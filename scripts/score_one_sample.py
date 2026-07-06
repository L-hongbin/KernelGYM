#!/usr/bin/env python3
"""Score ONE load_inline submission against its reference, in an isolated process.

Run per sample by ``rescore_musacoder_dump.py`` so each load_inline JIT build /
module import is isolated (avoids cross-sample ``name=`` collisions). Reads the
reference module and the raw model response from files, evaluates via KernelGym's
``backend="load_inline"`` path, and prints a one-line result JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import time
import torch

from kernelgym.backend.registry import get_backend
from kernelgym.toolkit.kernelbench import pipeline as kb
from kernelgym.toolkit.kernelbench.binding_detection import extract_model_code, resolve_kernel_backend


def _tf32_state() -> dict[str, str]:
    state = {}
    paths = {
        "cudnn.allow_tf32": (torch.backends.cudnn, "allow_tf32"),
        "cuda.matmul.allow_tf32": (torch.backends.cuda.matmul, "allow_tf32"),
        "fp32_precision": (torch.backends, "fp32_precision"),
        "cudnn.fp32_precision": (torch.backends.cudnn, "fp32_precision"),
        "cudnn.conv.fp32_precision": (getattr(torch.backends.cudnn, "conv", None), "fp32_precision"),
        "cuda.matmul.fp32_precision": (torch.backends.cuda.matmul, "fp32_precision"),
    }
    for label, (obj, attr) in paths.items():
        if obj is None:
            continue
        try:
            state[label] = str(getattr(obj, attr))
        except Exception:
            continue
    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            state["float32_matmul_precision"] = str(torch.get_float32_matmul_precision())
        except Exception:
            pass
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-file", required=True)
    ap.add_argument("--response-file", required=True)
    ap.add_argument("--entry-point", default="Model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-correct", type=int, default=5)
    ap.add_argument("--num-perf", type=int, default=20)
    ap.add_argument("--backend", default="load_inline", help="KernelGym backend (load_inline | cuda_agent | ...).")
    args = ap.parse_args()

    reference = open(args.reference_file, encoding="utf-8").read()
    response = open(args.response_file, encoding="utf-8").read()
    detected = resolve_kernel_backend(response, "auto")

    out: dict = {"detected_backend": detected}
    eval_wall_start = time.perf_counter()
    try:
        device = torch.device(args.device)
        adapter = get_backend("kernelbench")
        kres = kb.eval_kernel_against_ref(
            original_model_src=reference,
            custom_model_src=response,
            num_correct_trials=args.num_correct,
            num_perf_trials=args.num_perf,
            num_warmup=3,
            measure_performance=True,
            verbose=False,
            device=device,
            backend=args.backend,
            entry_point=args.entry_point,
            enable_profiling=False,
            enable_triton_detection=False,
            detect_decoy_kernel=True,
            backend_adapter=adapter,
        )
        ref_rt = kb.eval_reference_only(
            original_model_src=reference,
            num_perf_trials=args.num_perf,
            num_warmup=3,
            verbose=False,
            device=device,
            entry_point=args.entry_point,
            backend_adapter=adapter,
        ).runtime
        kernel_rt = kres.runtime if (kres.runtime and kres.runtime > 0) else None
        speedup = (ref_rt / kernel_rt) if (kernel_rt and ref_rt) else 0.0
        md = kres.metadata or {}
        err = md.get("compilation_error") or md.get("runtime_error") or md.get("correctness_issue")
        out.update(
            compiled=bool(kres.compiled),
            correctness=bool(kres.correctness),
            decoy=bool(getattr(kres, "decoy_kernel", False)),
            reference_runtime_ms=ref_rt,
            kernel_runtime_ms=kernel_rt,
            speedup=round(speedup, 4),
            extracted_chars=len(extract_model_code(response)),
            error=str(err)[:500] if err else None,
            correctness_atol=md.get("correctness_atol"),
            correctness_rtol=md.get("correctness_rtol"),
            correctness_issue_name=md.get("correctness_issue_name"),
            max_difference=md.get("max_difference"),
            avg_difference=md.get("avg_difference"),
            correctness_tf32_disabled=md.get("correctness_tf32_disabled"),
            correctness_tf32_state_before=md.get("correctness_tf32_state_before"),
            correctness_tf32_state_forced=md.get("correctness_tf32_state_forced"),
            post_eval_tf32_state=_tf32_state(),
            # Phase timing, purely additive: lets an offline rescore distinguish
            # "compile took long" from "correctness-check execution took long"
            # without needing to raise the outer subprocess timeout blindly.
            eval_wall_s=round(time.perf_counter() - eval_wall_start, 3),
            kg_kernel_total_s=md.get("kg_kernel_total_s"),
            kg_kernel_backend_compile_s=md.get("kg_kernel_backend_compile_s"),
            correctness_trial_s=md.get("correctness_trial_s"),
            correctness_reference_trial_s=md.get("correctness_reference_trial_s"),
            correctness_custom_trial_s=md.get("correctness_custom_trial_s"),
        )
    except Exception as exc:  # noqa: BLE001
        out.update(
            compiled=False, correctness=False, decoy=False, error=f"{type(exc).__name__}: {exc}"[:500],
            eval_wall_s=round(time.perf_counter() - eval_wall_start, 3),
        )

    print(json.dumps(out))


if __name__ == "__main__":
    main()
