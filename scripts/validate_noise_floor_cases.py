#!/usr/bin/env python3
"""Compile and correctness-smoke the fixed noise-floor calibration cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


def _run_child(case_id: str, output: Path) -> int:
    import torch

    from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
    from kernelgym.server.api.noise_floor_cases import CASE_BY_ID
    from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref

    case = CASE_BY_ID[case_id]
    started = perf_counter()
    result = eval_kernel_against_ref(
        original_model_src=case.reference_code,
        custom_model_src=case.kernel_code,
        seed_num=42,
        num_correct_trials=1,
        num_perf_trials=2,
        num_warmup=1,
        perf_trim_count=0,
        verbose=False,
        measure_performance=True,
        device=torch.device("cuda:0"),
        backend="tvm_ffi",
        precision="fp32",
        entry_point="Model",
        enable_profiling=False,
        enable_ncu=False,
        enable_compute_sanitizer=False,
        enable_correctness_input_perturbations=False,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend_adapter=KernelBenchBackend(),
        enable_compile_artifact_cache=False,
        adaptive_perf_trials=False,
    )
    error_code = result.metadata.get("error_code")
    if hasattr(error_code, "value"):
        error_code = error_code.value
    payload = {
        "case_id": case_id,
        "case": case.public_metadata(),
        "gpu": torch.cuda.get_device_name(0),
        "compiled": result.compiled,
        "correctness": result.correctness,
        "kernel_runtime_ms": result.runtime,
        "error_code": error_code,
        "error_message": result.metadata.get("error_message") or result.metadata.get("correctness_issue"),
        "elapsed_s": perf_counter() - started,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if result.compiled and result.correctness else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-case")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if args.child_case:
        if args.child_output is None:
            parser.error("--child-output is required with --child-case")
        return _run_child(args.child_case, args.child_output)

    from kernelgym.server.api.noise_floor_cases import get_noise_floor_cases

    cases = get_noise_floor_cases()
    results = []
    with tempfile.TemporaryDirectory(prefix="kernelgym-noise-floor-smoke-") as temp_dir:
        for case in cases:
            child_output = Path(temp_dir) / f"{case.case_id}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-case",
                case.case_id,
                "--child-output",
                str(child_output),
            ]
            child = subprocess.run(command, text=True, capture_output=True)
            if child_output.exists():
                result = json.loads(child_output.read_text(encoding="utf-8"))
            else:
                result = {
                    "case_id": case.case_id,
                    "compiled": False,
                    "correctness": False,
                    "error_message": child.stderr or child.stdout,
                }
            result["child_return_code"] = child.returncode
            results.append(result)
            print(
                f"{case.case_id}: compiled={result.get('compiled')} "
                f"correctness={result.get('correctness')} runtime_ms={result.get('kernel_runtime_ms')}",
                flush=True,
            )

    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "gpu": next((result.get("gpu") for result in results if result.get("gpu")), "unknown"),
        "protocol": "fresh process per case; 1 correctness, 1 warmup, 2 performance trials",
        "source_dataset": next(
            (
                case.public_metadata()["dataset_path"]
                for case in cases
                if case.public_metadata()["dataset_path"] is not None
            ),
            None,
        ),
        "passed": sum(1 for result in results if result.get("compiled") and result.get("correctness")),
        "total": len(results),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
