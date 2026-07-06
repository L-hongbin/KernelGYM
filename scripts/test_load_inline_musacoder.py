#!/usr/bin/env python3
"""End-to-end test for the load_inline backend on real MusaCoder generations.

Runs the KernelBench `add` example (known-correct self-test) plus a few real
MusaCoder-27B generations pulled from a slime eval dump through
``eval_kernel_against_ref(backend="load_inline", ...)`` and reports
detect/compiled/correct/speedup. Includes a WEIGHTED problem (conv) to exercise
seeded reference-vs-ModelNew weight matching.

Run on a GPU host with the KernelGym uv venv active and PYTHONPATH set to this
worktree, e.g.:

    source /nfs/.../KernelGYM-reward-only/.venv/bin/activate
    PYTHONPATH=/nfs/.../KernelGYM-load-inline \
        python scripts/test_load_inline_musacoder.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kernelgym.backend.registry import get_backend
from kernelgym.toolkit.kernelbench import pipeline as kb_pipeline
from kernelgym.toolkit.kernelbench.binding_detection import resolve_kernel_backend

DEFAULT_DUMP = (
    "/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/experiments/"
    "EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/"
    "MusaCoder-27B.kb_l1_musa_coder.load_inline.cuda_agent/dumps/rollout_data/eval_0.pt"
)

# KernelBench `add` example as a known-correct (reference, ModelNew) self-test.
ADD_REFERENCE = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, a, b):
        return a + b

def get_inputs():
    a = torch.rand(1, 128)
    b = torch.rand(1, 128)
    return [a, b]

def get_init_inputs():
    return []
"""

ADD_MODEL_NEW = """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_add_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { out[idx] = a[idx] + b[idx]; }
}
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
\"\"\"
elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add
    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
"""

# A decoy: compiles the same kernel but `forward` quietly uses torch.add — output
# is correct, yet the custom extension is never called. Expect decoy=True.
ADD_DECOY = ADD_MODEL_NEW.replace(
    "        return self.elementwise_add.elementwise_add_cuda(a, b)",
    "        return torch.add(a, b)",
)


def _ref_of(sample: dict) -> str | None:
    label = sample.get("label")
    if isinstance(label, dict) and label.get("ground_truth"):
        return label["ground_truth"]
    meta = sample.get("metadata") or {}
    return meta.get("ground_truth")


def _meta(sample: dict):
    meta = sample.get("metadata") or {}
    ei = meta.get("extra_info") or {}
    return meta.get("problem_id") or ei.get("problem_id"), meta.get("name") or ei.get("name")


def evaluate(reference: str, response: str, *, device, adapter, num_correct, num_perf) -> dict:
    detected = resolve_kernel_backend(response, "auto")
    kres = kb_pipeline.eval_kernel_against_ref(
        original_model_src=reference,
        custom_model_src=response,
        num_correct_trials=num_correct,
        num_perf_trials=num_perf,
        num_warmup=3,
        measure_performance=True,
        verbose=False,
        device=device,
        backend="load_inline",
        entry_point="Model",
        enable_profiling=False,
        enable_triton_detection=False,
        detect_decoy_kernel=True,
        backend_adapter=adapter,
    )
    ref_rt = kb_pipeline.eval_reference_only(
        original_model_src=reference,
        num_perf_trials=num_perf,
        num_warmup=3,
        verbose=False,
        device=device,
        entry_point="Model",
        backend_adapter=adapter,
    ).runtime
    kernel_rt = kres.runtime if kres.runtime and kres.runtime > 0 else None
    speedup = (ref_rt / kernel_rt) if (kernel_rt and ref_rt) else 0.0
    md = kres.metadata or {}
    err = md.get("compilation_error") or md.get("runtime_error") or md.get("correctness_issue")
    return {
        "detected_backend": detected,
        "compiled": bool(kres.compiled),
        "correctness": bool(kres.correctness),
        "decoy": bool(getattr(kres, "decoy_kernel", False)),
        "reference_runtime_ms": ref_rt,
        "kernel_runtime_ms": kernel_rt,
        "speedup": round(speedup, 3),
        "error": str(err)[:300] if err else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 216, 280, 448])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-correct", type=int, default=3)
    ap.add_argument("--num-perf", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    adapter = get_backend("kernelbench")
    results = []

    print("=" * 80)
    print("[self-test] KernelBench add example (expect compiled+correct)")
    r = evaluate(
        ADD_REFERENCE, ADD_MODEL_NEW, device=device, adapter=adapter,
        num_correct=args.num_correct, num_perf=args.num_perf,
    )
    r["case"] = "add_self_test"
    results.append(r)
    print(json.dumps(r, indent=2))

    print("=" * 80)
    print("[decoy-test] add kernel compiled but forward uses torch.add (expect compiled+correct+decoy)")
    r = evaluate(
        ADD_REFERENCE, ADD_DECOY, device=device, adapter=adapter,
        num_correct=args.num_correct, num_perf=args.num_perf,
    )
    r["case"] = "add_decoy_test"
    results.append(r)
    print(json.dumps(r, indent=2))

    obj = torch.load(args.dump, map_location="cpu", weights_only=False)
    samples = obj["samples"] if isinstance(obj, dict) else obj
    for idx in args.indices:
        if idx >= len(samples):
            continue
        s = samples[idx]
        pid, name = _meta(s)
        reference = _ref_of(s)
        response = s.get("response", "") or ""
        if not reference:
            print(f"[idx {idx}] no reference; skipping")
            continue
        print("=" * 80)
        print(f"[idx {idx}] problem_id={pid} name={name}")
        r = evaluate(
            reference, response, device=device, adapter=adapter,
            num_correct=args.num_correct, num_perf=args.num_perf,
        )
        r.update({"case": f"dump_idx_{idx}", "problem_id": pid, "name": name})
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "name"}, indent=2))

    n_compiled = sum(x["compiled"] for x in results)
    n_correct = sum(x["correctness"] for x in results)
    print("=" * 80)
    print(f"SUMMARY: {len(results)} cases, compiled={n_compiled}, correct={n_correct}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
