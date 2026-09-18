"""Reproduce fresh-process and steady-state detailed comparator measurements.

Run from the checkout: .venv/bin/python -m benchmarks.benchmark_correctness_fusion
The JSON report includes every sample, representative outputs and build cost.
No evaluator restart or modification of service settings is required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid


def child(mode, shape, correct):
    import torch
    from kernelgym.toolkit.kernelbench import correctness as c

    torch.manual_seed(123)
    reference = torch.randn(shape, device="cuda") + 1
    candidate = reference.clone() if correct else reference - 1
    functions = {
        "legacy": c._compare_tensors_inplace,
        "before": c._compare_tensors_inplace_with_diagnostics_torch,
        "after": c._compare_tensors_inplace_with_diagnostics,
    }
    function = functions[mode]

    def once():
        a, b = reference.clone(), candidate.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function(a, b, atol=0.001, rtol=0.001)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000, result

    first_ms, example = once()
    for _ in range(3):
        once()
    samples = [once()[0] for _ in range(20)]
    return {
        "pid": os.getpid(),
        "device": torch.cuda.get_device_name(),
        "mode": mode,
        "shape": shape,
        "correct_output": correct,
        "first_ms": first_ms,
        "steady_ms": samples,
        "example": example,
    }


def pipeline_child(mode):
    import torch
    from benchmarks.kernels.tvm_ffi_vector_add import KERNEL_CODE, REFERENCE_CODE
    from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend
    from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref

    # A real TVM-FFI candidate, with the same 1M-output all-mismatch diagnostic
    # workload as the comparator benchmark. Explicit controls matter: disabling
    # CUDA profiling alone does not disable the independent ATen/decoy profiler.
    ref = REFERENCE_CODE.replace("4096", "(4, 512, 512)").replace("return a + b", "return a + b + 1.0")
    backend = KernelBenchTvmFfiBackend()
    artifact = backend.compile(KERNEL_CODE, device="cuda:0", enable_compile_artifact_cache=True)
    assert artifact["compiled"], artifact
    os.environ["KERNELGYM_FUSED_CORRECTNESS"] = "true" if mode == "after" else "false"
    result = eval_kernel_against_ref(
        ref,
        KERNEL_CODE,
        backend="tvm_ffi",
        device=torch.device("cuda:0"),
        backend_adapter=backend,
        precompiled_artifact=artifact,
        num_correct_trials=1,
        num_perf_trials=1,
        measure_performance=False,
        enable_profiling=False,
        enable_ncu=False,
        enable_compute_sanitizer=False,
        enable_correctness_input_perturbations=False,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        return_detail_correctness=mode != "legacy",
        verbose=False,
    )
    assert result.compiled and result.correctness is False
    return {"mode": mode, "pid": os.getpid(), "metadata": result.metadata}


def http_validation(endpoint, blocks):
    from benchmarks.kernels.tvm_ffi_vector_add import KERNEL_CODE, REFERENCE_CODE

    payload = {
        "reference_code": REFERENCE_CODE.replace("4096", "(4, 512, 512)").replace(
            "return a + b", "return a + b + 1.0"
        ),
        "kernel_code": KERNEL_CODE,
        "backend": "tvm_ffi",
        "num_correct_trials": 1,
        "num_perf_trials": 1,
        "timeout": 180,
        "force_refresh": True,
        "run_performance": False,
        "enable_profiling": False,
        "enable_ncu": False,
        "enable_compute_sanitizer": False,
        "enable_correctness_input_perturbations": False,
        "use_reference_cache": False,
        "enable_compile_artifact_cache": True,
        "split_compile_and_execute": True,
        "detect_decoy_kernel": False,
        "enable_triton_detection": False,
        "run_triton_detection": False,
    }
    runs = []
    for block in range(-2, blocks):
        for detail in (False, True) if block % 2 == 0 else (True, False):
            body = {**payload, "task_id": f"fusion-http-{uuid.uuid4().hex}", "return_detail_correctness": detail}
            request = urllib.request.Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=240) as response:
                result = json.load(response)
            elapsed = time.perf_counter() - start
            assert (
                result["compiled"] and result["correctness"] is False and result["error_code"] == "CORRECTNESS_ERROR"
            )
            assert result["metadata"]["aten_detection_enabled"] is False
            fields = ("element_correctness_curve", "nan_count", "inf_count", "mismatch_localization")
            assert all((field in result["metadata"]) == detail for field in fields)
            if block >= 0:
                assert result["metadata"]["compile_artifact_cache_hit"]
                runs.append({"block": block, "detail": detail, "wall_s": elapsed, "response": result})
        print(f"Completed HTTP pair {block + 1}/{blocks}", file=sys.stderr, flush=True)
    summary = {}
    for detail in (False, True):
        selected = [r for r in runs if r["detail"] == detail]
        summary["detail" if detail else "legacy"] = {
            "http_mean_s": statistics.mean(r["wall_s"] for r in selected),
            "correctness_mean_s": statistics.mean(
                r["response"]["metadata"]["kg_kernel_correctness_s"] for r in selected
            ),
            "compare_mean_s": statistics.mean(
                r["response"]["metadata"]["correctness_compare_trial_s"][0] for r in selected
            ),
        }
    return {
        "scope": "HTTP current implementation, legacy versus fused detail, two warmup pairs excluded",
        "payload": payload,
        "summary_seconds": summary,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=["legacy", "before", "after"])
    parser.add_argument("--shape", default="4,512,512")
    parser.add_argument("--correct", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--http", help="Optional evaluator URL for current-service validation")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.http:
        report = http_validation(args.http, args.blocks)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary_seconds"], indent=2))
        return
    if args.child:
        result = (
            pipeline_child(args.child)
            if args.pipeline
            else child(args.child, tuple(int(d) for d in args.shape.split(",")), args.correct)
        )
        print(json.dumps(result, default=str))
        return

    def execute(mode, extra=(), env=None):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.benchmark_correctness_fusion",
                "--child",
                mode,
                "--shape",
                args.shape,
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads(result.stdout)

    if args.pipeline:
        runs = []
        for block in range(args.blocks):
            for mode in ("legacy", "before", "after") if block % 2 == 0 else ("after", "before", "legacy"):
                run = execute(mode, ("--pipeline",))
                run["block"] = block
                runs.append(run)
            print(f"Completed pipeline block {block + 1}/{args.blocks}", file=sys.stderr, flush=True)
        summary = {
            mode: {
                key: statistics.mean(r["metadata"][key] for r in runs if r["mode"] == mode)
                for key in ("kg_kernel_correctness_s", "kg_kernel_total_s")
            }
            for mode in ("legacy", "before", "after")
        }
        report = {
            "scope": "fresh-process TVM-FFI evaluator pipeline; not HTTP",
            "summary_seconds": summary,
            "runs": runs,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(summary, indent=2))
        return

    # Compile miss is measured separately, never silently excluded from a
    # "first call" claim. Its artifact is disposable and the source is trusted.
    with tempfile.TemporaryDirectory(prefix="kg-correctness-cold-build-") as cache:
        build = execute(
            "after",
            env={**os.environ, "KERNELGYM_CORRECTNESS_CACHE_DIR": cache, "KERNELGYM_FUSED_CORRECTNESS": "true"},
        )
    runs = []
    for block in range(args.blocks):
        modes = ("legacy", "before", "after") if block % 2 == 0 else ("after", "before", "legacy")
        for mode in modes:
            run = execute(mode)
            run["block"] = block
            runs.append(run)
        print(f"Completed block {block + 1}/{args.blocks}", file=sys.stderr, flush=True)
    correct_runs = [execute(mode, ("--correct",)) for mode in ("legacy", "before", "after")]

    def stats(values):
        return {
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }

    summary = {}
    for mode in ("legacy", "before", "after"):
        selected = [r for r in runs if r["mode"] == mode]
        summary[mode] = {
            "fresh_process_first_call": stats([r["first_ms"] for r in selected]),
            "steady_state": stats([v for r in selected for v in r["steady_ms"]]),
        }
    report = {
        "scope": "direct comparator, identical FP32 inputs, independent CUDA processes; not HTTP latency",
        "build_miss_first_call_ms": build["first_ms"],
        "summary": summary,
        "runs": runs,
        "correct_output_runs": correct_runs,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"build_miss_first_call_ms": build["first_ms"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
