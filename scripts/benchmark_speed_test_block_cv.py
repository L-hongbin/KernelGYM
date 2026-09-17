#!/usr/bin/env python3
"""Measure block-level runtime CV for the fixed GEMM + RMSNorm speed-test case."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
from kernelgym.server.api.speed_test import BACKEND, CASE_NAME, INPUT_SHAPES, KERNEL_CODE, REFERENCE_CODE
from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref, eval_reference_only


def _block_mean_ms(metadata: dict[str, Any], *, prefix: str) -> float:
    total_s = float(metadata[f"kg_{prefix}_perf_measure_cuda_event_s"])
    trials = int(metadata[f"kg_{prefix}_perf_num_trials"])
    if trials <= 0:
        raise RuntimeError(f"{prefix} returned no timed trials")
    return 1000.0 * total_s / trials


def _distribution(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean if mean > 0 else 0.0,
        "cv_percent": 100.0 * std / mean if mean > 0 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _within_block_cv_summary(blocks: list[dict[str, Any]], *, prefix: str) -> dict[str, float]:
    values = [
        float(block[f"{prefix}_within_block_std_ms"]) / float(block[f"{prefix}_block_mean_ms"]) for block in blocks
    ]
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def run(*, blocks: int, warmup: int, trials: int, device: str) -> dict[str, Any]:
    if blocks < 1:
        raise ValueError("blocks must be at least 1")
    if warmup < 0 or trials < 1:
        raise ValueError("warmup must be nonnegative and trials must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch_device = torch.device(device)
    backend_adapter = KernelBenchBackend()
    block_results: list[dict[str, Any]] = []

    for block_index in range(blocks):
        block_start = perf_counter()
        unique_kernel_code = KERNEL_CODE.replace(
            "__global__ void gemm_rmsnorm_kernel",
            f"// block-cv-run: {block_index}\n__global__ void gemm_rmsnorm_kernel",
            1,
        )
        kernel_result = eval_kernel_against_ref(
            original_model_src=REFERENCE_CODE,
            custom_model_src=unique_kernel_code,
            seed_num=42,
            num_correct_trials=1,
            num_perf_trials=trials,
            num_warmup=warmup,
            perf_trim_count=0,
            verbose=False,
            measure_performance=True,
            device=torch_device,
            backend=BACKEND,
            precision="fp32",
            entry_point="Model",
            enable_profiling=False,
            enable_ncu=False,
            enable_compute_sanitizer=False,
            enable_correctness_input_perturbations=False,
            enable_triton_detection=False,
            detect_decoy_kernel=False,
            backend_adapter=backend_adapter,
            enable_compile_artifact_cache=False,
            adaptive_perf_trials=False,
        )
        if not kernel_result.compiled or not kernel_result.correctness or kernel_result.runtime <= 0:
            raise RuntimeError(
                f"block {block_index + 1} kernel evaluation failed: "
                f"compiled={kernel_result.compiled}, correctness={kernel_result.correctness}, "
                f"metadata={kernel_result.metadata}"
            )

        reference_result = eval_reference_only(
            original_model_src=REFERENCE_CODE,
            seed_num=42,
            num_perf_trials=trials,
            num_warmup=warmup,
            perf_trim_count=0,
            verbose=False,
            device=torch_device,
            entry_point="Model",
            backend_adapter=backend_adapter,
        )
        if reference_result.runtime <= 0:
            raise RuntimeError(f"block {block_index + 1} reference timing failed: {reference_result.metadata}")

        kernel_mean_ms = _block_mean_ms(kernel_result.metadata, prefix="kernel")
        reference_mean_ms = _block_mean_ms(reference_result.metadata, prefix="reference")
        block_results.append(
            {
                "block": block_index + 1,
                "kernel_block_mean_ms": kernel_mean_ms,
                "reference_block_mean_ms": reference_mean_ms,
                "block_speedup": reference_mean_ms / kernel_mean_ms,
                "kernel_returned_runtime_ms": kernel_result.runtime,
                "reference_returned_runtime_ms": reference_result.runtime,
                "kernel_within_block_std_ms": kernel_result.runtime_stats.get("std"),
                "reference_within_block_std_ms": reference_result.runtime_stats.get("std"),
                "kernel_within_block_min_ms": kernel_result.runtime_stats.get("min"),
                "kernel_within_block_max_ms": kernel_result.runtime_stats.get("max"),
                "reference_within_block_min_ms": reference_result.runtime_stats.get("min"),
                "reference_within_block_max_ms": reference_result.runtime_stats.get("max"),
                "kernel_compile_artifact_cache_enabled": kernel_result.metadata.get("compile_artifact_cache_enabled"),
                "kernel_compile_artifact_cache_hit": kernel_result.metadata.get("compile_artifact_cache_hit"),
                "wall_time_s": perf_counter() - block_start,
            }
        )
        print(
            f"block {block_index + 1:02d}/{blocks}: "
            f"reference={reference_mean_ms:.6f} ms, kernel={kernel_mean_ms:.6f} ms, "
            f"speedup={reference_mean_ms / kernel_mean_ms:.4f}x",
            flush=True,
        )

    kernel_means = [item["kernel_block_mean_ms"] for item in block_results]
    reference_means = [item["reference_block_mean_ms"] for item in block_results]
    block_speedups = [item["block_speedup"] for item in block_results]
    kernel_trial_min = min(float(item["kernel_within_block_min_ms"]) for item in block_results)
    kernel_trial_max = max(float(item["kernel_within_block_max_ms"]) for item in block_results)
    reference_trial_min = min(float(item["reference_within_block_min_ms"]) for item in block_results)
    reference_trial_max = max(float(item["reference_within_block_max_ms"]) for item in block_results)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "case_name": CASE_NAME,
        "backend": BACKEND,
        "input_shapes": INPUT_SHAPES,
        "device": device,
        "gpu": torch.cuda.get_device_name(torch_device),
        "protocol": {
            "blocks": blocks,
            "warmup_per_block": warmup,
            "measurements_per_block": trials,
            "std_ddof": 0,
            "block_order": "kernel_then_reference",
            "result_cache": False,
            "reference_cache": False,
            "compile_artifact_cache": False,
            "adaptive_perf_trials": False,
            "profiling": False,
            "ncu": False,
            "compute_sanitizer": False,
        },
        "summary": {
            "kernel": {
                **_distribution(kernel_means),
                "within_block_trial_cv": _within_block_cv_summary(block_results, prefix="kernel"),
            },
            "reference": {
                **_distribution(reference_means),
                "within_block_trial_cv": _within_block_cv_summary(block_results, prefix="reference"),
            },
            "block_speedup": {
                "mean": statistics.fmean(block_speedups),
                "std": statistics.pstdev(block_speedups),
                "cv": statistics.pstdev(block_speedups) / statistics.fmean(block_speedups),
                "min": min(block_speedups),
                "max": max(block_speedups),
            },
            "within_block_trial_runtime_range_ms": {
                "kernel": {"min": kernel_trial_min, "max": kernel_trial_max},
                "reference": {"min": reference_trial_min, "max": reference_trial_max},
            },
            "unpaired_trial_speedup_envelope": {
                "min": reference_trial_min / kernel_trial_max,
                "max": reference_trial_max / kernel_trial_min,
                "interpretation": "conservative envelope from independent reference/kernel trial extrema",
            },
        },
        "blocks": block_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(blocks=args.blocks, warmup=args.warmup, trials=args.trials, device=args.device)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
