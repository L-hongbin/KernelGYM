#!/usr/bin/env python3
"""Collect comparable NCU reports for both sides of the fixed speed-test case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kernelgym.config import settings
from kernelgym.server.api.speed_test import BACKEND, INPUT_SHAPES, KERNEL_CODE, REFERENCE_CODE
from kernelgym.toolkit.kernelbench.ncu_profiler import run_ncu_profile


def _duration_us(profile: dict[str, Any]) -> float:
    total = 0.0
    for kernel in profile.get("kernels", []):
        metric = kernel.get("metrics", {}).get("gpu__time_duration.sum", {})
        value = metric.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return total


def collect() -> dict[str, Any]:
    common = dict(
        original_model_src=REFERENCE_CODE,
        custom_model_src=KERNEL_CODE,
        artifact=None,
        backend=BACKEND,
        entry_point="Model",
        device="cuda:0",
        ncu_path=settings.ncu_path,
        metrics=settings.ncu_metrics,
        timeout_s=settings.ncu_timeout_s,
        max_kernels=settings.ncu_max_kernels,
        warmup=settings.ncu_warmup,
        profile_version=settings.ncu_profile_version,
    )
    reference = run_ncu_profile(kernel_names=[], target="reference", **common)
    candidate = run_ncu_profile(kernel_names=["gemm_rmsnorm_kernel"], target="candidate", **common)
    reference_duration_us = _duration_us(reference)
    candidate_duration_us = _duration_us(candidate)
    return {
        "case_name": "gemm_rmsnorm_fp32",
        "backend": BACKEND,
        "input_shapes": INPUT_SHAPES,
        "summary": {
            "reference_status": reference.get("status"),
            "candidate_status": candidate.get("status"),
            "reference_kernel_count": reference.get("profiled_kernel_count", 0),
            "candidate_kernel_count": candidate.get("profiled_kernel_count", 0),
            "reference_total_kernel_time_us": reference_duration_us,
            "candidate_total_kernel_time_us": candidate_duration_us,
            "kernel_time_speedup": (
                reference_duration_us / candidate_duration_us if candidate_duration_us > 0 else None
            ),
        },
        "reference": reference,
        "candidate": candidate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the full JSON result to this path")
    args = parser.parse_args()

    result = collect()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text, end="")
    return 0 if result["summary"]["reference_status"] == result["summary"]["candidate_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
