#!/usr/bin/env python3
"""Measure block CV with one fresh Python/CUDA process per block."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


BASE_SCRIPT = Path(__file__).with_name("benchmark_speed_test_block_cv.py")


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


def _within_block_cv(blocks: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    values = [
        float(block[f"{prefix}_within_block_std_ms"]) / float(block[f"{prefix}_block_mean_ms"])
        for block in blocks
    ]
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def run(*, blocks: int, warmup: int, trials: int, device: str) -> dict[str, Any]:
    if blocks < 2:
        raise ValueError("blocks must be at least 2")
    if warmup < 0 or trials < 1:
        raise ValueError("warmup must be nonnegative and trials must be positive")

    block_results: list[dict[str, Any]] = []
    first_payload: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="kernelgym-fresh-block-cv-") as temp_dir:
        for block_index in range(blocks):
            child_output = Path(temp_dir) / f"block_{block_index + 1:02d}.json"
            command = [
                sys.executable,
                str(BASE_SCRIPT),
                "--blocks",
                "1",
                "--warmup",
                str(warmup),
                "--trials",
                str(trials),
                "--device",
                device,
                "--output",
                str(child_output),
            ]
            child_start = perf_counter()
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = child.communicate()
            child_wall_s = perf_counter() - child_start
            if child.returncode != 0:
                raise RuntimeError(
                    f"fresh block {block_index + 1} failed with exit code {child.returncode}:\n{stderr}\n{stdout}"
                )

            payload = json.loads(child_output.read_text(encoding="utf-8"))
            if first_payload is None:
                first_payload = payload
            block = dict(payload["blocks"][0])
            block.update(
                {
                    "block": block_index + 1,
                    "fresh_process": True,
                    "process_pid": child.pid,
                    "process_wall_time_s": child_wall_s,
                    "child_captured_at": payload["captured_at"],
                }
            )
            block_results.append(block)
            print(
                f"block {block_index + 1:02d}/{blocks} pid={child.pid}: "
                f"reference={block['reference_block_mean_ms']:.6f} ms, "
                f"kernel={block['kernel_block_mean_ms']:.6f} ms, "
                f"speedup={block['block_speedup']:.4f}x",
                flush=True,
            )

    assert first_payload is not None
    kernel_means = [float(item["kernel_block_mean_ms"]) for item in block_results]
    reference_means = [float(item["reference_block_mean_ms"]) for item in block_results]
    block_speedups = [float(item["block_speedup"]) for item in block_results]
    kernel_trial_min = min(float(item["kernel_within_block_min_ms"]) for item in block_results)
    kernel_trial_max = max(float(item["kernel_within_block_max_ms"]) for item in block_results)
    reference_trial_min = min(float(item["reference_within_block_min_ms"]) for item in block_results)
    reference_trial_max = max(float(item["reference_within_block_max_ms"]) for item in block_results)
    speedup_mean = statistics.fmean(block_speedups)
    speedup_std = statistics.pstdev(block_speedups)

    protocol = dict(first_payload["protocol"])
    protocol.update(
        {
            "blocks": blocks,
            "fresh_process_per_block": True,
            "process_state_reset": "child process exit after each block",
            "gpu_device_state_reset": False,
        }
    )
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "case_name": first_payload["case_name"],
        "backend": first_payload["backend"],
        "input_shapes": first_payload["input_shapes"],
        "device": device,
        "gpu": first_payload["gpu"],
        "protocol": protocol,
        "summary": {
            "kernel": {
                **_distribution(kernel_means),
                "within_block_trial_cv": _within_block_cv(block_results, "kernel"),
            },
            "reference": {
                **_distribution(reference_means),
                "within_block_trial_cv": _within_block_cv(block_results, "reference"),
            },
            "block_speedup": {
                "mean": speedup_mean,
                "std": speedup_std,
                "cv": speedup_std / speedup_mean,
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
    parser.add_argument("--warmup", type=int, default=10)
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
