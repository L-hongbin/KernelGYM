#!/usr/bin/env python3
"""Compare fresh-process import latency between shared and node-local Python environments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


RESULT_PREFIX = "KERNELGYM_IMPORT_RESULT="
CHILD_CODE = f"""
import importlib
import json
import os
import sys
import time

started = time.perf_counter()
module = importlib.import_module(sys.argv[1])
payload = {{
    "child_import_s": time.perf_counter() - started,
    "module_origin": getattr(module, "__file__", None),
    "module_version": str(getattr(module, "__version__", "")),
    "pid": os.getpid(),
    "python": sys.executable,
}}
print({RESULT_PREFIX!r} + json.dumps(payload, sort_keys=True), flush=True)
"""


def _run_once(label: str, python: Path, module_name: str, timeout_s: float, mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(python), "-I", "-c", CHILD_CODE, module_name],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    parent_wall_s = time.perf_counter() - started
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise RuntimeError(f"{label} child failed with exit {completed.returncode}: {detail[-2000:]}")
    result_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise RuntimeError(f"{label} child produced no result marker")
    result = json.loads(result_line[len(RESULT_PREFIX) :])
    result.update({"label": label, "mode": mode, "parent_wall_s": parent_wall_s})
    return result


def _run_batch(
    label: str,
    python: Path,
    count: int,
    module_name: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(_run_once, label, python, module_name, timeout_s, "parallel") for _ in range(count)]
        return [future.result() for future in futures]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "child_import_s": _stats([float(record["child_import_s"]) for record in records]),
        "module_origins": sorted({str(record["module_origin"]) for record in records}),
        "n": len(records),
        "parent_wall_s": _stats([float(record["parent_wall_s"]) for record in records]),
        "python_reported": sorted({str(record["python"]) for record in records}),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-python", required=True, help="Python interpreter in the shared environment")
    parser.add_argument("--local-python", required=True, help="Python interpreter in the node-local environment")
    parser.add_argument("--module", default="torch", help="Module imported by every fresh child (default: torch)")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Unreported warmup runs per environment")
    parser.add_argument("--serial-runs", type=int, default=6, help="Measured serial runs per environment")
    parser.add_argument("--parallel-runs", type=int, default=8, help="Measured concurrent runs per environment")
    parser.add_argument("--parallelism", type=int, default=4, help="Fresh child processes per concurrent batch")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-child timeout in seconds")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be >= 0")
    if min(args.serial_runs, args.parallel_runs, args.parallelism) < 1:
        raise SystemExit("--serial-runs, --parallel-runs, and --parallelism must be >= 1")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")

    environments = [
        ("shared", Path(args.shared_python).expanduser().resolve()),
        ("local", Path(args.local_python).expanduser().resolve()),
    ]
    for label, python in environments:
        if not python.is_file() or not os.access(python, os.X_OK):
            raise SystemExit(f"{label} interpreter is missing or not executable: {python}")

    for _ in range(args.warmup_runs):
        for label, python in environments:
            _run_once(label, python, args.module, args.timeout, "warmup")

    records: list[dict[str, Any]] = []
    for run_index in range(args.serial_runs):
        ordered = environments if run_index % 2 == 0 else list(reversed(environments))
        for label, python in ordered:
            records.append(_run_once(label, python, args.module, args.timeout, "serial"))

    remaining = {label: args.parallel_runs for label, _ in environments}
    batch_index = 0
    while any(remaining.values()):
        ordered = environments if batch_index % 2 == 0 else list(reversed(environments))
        for label, python in ordered:
            count = min(args.parallelism, remaining[label])
            if count:
                records.extend(_run_batch(label, python, count, args.module, args.timeout))
                remaining[label] -= count
        batch_index += 1

    summaries = {
        label: {
            mode: _summarize([record for record in records if record["label"] == label and record["mode"] == mode])
            for mode in ("serial", "parallel")
        }
        for label, _ in environments
    }
    for label, modes in summaries.items():
        for mode, summary in modes.items():
            timing = summary["child_import_s"]
            print(
                f"{label:>6} {mode:>8}: n={summary['n']} mean={timing['mean']:.3f}s "
                f"p95={timing['p95']:.3f}s max={timing['max']:.3f}s",
                file=sys.stderr,
            )

    print(
        json.dumps(
            {
                "config": vars(args),
                "records": records,
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
