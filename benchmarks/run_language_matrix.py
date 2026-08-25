#!/usr/bin/env python3
"""Run the CUDA (TVM-FFI), Triton, and TileLang KernelBench fixture matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LANGUAGE_MODULES = {
    "cuda": "benchmarks.kernels.cuda.cases",
    "triton": "benchmarks.kernels.triton.cases",
    "tilelang": "benchmarks.kernels.tilelang.cases",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:20111")
    parser.add_argument(
        "--languages",
        default="cuda,triton,tilelang",
        help="comma-separated subset of cuda,triton,tilelang; CUDA uses backend=tvm_ffi",
    )
    parser.add_argument("--cases", default="", help="comma-separated case-name subset")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--split", action="store_true", help="use split CPU compile / GPU execute")
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--out", help="write the final JSON report to this path")
    return parser.parse_args()


def disable_proxy() -> None:
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def health(api: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{api.rstrip('/')}/health", timeout=10) as response:
        return json.load(response)


def load_modules(languages: list[str], api: str) -> dict[str, ModuleType]:
    unknown = sorted(set(languages) - LANGUAGE_MODULES.keys())
    if unknown:
        raise ValueError(f"Unknown languages: {', '.join(unknown)}")
    modules = {}
    for language in languages:
        module = importlib.import_module(LANGUAGE_MODULES[language])
        module.URL = f"{api.rstrip('/')}/evaluate"
        modules[language] = module
    return modules


def expected_success(name: str, result: dict[str, Any]) -> bool:
    if name.endswith("_rejected"):
        return result.get("compiled") is False
    return (
        result.get("compiled") is True
        and result.get("correctness") is True
        and result.get("decoy") is False
        and (result.get("profile_kernels") or 0) >= 1
    )


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    selected_names = {item.strip() for item in args.cases.split(",") if item.strip()}
    disable_proxy()
    probe = health(args.api)
    if probe.get("status") != "healthy":
        raise RuntimeError(f"KernelGym is not healthy: {probe}")
    modules = load_modules(languages, args.api)

    jobs = []
    for language, module in modules.items():
        for case in module.CASES:
            name = case[0]
            if not args.include_negative and name.endswith("_rejected"):
                continue
            if selected_names and name not in selected_names:
                continue
            jobs.append((language, module, case))
    if not jobs:
        raise ValueError("No cases matched the selected languages/case names")

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(module.run, case, args.split): (language, case[0]) for language, module, case in jobs}
        for future in concurrent.futures.as_completed(futures):
            language, name = futures[future]
            result = future.result()
            record = {
                "language": language,
                **result,
                "passed": expected_success(name, result),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    records.sort(key=lambda item: (item["language"], item["name"]))
    report = {
        "api": args.api,
        "languages": languages,
        "split": args.split,
        "concurrency": args.concurrency,
        "total": len(records),
        "passed": sum(record["passed"] for record in records),
        "failed": sum(not record["passed"] for record in records),
        "results": records,
    }
    print("SUMMARY=" + json.dumps(report, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
