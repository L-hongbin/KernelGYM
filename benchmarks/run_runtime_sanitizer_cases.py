"""Compile and run the Runtime Sanitizer TVM-FFI fixtures on a CUDA device."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.runtime_sanitizer_cases import CASES, KERNEL_CODE
from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
from kernelgym.toolkit.kernelbench.compute_sanitizer import run_compute_sanitizer


def run_cases(device: str = "cuda:0") -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    backend = KernelBenchBackend()
    artifact = backend.compile(
        KERNEL_CODE,
        device=device,
        backend="tvm_ffi",
        entry_point="ModelNew",
        enable_compile_artifact_cache=True,
    )
    if not artifact.get("compiled"):
        raise RuntimeError(f"TVM-FFI fixture compilation failed: {artifact.get('error')}")

    results = []
    for case in CASES:
        result = run_compute_sanitizer(
            original_model_src=case.reference_code,
            custom_model_src=KERNEL_CODE,
            artifact=artifact,
            backend="tvm_ffi",
            entry_point="Model",
            device=device,
            kernel_names=[case.kernel_name],
            sanitizer_path="/usr/local/cuda-12.9/bin/compute-sanitizer",
            timeout_s=60,
            max_kernels=4,
            max_issues=4,
            mode=case.tool,
        )
        issues = [
            issue for check_result in result.get("check_results", []) for issue in check_result.get("issues", [])
        ]
        observed_text = " ".join(
            str(value)
            for issue in issues
            for value in (
                issue.get("hazard_type"),
                issue.get("message"),
                issue.get("raw_excerpt"),
            )
            if value
        ).lower()
        status_matches = result.get("status") == case.expected_status
        hazard_matches = (
            case.expected_hazard_fragment is None or case.expected_hazard_fragment.lower() in observed_text
        )
        results.append(
            {
                "case": case.name,
                "mode": case.mode,
                "check": case.tool,
                "kernel_name": case.kernel_name,
                "expected_status": case.expected_status,
                "status_matches": status_matches,
                "hazard_matches": hazard_matches,
                "result": result,
            }
        )

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "backend": "tvm_ffi",
        "compiled": True,
        "all_expectations_met": all(item["status_matches"] and item["hazard_matches"] for item in results),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = run_cases(args.device)
    rendered = json.dumps(evidence, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if evidence["all_expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
