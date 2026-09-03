"""Exercise deferred Runtime Sanitizer replay through the real subprocess pool."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.runtime_sanitizer_cases import CASES, KERNEL_CODE
from kernelgym.worker.subprocess_pool import FAULT_CONTEXT, SubprocessWorkerPool


def _faulting_kernel_code() -> str:
    return KERNEL_CODE.replace(
        "output[index] = index < n ? input[index] : 0.0f;",
        "if (index == 0) { *reinterpret_cast<volatile float*>(0x1) = 1.0f; }",
    )


async def _run(device_id: int) -> dict:
    case = next(item for item in CASES if item.name == "global_oob")
    pool = SubprocessWorkerPool(
        device_id=device_id,
        pool_size=1,
        worker_prefix="sanitizer_containment_evidence",
        max_tasks_per_worker=2,
    )
    shutdown_safe = False
    wrapped = None
    try:
        wrapped = await pool.execute_task(
            {
                "task_id": "sanitizer-containment-evidence",
                "base_task_id": "sanitizer-containment-evidence",
                "task_type": "kernel_evaluation",
                "toolkit": "kernelbench",
                "backend_adapter": "kernelbench",
                "backend": "tvm_ffi",
                "reference_code": case.reference_code,
                "kernel_code": _faulting_kernel_code(),
                "entry_point": "Model",
                "device": f"cuda:{device_id}",
                "num_correct_trials": 1,
                "num_perf_trials": 1,
                "run_performance": False,
                "enable_profiling": False,
                "enable_compute_sanitizer": True,
                "compute_sanitizer_mode": "error_based",
                "enable_triton_detection": False,
                "detect_decoy_kernel": False,
                "timeout": 180,
            },
            timeout=180,
            max_retries=0,
        )
    finally:
        shutdown_safe = await pool.shutdown(timeout=30)

    if not isinstance(wrapped, dict) or not isinstance(wrapped.get("result"), dict):
        raise RuntimeError(f"Pool returned an invalid containment result: {wrapped!r}")
    result = wrapped["result"]
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    diagnostic = result.get("runtime_sanitizer") if isinstance(result.get("runtime_sanitizer"), dict) else {}
    checks = diagnostic.get("check_results") if isinstance(diagnostic.get("check_results"), list) else []
    issues = [
        issue
        for check in checks
        if isinstance(check, dict)
        for issue in check.get("issues", [])
        if isinstance(issue, dict)
    ]
    expectations = {
        "structured_result_preserved": wrapped.get("success") is True,
        "fault_context_classified": wrapped.get("fault_severity") == FAULT_CONTEXT,
        "faulting_worker_retired": wrapped.get("worker_exiting") is True,
        "private_request_removed": "_runtime_sanitizer_request" not in metadata,
        "memcheck_issue_found": diagnostic.get("status") == "issues_found",
        "invalid_global_write_parsed": any(issue.get("hazard_type") == "invalid_global_write" for issue in issues),
        "replacement_path_executed": float((wrapped.get("pool_timing") or {}).get("pool_restart_s") or 0) > 0,
        "pool_shutdown_safe": shutdown_safe,
    }
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "device": f"cuda:{device_id}",
        "gpu": torch.cuda.get_device_name(torch.device(f"cuda:{device_id}")),
        "backend": "tvm_ffi",
        "all_expectations_met": all(expectations.values()),
        "expectations": expectations,
        "fault_severity": wrapped.get("fault_severity"),
        "error_type": wrapped.get("error_type"),
        "pool_timing": wrapped.get("pool_timing"),
        "pool_health": wrapped.get("pool_health"),
        "runtime_sanitizer": diagnostic,
    }


def run_containment_case(device_id: int = 0) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return asyncio.run(_run(device_id))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = run_containment_case(args.device_id)
    rendered = json.dumps(evidence, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if evidence["all_expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
