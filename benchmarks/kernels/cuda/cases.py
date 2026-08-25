"""CUDA TVM-FFI cases exposed through the shared language-matrix interface."""

from __future__ import annotations

import json
import time
import urllib.request
import uuid
from typing import Any

from benchmarks.kernels.tvm_ffi_vector_add import BACKEND, KERNEL_CODE, REFERENCE_CODE

URL = "http://127.0.0.1:20111/evaluate"

# Keep the tuple shape common with the Python-DSL case modules:
# name, reference source, candidate source, precision, backend.
CASES = [("vector_add_fp32", REFERENCE_CODE, KERNEL_CODE, "fp32", BACKEND)]


def run(case: tuple[str, str, str, str, str], split: bool = False) -> dict[str, Any]:
    name, reference, kernel, precision, backend = case
    payload = {
        "task_id": f"language-matrix-cuda-tvm-ffi-{name}-{uuid.uuid4().hex[:8]}",
        "reference_code": reference,
        "kernel_code": kernel,
        "backend": backend,
        "precision": precision,
        "num_correct_trials": 3,
        "num_perf_trials": 10,
        "num_warmup": 3,
        "perf_trim_count": 1,
        "timeout": 600,
        "force_refresh": True,
        "enable_profiling": True,
        "detect_decoy_kernel": True,
        "split_compile_and_execute": split,
        "enable_compile_artifact_cache": split,
    }
    started = time.time()
    request = urllib.request.Request(
        URL,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=660) as response:
            result = json.load(response)
    except Exception as exc:  # noqa: BLE001 - benchmark must report transport failures
        return {"name": name, "transport_error": repr(exc), "wall_s": round(time.time() - started, 2)}
    metadata = result.get("metadata") or {}
    profiling = metadata.get("profiling") or {}
    return {
        "name": name,
        "backend": backend,
        "status": result.get("status"),
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "decoy": result.get("decoy_kernel"),
        "runtime_ms": result.get("kernel_runtime"),
        "speedup": result.get("speedup"),
        "profile_kernels": profiling.get("kernel_count"),
        "precompiled": metadata.get("precompiled_artifact_used"),
        "error": result.get("error_message"),
        "wall_s": round(time.time() - started, 2),
    }
