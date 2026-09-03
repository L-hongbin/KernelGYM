from __future__ import annotations

import asyncio
import os

import pytest
import torch

from benchmarks.run_runtime_sanitizer_cases import run_cases
from benchmarks.runtime_sanitizer_cases import CASES, KERNEL_CODE
from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref
from kernelgym.worker.subprocess_pool import FAULT_CONTEXT, SubprocessWorkerPool


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_COMPUTE_SANITIZER_INTEGRATION") != "1",
    reason="set RUN_COMPUTE_SANITIZER_INTEGRATION=1 to run Compute Sanitizer GPU cases",
)
def test_tvm_ffi_runtime_sanitizer_cases() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    evidence = run_cases("cuda:0")

    assert evidence["compiled"] is True
    assert evidence["all_expectations_met"] is True


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_COMPUTE_SANITIZER_INTEGRATION") != "1",
    reason="set RUN_COMPUTE_SANITIZER_INTEGRATION=1 to run Compute Sanitizer GPU cases",
)
def test_pipeline_defers_sanitizer_until_faulting_context_is_reaped() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    case = next(item for item in CASES if item.name == "global_oob")
    kernel_code = KERNEL_CODE.replace(
        "output[index] = index < n ? input[index] : 0.0f;",
        "if (index == 0) { *reinterpret_cast<volatile float*>(0x1) = 1.0f; }",
    )
    result = eval_kernel_against_ref(
        original_model_src=case.reference_code,
        custom_model_src=kernel_code,
        num_correct_trials=1,
        num_perf_trials=1,
        measure_performance=False,
        device="cuda:0",
        backend="tvm_ffi",
        entry_point="Model",
        enable_profiling=False,
        enable_compute_sanitizer=True,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend_adapter=KernelBenchBackend(),
        enable_compile_artifact_cache=True,
    )

    assert result.compiled is True
    assert result.correctness is False
    assert result.metadata["correctness_runtime_error_stage"] == "custom_forward"
    assert result.runtime_sanitizer["status"] == "pending"
    request = result.metadata["_runtime_sanitizer_request"]
    assert request["input_seed"] == result.metadata["correctness_failed_trial_seed"]
    assert request["mode"] == "memcheck"
    assert request["primary_tool"] == "memcheck"
    assert result.metadata["runtime_sanitizer_execution_location"] == "pool_parent_after_fault_reap"


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_COMPUTE_SANITIZER_INTEGRATION") != "1",
    reason="set RUN_COMPUTE_SANITIZER_INTEGRATION=1 to run Compute Sanitizer GPU cases",
)
def test_subprocess_pool_preserves_diagnostics_and_replaces_poisoned_context() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    case = next(item for item in CASES if item.name == "global_oob")
    kernel_code = KERNEL_CODE.replace(
        "output[index] = index < n ? input[index] : 0.0f;",
        "if (index == 0) { *reinterpret_cast<volatile float*>(0x1) = 1.0f; }",
    )

    async def scenario() -> tuple[dict, bool]:
        pool = SubprocessWorkerPool(
            device_id=0,
            pool_size=1,
            worker_prefix="sanitizer_containment",
            max_tasks_per_worker=2,
        )
        try:
            wrapped = await pool.execute_task(
                {
                    "task_id": "sanitizer-containment",
                    "base_task_id": "sanitizer-containment",
                    "task_type": "kernel_evaluation",
                    "toolkit": "kernelbench",
                    "backend_adapter": "kernelbench",
                    "backend": "tvm_ffi",
                    "reference_code": case.reference_code,
                    "kernel_code": kernel_code,
                    "entry_point": "Model",
                    "device": "cuda:0",
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
            return wrapped, await pool.shutdown(timeout=30)
        except BaseException:
            await pool.shutdown(timeout=30)
            raise

    wrapped, shutdown_safe = asyncio.run(scenario())
    result = wrapped["result"]

    assert wrapped["success"] is True
    assert wrapped["worker_exiting"] is True
    assert wrapped["fault_severity"] == FAULT_CONTEXT
    assert result["runtime_sanitizer"]["status"] == "issues_found"
    assert result["metadata"]["runtime_sanitizer_trigger"] == "correctness_runtime_error"
    assert "_runtime_sanitizer_request" not in result["metadata"]
    first_issue = result["runtime_sanitizer"]["check_results"][0]["issues"][0]
    assert first_issue["hazard_type"] == "invalid_global_write"
    assert wrapped["pool_timing"]["pool_restart_s"] > 0
    assert shutdown_safe is True
