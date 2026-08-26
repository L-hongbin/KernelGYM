from __future__ import annotations

import os

import pytest
import torch

from benchmarks.run_runtime_sanitizer_cases import run_cases
from benchmarks.runtime_sanitizer_cases import CASES, KERNEL_CODE
from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref


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
def test_tvm_ffi_sanitizer_runs_after_correctness_runtime_failure() -> None:
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
        enable_ncu=False,
        enable_compute_sanitizer=True,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend_adapter=KernelBenchBackend(),
        enable_compile_artifact_cache=True,
    )

    assert result.compiled is True
    assert result.correctness is False
    assert result.metadata["correctness_runtime_error_stage"] == "custom_forward"
    assert result.runtime_sanitizer["status"] == "issues_found"
    assert result.runtime_sanitizer["replayed_input_seed"] == result.metadata["correctness_failed_trial_seed"]
    assert result.runtime_sanitizer["executed_checks"] == ["memcheck"]
    first_issue = result.runtime_sanitizer["check_results"][0]["issues"][0]
    assert first_issue["hazard_type"] == "invalid_global_write"
    assert first_issue["kernel_info"] == [
        {
            "name": "sanitizer_oob_kernel",
            "source": "file generated.cu line 12",
        }
    ]
