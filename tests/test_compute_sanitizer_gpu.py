from __future__ import annotations

import os

import pytest
import torch

from benchmarks.run_runtime_sanitizer_cases import run_cases
from benchmarks.runtime_sanitizer_cases import CASES, KERNEL_CODE
from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
from kernelgym.toolkit.kernelbench import pipeline as kernelbench_pipeline
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
@pytest.mark.parametrize("detail", [False, True])
def test_tvm_ffi_sanitizer_runs_after_correctness_runtime_failure(detail) -> None:
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
        return_detail_correctness=detail,
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


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_COMPUTE_SANITIZER_INTEGRATION") != "1",
    reason="set RUN_COMPUTE_SANITIZER_INTEGRATION=1 to run Compute Sanitizer GPU cases",
)
@pytest.mark.parametrize("detail", [False, True])
def test_tvm_ffi_sanitizer_runs_after_output_mismatch_and_omits_clean_result(monkeypatch, detail) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    case = next(item for item in CASES if item.name == "safe")
    mismatching_reference = case.reference_code.replace("return x.clone()", "return x.clone() + 1.0")
    observed_sanitizer_result = {}
    original_run_compute_sanitizer = kernelbench_pipeline.run_compute_sanitizer

    def recording_run_compute_sanitizer(**kwargs):
        sanitizer_result = original_run_compute_sanitizer(**kwargs)
        observed_sanitizer_result.update(sanitizer_result)
        return sanitizer_result

    monkeypatch.setattr(kernelbench_pipeline, "run_compute_sanitizer", recording_run_compute_sanitizer)
    result = eval_kernel_against_ref(
        original_model_src=mismatching_reference,
        custom_model_src=KERNEL_CODE,
        num_correct_trials=1,
        num_perf_trials=1,
        measure_performance=False,
        device="cuda:0",
        backend="tvm_ffi",
        entry_point="Model",
        enable_profiling=False,
        enable_ncu=False,
        enable_compute_sanitizer=True,
        return_detail_correctness=detail,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend_adapter=KernelBenchBackend(),
        enable_compile_artifact_cache=True,
    )

    assert result.compiled is True
    assert result.correctness is False
    assert result.metadata["correctness_issue_name"] == "numerical_mismatch"
    assert observed_sanitizer_result["status"] == "clean"
    assert observed_sanitizer_result["requested_checks"] == [
        "racecheck",
        "initcheck",
        "memcheck",
        "synccheck",
    ]
    assert observed_sanitizer_result["executed_checks"] == observed_sanitizer_result["requested_checks"]
    assert observed_sanitizer_result["diagnostic_policy_complete"] is True
    assert result.runtime_sanitizer == {}
    assert not any(key.startswith("runtime_sanitizer_") for key in result.metadata)
    assert ("element_correctness_curve" in result.metadata) is detail
    assert "correctness_diagnosis" not in result.metadata


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_COMPUTE_SANITIZER_INTEGRATION") != "1",
    reason="set RUN_COMPUTE_SANITIZER_INTEGRATION=1 to run Compute Sanitizer GPU cases",
)
@pytest.mark.parametrize("detail", [False, True])
def test_tvm_ffi_sanitizer_reports_issue_after_output_mismatch(detail) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    case = next(item for item in CASES if item.name == "shared_race")
    mismatching_kernel = KERNEL_CODE.replace(
        """    __syncthreads();
    if (index < n) {
        output[index] = input[index];
    }
}

__global__ void sanitizer_sync_kernel""",
        """    __syncthreads();
    if (index < n) {
        output[index] = input[index] + 1.0f;
    }
}

__global__ void sanitizer_sync_kernel""",
    )
    assert mismatching_kernel != KERNEL_CODE
    result = eval_kernel_against_ref(
        original_model_src=case.reference_code,
        custom_model_src=mismatching_kernel,
        num_correct_trials=1,
        num_perf_trials=1,
        measure_performance=False,
        device="cuda:0",
        backend="tvm_ffi",
        entry_point="Model",
        enable_profiling=False,
        enable_ncu=False,
        enable_compute_sanitizer=True,
        return_detail_correctness=detail,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend_adapter=KernelBenchBackend(),
        enable_compile_artifact_cache=True,
    )

    assert result.compiled is True
    assert result.correctness is False
    assert result.metadata["correctness_issue_name"] == "numerical_mismatch"
    assert result.metadata["runtime_sanitizer_trigger"] == "correctness_output_mismatch"
    assert result.runtime_sanitizer["status"] == "issues_found"
    assert result.runtime_sanitizer["replayed_input_seed"] == result.metadata["correctness_failed_trial_seed"]
    assert result.metadata["runtime_sanitizer_execution_policy"] == "mismatch_likely_first"
    assert result.metadata["runtime_sanitizer_run_all_checks"] is False
    assert result.runtime_sanitizer["executed_checks"] == ["racecheck"]
    assert result.runtime_sanitizer["skipped_checks"] == ["initcheck", "memcheck", "synccheck"]
    assert result.runtime_sanitizer["stop_reason"] == "first_issue"
    assert result.runtime_sanitizer["measurement_complete"] is False
    assert result.runtime_sanitizer["diagnostic_policy_complete"] is True
    assert result.runtime_sanitizer["issue_count_by_check"]["racecheck"] > 0
    assert ("element_correctness_curve" in result.metadata) is detail
    assert "correctness_diagnosis" not in result.metadata
