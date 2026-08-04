"""ATen fallback legality and conservative coverage-decoy behavior."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kernelgym.toolkit.kernelbench import pipeline, profiling
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult


@pytest.mark.parametrize(
    "name",
    [
        "aten::empty",
        "aten::view",
        "aten::reshape",
        "aten::copy_",
        "aten::rand",
        "aten::item",
        "aten::view.default",
    ],
)
def test_musacoder_allowlist_accepts_tensor_plumbing(name: str) -> None:
    assert profiling.is_allowed_aten_operator(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "aten::to",
        "aten::_to_copy",
        "aten::detach",
        "aten::fill_",
        "aten::zero_",
        "aten::new_empty",
        "aten::empty_strided",
        "aten::to.dtype",
    ],
)
def test_kernelgym_compat_allowlist_accepts_modern_tensor_plumbing(name: str) -> None:
    assert profiling.is_allowed_aten_operator(name) is True


def test_musacoder_source_allowlist_is_kept_separate_from_compat_additions() -> None:
    assert "aten::_cast_Float" in profiling.MUSACODER_APPENDIX_J_ALLOWED_ATEN_OPERATORS
    assert "aten::to" not in profiling.MUSACODER_APPENDIX_J_ALLOWED_ATEN_OPERATORS
    assert "aten::to" in profiling.KERNELGYM_COMPAT_ALLOWED_ATEN_OPERATORS


@pytest.mark.parametrize(
    "name",
    [
        "aten::mm",
        "aten::matmul",
        "aten::convolution",
        "aten::sum",
        "aten::softmax",
        "aten::mm.default",
    ],
)
def test_unlisted_aten_compute_is_forbidden(name: str) -> None:
    assert profiling.is_allowed_aten_operator(name) is False


def test_extract_profiling_metrics_uses_only_cuda_device_events() -> None:
    cpu_aggregate = SimpleNamespace(
        key="aten::mm",
        device_type=torch.profiler.DeviceType.CPU,
        device_time_total=20.0,
        cpu_time_total=5.0,
        count=1,
    )
    cuda_kernel = SimpleNamespace(
        key="sgemm_kernel",
        device_type=torch.profiler.DeviceType.CUDA,
        device_time_total=20.0,
        cpu_time_total=0.0,
        count=1,
    )
    prof = SimpleNamespace(key_averages=lambda: [cpu_aggregate, cuda_kernel])

    metrics = profiling.extract_profiling_metrics(prof)

    assert metrics["kernel_count"] == 1
    assert metrics["kernels"][0]["name"] == "sgemm_kernel"
    assert metrics["total_cuda_time_us"] == 20.0


def _apply_coverage(custom_time: float, total_time: float) -> tuple[KernelExecResult, dict]:
    metadata: dict = {}
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    pipeline._apply_coverage_metadata(
        metadata=metadata,
        kernel_exec_result=result,
        coverage_result_dict={
            "num_custom_kernels": int(custom_time > 0),
            "num_total_kernels": int(total_time > 0),
            "custom_kernels_not_in_profiling": [],
            "custom_kernels_in_profiling": ["candidate_kernel"] if custom_time > 0 else [],
            "total_kernel_run_time_in_profiling_us": total_time,
            "total_kernel_cuda_time_in_profiling_us": total_time,
            "total_kernel_run_time_in_profiling_us_cpu_cuda": total_time,
            "custom_kernel_cuda_time_in_profiling_us": custom_time,
        },
        coverage_backend="cuda_agent",
        detect_decoy_kernel=True,
    )
    return result, metadata


def test_low_named_kernel_coverage_is_suspicion_not_hard_decoy() -> None:
    result, metadata = _apply_coverage(custom_time=0.05, total_time=100.0)

    assert result.decoy_kernel is False
    assert metadata["raw_custom_kernel_time_coverage"] == pytest.approx(0.0005)
    assert metadata["suspected_decoy"] is True
    assert metadata["suspected_decoy_enforced"] is False
    assert metadata["suspected_decoy_effect"] == "DIAGNOSTIC_ONLY"
    assert metadata["hard_decoy_coverage_candidate"] is True
    assert metadata["hard_decoy_coverage_gate_applied"] is False
    assert metadata["hard_decoy_coverage_gate_skip_reason"] == "ALLOWED_LIBRARY_PROVENANCE_UNAVAILABLE"


def test_high_named_kernel_coverage_is_not_suspected() -> None:
    result, metadata = _apply_coverage(custom_time=80.0, total_time=100.0)

    assert result.decoy_kernel is False
    assert metadata["raw_custom_kernel_time_coverage"] == pytest.approx(0.8)
    assert "suspected_decoy" not in metadata


def test_empty_coverage_capture_is_unavailable_not_decoy() -> None:
    result, metadata = _apply_coverage(custom_time=0.0, total_time=0.0)

    assert result.decoy_kernel is False
    assert metadata["coverage_measurement_valid"] is False
    assert metadata["coverage_unavailable"] is True
