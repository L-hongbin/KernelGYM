"""KernelBench ATen fallback legality and coverage-decoy behavior."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kernelgym.toolkit.kernelbench import pipeline, profiling
from kernelgym.toolkit.kernelbench.correctness import apply_aten_detection_policy
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


def test_unavailable_aten_capture_is_a_fail_closed_policy_violation() -> None:
    metrics = profiling.extract_aten_operator_metrics(None)
    assert metrics["aten_detection_valid"] is False
    metadata: dict = {}
    apply_aten_detection_policy(metadata, [{"trial": 0, "error": metrics["aten_detection_error"]}])
    assert metadata["policy_violation"] is True
    assert metadata["policy_violation_reason"] == "ATEN_DETECTION_UNAVAILABLE"


def test_profiler_stop_failure_invalidates_aten_capture(monkeypatch) -> None:
    class StopFailingProfiler:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise RuntimeError("stop failed")

        def key_averages(self):
            raise AssertionError("invalid capture must not be read")

    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: StopFailingProfiler())
    with profiling.aten_operator_profiling_context(enabled=True) as capture:
        pass

    metrics = profiling.extract_aten_operator_metrics(capture)
    assert metrics["aten_detection_valid"] is False
    assert "failed to stop: stop failed" in metrics["aten_detection_error"]


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


class _ProbeModel:
    def __init__(self) -> None:
        self.calls = 0

    def cuda(self, device=None):
        return self

    def eval(self):
        return self

    def __call__(self, *_args):
        self.calls += 1
        return None


def _incorrect_probe_result(**metadata_overrides) -> tuple[KernelExecResult, dict]:
    metadata = {
        "correctness_candidate_forward_completed": True,
        "correctness_output_mismatch": True,
        **metadata_overrides,
    }
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata), metadata


def test_incorrect_cuda_probe_marks_missing_expected_kernel_as_decoy(monkeypatch) -> None:
    result, metadata = _incorrect_probe_result()
    model = _ProbeModel()
    monkeypatch.setattr(pipeline.torch.cuda, "synchronize", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "set_seed", lambda _seed: None)

    def fake_profile(kernel_fn, *args, num_trials, verbose, device):
        assert num_trials == 1
        assert verbose is False
        kernel_fn(*args)
        return {
            "kernels": [{"name": "unrelated_kernel", "cuda_time_us": 2.0, "cpu_time_us": 0.0}],
            "kernel_count": 1,
            "total_cuda_time_us": 2.0,
        }

    monkeypatch.setattr(pipeline, "run_profiling_only", fake_profile)

    detected = pipeline._run_incorrect_backend_usage_probe(
        kernel_exec_result=result,
        custom_model=model,
        get_inputs=lambda: [object()],
        metadata=metadata,
        seed_num=42,
        device="cuda:0",
        backend="cuda_agent",
        backend_profiling_hints={"custom_kernel_names": ["candidate_kernel"]},
        detect_decoy_kernel=True,
    )

    assert detected is True
    assert model.calls == 1
    assert result.decoy_kernel is True
    assert metadata["policy_violation_reason"] == "BACKEND_CUSTOM_KERNEL_NOT_OBSERVED"
    assert metadata["incorrect_backend_usage_probe"]["num_forwards"] == 1
    assert metadata["incorrect_backend_usage_probe"]["valid"] is True


def test_incorrect_tvm_ffi_probe_accepts_observed_expected_kernel(monkeypatch) -> None:
    result, metadata = _incorrect_probe_result()
    model = _ProbeModel()
    monkeypatch.setattr(pipeline.torch.cuda, "synchronize", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "set_seed", lambda _seed: None)

    def fake_profile(kernel_fn, *args, num_trials, verbose, device):
        assert num_trials == 1
        kernel_fn(*args)
        return {
            "kernels": [{"name": "void candidate_kernel(float*)", "cuda_time_us": 2.0, "cpu_time_us": 0.0}],
            "kernel_count": 1,
            "total_cuda_time_us": 2.0,
        }

    monkeypatch.setattr(pipeline, "run_profiling_only", fake_profile)

    detected = pipeline._run_incorrect_backend_usage_probe(
        kernel_exec_result=result,
        custom_model=model,
        get_inputs=lambda: [object()],
        metadata=metadata,
        seed_num=42,
        device="cuda:0",
        backend="tvm_ffi",
        backend_profiling_hints={"custom_kernel_names": ["candidate_kernel"]},
        detect_decoy_kernel=True,
    )

    assert detected is False
    assert model.calls == 1
    assert result.decoy_kernel is False
    assert metadata["incorrect_backend_usage_probe"]["custom_kernel_observed"] is True


@pytest.mark.parametrize(
    ("metadata_overrides", "expected_skip_reason"),
    [
        ({"runtime_error": "CUDA illegal memory access"}, "RUNTIME_ERROR"),
        ({"correctness_candidate_forward_completed": False}, "CANDIDATE_FORWARD_NOT_COMPLETED"),
        ({"correctness_output_mismatch": False}, "NO_OUTPUT_MISMATCH"),
    ],
)
def test_incorrect_backend_probe_does_not_rerun_unsafe_or_non_mismatch_failures(
    monkeypatch, metadata_overrides, expected_skip_reason
) -> None:
    result, metadata = _incorrect_probe_result(**metadata_overrides)
    monkeypatch.setattr(
        pipeline,
        "run_profiling_only",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    detected = pipeline._run_incorrect_backend_usage_probe(
        kernel_exec_result=result,
        custom_model=_ProbeModel(),
        get_inputs=lambda: [object()],
        metadata=metadata,
        seed_num=42,
        device="cuda:0",
        backend="cuda_agent",
        backend_profiling_hints={"custom_kernel_names": ["candidate_kernel"]},
        detect_decoy_kernel=True,
    )

    assert detected is False
    assert result.decoy_kernel is False
    assert metadata["incorrect_backend_usage_probe"]["attempted"] is False
    assert metadata["incorrect_backend_usage_probe"]["skip_reason"] == expected_skip_reason


def test_incorrect_backend_probe_respects_disabled_decoy_detection(monkeypatch) -> None:
    result, metadata = _incorrect_probe_result()
    monkeypatch.setattr(
        pipeline,
        "run_profiling_only",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    detected = pipeline._run_incorrect_backend_usage_probe(
        kernel_exec_result=result,
        custom_model=_ProbeModel(),
        get_inputs=lambda: [object()],
        metadata=metadata,
        seed_num=42,
        device="cuda:0",
        backend="cuda_agent",
        backend_profiling_hints={"custom_kernel_names": ["candidate_kernel"]},
        detect_decoy_kernel=False,
    )

    assert detected is False
    assert result.decoy_kernel is False
    assert metadata["incorrect_backend_usage_probe"]["skip_reason"] == "DECOY_DETECTION_DISABLED"


def test_incorrect_backend_probe_empty_capture_fails_open(monkeypatch) -> None:
    result, metadata = _incorrect_probe_result()
    model = _ProbeModel()
    monkeypatch.setattr(pipeline.torch.cuda, "synchronize", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "set_seed", lambda _seed: None)

    def fake_empty_profile(kernel_fn, *args, num_trials, verbose, device):
        assert num_trials == 1
        kernel_fn(*args)
        return {"kernels": [], "kernel_count": 0, "total_cuda_time_us": 0.0}

    monkeypatch.setattr(pipeline, "run_profiling_only", fake_empty_profile)

    detected = pipeline._run_incorrect_backend_usage_probe(
        kernel_exec_result=result,
        custom_model=model,
        get_inputs=lambda: [object()],
        metadata=metadata,
        seed_num=42,
        device="cuda:0",
        backend="tvm_ffi",
        backend_profiling_hints={"custom_kernel_names": ["candidate_kernel"]},
        detect_decoy_kernel=True,
    )

    assert detected is False
    assert model.calls == 1
    assert result.decoy_kernel is False
    assert metadata["incorrect_backend_usage_probe"]["valid"] is False
    assert metadata["incorrect_backend_usage_probe"]["skip_reason"] == "EMPTY_PROFILER_CAPTURE"


def test_incorrect_backend_probe_invalid_coverage_fails_open(monkeypatch) -> None:
    result, metadata = _incorrect_probe_result()
    model = _ProbeModel()
    monkeypatch.setattr(pipeline.torch.cuda, "synchronize", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "set_seed", lambda _seed: None)

    def fake_profile(kernel_fn, *args, num_trials, verbose, device):
        assert num_trials == 1
        kernel_fn(*args)
        return {
            "kernels": [{"name": "unrelated_kernel", "cuda_time_us": 1.0}],
            "kernel_count": 1,
            "total_cuda_time_us": 1.0,
        }

    monkeypatch.setattr(pipeline, "run_profiling_only", fake_profile)
    monkeypatch.setattr(
        pipeline,
        "compute_named_kernel_coverage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("malformed coverage")),
    )

    detected = pipeline._run_incorrect_backend_usage_probe(
        kernel_exec_result=result,
        custom_model=model,
        get_inputs=lambda: [object()],
        metadata=metadata,
        seed_num=42,
        device="cuda:0",
        backend="cuda_agent",
        backend_profiling_hints={"custom_kernel_names": ["candidate_kernel"]},
        detect_decoy_kernel=True,
    )

    assert detected is False
    assert model.calls == 1
    assert result.decoy_kernel is False
    assert metadata["incorrect_backend_usage_probe"]["valid"] is False
    assert metadata["incorrect_backend_usage_probe"]["error"] == "malformed coverage"


def test_incorrect_backend_probe_without_expected_names_does_not_rerun(monkeypatch) -> None:
    result, metadata = _incorrect_probe_result()
    monkeypatch.setattr(
        pipeline,
        "run_profiling_only",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    detected = pipeline._run_incorrect_backend_usage_probe(
        kernel_exec_result=result,
        custom_model=_ProbeModel(),
        get_inputs=lambda: [object()],
        metadata=metadata,
        seed_num=42,
        device="cuda:0",
        backend="cuda_agent",
        backend_profiling_hints={"custom_kernel_names": []},
        detect_decoy_kernel=True,
    )

    assert detected is False
    assert result.decoy_kernel is False
    assert metadata["incorrect_backend_usage_probe"]["attempted"] is False
    assert metadata["incorrect_backend_usage_probe"]["skip_reason"] == "NO_EXPECTED_CUSTOM_KERNEL_NAMES"
