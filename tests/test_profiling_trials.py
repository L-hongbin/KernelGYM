"""Profiling trial resolution and empty-capture retry semantics.

Background: CUDA 12.6u2-13.0 CUPTI can emit kernel activities with start=0 when
Kineto registers its TSC timestamp callback; Kineto drops those records and the
reward profiler captures zero CUDA kernels. See
docs/design-doc/PROFILER_EMPTY_CAPTURE.md. These tests pin the version gate,
the explicit/env overrides, and the pipeline retry + metadata bookkeeping.
"""

import pytest


def _get_timing():
    pytest.importorskip("torch")
    from kernelgym.toolkit.kernelbench import timing

    return timing


@pytest.mark.parametrize(
    ("cuda_version", "suspected"),
    [
        ("12.6", True),  # API version 24 cannot distinguish GA/U1/U2; conservative
        ("12.8", True),
        ("12.9", True),  # current .21/.22 runtime
        ("13.0", True),
        ("12.5", False),
        ("12.4", False),
        ("13.1", False),  # vendor fix ships here
        ("13.2", False),
        ("14.0", False),
    ],
)
def test_cupti_tsc_bug_version_gate(cuda_version, suspected) -> None:
    timing = _get_timing()
    assert timing.cupti_tsc_timestamp_bug_suspected(cuda_version=cuda_version, kineto_tsc_fixed=False) is suspected


@pytest.mark.parametrize("cuda_version", [None, "", "unknown", "garbage.version"])
def test_cupti_tsc_bug_gate_fails_safe_on_unknown_version(cuda_version) -> None:
    """An unparseable CUDA version must keep the workaround active, not drop it."""
    timing = _get_timing()
    assert timing.cupti_tsc_timestamp_bug_suspected(cuda_version=cuda_version, kineto_tsc_fixed=False) is True


def test_kineto_tsc_fixed_flag_disables_gate() -> None:
    """A patched Kineto build (declared via KINETO_TSC_FIXED) clears the suspicion."""
    timing = _get_timing()
    assert timing.cupti_tsc_timestamp_bug_suspected(cuda_version="12.9", kineto_tsc_fixed=True) is False


def test_resolve_explicit_configuration_wins() -> None:
    timing = _get_timing()
    assert timing.resolve_num_profiling_trials(100, configured=1, cuda_version="12.9") == 1
    assert timing.resolve_num_profiling_trials(100, configured=5, cuda_version="13.1") == 5


def test_resolve_auto_keeps_legacy_workaround_on_affected_cupti() -> None:
    timing = _get_timing()
    resolve = timing.resolve_num_profiling_trials
    # Production shape: num_trials=100 keeps the historical 10-forward workaround.
    assert resolve(100, configured=-1, cuda_version="12.9") == 10
    # Legacy min(10, num_trials) shape is preserved for small trial counts.
    assert resolve(4, configured=-1, cuda_version="12.9") == 4
    # At least one forward even for degenerate trial counts.
    assert resolve(0, configured=-1, cuda_version="12.9") == 1


def test_resolve_auto_single_trial_once_bug_absent() -> None:
    timing = _get_timing()
    resolve = timing.resolve_num_profiling_trials
    assert resolve(100, configured=-1, cuda_version="13.1") == 1
    assert resolve(100, configured=-1, cuda_version="12.9", kineto_tsc_fixed=True) == 1


def test_settings_defaults_and_env_overrides(monkeypatch) -> None:
    pytest.importorskip("torch")
    from kernelgym.config.settings import Settings

    defaults = Settings(_env_file=None)
    assert defaults.num_profiling_trials == -1
    assert defaults.kineto_tsc_fixed is False

    monkeypatch.setenv("NUM_PROFILING_TRIALS", "1")
    monkeypatch.setenv("KINETO_TSC_FIXED", "true")
    overridden = Settings(_env_file=None)
    assert overridden.num_profiling_trials == 1
    assert overridden.kineto_tsc_fixed is True


def _make_pipeline_mocks(monkeypatch, retry_metrics_factory, retry_count, num_perf_trials=100):
    import torch

    from kernelgym.toolkit.kernelbench import pipeline, timing

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.settings, "profiling_retry_count", retry_count)

    resolved = timing.resolve_num_profiling_trials(num_perf_trials)
    timing_info = {
        "warmup_wall_s": 0.0,
        "measure_wall_s": 0.0,
        "profiling_wall_s": 0.0,
        "timed_trials_cuda_event_s": 0.0,
        "num_warmup": 1,
        "num_trials": num_perf_trials,
        "num_profiling_trials": resolved,
        "total_wall_s": 0.0,
    }
    monkeypatch.setattr(
        pipeline,
        "time_execution_with_cuda_event",
        lambda *args, **kwargs: ([1.0, 1.0], {}, timing_info),
    )

    retry_calls = []

    def fake_run_profiling_only(model, *args, num_trials, verbose, device):
        retry_calls.append(num_trials)
        return retry_metrics_factory()

    monkeypatch.setattr(pipeline, "run_profiling_only", fake_run_profiling_only)
    return pipeline, retry_calls, resolved


def _run_perf_step(pipeline, num_perf_trials=100):
    from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult

    class FakeModel:
        def cuda(self, device=None):
            return self

    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    metadata = {}
    pipeline._run_performance_step(
        kernel_exec_result=result,
        custom_model=FakeModel(),
        get_inputs=lambda: [],
        metadata=metadata,
        num_perf_trials=num_perf_trials,
        num_warmup=1,
        perf_trim_count=0,
        verbose=False,
        seed_num=42,
        device=None,
        enable_profiling=True,
        enable_triton_detection=False,
        detect_decoy_kernel=True,
        backend="tvm_ffi",
        backend_profiling_hints={"custom_kernel_names": ["my_kernel"]},
    )
    return result, metadata


def test_empty_capture_retry_uses_resolved_trials_and_records_metadata(monkeypatch) -> None:
    pytest.importorskip("torch")

    def non_empty_retry():
        return {
            "kernels": [{"name": "my_kernel", "cuda_time_us": 10.0, "cpu_time_us": 1.0, "count": 1}],
            "kernel_count": 1,
            "total_cuda_time_us": 10.0,
            "total_cpu_time_us": 1.0,
        }

    pipeline, retry_calls, resolved = _make_pipeline_mocks(monkeypatch, non_empty_retry, retry_count=1)
    result, metadata = _run_perf_step(pipeline)

    # The retry must use the resolved profiling-trial count, not a hardcoded 10.
    assert retry_calls == [resolved]
    assert metadata["kg_kernel_profiling_empty_initial"] is True
    assert metadata["kg_kernel_profiling_retries_used"] == 1
    assert metadata["kg_kernel_profiling_empty_final"] is False
    assert metadata["kg_kernel_perf_num_profile_trials"] == resolved
    # Recovered capture flows into coverage and no decoy flag is raised.
    assert result.decoy_kernel is False
    assert metadata["profiling"]["kernel_count"] == 1


def test_empty_capture_after_all_retries_is_not_marked_decoy(monkeypatch) -> None:
    pytest.importorskip("torch")

    def still_empty_retry():
        return {
            "kernels": [],
            "kernel_count": 0,
            "total_cuda_time_us": 0.0,
            "total_cpu_time_us": 0.0,
            "profiling_warning": "Profiler captured no CUDA kernels. This may indicate a profiler failure.",
        }

    pipeline, retry_calls, resolved = _make_pipeline_mocks(monkeypatch, still_empty_retry, retry_count=2)
    result, metadata = _run_perf_step(pipeline)

    assert retry_calls == [resolved, resolved]
    assert metadata["kg_kernel_profiling_empty_initial"] is True
    assert metadata["kg_kernel_profiling_retries_used"] == 2
    assert metadata["kg_kernel_profiling_empty_final"] is True
    # Empty capture is a profiler failure, never a decoy verdict.
    assert result.decoy_kernel is False


def test_non_empty_first_capture_records_zero_retries(monkeypatch) -> None:
    pytest.importorskip("torch")
    import torch

    from kernelgym.toolkit.kernelbench import pipeline, timing

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.settings, "profiling_retry_count", 1)

    resolved = timing.resolve_num_profiling_trials(100)
    timing_info = {
        "warmup_wall_s": 0.0,
        "measure_wall_s": 0.0,
        "profiling_wall_s": 0.0,
        "timed_trials_cuda_event_s": 0.0,
        "num_warmup": 1,
        "num_trials": 100,
        "num_profiling_trials": resolved,
        "total_wall_s": 0.0,
    }
    profiling_metrics = {
        "kernels": [{"name": "my_kernel", "cuda_time_us": 5.0, "cpu_time_us": 1.0, "count": 1}],
        "kernel_count": 1,
        "total_cuda_time_us": 5.0,
        "total_cpu_time_us": 1.0,
    }
    monkeypatch.setattr(
        pipeline,
        "time_execution_with_cuda_event",
        lambda *args, **kwargs: ([1.0, 1.0], profiling_metrics, timing_info),
    )

    def fail_retry(*args, **kwargs):
        raise AssertionError("run_profiling_only must not be called when the first capture is non-empty")

    monkeypatch.setattr(pipeline, "run_profiling_only", fail_retry)

    result, metadata = _run_perf_step(pipeline)
    assert metadata["kg_kernel_profiling_empty_initial"] is False
    assert metadata["kg_kernel_profiling_retries_used"] == 0
    assert metadata["kg_kernel_profiling_empty_final"] is False
    assert result.decoy_kernel is False
