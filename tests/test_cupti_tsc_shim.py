"""CUPTI TSC shim: build, service-env injection, and verification gates.

See docs/design-doc/PROFILER_EMPTY_CAPTURE.md. The shim suppresses Kineto's
TSC timestamp callback on CUPTI versions affected by the CUDA 12.6u2-13.0
start=0 bug and flips Kineto to native timestamps, enabling single-forward
profiling. These tests cover the pieces that do not need a GPU.
"""

import ctypes
import shutil

import pytest

from kernelgym.utils import cupti_tsc_shim


def test_builder_produces_loadable_shim_with_state_symbols(tmp_path, monkeypatch) -> None:
    if shutil.which("g++") is None and shutil.which("c++") is None:
        pytest.skip("no C++ compiler available")
    monkeypatch.setattr(cupti_tsc_shim, "SHIM_BUILD_DIR", tmp_path)

    artifact = cupti_tsc_shim.ensure_shim_built()
    assert artifact is not None and artifact.exists()

    lib = ctypes.CDLL(str(artifact))
    # Freshly loaded, never invoked by Kineto: state must be NOT_CALLED.
    assert int(lib.kernelgym_cupti_tsc_shim_state()) == cupti_tsc_shim.STATE_NOT_CALLED
    assert int(lib.kernelgym_cupti_tsc_shim_cupti_version()) == 0

    # A second call is a no-op cache hit on the same artifact.
    assert cupti_tsc_shim.ensure_shim_built() == artifact


def test_service_env_injection_and_fail_open(monkeypatch, tmp_path) -> None:
    from kernelgym.cli import service

    monkeypatch.delenv(cupti_tsc_shim.SHIM_FLAG_ENV, raising=False)
    fake = tmp_path / "libshim.so"
    fake.write_bytes(b"")
    monkeypatch.setattr(cupti_tsc_shim, "ensure_shim_built", lambda: fake)

    env = service._with_cupti_tsc_shim({"KERNELGYM_CUPTI_TSC_SHIM": "true", "LD_PRELOAD": "/opt/other.so"})
    assert env["LD_PRELOAD"] == f"{fake}:/opt/other.so"
    assert env["KINETO_TSC_FIXED"] == "true"
    assert env[cupti_tsc_shim.SHIM_EXPECTED_ENV] == str(fake)

    # Build failure: fail open — nothing injected, legacy workaround stays.
    monkeypatch.setattr(cupti_tsc_shim, "ensure_shim_built", lambda: None)
    env = service._with_cupti_tsc_shim({"KERNELGYM_CUPTI_TSC_SHIM": "true"})
    assert "LD_PRELOAD" not in env
    assert "KINETO_TSC_FIXED" not in env
    assert cupti_tsc_shim.SHIM_EXPECTED_ENV not in env

    # Flag off: untouched.
    env = service._with_cupti_tsc_shim({"KERNELGYM_CUPTI_TSC_SHIM": "false"})
    assert "KINETO_TSC_FIXED" not in env

    # Operator's ambient env is the emergency off switch over the profile value.
    monkeypatch.setattr(cupti_tsc_shim, "ensure_shim_built", lambda: fake)
    monkeypatch.setenv(cupti_tsc_shim.SHIM_FLAG_ENV, "false")
    env = service._with_cupti_tsc_shim({"KERNELGYM_CUPTI_TSC_SHIM": "true"})
    assert "KINETO_TSC_FIXED" not in env


@pytest.mark.parametrize(
    ("kineto_tsc_fixed", "expected_path", "state", "verdict"),
    [
        (False, None, None, None),  # no fix declared
        (True, None, None, True),  # custom Kineto build declared, no shim: trusted
        (True, "/x/shim.so", cupti_tsc_shim.STATE_ENGAGED_NATIVE, True),
        (True, "/x/shim.so", cupti_tsc_shim.STATE_PASSTHROUGH_FIXED, True),
        (True, "/x/shim.so", cupti_tsc_shim.STATE_NOT_CALLED, False),
        (True, "/x/shim.so", cupti_tsc_shim.STATE_PASSTHROUGH_ERROR, False),
        (True, "/x/shim.so", cupti_tsc_shim.STATE_FAILED, False),
        (True, "/x/shim.so", None, False),  # shim expected but not even loaded
    ],
)
def test_kineto_tsc_fix_verified(monkeypatch, kineto_tsc_fixed, expected_path, state, verdict) -> None:
    if expected_path is None:
        monkeypatch.delenv(cupti_tsc_shim.SHIM_EXPECTED_ENV, raising=False)
    else:
        monkeypatch.setenv(cupti_tsc_shim.SHIM_EXPECTED_ENV, expected_path)
    monkeypatch.setattr(cupti_tsc_shim, "shim_state", lambda: state)

    assert cupti_tsc_shim.kineto_tsc_fix_verified(kineto_tsc_fixed) is verdict


def test_retry_falls_back_to_legacy_trials_when_shim_not_engaged(monkeypatch) -> None:
    """With KINETO_TSC_FIXED declared but the shim not engaged, the empty-capture
    retry must use the legacy multi-forward count, not a single forward."""
    pytest.importorskip("torch")
    import torch

    from kernelgym.toolkit.kernelbench import pipeline
    from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.settings, "profiling_retry_count", 1)
    monkeypatch.setattr(pipeline.settings, "kineto_tsc_fixed", True)
    monkeypatch.setattr(pipeline, "timing_kineto_tsc_fix_verified", lambda: False)

    timing_info = {
        "warmup_wall_s": 0.0,
        "measure_wall_s": 0.0,
        "profiling_wall_s": 0.0,
        "timed_trials_cuda_event_s": 0.0,
        "num_warmup": 1,
        "num_trials": 100,
        "num_profiling_trials": 1,
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
        return {"kernels": [], "kernel_count": 0, "total_cuda_time_us": 0.0, "total_cpu_time_us": 0.0}

    monkeypatch.setattr(pipeline, "run_profiling_only", fake_run_profiling_only)

    class FakeModel:
        def cuda(self, device=None):
            return self

    metadata = {}
    pipeline._run_performance_step(
        kernel_exec_result=KernelExecResult(compiled=True, correctness=True, metadata={}),
        custom_model=FakeModel(),
        get_inputs=lambda: [],
        metadata=metadata,
        num_perf_trials=100,
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

    # Legacy min(10, num_trials) workaround, not the declared-but-unverified 1.
    assert retry_calls == [10]
    assert metadata["kg_kernel_profiling_retries_used"] == 1
