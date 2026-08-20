"""KernelBench GPU tests for profiler empty-capture behavior.

CUDA 12.6u2-13.0 CUPTI can emit start=0 kernel timestamps under Kineto's TSC
callback, which Kineto drops, leaving the reward profiler with zero CUDA
kernels — mostly on slow kernels (~1/10 non-empty for a single forward on an
L2 P90 sample). The resolved profiling-trial count must keep captures
non-empty on the current runtime, whether via the legacy multi-forward
workaround (affected CUPTI) or a single forward (fixed CUPTI / patched
Kineto). See docs/design-doc/PROFILER_EMPTY_CAPTURE.md.
"""

import pytest


def _require_cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is not available")
    return torch


def _get_timing_module():
    pytest.importorskip("torch")
    from kernelgym.toolkit.kernelbench import timing

    return timing


def _calibrate_sleep_cycles(torch, device, target_ms: float) -> int:
    """Scale torch.cuda._sleep cycles so one call busy-spins for ~target_ms."""
    probe_cycles = 2_000_000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device=device)
    start.record()
    torch.cuda._sleep(probe_cycles)
    end.record()
    torch.cuda.synchronize(device=device)
    probe_ms = max(start.elapsed_time(end), 1e-3)
    return max(probe_cycles, int(probe_cycles * target_ms / probe_ms))


@pytest.mark.gpu
def test_slow_kernel_repeated_profiler_contexts_capture_kernels() -> None:
    """Consecutive profiler contexts on a slow kernel must each capture CUDA kernels.

    This is the regression test for the empty-capture failure mode: every
    context must report at least one kernel with a name and a positive CUDA
    duration when using the production-resolved profiling-trial count.
    """
    torch = _require_cuda_runtime()
    timing = _get_timing_module()

    device = torch.device("cuda:0")
    cycles = _calibrate_sleep_cycles(torch, device, target_ms=120.0)

    def slow_forward(x):
        torch.cuda._sleep(cycles)
        return x

    inputs = torch.randn(64, device=device)
    # The same count production uses for the default num_trials=100 shape.
    resolved_trials = timing.resolve_num_profiling_trials(100)

    for context_idx in range(3):
        _, profiling_metrics, timing_info = timing.time_execution_with_cuda_event(
            slow_forward,
            inputs,
            num_warmup=1,
            num_trials=2,
            verbose=False,
            device=device,
            enable_profiling=True,
            num_profiling_trials=resolved_trials,
        )

        assert timing_info["num_profiling_trials"] == resolved_trials
        assert "profiling_error" not in profiling_metrics, (
            f"context {context_idx}: profiler error {profiling_metrics.get('profiling_error')}"
        )
        kernels = profiling_metrics.get("kernels", [])
        assert len(kernels) > 0, f"context {context_idx}: profiler captured no CUDA kernels"
        for kernel in kernels:
            assert isinstance(kernel["name"], str) and kernel["name"], (
                f"context {context_idx}: kernel entry without a name: {kernel}"
            )
            assert kernel["cuda_time_us"] > 0.0, (
                f"context {context_idx}: kernel {kernel['name']} has non-positive CUDA duration"
            )


@pytest.mark.gpu
def test_explicit_num_profiling_trials_controls_forward_count() -> None:
    torch = _require_cuda_runtime()
    timing = _get_timing_module()

    device = torch.device("cuda:0")
    forward_count = {"n": 0}

    def counted_forward(x):
        forward_count["n"] += 1
        return x + 1

    _, _, timing_info = timing.time_execution_with_cuda_event(
        counted_forward,
        torch.randn(64, device=device),
        num_warmup=1,
        num_trials=2,
        verbose=False,
        device=device,
        enable_profiling=True,
        num_profiling_trials=3,
    )

    assert timing_info["num_profiling_trials"] == 3
    # warmup (1) + CUDA-event trials (2) + profiling forwards (3)
    assert forward_count["n"] == 1 + 2 + 3


@pytest.mark.gpu
def test_auto_resolution_reports_resolved_count_in_timing_info() -> None:
    torch = _require_cuda_runtime()
    timing = _get_timing_module()

    device = torch.device("cuda:0")

    def forward(x):
        return x + 1

    _, _, timing_info = timing.time_execution_with_cuda_event(
        forward,
        torch.randn(64, device=device),
        num_warmup=1,
        num_trials=2,
        verbose=False,
        device=device,
        enable_profiling=True,
    )

    assert timing_info["num_profiling_trials"] == timing.resolve_num_profiling_trials(2)
