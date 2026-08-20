"""KernelBench timing GPU tests."""

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


@pytest.mark.gpu
def test_timing_disables_autograd_graph_for_parameterized_model() -> None:
    """The perf-timing window must run forwards with autograd disabled.

    A model holding an nn.Parameter (requires_grad=True by default) would
    otherwise build and discard an autograd graph on every trial. We assert
    the timing helper disables grad and that produced outputs carry no grad
    bookkeeping.
    """
    torch = _require_cuda_runtime()
    timing = _get_timing_module()

    device = torch.device("cuda:0")

    class Param(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(256, 256, device=device))

        def forward(self, x):
            return x @ self.w

    model = Param().to(device)
    # Sanity: the parameter really does request gradients.
    assert model.w.requires_grad is True

    observed = {"grad_enabled": [], "out_requires_grad": []}

    def spy(x):
        observed["grad_enabled"].append(torch.is_grad_enabled())
        out = model(x)
        observed["out_requires_grad"].append(out.requires_grad)
        return out

    inputs = torch.randn(256, 256, device=device)

    elapsed_times, _, timing_info = timing.time_execution_with_cuda_event(
        spy,
        inputs,
        num_warmup=2,
        num_trials=3,
        verbose=False,
        device=device,
        enable_profiling=False,
    )

    # The helper actually ran the requested forwards (warmup + trials).
    assert len(observed["grad_enabled"]) == 2 + 3
    assert len(elapsed_times) == 3
    assert timing_info["num_trials"] == 3

    # Every forward in the measurement window ran with autograd off ...
    assert all(enabled is False for enabled in observed["grad_enabled"])
    # ... so no autograd graph was built despite the trainable parameter.
    assert all(req is False for req in observed["out_requires_grad"])


@pytest.mark.gpu
def test_timing_disables_autograd_in_profiling_path() -> None:
    """The optional profiling iterations must also run with autograd disabled."""
    torch = _require_cuda_runtime()
    timing = _get_timing_module()

    device = torch.device("cuda:0")

    class Param(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(128, 128, device=device))

        def forward(self, x):
            return x @ self.w

    model = Param().to(device)
    grad_states = []

    def spy(x):
        grad_states.append(torch.is_grad_enabled())
        return model(x)

    inputs = torch.randn(128, 128, device=device)

    timing.time_execution_with_cuda_event(
        spy,
        inputs,
        num_warmup=1,
        num_trials=2,
        verbose=False,
        device=device,
        enable_profiling=True,
    )

    # warmup + measure + profiling forwards all happened ...
    assert len(grad_states) > 1 + 2
    # ... and none of them re-enabled autograd.
    assert all(enabled is False for enabled in grad_states)


@pytest.mark.gpu
def test_timing_restores_grad_mode_after_return() -> None:
    """no_grad must be scoped to the call, not leak to the caller."""
    torch = _require_cuda_runtime()
    timing = _get_timing_module()

    device = torch.device("cuda:0")
    assert torch.is_grad_enabled() is True

    def noop(x):
        return x + 1

    timing.time_execution_with_cuda_event(
        noop,
        torch.randn(64, device=device),
        num_warmup=1,
        num_trials=2,
        verbose=False,
        device=device,
        enable_profiling=False,
    )

    # Grad mode is back to normal for code running after timing.
    assert torch.is_grad_enabled() is True
