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


def _tf32_is_enabled(torch) -> bool:  # noqa: ANN001
    from kernelgym.toolkit.kernelbench.execution_policy import _read_backend_attr

    has_conv_precision, conv_precision = _read_backend_attr(
        torch.backends,
        ("cudnn", "conv", "fp32_precision"),
    )
    has_matmul_precision, matmul_precision = _read_backend_attr(
        torch.backends,
        ("cuda", "matmul", "fp32_precision"),
    )
    conv_enabled = (
        conv_precision == "tf32"
        if has_conv_precision
        else _read_backend_attr(torch.backends, ("cudnn", "allow_tf32"))[1] is True
    )
    matmul_enabled = (
        matmul_precision == "tf32"
        if has_matmul_precision
        else _read_backend_attr(torch.backends, ("cuda", "matmul", "allow_tf32"))[1] is True
    )
    return conv_enabled and matmul_enabled


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

    observed = {"grad_enabled": [], "out_requires_grad": [], "tf32_enabled": []}

    def spy(x):
        observed["grad_enabled"].append(torch.is_grad_enabled())
        observed["tf32_enabled"].append(_tf32_is_enabled(torch))
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
    assert timing_info["timing_tf32_enabled"] is True
    assert timing_info["timing_tf32_state_forced"]

    # Every forward in the measurement window ran with autograd off ...
    assert all(enabled is False for enabled in observed["grad_enabled"])
    # ... so no autograd graph was built despite the trainable parameter.
    assert all(req is False for req in observed["out_requires_grad"])
    assert all(observed["tf32_enabled"])


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


@pytest.mark.gpu
def test_candidate_gpu_policy_is_eval_no_grad_not_inference() -> None:
    """The production candidate timing entry applies the complete policy on CUDA."""
    torch = _require_cuda_runtime()
    from kernelgym.toolkit.kernelbench import pipeline
    from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult

    device = torch.device("cuda:0")
    observed = []

    class ObservedModel(torch.nn.Module):
        def forward(self, x):
            observed.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            return x + 1

    model = ObservedModel().train().to(device)
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    pipeline._run_performance_step(
        kernel_exec_result=result,
        custom_model=model,
        get_inputs=lambda: [torch.ones(64, device=device)],
        metadata=result.metadata,
        num_perf_trials=2,
        num_warmup=1,
        perf_trim_count=0,
        verbose=False,
        seed_num=42,
        device=device,
        enable_profiling=False,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend="cuda_agent",
        backend_profiling_hints=None,
    )

    assert observed == [(False, False, False)] * 3
    assert result.runtime is not None and result.runtime > 0
