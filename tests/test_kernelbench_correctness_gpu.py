import pytest


def _require_cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is not available")
    return torch


def _get_correctness_module():
    pytest.importorskip("torch")
    from kernelgym.toolkit.kernelbench import correctness

    return correctness


@pytest.mark.gpu
def test_correctness_runs_zero_like_cache_poison_before_custom_forward() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x):
            intermediate = x + 1
            return intermediate.clone()

    class EmptyOutput(torch.nn.Module):
        def forward(self, x):
            return torch.empty_like(x)

    device = torch.device("cuda:0")

    def get_inputs():
        return [torch.randn((256, 256), device=device)]

    result = correctness.run_and_check_correctness(
        Reference(),
        EmptyOutput(),
        get_inputs,
        metadata={},
        num_correct_trials=2,
        device=device,
    )

    assert result.correctness is False
    assert result.metadata["correctness_reference_cache_poison_enabled"] is True
    assert result.metadata["correctness_failed_trial"] == 0


@pytest.mark.gpu
def test_without_cache_poison_empty_output_can_reuse_reference_intermediate(monkeypatch) -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()
    monkeypatch.setattr(correctness, "_zero_poison_like", lambda value: None)

    class Reference(torch.nn.Module):
        def forward(self, x):
            intermediate = x + 1
            return intermediate.clone()

    class EmptyOutput(torch.nn.Module):
        def forward(self, x):
            return torch.empty_like(x)

    device = torch.device("cuda:0")

    def get_inputs():
        return [torch.randn((256, 256), device=device)]

    result = correctness.run_and_check_correctness(
        Reference(),
        EmptyOutput(),
        get_inputs,
        metadata={},
        num_correct_trials=2,
        device=device,
    )

    assert result.correctness is True
    assert result.metadata["correctness_trials"] == "(2 / 2)"


@pytest.mark.gpu
def test_correctness_accepts_matching_cuda_model_with_cache_poison() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class Matching(torch.nn.Module):
        def forward(self, x):
            return x + 1

    device = torch.device("cuda:0")

    def get_inputs():
        return [torch.randn((128, 128), device=device)]

    result = correctness.run_and_check_correctness(
        Reference(),
        Matching(),
        get_inputs,
        metadata={},
        num_correct_trials=2,
        device=device,
    )

    assert result.correctness is True
    assert result.metadata["correctness_trials"] == "(2 / 2)"
    assert result.metadata["correctness_reference_cache_poison_enabled"] is True
