"""KernelBench correctness GPU tests."""

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


@pytest.mark.gpu
def test_correctness_resets_seed_before_each_stochastic_forward() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class RandomReference(torch.nn.Module):
        def forward(self, x):
            return x + torch.rand_like(x)

    class RandomMatching(torch.nn.Module):
        def forward(self, x):
            return x + torch.rand_like(x)

    device = torch.device("cuda:0")

    def get_inputs():
        # Deliberately consume the same CUDA RNG used by the forwards.
        return [torch.randn((128, 128), device=device)]

    result = correctness.run_and_check_correctness(
        RandomReference(),
        RandomMatching(),
        get_inputs,
        metadata={},
        num_correct_trials=2,
        seed=1234,
        device=device,
    )

    assert result.correctness is True
    assert result.metadata["correctness_trials"] == "(2 / 2)"
    assert result.metadata["correctness_forward_seed_reset_enabled"] is True


@pytest.mark.gpu
def test_correctness_marks_forbidden_aten_compute_as_decoy() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.mm(x, y)

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        MatmulModel(),
        MatmulModel(),
        lambda: [
            torch.randn((64, 64), device=device),
            torch.randn((64, 64), device=device),
        ],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
        detect_aten_fallback=True,
    )

    assert result.correctness is True
    assert result.decoy_kernel is True
    assert result.metadata["policy_violation_reason"] == "DISALLOWED_ATEN_COMPUTE"
    assert "aten::mm" in result.metadata["forbidden_aten_op_names"]


@pytest.mark.gpu
def test_incorrect_output_preserves_forbidden_aten_decoy_verdict() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x, y):
            return torch.mm(x, y) + 1

    class IncorrectAtenFallback(torch.nn.Module):
        def forward(self, x, y):
            return torch.mm(x, y)

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        Reference(),
        IncorrectAtenFallback(),
        lambda: [
            torch.randn((64, 64), device=device),
            torch.randn((64, 64), device=device),
        ],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
        detect_aten_fallback=True,
    )

    assert result.correctness is False
    assert result.decoy_kernel is True
    assert result.metadata["policy_violation_reason"] == "DISALLOWED_ATEN_COMPUTE"
    assert "aten::mm" in result.metadata["forbidden_aten_op_names"]
    assert result.metadata["correctness_candidate_forward_completed"] is True
    assert result.metadata["correctness_candidate_forward_completed_trials"] == [0]
    assert result.metadata["correctness_output_mismatch"] is True


@pytest.mark.gpu
def test_correctness_allows_allowlisted_aten_view() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class ViewModel(torch.nn.Module):
        def forward(self, x):
            return x.view(32, 128)

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        ViewModel(),
        ViewModel(),
        lambda: [torch.randn((64, 64), device=device)],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
        detect_aten_fallback=True,
    )

    assert result.correctness is True
    assert result.decoy_kernel is False
    assert result.metadata["forbidden_aten_op_names"] == []
    assert any(item["name"] == "aten::view" for item in result.metadata["allowed_aten_ops"])


@pytest.mark.gpu
def test_correctness_profiles_aten_on_every_candidate_trial() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x):
            return x.view(32, 128)

    class BranchingCandidate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls == 1:
                return x.view(32, 128)
            return x.view(32, 128) + 0

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        Reference(),
        BranchingCandidate(),
        lambda: [torch.randn((64, 64), device=device)],
        metadata={},
        num_correct_trials=2,
        seed=1234,
        device=device,
        detect_aten_fallback=True,
    )

    assert result.correctness is True
    assert result.decoy_kernel is True
    assert result.metadata["aten_detection_trials_run"] == 2
    assert "aten::add" in result.metadata["forbidden_aten_op_names"]
