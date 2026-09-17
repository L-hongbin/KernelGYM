"""KernelBench correctness GPU tests."""

import re

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
def test_correctness_input_perturbations_are_disabled_by_default() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    class PositiveOnlyCandidate(torch.nn.Module):
        def forward(self, x):
            return x

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        Reference(),
        PositiveOnlyCandidate(),
        lambda: [torch.rand((128, 128), device=device)],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
    )

    assert result.correctness is True
    assert result.metadata["correctness_input_perturbations_enabled"] is False
    assert result.metadata["correctness_effective_trials"] == 1
    assert "correctness_input_perturbation_trials" not in result.metadata


@pytest.mark.gpu
def test_numerical_mismatch_defaults_to_legacy_metadata(monkeypatch) -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x):
            return x

    class IncorrectCandidate(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    def reject_detailed_comparison(*args, **kwargs):
        raise AssertionError("default correctness must not compute detailed diagnostics")

    monkeypatch.setattr(
        correctness,
        "_compare_outputs_inplace_with_diagnostics",
        reject_detailed_comparison,
    )

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        Reference(),
        IncorrectCandidate(),
        lambda: [torch.rand((2, 3), device=device)],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
    )

    assert result.correctness is False
    assert result.metadata["correctness_issue_name"] == "numerical_mismatch"
    assert "max_difference=" in result.metadata["correctness_issue"]
    assert "avg_difference=" in result.metadata["correctness_issue"]
    assert "element_correctness=" not in result.metadata["correctness_issue"]
    for field in (
        "element_correctness",
        "element_correctness_curve",
        "nan_count",
        "inf_count",
        "mismatch_coordinate",
        "batch_correctness",
        "row_correctness",
        "tile_correctness",
        "output_space_localization",
        "correctness_failed_trial_seed",
    ):
        assert field not in result.metadata


@pytest.mark.gpu
def test_rand_sign_perturbation_returns_numerical_mismatch_details() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    class PositiveOnlyCandidate(torch.nn.Module):
        def forward(self, x):
            return x

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        Reference(),
        PositiveOnlyCandidate(),
        lambda: [torch.rand((128, 128), device=device)],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
        enable_input_perturbations=True,
        return_detail_correctness=True,
    )

    assert result.correctness is False
    assert result.metadata["correctness_effective_trials"] == 4
    assert result.metadata["correctness_failed_input_perturbation"] == "sign_challenge"
    assert isinstance(result.metadata["correctness_failed_trial_seed"], int)
    assert result.metadata["correctness_issue_name"] == "numerical_mismatch"
    assert float(result.metadata["max_difference"][0]) > 0
    assert float(result.metadata["avg_difference"][0]) > 0
    element_correctness = result.metadata["element_correctness"][0]
    assert re.fullmatch(r"\d+\.\d{2}%", element_correctness)
    assert f"element_correctness={element_correctness}" in result.metadata["correctness_issue"]
    element_correctness_curve = result.metadata["element_correctness_curve"][0]
    expected_multipliers = ["1x", "2x", "4x", "8x", "16x"]
    assert list(element_correctness_curve) == expected_multipliers[: len(element_correctness_curve)]
    assert element_correctness_curve["1x"] == element_correctness
    assert all(re.fullmatch(r"\d+\.\d{2}%", value) for value in element_correctness_curve.values())
    assert all(
        float(element_correctness_curve[left][:-1]) <= float(element_correctness_curve[right][:-1])
        for left, right in zip(element_correctness_curve, list(element_correctness_curve)[1:])
    )
    if "100.00%" in element_correctness_curve.values():
        assert list(element_correctness_curve.values())[-1] == "100.00%"
    curve_points = ",".join(
        f"{multiplier}:{percentage}" for multiplier, percentage in element_correctness_curve.items()
    )
    curve_text = f"{{{curve_points}}}"
    assert f"element_correctness_curve={curve_text}" in result.metadata["correctness_issue"]
    assert result.metadata["nan_count"] == [0]
    assert result.metadata["inf_count"] == [0]
    mismatch_coordinate = result.metadata["mismatch_coordinate"][0]
    assert set(mismatch_coordinate) == {"first", "last", "top_3"}
    assert mismatch_coordinate["first"]["output_path"] == "output"
    assert mismatch_coordinate["last"]["output_path"] == "output"
    assert len(mismatch_coordinate["top_3"]) == 3
    assert 0.0 <= result.metadata["batch_correctness"][0] <= 1.0
    assert 0.0 <= result.metadata["row_correctness"][0] <= 1.0
    assert 0.0 <= result.metadata["tile_correctness"][0] <= 1.0
    assert result.metadata["output_space_localization"][0]["tile_shape"] == [32, 32]
    assert "correctness_numerical_errors" not in result.metadata
    sign_trial = result.metadata["correctness_input_perturbation_trials"][-1]
    assert sign_trial["detected_input_kinds"] == {"torch.rand": 1}
    assert sign_trial["transforms"] == {"negate": 1}


@pytest.mark.gpu
def test_nonfinite_candidate_returns_counts_and_mismatch_coordinates() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    class NonfiniteCandidate(torch.nn.Module):
        def forward(self, x):
            output = torch.zeros_like(x)
            flat_output = output.view(-1)
            flat_output[1] = torch.nan
            flat_output[3] = torch.inf
            flat_output[5] = 1.0
            return output

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        Reference(),
        NonfiniteCandidate(),
        lambda: [torch.rand((2, 3), device=device)],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
        return_detail_correctness=True,
    )

    assert result.correctness is False
    assert result.metadata["nan_count"] == [1]
    assert result.metadata["inf_count"] == [1]
    assert isinstance(result.metadata["correctness_failed_trial_seed"], int)
    mismatch_coordinate = result.metadata["mismatch_coordinate"][0]
    assert mismatch_coordinate["first"] == {
        "output_path": "output",
        "coordinate": [0, 1],
    }
    assert mismatch_coordinate["last"] == {
        "output_path": "output",
        "coordinate": [1, 2],
    }
    assert len(mismatch_coordinate["top_3"]) == 3
    assert result.metadata["batch_correctness"] == [0.0]
    assert result.metadata["row_correctness"] == [0.0]
    assert result.metadata["tile_correctness"] == [0.0]
    tensor_localization = result.metadata["output_space_localization"][0]["tensors"][0]
    assert tensor_localization["mismatch_bounds"] == {"M": [0, 1], "N": [0, 2]}


@pytest.mark.gpu
def test_reference_error_skips_only_failing_input_perturbation() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class NonnegativeOnly(torch.nn.Module):
        def forward(self, x):
            if bool((x < 0).any().item()):
                raise ValueError("negative inputs are unsupported")
            return x + 1

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        NonnegativeOnly(),
        NonnegativeOnly(),
        lambda: [torch.rand((128, 128), device=device)],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
        enable_input_perturbations=True,
    )

    assert result.correctness is True
    assert result.metadata["correctness_trials"] == "(3 / 3)"
    assert result.metadata["correctness_reference_skipped_perturbation_count"] == 1
    skipped = result.metadata["correctness_reference_skipped_perturbations"][0]
    assert skipped["perturbation"] == "sign_challenge"
    assert "negative inputs are unsupported" in skipped["reason"]


@pytest.mark.gpu
def test_reference_nonfinite_output_skips_only_failing_input_perturbation() -> None:
    torch = _require_cuda_runtime()
    correctness = _get_correctness_module()

    class SquareRoot(torch.nn.Module):
        def forward(self, x):
            return torch.sqrt(x)

    device = torch.device("cuda:0")
    result = correctness.run_and_check_correctness(
        SquareRoot(),
        SquareRoot(),
        lambda: [torch.rand((128, 128), device=device)],
        metadata={},
        num_correct_trials=1,
        seed=1234,
        device=device,
        enable_input_perturbations=True,
    )

    assert result.correctness is True
    assert result.metadata["correctness_trials"] == "(3 / 3)"
    skipped = result.metadata["correctness_reference_skipped_perturbations"][0]
    assert skipped["perturbation"] == "sign_challenge"
    assert skipped["reason"] == "reference output contains NaN or Inf"


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
