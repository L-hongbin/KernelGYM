"""Element-level KernelBench correctness statistics."""

import pytest
import torch

from kernelgym.toolkit.kernelbench.correctness import (
    _compare_outputs_inplace_with_diagnostics,
    _compare_outputs_inplace_with_element_counts,
    _compare_tensors_inplace_with_correctness_curve,
    _compare_tensors_inplace_with_element_counts,
    _format_element_correctness,
    _format_element_correctness_curve,
    _format_mismatch_coordinate,
    _format_output_space_localization,
)


def test_element_correctness_uses_reference_scaled_tolerance() -> None:
    reference = torch.tensor([1.0])
    candidate = torch.tensor([1.105])

    outputs_close, _max_diff, _avg_diff, correct, total = _compare_tensors_inplace_with_element_counts(
        reference,
        candidate,
        atol=0.0,
        rtol=0.1,
    )

    assert outputs_close is False
    assert (correct, total) == (0, 1)


def test_element_correctness_weights_nested_outputs_by_element_count() -> None:
    reference = {
        "large": torch.zeros(100),
        "small": torch.zeros(1),
    }
    candidate = {
        "large": torch.zeros(100),
        "small": torch.ones(1),
    }
    candidate["large"][0] = 1.0

    outputs_close, _max_diff, _avg_diff, correct, total = _compare_outputs_inplace_with_element_counts(
        reference,
        candidate,
    )

    assert outputs_close is False
    assert (correct, total) == (99, 101)
    assert 100.0 * correct / total == pytest.approx(98.01980198)
    assert _format_element_correctness(correct, total) == "98.02%"


def test_element_correctness_curve_counts_multiple_tolerances() -> None:
    reference = torch.zeros(7)
    candidate = torch.tensor([0.0, 1.0, 1.5, 2.5, 4.5, 8.5, 16.5])

    outputs_close, _max_diff, _avg_diff, curve_counts, total = _compare_tensors_inplace_with_correctness_curve(
        reference,
        candidate,
        atol=1.0,
        rtol=0.0,
    )

    assert outputs_close is False
    assert curve_counts == {1: 2, 2: 3, 4: 4, 8: 5, 16: 6}
    assert _format_element_correctness_curve(curve_counts, total) == {
        "1x": "28.57%",
        "2x": "42.86%",
        "4x": "57.14%",
        "8x": "71.43%",
        "16x": "85.71%",
    }


def test_element_correctness_curve_stops_comparing_after_reaching_100_percent(monkeypatch) -> None:
    reference = torch.zeros(3)
    candidate = torch.tensor([0.0, 1.5, 3.0])
    original_count_nonzero = torch.count_nonzero
    comparison_calls = 0

    def counted_count_nonzero(*args, **kwargs):
        nonlocal comparison_calls
        comparison_calls += 1
        return original_count_nonzero(*args, **kwargs)

    monkeypatch.setattr(torch, "count_nonzero", counted_count_nonzero)
    _outputs_close, _max_diff, _avg_diff, curve_counts, total = _compare_tensors_inplace_with_correctness_curve(
        reference,
        candidate,
        atol=1.0,
        rtol=0.0,
    )

    assert comparison_calls == 2
    assert curve_counts == {1: 1, 2: 2, 4: 3, 8: 3, 16: 3}
    assert _format_element_correctness_curve(curve_counts, total) == {
        "1x": "33.33%",
        "2x": "66.67%",
        "4x": "100.00%",
    }


def test_mismatch_diagnostics_count_nonfinite_values_and_nested_coordinates() -> None:
    reference = {
        "matrix": torch.zeros((2, 3)),
        "tail": torch.zeros(2),
    }
    candidate = {
        "matrix": torch.zeros((2, 3)),
        "tail": torch.zeros(2),
    }
    candidate["matrix"][0, 1] = torch.nan
    candidate["matrix"][1, 2] = 2.0
    candidate["tail"][1] = torch.inf

    outputs_close, _max_diff, _avg_diff, curve_counts, total, nan_count, inf_count, coordinates = (
        _compare_outputs_inplace_with_diagnostics(reference, candidate)
    )
    public_coordinates = _format_mismatch_coordinate(coordinates)

    assert outputs_close is False
    assert total - curve_counts[1] == 3
    assert nan_count == 1
    assert inf_count == 1
    assert public_coordinates["first"] == {
        "output_path": "output.matrix",
        "coordinate": [0, 1],
    }
    assert public_coordinates["last"] == {
        "output_path": "output.tail",
        "coordinate": [1],
    }
    assert len(public_coordinates["top_3"]) == 3
    assert {(entry["output_path"], tuple(entry["coordinate"])) for entry in public_coordinates["top_3"]} == {
        ("output.matrix", (0, 1)),
        ("output.matrix", (1, 2)),
        ("output.tail", (1,)),
    }


def test_output_space_localization_reports_batch_row_tile_and_axis_bounds() -> None:
    reference = torch.zeros((1, 32, 64))
    candidate = torch.zeros_like(reference)
    candidate[0, 0, 32:48] = 1.0

    _close, _max_diff, _avg_diff, _curve, _total, _nan, _inf, coordinates = _compare_outputs_inplace_with_diagnostics(
        reference, candidate
    )
    localization = _format_output_space_localization(coordinates)

    assert localization is not None
    assert localization["tile_shape"] == [32, 32]
    assert localization["batch_correctness"] == 0.0
    assert localization["row_correctness"] == pytest.approx(31 / 32)
    assert localization["tile_correctness"] == 0.5
    tensor_localization = localization["tensors"][0]
    assert tensor_localization["mismatch_bounds"] == {
        "B": [0, 0],
        "M": [0, 0],
        "N": [32, 47],
    }
    assert tensor_localization["batch"]["failed"] == [{"batch": 0}]
    assert tensor_localization["row"]["failed"] == [{"batch_index": [0], "row": 0}]
    assert tensor_localization["tile"]["failed"] == [
        {
            "batch_index": [0],
            "M": [0, 32],
            "N": [32, 64],
        }
    ]
