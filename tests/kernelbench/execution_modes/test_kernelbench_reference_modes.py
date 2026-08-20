"""Exercise real KernelBench references under no-grad and inference mode."""

from __future__ import annotations

from typing import Any

import pytest

from tests.kernelbench.execution_modes.reference_cases import (
    REFERENCE_CASES,
    ReferenceCase,
    find_kernelbench_data_root,
    load_reference_namespace,
)


torch = pytest.importorskip("torch")
DATA_ROOT = find_kernelbench_data_root()
pytestmark = pytest.mark.skipif(DATA_ROOT is None, reason="KernelBench reference checkout is unavailable")


def _build_model_and_inputs(case: ReferenceCase, *, seed: int = 1234):
    assert DATA_ROOT is not None
    namespace = load_reference_namespace(DATA_ROOT, case)
    torch.manual_seed(seed)
    init_inputs = namespace["get_init_inputs"]()
    model = namespace["Model"](*init_inputs).eval()
    inputs = namespace["get_inputs"]()
    return model, inputs


def _assert_tree_close(actual: Any, expected: Any) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_close(actual_item, expected_item)
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_close(actual[key], expected[key])
        return
    assert actual == expected


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda case: case.test_id)
def test_eval_reference_matches_between_no_grad_and_inference_mode(case: ReferenceCase) -> None:
    no_grad_model, no_grad_inputs = _build_model_and_inputs(case)
    inference_model, inference_inputs = _build_model_and_inputs(case)

    assert all(module.training is False for module in no_grad_model.modules())
    assert all(module.training is False for module in inference_model.modules())

    torch.manual_seed(5678)
    with torch.no_grad():
        no_grad_output = no_grad_model(*no_grad_inputs)

    torch.manual_seed(5678)
    with torch.inference_mode():
        inference_output = inference_model(*inference_inputs)

    _assert_tree_close(inference_output, no_grad_output)


def test_vanilla_rnn_inference_state_cannot_be_reused_by_no_grad() -> None:
    case = next(case for case in REFERENCE_CASES if case.relative_path == "level3/33_VanillaRNN.py")
    model, inputs = _build_model_and_inputs(case)

    assert torch.is_inference(model.hidden) is False
    with torch.inference_mode():
        model(*inputs)
    assert torch.is_inference(model.hidden) is True

    with pytest.raises(RuntimeError, match="Inplace update to inference tensor outside InferenceMode"):
        with torch.no_grad():
            model(*inputs)


def test_vanilla_rnn_no_grad_fallback_succeeds_with_fresh_model() -> None:
    case = next(case for case in REFERENCE_CASES if case.relative_path == "level3/33_VanillaRNN.py")
    attempted_model, attempted_inputs = _build_model_and_inputs(case)
    with torch.inference_mode():
        attempted_model(*attempted_inputs)

    fallback_model, fallback_inputs = _build_model_and_inputs(case)
    with torch.no_grad():
        fallback_output = fallback_model(*fallback_inputs)

    assert fallback_output.shape == (2, 4)
    assert torch.is_inference(fallback_model.hidden) is False
