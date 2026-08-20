"""Characterize inference-mode failures that a no-grad fallback must handle."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_inference_tensor_has_no_readable_version_counter() -> None:
    with torch.no_grad():
        no_grad_tensor = torch.ones(2)
    assert no_grad_tensor._version == 0
    assert torch.is_inference(no_grad_tensor) is False

    with torch.inference_mode():
        inference_tensor = torch.ones(2)

    assert torch.is_inference(inference_tensor) is True
    with pytest.raises(RuntimeError, match="Inference tensors do not track version counter"):
        _ = inference_tensor._version


def test_inference_tensor_rejects_mutation_after_context_exit() -> None:
    with torch.no_grad():
        no_grad_tensor = torch.ones(2)
    no_grad_tensor.add_(1)
    assert torch.equal(no_grad_tensor, torch.full((2,), 2.0))

    with torch.inference_mode():
        inference_tensor = torch.ones(2)

    with pytest.raises(RuntimeError, match="Inplace update to inference tensor outside InferenceMode"):
        inference_tensor.add_(1)


def test_inference_tensor_rejects_requires_grad_after_context_exit() -> None:
    with torch.no_grad():
        no_grad_tensor = torch.ones(2)
    no_grad_tensor.requires_grad_(True)
    assert no_grad_tensor.requires_grad is True

    with torch.inference_mode():
        inference_tensor = torch.ones(2)

    with pytest.raises(RuntimeError, match="Setting requires_grad=True on inference tensor outside InferenceMode"):
        inference_tensor.requires_grad_(True)


def test_nested_enable_grad_cannot_escape_inference_mode() -> None:
    source = torch.ones(2, requires_grad=True)

    with torch.no_grad():
        with torch.enable_grad():
            no_grad_output = source * 2

    with torch.inference_mode():
        with torch.enable_grad():
            inference_output = source * 2

    assert no_grad_output.requires_grad is True
    assert torch.is_inference(no_grad_output) is False
    assert inference_output.requires_grad is False
    assert torch.is_inference(inference_output) is True
