"""Shared execution policy for KernelBench correctness and timing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


EXECUTION_POLICY_VERSION = "eval_no_grad_tf32_decoy_v3"
MODEL_MODE = "eval"
GRAD_MODE = "no_grad"
FP32_MATH_MODE = "tf32"


def _read_backend_attr(root: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    obj = root
    for name in path[:-1]:
        try:
            obj = getattr(obj, name)
        except Exception:
            return False, None
        if obj is None:
            return False, None
    try:
        return True, getattr(obj, path[-1])
    except Exception:
        return False, None


def _write_backend_attr(root: Any, path: tuple[str, ...], value: Any) -> bool:
    obj = root
    for name in path[:-1]:
        try:
            obj = getattr(obj, name)
        except Exception:
            return False
        if obj is None:
            return False
    try:
        setattr(obj, path[-1], value)
        return True
    except Exception:
        return False


@contextmanager
def tf32_execution_context(
    metadata: dict[str, Any] | None = None,
    *,
    stage: str,
    enabled: bool = True,
):
    """Enable TF32 for one execution stage and restore prior process state."""

    if metadata is not None:
        metadata[f"{stage}_tf32_enabled"] = bool(enabled)
    if not enabled:
        yield
        return

    # PyTorch 2.9+ prefers per-op fp32_precision settings. Avoid mixing them
    # with legacy allow_tf32 flags when available because newer PyTorch rejects
    # contradictory old/new API combinations.
    has_cudnn_conv_precision, _ = _read_backend_attr(
        torch.backends,
        ("cudnn", "conv", "fp32_precision"),
    )
    has_matmul_precision, _ = _read_backend_attr(
        torch.backends,
        ("cuda", "matmul", "fp32_precision"),
    )

    attr_targets = []
    if has_cudnn_conv_precision:
        attr_targets.append(("cudnn.conv.fp32_precision", ("cudnn", "conv", "fp32_precision"), "tf32"))
    else:
        attr_targets.append(("cudnn.allow_tf32", ("cudnn", "allow_tf32"), True))
    if has_matmul_precision:
        attr_targets.append(("cuda.matmul.fp32_precision", ("cuda", "matmul", "fp32_precision"), "tf32"))
    else:
        attr_targets.append(("cuda.matmul.allow_tf32", ("cuda", "matmul", "allow_tf32"), True))

    saved_attrs: list[tuple[tuple[str, ...], Any]] = []
    before: dict[str, str] = {}
    applied: dict[str, str] = {}
    for label, path, value in attr_targets:
        exists, old_value = _read_backend_attr(torch.backends, path)
        if not exists:
            continue
        before[label] = str(old_value)
        if _write_backend_attr(torch.backends, path, value):
            saved_attrs.append((path, old_value))
            applied[label] = str(value)

    old_matmul_precision = None
    if (
        not has_matmul_precision
        and hasattr(torch, "get_float32_matmul_precision")
        and hasattr(torch, "set_float32_matmul_precision")
    ):
        try:
            old_matmul_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("high")
            before["float32_matmul_precision"] = str(old_matmul_precision)
            applied["float32_matmul_precision"] = "high"
        except Exception:
            old_matmul_precision = None

    if metadata is not None:
        metadata[f"{stage}_tf32_state_before"] = before
        metadata[f"{stage}_tf32_state_forced"] = applied

    try:
        yield
    finally:
        if old_matmul_precision is not None:
            try:
                torch.set_float32_matmul_precision(old_matmul_precision)
            except Exception:
                pass
        for path, old_value in reversed(saved_attrs):
            _write_backend_attr(torch.backends, path, old_value)


def prepare_model_for_execution(model: Any) -> Any:
    """Put a KernelBench model in the required evaluation mode."""
    model.eval()
    return model


def record_execution_policy(metadata: dict[str, Any]) -> None:
    metadata["execution_policy"] = EXECUTION_POLICY_VERSION
    metadata["model_mode"] = MODEL_MODE
    metadata["grad_mode"] = GRAD_MODE
    metadata["fp32_math_mode"] = FP32_MATH_MODE
