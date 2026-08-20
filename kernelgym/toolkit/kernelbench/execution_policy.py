"""Shared execution policy for KernelBench correctness and timing."""

from __future__ import annotations

from typing import Any


EXECUTION_POLICY_VERSION = "eval_no_grad_v1"
MODEL_MODE = "eval"
GRAD_MODE = "no_grad"


def prepare_model_for_execution(model: Any) -> Any:
    """Put a KernelBench model in the required evaluation mode."""
    model.eval()
    return model


def record_execution_policy(metadata: dict[str, Any]) -> None:
    metadata["execution_policy"] = EXECUTION_POLICY_VERSION
    metadata["model_mode"] = MODEL_MODE
    metadata["grad_mode"] = GRAD_MODE
