"""Runtime request defaults applied before workflow execution."""

from __future__ import annotations

from typing import Any, Dict


def apply_runtime_defaults(
    payload: Dict[str, Any],
    *,
    workflow_name: str,
    split_compile_and_execute: bool,
    enable_ncu: bool | None = None,
    ncu_profile_version: str = "",
    enable_compute_sanitizer: bool | None = None,
    compute_sanitizer_profile_version: str = "",
    enable_correctness_input_perturbations: bool | None = None,
) -> Dict[str, Any]:
    """Apply deployment-level defaults to an external workflow payload."""
    if payload.get("resources") is None:
        payload["resources"] = None
    if (workflow_name or "kernelbench") == "kernelbench":
        if payload.get("enable_ncu") is None and enable_ncu is not None:
            payload["enable_ncu"] = bool(enable_ncu)
        if payload.get("enable_ncu") and ncu_profile_version:
            payload["_ncu_profile_version"] = ncu_profile_version
        if payload.get("enable_compute_sanitizer") is None and enable_compute_sanitizer is not None:
            payload["enable_compute_sanitizer"] = bool(enable_compute_sanitizer)
        if payload.get("enable_compute_sanitizer") and compute_sanitizer_profile_version:
            payload["_compute_sanitizer_profile_version"] = compute_sanitizer_profile_version
        if (
            payload.get("enable_correctness_input_perturbations") is None
            and enable_correctness_input_perturbations is not None
        ):
            payload["enable_correctness_input_perturbations"] = bool(enable_correctness_input_perturbations)
    if (
        (workflow_name or "kernelbench") == "kernelbench"
        and split_compile_and_execute
        and not payload.get("pure_compile_task")
        and not payload.get("task_stage")
    ):
        payload["split_compile_and_execute"] = True
    return payload
