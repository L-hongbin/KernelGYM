"""KernelGym reward package.

The package root is intentionally import-light. Importing ``kernelgym`` should
not eagerly import CUDA/Torch backends or server modules.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Artifact",
    "Metric",
    "Result",
    "TaskSpec",
    "TaskGroup",
    "SchedulerAPI",
    "WorkflowController",
    "WorkflowState",
    "Registry",
    "KernelBenchWorkflowController",
    "TaskManagerScheduler",
]


def __getattr__(name: str) -> Any:
    if name in {
        "Artifact",
        "Metric",
        "Result",
        "TaskSpec",
        "TaskGroup",
        "SchedulerAPI",
        "WorkflowController",
        "WorkflowState",
        "Registry",
    }:
        from . import core

        return getattr(core, name)
    if name == "KernelBenchWorkflowController":
        from .workflow import KernelBenchWorkflowController

        return KernelBenchWorkflowController
    if name == "TaskManagerScheduler":
        from .server import TaskManagerScheduler

        return TaskManagerScheduler
    raise AttributeError(name)
