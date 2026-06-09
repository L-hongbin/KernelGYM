"""Scheduler API for task submission and tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import TaskSpec


class SchedulerAPI(ABC):
    @abstractmethod
    async def submit(self, task: TaskSpec) -> str:
        """Submit a task and return its task_id."""

    @abstractmethod
    async def wait(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for a task result and return the raw result payload."""

    @abstractmethod
    async def get_status(self, task_id: str) -> Dict[str, Any]:
        """Return status metadata for a task."""

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Cancel a task if possible."""

    async def begin_workflow(self, base_id: str) -> None:
        """Mark a multi-task workflow as active so its parent id is cancellable.

        Default is a no-op; schedulers backed by a shared store override this.
        """
        return None

    async def end_workflow(self, base_id: str) -> None:
        """Clear the active marker for a workflow. Default no-op."""
        return None

    async def is_cancelled(self, task_id: str) -> bool:
        """Whether cancellation was requested for ``task_id``. Default False."""
        return False

    async def wait_unless_cancelled(
        self, task_id: str, base_id: str, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Like ``wait`` but return ``None`` promptly if ``base_id`` is cancelled.

        Lets a workflow abort while a sub-task is still queued (never dispatched)
        or running on a worker without a cancellation watcher (e.g. CPU compile),
        instead of blocking in ``wait`` until the sub-task finishes. Default
        implementation ignores cancellation and falls back to ``wait``.
        """
        return await self.wait(task_id, timeout)
