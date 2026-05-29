"""Scheduler adapter for TaskManager."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from ..core.scheduler import SchedulerAPI
from ..core.types import TaskSpec


class TaskManagerScheduler(SchedulerAPI):
    def __init__(self, task_manager: Any, poll_interval: float = 0.5):
        self._task_manager = task_manager
        self._poll_interval = poll_interval

    async def submit(self, task: TaskSpec) -> str:
        payload = task.payload
        if not isinstance(payload, dict):
            raise ValueError("TaskSpec.payload must be a dict")
        if "task_id" not in payload:
            raise ValueError("TaskSpec.payload must include task_id")
        if task.resources is not None and isinstance(payload, dict) and "resources" not in payload:
            payload = dict(payload)
            payload["resources"] = task.resources
        if task.kind == "kernelbench.evaluation":
            return await self._task_manager.submit_evaluation_task(payload)
        return await self._task_manager.submit_task(payload)

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        start = time.monotonic()
        while True:
            result = await self._task_manager.get_task_result(task_id)
            if result:
                return result
            if timeout is not None and (time.monotonic() - start) >= timeout:
                raise TimeoutError(f"Timed out waiting for task {task_id}")
            await asyncio.sleep(self._poll_interval)

    async def get_status(self, task_id: str) -> Dict[str, Any]:
        status = await self._task_manager.get_task_status(task_id)
        return status or {}

    async def cancel(self, task_id: str) -> bool:
        return await self._task_manager.cancel_task(task_id)

    async def select_idle_worker(
        self,
        resource: str,
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        start = time.monotonic()
        interval = self._poll_interval if poll_interval is None else poll_interval
        while True:
            worker = await self._task_manager.select_idle_worker(resource)
            if worker:
                return worker
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return None
            await asyncio.sleep(interval)

    async def select_worker_by_task_id(
        self,
        resource: str,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._task_manager.select_worker_by_task_id(resource, task_id)
