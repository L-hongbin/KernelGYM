"""Scheduler adapter for TaskManager."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from ..core.scheduler import SchedulerAPI
from ..core.types import TaskSpec
from .workflow_lifecycle import WorkflowStoppedError


class TaskManagerScheduler(SchedulerAPI):
    def __init__(self, task_manager: Any, poll_interval: float = 0.5, *, workflow: Optional[Dict[str, str]] = None):
        self._task_manager = task_manager
        self._poll_interval = poll_interval
        self._workflow = workflow

    async def _check_workflow(self) -> None:
        if self._workflow is None:
            return
        record = await self._task_manager.workflows.reconcile(self._workflow["task_id"])
        if record.get("workflow_generation") != self._workflow["workflow_generation"] or record.get("status") not in {
            "pending",
            "processing",
        }:
            raise WorkflowStoppedError("Parent workflow is no longer active")

    async def submit(self, task: TaskSpec) -> str:
        payload = task.payload
        if not isinstance(payload, dict):
            raise ValueError("TaskSpec.payload must be a dict")
        if "task_id" not in payload:
            raise ValueError("TaskSpec.payload must include task_id")
        if self._workflow is not None:
            await self._check_workflow()
            payload = dict(payload)
            base_id = self._workflow["task_id"]
            generation = self._workflow["workflow_generation"]
            keys = self._task_manager.workflows.keys(base_id)
            # Generation-scoped child IDs isolate force-refresh and late CPU
            # completions from the next invocation of the same parent ID.
            original_id = payload["task_id"]
            if original_id == base_id:
                original_id = f"{base_id}_kernel"
            payload.update(
                task_id=f"{original_id}_{generation}",
                base_task_id=base_id,
                force_refresh=False,
                workflow_generation=generation,
                workflow_source_task_id=original_id,
                workflow_parent_key=keys[0],
                workflow_lease_key=keys[2],
                workflow_deadline=self._workflow["workflow_deadline"],
                workflow_children_key=self._task_manager.workflows.children_key(base_id, generation),
            )
        if task.resources is not None and isinstance(payload, dict) and "resources" not in payload:
            payload = dict(payload)
            payload["resources"] = task.resources
        if task.kind == "kernelbench.evaluation":
            return await self._task_manager.submit_evaluation_task(payload)
        return await self._task_manager.submit_task(payload)

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        start = time.monotonic()
        while True:
            await self._check_workflow()
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

    async def is_cancelled(self, task_id: str) -> bool:
        return await self._task_manager.is_task_cancelled(task_id)

    async def wait_unless_cancelled(
        self, task_id: str, base_id: str, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        start = time.monotonic()
        while True:
            await self._check_workflow()
            result = await self._task_manager.get_task_result(task_id)
            if result:
                return result
            if base_id and await self._task_manager.is_task_cancelled(base_id):
                # Pull this specific child out of its queue (and mark it terminal)
                # so it can't be dispatched and run orphaned after the parent
                # cancel marker eventually expires.
                try:
                    await self._task_manager.cancel_task(task_id)
                except Exception:  # pragma: no cover - best effort
                    pass
                return None
            if timeout is not None and (time.monotonic() - start) >= timeout:
                raise TimeoutError(f"Timed out waiting for task {task_id}")
            await asyncio.sleep(self._poll_interval)

    async def select_idle_worker(
        self,
        resource: str,
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        target_node_id: Optional[str] = None,
        target_hostname: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        start = time.monotonic()
        interval = self._poll_interval if poll_interval is None else poll_interval
        while True:
            worker = await self._task_manager.select_idle_worker(
                resource,
                target_node_id=target_node_id,
                target_hostname=target_hostname,
            )
            if worker:
                return worker
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return None
            await asyncio.sleep(interval)

    async def select_worker_by_task_id(
        self,
        resource: str,
        task_id: str,
        *,
        target_node_id: Optional[str] = None,
        target_hostname: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self._task_manager.select_worker_by_task_id(
            resource,
            task_id,
            target_node_id=target_node_id,
            target_hostname=target_hostname,
        )
