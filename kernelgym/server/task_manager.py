"""Core TaskManager for KernelGym server.

This is a minimal, generic scheduler-backed task manager without workflow semantics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

from kernelgym.common import TaskStatus, Priority, ErrorCode
from kernelgym.config import settings
from kernelgym.backend import list_backends
from kernelgym.server.code_retry_manager import CodeRetryManager
from kernelgym.toolkit import list_toolkits

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    task_id: str
    status: TaskStatus
    priority: Priority
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class WorkerLoadBalancer:
    """Simple round-robin worker load balancer."""

    def __init__(self):
        self.available_workers: Dict[str, Dict[str, Any]] = {}
        self.current_index = 0
        self._lock = asyncio.Lock()

    async def register_worker(self, worker_id: str, device: str):
        async with self._lock:
            self.available_workers[worker_id] = {
                "device": device,
                "status": "online",
                "last_heartbeat": datetime.now(),
            }

    async def unregister_worker(self, worker_id: str):
        async with self._lock:
            self.available_workers.pop(worker_id, None)

    async def update_worker_heartbeat(self, worker_id: str):
        async with self._lock:
            if worker_id in self.available_workers:
                self.available_workers[worker_id]["last_heartbeat"] = datetime.now()

    async def get_next_worker(self) -> Optional[str]:
        async with self._lock:
            now = datetime.now()
            fresh_online = []
            for wid, info in self.available_workers.items():
                if info.get("status") != "online":
                    continue
                last_hb = info.get("last_heartbeat")
                if not isinstance(last_hb, datetime):
                    continue
                if (now - last_hb).total_seconds() <= 30:
                    fresh_online.append(wid)

            if not fresh_online:
                return None

            worker = fresh_online[self.current_index % len(fresh_online)]
            self.current_index = (self.current_index + 1) % len(fresh_online)
            return worker


class TaskManager:
    """Manages task queue and worker registry."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = settings.redis_key_prefix
        self.legacy_prefix = settings.redis_key_prefix_legacy
        self.task_prefix = f"{self.key_prefix}:task:"
        self.queue_prefix = f"{self.key_prefix}:queue:"
        self.result_prefix = f"{self.key_prefix}:result:"
        self.worker_prefix = f"{self.key_prefix}:worker:"
        self.node_map_key = f"{self.key_prefix}:nodes"
        self.status_prefix = f"{self.key_prefix}:status:"

        self.priority_queues = {
            Priority.HIGH: f"{self.queue_prefix}priority:high",
            Priority.NORMAL: f"{self.queue_prefix}priority:normal",
            Priority.LOW: f"{self.queue_prefix}priority:low",
        }
        self.resource_queues = {
            "gpu": {
                Priority.HIGH: f"{self.queue_prefix}resource:gpu:priority:high",
                Priority.NORMAL: f"{self.queue_prefix}resource:gpu:priority:normal",
                Priority.LOW: f"{self.queue_prefix}resource:gpu:priority:low",
            },
            "cpu": {
                Priority.HIGH: f"{self.queue_prefix}resource:cpu:priority:high",
                Priority.NORMAL: f"{self.queue_prefix}resource:cpu:priority:normal",
                Priority.LOW: f"{self.queue_prefix}resource:cpu:priority:low",
            },
        }
        self.worker_queues: Dict[str, str] = {}
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.worker_registry: Dict[str, Dict[str, Any]] = {}
        self.worker_load_balancer = WorkerLoadBalancer()
        self.retry_manager = CodeRetryManager(redis_client)
        self._background_tasks: list[asyncio.Task] = []

    def _prefixes_for_read(self):
        prefixes = [self.key_prefix]
        if self.legacy_prefix and self.legacy_prefix != self.key_prefix:
            prefixes.append(self.legacy_prefix)
        return prefixes

    def _key(self, prefix: str, suffix: str) -> str:
        return f"{prefix}:{suffix}"

    async def initialize(self):
        logger.info("TaskManager initialized")
        self._start_background_tasks()

    async def shutdown(self):
        for task in list(self._background_tasks):
            task.cancel()
        for task in list(self._background_tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("TaskManager shutdown")
        self._background_tasks.clear()

    def _start_background_tasks(self) -> None:
        timeout_sec = getattr(settings, "worker_queue_wait_timeout_sec", 0)
        interval_raw = getattr(settings, "worker_queue_wait_monitor_interval", 20)
        if timeout_sec > 0 and interval_raw > 0:
            self._background_tasks.append(asyncio.create_task(self._queue_wait_monitor()))

    def _parse_iso_datetime(self, value: Optional[Any]) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, bytes):
            value = value.decode()
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    def _load_task_json(self, task_data: Dict[bytes, bytes]) -> Dict[str, Any]:
        raw = task_data.get(b"data")
        if not raw:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _normalize_result_metadata(self, metadata: Any) -> Any:
        if not isinstance(metadata, dict):
            return metadata

        normalized = dict(metadata)

        if "is_kernel_used" not in normalized:
            for old_key in (
                "custom_kernel_used",
                "profiler_used",
                "cuda_profiler_used",
                "triton_profiler_used",
            ):
                if old_key in normalized:
                    normalized["is_kernel_used"] = normalized[old_key]
                    break

        for old_key in (
            "custom_kernel_used",
            "profiler_used",
            "cuda_profiler_used",
            "triton_profiler_used",
        ):
            normalized.pop(old_key, None)

        if "profiler_matches" not in normalized:
            for old_key in ("cuda_profiler_matches", "triton_profiler_matches"):
                if old_key in normalized:
                    normalized["profiler_matches"] = normalized[old_key]
                    break

        for old_key in ("cuda_profiler_matches", "triton_profiler_matches"):
            normalized.pop(old_key, None)

        if not normalized.get("profiler_matches"):
            normalized.pop("profiler_matches", None)

        if "device_info" not in normalized and "hardware" in normalized:
            hardware = normalized.get("hardware")
            if isinstance(hardware, dict):
                normalized["device_info"] = hardware
            elif hardware is not None:
                normalized["device_info"] = {"gpu_name": hardware}
        normalized.pop("hardware", None)
        if isinstance(normalized.get("device_info"), dict):
            normalized.pop("gpu_name", None)

        return normalized

    def _normalize_task_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(result)
        if "metadata" in normalized:
            normalized["metadata"] = self._normalize_result_metadata(
                normalized.get("metadata")
            )
        return normalized

    def _get_task_timeout_sec(self, task_data: Dict[bytes, bytes], task_json: Dict[str, Any]) -> int:
        timeout_val = task_json.get("timeout", task_json.get("per_task_timeout"))
        if timeout_val is None:
            timeout_val = settings.default_timeout
        try:
            timeout_sec = int(timeout_val)
        except Exception:
            timeout_sec = settings.default_timeout
        return max(0, timeout_sec)

    def _get_queue_wait_timeout_sec(
        self,
        task_data: Dict[bytes, bytes],
        task_json: Dict[str, Any],
        task_timeout_sec: int,
        default_timeout_sec: int,
    ) -> int:
        queue_timeout = task_json.get("queue_wait_timeout", task_json.get("queue_timeout"))
        if queue_timeout is None:
            queue_timeout = default_timeout_sec
        try:
            queue_timeout_sec = int(queue_timeout)
        except Exception:
            queue_timeout_sec = default_timeout_sec
        if task_timeout_sec > 0 and queue_timeout_sec > task_timeout_sec:
            queue_timeout_sec = task_timeout_sec
        return max(0, queue_timeout_sec)

    async def _requeue_task(
        self,
        task_id: str,
        task_data: Dict[bytes, bytes],
        task_json: Dict[str, Any],
        reason: str,
        now_iso: str,
    ) -> None:
        try:
            try:
                prio = task_data.get(b"priority", b"normal").decode()
                prio_enum = Priority(prio)
            except Exception:
                prio_enum = Priority.NORMAL
            task_json["assigned_worker"] = ""
            task_json["queue_timeout_reason"] = reason
            task_json["queue_timeout_at"] = now_iso
            await self.redis.hset(
                f"{self.task_prefix}{task_id}",
                mapping={
                    "data": json.dumps(task_json),
                    "assigned_worker": "",
                    "assigned_at": "",
                    "queue_timeout_reason": reason,
                    "queue_timeout_at": now_iso,
                    "updated_at": now_iso,
                },
            )
            queue_key = self.priority_queues[prio_enum]
            await self.redis.lpush(queue_key, task_id)
        except Exception as e:
            logger.error(f"Failed to requeue task {task_id}: {e}")

    async def _queue_wait_monitor(self) -> None:
        timeout_sec = getattr(settings, "worker_queue_wait_timeout_sec", 0)
        interval_raw = getattr(settings, "worker_queue_wait_monitor_interval", 20)
        interval = max(5, interval_raw)
        scan_limit = max(1, getattr(settings, "worker_queue_wait_scan_limit", 200))
        if timeout_sec <= 0 or interval_raw <= 0:
            logger.info("Queue wait monitor disabled (worker_queue_wait_timeout_sec<=0)")
            return

        while True:
            try:
                worker_keys = await self.redis.keys(f"{self.worker_prefix}*")
                worker_ids = [key.decode().replace(self.worker_prefix, "") for key in worker_keys]
                now = datetime.now()
                now_iso = now.isoformat()

                for worker_id in worker_ids:
                    worker_queue_key = self.worker_queues.get(
                        worker_id, f"{self.queue_prefix}worker:{worker_id}"
                    )
                    self.worker_queues[worker_id] = worker_queue_key
                    keep: list[str] = []
                    requeue: list[tuple[str, Dict[bytes, bytes], Dict[str, Any], str]] = []

                    for _ in range(scan_limit):
                        tid = await self.redis.rpop(worker_queue_key)
                        if not tid:
                            break
                        task_id = tid.decode() if isinstance(tid, bytes) else tid
                        task_data = await self.redis.hgetall(f"{self.task_prefix}{task_id}")
                        if not task_data:
                            continue
                        status = task_data.get(b"status", b"pending").decode()
                        task_json = self._load_task_json(task_data)

                        assigned_worker = task_json.get("assigned_worker") or (
                            task_data.get(b"assigned_worker", b"").decode()
                        )
                        if not assigned_worker or assigned_worker != worker_id:
                            requeue.append((task_id, task_data, task_json, "stale_assignment"))
                            continue

                        if status != TaskStatus.PENDING.value:
                            keep.append(task_id)
                            continue

                        assigned_at = self._parse_iso_datetime(task_data.get(b"assigned_at"))
                        if not assigned_at:
                            assigned_at = self._parse_iso_datetime(task_data.get(b"submitted_at"))
                        if not assigned_at:
                            keep.append(task_id)
                            continue

                        task_timeout_sec = self._get_task_timeout_sec(task_data, task_json)
                        queue_timeout_sec = self._get_queue_wait_timeout_sec(
                            task_data, task_json, task_timeout_sec, timeout_sec
                        )
                        if queue_timeout_sec <= 0:
                            keep.append(task_id)
                            continue

                        wait_sec = (now - assigned_at).total_seconds()
                        if wait_sec > queue_timeout_sec:
                            requeue.append((task_id, task_data, task_json, "queue_wait_timeout"))
                        else:
                            keep.append(task_id)

                    for task_id in reversed(keep):
                        await self.redis.rpush(worker_queue_key, task_id)

                    if requeue:
                        for task_id, task_data, task_json, reason in requeue:
                            await self._requeue_task(task_id, task_data, task_json, reason, now_iso)
                        logger.warning(
                            f"Requeued {len(requeue)} tasks from {worker_id} due to queue wait timeout"
                        )

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in queue wait monitor: {e}")
                await asyncio.sleep(interval)

    async def submit_evaluation_task(self, task_data: Dict[str, Any]) -> str:
        """Compatibility entrypoint: treat as a normal task submission."""
        return await self.submit_task(task_data)

    def _resolve_required_resource(self, task_data: Dict[str, Any]) -> str:
        raw = (
            task_data.get("required_resource")
            or task_data.get("resource_type")
            or task_data.get("worker_resource")
        )
        if raw:
            value = str(raw).strip().lower()
            if value in {"cpu", "compile", "compilation"}:
                return "cpu"
            if value in {"gpu", "cuda"}:
                return "gpu"

        stage = str(task_data.get("task_stage") or task_data.get("stage") or "").strip().lower()
        task_type = str(task_data.get("task_type") or "").strip().lower()
        if bool(task_data.get("pure_compile_task")):
            return "cpu"
        if stage in {"compile", "compilation"} or task_type in {
            "compile",
            "compilation",
            "kernel_compile",
            "cpu_compile",
        }:
            return "cpu"
        return "gpu"

    def _resource_poll_order(self, worker_resource: str) -> list[str]:
        resource = (worker_resource or "gpu").strip().lower()
        if resource == "cpu":
            return ["cpu"]
        order = ["gpu"]
        if getattr(settings, "gpu_workers_poll_cpu_tasks", True):
            order.append("cpu")
        return order

    async def submit_task(self, task_data: Dict[str, Any]) -> str:
        task_data = dict(task_data)
        if not task_data.get("toolkit"):
            task_data["toolkit"] = settings.default_toolkit
        if not task_data.get("backend_adapter"):
            task_data["backend_adapter"] = settings.default_backend_adapter
        if not task_data.get("backend"):
            task_data["backend"] = settings.default_backend

        toolkit_name = task_data.get("toolkit")
        backend_adapter = task_data.get("backend_adapter")
        if toolkit_name not in list_toolkits():
            raise ValueError(f"Unknown toolkit '{toolkit_name}'")
        if backend_adapter not in list_backends():
            raise ValueError(f"Unknown backend adapter '{backend_adapter}'")

        task_id = task_data["task_id"]
        priority = Priority(task_data.get("priority", Priority.NORMAL))
        required_resource = self._resolve_required_resource(task_data)
        task_data["required_resource"] = required_resource

        if await self.redis.exists(f"{self.task_prefix}{task_id}"):
            logger.info(f"Task {task_id} already exists, returning existing task")
            return task_id

        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            priority=priority,
            submitted_at=datetime.now(),
        )

        submitted_at = datetime.now()
        assigned_worker = task_data.get("assigned_worker") or ""
        assigned_at = submitted_at.isoformat() if assigned_worker else ""
        await self.redis.hset(
            f"{self.task_prefix}{task_id}",
            mapping={
                "data": json.dumps(task_data),
                "status": task_info.status.value,
                "priority": task_info.priority.value,
                "submitted_at": submitted_at.isoformat(),
                "assigned_worker": assigned_worker,
                "assigned_at": assigned_at,
            },
        )

        if assigned_worker:
            worker_queue_key = f"{self.queue_prefix}worker:{assigned_worker}"
            self.worker_queues.setdefault(assigned_worker, worker_queue_key)
            await self.redis.lpush(worker_queue_key, task_id)
        else:
            queue_key = self.resource_queues[required_resource][priority]
            await self.redis.lpush(queue_key, task_id)

        self.active_tasks[task_id] = task_info
        logger.info(
            f"Task {task_id} submitted with priority {priority.value}, "
            f"required_resource={required_resource}"
        )
        return task_id

    async def get_next_task(self, worker_id: str, worker_resource: str = "gpu") -> Optional[Dict[str, Any]]:
        queue_keys: list[str] = []
        for prefix in self._prefixes_for_read():
            queue_keys.append(f"{prefix}:queue:worker:{worker_id}")

            for resource in self._resource_poll_order(worker_resource):
                for priority in (Priority.HIGH, Priority.NORMAL, Priority.LOW):
                    queue_keys.append(
                        f"{prefix}:queue:resource:{resource}:priority:{priority.value}"
                    )

            if worker_resource == "gpu":
                for priority in (Priority.HIGH, Priority.NORMAL, Priority.LOW):
                    queue_keys.append(f"{prefix}:queue:priority:{priority.value}")

        use_blocking_poll = bool(getattr(settings, "worker_use_blocking_task_poll", True))
        if use_blocking_poll:
            timeout = max(0, int(getattr(settings, "worker_task_poll_block_timeout_sec", 1)))
            popped = await self.redis.brpop(queue_keys, timeout=timeout)
            if not popped:
                return None
            _, task_id = popped
        else:
            task_id = None
            for queue_key in queue_keys:
                task_id = await self.redis.rpop(queue_key)
                if task_id is not None:
                    break
            if task_id is None:
                return None

        if isinstance(task_id, bytes):
            task_id = task_id.decode()

        task_data = None
        task_key = None
        for prefix in self._prefixes_for_read():
            candidate_key = f"{prefix}:task:{task_id}"
            data = await self.redis.hgetall(candidate_key)
            if data:
                task_key = candidate_key
                task_data = data
                break
        if not task_data or not task_key:
            return None

        raw = task_data.get(b"data")
        if not raw:
            return None

        task_json = json.loads(raw.decode())
        started_at = datetime.now().isoformat()
        await self.redis.hset(task_key, mapping={"status": TaskStatus.PROCESSING.value, "started_at": started_at})
        return task_json

    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        completed_at = datetime.now().isoformat()
        payload = json.dumps(result)
        await self.redis.hset(
            f"{self.result_prefix}{task_id}",
            mapping={"result": payload, "completed_at": completed_at},
        )
        await self.redis.hset(
            f"{self.task_prefix}{task_id}",
            mapping={"status": TaskStatus.COMPLETED.value, "completed_at": completed_at},
        )
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = TaskStatus.COMPLETED
            self.active_tasks[task_id].completed_at = datetime.fromisoformat(completed_at)

    async def fail_task(
        self,
        task_id: str,
        error_message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        prefix: Optional[str] = None,
    ):
        failed_at = datetime.now().isoformat()
        result_prefix = f"{prefix}:result:" if prefix else self.result_prefix
        task_prefix = f"{prefix}:task:" if prefix else self.task_prefix
        await self.redis.hset(
            f"{result_prefix}{task_id}",
            mapping={"error": error_message, "failed_at": failed_at, "error_code": error_code.value},
        )
        await self.redis.hset(
            f"{task_prefix}{task_id}",
            mapping={"status": TaskStatus.FAILED.value, "failed_at": failed_at},
        )
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = TaskStatus.FAILED
            self.active_tasks[task_id].error_message = error_message

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        for prefix in self._prefixes_for_read():
            result_data = await self.redis.hgetall(f"{prefix}:result:{task_id}")
            if result_data:
                status = TaskStatus.COMPLETED if b"result" in result_data else TaskStatus.FAILED
                if b"result" in result_data:
                    try:
                        payload = json.loads(result_data[b"result"].decode())
                        if payload.get("status") == "failed":
                            status = TaskStatus.FAILED
                    except Exception:
                        pass
                return {
                    "task_id": task_id,
                    "status": status.value,
                    "completed_at": result_data.get(b"completed_at", b"").decode()
                    if b"completed_at" in result_data
                    else None,
                    "failed_at": result_data.get(b"failed_at", b"").decode()
                    if b"failed_at" in result_data
                    else None,
                    "error_message": result_data.get(b"error", b"").decode()
                    if b"error" in result_data
                    else None,
                }

            task_data = await self.redis.hgetall(f"{prefix}:task:{task_id}")
            if task_data:
                return {
                    "task_id": task_id,
                    "status": task_data.get(b"status", b"pending").decode(),
                    "submitted_at": task_data.get(b"submitted_at", b"").decode(),
                    "started_at": task_data.get(b"started_at", b"").decode(),
                }

        return None

    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        for prefix in self._prefixes_for_read():
            result_data = await self.redis.hgetall(f"{prefix}:result:{task_id}")
            if not result_data:
                continue

            if b"result" in result_data:
                result = json.loads(result_data[b"result"].decode())
                result = self._normalize_task_result(result)
                return {
                    "completed_at": result_data.get(b"completed_at", b"").decode(),
                    **result,
                }
            if b"error" in result_data:
                return {
                    "failed_at": result_data.get(b"failed_at", b"").decode(),
                    "error_message": result_data[b"error"].decode(),
                    "error_code": result_data.get(b"error_code", b"UNKNOWN_ERROR").decode(),
                }
        return None

    async def cancel_task(self, task_id: str) -> bool:
        for prefix in self._prefixes_for_read():
            task_data = await self.redis.hgetall(f"{prefix}:task:{task_id}")
            if not task_data:
                continue
            status = task_data.get(b"status", b"").decode()
            if status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
                return False
            await self.fail_task(task_id, "Task cancelled", ErrorCode.SYSTEM_ERROR, prefix=prefix)
            return True
        return False

    async def get_queue_status(self) -> Dict[str, Any]:
        pending = 0
        pending_by_prefix: Dict[str, int] = {}
        resource_pending: Dict[str, int] = {"cpu": 0, "gpu": 0}
        for prefix in self._prefixes_for_read():
            prefix_pending = 0
            for priority in (Priority.HIGH, Priority.NORMAL, Priority.LOW):
                queue_key = f"{prefix}:queue:priority:{priority.value}"
                prefix_pending += await self.redis.llen(queue_key)
            for resource in ("cpu", "gpu"):
                for priority in (Priority.HIGH, Priority.NORMAL, Priority.LOW):
                    queue_key = f"{prefix}:queue:resource:{resource}:priority:{priority.value}"
                    count = await self.redis.llen(queue_key)
                    resource_pending[resource] += count
                    prefix_pending += count
            pending_by_prefix[prefix] = prefix_pending
            pending += prefix_pending
        worker_queues = {k: await self.redis.llen(v) for k, v in self.worker_queues.items()}
        return {
            "pending": pending,
            "pending_by_prefix": pending_by_prefix,
            "resource_pending": resource_pending,
            "worker_queues": worker_queues,
        }

    async def register_worker(
        self,
        worker_id: str,
        device: str,
        node_id: Optional[str] = None,
        hostname: Optional[str] = None,
        worker_type: Optional[str] = None,
    ) -> bool:
        now = datetime.now().isoformat()
        resolved_type = worker_type or ("cpu" if device.startswith("cpu") else "gpu")
        await self.redis.hset(
            f"{self.worker_prefix}{worker_id}",
            mapping={
                "device": device,
                "worker_type": resolved_type,
                "status": "online",
                "last_heartbeat": now,
                "node_id": node_id or "",
                "hostname": hostname or "",
            },
        )
        self.worker_registry[worker_id] = {
            "device": device,
            "worker_type": resolved_type,
            "status": "online",
            "last_heartbeat": now,
            "node_id": node_id or "",
            "hostname": hostname or "",
        }
        await self.worker_load_balancer.register_worker(worker_id, device)
        return True

    async def unregister_worker(self, worker_id: str) -> bool:
        await self.redis.hset(
            f"{self.worker_prefix}{worker_id}",
            mapping={"status": "offline", "last_heartbeat": datetime.now().isoformat()},
        )
        self.worker_registry.pop(worker_id, None)
        await self.worker_load_balancer.unregister_worker(worker_id)
        return True

    async def get_worker_data(self, worker_id: str) -> Dict[bytes, bytes]:
        for prefix in self._prefixes_for_read():
            data = await self.redis.hgetall(f"{prefix}:worker:{worker_id}")
            if data:
                return data
        return {}

    async def get_workers_status(self) -> Dict[str, Any]:
        return self.worker_registry

    async def update_worker_heartbeat(self, worker_id: str) -> None:
        now = datetime.now().isoformat()
        await self.redis.hset(
            f"{self.worker_prefix}{worker_id}",
            mapping={"last_heartbeat": now, "status": "online"},
        )
        if worker_id in self.worker_registry:
            self.worker_registry[worker_id]["last_heartbeat"] = now
            self.worker_registry[worker_id]["status"] = "online"
        await self.worker_load_balancer.update_worker_heartbeat(worker_id)


__all__ = ["TaskManager"]
