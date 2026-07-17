"""Core TaskManager for KernelGym server.

This is a minimal, generic scheduler-backed task manager without workflow semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

from kernelgym.common import TaskStatus, Priority, ErrorCode
from kernelgym.config import settings
from kernelgym.backend import list_backends
from kernelgym.server.code_retry_manager import CodeRetryManager
from kernelgym.toolkit import list_toolkits
from kernelgym.utils.task_status import task_status_from_result_payload

logger = logging.getLogger(__name__)

# A queued task in one of these terminal states was cancelled (or already
# finished) after it was enqueued; it must be dropped instead of dispatched.
_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED.value,
        TaskStatus.FAILED.value,
        TaskStatus.TIMEOUT.value,
    }
)


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
        self.worker_index_key = f"{self.key_prefix}:workers"
        self.node_map_key = f"{self.key_prefix}:nodes"
        self.status_prefix = f"{self.key_prefix}:status:"

        self.resource_queues = {
            "cpu": f"{self.queue_prefix}resource:cpu",
            "gpu": f"{self.queue_prefix}resource:gpu",
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

    def _worker_index_for_prefix(self, prefix: str) -> str:
        return f"{prefix}:workers"

    @staticmethod
    def _decode_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @classmethod
    def _node_identity(cls, worker_info: Dict[Any, Any]) -> tuple[str, str]:
        node_id = cls._decode_value(worker_info.get("node_id") or worker_info.get(b"node_id"))
        hostname = cls._decode_value(worker_info.get("hostname") or worker_info.get(b"hostname"))
        return node_id, hostname

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @classmethod
    def _task_node_affinity(cls, task_json: Dict[str, Any]) -> tuple[str, str, bool]:
        affinity = str(task_json.get("node_affinity") or "").strip().lower()
        node_id = str(
            task_json.get("target_node_id")
            or task_json.get("artifact_node_id")
            or task_json.get("preferred_node_id")
            or ""
        ).strip()
        hostname = str(task_json.get("target_hostname") or task_json.get("artifact_hostname") or "").strip()
        required = affinity == "required" or bool(node_id or hostname)
        return node_id, hostname, required

    @classmethod
    def _task_matches_worker_node(cls, task_json: Dict[str, Any], worker_info: Dict[Any, Any]) -> bool:
        required_node_id, required_hostname, required = cls._task_node_affinity(task_json)
        if not required:
            return True
        worker_node_id, worker_hostname = cls._node_identity(worker_info)
        node_ok = not required_node_id or required_node_id in {worker_node_id, worker_hostname}
        host_ok = not required_hostname or required_hostname in {worker_hostname, worker_node_id}
        return node_ok and host_ok

    async def _load_task_data(
        self, task_id: str
    ) -> tuple[str | None, Dict[bytes, bytes] | None, Dict[str, Any] | None]:
        for prefix in self._prefixes_for_read():
            candidate_key = f"{prefix}:task:{task_id}"
            data = await self.redis.hgetall(candidate_key)
            if not data:
                continue
            raw = data.get(b"data")
            if not raw:
                return candidate_key, data, None
            return candidate_key, data, json.loads(raw.decode())
        return None, None, None

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
        if getattr(settings, "dead_worker_reaper_enabled", True):
            self._background_tasks.append(asyncio.create_task(self._dead_worker_reaper()))

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
            task_json["assigned_worker"] = ""
            task_json["queue_timeout_reason"] = reason
            task_json["queue_timeout_at"] = now_iso
            required_resource = self._resolve_required_resource(task_json)
            task_json["required_resource"] = required_resource
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
            queue_key = self.resource_queues[required_resource]
            await self.redis.lpush(queue_key, task_id)
        except Exception as e:
            logger.error(f"Failed to requeue task {task_id}: {e}")

    async def _requeue_or_fail(
        self,
        task_id: str,
        task_data: Dict[bytes, bytes],
        task_json: Dict[str, Any],
        reason: str,
        now_iso: str,
    ) -> str:
        """Requeue a stuck task, or fail it once it has exhausted its requeue budget.

        Prevents requeue loops: a task whose worker keeps dying (or whose resource
        never gets a live worker) is failed cleanly after ``max_requeue_attempts``
        instead of bouncing forever. Returns "requeued" or "failed".
        """
        max_attempts = max(0, int(getattr(settings, "max_requeue_attempts", 3)))
        try:
            prev = int(self._decode_value(task_data.get(b"requeue_count")) or 0)
        except (TypeError, ValueError):
            prev = 0
        if max_attempts and prev >= max_attempts:
            await self.fail_task(
                task_id,
                error_message=(
                    f"Task failed after {prev} requeue attempt(s) (limit {max_attempts}); last reason: {reason}"
                ),
                error_code=ErrorCode.SYSTEM_ERROR,
            )
            logger.warning("Reaper failing task %s after %s requeue(s) (last reason=%s)", task_id, prev, reason)
            return "failed"
        new_count = prev + 1
        task_json["requeue_count"] = new_count
        await self._requeue_task(task_id, task_data, task_json, reason, now_iso)
        await self.redis.hset(f"{self.task_prefix}{task_id}", mapping={"requeue_count": str(new_count)})
        return "requeued"

    def _worker_is_dead(
        self,
        worker_id: str,
        workers: Dict[str, Any],
        now: datetime,
        dead_after_s: int,
    ) -> bool:
        """A worker is dead if unregistered, offline, or silent longer than dead_after_s."""
        info = workers.get(worker_id)
        if not info:
            return True
        if str(info.get("status") or "").lower() == "offline":
            return True
        heartbeat = self._parse_datetime(str(info.get("last_heartbeat") or ""))
        if heartbeat is None:
            return True
        return (now - heartbeat).total_seconds() > dead_after_s

    async def _reap_orphaned_task(self, task_id: str, worker_id: str, now_iso: str) -> None:
        """Recover one task stranded in a dead worker's queue (requeue, or fail if capped)."""
        task_data = await self.redis.hgetall(f"{self.task_prefix}{task_id}")
        if not task_data:
            return  # task record already gone; nothing to recover
        status = self._decode_value(task_data.get(b"status")) or TaskStatus.PENDING.value
        if status in _TERMINAL_TASK_STATUSES:
            return  # already finished/failed/timed out; drop
        task_json = self._load_task_json(task_data)
        base_id = str(task_json.get("base_task_id") or "")
        try:
            if await self._dequeued_task_cancelled(self.key_prefix, task_id, base_id):
                return  # cancelled after enqueue; drop
        except Exception:
            pass
        outcome = await self._requeue_or_fail(task_id, task_data, task_json, f"dead_worker:{worker_id}", now_iso)
        logger.info("Reaped orphaned task %s from dead worker %s -> %s", task_id, worker_id, outcome)

    async def _dead_worker_reaper(self) -> None:
        """Recover tasks stranded in the queues of dead / unregistered workers.

        The queue-wait monitor only scans queues of *currently registered* workers,
        so a task left in a dead worker's queue is invisible to it and sits pending
        until its client-side timeout. This reaper scans *all* worker queues in redis
        and, for workers confirmed dead (stale heartbeat, offline, or unregistered),
        drains their queues back to the shared resource queue (subject to the requeue
        cap in ``_requeue_or_fail``).

        Safety: only DEAD workers' queues are touched (liveness gate), and draining
        uses atomic ``rpop`` so a task is in exactly one place -- no double-dispatch.
        Resource queues are intentionally left alone (that is capacity, not an orphan).
        """
        interval = max(5, int(getattr(settings, "dead_worker_reaper_interval_s", 20)))
        dead_after = max(1, int(getattr(settings, "dead_worker_timeout_s", 90)))
        scan_limit = max(1, int(getattr(settings, "worker_queue_wait_scan_limit", 200)))
        logger.info("Dead-worker reaper started (interval=%ss, dead_after=%ss)", interval, dead_after)
        while True:
            try:
                now = datetime.now()
                now_iso = now.isoformat()
                workers = await self.get_workers_status()
                queue_keys: list[str] = []
                for prefix in self._prefixes_for_read():
                    async for key in self.redis.scan_iter(f"{prefix}:queue:worker:*", count=500):
                        queue_keys.append(key.decode() if isinstance(key, bytes) else str(key))
                total_reaped = 0
                for queue_key in queue_keys:
                    worker_id = queue_key.rsplit(":queue:worker:", 1)[-1]
                    if not worker_id or not self._worker_is_dead(worker_id, workers, now, dead_after):
                        continue  # liveness gate: never touch a live worker's queue
                    for _ in range(scan_limit):
                        tid = await self.redis.rpop(queue_key)
                        if not tid:
                            break
                        task_id = tid.decode() if isinstance(tid, bytes) else tid
                        try:
                            await self._reap_orphaned_task(task_id, worker_id, now_iso)
                            total_reaped += 1
                        except Exception as exc:
                            logger.error("Reaper failed on task %s (worker %s): %s", task_id, worker_id, exc)
                if total_reaped:
                    logger.warning("Dead-worker reaper recovered %s orphaned task(s)", total_reaped)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Dead-worker reaper loop error: %s", exc)
            await asyncio.sleep(interval)

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
                worker_ids = [
                    key.decode().replace(self.worker_prefix, "")
                    async for key in self.redis.scan_iter(f"{self.worker_prefix}*", count=500)
                ]
                now = datetime.now()
                now_iso = now.isoformat()

                for worker_id in worker_ids:
                    worker_queue_key = self.worker_queues.get(worker_id, f"{self.queue_prefix}worker:{worker_id}")
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
                            await self._requeue_or_fail(task_id, task_data, task_json, reason, now_iso)
                        logger.warning(f"Requeued {len(requeue)} tasks from {worker_id} due to queue wait timeout")

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
        explicit = str(task_data.get("required_resource") or "").strip().lower()
        if explicit in self.resource_queues:
            return explicit
        stage = str(task_data.get("task_stage") or task_data.get("stage") or "").strip().lower()
        if stage == "compile" or bool(task_data.get("pure_compile_task")):
            return "cpu"
        return "gpu"

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
        force_refresh = bool(task_data.get("force_refresh"))

        if await self.redis.exists(f"{self.task_prefix}{task_id}"):
            if force_refresh:
                logger.info("Task %s already exists; force_refresh requested, resubmitting", task_id)
                await self._clear_task_for_refresh(task_id)
            else:
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
        if "assigned_worker" in task_data:
            task_data["assigned_worker"] = assigned_worker
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
            queue_key = self.resource_queues[required_resource]
            await self.redis.lpush(queue_key, task_id)

        self.active_tasks[task_id] = task_info
        logger.info(f"Task {task_id} submitted with resource {required_resource}")
        return task_id

    async def _clear_task_for_refresh(self, task_id: str) -> None:
        """Remove completed or queued state so force_refresh can submit fresh work."""
        for queue_key in [*self.resource_queues.values(), *self.worker_queues.values()]:
            try:
                await self.redis.lrem(queue_key, 0, task_id)
            except AttributeError:
                pass
        await self.redis.delete(f"{self.task_prefix}{task_id}", f"{self.result_prefix}{task_id}")
        self.active_tasks.pop(task_id, None)

    async def get_next_task(self, worker_id: str, resources: Optional[list[str]] = None) -> Optional[Dict[str, Any]]:
        resources = resources or ["gpu"]
        worker_info = await self.get_worker_data(worker_id)
        scan_limit = max(1, getattr(settings, "worker_queue_wait_scan_limit", 200))

        for prefix in self._prefixes_for_read():
            worker_queue_key = f"{prefix}:queue:worker:{worker_id}"
            direct_task_id = await self.redis.rpop(worker_queue_key)
            if direct_task_id is not None:
                task_id = direct_task_id.decode() if isinstance(direct_task_id, bytes) else direct_task_id
                task_key, task_hash, task_json = await self._load_task_data(task_id)
                dispatchable = (
                    task_key
                    and task_json is not None
                    and not self._dropped_before_dispatch(task_hash)
                    and not await self._dequeued_task_cancelled(prefix, task_id, task_json.get("base_task_id") or "")
                )
                if dispatchable:
                    started_at = datetime.now().isoformat()
                    await self.redis.hset(
                        task_key,
                        mapping={"status": TaskStatus.PROCESSING.value, "started_at": started_at},
                    )
                    return task_json
                # Cancelled/terminal/missing: discard it and fall through to the
                # resource queues instead of handing a dead task to the worker.
                logger.info("Dropping cancelled/terminal/missing task %s from worker queue", task_id)

            for resource in resources:
                queue_key = f"{prefix}:queue:resource:{resource}"
                deferred_task_ids: list[str] = []
                try:
                    for _ in range(scan_limit):
                        queued_task_id = await self.redis.rpop(queue_key)
                        if queued_task_id is None:
                            break
                        task_id = queued_task_id.decode() if isinstance(queued_task_id, bytes) else queued_task_id
                        task_key, task_hash, task_json = await self._load_task_data(task_id)
                        if not task_key or task_json is None:
                            continue
                        if self._dropped_before_dispatch(task_hash) or await self._dequeued_task_cancelled(
                            prefix, task_id, task_json.get("base_task_id") or ""
                        ):
                            # Cancelled/terminal task (or a sub-task whose workflow
                            # parent was cancelled): drop it from the queue (do not
                            # re-defer) so it is never dispatched to a worker.
                            logger.info("Dropping cancelled/terminal task %s from resource queue", task_id)
                            continue
                        if not self._task_matches_worker_node(task_json, worker_info):
                            deferred_task_ids.append(task_id)
                            continue
                        started_at = datetime.now().isoformat()
                        await self.redis.hset(
                            task_key,
                            mapping={"status": TaskStatus.PROCESSING.value, "started_at": started_at},
                        )
                        return task_json
                finally:
                    for deferred_task_id in deferred_task_ids:
                        await self.redis.lpush(queue_key, deferred_task_id)
        return None

    async def complete_task(self, task_id: str, result: Dict[str, Any], request_hash: Optional[str] = None):
        timing_start = time.time()
        timing_start_mono_ns = time.monotonic_ns()
        completed_at = datetime.now().isoformat()
        metadata = result.setdefault("metadata", {})
        metadata["tm_enter_monotonic_ns"] = timing_start_mono_ns
        task_status = task_status_from_result_payload(result)
        result["status"] = task_status.value

        status_hset_start = time.time()
        status_mapping = {"status": task_status.value, "completed_at": completed_at}
        if task_status == TaskStatus.FAILED:
            status_mapping["failed_at"] = completed_at
        elif task_status == TaskStatus.TIMEOUT:
            status_mapping["timeout_at"] = completed_at
        task_key = f"{self.task_prefix}{task_id}"
        result_key = f"{self.result_prefix}{task_id}"
        await self.redis.hset(task_key, mapping=status_mapping)
        status_hset_s = time.time() - status_hset_start
        metadata["tm_status_hset_s"] = status_hset_s

        json_start = time.time()
        payload = json.dumps(result)
        json_dumps_s = time.time() - json_start

        result_hset_start = time.time()
        result_mapping: Dict[str, Any] = {"result": payload, "completed_at": completed_at}
        if request_hash:
            result_mapping["request_hash"] = request_hash
        await self.redis.hset(result_key, mapping=result_mapping)
        await self._expire_terminal_records(task_key, result_key)
        result_hset_s = time.time() - result_hset_start

        metadata["tm_json_dumps_s"] = json_dumps_s
        metadata["tm_result_hset_s"] = result_hset_s
        metadata["tm_complete_task_s"] = time.time() - timing_start
        metadata["tm_exit_monotonic_ns"] = time.monotonic_ns()
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = task_status
            self.active_tasks[task_id].completed_at = datetime.fromisoformat(completed_at)
            if task_status != TaskStatus.COMPLETED:
                self.active_tasks[task_id].error_message = result.get("error_message")

        logger.info(
            "[TaskManagerTiming] task=%s json_dumps_s=%.4f result_hset_s=%.4f status_hset_s=%.4f total_s=%.4f",
            task_id,
            json_dumps_s,
            result_hset_s,
            status_hset_s,
            metadata["tm_complete_task_s"],
        )

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
        task_status = task_status_from_result_payload({"status": "failed", "error_code": error_code})
        timing_key = "timeout_at" if task_status == TaskStatus.TIMEOUT else "failed_at"
        task_key = f"{task_prefix}{task_id}"
        result_key = f"{result_prefix}{task_id}"
        await self.redis.hset(
            result_key,
            mapping={
                "error": error_message,
                timing_key: failed_at,
                "error_code": error_code.value,
            },
        )
        await self.redis.hset(
            task_key,
            mapping={"status": task_status.value, timing_key: failed_at},
        )
        await self._expire_terminal_records(task_key, result_key)
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = task_status
            self.active_tasks[task_id].error_message = error_message

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        for prefix in self._prefixes_for_read():
            result_data = await self.redis.hgetall(f"{prefix}:result:{task_id}")
            if result_data:
                status = TaskStatus.COMPLETED if b"result" in result_data else TaskStatus.FAILED
                error_message = None
                if b"result" in result_data:
                    try:
                        payload = json.loads(result_data[b"result"].decode())
                        status = task_status_from_result_payload(payload)
                        if status != TaskStatus.COMPLETED:
                            error_message = payload.get("error_message")
                    except Exception:
                        pass
                elif result_data.get(b"error_code"):
                    status = task_status_from_result_payload(
                        {"status": status.value, "error_code": result_data[b"error_code"].decode()}
                    )
                if error_message is None and b"error" in result_data:
                    error_message = result_data.get(b"error", b"").decode()
                return {
                    "task_id": task_id,
                    "status": status.value,
                    "completed_at": result_data.get(b"completed_at", b"").decode()
                    if b"completed_at" in result_data
                    else None,
                    "failed_at": result_data.get(b"failed_at", b"").decode() if b"failed_at" in result_data else None,
                    "error_message": error_message,
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

    async def get_task_result(
        self,
        task_id: str,
        expected_request_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        for prefix in self._prefixes_for_read():
            result_data = await self.redis.hgetall(f"{prefix}:result:{task_id}")
            if not result_data:
                continue

            stored_request_hash = result_data.get(b"request_hash")
            if expected_request_hash is not None:
                if not stored_request_hash:
                    logger.info(
                        "Ignoring cached result for task=%s because it has no request_hash",
                        task_id,
                    )
                    return None
                if stored_request_hash.decode() != expected_request_hash:
                    logger.warning(
                        "Ignoring cached result for task=%s due to request_hash mismatch stored=%s expected=%s",
                        task_id,
                        stored_request_hash.decode(),
                        expected_request_hash,
                    )
                    return None

            if b"result" in result_data:
                result = json.loads(result_data[b"result"].decode())
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

    @staticmethod
    def _dropped_before_dispatch(task_hash: Optional[Dict[bytes, bytes]]) -> bool:
        """True if a just-dequeued task is already terminal and must be discarded."""
        if not task_hash:
            return False
        status = task_hash.get(b"status", b"")
        if isinstance(status, bytes):
            status = status.decode()
        return status in _TERMINAL_TASK_STATUSES

    def _cancel_key(self, task_id: str, prefix: Optional[str] = None) -> str:
        base = prefix or self.key_prefix
        return f"{base}:cancel:{task_id}"

    def _workflow_key(self, base_id: str, prefix: Optional[str] = None) -> str:
        base = prefix or self.key_prefix
        return f"{base}:workflow:{base_id}"

    def _marker_ttl(self) -> int:
        return max(60, int(getattr(settings, "default_timeout", 300)) * 2)

    async def _expire_terminal_records(self, task_key: str, result_key: str) -> None:
        task_ttl = int(getattr(settings, "terminal_task_ttl_sec", 0) or 0)
        result_ttl = int(getattr(settings, "terminal_result_ttl_sec", 0) or 0)
        if task_ttl > 0:
            await self.redis.expire(task_key, task_ttl)
        if result_ttl > 0:
            await self.redis.expire(result_key, result_ttl)

    async def _mark_task_cancelled(self, task_id: str, prefix: str) -> None:
        """Publish a short-lived cancellation marker that running workers poll."""
        try:
            await self.redis.set(self._cancel_key(task_id, prefix), "1", ex=self._marker_ttl())
        except Exception as exc:  # pragma: no cover - best effort marker
            logger.warning("Failed to set cancellation marker for %s: %s", task_id, exc)

    async def register_workflow(self, base_id: str) -> None:
        """Mark a workflow (whose parent id has no task hash mid-flight) active.

        This lets ``cancel_task`` recognize an in-flight ``/evaluate`` parent id
        and publish a cancellation marker that its running sub-tasks poll via
        their ``base_task_id``.
        """
        if not base_id:
            return
        try:
            await self.redis.set(self._workflow_key(base_id), "1", ex=self._marker_ttl())
        except Exception as exc:  # pragma: no cover - best effort registration
            logger.warning("Failed to register workflow %s: %s", base_id, exc)

    async def unregister_workflow(self, base_id: str) -> None:
        if not base_id:
            return
        for prefix in self._prefixes_for_read():
            try:
                await self.redis.delete(self._workflow_key(base_id, prefix))
            except Exception:  # pragma: no cover - best effort cleanup
                continue

    async def _is_workflow_active(self, base_id: str) -> bool:
        for prefix in self._prefixes_for_read():
            try:
                if await self.redis.exists(self._workflow_key(base_id, prefix)):
                    return True
            except Exception:  # pragma: no cover
                continue
        return False

    async def is_task_cancelled(self, task_id: str) -> bool:
        """Whether a cancellation has been requested for ``task_id`` (any prefix)."""
        for prefix in self._prefixes_for_read():
            try:
                if await self.redis.exists(self._cancel_key(task_id, prefix)):
                    return True
            except Exception:  # pragma: no cover - treat redis hiccup as "not cancelled"
                continue
        return False

    async def _dequeued_task_cancelled(self, prefix: str, task_id: str, base_id: str = "") -> bool:
        """Prefix-local check: was this just-dequeued task (or its workflow parent) cancelled?

        One ``EXISTS`` over the task's own cancel marker plus its ``base_task_id``
        marker, so a sub-task whose parent ``/evaluate`` was cancelled while it sat
        queued is dropped instead of dispatched.
        """
        keys = [self._cancel_key(task_id, prefix)]
        if base_id and base_id != task_id:
            keys.append(self._cancel_key(base_id, prefix))
        try:
            return bool(await self.redis.exists(*keys))
        except Exception:  # pragma: no cover
            return False

    async def _remove_task_from_queues(self, task_id: str, prefix: str, assigned_worker: str = "") -> int:
        """Remove a pending task id from every queue it could be waiting in."""
        queue_keys = [
            f"{prefix}:queue:resource:cpu",
            f"{prefix}:queue:resource:gpu",
        ]
        if assigned_worker:
            queue_keys.append(f"{prefix}:queue:worker:{assigned_worker}")
        for worker_queue_key in self.worker_queues.values():
            if worker_queue_key not in queue_keys:
                queue_keys.append(worker_queue_key)
        removed = 0
        for queue_key in queue_keys:
            try:
                removed += await self.redis.lrem(queue_key, 0, task_id)
            except Exception:  # pragma: no cover - best effort cleanup
                continue
        if removed:
            logger.info("Removed cancelled task %s from %d queue position(s)", task_id, removed)
        return removed

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task or an in-flight ``/evaluate`` workflow.

        Direct task: pulled from the queue if pending (never dispatched) and
        recorded with a terminal cancelled result; if running, a cancellation
        marker lets the worker kill its CUDA subprocess promptly.

        Workflow parent id (no task hash while sub-tasks run): a cancellation
        marker is published under the parent id, which its running sub-tasks
        poll via their ``base_task_id`` and abort.

        Returns False only for unknown or already-terminal ids.
        """
        for prefix in self._prefixes_for_read():
            task_data = await self.redis.hgetall(f"{prefix}:task:{task_id}")
            if not task_data:
                continue
            status = task_data.get(b"status", b"").decode()
            if status in _TERMINAL_TASK_STATUSES:
                return False
            assigned_worker = task_data.get(b"assigned_worker", b"").decode()
            # 1. Publish a cancellation marker so a worker already running this
            #    task can detect it and kill its CUDA subprocess promptly.
            await self._mark_task_cancelled(task_id, prefix)
            # 2. Pull the task out of any pending queue so it is never dispatched.
            await self._remove_task_from_queues(task_id, prefix, assigned_worker)
            # 3. Record the terminal cancelled result/status.
            await self.fail_task(task_id, "Task cancelled", ErrorCode.SYSTEM_ERROR, prefix=prefix)
            cancelled_at = datetime.now().isoformat()
            await self.redis.hset(f"{prefix}:task:{task_id}", mapping={"cancelled_at": cancelled_at})
            logger.info(
                "Cancelled task %s (prior_status=%s, assigned_worker=%s)",
                task_id,
                status or "pending",
                assigned_worker or "-",
            )
            return True

        # No direct task hash. If this is the parent id of an in-flight workflow,
        # publish a cancellation marker that its running sub-tasks (which carry
        # base_task_id == task_id) poll and act on. The workflow controller
        # writes the parent's terminal result when it aborts.
        if await self._is_workflow_active(task_id):
            await self._mark_task_cancelled(task_id, self.key_prefix)
            logger.info("Cancelled in-flight workflow %s (no direct task; marked base scope)", task_id)
            return True
        return False

    async def get_queue_status(self) -> Dict[str, Any]:
        pending = 0
        pending_by_prefix: Dict[str, int] = {}
        for prefix in self._prefixes_for_read():
            prefix_pending = 0
            for resource in ("cpu", "gpu"):
                queue_key = f"{prefix}:queue:resource:{resource}"
                prefix_pending += await self.redis.llen(queue_key)
            pending_by_prefix[prefix] = prefix_pending
            pending += prefix_pending
        worker_queues = {k: await self.redis.llen(v) for k, v in self.worker_queues.items()}
        return {
            "pending": pending,
            "pending_by_prefix": pending_by_prefix,
            "worker_queues": worker_queues,
        }

    async def register_worker(
        self, worker_id: str, device: str, node_id: Optional[str] = None, hostname: Optional[str] = None
    ) -> bool:
        now = datetime.now().isoformat()
        await self.redis.hset(
            f"{self.worker_prefix}{worker_id}",
            mapping={
                "device": device,
                "status": "online",
                "last_heartbeat": now,
                "node_id": node_id or "",
                "hostname": hostname or "",
            },
        )
        await self.redis.sadd(self.worker_index_key, worker_id)
        self.worker_registry[worker_id] = {
            "device": device,
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
        await self.redis.sadd(self.worker_index_key, worker_id)
        self.worker_registry.pop(worker_id, None)
        await self.worker_load_balancer.unregister_worker(worker_id)
        return True

    async def get_worker_data(self, worker_id: str) -> Dict[bytes, bytes]:
        for prefix in self._prefixes_for_read():
            data = await self.redis.hgetall(f"{prefix}:worker:{worker_id}")
            if data:
                return data
        return {}

    @staticmethod
    def _decode_redis_hash(data: Dict[Any, Any]) -> Dict[str, Any]:
        decoded: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="replace")
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            decoded[str(key)] = value
        return decoded

    async def get_workers_status(self) -> Dict[str, Any]:
        workers = {worker_id: dict(info) for worker_id, info in self.worker_registry.items()}
        for prefix in self._prefixes_for_read():
            index_key = self._worker_index_for_prefix(prefix)
            indexed_ids = await self.redis.smembers(index_key)
            if not indexed_ids:
                indexed_ids = set()
                async for key in self.redis.scan_iter(f"{prefix}:worker:*", count=1000):
                    key_text = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
                    worker_id = key_text.rsplit(":worker:", 1)[-1]
                    indexed_ids.add(worker_id)
                if indexed_ids:
                    await self.redis.sadd(index_key, *indexed_ids)

            for raw_worker_id in indexed_ids:
                worker_id = (
                    raw_worker_id.decode("utf-8", errors="replace")
                    if isinstance(raw_worker_id, bytes)
                    else str(raw_worker_id)
                )
                data = await self.redis.hgetall(f"{prefix}:worker:{worker_id}")
                if not data:
                    await self.redis.srem(index_key, worker_id)
                    continue
                workers.setdefault(worker_id, {}).update(self._decode_redis_hash(data))
        return workers

    async def _resource_worker_candidates(
        self,
        resource: str,
        *,
        idle_only: bool,
        target_node_id: Optional[str] = None,
        target_hostname: Optional[str] = None,
        max_heartbeat_age_s: int = 30,
    ) -> list[tuple[str, Dict[str, Any]]]:
        now = datetime.now()
        candidates: list[tuple[str, Dict[str, Any]]] = []
        for worker_id, info in (await self.get_workers_status()).items():
            device = str(info.get("device") or "")
            if resource == "gpu" and not device.startswith("cuda:"):
                continue
            if resource == "cpu" and device != "cpu":
                continue
            if str(info.get("status") or "").lower() != "online":
                continue
            if idle_only and str(info.get("current_task") or "").strip():
                continue
            heartbeat = self._parse_datetime(str(info.get("last_heartbeat") or ""))
            if heartbeat is None or (now - heartbeat).total_seconds() > max_heartbeat_age_s:
                continue
            node_id, hostname = self._node_identity(info)
            if target_node_id and target_node_id not in {node_id, hostname}:
                continue
            if target_hostname and target_hostname not in {hostname, node_id}:
                continue
            candidates.append((worker_id, info))
        candidates.sort(key=lambda item: item[0])
        return candidates

    async def select_idle_worker(
        self,
        resource: str,
        *,
        target_node_id: Optional[str] = None,
        target_hostname: Optional[str] = None,
        max_heartbeat_age_s: int = 30,
    ) -> Optional[Dict[str, Any]]:
        candidates = await self._resource_worker_candidates(
            resource,
            idle_only=True,
            target_node_id=target_node_id,
            target_hostname=target_hostname,
            max_heartbeat_age_s=max_heartbeat_age_s,
        )
        if not candidates:
            return None
        index = 0
        try:
            index = int(await self.redis.incr(f"{self.key_prefix}:worker_select:{resource}:rr")) - 1
        except Exception:
            pass
        worker_id, info = candidates[index % len(candidates)]
        node_id, hostname = self._node_identity(info)
        return {
            "worker_id": worker_id,
            "device": str(info.get("device") or ""),
            "node_id": node_id,
            "hostname": hostname,
        }

    async def select_worker_by_task_id(
        self,
        resource: str,
        task_id: str,
        *,
        target_node_id: Optional[str] = None,
        target_hostname: Optional[str] = None,
        max_heartbeat_age_s: int = 30,
    ) -> Optional[Dict[str, Any]]:
        candidates = await self._resource_worker_candidates(
            resource,
            idle_only=False,
            target_node_id=target_node_id,
            target_hostname=target_hostname,
            max_heartbeat_age_s=max_heartbeat_age_s,
        )
        if not candidates:
            return None
        digest = hashlib.sha256(str(task_id).encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(candidates)
        worker_id, info = candidates[index]
        node_id, hostname = self._node_identity(info)
        return {
            "worker_id": worker_id,
            "device": str(info.get("device") or ""),
            "node_id": node_id,
            "hostname": hostname,
        }

    async def update_worker_heartbeat(
        self,
        worker_id: str,
        node_id: Optional[str] = None,
        hostname: Optional[str] = None,
    ) -> None:
        now = datetime.now().isoformat()
        mapping = {"last_heartbeat": now, "status": "online"}
        if node_id:
            mapping["node_id"] = node_id
        if hostname:
            mapping["hostname"] = hostname
        await self.redis.hset(
            f"{self.worker_prefix}{worker_id}",
            mapping=mapping,
        )
        await self.redis.sadd(self.worker_index_key, worker_id)
        if worker_id in self.worker_registry:
            self.worker_registry[worker_id]["last_heartbeat"] = now
            self.worker_registry[worker_id]["status"] = "online"
            if node_id:
                self.worker_registry[worker_id]["node_id"] = node_id
            if hostname:
                self.worker_registry[worker_id]["hostname"] = hostname
        await self.worker_load_balancer.update_worker_heartbeat(worker_id)


__all__ = ["TaskManager"]
