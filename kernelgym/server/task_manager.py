"""Core TaskManager for KernelGym server.

This is a minimal, generic scheduler-backed task manager without workflow semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

from kernelgym.common import TaskStatus, Priority, ErrorCode
from kernelgym.config import settings
from kernelgym.backend import list_backends
from kernelgym.server.code_retry_manager import CodeRetryManager
from kernelgym.toolkit import list_toolkits
from kernelgym.utils.gpu_quarantine import read_gpu_quarantine
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


class StaleTaskClaimError(RuntimeError):
    """A superseded worker attempt tried to mutate a newer task claim."""


class FrozenTaskClaimError(RuntimeError):
    """Terminal publication was blocked by a containment-owned claim."""


class TaskRefreshConflictError(RuntimeError):
    """A force refresh tried to replace work that may still be executing."""


@dataclass(frozen=True)
class TaskClaim:
    prefix: str
    inflight_queue: str
    source_queue: str
    token: str
    entry: str
    worker_instance: str


_CLAIM_GPU_TASK_LUA = r"""
-- kernelgym:claim-gpu-task-v2
-- KEYS: source queue, worker inflight queue
-- ARGV: token, worker id, worker instance, task-key prefix, scan limit
--
-- Queue ids whose task is missing/terminal, or which duplicate an active
-- tokenized claim, are stale queue debris.  Skip them atomically without ever
-- overwriting the live attempt's token.
local scan_limit = tonumber(ARGV[5]) or 32
for _ = 1, scan_limit do
    local task_id = redis.call('RPOP', KEYS[1])
    if not task_id then
        return false
    end
    local task_key = ARGV[4] .. task_id
    if redis.call('EXISTS', task_key) == 1 then
        local status = redis.call('HGET', task_key, 'status')
        local current_token = redis.call('HGET', task_key, 'claim_token')
        if status == 'pending' and (not current_token or current_token == '') then
            local entry = ARGV[1] .. '|' .. task_id
            redis.call('LPUSH', KEYS[2], entry)
            redis.call(
                'HSET', task_key,
                'claim_token', ARGV[1],
                'claim_worker', ARGV[2],
                'claim_worker_instance', ARGV[3],
                'claim_source_queue', KEYS[1],
                'claim_inflight_queue', KEYS[2],
                'claim_recovery_state', '',
                'claim_recovery_reason', '',
                'claim_recovery_at', ''
            )
            return task_id
        end
    end
end
return false
"""


_SUBMIT_TASK_IF_ABSENT_LUA = r"""
-- kernelgym:submit-task-if-absent-v1
-- KEYS: task hash, destination queue
-- ARGV: task id, task mapping JSON
--
-- Creating the hash and publishing its queue id are one Redis operation.  A
-- retry or concurrent submit observes either the complete task or no task; it
-- can never observe a pending hash whose queue publication was skipped.
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
local task_mapping = cjson.decode(ARGV[2])
for field, value in pairs(task_mapping) do
    redis.call('HSET', KEYS[1], field, value)
end
redis.call('LREM', KEYS[2], 0, ARGV[1])
redis.call('LPUSH', KEYS[2], ARGV[1])
return 1
"""


_FORCE_REFRESH_TASK_LUA = r"""
-- kernelgym:force-refresh-task-v1
-- KEYS: task hash, result hash, destination queue, cleanup queues...
-- ARGV: task id, replacement task mapping JSON, worker-queue prefix
--
-- A force refresh is an atomic replacement, not delete-then-create.  It may
-- create a previously absent task or replace a terminal task whose attempt is
-- fully released.  Pending/processing tasks and every tokenized/frozen claim
-- are immutable here, so a concurrent GPU claim either wins before this script
-- and fences the refresh, or runs afterwards against the replacement payload.
local task_exists = redis.call('EXISTS', KEYS[1]) == 1
if task_exists then
    if redis.call('HGET', KEYS[1], 'claim_recovery_state') == 'frozen' then
        return -2
    end
    local claim_token = redis.call('HGET', KEYS[1], 'claim_token')
    if claim_token and claim_token ~= '' then
        return -1
    end
    local status = redis.call('HGET', KEYS[1], 'status') or ''
    if status ~= 'completed' and status ~= 'failed' and status ~= 'timeout' then
        return -4
    end

    -- Direct queues are data-dependent, so derive the authoritative old queue
    -- while the task hash is still protected by this Redis transaction.
    local old_assigned_worker = redis.call('HGET', KEYS[1], 'assigned_worker') or ''
    if old_assigned_worker ~= '' then
        redis.call('LREM', ARGV[3] .. old_assigned_worker, 0, ARGV[1])
    end
end

for index = 3, #KEYS do
    redis.call('LREM', KEYS[index], 0, ARGV[1])
end
redis.call('DEL', KEYS[1], KEYS[2])
local replacement_mapping = cjson.decode(ARGV[2])
for field, value in pairs(replacement_mapping) do
    redis.call('HSET', KEYS[1], field, value)
end
redis.call('LPUSH', KEYS[3], ARGV[1])
return task_exists and 2 or 1
"""


_QUEUE_WAIT_REQUEUE_LUA = r"""
-- kernelgym:queue-wait-requeue-v2
-- KEYS: task hash, source worker queue, destination resource queue,
--       task cancel, optional workflow-parent cancel
-- ARGV: task id, source worker id, operation, expected assigned worker,
--       expected assigned-at, expected submitted-at, expected serialized data,
--       replacement data, reason, timestamp, move token
if (redis.call('HGET', KEYS[1], 'assigned_worker') or '') ~= ARGV[4] then
    return -3
end
if (redis.call('HGET', KEYS[1], 'assigned_at') or '') ~= ARGV[5] then
    return -4
end
if (redis.call('HGET', KEYS[1], 'submitted_at') or '') ~= ARGV[6] then
    return -5
end
if (redis.call('HGET', KEYS[1], 'data') or '') ~= ARGV[7] then
    return -6
end
if ARGV[3] == 'remove_stale_copy' then
    -- A task assigned to worker B may have an obsolete copy in worker A's
    -- queue.  Remove only A's list entry; never unassign or requeue B's task.
    if ARGV[4] == '' or ARGV[4] == ARGV[2] then
        return -8
    end
    if redis.call('LREM', KEYS[2], 1, ARGV[1]) == 1 then
        return 2
    end
    return -7
end
if ARGV[3] ~= 'requeue' or (ARGV[4] ~= '' and ARGV[4] ~= ARGV[2]) then
    return -8
end
if redis.call('HGET', KEYS[1], 'status') ~= 'pending' then
    return 0
end
local claim_token = redis.call('HGET', KEYS[1], 'claim_token')
if claim_token and claim_token ~= '' then
    return -1
end
if redis.call('EXISTS', KEYS[4]) == 1 or redis.call('EXISTS', KEYS[5]) == 1 then
    return -2
end
if redis.call('LREM', KEYS[2], 1, ARGV[1]) ~= 1 then
    -- The worker atomically claimed it after LRANGE but before this script.
    return -7
end
redis.call('LREM', KEYS[3], 0, ARGV[1])
redis.call(
    'HSET', KEYS[1],
    'data', ARGV[8],
    'status', 'pending',
    'assigned_worker', '',
    'assigned_at', '',
    'started_at', '',
    'claim_token', '',
    'claim_worker', '',
    'claim_worker_instance', '',
    'claim_source_queue', '',
    'claim_inflight_queue', '',
    'claim_recovery_state', '',
    'claim_recovery_reason', '',
    'claim_recovery_at', '',
    'queue_timeout_reason', ARGV[9],
    'queue_timeout_at', ARGV[10],
    'queue_move_token', ARGV[11],
    'updated_at', ARGV[10]
)
redis.call('LPUSH', KEYS[3], ARGV[1])
return 1
"""


_CONDITIONAL_REQUEUE_LUA = r"""
-- kernelgym:conditional-requeue-v2
-- KEYS: task hash, destination queue, worker inflight queue, task cancel,
--       optional workflow-parent cancel
-- ARGV: task id, claim entry, token, serialized data, reason, timestamp,
--       destination assigned worker, destination assigned-at,
--       recovery-fence release mode (none/execution/all)
local current_token = redis.call('HGET', KEYS[1], 'claim_token')
if not current_token or current_token ~= ARGV[3] then
    redis.call('LREM', KEYS[3], 1, ARGV[2])
    return -3
end
local recovery_state = redis.call('HGET', KEYS[1], 'claim_recovery_state') or ''
if recovery_state ~= '' then
    local release_allowed =
        (recovery_state == 'execution_fenced' and (ARGV[9] == 'execution' or ARGV[9] == 'all'))
        or (recovery_state == 'frozen' and ARGV[9] == 'all')
    if not release_allowed then
        return -4
    end
end
local function clear_claim()
    redis.call(
        'HSET', KEYS[1],
        'claim_token', '',
        'claim_worker', '',
        'claim_worker_instance', '',
        'claim_source_queue', '',
        'claim_inflight_queue', '',
        'claim_recovery_state', '',
        'claim_recovery_reason', '',
        'claim_recovery_at', ''
    )
end
local status = redis.call('HGET', KEYS[1], 'status')
if not status then
    redis.call('LREM', KEYS[3], 1, ARGV[2])
    clear_claim()
    return 0
end
if status == 'completed' or status == 'failed' or status == 'timeout' then
    redis.call('LREM', KEYS[3], 1, ARGV[2])
    clear_claim()
    return -1
end
if redis.call('EXISTS', KEYS[4]) == 1 or redis.call('EXISTS', KEYS[5]) == 1 then
    redis.call('LREM', KEYS[3], 1, ARGV[2])
    clear_claim()
    return -2
end
redis.call('LREM', KEYS[3], 1, ARGV[2])
redis.call('LREM', KEYS[2], 0, ARGV[1])
redis.call(
    'HSET', KEYS[1],
    'data', ARGV[4],
    'status', 'pending',
    'assigned_worker', ARGV[7],
    'assigned_at', ARGV[8],
    'started_at', '',
    'claim_token', '',
    'claim_worker', '',
    'claim_worker_instance', '',
    'claim_source_queue', '',
    'claim_inflight_queue', '',
    'claim_recovery_state', '',
    'claim_recovery_reason', '',
    'claim_recovery_at', '',
    'queue_timeout_reason', ARGV[5],
    'queue_timeout_at', ARGV[6],
    'updated_at', ARGV[6]
)
redis.call('LPUSH', KEYS[2], ARGV[1])
return 1
"""


_MARK_CLAIM_PROCESSING_LUA = r"""
-- kernelgym:mark-claim-processing-v3
-- KEYS: task hash, inflight queue, task cancel, optional parent cancel
-- ARGV: task id, claim entry, token, started timestamp, worker id,
--       worker instance, source queue
local current_token = redis.call('HGET', KEYS[1], 'claim_token')
if not current_token or current_token ~= ARGV[3] then
    redis.call('LREM', KEYS[2], 1, ARGV[2])
    return -2
end
local function clear_claim()
    redis.call(
        'HSET', KEYS[1],
        'claim_token', '',
        'claim_worker', '',
        'claim_worker_instance', '',
        'claim_source_queue', '',
        'claim_inflight_queue', '',
        'claim_recovery_state', '',
        'claim_recovery_reason', '',
        'claim_recovery_at', ''
    )
end
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'pending' then
    redis.call('LREM', KEYS[2], 1, ARGV[2])
    clear_claim()
    return 0
end
if redis.call('EXISTS', KEYS[3]) == 1 or redis.call('EXISTS', KEYS[4]) == 1 then
    redis.call('LREM', KEYS[2], 1, ARGV[2])
    clear_claim()
    return -1
end
redis.call(
    'HSET', KEYS[1],
    'status', 'processing',
    'started_at', ARGV[4],
    'claim_worker', ARGV[5],
    'claim_worker_instance', ARGV[6],
    'claim_source_queue', ARGV[7],
    'claim_inflight_queue', KEYS[2],
    'claim_recovery_state', 'execution_fenced',
    'claim_recovery_reason', 'GPU execution is in flight; automatic recovery requires proven process containment',
    'claim_recovery_at', ARGV[4]
)
return 1
"""


_RETURN_CLAIM_LUA = r"""
-- kernelgym:return-claim-v2
-- KEYS: source queue, inflight queue, task hash
-- ARGV: task id, claim entry, token
local current_token = redis.call('HGET', KEYS[3], 'claim_token')
if not current_token or current_token ~= ARGV[3] then
    redis.call('LREM', KEYS[2], 1, ARGV[2])
    return 0
end
redis.call('LREM', KEYS[2], 1, ARGV[2])
redis.call('LREM', KEYS[1], 0, ARGV[1])
redis.call('LPUSH', KEYS[1], ARGV[1])
redis.call(
    'HSET', KEYS[3],
    'claim_token', '',
    'claim_worker', '',
    'claim_worker_instance', '',
    'claim_source_queue', '',
    'claim_inflight_queue', '',
    'claim_recovery_state', '',
    'claim_recovery_reason', '',
    'claim_recovery_at', ''
)
return 1
"""


_ACK_CLAIM_LUA = r"""
-- kernelgym:ack-claim-v3
-- KEYS: task hash, inflight queue
-- ARGV: claim entry, token, allow recovery-fence release (0/1)
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('LREM', KEYS[2], 1, ARGV[1])
    return 1
end
local current_token = redis.call('HGET', KEYS[1], 'claim_token')
if not current_token or current_token ~= ARGV[2] then
    redis.call('LREM', KEYS[2], 1, ARGV[1])
    return 0
end
local recovery_state = redis.call('HGET', KEYS[1], 'claim_recovery_state') or ''
if recovery_state ~= '' and ARGV[3] ~= '1' then
    return -2
end
redis.call('LREM', KEYS[2], 1, ARGV[1])
redis.call(
    'HSET', KEYS[1],
    'claim_token', '',
    'claim_worker', '',
    'claim_worker_instance', '',
    'claim_source_queue', '',
    'claim_inflight_queue', '',
    'claim_recovery_state', '',
    'claim_recovery_reason', '',
    'claim_recovery_at', ''
)
return 1
"""


_FREEZE_CLAIM_RECOVERY_LUA = r"""
-- kernelgym:freeze-claim-recovery-v1
-- KEYS: task hash
-- ARGV: token, reason, timestamp
local current_token = redis.call('HGET', KEYS[1], 'claim_token')
if not current_token or current_token ~= ARGV[1] then
    return 0
end
redis.call(
    'HSET', KEYS[1],
    'claim_recovery_state', 'frozen',
    'claim_recovery_reason', ARGV[2],
    'claim_recovery_at', ARGV[3]
)
return 1
"""


_FINALIZE_CLAIM_LUA = r"""
-- kernelgym:finalize-claim-v2
-- KEYS: task hash, result hash, inflight queue
-- ARGV: token, claim entry, task mapping JSON, result mapping JSON,
--       task TTL, result TTL, allow containment-frozen terminal (0/1)
local current_token = redis.call('HGET', KEYS[1], 'claim_token')
if not current_token or current_token ~= ARGV[1] then
    redis.call('LREM', KEYS[3], 1, ARGV[2])
    return 0
end
if redis.call('HGET', KEYS[1], 'claim_recovery_state') == 'frozen' and ARGV[7] ~= '1' then
    return -2
end
local task_mapping = cjson.decode(ARGV[3])
for field, value in pairs(task_mapping) do
    redis.call('HSET', KEYS[1], field, value)
end
local result_mapping = cjson.decode(ARGV[4])
for field, value in pairs(result_mapping) do
    redis.call('HSET', KEYS[2], field, value)
end
redis.call('LREM', KEYS[3], 1, ARGV[2])
local task_ttl = tonumber(ARGV[5])
local result_ttl = tonumber(ARGV[6])
if task_ttl and task_ttl > 0 then
    redis.call('EXPIRE', KEYS[1], task_ttl)
end
if result_ttl and result_ttl > 0 then
    redis.call('EXPIRE', KEYS[2], result_ttl)
end
return 1
"""


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
        # GPU dequeues use an atomic source->inflight claim.  The in-memory
        # index is only an optimization; the Redis list is the crash-recovery
        # authority and survives a worker-process exit at every dequeue point.
        self._task_claims: Dict[str, TaskClaim] = {}
        # Recovery is repeatable because an earlier process can leave durable
        # inflight state after this process's first empty scan.  Serialize scans
        # per stable worker id so concurrent get_next_task calls cannot both
        # adopt the same stale token into the in-memory claim map.
        self._inflight_recovery_locks: Dict[tuple[str, str], asyncio.Lock] = {}
        # Distinguishes overlapping/replacement processes that reuse a stable
        # worker id.  The per-task token fences mutations; this instance id
        # prevents a process from treating its own active claim as crash debris.
        self.worker_instance_id = uuid.uuid4().hex

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

    async def _gpu_worker_admission_open(self, worker_id: str, worker_info: Dict[Any, Any]) -> bool:
        """Require an explicit healthy heartbeat plus no persistent latch."""

        decoded = self._decode_redis_hash(worker_info)
        device = str(decoded.get("device") or "")
        health_state = str(decoded.get("health_state") or "").lower()
        accepting = str(decoded.get("accepting_tasks") or "").lower()
        hostname = str(decoded.get("hostname") or "").strip()
        if (
            not device.startswith("cuda:")
            or not hostname
            or str(decoded.get("status") or "").lower() != "online"
            or str(decoded.get("online") or "").lower() != "true"
            or accepting != "true"
            or health_state not in {"healthy", "degraded_check"}
        ):
            return False
        heartbeat = self._parse_datetime(str(decoded.get("last_heartbeat") or ""))
        if heartbeat is None:
            return False
        now = datetime.now(tz=heartbeat.tzinfo)
        max_age = max(1, int(getattr(settings, "worker_monitor_heartbeat_timeout", 120)))
        if (now - heartbeat).total_seconds() > max_age:
            return False
        try:
            quarantine = await read_gpu_quarantine(
                self.redis,
                worker_id,
                device=device,
                hostname=hostname,
            )
        except Exception as exc:
            logger.error("GPU admission check failed for %s; failing closed: %s", worker_id, exc)
            return False
        return quarantine is None

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

    async def _load_task_data_for_prefix(
        self,
        prefix: str,
        task_id: str,
    ) -> tuple[str | None, Dict[bytes, bytes] | None, Dict[str, Any] | None]:
        """Load only the namespace from which this claim was created."""

        task_key = f"{prefix}:task:{task_id}"
        data = await self.redis.hgetall(task_key)
        if not data:
            return None, None, None
        raw = data.get(b"data")
        if not raw:
            return task_key, data, None
        return task_key, data, json.loads(raw.decode())

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

    async def _conditionally_requeue_waiting_task(
        self,
        *,
        prefix: str,
        worker_queue_key: str,
        task_id: str,
        task_data: Dict[bytes, bytes],
        task_json: Dict[str, Any],
        reason: str,
        now_iso: str,
        operation: str,
    ) -> int:
        """Atomically move one unchanged, still-pending direct task.

        ``LRANGE`` makes the monitor scan non-destructive.  This Lua CAS is the
        only mutation: it rechecks the task snapshot, cancellation markers,
        absence of a GPU claim token, and ownership of one source-list entry.
        """

        decoded = self._decode_redis_hash(task_data)
        raw_data = str(decoded.get("data") or "")
        restored_task = dict(task_json)
        restored_task["assigned_worker"] = ""
        restored_task["queue_timeout_reason"] = reason
        restored_task["queue_timeout_at"] = now_iso
        required_resource = self._resolve_required_resource(restored_task)
        restored_task["required_resource"] = required_resource
        queue_key = f"{prefix}:queue:resource:{required_resource}"
        own_cancel_key = self._cancel_key(task_id, prefix)
        base_task_id = str(restored_task.get("base_task_id") or "")
        parent_cancel_key = self._cancel_key(base_task_id, prefix) if base_task_id else own_cancel_key
        result = await self.redis.eval(
            _QUEUE_WAIT_REQUEUE_LUA,
            5,
            f"{prefix}:task:{task_id}",
            worker_queue_key,
            queue_key,
            own_cancel_key,
            parent_cancel_key,
            task_id,
            worker_queue_key.rsplit(":queue:worker:", 1)[-1],
            operation,
            str(decoded.get("assigned_worker") or ""),
            str(decoded.get("assigned_at") or ""),
            str(decoded.get("submitted_at") or ""),
            raw_data,
            json.dumps(restored_task),
            reason,
            now_iso,
            uuid.uuid4().hex,
        )
        return int(result)

    async def requeue_unstarted_task(
        self,
        task_data: Dict[str, Any],
        reason: str,
        *,
        restore_claim_source: bool = False,
        release_frozen_claim: bool = False,
        release_execution_fence: bool = False,
    ) -> bool:
        """Conditionally put an unstarted GPU claim back behind admission.

        The Lua transaction refuses to overwrite a terminal/cancelled task and
        de-duplicates both the inflight and destination lists.  This closes the
        cancellation race that could otherwise resurrect a terminal task.
        ``release_execution_fence`` is limited to a task proven not to have
        reached a child; only proven-safe startup recovery may set
        ``release_frozen_claim`` and release a containment-upgraded fence.
        """

        task_id = str(task_data.get("task_id") or "")
        if not task_id:
            raise ValueError("Cannot requeue task without task_id")
        if release_frozen_claim and release_execution_fence:
            raise ValueError("Choose one recovery-fence release authority")
        claim = self._task_claims.get(task_id)
        if claim is None:
            logger.warning("Refusing to requeue task %s without its claim token", task_id)
            return False
        now_iso = datetime.now().isoformat()
        prefix = claim.prefix
        inflight_queue = claim.inflight_queue
        task_key = f"{prefix}:task:{task_id}"
        task_hash = await self.redis.hgetall(task_key)
        decoded_hash = self._decode_redis_hash(task_hash)
        restored_task = dict(task_data)
        required_resource = self._resolve_required_resource(restored_task)
        restored_task["required_resource"] = required_resource
        if restore_claim_source:
            assigned_worker = str(decoded_hash.get("assigned_worker") or "")
            assigned_at = str(decoded_hash.get("assigned_at") or "")
            restored_task["assigned_worker"] = assigned_worker
            queue_key = claim.source_queue
        else:
            assigned_worker = ""
            assigned_at = ""
            restored_task["assigned_worker"] = ""
            queue_key = f"{prefix}:queue:resource:{required_resource}"
        restored_task["queue_timeout_reason"] = reason
        restored_task["queue_timeout_at"] = now_iso

        own_cancel_key = self._cancel_key(task_id, prefix)
        base_task_id = str(restored_task.get("base_task_id") or "")
        parent_cancel_key = self._cancel_key(base_task_id, prefix) if base_task_id else own_cancel_key
        result = await self.redis.eval(
            _CONDITIONAL_REQUEUE_LUA,
            5,
            task_key,
            queue_key,
            inflight_queue,
            own_cancel_key,
            parent_cancel_key,
            task_id,
            claim.entry,
            claim.token,
            json.dumps(restored_task),
            reason,
            now_iso,
            assigned_worker,
            assigned_at,
            "all" if release_frozen_claim else "execution" if release_execution_fence else "none",
        )
        result_code = int(result)
        # ``-4`` is the execution/containment-fence guard. Redis deliberately leaves the
        # token and inflight entry untouched, so retain the matching local
        # claim as well; stop() may still need that exact token to finalize a
        # safely contained task. Every other non-success result either
        # acknowledged the inflight entry or proved this token stale.
        if result_code != -4:
            self._task_claims.pop(task_id, None)
        restored = result_code == 1
        if not restored:
            logger.info(
                "Did not requeue task %s because it became missing, terminal, or cancelled (result=%s)",
                task_id,
                result,
            )
        return restored

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
                now = datetime.now()
                now_iso = now.isoformat()

                for prefix in self._prefixes_for_read():
                    worker_queue_prefix = f"{prefix}:queue:worker:"
                    worker_queue_keys = [
                        self._decode_value(key)
                        async for key in self.redis.scan_iter(f"{worker_queue_prefix}*", count=500)
                        if self._decode_value(key).startswith(worker_queue_prefix)
                    ]
                    for worker_queue_key in worker_queue_keys:
                        worker_id = worker_queue_key[len(worker_queue_prefix) :]
                        if prefix == self.key_prefix:
                            self.worker_queues[worker_id] = worker_queue_key
                        # LPUSH/RPOP queues dispatch from the right, so inspect
                        # the oldest (next-to-dispatch) bounded slice.
                        raw_ids = await self.redis.lrange(worker_queue_key, -scan_limit, -1)
                        moved = 0
                        removed_stale = 0

                        for raw_task_id in raw_ids:
                            task_id = self._decode_value(raw_task_id)
                            task_data = await self.redis.hgetall(f"{prefix}:task:{task_id}")
                            if not task_data:
                                continue
                            status = self._decode_value(task_data.get(b"status") or b"pending")
                            if status != TaskStatus.PENDING.value:
                                continue
                            task_json = self._load_task_json(task_data)
                            if not task_json:
                                continue

                            assigned_worker = self._decode_value(task_data.get(b"assigned_worker"))
                            reason = ""
                            operation = "requeue"
                            if assigned_worker and assigned_worker != worker_id:
                                # This is only an obsolete A-queue copy of a task
                                # currently owned by B.  Never steal B's task.
                                operation = "remove_stale_copy"
                                reason = "stale_assignment_copy"
                            elif not assigned_worker:
                                reason = "stale_assignment"
                            else:
                                assigned_at = self._parse_iso_datetime(task_data.get(b"assigned_at"))
                                if not assigned_at:
                                    assigned_at = self._parse_iso_datetime(task_data.get(b"submitted_at"))
                                if not assigned_at:
                                    continue
                                task_timeout_sec = self._get_task_timeout_sec(task_data, task_json)
                                queue_timeout_sec = self._get_queue_wait_timeout_sec(
                                    task_data,
                                    task_json,
                                    task_timeout_sec,
                                    timeout_sec,
                                )
                                if queue_timeout_sec <= 0:
                                    continue
                                if (now - assigned_at).total_seconds() <= queue_timeout_sec:
                                    continue
                                reason = "queue_wait_timeout"

                            move_result = await self._conditionally_requeue_waiting_task(
                                prefix=prefix,
                                worker_queue_key=worker_queue_key,
                                task_id=task_id,
                                task_data=task_data,
                                task_json=task_json,
                                reason=reason,
                                now_iso=now_iso,
                                operation=operation,
                            )
                            if move_result == 1:
                                moved += 1
                            elif move_result == 2:
                                removed_stale += 1

                        if moved or removed_stale:
                            logger.warning(
                                "Queue-wait cleanup for %s/%s: requeued=%d stale_copies_removed=%d",
                                prefix,
                                worker_id,
                                moved,
                                removed_stale,
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
        if assigned_worker:
            worker_queue_key = f"{self.queue_prefix}worker:{assigned_worker}"
            self.worker_queues.setdefault(assigned_worker, worker_queue_key)
            destination_queue = worker_queue_key
        else:
            destination_queue = self.resource_queues[required_resource]

        task_mapping = {
            "data": json.dumps(task_data),
            "status": task_info.status.value,
            "priority": task_info.priority.value,
            "submitted_at": submitted_at.isoformat(),
            "assigned_worker": assigned_worker,
            "assigned_at": assigned_at,
        }
        if force_refresh:
            refresh_result = await self._force_refresh_task(
                task_id,
                task_mapping=task_mapping,
                destination_queue=destination_queue,
            )
            if refresh_result == 2:
                logger.info("Task %s terminal state atomically replaced by force_refresh", task_id)
        else:
            created = await self._submit_task_if_absent(
                task_id,
                task_mapping=task_mapping,
                destination_queue=destination_queue,
            )
            if not created:
                logger.info("Task %s already exists, returning existing task", task_id)
                return task_id

        self.active_tasks[task_id] = task_info
        logger.info(f"Task {task_id} submitted with resource {required_resource}")
        return task_id

    async def _submit_task_if_absent(
        self,
        task_id: str,
        *,
        task_mapping: Dict[str, Any],
        destination_queue: str,
    ) -> bool:
        """Atomically create and enqueue one task, or preserve its existing state."""

        created = await self.redis.eval(
            _SUBMIT_TASK_IF_ABSENT_LUA,
            2,
            f"{self.task_prefix}{task_id}",
            destination_queue,
            task_id,
            json.dumps(task_mapping),
        )
        return int(created) == 1

    async def _force_refresh_task(
        self,
        task_id: str,
        *,
        task_mapping: Dict[str, Any],
        destination_queue: str,
    ) -> int:
        """Atomically create or safely replace one terminal, unclaimed task."""

        cleanup_queues = list(
            dict.fromkeys(
                [
                    destination_queue,
                    *self.resource_queues.values(),
                    *self.worker_queues.values(),
                ]
            )
        )
        keys = [
            f"{self.task_prefix}{task_id}",
            f"{self.result_prefix}{task_id}",
            *cleanup_queues,
        ]
        result = int(
            await self.redis.eval(
                _FORCE_REFRESH_TASK_LUA,
                len(keys),
                *keys,
                task_id,
                json.dumps(task_mapping),
                f"{self.queue_prefix}worker:",
            )
        )
        if result < 0:
            reason = {
                -1: "an active claim token exists",
                -2: "claim recovery is frozen pending safe GPU containment",
                -4: "the existing task is not terminal",
            }.get(result, f"the atomic refresh guard returned {result}")
            raise TaskRefreshConflictError(f"Cannot force-refresh task {task_id}: {reason}")
        return result

    @staticmethod
    def _gpu_inflight_queue(prefix: str, worker_id: str) -> str:
        return f"{prefix}:queue:inflight:{worker_id}"

    async def _claim_gpu_task(self, prefix: str, worker_id: str, source_queue: str) -> Optional[str]:
        """Atomically create a token-fenced, crash-recoverable GPU claim."""

        inflight_queue = self._gpu_inflight_queue(prefix, worker_id)
        token = uuid.uuid4().hex
        raw_task_id = await self.redis.eval(
            _CLAIM_GPU_TASK_LUA,
            2,
            source_queue,
            inflight_queue,
            token,
            worker_id,
            self.worker_instance_id,
            f"{prefix}:task:",
            32,
        )
        if raw_task_id is None:
            return None
        task_id = raw_task_id.decode() if isinstance(raw_task_id, bytes) else str(raw_task_id)
        entry = f"{token}|{task_id}"
        self._task_claims[task_id] = TaskClaim(
            prefix=prefix,
            inflight_queue=inflight_queue,
            source_queue=source_queue,
            token=token,
            entry=entry,
            worker_instance=self.worker_instance_id,
        )
        return task_id

    async def _return_claim_to_source(self, task_id: str) -> None:
        """Atomically return a node-affinity mismatch without losing its claim."""

        claim = self._task_claims.get(task_id)
        if claim is None:
            raise StaleTaskClaimError(f"No claim token available while returning task {task_id}")
        result = await self.redis.eval(
            _RETURN_CLAIM_LUA,
            3,
            claim.source_queue,
            claim.inflight_queue,
            f"{claim.prefix}:task:{task_id}",
            task_id,
            claim.entry,
            claim.token,
        )
        self._task_claims.pop(task_id, None)
        if int(result) != 1:
            raise StaleTaskClaimError(f"Task {task_id} claim was superseded before return")

    async def acknowledge_task_claim(self, task_id: str, *, release_fenced_claim: bool = False) -> None:
        """Remove only this exact attempt from its durable inflight list."""

        claim = self._task_claims.get(task_id)
        if claim is None:
            return
        acknowledged = await self.redis.eval(
            _ACK_CLAIM_LUA,
            2,
            f"{claim.prefix}:task:{task_id}",
            claim.inflight_queue,
            claim.entry,
            claim.token,
            1 if release_fenced_claim else 0,
        )
        if int(acknowledged) != -2:
            self._task_claims.pop(task_id, None)

    async def freeze_task_claim(self, task_id: str, reason: str) -> bool:
        """Fence crash recovery for this exact attempt during unsafe containment.

        The marker is token-scoped: an old process cannot freeze a replacement
        attempt. Ordinary terminal publication is blocked after this upgrade;
        only a caller that has affirmatively proven containment may opt in to
        finalizing the same token. A replacement process likewise must not
        requeue it until an explicit safe startup releases fenced claims.
        """

        claim = self._task_claims.get(task_id)
        if claim is None:
            logger.critical("Cannot freeze recovery for task %s without its local claim token", task_id)
            return False
        frozen = await self.redis.eval(
            _FREEZE_CLAIM_RECOVERY_LUA,
            1,
            f"{claim.prefix}:task:{task_id}",
            claim.token,
            reason,
            datetime.now().isoformat(),
        )
        if int(frozen) != 1:
            self._task_claims.pop(task_id, None)
            logger.info("Task %s claim was superseded before recovery could be frozen", task_id)
            return False
        logger.critical("Froze automatic recovery for task %s: %s", task_id, reason)
        return True

    async def _mark_claim_processing(
        self,
        *,
        prefix: str,
        task_id: str,
        task_key: str,
        task_json: Dict[str, Any],
        worker_id: str,
        inflight_queue: str,
        source_queue: str,
    ) -> bool:
        """Atomically mark a GPU claim running and fence crash recovery.

        The fence is installed before the task is returned to GPUWorker, so a
        process crash at any later instruction cannot make an unproven CUDA
        attempt automatically retryable. Same-token terminal publication
        clears it; an explicitly proven-safe fresh startup may also release it.
        """

        claim = self._task_claims.get(task_id)
        if claim is None:
            raise StaleTaskClaimError(f"No claim token available while dispatching task {task_id}")
        base_task_id = str(task_json.get("base_task_id") or "")
        own_cancel_key = self._cancel_key(task_id, prefix)
        parent_cancel_key = self._cancel_key(base_task_id, prefix) if base_task_id else own_cancel_key
        result = await self.redis.eval(
            _MARK_CLAIM_PROCESSING_LUA,
            4,
            task_key,
            inflight_queue,
            own_cancel_key,
            parent_cancel_key,
            task_id,
            claim.entry,
            claim.token,
            datetime.now().isoformat(),
            worker_id,
            claim.worker_instance,
            source_queue,
        )
        if int(result) != 1:
            self._task_claims.pop(task_id, None)
            return False
        return True

    async def _recover_gpu_inflight(
        self,
        prefix: str,
        worker_id: str,
        *,
        release_frozen_claims: bool = False,
    ) -> None:
        """Restore claims left by an earlier process before this worker dequeues."""

        lock_key = (prefix, worker_id)
        recovery_lock = self._inflight_recovery_locks.get(lock_key)
        if recovery_lock is None:
            recovery_lock = asyncio.Lock()
            self._inflight_recovery_locks[lock_key] = recovery_lock
        inflight_queue = self._gpu_inflight_queue(prefix, worker_id)
        scan_limit = max(1, int(getattr(settings, "gpu_inflight_recovery_scan_limit", 256)))
        async with recovery_lock:
            raw_entries = await self.redis.lrange(inflight_queue, 0, scan_limit - 1)
            for raw_entry in raw_entries:
                entry = raw_entry.decode() if isinstance(raw_entry, bytes) else str(raw_entry)
                token, separator, task_id = entry.partition("|")
                if not separator or not token or not task_id:
                    # This working tree has not been deployed with the older
                    # bare-id inflight format.  Refuse to guess ownership.
                    logger.critical("Malformed/legacy GPU inflight entry retained for manual review: %s", entry)
                    continue
                task_key = f"{prefix}:task:{task_id}"
                task_hash = await self.redis.hgetall(task_key)
                if not task_hash:
                    await self.redis.lrem(inflight_queue, 1, entry)
                    continue
                decoded_hash = self._decode_redis_hash(task_hash)
                if str(decoded_hash.get("claim_token") or "") != token:
                    await self.redis.lrem(inflight_queue, 1, entry)
                    continue
                recovery_state = str(decoded_hash.get("claim_recovery_state") or "")
                if recovery_state and not release_frozen_claims:
                    logger.critical(
                        "Retaining fenced inflight claim for task %s on GPU worker %s: %s",
                        task_id,
                        worker_id,
                        str(decoded_hash.get("claim_recovery_reason") or "unsafe containment"),
                    )
                    continue
                claim_instance = str(decoded_hash.get("claim_worker_instance") or "")
                if self.worker_instance_id and claim_instance == self.worker_instance_id:
                    # This process owns the attempt; a concurrent get_next call
                    # must not recover work it is already executing.
                    continue
                source_queue = str(
                    decoded_hash.get("claim_source_queue")
                    or f"{prefix}:queue:resource:{self._resolve_required_resource(self._load_task_json(task_hash))}"
                )
                self._task_claims[task_id] = TaskClaim(
                    prefix=prefix,
                    inflight_queue=inflight_queue,
                    source_queue=source_queue,
                    token=token,
                    entry=entry,
                    worker_instance=claim_instance,
                )
                task_json = self._load_task_json(task_hash)
                if not task_json or self._dropped_before_dispatch(task_hash):
                    await self.acknowledge_task_claim(
                        task_id,
                        release_fenced_claim=release_frozen_claims,
                    )
                    continue
                if await self._dequeued_task_cancelled(
                    prefix,
                    task_id,
                    str(task_json.get("base_task_id") or ""),
                ):
                    # A cancellation marker can race an unsafe-shutdown fence:
                    # cancel_task cannot finalize a frozen claim, but a proven-
                    # safe fresh startup owns the authority to close it.  Write
                    # the terminal result and ACK the exact token atomically so
                    # waiters cannot remain stuck in ``processing`` forever.
                    try:
                        await self.fail_task(
                            task_id,
                            "Task cancelled",
                            ErrorCode.SYSTEM_ERROR,
                            prefix=prefix,
                            allow_frozen_claim=release_frozen_claims,
                        )
                    except (StaleTaskClaimError, FrozenTaskClaimError) as exc:
                        # Another actor changed ownership after this scan read
                        # the token.  Keep recovering the remaining entries;
                        # the winner owns this task's terminal/recovery state.
                        self._task_claims.pop(task_id, None)
                        logger.warning(
                            "Skipped superseded cancelled claim %s while recovering GPU worker %s: %s",
                            task_id,
                            worker_id,
                            exc,
                        )
                        continue
                    try:
                        await self.redis.hset(
                            task_key,
                            mapping={"cancelled_at": datetime.now().isoformat()},
                        )
                    except Exception as exc:
                        # The Lua transaction above already committed the
                        # terminal result and exact-token ACK. cancelled_at is
                        # diagnostic metadata and must not stop this scan.
                        logger.warning(
                            "Failed to add cancelled_at metadata for recovered task %s: %s",
                            task_id,
                            exc,
                        )
                    continue
                restored = await self.requeue_unstarted_task(
                    task_json,
                    reason="worker_process_recovered_inflight_claim",
                    restore_claim_source=True,
                    release_frozen_claim=release_frozen_claims,
                )
                if restored:
                    logger.warning(
                        "Recovered task %s from stale inflight list for GPU worker %s",
                        task_id,
                        worker_id,
                    )

    async def recover_gpu_inflight(self, worker_id: str, *, release_frozen_claims: bool = False) -> None:
        """Recover ordinary crash claims, optionally after a proven-safe restart.

        ``release_frozen_claims`` is a trust boundary.  GPUWorker uses it only
        after the persistent quarantine is absent and a fresh worker pool has
        passed CUDA initialization.  Routine control-plane recovery leaves
        unsafe-shutdown claims untouched.
        """

        for prefix in self._prefixes_for_read():
            await self._recover_gpu_inflight(
                prefix,
                worker_id,
                release_frozen_claims=release_frozen_claims,
            )

    async def get_next_task(self, worker_id: str, resources: Optional[list[str]] = None) -> Optional[Dict[str, Any]]:
        resources = resources or ["gpu"]
        worker_info = await self.get_worker_data(worker_id)
        decoded_worker_info = self._decode_redis_hash(worker_info)
        is_gpu_worker = str(decoded_worker_info.get("device") or "").startswith("cuda:") or "gpu" in resources
        if is_gpu_worker and not await self._gpu_worker_admission_open(worker_id, worker_info):
            logger.warning("GPU worker %s is not accepting tasks; queue pop skipped", worker_id)
            return None
        if is_gpu_worker:
            # Admission must close before recovery as well as before dequeue:
            # a quarantined replacement cannot expose its predecessor's claim
            # to another GPU.  Frozen unsafe-shutdown claims additionally need
            # the explicit safe-start release performed by GPUWorker.start().
            await self.recover_gpu_inflight(worker_id)
        scan_limit = max(1, getattr(settings, "worker_queue_wait_scan_limit", 200))

        for prefix in self._prefixes_for_read():
            worker_queue_key = f"{prefix}:queue:worker:{worker_id}"
            direct_inflight_queue = self._gpu_inflight_queue(prefix, worker_id)
            direct_task_id = (
                await self._claim_gpu_task(prefix, worker_id, worker_queue_key)
                if is_gpu_worker
                else await self.redis.rpop(worker_queue_key)
            )
            if direct_task_id is not None:
                task_id = direct_task_id.decode() if isinstance(direct_task_id, bytes) else direct_task_id
                task_key, task_hash, task_json = await self._load_task_data_for_prefix(prefix, task_id)
                dispatchable = (
                    task_key
                    and task_json is not None
                    and not self._dropped_before_dispatch(task_hash)
                    and not await self._dequeued_task_cancelled(prefix, task_id, task_json.get("base_task_id") or "")
                )
                if dispatchable:
                    fresh_worker_info = await self.get_worker_data(worker_id) if is_gpu_worker else worker_info
                    if is_gpu_worker and not await self._gpu_worker_admission_open(worker_id, fresh_worker_info):
                        await self.requeue_unstarted_task(
                            task_json,
                            reason="gpu_admission_closed_after_direct_dequeue",
                        )
                        return None
                    if is_gpu_worker:
                        if not await self._mark_claim_processing(
                            prefix=prefix,
                            task_id=task_id,
                            task_key=task_key,
                            task_json=task_json,
                            worker_id=worker_id,
                            inflight_queue=direct_inflight_queue,
                            source_queue=worker_queue_key,
                        ):
                            continue
                    else:
                        started_at = datetime.now().isoformat()
                        await self.redis.hset(
                            task_key,
                            mapping={"status": TaskStatus.PROCESSING.value, "started_at": started_at},
                        )
                    return task_json
                # Cancelled/terminal/missing: discard it and fall through to the
                # resource queues instead of handing a dead task to the worker.
                if is_gpu_worker:
                    await self.acknowledge_task_claim(task_id)
                logger.info("Dropping cancelled/terminal/missing task %s from worker queue", task_id)

            for resource in resources:
                queue_key = f"{prefix}:queue:resource:{resource}"
                deferred_task_ids: list[str] = []
                deferred_claims: list[str] = []
                try:
                    for _ in range(scan_limit):
                        queued_task_id = (
                            await self._claim_gpu_task(prefix, worker_id, queue_key)
                            if is_gpu_worker
                            else await self.redis.rpop(queue_key)
                        )
                        if queued_task_id is None:
                            break
                        task_id = queued_task_id.decode() if isinstance(queued_task_id, bytes) else queued_task_id
                        task_key, task_hash, task_json = await self._load_task_data_for_prefix(prefix, task_id)
                        if not task_key or task_json is None:
                            if is_gpu_worker:
                                await self.acknowledge_task_claim(task_id)
                            continue
                        if self._dropped_before_dispatch(task_hash) or await self._dequeued_task_cancelled(
                            prefix, task_id, task_json.get("base_task_id") or ""
                        ):
                            # Cancelled/terminal task (or a sub-task whose workflow
                            # parent was cancelled): drop it from the queue (do not
                            # re-defer) so it is never dispatched to a worker.
                            if is_gpu_worker:
                                await self.acknowledge_task_claim(task_id)
                            logger.info("Dropping cancelled/terminal task %s from resource queue", task_id)
                            continue
                        if not self._task_matches_worker_node(task_json, worker_info):
                            if is_gpu_worker:
                                deferred_claims.append(task_id)
                            else:
                                deferred_task_ids.append(task_id)
                            continue
                        fresh_worker_info = await self.get_worker_data(worker_id) if is_gpu_worker else worker_info
                        if is_gpu_worker and not await self._gpu_worker_admission_open(worker_id, fresh_worker_info):
                            await self.requeue_unstarted_task(
                                task_json,
                                reason="gpu_admission_closed_after_resource_dequeue",
                            )
                            return None
                        if is_gpu_worker:
                            if not await self._mark_claim_processing(
                                prefix=prefix,
                                task_id=task_id,
                                task_key=task_key,
                                task_json=task_json,
                                worker_id=worker_id,
                                inflight_queue=self._gpu_inflight_queue(prefix, worker_id),
                                source_queue=queue_key,
                            ):
                                continue
                        else:
                            started_at = datetime.now().isoformat()
                            await self.redis.hset(
                                task_key,
                                mapping={"status": TaskStatus.PROCESSING.value, "started_at": started_at},
                            )
                        return task_json
                finally:
                    for deferred_task_id in deferred_claims:
                        await self._return_claim_to_source(deferred_task_id)
                    for deferred_task_id in deferred_task_ids:
                        await self.redis.lpush(queue_key, deferred_task_id)
        return None

    async def _finalize_task_records(
        self,
        *,
        task_id: str,
        task_key: str,
        result_key: str,
        task_mapping: Dict[str, Any],
        result_mapping: Dict[str, Any],
        claim: Optional[TaskClaim],
        allow_control_plane_claim: bool = False,
        allow_frozen_claim: bool = False,
    ) -> None:
        """Atomically publish terminal records and release exactly one attempt."""

        if claim is None:
            existing_task = await self.redis.hgetall(task_key)
            decoded = self._decode_redis_hash(existing_task)
            current_token = str(decoded.get("claim_token") or "")
            if current_token:
                if not allow_control_plane_claim:
                    raise StaleTaskClaimError(
                        f"Task {task_id} is owned by a claim token unavailable to this completion"
                    )
                inflight_queue = str(decoded.get("claim_inflight_queue") or "")
                if not inflight_queue:
                    raise StaleTaskClaimError(f"Task {task_id} has a token but no inflight queue")
                prefix = task_key.rsplit(":task:", 1)[0]
                claim = TaskClaim(
                    prefix=prefix,
                    inflight_queue=inflight_queue,
                    source_queue=str(decoded.get("claim_source_queue") or ""),
                    token=current_token,
                    entry=f"{current_token}|{task_id}",
                    worker_instance=str(decoded.get("claim_worker_instance") or ""),
                )

        task_mapping = {
            **task_mapping,
            "claim_token": "",
            "claim_worker": "",
            "claim_worker_instance": "",
            "claim_source_queue": "",
            "claim_inflight_queue": "",
            "claim_recovery_state": "",
            "claim_recovery_reason": "",
            "claim_recovery_at": "",
        }
        task_ttl = int(getattr(settings, "terminal_task_ttl_sec", 0) or 0)
        result_ttl = int(getattr(settings, "terminal_result_ttl_sec", 0) or 0)

        if claim is not None:
            finalized = await self.redis.eval(
                _FINALIZE_CLAIM_LUA,
                3,
                task_key,
                result_key,
                claim.inflight_queue,
                claim.token,
                claim.entry,
                json.dumps(task_mapping),
                json.dumps(result_mapping),
                task_ttl,
                result_ttl,
                1 if allow_frozen_claim else 0,
            )
            if int(finalized) == -2:
                raise FrozenTaskClaimError(f"Task {task_id} terminal commit is owned by containment")
            if int(finalized) != 1:
                raise StaleTaskClaimError(f"Task {task_id} claim was superseded before terminal commit")
        else:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.hset(task_key, mapping=task_mapping)
                pipe.hset(result_key, mapping=result_mapping)
                if task_ttl > 0:
                    pipe.expire(task_key, task_ttl)
                if result_ttl > 0:
                    pipe.expire(result_key, result_ttl)
                await pipe.execute()
        self._task_claims.pop(task_id, None)

    async def complete_task(self, task_id: str, result: Dict[str, Any], request_hash: Optional[str] = None):
        timing_start = time.time()
        timing_start_mono_ns = time.monotonic_ns()
        completed_at = datetime.now().isoformat()
        metadata = result.setdefault("metadata", {})
        metadata["tm_enter_monotonic_ns"] = timing_start_mono_ns
        task_status = task_status_from_result_payload(result)
        result["status"] = task_status.value

        status_mapping = {"status": task_status.value, "completed_at": completed_at}
        if task_status == TaskStatus.FAILED:
            status_mapping["failed_at"] = completed_at
        elif task_status == TaskStatus.TIMEOUT:
            status_mapping["timeout_at"] = completed_at

        claim = self._task_claims.get(task_id)
        terminal_prefix = claim.prefix if claim is not None else self.key_prefix
        task_key = f"{terminal_prefix}:task:{task_id}"
        result_key = f"{terminal_prefix}:result:{task_id}"

        json_start = time.time()
        payload = json.dumps(result)
        json_dumps_s = time.time() - json_start

        result_mapping: Dict[str, Any] = {"result": payload, "completed_at": completed_at}
        if request_hash:
            result_mapping["request_hash"] = request_hash

        # Publish the terminal task state, result payload, TTLs, and inflight
        # acknowledgement in one Redis transaction.  If EXEC fails, the claim
        # remains recoverable and no partial terminal state can make recovery
        # silently discard a result-less task.
        transaction_start = time.time()
        await self._finalize_task_records(
            task_id=task_id,
            task_key=task_key,
            result_key=result_key,
            task_mapping=status_mapping,
            result_mapping=result_mapping,
            claim=claim,
        )
        terminal_transaction_s = time.time() - transaction_start
        status_hset_s = terminal_transaction_s
        result_hset_s = terminal_transaction_s
        metadata["tm_status_hset_s"] = status_hset_s

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
        *,
        adopt_current_claim: bool = False,
        allow_frozen_claim: bool = False,
    ):
        failed_at = datetime.now().isoformat()
        result_prefix = f"{prefix}:result:" if prefix else self.result_prefix
        task_prefix = f"{prefix}:task:" if prefix else self.task_prefix
        claim = self._task_claims.get(task_id)
        if prefix is None and claim is not None:
            claim_prefix = claim.prefix
            result_prefix = f"{claim_prefix}:result:"
            task_prefix = f"{claim_prefix}:task:"
        task_status = task_status_from_result_payload({"status": "failed", "error_code": error_code})
        timing_key = "timeout_at" if task_status == TaskStatus.TIMEOUT else "failed_at"
        task_key = f"{task_prefix}{task_id}"
        result_key = f"{result_prefix}{task_id}"
        await self._finalize_task_records(
            task_id=task_id,
            task_key=task_key,
            result_key=result_key,
            task_mapping={"status": task_status.value, timing_key: failed_at},
            result_mapping={
                "error": error_message,
                timing_key: failed_at,
                "error_code": error_code.value,
            },
            claim=claim,
            allow_control_plane_claim=adopt_current_claim,
            allow_frozen_claim=allow_frozen_claim,
        )
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
            await self.fail_task(
                task_id,
                "Task cancelled",
                ErrorCode.SYSTEM_ERROR,
                prefix=prefix,
                adopt_current_claim=True,
            )
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
        admission_defaults = {}
        if device.startswith("cuda:"):
            admission_defaults = {
                "health_state": "initializing",
                "accepting_tasks": "false",
            }
        await self.redis.hset(
            f"{self.worker_prefix}{worker_id}",
            mapping={
                "device": device,
                "status": "online",
                "last_heartbeat": now,
                "node_id": node_id or "",
                "hostname": hostname or "",
                **admission_defaults,
            },
        )
        await self.redis.sadd(self.worker_index_key, worker_id)
        self.worker_registry[worker_id] = {
            "device": device,
            "status": "online",
            "last_heartbeat": now,
            "node_id": node_id or "",
            "hostname": hostname or "",
            **admission_defaults,
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
            if resource == "gpu":
                if not await self._gpu_worker_admission_open(worker_id, info):
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
