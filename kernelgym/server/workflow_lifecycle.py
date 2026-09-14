"""Redis-owned parent workflow lifecycle and child admission fences."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from kernelgym.config import settings
from kernelgym.utils.gpu_quarantine import read_gpu_quarantine
from kernelgym.utils.task_status import task_status_from_result_payload


class WorkflowConflictError(RuntimeError):
    pass


class WorkflowStoppedError(RuntimeError):
    pass


DISCARD_WORKFLOW_RECORDS_LUA = r"""
if redis.call('HGET', KEYS[1], 'workflow_generation') ~= ARGV[1] then return 0 end
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'completed' and status ~= 'failed' and status ~= 'timeout' then return 0 end
if (redis.call('HGET', KEYS[1], 'claim_token') or '') ~= '' then return 0 end
return redis.call('DEL', unpack(KEYS))
"""


# Shared by submit and CPU/GPU dispatch transactions. Legacy standalone tasks
# have no workflow generation and retain their existing scheduling semantics.
WORKFLOW_GUARD_LUA = r"""
local function workflow_allowed(mapping)
    for _, key in ipairs(cjson.decode(mapping.cancellation_keys or '[]')) do
        if redis.call('EXISTS', key) == 1 then return false end
    end
    local generation = mapping.workflow_generation
    if not generation or generation == '' then return true end
    local parent = mapping.workflow_parent_key
    local lease = mapping.workflow_lease_key
    if not parent or not lease then return false end
    local cancelled = redis.call('HGET', parent, 'cancellation_key')
    if cancelled and redis.call('EXISTS', cancelled) == 1 then return false end
    if redis.call('HGET', parent, 'workflow_generation') ~= generation then return false end
    local status = redis.call('HGET', parent, 'status')
    if status ~= 'pending' and status ~= 'processing' then return false end
    if redis.call('GET', lease) ~= generation then return false end
    local clock = redis.call('TIME')
    local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
    local deadline = tonumber(redis.call('HGET', parent, 'workflow_deadline')) or 0
    return now < deadline and now < (tonumber(mapping.workflow_deadline) or 0)
end
local function workflow_task_allowed(task_key)
    return workflow_allowed({
        cancellation_keys = redis.call('HGET', task_key, 'cancellation_keys') or '[]',
        workflow_generation = redis.call('HGET', task_key, 'workflow_generation'),
        workflow_parent_key = redis.call('HGET', task_key, 'workflow_parent_key'),
        workflow_lease_key = redis.call('HGET', task_key, 'workflow_lease_key'),
        workflow_deadline = redis.call('HGET', task_key, 'workflow_deadline')
    })
end
"""

WORKFLOW_TASK_ALIVE_LUA = (
    WORKFLOW_GUARD_LUA
    + r"""
if redis.call('HGET', KEYS[1], 'workflow_parent_key') then
    return workflow_task_allowed(KEYS[1]) and 1 or 0
end
return workflow_allowed({workflow_generation=redis.call('HGET', KEYS[1], 'workflow_generation'),
    workflow_parent_key=KEYS[1], workflow_lease_key=KEYS[2],
    workflow_deadline=redis.call('HGET', KEYS[1], 'workflow_deadline')}) and 1 or 0
"""
)

_ACCEPT = r"""
-- KEYS: parent task, result, lease, cancel, active index, durable tombstone
-- ARGV: id, generation, hash, name, timeout, lease_ms, force, submitted_at, children JSON
-- Do not overwrite an old-version active marker during a rolling deployment.
if redis.call('EXISTS', KEYS[6]) == 1 then return -2 end
if redis.call('EXISTS', KEYS[1]) == 0 and redis.call('EXISTS', KEYS[3]) == 1 then return -1 end
if redis.call('EXISTS', KEYS[1]) == 1 then
    local status = redis.call('HGET', KEYS[1], 'status')
    local same = redis.call('HGET', KEYS[1], 'request_hash') == ARGV[3]
    if status == 'pending' or status == 'processing' then
        if not redis.call('HGET', KEYS[1], 'workflow_generation') then return -1 end
        if not same then return -1 end
        return 0
    end
    if status ~= 'completed' and status ~= 'failed' and status ~= 'timeout' then return -1 end
    if (redis.call('HGET', KEYS[1], 'claim_token') or '') ~= '' then return -1 end
    if ARGV[7] ~= '1' then
        if same and redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
        return -1
    end
end
local clock = redis.call('TIME')
local deadline = tonumber(clock[1]) + tonumber(clock[2]) / 1000000 + tonumber(ARGV[5])
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3], KEYS[4])
redis.call('HSET', KEYS[1], 'status', 'pending', 'task_id', ARGV[1],
    'workflow_generation', ARGV[2], 'request_hash', ARGV[3], 'workflow', ARGV[4],
    'workflow_deadline', tostring(deadline), 'submitted_at', ARGV[8], 'children', ARGV[9],
    'cancellation_key', KEYS[6])
redis.call('SET', KEYS[3], ARGV[2], 'PX', ARGV[6])
redis.call('SADD', KEYS[5], ARGV[1])
return 1
"""

CANCEL_TASK_LUA = r"""
-- kernelgym:cancel-task-v2
-- KEYS: task, result, durable tombstone, worker cancel marker
-- ARGV: id, timestamp, prefix, task TTL, result TTL
-- Persist even absent IDs: a DELETE arriving before POST is an admission fence.
redis.call('SET', KEYS[3], ARGV[2])
redis.call('SET', KEYS[4], '1')
for _, stage in ipairs({'compile', 'kernel', 'ref'}) do
    redis.call('SET', ARGV[3] .. ':cancelled:' .. ARGV[1] .. '_' .. stage, ARGV[2])
end
if redis.call('EXISTS', KEYS[1]) == 0 then return 1 end
redis.call('HSET', KEYS[1], 'cancelled_at', ARGV[2])
local children = redis.call('HGET', KEYS[1], 'children')
if children then
    for stage, child in pairs(cjson.decode(children)) do
        redis.call('SET', ARGV[3] .. ':cancelled:' .. child, ARGV[2])
        redis.call('SET', ARGV[3] .. ':cancel:' .. child, '1')
        -- Also fence pre-generation child IDs from delayed/older clients.
        redis.call('SET', ARGV[3] .. ':cancelled:' .. ARGV[1] .. '_' .. stage, ARGV[2])
    end
    return 1 -- Parent terminalization uses its generation-fenced finish path.
end
redis.call('LREM', ARGV[3] .. ':queue:resource:cpu', 0, ARGV[1])
redis.call('LREM', ARGV[3] .. ':queue:resource:gpu', 0, ARGV[1])
local worker = redis.call('HGET', KEYS[1], 'assigned_worker') or ''
if worker ~= '' then redis.call('LREM', ARGV[3] .. ':queue:worker:' .. worker, 0, ARGV[1]) end
-- Never finalize a running/frozen/execution-fenced attempt from the control plane.
if redis.call('HGET', KEYS[1], 'status') ~= 'pending' then return 1 end
if (redis.call('HGET', KEYS[1], 'claim_recovery_state') or '') ~= '' then return 1 end
local token = redis.call('HGET', KEYS[1], 'claim_token') or ''
local inflight = redis.call('HGET', KEYS[1], 'claim_inflight_queue') or ''
if token ~= '' and inflight ~= '' then redis.call('LREM', inflight, 1, token .. '|' .. ARGV[1]) end
redis.call('HSET', KEYS[1], 'status', 'failed', 'completed_at', ARGV[2],
    'claim_token', '', 'claim_worker', '', 'claim_worker_instance', '',
    'claim_source_queue', '', 'claim_inflight_queue', '')
redis.call('HSET', KEYS[2], 'result', cjson.encode({task_id=ARGV[1], status='failed',
    error_message='Task cancelled', error_code='SYSTEM_ERROR'}), 'error', 'Task cancelled',
    'error_code', 'SYSTEM_ERROR', 'completed_at', ARGV[2])
if tonumber(ARGV[4]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[4]) end
if tonumber(ARGV[5]) > 0 then redis.call('EXPIRE', KEYS[2], ARGV[5]) end
return 1
"""

_RENEW = (
    WORKFLOW_GUARD_LUA
    + r"""
if not workflow_allowed({workflow_generation = ARGV[1], workflow_parent_key = KEYS[1],
    workflow_lease_key = KEYS[2], workflow_deadline = redis.call('HGET', KEYS[1], 'workflow_deadline')}) then return 0 end
if redis.call('HGET', KEYS[1], 'workflow_generation') ~= ARGV[1] then return 0 end
redis.call('PEXPIRE', KEYS[2], ARGV[2])
redis.call('HSET', KEYS[1], 'status', 'processing', 'heartbeat_at', ARGV[3])
redis.call('HSETNX', KEYS[1], 'started_at', ARGV[3])
return 1
"""
)

_FINISH = r"""
-- KEYS: task, result, lease, cancel, active index, generation children set
-- ARGV: generation, result JSON, timestamp, task TTL, result TTL, id, reason, prefix
if redis.call('HGET', KEYS[1], 'workflow_generation') ~= ARGV[1] then return 0 end
local current = redis.call('HGET', KEYS[1], 'status')
if current ~= 'pending' and current ~= 'processing' then return 0 end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local deadline = tonumber(redis.call('HGET', KEYS[1], 'workflow_deadline')) or 0
local result = cjson.decode(ARGV[2])
local reason = ARGV[7]
local tombstone = redis.call('HGET', KEYS[1], 'cancellation_key')
if tombstone and redis.call('EXISTS', tombstone) == 1 then
    reason = 'Task cancelled'
    result.status = 'failed'
    result.error_code = 'SYSTEM_ERROR'
end
if now >= deadline then
    reason = 'Workflow end-to-end deadline exceeded'
    result.status = 'timeout'
    result.error_code = 'TIMEOUT_ERROR'
elseif redis.call('GET', KEYS[3]) ~= ARGV[1] and reason == '' then
    reason = 'Workflow owner lease expired'
    result.status = 'failed'
    result.error_code = 'SYSTEM_ERROR'
end
if reason ~= '' then
    result.error_message = reason
    result.compiled = false
    result.correctness = false
end
result.task_id = ARGV[6]
result.workflow_generation = ARGV[1]
result.workflow_children = cjson.decode(redis.call('HGET', KEYS[1], 'children') or '{}')
redis.call('HSET', KEYS[1], 'status', result.status, 'completed_at', ARGV[3],
    'error_message', type(result.error_message) == 'string' and result.error_message or '')
redis.call('HSET', KEYS[2], 'result', cjson.encode(result), 'completed_at', ARGV[3],
    'request_hash', redis.call('HGET', KEYS[1], 'request_hash'))
redis.call('DEL', KEYS[3])
redis.call('SET', KEYS[4], '1')
redis.call('SREM', KEYS[5], ARGV[6])
for _, child_id in ipairs(redis.call('SMEMBERS', KEYS[6])) do
    local child = ARGV[8] .. ':task:' .. child_id
    local worker = redis.call('HGET', child, 'assigned_worker') or ''
    redis.call('LREM', ARGV[8] .. ':queue:resource:cpu', 0, child_id)
    redis.call('LREM', ARGV[8] .. ':queue:resource:gpu', 0, child_id)
    if worker ~= '' then redis.call('LREM', ARGV[8] .. ':queue:worker:' .. worker, 0, child_id) end
    -- Running/frozen claims remain owned by the worker's safe-reap path.
    -- A pending, unfenced claim has not passed the atomic execution gate;
    -- cancelling it here also fences a worker that dequeued just before us.
    local token = redis.call('HGET', child, 'claim_token') or ''
    local fence = redis.call('HGET', child, 'claim_recovery_state') or ''
    if redis.call('HGET', child, 'status') == 'pending' and fence == '' then
        local child_result = ARGV[8] .. ':result:' .. child_id
        local inflight = redis.call('HGET', child, 'claim_inflight_queue') or ''
        if token ~= '' and inflight ~= '' then redis.call('LREM', inflight, 1, token .. '|' .. child_id) end
        redis.call('HSET', child, 'claim_token', '', 'claim_worker', '', 'claim_worker_instance', '',
            'claim_source_queue', '', 'claim_inflight_queue', '')
        redis.call('HSET', child, 'status', 'failed', 'completed_at', ARGV[3])
        redis.call('HSET', child_result, 'result', cjson.encode({task_id=child_id,
            status='failed', error_message='Parent workflow ended', error_code='SYSTEM_ERROR'}),
            'completed_at', ARGV[3])
        if tonumber(ARGV[4]) > 0 then redis.call('EXPIRE', child, ARGV[4]) end
        if tonumber(ARGV[5]) > 0 then redis.call('EXPIRE', child_result, ARGV[5]) end
    end
end
if tonumber(ARGV[4]) > 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[4])
    redis.call('EXPIRE', KEYS[4], ARGV[4])
    redis.call('EXPIRE', KEYS[6], ARGV[4])
end
if tonumber(ARGV[5]) > 0 then redis.call('EXPIRE', KEYS[2], ARGV[5]) end
return 1
"""


class WorkflowLifecycle:
    def __init__(self, task_manager: Any):
        self.manager = task_manager
        self.redis = task_manager.redis
        self.prefix = task_manager.key_prefix
        self.active_key = f"{self.prefix}:workflows:active"

    def keys(self, task_id: str) -> list[str]:
        return [
            f"{self.prefix}:task:{task_id}",
            f"{self.prefix}:result:{task_id}",
            f"{self.prefix}:workflow:{task_id}",
            f"{self.prefix}:cancel:{task_id}",
            self.active_key,
        ]

    def children_key(self, task_id: str, generation: str) -> str:
        return f"{self.prefix}:workflow_children:{task_id}:{generation}"

    async def read(self, task_id: str) -> dict[str, str]:
        return self.manager._decode_redis_hash(await self.redis.hgetall(self.keys(task_id)[0]))

    async def accept(self, task_id: str, request_hash: str, name: str, timeout: float, force: bool):
        generation = uuid.uuid4().hex
        children = {stage: f"{task_id}_{stage}_{generation}" for stage in ("compile", "kernel", "ref")}
        result = await self.redis.eval(
            _ACCEPT,
            6,
            *self.keys(task_id),
            self.manager._tombstone_key(task_id),
            task_id,
            generation,
            request_hash,
            name,
            timeout,
            int(settings.workflow_lease_seconds * 1000),
            int(force),
            datetime.now().isoformat(),
            json.dumps(children),
        )
        if int(result) == -2:
            raise WorkflowConflictError(f"Task {task_id} was cancelled; use a new ID")
        if int(result) < 0:
            raise WorkflowConflictError(f"Task {task_id} already exists with incompatible content/state; use a new ID")
        record = await self.read(task_id)
        return int(result) == 1, record

    async def renew(self, task_id: str, generation: str) -> bool:
        keys = self.keys(task_id)
        return bool(
            await self.redis.eval(
                _RENEW,
                2,
                keys[0],
                keys[2],
                generation,
                int(settings.workflow_lease_seconds * 1000),
                datetime.now().isoformat(),
            )
        )

    async def run(self, task_id: str, record: dict[str, str], operation: Any) -> None:
        """Own one controller independently of individual HTTP waiters."""
        generation = record["workflow_generation"]

        async def heartbeat():
            while await self.renew(task_id, generation):
                await asyncio.sleep(min(1.0, settings.workflow_lease_seconds / 3))
            raise WorkflowStoppedError("Workflow lease/deadline is no longer valid")

        controller_task = asyncio.create_task(operation())
        heartbeat_task = asyncio.create_task(heartbeat())
        reason = ""
        cancelled = False
        try:
            clock = await self.redis.time()
            remaining = float(record["workflow_deadline"]) - float(clock[0]) - float(clock[1]) / 1_000_000
            async with asyncio.timeout(max(0, remaining)):
                done, _ = await asyncio.wait({controller_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED)
                if heartbeat_task in done:
                    heartbeat_task.result()
                result = controller_task.result()
                result = dict(result)
                result["task_id"] = task_id
                result["status"] = task_status_from_result_payload(result).value
        except TimeoutError:
            reason = "Workflow end-to-end deadline exceeded"
            result = self.failure(task_id, reason)
            result["status"] = "timeout"
            result["error_code"] = "TIMEOUT_ERROR"
        except asyncio.CancelledError:
            cancelled = True
            reason = "Workflow controller cancelled during shutdown"
            result = self.failure(task_id, reason)
        except Exception as exc:
            reason = f"Workflow controller failed: {type(exc).__name__}: {exc}"
            result = self.failure(task_id, reason)
        finally:
            controller_task.cancel()
            heartbeat_task.cancel()
            done, pending = await asyncio.wait({controller_task, heartbeat_task}, timeout=5)
            for task in done:
                if not task.cancelled():
                    task.exception()
            for task in pending:
                logging.getLogger(__name__).error("Workflow %s coroutine did not stop within cleanup grace", task_id)
                task.add_done_callback(lambda completed: None if completed.cancelled() else completed.exception())
        # CAS refuses late completions after explicit cancellation, refresh,
        # deadline reconciliation, or another terminal writer.
        async with asyncio.timeout(5):
            await self.finish(task_id, generation, result, reason)
        if cancelled:
            raise asyncio.CancelledError

    async def wait(self, task_id: str, generation: str) -> dict[str, Any]:
        while True:
            async with asyncio.timeout(5):
                record = await self.reconcile(task_id)
                if record.get("workflow_generation") != generation:
                    raise WorkflowConflictError(f"Workflow {task_id} was replaced while this request was waiting")
                result = await self.manager.get_task_result(task_id)
            if result:
                if result.get("workflow_generation") != generation:
                    raise WorkflowConflictError(f"Workflow {task_id} result belongs to another generation")
                return result
            if record.get("status") not in {"pending", "processing"}:
                raise WorkflowConflictError(f"Workflow {task_id} finished but its retained result is unavailable")
            await asyncio.sleep(0.2)

    async def finish(self, task_id: str, generation: str, result: dict[str, Any], reason: str = "") -> bool:
        keys = [*self.keys(task_id), self.children_key(task_id, generation)]
        changed = bool(
            await self.redis.eval(
                _FINISH,
                len(keys),
                *keys,
                generation,
                json.dumps(result),
                datetime.now().isoformat(),
                settings.terminal_task_ttl_sec,
                settings.terminal_result_ttl_sec,
                task_id,
                reason,
                self.prefix,
            )
        )
        return changed

    async def reconcile(self, task_id: str) -> dict[str, str]:
        record = await self.read(task_id)
        if record.get("status") not in {"pending", "processing"} or not record.get("workflow_generation"):
            return record
        clock = await self.redis.time()
        now = float(clock[0]) + float(clock[1]) / 1_000_000
        reason = ""
        if now >= float(record["workflow_deadline"]):
            reason = "Workflow end-to-end deadline exceeded"
        elif await self.redis.get(self.keys(task_id)[2]) != record["workflow_generation"].encode():
            reason = "Workflow owner lease expired"
        elif await self.redis.exists(self.manager._tombstone_key(task_id)):
            reason = "Task cancelled"
        else:
            children = set(json.loads(record.get("children", "{}")).values())
            children.update(
                raw.decode() if isinstance(raw, bytes) else str(raw)
                for raw in await self.redis.smembers(self.children_key(task_id, record["workflow_generation"]))
            )
            for child_id in sorted(children):
                child = self.manager._decode_redis_hash(await self.redis.hgetall(f"{self.prefix}:task:{child_id}"))
                cancellation_keys = json.loads(child.get("cancellation_keys", "[]"))
                if await self.redis.exists(self.manager._tombstone_key(child_id), *cancellation_keys):
                    reason = f"Child task {child_id} cancelled"
                    break
                if child.get("claim_recovery_state") == "frozen":
                    reason = (
                        f"Infrastructure failure: child {child_id} frozen: {child.get('claim_recovery_reason', '')}"
                    )
                    break
                if child.get("status") not in {"pending", "processing"}:
                    continue
                worker_id = child.get("claim_worker") or child.get("assigned_worker")
                if not worker_id:
                    continue
                worker = self.manager._decode_redis_hash(await self.redis.hgetall(f"{self.prefix}:worker:{worker_id}"))
                if await read_gpu_quarantine(
                    self.redis,
                    worker_id,
                    device=worker.get("device", ""),
                    hostname=worker.get("hostname", ""),
                    read_only=True,
                ):
                    reason = f"Infrastructure failure: child {child_id} worker {worker_id} quarantined"
                    break
        if reason:
            await self.finish(task_id, record["workflow_generation"], self.failure(task_id, reason), reason)
            record = await self.read(task_id)
        return record

    @staticmethod
    def failure(task_id: str, reason: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "status": "failed",
            "error_message": reason,
            "error_code": "SYSTEM_ERROR",
            "compiled": False,
            "correctness": False,
            "decoy_kernel": False,
            "speedup": 0.0,
            "kernel_runtime": -1.0,
            "reference_runtime": -1.0,
            "metadata": {},
        }

    async def watch(self) -> None:
        while True:
            try:
                for raw_id in await self.redis.smembers(self.active_key):
                    task_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                    async with asyncio.timeout(5):
                        await self.reconcile(task_id)
            except Exception:
                # Keep the index for retry; admission still fails closed
                # against the absolute deadline and the expiring lease.
                logging.getLogger(__name__).exception("Workflow reconciliation failed")
            await asyncio.sleep(1)
