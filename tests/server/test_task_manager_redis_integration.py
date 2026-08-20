"""Server integration coverage for TaskManager's Redis Lua transactions.

Set ``KERNELGYM_TEST_REDIS_URL`` to a disposable standalone Redis instance to
run these tests.  Each test owns a UUID namespace and deletes only keys in that
namespace; it deliberately never calls FLUSHDB.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta

import pytest
import redis.asyncio as redis_async

from kernelgym.common import TaskStatus
from kernelgym.server.task_manager import (
    FrozenTaskClaimError,
    StaleTaskClaimError,
    TaskManager,
    TaskRefreshConflictError,
)


_REDIS_URL = os.getenv("KERNELGYM_TEST_REDIS_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _REDIS_URL,
        reason="set KERNELGYM_TEST_REDIS_URL to a disposable standalone Redis",
    ),
]


def _namespaced_manager(client: redis_async.Redis, prefix: str) -> TaskManager:
    """Bind one manager to the test namespace without changing global settings."""

    manager = TaskManager(client)
    manager.key_prefix = prefix
    manager.legacy_prefix = prefix
    manager.task_prefix = f"{prefix}:task:"
    manager.queue_prefix = f"{prefix}:queue:"
    manager.result_prefix = f"{prefix}:result:"
    manager.worker_prefix = f"{prefix}:worker:"
    manager.worker_index_key = f"{prefix}:workers"
    manager.node_map_key = f"{prefix}:nodes"
    manager.status_prefix = f"{prefix}:status:"
    manager.resource_queues = {
        "cpu": f"{prefix}:queue:resource:cpu",
        "gpu": f"{prefix}:queue:resource:gpu",
    }
    return manager


async def _delete_namespace(client: redis_async.Redis, prefix: str) -> None:
    keys = [key async for key in client.scan_iter(match=f"{prefix}:*", count=100)]
    if keys:
        await client.delete(*keys)


async def _close(client: redis_async.Redis) -> None:
    close = getattr(client, "aclose", None)
    if close is not None:
        await close()
    else:  # pragma: no cover - compatibility with older redis-py
        await client.close()


def test_real_redis_frozen_gpu_claim_recovery_and_completion_fencing() -> None:
    async def scenario() -> None:
        assert _REDIS_URL is not None
        client = redis_async.Redis.from_url(_REDIS_URL, decode_responses=False)
        prefix = f"kernelgym:test:claim-fencing:{uuid.uuid4().hex}"
        task_id = f"task-{uuid.uuid4().hex}"
        worker_id = "gpu-worker"

        try:
            await client.ping()
            first = _namespaced_manager(client, prefix)
            task_key = f"{first.task_prefix}{task_id}"
            result_key = f"{first.result_prefix}{task_id}"
            source_queue = first.resource_queues["gpu"]
            inflight_queue = first._gpu_inflight_queue(prefix, worker_id)
            payload = {"task_id": task_id, "required_resource": "gpu"}
            submitted_at = datetime.now().isoformat()
            await client.hset(
                task_key,
                mapping={
                    "data": json.dumps(payload),
                    "status": TaskStatus.PENDING.value,
                    "submitted_at": submitted_at,
                    "assigned_worker": "",
                    "assigned_at": "",
                },
            )
            await client.lpush(source_queue, task_id)

            assert await first._claim_gpu_task(prefix, worker_id, source_queue) == task_id
            old_claim = first._task_claims[task_id]
            assert await first._mark_claim_processing(
                prefix=prefix,
                task_id=task_id,
                task_key=task_key,
                task_json=payload,
                worker_id=worker_id,
                inflight_queue=inflight_queue,
                source_queue=source_queue,
            )
            assert await client.lrange(inflight_queue, 0, -1) == [old_claim.entry.encode()]
            replacement_mapping = {
                "data": json.dumps({**payload, "generation": "replacement"}),
                "status": TaskStatus.PENDING.value,
                "priority": "normal",
                "submitted_at": datetime.now().isoformat(),
                "assigned_worker": "",
                "assigned_at": "",
            }
            active_snapshot = await client.hgetall(task_key)
            with pytest.raises(TaskRefreshConflictError, match="active claim token"):
                await first._force_refresh_task(
                    task_id,
                    task_mapping=replacement_mapping,
                    destination_queue=source_queue,
                )
            assert await client.hgetall(task_key) == active_snapshot
            assert await client.lrange(inflight_queue, 0, -1) == [old_claim.entry.encode()]

            assert await first.freeze_task_claim(task_id, "unsafe CUDA child still exists") is True
            frozen_snapshot = await client.hgetall(task_key)
            with pytest.raises(FrozenTaskClaimError, match="owned by containment"):
                await first.complete_task(
                    task_id,
                    {"task_id": task_id, "status": "failed", "error_message": "must remain frozen"},
                )
            assert await client.hgetall(task_key) == frozen_snapshot
            assert await client.lrange(inflight_queue, 0, -1) == [old_claim.entry.encode()]
            with pytest.raises(TaskRefreshConflictError, match="recovery is frozen"):
                await first._force_refresh_task(
                    task_id,
                    task_mapping=replacement_mapping,
                    destination_queue=source_queue,
                )
            assert await client.hgetall(task_key) == frozen_snapshot
            assert await client.lrange(inflight_queue, 0, -1) == [old_claim.entry.encode()]

            # A replacement must not expose the old task while containment is
            # still unsafe. Only the explicit safe-start path may release it.
            replacement = _namespaced_manager(client, prefix)
            await replacement.recover_gpu_inflight(worker_id)
            frozen_hash = await client.hgetall(task_key)
            assert frozen_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
            assert frozen_hash[b"claim_token"] == old_claim.token.encode()
            assert frozen_hash[b"claim_recovery_state"] == b"frozen"
            assert await client.lrange(inflight_queue, 0, -1) == [old_claim.entry.encode()]
            assert await client.lrange(source_queue, 0, -1) == []

            # This flag represents GPUWorker's proven-safe startup gate: the
            # quarantine is absent and one fresh CUDA pool initialized healthy.
            await replacement.recover_gpu_inflight(worker_id, release_frozen_claims=True)
            recovered_hash = await client.hgetall(task_key)
            assert recovered_hash[b"status"] == TaskStatus.PENDING.value.encode()
            assert recovered_hash[b"claim_token"] == b""
            assert await client.lrange(inflight_queue, 0, -1) == []
            assert await client.lrange(source_queue, 0, -1) == [task_id.encode()]

            assert await replacement._claim_gpu_task(prefix, worker_id, source_queue) == task_id
            new_claim = replacement._task_claims[task_id]
            assert new_claim.token != old_claim.token
            assert await replacement._mark_claim_processing(
                prefix=prefix,
                task_id=task_id,
                task_key=task_key,
                task_json=payload,
                worker_id=worker_id,
                inflight_queue=inflight_queue,
                source_queue=source_queue,
            )

            with pytest.raises(StaleTaskClaimError, match="superseded"):
                await first.complete_task(
                    task_id,
                    {"task_id": task_id, "status": TaskStatus.COMPLETED.value, "compiled": True},
                )

            fenced_hash = await client.hgetall(task_key)
            assert fenced_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
            assert fenced_hash[b"claim_token"] == new_claim.token.encode()
            assert await client.hgetall(result_key) == {}
            assert await client.lrange(inflight_queue, 0, -1) == [new_claim.entry.encode()]

            await replacement.complete_task(
                task_id,
                {"task_id": task_id, "status": TaskStatus.COMPLETED.value, "compiled": True},
            )

            completed_hash = await client.hgetall(task_key)
            result_hash = await client.hgetall(result_key)
            assert completed_hash[b"status"] == TaskStatus.COMPLETED.value.encode()
            assert completed_hash[b"claim_token"] == b""
            assert b"result" in result_hash
            assert await client.lrange(inflight_queue, 0, -1) == []
        finally:
            await _delete_namespace(client, prefix)
            await _close(client)

    asyncio.run(scenario())


def test_real_redis_safe_recovery_finalizes_cancelled_frozen_claim() -> None:
    async def scenario() -> None:
        assert _REDIS_URL is not None
        client = redis_async.Redis.from_url(_REDIS_URL, decode_responses=False)
        prefix = f"kernelgym:test:frozen-cancel:{uuid.uuid4().hex}"
        task_id = f"task-{uuid.uuid4().hex}"
        worker_id = "gpu-worker"

        try:
            await client.ping()
            first = _namespaced_manager(client, prefix)
            task_key = f"{first.task_prefix}{task_id}"
            result_key = f"{first.result_prefix}{task_id}"
            source_queue = first.resource_queues["gpu"]
            inflight_queue = first._gpu_inflight_queue(prefix, worker_id)
            payload = {"task_id": task_id, "required_resource": "gpu"}
            await client.hset(
                task_key,
                mapping={
                    "data": json.dumps(payload),
                    "status": TaskStatus.PENDING.value,
                    "submitted_at": datetime.now().isoformat(),
                    "assigned_worker": "",
                    "assigned_at": "",
                },
            )
            await client.lpush(source_queue, task_id)

            assert await first._claim_gpu_task(prefix, worker_id, source_queue) == task_id
            old_claim = first._task_claims[task_id]
            assert await first._mark_claim_processing(
                prefix=prefix,
                task_id=task_id,
                task_key=task_key,
                task_json=payload,
                worker_id=worker_id,
                inflight_queue=inflight_queue,
                source_queue=source_queue,
            )
            assert await first.freeze_task_claim(task_id, "unsafe CUDA child still exists") is True
            await client.set(first._cancel_key(task_id, prefix), "1")

            replacement = _namespaced_manager(client, prefix)
            await replacement.recover_gpu_inflight(worker_id)
            still_frozen = await client.hgetall(task_key)
            assert still_frozen[b"status"] == TaskStatus.PROCESSING.value.encode()
            assert still_frozen[b"claim_recovery_state"] == b"frozen"

            await replacement.recover_gpu_inflight(worker_id, release_frozen_claims=True)

            terminal = await client.hgetall(task_key)
            result = await client.hgetall(result_key)
            assert terminal[b"status"] == TaskStatus.FAILED.value.encode()
            assert terminal[b"claim_token"] == b""
            assert terminal[b"claim_recovery_state"] == b""
            assert b"cancelled_at" in terminal
            assert result[b"error"] == b"Task cancelled"
            assert await client.lrange(inflight_queue, 0, -1) == []
            assert await client.lrange(source_queue, 0, -1) == []
            assert old_claim.entry.encode() not in await client.lrange(inflight_queue, 0, -1)
        finally:
            await _delete_namespace(client, prefix)
            await _close(client)

    asyncio.run(scenario())


def test_real_redis_atomic_submit_and_repeatable_recovery() -> None:
    async def scenario() -> None:
        assert _REDIS_URL is not None
        client = redis_async.Redis.from_url(_REDIS_URL, decode_responses=False)
        prefix = f"kernelgym:test:atomic-submit:{uuid.uuid4().hex}"
        worker_id = "gpu-worker"

        try:
            await client.ping()
            first = _namespaced_manager(client, prefix)
            second = _namespaced_manager(client, prefix)

            concurrent_id = f"submit-{uuid.uuid4().hex}"
            destination = f"{prefix}:queue:test:atomic-submit"
            mappings = [
                {
                    "data": json.dumps({"task_id": concurrent_id, "generation": generation}),
                    "status": TaskStatus.PENDING.value,
                    "priority": "normal",
                    "submitted_at": datetime.now().isoformat(),
                    "assigned_worker": "",
                    "assigned_at": "",
                }
                for generation in ("first", "second")
            ]
            created = await asyncio.gather(
                first._submit_task_if_absent(
                    concurrent_id,
                    task_mapping=mappings[0],
                    destination_queue=destination,
                ),
                second._submit_task_if_absent(
                    concurrent_id,
                    task_mapping=mappings[1],
                    destination_queue=destination,
                ),
            )
            assert sorted(created) == [False, True]
            assert await client.lrange(destination, 0, -1) == [concurrent_id.encode()]
            stored = await client.hgetall(f"{first.task_prefix}{concurrent_id}")
            assert json.loads(stored[b"data"].decode())["generation"] in {"first", "second"}

            race_id = f"normal-force-{uuid.uuid4().hex}"
            normal_mapping = {
                **mappings[0],
                "data": json.dumps({"task_id": race_id, "generation": "normal"}),
            }
            refresh_mapping = {
                **mappings[1],
                "data": json.dumps({"task_id": race_id, "generation": "refresh"}),
            }
            outcomes = await asyncio.gather(
                first._submit_task_if_absent(
                    race_id,
                    task_mapping=normal_mapping,
                    destination_queue=destination,
                ),
                second._force_refresh_task(
                    race_id,
                    task_mapping=refresh_mapping,
                    destination_queue=destination,
                ),
                return_exceptions=True,
            )
            assert isinstance(outcomes[0], bool)
            assert isinstance(outcomes[1], (int, TaskRefreshConflictError))
            assert (await client.lrange(destination, 0, -1)).count(race_id.encode()) == 1
            race_hash = await client.hgetall(f"{first.task_prefix}{race_id}")
            assert json.loads(race_hash[b"data"].decode())["generation"] in {"normal", "refresh"}

            # A stale claim can become visible after an earlier empty scan. The
            # replacement must scan again rather than caching the empty result.
            late_id = f"late-claim-{uuid.uuid4().hex}"
            late_task_key = f"{first.task_prefix}{late_id}"
            late_destination = f"{prefix}:queue:test:late-claim"
            late_payload = {"task_id": late_id, "required_resource": "gpu"}
            await second.recover_gpu_inflight(worker_id)
            await client.hset(
                late_task_key,
                mapping={
                    "data": json.dumps(late_payload),
                    "status": TaskStatus.PENDING.value,
                    "submitted_at": datetime.now().isoformat(),
                    "assigned_worker": "",
                    "assigned_at": "",
                },
            )
            await client.lpush(late_destination, late_id)
            assert await first._claim_gpu_task(prefix, worker_id, late_destination) == late_id
            old_claim = first._task_claims[late_id]

            await second.recover_gpu_inflight(worker_id)

            recovered = await client.hgetall(late_task_key)
            assert recovered[b"status"] == TaskStatus.PENDING.value.encode()
            assert recovered[b"claim_token"] == b""
            assert await client.lrange(old_claim.inflight_queue, 0, -1) == []
            assert await client.lrange(late_destination, 0, -1) == [late_id.encode()]
        finally:
            await _delete_namespace(client, prefix)
            await _close(client)

    asyncio.run(scenario())


def test_real_redis_direct_queue_wait_compare_and_swap() -> None:
    async def scenario() -> None:
        assert _REDIS_URL is not None
        client = redis_async.Redis.from_url(_REDIS_URL, decode_responses=False)
        prefix = f"kernelgym:test:queue-wait-cas:{uuid.uuid4().hex}"
        worker_id = "gpu-worker"

        try:
            await client.ping()
            manager = _namespaced_manager(client, prefix)
            worker_queue = f"{prefix}:queue:worker:{worker_id}"
            destination = manager.resource_queues["gpu"]

            task_id = f"move-{uuid.uuid4().hex}"
            task_key = f"{manager.task_prefix}{task_id}"
            assigned_at = (datetime.now() - timedelta(minutes=5)).isoformat()
            payload = {
                "task_id": task_id,
                "assigned_worker": worker_id,
                "required_resource": "gpu",
            }
            await client.hset(
                task_key,
                mapping={
                    "data": json.dumps(payload),
                    "status": TaskStatus.PENDING.value,
                    "submitted_at": assigned_at,
                    "assigned_worker": worker_id,
                    "assigned_at": assigned_at,
                },
            )
            await client.lpush(worker_queue, task_id)
            snapshot = await client.hgetall(task_key)

            moved = await manager._conditionally_requeue_waiting_task(
                prefix=prefix,
                worker_queue_key=worker_queue,
                task_id=task_id,
                task_data=snapshot,
                task_json=payload,
                reason="queue_wait_timeout",
                now_iso=datetime.now().isoformat(),
                operation="requeue",
            )
            assert moved == 1
            assert await client.lrange(worker_queue, 0, -1) == []
            assert await client.lrange(destination, 0, -1) == [task_id.encode()]
            moved_hash = await client.hgetall(task_key)
            assert moved_hash[b"assigned_worker"] == b""
            assert moved_hash[b"queue_timeout_reason"] == b"queue_wait_timeout"

            # The monitor's LRANGE snapshot is stale after this field changes.
            # The Lua CAS must leave both the task and source queue untouched.
            race_id = f"race-{uuid.uuid4().hex}"
            race_key = f"{manager.task_prefix}{race_id}"
            race_payload = {
                "task_id": race_id,
                "assigned_worker": worker_id,
                "required_resource": "gpu",
            }
            await client.hset(
                race_key,
                mapping={
                    "data": json.dumps(race_payload),
                    "status": TaskStatus.PENDING.value,
                    "submitted_at": assigned_at,
                    "assigned_worker": worker_id,
                    "assigned_at": assigned_at,
                },
            )
            await client.lpush(worker_queue, race_id)
            stale_snapshot = await client.hgetall(race_key)
            replacement_assigned_at = datetime.now().isoformat()
            await client.hset(race_key, mapping={"assigned_at": replacement_assigned_at})

            rejected = await manager._conditionally_requeue_waiting_task(
                prefix=prefix,
                worker_queue_key=worker_queue,
                task_id=race_id,
                task_data=stale_snapshot,
                task_json=race_payload,
                reason="queue_wait_timeout",
                now_iso=datetime.now().isoformat(),
                operation="requeue",
            )
            assert rejected == -4
            assert await client.lrange(worker_queue, 0, -1) == [race_id.encode()]
            assert race_id.encode() not in await client.lrange(destination, 0, -1)
            race_hash = await client.hgetall(race_key)
            assert race_hash[b"assigned_at"] == replacement_assigned_at.encode()
            assert race_hash[b"assigned_worker"] == worker_id.encode()
        finally:
            await _delete_namespace(client, prefix)
            await _close(client)

    asyncio.run(scenario())
