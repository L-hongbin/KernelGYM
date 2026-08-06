from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import timedelta
from typing import Any

import pytest

from kernelgym.server import task_manager as task_manager_module
from kernelgym.server.task_manager import (
    FrozenTaskClaimError,
    StaleTaskClaimError,
    TaskManager,
    TaskRefreshConflictError,
)
from kernelgym.server.request_hash import request_hash
from kernelgym.common import TaskStatus
from kernelgym.schema.task import EvaluationTask
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks


@pytest.fixture(autouse=True)
def _isolated_latch_dir(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """Keep Redis-only quarantine migration out of the shared runtime latch tree."""

    monkeypatch.setenv("KERNELGYM_SAFETY_LATCH_DIR", str(tmp_path / "safety_latches"))


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.lists: dict[str, list[str]] = defaultdict(list)
        self.sets: dict[str, set[str]] = defaultdict(set)
        self.strings: dict[str, bytes] = {}
        self.counters: dict[str, int] = defaultdict(int)
        self.expirations: dict[str, int] = {}

    @staticmethod
    def _bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        return str(value).encode()

    async def exists(self, *keys: str) -> int:
        return sum(key in self.hashes or key in self.strings or key in self.sets for key in keys)

    async def hset(self, key: str, mapping: dict[str, Any]) -> None:
        target = self.hashes.setdefault(key, {})
        for field, value in mapping.items():
            target[self._bytes(field)] = self._bytes(value)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return dict(self.hashes.get(key, {}))

    async def lpush(self, key: str, value: str) -> None:
        self.lists[key].insert(0, value)

    async def rpush(self, key: str, value: str) -> None:
        self.lists[key].append(value)

    async def rpop(self, key: str) -> bytes | None:
        if not self.lists[key]:
            return None
        return self._bytes(self.lists[key].pop())

    async def rpoplpush(self, source: str, destination: str) -> bytes | None:
        value = await self.rpop(source)
        if value is None:
            return None
        text = value.decode() if isinstance(value, bytes) else str(value)
        await self.lpush(destination, text)
        return value

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        values = self.lists[key]
        if end == -1:
            selected = values[start:]
        else:
            selected = values[start : end + 1]
        return [self._bytes(value) for value in selected]

    async def llen(self, key: str) -> int:
        return len(self.lists[key])

    async def lrem(self, key: str, count: int, value: str) -> int:
        removed = 0
        values = self.lists[key]
        kept = []
        for item in values:
            if item == value and (count == 0 or removed < abs(count)):
                removed += 1
                continue
            kept.append(item)
        self.lists[key] = kept
        return removed

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.hashes:
                removed += 1
                del self.hashes[key]
            if key in self.strings:
                removed += 1
                del self.strings[key]
            if key in self.sets:
                removed += 1
                del self.sets[key]
            self.expirations.pop(key, None)
        return removed

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        self.strings[key] = self._bytes(value)
        if ex is not None:
            self.expirations[key] = ex

    async def expire(self, key: str, ttl: int) -> bool:
        if key not in self.hashes and key not in self.strings and key not in self.sets:
            return False
        self.expirations[key] = ttl
        return True

    async def sadd(self, key: str, *values: Any) -> int:
        before = len(self.sets[key])
        self.sets[key].update(str(value) for value in values)
        return len(self.sets[key]) - before

    async def smembers(self, key: str) -> set[bytes]:
        return {self._bytes(value) for value in self.sets.get(key, set())}

    async def srem(self, key: str, *values: Any) -> int:
        removed = 0
        target = self.sets[key]
        for value in values:
            text = str(value)
            if text in target:
                target.remove(text)
                removed += 1
        return removed

    async def scan_iter(self, pattern: str, count: int | None = None):
        import fnmatch

        keys = set(self.hashes) | set(self.lists) | set(self.sets) | set(self.strings)
        for key in sorted(keys):
            if fnmatch.fnmatch(key, pattern):
                yield key

    async def incr(self, key: str) -> int:
        self.counters[key] += 1
        return self.counters[key]

    async def eval(self, script: str, numkeys: int, *values: Any) -> Any:
        keys = [str(value) for value in values[:numkeys]]
        argv = [str(value) for value in values[numkeys:]]

        def hash_field(task_hash: dict[bytes, bytes] | None, field: str) -> str:
            if not task_hash:
                return ""
            value = task_hash.get(field.encode(), b"")
            return value.decode() if isinstance(value, bytes) else str(value)

        async def clear_claim(task_key: str) -> None:
            await self.hset(
                task_key,
                mapping={
                    "claim_token": "",
                    "claim_worker": "",
                    "claim_worker_instance": "",
                    "claim_source_queue": "",
                    "claim_inflight_queue": "",
                    "claim_recovery_state": "",
                    "claim_recovery_reason": "",
                    "claim_recovery_at": "",
                },
            )

        if "kernelgym:submit-task-if-absent-v1" in script:
            task_key, destination = keys
            task_id, task_mapping_json = argv
            if task_key in self.hashes:
                return 0
            await self.hset(task_key, mapping=json.loads(task_mapping_json))
            await self.lrem(destination, 0, task_id)
            await self.lpush(destination, task_id)
            return 1

        if "kernelgym:force-refresh-task-v1" in script:
            task_key, result_key, destination, *cleanup_queues = keys
            task_id, replacement_mapping_json, worker_queue_prefix = argv
            task_hash = self.hashes.get(task_key)
            task_exists = bool(task_hash)
            if task_exists:
                if hash_field(task_hash, "claim_recovery_state") == "frozen":
                    return -2
                if hash_field(task_hash, "claim_token"):
                    return -1
                if hash_field(task_hash, "status") not in {
                    TaskStatus.COMPLETED.value,
                    TaskStatus.FAILED.value,
                    TaskStatus.TIMEOUT.value,
                }:
                    return -4
                old_assigned_worker = hash_field(task_hash, "assigned_worker")
                if old_assigned_worker:
                    await self.lrem(f"{worker_queue_prefix}{old_assigned_worker}", 0, task_id)
            for queue_key in [destination, *cleanup_queues]:
                await self.lrem(queue_key, 0, task_id)
            await self.delete(task_key, result_key)
            await self.hset(task_key, mapping=json.loads(replacement_mapping_json))
            await self.lpush(destination, task_id)
            return 2 if task_exists else 1

        if "kernelgym:claim-gpu-task-v2" in script:
            source_queue, inflight_queue = keys
            token, worker_id, worker_instance, task_prefix, scan_limit = argv
            for _ in range(int(scan_limit)):
                raw_task_id = await self.rpop(source_queue)
                if raw_task_id is None:
                    return None
                task_id = raw_task_id.decode() if isinstance(raw_task_id, bytes) else str(raw_task_id)
                task_key = f"{task_prefix}{task_id}"
                task_hash = self.hashes.get(task_key)
                if (
                    task_hash
                    and hash_field(task_hash, "status") == TaskStatus.PENDING.value
                    and not hash_field(task_hash, "claim_token")
                ):
                    entry = f"{token}|{task_id}"
                    await self.lpush(inflight_queue, entry)
                    await self.hset(
                        task_key,
                        mapping={
                            "claim_token": token,
                            "claim_worker": worker_id,
                            "claim_worker_instance": worker_instance,
                            "claim_source_queue": source_queue,
                            "claim_inflight_queue": inflight_queue,
                            "claim_recovery_state": "",
                            "claim_recovery_reason": "",
                            "claim_recovery_at": "",
                        },
                    )
                    return task_id
            return None

        if "kernelgym:queue-wait-requeue-v2" in script:
            task_key, source_queue, destination, own_cancel, parent_cancel = keys
            (
                task_id,
                source_worker,
                operation,
                expected_worker,
                expected_assigned_at,
                expected_submitted_at,
                expected_payload,
                replacement_payload,
                reason,
                timestamp,
                move_token,
            ) = argv
            task_hash = self.hashes.get(task_key)
            if hash_field(task_hash, "assigned_worker") != expected_worker:
                return -3
            if hash_field(task_hash, "assigned_at") != expected_assigned_at:
                return -4
            if hash_field(task_hash, "submitted_at") != expected_submitted_at:
                return -5
            if hash_field(task_hash, "data") != expected_payload:
                return -6
            if operation == "remove_stale_copy":
                if not expected_worker or expected_worker == source_worker:
                    return -8
                return 2 if await self.lrem(source_queue, 1, task_id) == 1 else -7
            if operation != "requeue" or (expected_worker and expected_worker != source_worker):
                return -8
            if hash_field(task_hash, "status") != TaskStatus.PENDING.value:
                return 0
            if hash_field(task_hash, "claim_token"):
                return -1
            if await self.exists(own_cancel, parent_cancel):
                return -2
            if await self.lrem(source_queue, 1, task_id) != 1:
                return -7
            await self.lrem(destination, 0, task_id)
            await self.hset(
                task_key,
                mapping={
                    "data": replacement_payload,
                    "status": TaskStatus.PENDING.value,
                    "assigned_worker": "",
                    "assigned_at": "",
                    "started_at": "",
                    "claim_token": "",
                    "claim_worker": "",
                    "claim_worker_instance": "",
                    "claim_source_queue": "",
                    "claim_inflight_queue": "",
                    "claim_recovery_state": "",
                    "claim_recovery_reason": "",
                    "claim_recovery_at": "",
                    "queue_timeout_reason": reason,
                    "queue_timeout_at": timestamp,
                    "queue_move_token": move_token,
                    "updated_at": timestamp,
                },
            )
            await self.lpush(destination, task_id)
            return 1

        if "kernelgym:return-claim-v2" in script:
            source_queue, inflight_queue, task_key = keys
            task_id, entry, token = argv
            task_hash = self.hashes.get(task_key)
            if hash_field(task_hash, "claim_token") != token:
                await self.lrem(inflight_queue, 1, entry)
                return 0
            await self.lrem(inflight_queue, 1, entry)
            await self.lrem(source_queue, 0, task_id)
            await self.lpush(source_queue, task_id)
            await clear_claim(task_key)
            return 1

        if "kernelgym:mark-claim-processing-v3" in script:
            task_key, inflight_queue, own_cancel, parent_cancel = keys
            task_id, entry, token, started_at, worker_id, worker_instance, source_queue = argv
            task_hash = self.hashes.get(task_key, {})
            if hash_field(task_hash, "claim_token") != token:
                await self.lrem(inflight_queue, 1, entry)
                return -2
            status = hash_field(task_hash, "status")
            if status != TaskStatus.PENDING.value:
                await self.lrem(inflight_queue, 1, entry)
                await clear_claim(task_key)
                return 0
            if await self.exists(own_cancel, parent_cancel):
                await self.lrem(inflight_queue, 1, entry)
                await clear_claim(task_key)
                return -1
            await self.hset(
                task_key,
                mapping={
                    "status": TaskStatus.PROCESSING.value,
                    "started_at": started_at,
                    "claim_worker": worker_id,
                    "claim_worker_instance": worker_instance,
                    "claim_source_queue": source_queue,
                    "claim_inflight_queue": inflight_queue,
                    "claim_recovery_state": "execution_fenced",
                    "claim_recovery_reason": (
                        "GPU execution is in flight; automatic recovery requires proven process containment"
                    ),
                    "claim_recovery_at": started_at,
                },
            )
            return 1

        if "kernelgym:conditional-requeue-v2" in script:
            task_key, destination, inflight_queue, own_cancel, parent_cancel = keys
            (
                task_id,
                entry,
                token,
                payload,
                reason,
                timestamp,
                assigned_worker,
                assigned_at,
                release_mode,
            ) = argv
            task_hash = self.hashes.get(task_key)
            if hash_field(task_hash, "claim_token") != token:
                await self.lrem(inflight_queue, 1, entry)
                return -3
            recovery_state = hash_field(task_hash, "claim_recovery_state")
            release_allowed = (recovery_state == "execution_fenced" and release_mode in {"execution", "all"}) or (
                recovery_state == "frozen" and release_mode == "all"
            )
            if recovery_state and not release_allowed:
                return -4
            if not task_hash:
                await self.lrem(inflight_queue, 1, entry)
                await clear_claim(task_key)
                return 0
            status = hash_field(task_hash, "status")
            if status in {
                TaskStatus.COMPLETED.value,
                TaskStatus.FAILED.value,
                TaskStatus.TIMEOUT.value,
            }:
                await self.lrem(inflight_queue, 1, entry)
                await clear_claim(task_key)
                return -1
            if await self.exists(own_cancel, parent_cancel):
                await self.lrem(inflight_queue, 1, entry)
                await clear_claim(task_key)
                return -2
            await self.lrem(inflight_queue, 1, entry)
            await self.lrem(destination, 0, task_id)
            await self.hset(
                task_key,
                mapping={
                    "data": payload,
                    "status": TaskStatus.PENDING.value,
                    "assigned_worker": assigned_worker,
                    "assigned_at": assigned_at,
                    "started_at": "",
                    "claim_token": "",
                    "claim_worker": "",
                    "claim_worker_instance": "",
                    "claim_source_queue": "",
                    "claim_inflight_queue": "",
                    "claim_recovery_state": "",
                    "claim_recovery_reason": "",
                    "claim_recovery_at": "",
                    "queue_timeout_reason": reason,
                    "queue_timeout_at": timestamp,
                    "updated_at": timestamp,
                },
            )
            await self.lpush(destination, task_id)
            return 1

        if "kernelgym:freeze-claim-recovery-v1" in script:
            (task_key,) = keys
            token, reason, timestamp = argv
            task_hash = self.hashes.get(task_key)
            if hash_field(task_hash, "claim_token") != token:
                return 0
            await self.hset(
                task_key,
                mapping={
                    "claim_recovery_state": "frozen",
                    "claim_recovery_reason": reason,
                    "claim_recovery_at": timestamp,
                },
            )
            return 1

        if "kernelgym:ack-claim-v3" in script:
            task_key, inflight_queue = keys
            entry, token, release_fenced = argv
            task_hash = self.hashes.get(task_key)
            if not task_hash:
                await self.lrem(inflight_queue, 1, entry)
                return 1
            if hash_field(task_hash, "claim_token") != token:
                await self.lrem(inflight_queue, 1, entry)
                return 0
            if hash_field(task_hash, "claim_recovery_state") and release_fenced != "1":
                return -2
            await self.lrem(inflight_queue, 1, entry)
            await clear_claim(task_key)
            return 1

        if "kernelgym:finalize-claim-v2" in script:
            task_key, result_key, inflight_queue = keys
            token, entry, task_mapping_json, result_mapping_json, task_ttl, result_ttl, allow_frozen = argv
            task_hash = self.hashes.get(task_key)
            if hash_field(task_hash, "claim_token") != token:
                await self.lrem(inflight_queue, 1, entry)
                return 0
            if hash_field(task_hash, "claim_recovery_state") == "frozen" and allow_frozen != "1":
                return -2
            await self.hset(task_key, mapping=json.loads(task_mapping_json))
            await self.hset(result_key, mapping=json.loads(result_mapping_json))
            await self.lrem(inflight_queue, 1, entry)
            if int(task_ttl) > 0:
                await self.expire(task_key, int(task_ttl))
            if int(result_ttl) > 0:
                await self.expire(result_key, int(result_ttl))
            return 1

        raise AssertionError("unexpected Lua script")

    def pipeline(self, transaction: bool = True):  # noqa: ARG002
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
        return False

    def hset(self, *args: Any, **kwargs: Any):
        self.operations.append(("hset", args, kwargs))
        return self

    def lpush(self, *args: Any, **kwargs: Any):
        self.operations.append(("lpush", args, kwargs))
        return self

    def lrem(self, *args: Any, **kwargs: Any):
        self.operations.append(("lrem", args, kwargs))
        return self

    def expire(self, *args: Any, **kwargs: Any):
        self.operations.append(("expire", args, kwargs))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for method_name, args, kwargs in self.operations:
            results.append(await getattr(self.redis, method_name)(*args, **kwargs))
        return results


def _patch_registry(monkeypatch) -> None:
    monkeypatch.setattr(task_manager_module, "list_toolkits", lambda: ["kernelbench"])
    monkeypatch.setattr(task_manager_module, "list_backends", lambda: ["kernelbench"])


def _base_payload(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": "cuda_agent",
        "kernel_code": "code",
        "reference_code": "ref",
    }


async def _register_healthy_gpu(
    manager: TaskManager,
    redis: FakeRedis,
    worker_id: str,
    device: str = "cuda:0",
    *,
    node_id: str = "",
    hostname: str = "host-a",
) -> None:
    await manager.register_worker(worker_id, device, node_id=node_id, hostname=hostname)
    await redis.hset(
        f"{manager.worker_prefix}{worker_id}",
        mapping={"health_state": "healthy", "accepting_tasks": "true", "online": "true"},
    )


async def _supersede_active_gpu_claim(
    first: TaskManager,
    redis: FakeRedis,
    task_id: str,
) -> tuple[dict[str, Any], TaskManager]:
    """Replace ``first``'s active token through normal crash recovery."""

    task_data = await first.get_next_task("gpu-worker", resources=["gpu"])
    assert task_data is not None
    replacement = TaskManager(redis)  # type: ignore[arg-type]
    await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)
    replacement_data = await replacement.get_next_task("gpu-worker", resources=["gpu"])
    assert replacement_data is not None
    assert replacement_data["task_id"] == task_id
    assert first._task_claims[task_id].token != replacement._task_claims[task_id].token
    return task_data, replacement


def test_task_manager_routes_compile_and_execute_by_resource(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")

        await manager.submit_task({**_base_payload("compile-task"), "task_stage": "compile"})
        await manager.submit_task({**_base_payload("execute-task"), "task_stage": "execute"})

        status = await manager.get_queue_status()
        assert status["pending"] == 2

        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) == {
            **_base_payload("execute-task"),
            "task_stage": "execute",
            "required_resource": "gpu",
        }
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is None

        compile_payload = await manager.get_next_task("cpu-worker", resources=["cpu"])
        assert compile_payload == {
            **_base_payload("compile-task"),
            "task_stage": "compile",
            "required_resource": "cpu",
        }
        task_hash = await redis.hgetall(f"{manager.task_prefix}compile-task")
        assert task_hash[b"status"] == TaskStatus.PROCESSING.value.encode()

    asyncio.run(scenario())


def test_quarantined_gpu_does_not_pop_queued_task(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await manager.register_worker("gpu-worker", "cuda:0", node_id="node-a", hostname="host-a")
        await redis.hset(
            f"{manager.worker_prefix}gpu-worker",
            mapping={"health_state": "quarantined", "accepting_tasks": "false"},
        )
        await manager.submit_task({**_base_payload("held-task"), "task_stage": "execute"})

        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is None
        assert await redis.llen(manager.resource_queues["gpu"]) == 1

    asyncio.run(scenario())


def test_gpu_with_missing_admission_fields_does_not_pop_queued_task(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await redis.hset(
            f"{manager.worker_prefix}gpu-worker",
            mapping={"device": "cuda:0", "status": "online"},
        )
        await manager.submit_task({**_base_payload("held-without-heartbeat"), "task_stage": "execute"})

        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is None
        assert await redis.llen(manager.resource_queues["gpu"]) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("closed_field", ["offline", "stale", "missing_hostname"])
def test_gpu_with_closed_heartbeat_admission_does_not_claim_task(monkeypatch, closed_field: str) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        hostname = "" if closed_field == "missing_hostname" else "host-a"
        await _register_healthy_gpu(manager, redis, "gpu-worker", hostname=hostname)
        if closed_field == "offline":
            await redis.hset(f"{manager.worker_prefix}gpu-worker", mapping={"online": "false"})
        elif closed_field == "stale":
            stale = task_manager_module.datetime.now() - timedelta(
                seconds=task_manager_module.settings.worker_monitor_heartbeat_timeout + 1
            )
            await redis.hset(
                f"{manager.worker_prefix}gpu-worker",
                mapping={"last_heartbeat": stale.isoformat()},
            )
        await manager.submit_task({**_base_payload(f"held-{closed_field}"), "task_stage": "execute"})

        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is None
        assert await redis.llen(manager.resource_queues["gpu"]) == 1
        assert await redis.llen(manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")) == 0

    asyncio.run(scenario())


def test_gpu_claim_remains_recoverable_until_acknowledged(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("claimed-task"), "task_stage": "execute"})

        claimed = await manager.get_next_task("gpu-worker", resources=["gpu"])

        assert claimed is not None
        assert claimed["task_id"] == "claimed-task"
        inflight = manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")
        assert redis.lists[inflight] == [manager._task_claims["claimed-task"].entry]
        assert await redis.llen(manager.resource_queues["gpu"]) == 0
        await manager.acknowledge_task_claim(
            "claimed-task",
            release_fenced_claim=True,
        )
        assert redis.lists[inflight] == []

    asyncio.run(scenario())


def test_duplicate_queue_id_cannot_overwrite_active_claim_token(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("duplicate-id"), "task_stage": "execute"})
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is not None
        active_claim = manager._task_claims["duplicate-id"]
        task_key = f"{manager.task_prefix}duplicate-id"
        inflight = manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")

        # A stale duplicate queue entry must be discarded, not turned into a
        # second attempt that overwrites the active token.
        await redis.lpush(manager.resource_queues["gpu"], "duplicate-id")
        assert (
            await manager._claim_gpu_task(
                manager.key_prefix,
                "gpu-worker",
                manager.resource_queues["gpu"],
            )
            is None
        )

        task_hash = await redis.hgetall(task_key)
        assert task_hash[b"claim_token"] == active_claim.token.encode()
        assert redis.lists[inflight] == [active_claim.entry]
        assert await redis.llen(manager.resource_queues["gpu"]) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("stale_action", ["finalize", "requeue", "ack"])
def test_old_claim_cannot_mutate_replacement_attempt(monkeypatch, stale_action: str) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(first, redis, "gpu-worker")
        await first.submit_task({**_base_payload("fenced-task"), "task_stage": "execute"})
        old_task_data, replacement = await _supersede_active_gpu_claim(first, redis, "fenced-task")
        new_claim = replacement._task_claims["fenced-task"]

        if stale_action == "finalize":
            with pytest.raises(StaleTaskClaimError, match="superseded"):
                await first.complete_task(
                    "fenced-task",
                    {"task_id": "fenced-task", "status": "completed", "compiled": True},
                )
        elif stale_action == "requeue":
            assert await first.requeue_unstarted_task(old_task_data, "stale_old_worker") is False
        else:
            await first.acknowledge_task_claim("fenced-task")

        task_hash = await redis.hgetall(f"{replacement.task_prefix}fenced-task")
        assert task_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
        assert task_hash[b"claim_token"] == new_claim.token.encode()
        assert task_hash[b"claim_worker_instance"] == new_claim.worker_instance.encode()
        assert await redis.hgetall(f"{replacement.result_prefix}fenced-task") == {}
        assert redis.lists[new_claim.inflight_queue] == [new_claim.entry]

    asyncio.run(scenario())


def test_fail_task_does_not_adopt_unknown_active_claim_by_default(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        worker = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(worker, redis, "gpu-worker")
        await worker.submit_task({**_base_payload("unknown-claim"), "task_stage": "execute"})
        assert await worker.get_next_task("gpu-worker", resources=["gpu"]) is not None
        active_claim = worker._task_claims["unknown-claim"]

        controller = TaskManager(redis)  # type: ignore[arg-type]
        with pytest.raises(StaleTaskClaimError, match="unavailable"):
            await controller.fail_task("unknown-claim", "must not steal replacement claim")

        task_hash = await redis.hgetall(f"{worker.task_prefix}unknown-claim")
        assert task_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
        assert task_hash[b"claim_token"] == active_claim.token.encode()
        assert redis.lists[active_claim.inflight_queue] == [active_claim.entry]

    asyncio.run(scenario())


def test_cancel_task_explicitly_adopts_and_finalizes_active_claim(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        worker = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(worker, redis, "gpu-worker")
        await worker.submit_task({**_base_payload("cancel-active"), "task_stage": "execute"})
        assert await worker.get_next_task("gpu-worker", resources=["gpu"]) is not None
        active_claim = worker._task_claims["cancel-active"]

        controller = TaskManager(redis)  # type: ignore[arg-type]
        assert await controller.cancel_task("cancel-active") is True

        task_hash = await redis.hgetall(f"{worker.task_prefix}cancel-active")
        assert task_hash[b"status"] == TaskStatus.FAILED.value.encode()
        assert task_hash[b"claim_token"] == b""
        assert redis.lists[active_claim.inflight_queue] == []

    asyncio.run(scenario())


def test_terminal_result_transaction_acknowledges_gpu_claim(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("terminal-ack"), "task_stage": "execute"})
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is not None

        await manager.complete_task(
            "terminal-ack",
            {"task_id": "terminal-ack", "status": "completed", "compiled": True},
        )

        inflight = manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")
        assert redis.lists[inflight] == []
        task_hash = await redis.hgetall(f"{manager.task_prefix}terminal-ack")
        result_hash = await redis.hgetall(f"{manager.result_prefix}terminal-ack")
        assert task_hash[b"status"] == TaskStatus.COMPLETED.value.encode()
        assert b"result" in result_hash
        assert "terminal-ack" not in manager._task_claims

    asyncio.run(scenario())


def test_terminal_transaction_failure_preserves_inflight_recovery_authority(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("terminal-failure"), "task_stage": "execute"})
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is not None
        claim_entry = manager._task_claims["terminal-failure"].entry
        original_eval = redis.eval

        async def fail_finalize(script: str, numkeys: int, *values: Any) -> Any:
            if "kernelgym:finalize-claim-v2" in script:
                raise RuntimeError("terminal transaction unavailable")
            return await original_eval(script, numkeys, *values)

        monkeypatch.setattr(redis, "eval", fail_finalize)

        with pytest.raises(RuntimeError, match="terminal transaction unavailable"):
            await manager.complete_task(
                "terminal-failure",
                {"task_id": "terminal-failure", "status": "completed", "compiled": True},
            )

        inflight = manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")
        assert redis.lists[inflight] == [claim_entry]
        task_hash = await redis.hgetall(f"{manager.task_prefix}terminal-failure")
        assert task_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
        assert await redis.hgetall(f"{manager.result_prefix}terminal-failure") == {}
        assert "terminal-failure" in manager._task_claims

    asyncio.run(scenario())


def test_replacement_gpu_worker_recovers_crash_window_claim(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(first, redis, "gpu-worker")
        await first.submit_task({**_base_payload("recover-task"), "task_stage": "execute"})
        assert await first.get_next_task("gpu-worker", resources=["gpu"]) is not None

        # Simulate a process exit before acknowledgement by constructing the
        # replacement worker's TaskManager against the same Redis state. A
        # proven-safe fresh pool/SID containment is required before release.
        replacement = TaskManager(redis)  # type: ignore[arg-type]
        await replacement.recover_gpu_inflight("gpu-worker")
        fenced_hash = await redis.hgetall(f"{replacement.task_prefix}recover-task")
        assert fenced_hash[b"claim_recovery_state"] == b"execution_fenced"
        assert fenced_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)
        recovered = await replacement.get_next_task("gpu-worker", resources=["gpu"])

        assert recovered is not None
        assert recovered["task_id"] == "recover-task"
        task_hash = await redis.hgetall(f"{replacement.task_prefix}recover-task")
        assert task_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
        assert task_hash[b"queue_timeout_reason"] == b"worker_process_recovered_inflight_claim"
        assert redis.lists[replacement._gpu_inflight_queue(replacement.key_prefix, "gpu-worker")] == [
            replacement._task_claims["recover-task"].entry
        ]

    asyncio.run(scenario())


def test_recovery_repeats_after_initial_empty_scan(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        old_process = TaskManager(redis)  # type: ignore[arg-type]
        replacement = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(old_process, redis, "gpu-worker")

        # The replacement can start its recovery pass before an earlier process's
        # final Redis claim write becomes visible. A later pass must not be
        # suppressed merely because the first scan was empty.
        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)
        await old_process.submit_task({**_base_payload("late-stale-claim"), "task_stage": "execute"})
        assert await old_process.get_next_task("gpu-worker", resources=["gpu"]) is not None
        old_claim = old_process._task_claims["late-stale-claim"]
        # The old process is considered crashed after its durable claim write;
        # the production launcher proves its process group absent before this
        # replacement is allowed to execute recovery.

        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)

        task_hash = await redis.hgetall(f"{replacement.task_prefix}late-stale-claim")
        assert task_hash[b"status"] == TaskStatus.PENDING.value.encode()
        assert task_hash[b"claim_token"] == b""
        assert redis.lists[old_claim.inflight_queue] == []
        assert redis.lists[replacement.resource_queues["gpu"]] == ["late-stale-claim"]

    asyncio.run(scenario())


def test_concurrent_recovery_scans_are_serialized_per_worker(monkeypatch) -> None:
    class ObservedRecoveryRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.target_queue = ""
            self.active_scans = 0
            self.max_active_scans = 0

        async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
            if key == self.target_queue:
                self.active_scans += 1
                self.max_active_scans = max(self.max_active_scans, self.active_scans)
                await asyncio.sleep(0)
                try:
                    return await super().lrange(key, start, end)
                finally:
                    self.active_scans -= 1
            return await super().lrange(key, start, end)

    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = ObservedRecoveryRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        redis.target_queue = manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")

        await asyncio.gather(
            manager._recover_gpu_inflight(manager.key_prefix, "gpu-worker"),
            manager._recover_gpu_inflight(manager.key_prefix, "gpu-worker"),
        )

        assert redis.max_active_scans == 1

    asyncio.run(scenario())


def test_recovery_restores_direct_claim_to_original_queue_and_assignment(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(first, redis, "gpu-worker")
        payload = {
            **_base_payload("recover-direct"),
            "task_stage": "execute",
            "assigned_worker": "gpu-worker",
        }
        await first.submit_task(payload)
        task_key = f"{first.task_prefix}recover-direct"
        before_claim = await redis.hgetall(task_key)
        original_assigned_at = before_claim[b"assigned_at"]
        assert await first.get_next_task("gpu-worker", resources=["gpu"]) is not None
        old_claim = first._task_claims["recover-direct"]

        replacement = TaskManager(redis)  # type: ignore[arg-type]
        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)

        direct_queue = f"{replacement.key_prefix}:queue:worker:gpu-worker"
        assert redis.lists[direct_queue] == ["recover-direct"]
        assert redis.lists[replacement.resource_queues["gpu"]] == []
        assert redis.lists[old_claim.inflight_queue] == []
        task_hash = await redis.hgetall(task_key)
        assert task_hash[b"status"] == TaskStatus.PENDING.value.encode()
        assert task_hash[b"assigned_worker"] == b"gpu-worker"
        assert task_hash[b"assigned_at"] == original_assigned_at
        assert task_hash[b"claim_token"] == b""
        restored_payload = json.loads(task_hash[b"data"].decode())
        assert restored_payload["assigned_worker"] == "gpu-worker"

    asyncio.run(scenario())


def test_frozen_claim_waits_for_explicit_safe_recovery_release(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(first, redis, "gpu-worker")
        await first.submit_task({**_base_payload("frozen-claim"), "task_stage": "execute"})
        assert await first.get_next_task("gpu-worker", resources=["gpu"]) is not None
        old_claim = first._task_claims["frozen-claim"]
        assert await first.freeze_task_claim("frozen-claim", "unsafe CUDA child still exists") is True

        replacement = TaskManager(redis)  # type: ignore[arg-type]
        await replacement.recover_gpu_inflight("gpu-worker")

        task_key = f"{replacement.task_prefix}frozen-claim"
        frozen_hash = await redis.hgetall(task_key)
        assert frozen_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
        assert frozen_hash[b"claim_token"] == old_claim.token.encode()
        assert frozen_hash[b"claim_recovery_state"] == b"frozen"
        assert redis.lists[old_claim.inflight_queue] == [old_claim.entry]
        assert redis.lists[replacement.resource_queues["gpu"]] == []
        assert "frozen-claim" not in replacement._task_claims

        # Trusted only after quarantine clear plus a successful fresh CUDA pool
        # initialization; this second recovery mode has its own idempotence key.
        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)

        released_hash = await redis.hgetall(task_key)
        assert released_hash[b"status"] == TaskStatus.PENDING.value.encode()
        assert released_hash[b"claim_token"] == b""
        assert released_hash[b"claim_recovery_state"] == b""
        assert redis.lists[old_claim.inflight_queue] == []
        assert redis.lists[replacement.resource_queues["gpu"]] == ["frozen-claim"]

    asyncio.run(scenario())


def test_ordinary_requeue_cannot_release_frozen_claim(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("frozen-local"), "task_stage": "execute"})
        task_data = await manager.get_next_task("gpu-worker", resources=["gpu"])
        assert task_data is not None
        claim = manager._task_claims["frozen-local"]
        assert await manager.freeze_task_claim("frozen-local", "containment pending") is True

        assert await manager.requeue_unstarted_task(task_data, "ordinary_gate_close") is False
        assert (
            await manager.requeue_unstarted_task(
                task_data,
                "stale unstarted path must not release containment",
                release_execution_fence=True,
            )
            is False
        )
        await manager.acknowledge_task_claim("frozen-local")

        frozen_hash = await redis.hgetall(f"{manager.task_prefix}frozen-local")
        assert frozen_hash[b"claim_token"] == claim.token.encode()
        assert frozen_hash[b"claim_recovery_state"] == b"frozen"
        assert manager._task_claims["frozen-local"] == claim
        assert redis.lists[claim.inflight_queue] == [claim.entry]
        assert redis.lists[manager.resource_queues["gpu"]] == []

    asyncio.run(scenario())


def test_safe_shutdown_terminal_commit_clears_frozen_recovery_marker(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("safe-shutdown"), "task_stage": "execute"})
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is not None
        claim = manager._task_claims["safe-shutdown"]
        assert await manager.freeze_task_claim("safe-shutdown", "containment pending") is True

        await manager.fail_task(
            "safe-shutdown",
            "Worker shutdown",
            allow_frozen_claim=True,
        )

        task_hash = await redis.hgetall(f"{manager.task_prefix}safe-shutdown")
        assert task_hash[b"status"] == TaskStatus.FAILED.value.encode()
        assert task_hash[b"claim_token"] == b""
        assert task_hash[b"claim_recovery_state"] == b""
        assert task_hash[b"claim_recovery_reason"] == b""
        assert redis.lists[claim.inflight_queue] == []

    asyncio.run(scenario())


def test_ordinary_terminal_commit_cannot_ack_containment_frozen_claim(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("unsafe-terminal"), "task_stage": "execute"})
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is not None
        claim = manager._task_claims["unsafe-terminal"]
        assert await manager.freeze_task_claim("unsafe-terminal", "containment owns terminal") is True

        with pytest.raises(FrozenTaskClaimError, match="owned by containment"):
            await manager.complete_task(
                "unsafe-terminal",
                {"task_id": "unsafe-terminal", "status": "failed", "error_message": "unsafe"},
            )

        task_hash = await redis.hgetall(f"{manager.task_prefix}unsafe-terminal")
        assert task_hash[b"status"] == TaskStatus.PROCESSING.value.encode()
        assert task_hash[b"claim_token"] == claim.token.encode()
        assert task_hash[b"claim_recovery_state"] == b"frozen"
        assert redis.lists[claim.inflight_queue] == [claim.entry]
        assert "unsafe-terminal" in manager._task_claims

    asyncio.run(scenario())


@pytest.mark.parametrize("race", ["terminal", "cancel"])
def test_safe_release_does_not_resurrect_frozen_terminal_or_cancelled_task(monkeypatch, race: str) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(first, redis, "gpu-worker")
        task_id = f"frozen-{race}-race"
        await first.submit_task({**_base_payload(task_id), "task_stage": "execute"})
        assert await first.get_next_task("gpu-worker", resources=["gpu"]) is not None
        claim = first._task_claims[task_id]
        assert await first.freeze_task_claim(task_id, "unsafe containment") is True
        if race == "terminal":
            await redis.hset(f"{first.task_prefix}{task_id}", mapping={"status": TaskStatus.FAILED.value})
        else:
            await redis.set(first._cancel_key(task_id), "1")

        replacement = TaskManager(redis)  # type: ignore[arg-type]
        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)

        task_hash = await redis.hgetall(f"{first.task_prefix}{task_id}")
        assert task_hash[b"status"] == TaskStatus.FAILED.value.encode()
        assert task_hash[b"claim_token"] == b""
        assert task_hash[b"claim_recovery_state"] == b""
        assert redis.lists[claim.inflight_queue] == []
        assert redis.lists[replacement.resource_queues["gpu"]] == []
        if race == "cancel":
            result_hash = await redis.hgetall(f"{first.result_prefix}{task_id}")
            assert result_hash[b"error"] == b"Task cancelled"
            assert b"cancelled_at" in task_hash

    asyncio.run(scenario())


def test_cancelled_claim_race_does_not_abort_remaining_recovery(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(first, redis, "gpu-worker")
        task_ids = ("cancel-race-first", "cancel-race-next")
        for task_id in task_ids:
            await first.submit_task({**_base_payload(task_id), "task_stage": "execute"})
            assert await first.get_next_task("gpu-worker", resources=["gpu"]) is not None
            assert await first.freeze_task_claim(task_id, "unsafe containment") is True
            await redis.set(first._cancel_key(task_id), "1")

        replacement = TaskManager(redis)  # type: ignore[arg-type]
        original_fail = replacement.fail_task
        calls: list[str] = []

        async def race_once(task_id, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            calls.append(task_id)
            if len(calls) == 1:
                claim = replacement._task_claims[task_id]
                await redis.hset(
                    f"{replacement.task_prefix}{task_id}",
                    mapping={"status": TaskStatus.FAILED.value, "claim_token": ""},
                )
                await redis.lrem(claim.inflight_queue, 1, claim.entry)
                raise StaleTaskClaimError("another recovery won the claim")
            return await original_fail(task_id, *args, **kwargs)

        monkeypatch.setattr(replacement, "fail_task", race_once)
        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)

        assert len(calls) == 2
        assert not redis.lists[first._gpu_inflight_queue(first.key_prefix, "gpu-worker")]
        assert not replacement._task_claims
        recovered_result = await redis.hgetall(f"{replacement.result_prefix}{calls[1]}")
        assert recovered_result[b"error"] == b"Task cancelled"

    asyncio.run(scenario())


def test_cancelled_at_write_failure_does_not_abort_remaining_recovery(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(first, redis, "gpu-worker")
        task_ids = ("cancel-metadata-first", "cancel-metadata-next")
        for task_id in task_ids:
            await first.submit_task({**_base_payload(task_id), "task_stage": "execute"})
            assert await first.get_next_task("gpu-worker", resources=["gpu"]) is not None
            assert await first.freeze_task_claim(task_id, "unsafe containment") is True
            await redis.set(first._cancel_key(task_id), "1")

        replacement = TaskManager(redis)  # type: ignore[arg-type]
        original_hset = redis.hset
        metadata_writes = 0

        async def fail_first_metadata_write(key, mapping):  # noqa: ANN001
            nonlocal metadata_writes
            if set(mapping) == {"cancelled_at"}:
                metadata_writes += 1
                if metadata_writes == 1:
                    raise RuntimeError("diagnostic write unavailable")
            return await original_hset(key, mapping)

        monkeypatch.setattr(redis, "hset", fail_first_metadata_write)
        await replacement.recover_gpu_inflight("gpu-worker", release_frozen_claims=True)

        assert metadata_writes == 2
        assert not redis.lists[first._gpu_inflight_queue(first.key_prefix, "gpu-worker")]
        for task_id in task_ids:
            result = await redis.hgetall(f"{replacement.result_prefix}{task_id}")
            assert result[b"error"] == b"Task cancelled"

    asyncio.run(scenario())


def test_conditional_requeue_never_resurrects_terminal_task(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("terminal-race"), "task_stage": "execute"})
        task_data = await manager.get_next_task("gpu-worker", resources=["gpu"])
        assert task_data is not None
        await redis.hset(
            f"{manager.task_prefix}terminal-race",
            mapping={"status": TaskStatus.FAILED.value},
        )

        restored = await manager.requeue_unstarted_task(
            task_data,
            "gpu_gate_closed",
            release_execution_fence=True,
        )

        assert restored is False
        assert await redis.llen(manager.resource_queues["gpu"]) == 0
        assert await redis.llen(manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")) == 0
        task_hash = await redis.hgetall(f"{manager.task_prefix}terminal-race")
        assert task_hash[b"status"] == TaskStatus.FAILED.value.encode()
        assert task_hash[b"claim_token"] == b""
        assert task_hash[b"claim_worker"] == b""
        assert task_hash[b"claim_worker_instance"] == b""

    asyncio.run(scenario())


def test_atomic_dispatch_rejects_cancel_marker_created_after_claim(monkeypatch) -> None:
    class CancelBeforeDispatchRedis(FakeRedis):
        async def eval(self, script: str, numkeys: int, *values: Any) -> Any:
            if "kernelgym:mark-claim-processing-v3" in script:
                keys = [str(value) for value in values[:numkeys]]
                await self.set(keys[2], "1")
            return await super().eval(script, numkeys, *values)

    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = CancelBeforeDispatchRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("cancel-race"), "task_stage": "execute"})

        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is None
        assert await redis.llen(manager.resource_queues["gpu"]) == 0
        assert await redis.llen(manager._gpu_inflight_queue(manager.key_prefix, "gpu-worker")) == 0
        task_hash = await redis.hgetall(f"{manager.task_prefix}cancel-race")
        assert task_hash[b"status"] == TaskStatus.PENDING.value.encode()
        assert task_hash[b"claim_token"] == b""
        assert task_hash[b"claim_worker"] == b""
        assert task_hash[b"claim_worker_instance"] == b""

    asyncio.run(scenario())


def test_task_manager_rechecks_gpu_gate_after_pop_and_restores_once(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("gate-race-task"), "task_stage": "execute"})

        checks = iter((True, False))

        async def admission_changes(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
            return next(checks)

        monkeypatch.setattr(manager, "_gpu_worker_admission_open", admission_changes)

        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is None
        assert await redis.llen(manager.resource_queues["gpu"]) == 1
        assert redis.lists[manager.resource_queues["gpu"]] == ["gate-race-task"]
        task_hash = await redis.hgetall(f"{manager.task_prefix}gate-race-task")
        assert task_hash[b"status"] == TaskStatus.PENDING.value.encode()
        assert task_hash[b"started_at"] == b""

    asyncio.run(scenario())


def test_unstarted_requeue_propagates_redis_transaction_failure(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("not-lost-silently"), "required_resource": "gpu"})
        task_data = await manager.get_next_task("gpu-worker", resources=["gpu"])
        assert task_data is not None
        original_eval = redis.eval

        async def fail_requeue(script: str, numkeys: int, *values: Any) -> Any:
            if "kernelgym:conditional-requeue-v2" in script:
                raise RuntimeError("transaction unavailable")
            return await original_eval(script, numkeys, *values)

        monkeypatch.setattr(redis, "eval", fail_requeue)

        with pytest.raises(RuntimeError, match="transaction unavailable"):
            await manager.requeue_unstarted_task(
                task_data,
                "gate_closed",
                release_execution_fence=True,
            )

    asyncio.run(scenario())


def test_unstarted_task_is_restored_after_admission_closes_post_pop(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        await manager.submit_task({**_base_payload("race-task"), "task_stage": "execute"})

        task_data = await manager.get_next_task("gpu-worker", resources=["gpu"])
        assert task_data is not None
        assert await redis.llen(manager.resource_queues["gpu"]) == 0

        await manager.requeue_unstarted_task(
            task_data,
            "gpu_admission_closed_after_dequeue",
            release_execution_fence=True,
        )

        assert await redis.llen(manager.resource_queues["gpu"]) == 1
        task_hash = await redis.hgetall(f"{manager.task_prefix}race-task")
        assert task_hash[b"status"] == TaskStatus.PENDING.value.encode()
        assert task_hash[b"started_at"] == b""

    asyncio.run(scenario())


def test_queue_wait_move_is_atomic_and_preserves_namespace(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        task_id = "legacy-direct-timeout"
        prefix = manager.legacy_prefix
        worker_id = "gpu-worker"
        worker_queue = f"{prefix}:queue:worker:{worker_id}"
        payload = {
            **_base_payload(task_id),
            "assigned_worker": worker_id,
            "required_resource": "gpu",
        }
        assigned_at = (task_manager_module.datetime.now() - timedelta(seconds=300)).isoformat()
        await redis.hset(
            f"{prefix}:task:{task_id}",
            mapping={
                "data": json.dumps(payload),
                "status": TaskStatus.PENDING.value,
                "assigned_worker": worker_id,
                "assigned_at": assigned_at,
                "submitted_at": assigned_at,
            },
        )
        await redis.lpush(worker_queue, task_id)
        snapshot = await redis.hgetall(f"{prefix}:task:{task_id}")

        assert (
            await manager._conditionally_requeue_waiting_task(
                prefix=prefix,
                worker_queue_key=worker_queue,
                task_id=task_id,
                task_data=snapshot,
                task_json=payload,
                reason="queue_wait_timeout",
                now_iso=task_manager_module.datetime.now().isoformat(),
                operation="requeue",
            )
            == 1
        )

        destination = f"{prefix}:queue:resource:gpu"
        assert redis.lists[worker_queue] == []
        assert redis.lists[destination] == [task_id]
        task_hash = await redis.hgetall(f"{prefix}:task:{task_id}")
        assert task_hash[b"status"] == TaskStatus.PENDING.value.encode()
        assert task_hash[b"assigned_worker"] == b""
        assert task_hash[b"queue_timeout_reason"] == b"queue_wait_timeout"

    asyncio.run(scenario())


@pytest.mark.parametrize("race", ["worker_claim", "assigned_at_change", "cancel"])
def test_queue_wait_move_loses_race_without_mutating_new_state(monkeypatch, race: str) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        task_id = f"queue-wait-race-{race}"
        payload = {
            **_base_payload(task_id),
            "assigned_worker": "gpu-worker",
            "required_resource": "gpu",
        }
        await manager.submit_task(payload)
        worker_queue = manager.worker_queues["gpu-worker"]
        task_key = f"{manager.task_prefix}{task_id}"
        snapshot = await redis.hgetall(task_key)

        active_claim = None
        changed_assigned_at = ""
        if race == "worker_claim":
            assert await manager._claim_gpu_task(manager.key_prefix, "gpu-worker", worker_queue) == task_id
            active_claim = manager._task_claims[task_id]
        elif race == "assigned_at_change":
            changed_assigned_at = task_manager_module.datetime.now().isoformat()
            await redis.hset(task_key, mapping={"assigned_at": changed_assigned_at})
        else:
            await redis.set(manager._cancel_key(task_id), "1")

        assert (
            await manager._conditionally_requeue_waiting_task(
                prefix=manager.key_prefix,
                worker_queue_key=worker_queue,
                task_id=task_id,
                task_data=snapshot,
                task_json=payload,
                reason="queue_wait_timeout",
                now_iso=task_manager_module.datetime.now().isoformat(),
                operation="requeue",
            )
            <= 0
        )

        task_hash = await redis.hgetall(task_key)
        assert redis.lists[manager.resource_queues["gpu"]] == []
        if active_claim is not None:
            assert task_hash[b"claim_token"] == active_claim.token.encode()
            assert redis.lists[active_claim.inflight_queue] == [active_claim.entry]
            assert redis.lists[worker_queue] == []
        else:
            assert b"claim_token" not in task_hash
            assert redis.lists[worker_queue] == [task_id]
            if changed_assigned_at:
                assert task_hash[b"assigned_at"] == changed_assigned_at.encode()

    asyncio.run(scenario())


def test_queue_wait_cleanup_only_removes_foreign_worker_stale_copy(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        task_id = "foreign-worker-copy"
        worker_a_queue = f"{manager.key_prefix}:queue:worker:worker-a"
        worker_b_queue = f"{manager.key_prefix}:queue:worker:worker-b"
        assigned_at = task_manager_module.datetime.now().isoformat()
        payload = {
            **_base_payload(task_id),
            "assigned_worker": "worker-b",
            "required_resource": "gpu",
        }
        task_key = f"{manager.task_prefix}{task_id}"
        await redis.hset(
            task_key,
            mapping={
                "data": json.dumps(payload),
                "status": TaskStatus.PENDING.value,
                "assigned_worker": "worker-b",
                "assigned_at": assigned_at,
                "submitted_at": assigned_at,
                "claim_token": "",
            },
        )
        await redis.lpush(worker_a_queue, task_id)
        await redis.lpush(worker_b_queue, task_id)
        snapshot = await redis.hgetall(task_key)

        result = await manager._conditionally_requeue_waiting_task(
            prefix=manager.key_prefix,
            worker_queue_key=worker_a_queue,
            task_id=task_id,
            task_data=snapshot,
            task_json=payload,
            reason="stale_assignment_copy",
            now_iso=task_manager_module.datetime.now().isoformat(),
            operation="remove_stale_copy",
        )

        assert result == 2
        assert redis.lists[worker_a_queue] == []
        assert redis.lists[worker_b_queue] == [task_id]
        assert redis.lists[manager.resource_queues["gpu"]] == []
        assert await redis.hgetall(task_key) == snapshot

    asyncio.run(scenario())


def test_queue_wait_monitor_recovers_queue_without_worker_heartbeat(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        monkeypatch.setattr(task_manager_module.settings, "worker_queue_wait_timeout_sec", 10)
        monkeypatch.setattr(task_manager_module.settings, "worker_queue_wait_monitor_interval", 1)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        task_id = "missing-heartbeat-worker"
        worker_id = "vanished-worker"
        worker_queue = f"{manager.key_prefix}:queue:worker:{worker_id}"
        assigned_at = (task_manager_module.datetime.now() - timedelta(seconds=300)).isoformat()
        payload = {
            **_base_payload(task_id),
            "assigned_worker": worker_id,
            "required_resource": "gpu",
        }
        await redis.hset(
            f"{manager.task_prefix}{task_id}",
            mapping={
                "data": json.dumps(payload),
                "status": TaskStatus.PENDING.value,
                "assigned_worker": worker_id,
                "assigned_at": assigned_at,
                "submitted_at": assigned_at,
                "claim_token": "",
            },
        )
        # Deliberately create only the list key: no worker heartbeat/hash exists.
        await redis.lpush(worker_queue, task_id)

        async def stop_after_first_pass(delay: float) -> None:  # noqa: ARG001
            raise asyncio.CancelledError

        monkeypatch.setattr(task_manager_module.asyncio, "sleep", stop_after_first_pass)
        with pytest.raises(asyncio.CancelledError):
            await manager._queue_wait_monitor()

        assert redis.lists[worker_queue] == []
        assert redis.lists[manager.resource_queues["gpu"]] == [task_id]
        task_hash = await redis.hgetall(f"{manager.task_prefix}{task_id}")
        assert task_hash[b"assigned_worker"] == b""
        assert task_hash[b"queue_timeout_reason"] == b"queue_wait_timeout"

    asyncio.run(scenario())


def test_persistent_quarantine_excludes_gpu_scheduler_candidate(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(
            manager,
            redis,
            "gpu-worker",
            node_id="node-a",
            hostname="host-a",
        )
        await redis.hset(
            f"{manager.worker_prefix}gpu-worker",
            mapping={"current_task": "", "last_heartbeat": task_manager_module.datetime.now().isoformat()},
        )
        await redis.hset(
            f"{manager.key_prefix}:quarantine:worker:gpu-worker",
            mapping={"state": "quarantined", "reason": "driver init failed"},
        )

        assert await manager.select_idle_worker("gpu") is None

    asyncio.run(scenario())


def test_task_manager_honors_required_node_affinity(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(
            manager,
            redis,
            "gpu-node-a",
            node_id="node-a",
            hostname="host-a",
        )
        await _register_healthy_gpu(
            manager,
            redis,
            "gpu-node-b",
            node_id="node-b",
            hostname="host-b",
        )

        await manager.submit_task(
            {
                **_base_payload("node-b-task"),
                "task_stage": "execute",
                "node_affinity": "required",
                "target_node_id": "node-b",
            }
        )

        assert await manager.get_next_task("gpu-node-a", resources=["gpu"]) is None
        matched = await manager.get_next_task("gpu-node-b", resources=["gpu"])
        assert matched is not None
        assert matched["task_id"] == "node-b-task"

    asyncio.run(scenario())


def test_task_manager_selects_idle_gpu_worker(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(
            manager,
            redis,
            "gpu-busy",
            node_id="node-a",
            hostname="host-a",
        )
        await _register_healthy_gpu(
            manager,
            redis,
            "gpu-idle",
            device="cuda:1",
            node_id="node-b",
            hostname="host-b",
        )
        await redis.hset(f"{manager.worker_prefix}gpu-busy", mapping={"current_task": "task"})
        await redis.hset(f"{manager.worker_prefix}gpu-idle", mapping={"current_task": ""})

        selected = await manager.select_idle_worker("gpu")

        assert selected is not None
        assert selected["worker_id"] == "gpu-idle"
        assert selected["node_id"] == "node-b"

    asyncio.run(scenario())


def test_task_manager_selects_busy_gpu_worker_by_task_id(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(
            manager,
            redis,
            "gpu-a",
            node_id="node-a",
            hostname="host-a",
        )
        await _register_healthy_gpu(
            manager,
            redis,
            "gpu-b",
            device="cuda:1",
            node_id="node-b",
            hostname="host-b",
        )
        await redis.hset(f"{manager.worker_prefix}gpu-a", mapping={"current_task": "task-a"})
        await redis.hset(f"{manager.worker_prefix}gpu-b", mapping={"current_task": "task-b"})

        task_id = "parallel_task_001692_bf77c294_kernel"
        selected = await manager.select_worker_by_task_id("gpu", task_id)

        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        expected_worker_id = ["gpu-a", "gpu-b"][int.from_bytes(digest[:8], "big") % 2]
        assert selected is not None
        assert selected["worker_id"] == expected_worker_id

    asyncio.run(scenario())


def test_task_manager_heartbeat_backfills_node_metadata(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await manager.register_worker("gpu-empty", "cuda:0")

        await manager.update_worker_heartbeat("gpu-empty", node_id="v1", hostname="ai-16-39")

        stored = await redis.hgetall(f"{manager.worker_prefix}gpu-empty")
        assert stored[b"node_id"] == b"v1"
        assert stored[b"hostname"] == b"ai-16-39"
        assert manager.worker_registry["gpu-empty"]["node_id"] == "v1"
        assert manager.worker_registry["gpu-empty"]["hostname"] == "ai-16-39"

    asyncio.run(scenario())


def test_task_manager_preserves_direct_worker_queue(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")

        payload = {**_base_payload("direct-task"), "required_resource": "cpu", "assigned_worker": "gpu-worker"}
        await manager.submit_task(payload)

        assert await manager.get_queue_status() == {
            "pending": 0,
            "pending_by_prefix": {"kernelgym": 0, "kernelserver": 0},
            "worker_queues": {"gpu-worker": 1},
        }
        direct_payload = await manager.get_next_task("gpu-worker", resources=["gpu"])
        assert direct_payload == {**payload, "required_resource": "cpu"}
        stored = await redis.hgetall(f"{manager.task_prefix}direct-task")
        stored_payload = json.loads(stored[b"data"].decode())
        assert stored_payload["assigned_worker"] == "gpu-worker"

    asyncio.run(scenario())


def test_task_manager_normalizes_none_assigned_worker(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

        await manager.submit_task({**_base_payload("none-worker-task"), "assigned_worker": None})

        assert await manager.get_queue_status() == {
            "pending": 1,
            "pending_by_prefix": {"kernelgym": 1, "kernelserver": 0},
            "worker_queues": {},
        }
        stored = await redis.hgetall(f"{manager.task_prefix}none-worker-task")
        assert stored[b"assigned_worker"] == b""
        stored_payload = json.loads(stored[b"data"].decode())
        assert stored_payload["assigned_worker"] == ""

    asyncio.run(scenario())


def test_task_manager_force_refresh_resubmits_existing_task(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

        await manager.submit_task({**_base_payload("refresh-task"), "kernel_code": "old"})
        await manager.complete_task("refresh-task", {"task_id": "refresh-task", "compiled": False})

        await manager.submit_task({**_base_payload("refresh-task"), "kernel_code": "new", "force_refresh": True})

        stored = await redis.hgetall(f"{manager.task_prefix}refresh-task")
        stored_payload = json.loads(stored[b"data"].decode())
        assert stored_payload["kernel_code"] == "new"
        assert await manager.get_task_result("refresh-task") is None
        assert await manager.get_queue_status() == {
            "pending": 1,
            "pending_by_prefix": {"kernelgym": 1, "kernelserver": 0},
            "worker_queues": {},
        }

    asyncio.run(scenario())


def test_concurrent_normal_submits_create_one_immutable_task(monkeypatch) -> None:
    class SerializedEvalRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.eval_lock = asyncio.Lock()

        async def eval(self, script: str, numkeys: int, *values: Any) -> Any:
            async with self.eval_lock:
                await asyncio.sleep(0)
                return await super().eval(script, numkeys, *values)

    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = SerializedEvalRedis()
        first = TaskManager(redis)  # type: ignore[arg-type]
        second = TaskManager(redis)  # type: ignore[arg-type]
        task_id = "concurrent-normal-submit"

        await asyncio.gather(
            first.submit_task({**_base_payload(task_id), "kernel_code": "first"}),
            second.submit_task({**_base_payload(task_id), "kernel_code": "second"}),
        )

        task_hash = await redis.hgetall(f"{first.task_prefix}{task_id}")
        stored_payload = json.loads(task_hash[b"data"].decode())
        assert stored_payload["kernel_code"] in {"first", "second"}
        assert redis.lists[first.resource_queues["gpu"]] == [task_id]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_point", ["before", "after"])
def test_normal_submit_failure_cannot_leave_unqueued_hash(monkeypatch, failure_point: str) -> None:
    class FailedAtomicSubmitRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def eval(self, script: str, numkeys: int, *values: Any) -> Any:
            if "kernelgym:submit-task-if-absent-v1" in script and not self.failed:
                self.failed = True
                if failure_point == "after":
                    await super().eval(script, numkeys, *values)
                raise ConnectionError("submit transaction unavailable")
            return await super().eval(script, numkeys, *values)

    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FailedAtomicSubmitRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        task_id = "atomic-submit-failure"

        with pytest.raises(ConnectionError, match="submit transaction unavailable"):
            await manager.submit_task(_base_payload(task_id))

        task_hash = await redis.hgetall(f"{manager.task_prefix}{task_id}")
        queued = redis.lists[manager.resource_queues["gpu"]]
        if failure_point == "before":
            assert task_hash == {}
            assert queued == []
        else:
            assert task_hash[b"status"] == TaskStatus.PENDING.value.encode()
            assert queued == [task_id]

        # Retrying an ambiguous call either creates the complete pair or sees
        # the pair already committed; it never duplicates the queue entry.
        await manager.submit_task(_base_payload(task_id))
        assert await redis.hgetall(f"{manager.task_prefix}{task_id}")
        assert redis.lists[manager.resource_queues["gpu"]] == [task_id]

    asyncio.run(scenario())


def test_absent_normal_submit_and_force_refresh_serialize(monkeypatch) -> None:
    class SerializedEvalRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.eval_lock = asyncio.Lock()

        async def eval(self, script: str, numkeys: int, *values: Any) -> Any:
            async with self.eval_lock:
                await asyncio.sleep(0)
                return await super().eval(script, numkeys, *values)

    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = SerializedEvalRedis()
        normal = TaskManager(redis)  # type: ignore[arg-type]
        refresh = TaskManager(redis)  # type: ignore[arg-type]
        task_id = "normal-force-absent-race"

        outcomes = await asyncio.gather(
            normal.submit_task({**_base_payload(task_id), "kernel_code": "normal"}),
            refresh.submit_task({**_base_payload(task_id), "kernel_code": "refresh", "force_refresh": True}),
            return_exceptions=True,
        )

        assert all(isinstance(outcome, (str, TaskRefreshConflictError)) for outcome in outcomes)
        task_hash = await redis.hgetall(f"{normal.task_prefix}{task_id}")
        stored_payload = json.loads(task_hash[b"data"].decode())
        assert stored_payload["kernel_code"] in {"normal", "refresh"}
        assert redis.lists[normal.resource_queues["gpu"]] == [task_id]

    asyncio.run(scenario())


def test_force_refresh_cannot_replace_active_gpu_claim(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        task_id = "refresh-active-claim"
        await manager.submit_task({**_base_payload(task_id), "kernel_code": "old", "task_stage": "execute"})
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is not None

        task_key = f"{manager.task_prefix}{task_id}"
        claim = manager._task_claims[task_id]
        before_hash = await redis.hgetall(task_key)
        before_inflight = list(redis.lists[claim.inflight_queue])

        with pytest.raises(TaskRefreshConflictError, match="active claim token"):
            await manager.submit_task(
                {
                    **_base_payload(task_id),
                    "kernel_code": "replacement",
                    "task_stage": "execute",
                    "force_refresh": True,
                }
            )

        assert await redis.hgetall(task_key) == before_hash
        assert redis.lists[claim.inflight_queue] == before_inflight
        assert redis.lists[manager.resource_queues["gpu"]] == []
        assert json.loads(before_hash[b"data"].decode())["kernel_code"] == "old"

    asyncio.run(scenario())


def test_force_refresh_cannot_delete_frozen_gpu_claim(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await _register_healthy_gpu(manager, redis, "gpu-worker")
        task_id = "refresh-frozen-claim"
        await manager.submit_task({**_base_payload(task_id), "kernel_code": "old", "task_stage": "execute"})
        assert await manager.get_next_task("gpu-worker", resources=["gpu"]) is not None
        assert await manager.freeze_task_claim(task_id, "unsafe child still exists") is True

        task_key = f"{manager.task_prefix}{task_id}"
        claim = manager._task_claims[task_id]
        before_hash = await redis.hgetall(task_key)
        before_inflight = list(redis.lists[claim.inflight_queue])

        with pytest.raises(TaskRefreshConflictError, match="recovery is frozen"):
            await manager.submit_task(
                {
                    **_base_payload(task_id),
                    "kernel_code": "replacement",
                    "task_stage": "execute",
                    "force_refresh": True,
                }
            )

        assert await redis.hgetall(task_key) == before_hash
        assert redis.lists[claim.inflight_queue] == before_inflight
        assert redis.lists[manager.resource_queues["gpu"]] == []
        assert json.loads(before_hash[b"data"].decode())["kernel_code"] == "old"

    asyncio.run(scenario())


def test_task_result_cache_checks_request_hash(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

        await manager.complete_task(
            "same-task",
            {"task_id": "same-task", "compiled": True, "correctness": True},
            request_hash="hash-a",
        )

        assert await manager.get_task_result("same-task", expected_request_hash="hash-b") is None
        cached = await manager.get_task_result("same-task", expected_request_hash="hash-a")
        assert cached is not None
        assert cached["compiled"] is True

    asyncio.run(scenario())


def test_task_result_cache_rejects_legacy_result_without_request_hash(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

        await manager.complete_task("legacy-task", {"task_id": "legacy-task", "compiled": True})

        assert await manager.get_task_result("legacy-task", expected_request_hash="hash-a") is None
        assert await manager.get_task_result("legacy-task") is not None

    asyncio.run(scenario())


def test_terminal_task_and_result_records_expire(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        monkeypatch.setattr(task_manager_module.settings, "terminal_task_ttl_sec", 123)
        monkeypatch.setattr(task_manager_module.settings, "terminal_result_ttl_sec", 456)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

        await manager.submit_task(_base_payload("ttl-task"))
        await manager.complete_task("ttl-task", {"task_id": "ttl-task", "compiled": True})

        assert redis.expirations[f"{manager.task_prefix}ttl-task"] == 123
        assert redis.expirations[f"{manager.result_prefix}ttl-task"] == 456

    asyncio.run(scenario())


def test_cancel_task_removes_pending_queue_and_records_terminal_result(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        monkeypatch.setattr(task_manager_module.settings, "terminal_task_ttl_sec", 123)
        monkeypatch.setattr(task_manager_module.settings, "terminal_result_ttl_sec", 456)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

        await manager.submit_task(_base_payload("cancel-task"))
        assert await redis.llen(manager.resource_queues["gpu"]) == 1

        assert await manager.cancel_task("cancel-task") is True

        assert await redis.llen(manager.resource_queues["gpu"]) == 0
        task_hash = await redis.hgetall(f"{manager.task_prefix}cancel-task")
        result_hash = await redis.hgetall(f"{manager.result_prefix}cancel-task")
        assert task_hash[b"status"] == TaskStatus.FAILED.value.encode()
        assert b"cancelled_at" in task_hash
        assert result_hash[b"error"] == b"Task cancelled"
        assert redis.expirations[f"{manager.task_prefix}cancel-task"] == 123
        assert redis.expirations[f"{manager.result_prefix}cancel-task"] == 456
        assert redis.expirations[f"{manager.key_prefix}:cancel:cancel-task"] == manager._marker_ttl()

    asyncio.run(scenario())


def test_request_hash_ignores_identity_and_provenance_fields() -> None:
    base = {
        "task_id": "task-a",
        "reference_code": "reference",
        "kernel_code": "kernel",
        "force_refresh": False,
        "metadata": {"turn_id": 1, "line_index": 10, "model_id": "model-a"},
    }
    same_semantics = {
        **base,
        "task_id": "task-b",
        "force_refresh": True,
        "turn_id": 2,
        "line_index": 20,
        "model_id": "model-b",
        "metadata": {"turn_id": 9, "line_index": 99, "model_id": "model-c"},
    }
    changed_payload = {**base, "kernel_code": "different kernel"}

    assert request_hash("kernelbench", base) == request_hash("kernelbench", same_semantics)
    assert request_hash("kernelbench", base) != request_hash("kernelbench", changed_payload)


def test_kernelbench_child_task_preserves_force_refresh() -> None:
    _, kernel_task = _create_paired_tasks(
        EvaluationTask(
            task_id="parent",
            reference_code="class Model: pass",
            kernel_code="class Model: pass",
            force_refresh=True,
        )
    )

    assert kernel_task.force_refresh is True
    assert kernel_task.to_dict()["force_refresh"] is True
