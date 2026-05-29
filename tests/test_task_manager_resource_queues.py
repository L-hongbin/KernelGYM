import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Any

from kernelgym.server import task_manager as task_manager_module
from kernelgym.server.task_manager import TaskManager
from kernelgym.server.request_hash import request_hash
from kernelgym.common import TaskStatus
from kernelgym.schema.task import EvaluationTask
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.lists: dict[str, list[str]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)

    @staticmethod
    def _bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        return str(value).encode()

    async def exists(self, key: str) -> bool:
        return key in self.hashes

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
        return removed

    async def scan_iter(self, pattern: str):
        import fnmatch

        for key in sorted(self.hashes):
            if fnmatch.fnmatch(key, pattern):
                yield key

    async def incr(self, key: str) -> int:
        self.counters[key] += 1
        return self.counters[key]


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


def test_task_manager_routes_compile_and_execute_by_resource(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

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


def test_task_manager_honors_required_node_affinity(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]
        await manager.register_worker("gpu-node-a", "cuda:0", node_id="node-a", hostname="host-a")
        await manager.register_worker("gpu-node-b", "cuda:0", node_id="node-b", hostname="host-b")

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
        await manager.register_worker("gpu-busy", "cuda:0", node_id="node-a", hostname="host-a")
        await manager.register_worker("gpu-idle", "cuda:1", node_id="node-b", hostname="host-b")
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
        await manager.register_worker("gpu-a", "cuda:0", node_id="node-a", hostname="host-a")
        await manager.register_worker("gpu-b", "cuda:1", node_id="node-b", hostname="host-b")
        await redis.hset(f"{manager.worker_prefix}gpu-a", mapping={"current_task": "task-a"})
        await redis.hset(f"{manager.worker_prefix}gpu-b", mapping={"current_task": "task-b"})

        task_id = "parallel_task_001692_bf77c294_kernel"
        selected = await manager.select_worker_by_task_id("gpu", task_id)

        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        expected_worker_id = ["gpu-a", "gpu-b"][int.from_bytes(digest[:8], "big") % 2]
        assert selected is not None
        assert selected["worker_id"] == expected_worker_id

    asyncio.run(scenario())


def test_task_manager_preserves_direct_worker_queue(monkeypatch) -> None:
    async def scenario() -> None:
        _patch_registry(monkeypatch)
        redis = FakeRedis()
        manager = TaskManager(redis)  # type: ignore[arg-type]

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
