import asyncio
import json
from collections import defaultdict
from typing import Any

from kernelgym.server import task_manager as task_manager_module
from kernelgym.server.task_manager import TaskManager
from kernelgym.common import TaskStatus


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.lists: dict[str, list[str]] = defaultdict(list)

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
