"""Server worker-heartbeat route tests."""

import asyncio
from datetime import datetime, timedelta

from kernelgym.server.api import server


class FakeLoadBalancer:
    def __init__(self) -> None:
        self.available_workers = {"worker_gpu_0": {"device": "cuda:0"}}


class FakeTaskManager:
    def __init__(self) -> None:
        old = (datetime.now() - timedelta(minutes=10)).isoformat()
        self.worker_registry = {
            "worker_gpu_0": {
                "device": "cuda:0",
                "status": "online",
                "last_heartbeat": old,
                "node_id": "v1",
                "hostname": "node",
            }
        }
        self.worker_load_balancer = FakeLoadBalancer()
        self.updated = False

    async def get_worker_data(self, worker_id):
        return {}

    async def register_worker(self, worker_id, device, node_id=None, hostname=None):
        self.worker_registry[worker_id] = {
            "device": device,
            "status": "online",
            "last_heartbeat": datetime.now().isoformat(),
            "node_id": node_id or "",
            "hostname": hostname or "",
        }
        return True

    async def update_worker_heartbeat(self, worker_id, node_id=None, hostname=None):
        self.updated = True
        self.worker_registry[worker_id]["last_heartbeat"] = datetime.now().isoformat()
        self.worker_registry[worker_id]["status"] = "online"
        if node_id:
            self.worker_registry[worker_id]["node_id"] = node_id
        if hostname:
            self.worker_registry[worker_id]["hostname"] = hostname


def test_worker_heartbeat_updates_status_registry() -> None:
    task_manager = FakeTaskManager()
    before = task_manager.worker_registry["worker_gpu_0"]["last_heartbeat"]

    result = asyncio.run(
        server.worker_heartbeat(
            "worker_gpu_0",
            device="cuda:0",
            node_id="v1",
            hostname="node",
            task_manager=task_manager,
        )
    )

    assert result["success"] is True
    assert task_manager.updated is True
    assert task_manager.worker_registry["worker_gpu_0"]["last_heartbeat"] != before


def test_worker_heartbeat_backfills_missing_node_metadata() -> None:
    task_manager = FakeTaskManager()
    task_manager.worker_registry["worker_gpu_0"]["node_id"] = ""
    task_manager.worker_registry["worker_gpu_0"]["hostname"] = ""

    result = asyncio.run(
        server.worker_heartbeat(
            "worker_gpu_0",
            device="cuda:0",
            node_id="v1",
            hostname="ai-16-39",
            task_manager=task_manager,
        )
    )

    assert result["success"] is True
    assert task_manager.worker_registry["worker_gpu_0"]["node_id"] == "v1"
    assert task_manager.worker_registry["worker_gpu_0"]["hostname"] == "ai-16-39"
