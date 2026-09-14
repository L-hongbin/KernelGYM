"""Parent lifecycle integration against a disposable, Unix-socket-only Redis."""

import asyncio
import json
import shutil
import subprocess
import time
import uuid

import pytest
import redis
import redis.asyncio as aioredis

from kernelgym.config import settings
from kernelgym.core.types import TaskSpec
from kernelgym.server.scheduler import TaskManagerScheduler
from kernelgym.server.task_manager import TaskManager, StaleTaskClaimError
from kernelgym.server.workflow_lifecycle import WorkflowConflictError, WorkflowStoppedError


@pytest.fixture(scope="module")
def redis_socket(tmp_path_factory):
    binary = shutil.which("redis-server")
    if not binary:
        pytest.skip("redis-server is required for real Lua lifecycle tests")
    directory = tmp_path_factory.mktemp("workflow-redis")
    socket = directory / "redis.sock"
    process = subprocess.Popen(
        [
            binary,
            "--port",
            "0",
            "--unixsocket",
            str(socket),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(directory),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        client = redis.Redis(unix_socket_path=str(socket), socket_timeout=1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if client.ping():
                    break
            except redis.ConnectionError:
                time.sleep(0.02)
        else:
            raise RuntimeError("isolated Redis failed to start")
        client.close()
        yield str(socket)
    finally:
        process.terminate()  # Only the disposable server owned by this fixture.
        process.wait(timeout=5)


@pytest.fixture
def run_case(redis_socket, monkeypatch, tmp_path):
    monkeypatch.setattr(type(settings), "redis_key_prefix", f"test_workflow_{uuid.uuid4().hex}")
    monkeypatch.setattr(type(settings), "redis_key_prefix_legacy", "")
    monkeypatch.setenv("KERNELGYM_SAFETY_LATCH_DIR", str(tmp_path / "latches"))

    def run(scenario):
        async def main():
            async with aioredis.Redis(unix_socket_path=redis_socket, socket_timeout=2) as client:
                manager = TaskManager(client)
                try:
                    await asyncio.wait_for(scenario(manager), timeout=15)
                finally:
                    await manager.shutdown()

        asyncio.run(main())

    return run


async def accept(manager, timeout=10):
    owner, record = await manager.workflows.accept("parent", "hash", "kernelbench", timeout, False)
    assert owner
    return record, TaskManagerScheduler(manager, poll_interval=0.01, workflow=record)


def child(stage="compile"):
    return TaskSpec(
        kind="kernelbench.kernel",
        payload={
            "task_id": f"parent_{stage}",
            "toolkit": "kernelbench",
            "backend_adapter": "kernelbench",
            "task_type": "kernel_evaluation",
            "required_resource": "cpu" if stage == "compile" else "gpu",
        },
    )


def success():
    return {
        "status": "completed",
        "compiled": True,
        "correctness": True,
        "error_message": None,
        "error_code": None,
        "metadata": {},
    }


def test_parent_exists_before_submit_and_lease_renews(run_case, monkeypatch):
    monkeypatch.setattr(settings, "workflow_lease_seconds", 3)

    async def scenario(manager):
        record, scheduler = await accept(manager)
        status = await manager.get_task_status("parent")
        assert status["status"] == "pending"
        assert set(status["children"]) == {"compile", "kernel", "ref"}
        keys = manager.workflows.keys("parent")
        assert await manager.redis.ttl(keys[0]) == -1
        await manager.redis.pexpire(keys[2], 300)
        assert await manager.workflows.renew("parent", record["workflow_generation"])
        assert await manager.redis.pttl(keys[2]) > 2000
        assert (await manager.get_task_status("parent"))["status"] == "processing"
        child_id = await scheduler.submit(child())
        assert child_id == status["children"]["compile"]
        assert await manager.redis.sismember(
            manager.workflows.children_key("parent", record["workflow_generation"]), child_id
        )
        assert await manager.workflows.finish("parent", record["workflow_generation"], success())
        assert not await manager.redis.exists(keys[2])
        assert (await manager.get_task_status(child_id))["status"] == "failed"

    run_case(scenario)


def test_atomic_accept_deduplicates_across_managers(run_case):
    async def scenario(manager):
        other = TaskManager(manager.redis)
        results = await asyncio.gather(
            *[tm.workflows.accept("parent", "hash", "kernelbench", 10, True) for tm in [manager, other] * 5]
        )
        assert sum(owner for owner, _ in results) == 1
        assert len({record["workflow_generation"] for _, record in results}) == 1
        with pytest.raises(WorkflowConflictError):
            await other.workflows.accept("parent", "different", "kernelbench", 10, False)

    run_case(scenario)


@pytest.mark.parametrize("cause", ["deadline", "lease", "cancel"])
def test_parent_stop_fences_submit_cpu_dispatch_and_late_completion(run_case, cause):
    async def scenario(manager):
        record, scheduler = await accept(manager)
        child_id = await scheduler.submit(child())
        key, _, payload = await manager._load_task_data(child_id)
        if cause == "deadline":
            await manager.redis.hset(manager.workflows.keys("parent")[0], mapping={"workflow_deadline": "1"})
        elif cause == "lease":
            await manager.redis.delete(manager.workflows.keys("parent")[2])
        else:
            assert await manager.cancel_task("parent")
        # Exercise the Lua dispatch gate directly, without first reconciling.
        assert not await manager._mark_cpu_processing(manager.key_prefix, key, child_id, payload)
        with pytest.raises(WorkflowStoppedError):
            await manager.submit_task({**payload, "task_id": "late_child"})
        state = await manager.get_task_status("parent")
        assert state["status"] == ("timeout" if cause == "deadline" else "failed")
        assert await manager.redis.llen(manager.resource_queues["cpu"]) == 0
        assert not await manager.workflows.finish("parent", record["workflow_generation"], success())
        assert (await manager.get_task_status("parent"))["status"] == state["status"]
        with pytest.raises(StaleTaskClaimError):
            await manager.complete_task("parent", success())

    run_case(scenario)


def test_cancel_preserves_frozen_gpu_claim_and_survives_marker_expiry(run_case):
    async def scenario(manager):
        record, scheduler = await accept(manager)
        child_id = await scheduler.submit(child("kernel"))
        key = f"{manager.task_prefix}{child_id}"
        frozen = {"status": "processing", "claim_token": "owner", "claim_recovery_state": "frozen"}
        await manager.redis.hset(key, mapping=frozen)
        assert await manager.cancel_task("parent")
        after = await manager.redis.hgetall(key)
        for field, value in frozen.items():
            assert after[field.encode()] == value.encode()
        await manager.redis.delete(manager._cancel_key("parent"))
        assert await manager.is_task_cancelled(child_id)
        assert await manager.is_task_cancelled("parent")

    run_case(scenario)


def test_gpu_claim_cannot_start_after_parent_deadline(run_case):
    async def scenario(manager):
        _, scheduler = await accept(manager)
        child_id = await scheduler.submit(child("kernel"))
        queue = manager.resource_queues["gpu"]
        assert await manager._claim_gpu_task(manager.key_prefix, "gpu_worker", queue) == child_id
        key, _, payload = await manager._load_task_data(child_id)
        await manager.redis.hset(manager.workflows.keys("parent")[0], mapping={"workflow_deadline": "1"})
        assert not await manager._mark_claim_processing(
            prefix=manager.key_prefix,
            task_id=child_id,
            task_key=key,
            task_json=payload,
            worker_id="gpu_worker",
            inflight_queue=manager._gpu_inflight_queue(manager.key_prefix, "gpu_worker"),
            source_queue=queue,
        )
        assert (await manager.redis.hgetall(key))[b"claim_recovery_state"] == b""

    run_case(scenario)


def test_refresh_fences_old_children_and_old_parent_owner(run_case):
    async def scenario(manager):
        old, scheduler = await accept(manager)
        old_child = await scheduler.submit(child())
        await manager.workflows.finish("parent", old["workflow_generation"], success())
        owner, new = await manager.workflows.accept("parent", "hash", "kernelbench", 10, True)
        assert owner and new["workflow_generation"] != old["workflow_generation"]
        current = TaskManagerScheduler(manager, workflow=new)
        new_child = await current.submit(child())
        assert old_child != new_child
        assert await manager.is_task_cancelled(old_child)
        assert not await manager.workflows.finish("parent", old["workflow_generation"], success())
        assert (await manager.get_task_status("parent"))["status"] == "pending"

    run_case(scenario)


def test_sync_post_deduplicates_and_status_is_queryable(run_case, monkeypatch):
    from kernelgym.server.api import server

    async def scenario(manager):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        class Controller:
            async def handle_request(self, payload, scheduler):
                nonlocal calls
                calls += 1
                entered.set()
                await release.wait()
                return success()

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: Controller())
        first = asyncio.create_task(server._execute_workflow(manager, "kernelbench", {"task_id": "parent"}))
        await entered.wait()
        second = asyncio.create_task(server._execute_workflow(manager, "kernelbench", {"task_id": "parent"}))
        state = await server.get_task_status("parent", manager)
        assert state.status.value in {"pending", "processing"}
        await asyncio.sleep(0.05)
        assert calls == 1
        release.set()
        a, b = await asyncio.gather(first, second)
        assert a[1] == b[1]
        assert a[1]["task_id"] == "parent"

    run_case(scenario)


def test_queue_time_counts_towards_parent_deadline(run_case, monkeypatch):
    from kernelgym.server.api import server

    async def scenario(manager):
        class Controller:
            async def handle_request(self, payload, scheduler):
                task_id = await scheduler.submit(child())
                return await scheduler.wait(task_id)

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: Controller())
        _, result, status = await server._execute_workflow(
            manager, "kernelbench", {"task_id": "parent", "workflow_timeout": 0.1}
        )
        assert status.value == "timeout"
        assert result["error_code"] == "TIMEOUT_ERROR"
        assert await manager.redis.llen(manager.resource_queues["cpu"]) == 0
        print(json.dumps({"queued_workflow_result": result, "status": await manager.get_task_status("parent")}))

    run_case(scenario)


def test_disconnect_does_not_cancel_shared_workflow(run_case, monkeypatch):
    from kernelgym.server.api import server

    async def scenario(manager):
        entered, release = asyncio.Event(), asyncio.Event()

        class Controller:
            async def handle_request(self, payload, scheduler):
                entered.set()
                await release.wait()
                return success()

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: Controller())
        waiter = asyncio.create_task(server._execute_workflow(manager, "kernelbench", {"task_id": "parent"}))
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert manager._workflow_tasks
        release.set()
        _, result, _ = await server._execute_workflow(manager, "kernelbench", {"task_id": "parent"})
        assert result["status"] == "completed"

    run_case(scenario)


def test_controller_exception_and_shutdown_commit_terminal_state(run_case, monkeypatch):
    from kernelgym.server.api import server

    async def scenario(manager):
        class BrokenController:
            async def handle_request(self, payload, scheduler):
                raise ValueError("controlled failure")

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: BrokenController())
        _, result, _ = await server._execute_workflow(manager, "kernelbench", {"task_id": "parent"})
        assert result["status"] == "failed"
        assert "controlled failure" in result["error_message"]

        entered = asyncio.Event()

        class WaitingController:
            async def handle_request(self, payload, scheduler):
                entered.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: WaitingController())
        request = asyncio.create_task(server._execute_workflow(manager, "kernelbench", {"task_id": "shutdown"}))
        await entered.wait()
        await manager.shutdown()
        _, result, _ = await request
        assert result["status"] == "failed"
        assert "shutdown" in result["error_message"]
        assert not await manager.redis.exists(manager.workflows.keys("shutdown")[2])

    run_case(scenario)


def test_watchdog_terminalizes_abandoned_owner(run_case):
    async def scenario(manager):
        await accept(manager)
        await manager.redis.delete(manager.workflows.keys("parent")[2])
        watcher = asyncio.create_task(manager.workflows.watch())
        try:
            for _ in range(100):
                result = await manager.get_task_result("parent")
                if result:
                    break
                await asyncio.sleep(0.01)
            assert result["status"] == "failed"
            assert "lease expired" in result["error_message"]
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    run_case(scenario)


def test_terminal_retention_starts_after_completion(run_case, monkeypatch):
    monkeypatch.setattr(settings, "terminal_task_ttl_sec", 17)
    monkeypatch.setattr(settings, "terminal_result_ttl_sec", 19)

    async def scenario(manager):
        record, _ = await accept(manager)
        keys = manager.workflows.keys("parent")
        assert await manager.redis.ttl(keys[0]) == -1
        await manager.workflows.finish("parent", record["workflow_generation"], success())
        assert 0 < await manager.redis.ttl(keys[0]) <= 17
        assert 0 < await manager.redis.ttl(keys[1]) <= 19
        assert not await manager.redis.sismember(keys[4], "parent")

    run_case(scenario)


def test_ephemeral_cleanup_keeps_frozen_child_ownership(run_case):
    async def scenario(manager):
        _, scheduler = await accept(manager)
        child_id = await scheduler.submit(child("kernel"))
        child_key = f"{manager.task_prefix}{child_id}"
        await manager.redis.hset(
            child_key, mapping={"status": "processing", "claim_token": "owner", "claim_recovery_state": "frozen"}
        )
        await manager.cancel_task("parent")
        await manager.discard_task_records(["parent", child_id])
        assert (await manager.redis.hgetall(child_key))[b"claim_token"] == b"owner"
        assert await manager.is_task_cancelled(child_id)

    run_case(scenario)


def test_http_status_and_cancel_during_sync_post(run_case, monkeypatch):
    import httpx
    from kernelgym.server.api import server

    async def scenario(manager):
        entered = asyncio.Event()

        class Controller:
            async def handle_request(self, payload, scheduler):
                entered.set()
                child_id = await scheduler.submit(child())
                return await scheduler.wait(child_id)

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: Controller())
        server.app.dependency_overrides[server.get_task_manager] = lambda: manager
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                request = asyncio.create_task(
                    client.post(
                        "/workflow/submit",
                        json={"workflow": "kernelbench", "task_id": "parent", "payload": {"task_id": "parent"}},
                    )
                )
                await entered.wait()
                response = await client.get("/status/parent")
                assert response.status_code == 200
                assert response.json()["status"] in {"pending", "processing"}
                assert response.json()["children"]["compile"].startswith("parent_compile_")
                assert (await client.delete("/tasks/parent")).status_code == 200
                submitted = await request
                assert submitted.status_code == 200
                assert submitted.json()["result"]["error_message"] == "Task cancelled"
                assert (await client.get("/status/parent")).json()["status"] == "failed"
        finally:
            server.app.dependency_overrides.pop(server.get_task_manager, None)

    run_case(scenario)


def test_real_kernelbench_controller_uses_generation_scoped_child(run_case):
    from kernelgym.server.api import server

    async def scenario(manager):
        request = asyncio.create_task(
            server._execute_workflow(
                manager,
                "kernelbench",
                {
                    "task_id": "parent",
                    "backend": "triton",
                    "use_reference_cache": False,
                    "reference_code": "class Model:\n    def forward(self, x): return x\n",
                    "kernel_code": "class ModelNew:\n    def forward(self, x): return x\n",
                },
            )
        )
        for _ in range(200):
            queued = await manager.redis.lrange(manager.resource_queues["gpu"], 0, -1)
            if queued or request.done():
                break
            await asyncio.sleep(0.01)
        assert queued, request.result() if request.done() else "controller did not submit a kernel child"
        child_id = queued[0].decode()
        parent = await manager.get_task_status("parent")
        assert parent["children"]["kernel"] == child_id
        # Synthetic worker feedback through the real TaskManager/controller;
        # no candidate Python or CUDA code is executed in this test.
        await manager.complete_task(
            child_id,
            {
                "task_id": child_id,
                "base_task_id": "parent",
                "compiled": True,
                "correctness": False,
                "decoy_kernel": False,
                "kernel_runtime": -1.0,
                "metadata": {},
                "status": "completed",
                "error_message": None,
                "error_code": None,
            },
        )
        _, result, _ = await request
        assert result["task_id"] == "parent"
        assert result["compiled"] is True and result["correctness"] is False
        assert result["error_message"] is None
        assert (await manager.get_task_status("parent"))["status"] == "completed"

    run_case(scenario)


def test_parent_cancel_finalizes_claimed_but_unstarted_child(run_case):
    async def scenario(manager):
        _, scheduler = await accept(manager)
        child_id = await scheduler.submit(child("kernel"))
        queue = manager.resource_queues["gpu"]
        assert await manager._claim_gpu_task(manager.key_prefix, "gpu_worker", queue) == child_id
        assert await manager.cancel_task("parent")
        key, data, payload = await manager._load_task_data(child_id)
        assert data[b"status"] == b"failed"
        assert data[b"claim_token"] == b""
        assert not await manager._mark_claim_processing(
            prefix=manager.key_prefix,
            task_id=child_id,
            task_key=key,
            task_json=payload,
            worker_id="gpu_worker",
            inflight_queue=manager._gpu_inflight_queue(manager.key_prefix, "gpu_worker"),
            source_queue=queue,
        )

    run_case(scenario)


def test_ephemeral_cleanup_cas_does_not_delete_new_parent_generation(run_case, monkeypatch):
    from kernelgym.server.workflow_lifecycle import DISCARD_WORKFLOW_RECORDS_LUA

    async def scenario(manager):
        record, _ = await accept(manager)
        await manager.workflows.finish("parent", record["workflow_generation"], success())
        original_eval = manager.redis.eval

        async def race(script, *args):
            if script == DISCARD_WORKFLOW_RECORDS_LUA:
                await manager.redis.hset(
                    manager.workflows.keys("parent")[0],
                    mapping={"workflow_generation": "replacement", "status": "pending"},
                )
            return await original_eval(script, *args)

        monkeypatch.setattr(manager.redis, "eval", race)
        assert await manager.discard_task_records(["parent"]) == 0
        assert (await manager.workflows.read("parent"))["workflow_generation"] == "replacement"

    run_case(scenario)


def test_terminal_missing_result_fails_instead_of_waiting_forever(run_case):
    async def scenario(manager):
        record, _ = await accept(manager)
        await manager.workflows.finish("parent", record["workflow_generation"], success())
        await manager.redis.delete(manager.workflows.keys("parent")[1])
        with pytest.raises(WorkflowConflictError, match="retained result is unavailable"):
            await manager.workflows.wait("parent", record["workflow_generation"])

    run_case(scenario)


def test_accept_does_not_overwrite_legacy_active_workflow_marker(run_case):
    async def scenario(manager):
        lease_key = manager.workflows.keys("parent")[2]
        await manager.redis.set(lease_key, "1", ex=60)
        with pytest.raises(WorkflowConflictError):
            await manager.workflows.accept("parent", "hash", "kernelbench", 10, True)
        assert await manager.redis.get(lease_key) == b"1"

    run_case(scenario)


@pytest.mark.parametrize("desired_status", ["pending", "processing"])
def test_live_cancel_probe_discovers_actual_child_id(monkeypatch, desired_status):
    from scripts import test_cancel

    calls = []

    def get(url, timeout):
        calls.append(url)
        if url.endswith("/status/parent"):
            return 200, {"status": "processing", "children": {"kernel": "parent_kernel_generation"}}
        assert url.endswith("/status/parent_kernel_generation")
        return 200, {"status": desired_status}

    monkeypatch.setattr(test_cancel, "_get", get)
    child_id, _ = test_cancel._wait_subtask_status("http://test", "parent", desired_status=desired_status)
    assert child_id == "parent_kernel_generation"
    assert len(calls) == 2


@pytest.mark.parametrize("force", [False, True])
def test_cancel_before_post_rejects_parent_and_delayed_children(run_case, force):
    from fastapi import HTTPException
    from kernelgym.server.api import server

    async def scenario(manager):
        assert await manager.cancel_task("parent")
        assert await manager.cancel_task("parent")  # Idempotent even before acceptance.
        assert await manager.get_task_status("parent") is None  # 404 is not an execution fence.
        with pytest.raises(HTTPException) as error:
            await server._execute_workflow(manager, "kernelbench", {"task_id": "parent"}, force_refresh=force)
        assert error.value.status_code == 409
        with pytest.raises(WorkflowConflictError, match="cancelled"):
            await manager.workflows.accept("parent", "hash", "kernelbench", 10, force)
        for stage in ("compile", "kernel", "ref"):
            with pytest.raises(WorkflowStoppedError):
                await manager.submit_task({**child(stage).payload, "force_refresh": force})
            with pytest.raises(WorkflowStoppedError):
                await manager.submit_task(
                    {
                        **child(stage).payload,
                        "task_id": f"delayed_{stage}",
                        "base_task_id": "parent",
                        "force_refresh": force,
                    }
                )
        assert await manager.redis.llen(manager.resource_queues["gpu"]) == 0
        assert await manager.redis.llen(manager.resource_queues["cpu"]) == 0
        assert not await manager.redis.exists(manager.workflows.keys("parent")[0])
        assert await manager.redis.ttl(manager._tombstone_key("parent")) == -1

    run_case(scenario)


@pytest.mark.parametrize("stage", ["compile", "kernel", "ref"])
@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("cancel_parent", [False, True])
def test_each_stage_cancel_preserves_running_fences_and_stops_parent(run_case, stage, running, cancel_parent):
    async def scenario(manager):
        record, scheduler = await accept(manager)
        child_id = await scheduler.submit(child(stage))
        key, _, payload = await manager._load_task_data(child_id)
        if running:
            if stage == "compile":
                assert await manager._mark_cpu_processing(manager.key_prefix, key, child_id, payload)
            else:
                queue = manager.resource_queues["gpu"]
                assert await manager._claim_gpu_task(manager.key_prefix, "gpu_worker", queue) == child_id
                assert await manager._mark_claim_processing(
                    prefix=manager.key_prefix,
                    task_id=child_id,
                    task_key=key,
                    task_json=payload,
                    worker_id="gpu_worker",
                    inflight_queue=manager._gpu_inflight_queue(manager.key_prefix, "gpu_worker"),
                    source_queue=queue,
                )
        before = await manager.redis.hgetall(key)
        assert await manager.cancel_task("parent" if cancel_parent else child_id)
        result = await asyncio.wait_for(manager.workflows.wait("parent", record["workflow_generation"]), 1)
        assert result["status"] == "failed" and "cancelled" in result["error_message"]
        after = await manager.redis.hgetall(key)
        if running:
            assert after[b"status"] == b"processing"
            for field in (b"claim_token", b"claim_recovery_state", b"claim_inflight_queue"):
                assert after.get(field) == before.get(field)
        else:
            assert after[b"status"] == b"failed"
        with pytest.raises(WorkflowStoppedError):
            await scheduler.submit(child("ref"))
        assert not await manager.workflows.finish("parent", record["workflow_generation"], success())

    run_case(scenario)


def test_tombstones_survive_cleanup_and_block_force_refresh(run_case):
    async def scenario(manager):
        record, scheduler = await accept(manager)
        child_id = await scheduler.submit(child())
        await manager.cancel_task("parent")
        await manager.discard_task_records(["parent", child_id])
        assert await manager.redis.ttl(manager._tombstone_key("parent")) == -1
        assert await manager.redis.ttl(manager._tombstone_key(child_id)) == -1
        with pytest.raises(WorkflowConflictError, match="cancelled"):
            await manager.workflows.accept("parent", "hash", "kernelbench", 10, True)
        with pytest.raises(WorkflowStoppedError):
            await manager.submit_task({**child().payload, "task_id": child_id, "force_refresh": True})
        assert await manager.is_task_cancelled("parent")
        assert await manager.is_task_cancelled(child_id)

    run_case(scenario)


@pytest.mark.parametrize("stage", ["compile", "kernel", "ref"])
def test_cancel_child_alias_before_generation_submit(run_case, stage):
    async def scenario(manager):
        _, scheduler = await accept(manager)
        await manager.cancel_task(f"parent_{stage}")
        with pytest.raises(WorkflowStoppedError):
            await scheduler.submit(child(stage))

    run_case(scenario)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("heartbeat_present", [False, True])
def test_quarantined_child_ends_http_wait_without_releasing_containment(
    run_case, monkeypatch, frozen, heartbeat_present
):
    from kernelgym.server.api import server
    from kernelgym.utils.gpu_quarantine import write_gpu_quarantine, read_gpu_quarantine

    async def scenario(manager):
        submitted = asyncio.Future()
        stages = []

        class Controller:
            async def handle_request(self, payload, scheduler):
                task_id = await scheduler.submit(child("kernel"))
                stages.append("kernel")
                submitted.set_result(task_id)
                await scheduler.wait(task_id)
                stages.append("ref")
                return success()

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: Controller())
        request = asyncio.create_task(server._execute_workflow(manager, "kernelbench", {"task_id": "parent"}))
        child_id = await submitted
        key = f"{manager.task_prefix}{child_id}"
        queue = manager.resource_queues["gpu"]
        assert await manager._claim_gpu_task(manager.key_prefix, "gpu_worker", queue) == child_id
        _, _, payload = await manager._load_task_data(child_id)
        assert await manager._mark_claim_processing(
            prefix=manager.key_prefix,
            task_id=child_id,
            task_key=key,
            task_json=payload,
            worker_id="gpu_worker",
            inflight_queue=manager._gpu_inflight_queue(manager.key_prefix, "gpu_worker"),
            source_queue=queue,
        )
        await manager.redis.hset(
            f"{manager.worker_prefix}gpu_worker", mapping={"device": "cuda:0", "hostname": "test-host"}
        )
        await write_gpu_quarantine(
            manager.redis,
            "gpu_worker",
            device="cuda:0",
            hostname="test-host",
            reason="containment uncertain",
            fault_class="test",
        )
        if frozen:
            await manager.freeze_task_claim(child_id, "containment uncertain")
        if not heartbeat_present:
            await manager.redis.delete(f"{manager.worker_prefix}gpu_worker")
        before = await manager.redis.hgetall(key)
        claim = manager._task_claims[child_id]
        inflight = await manager.redis.lrange(claim.inflight_queue, 0, -1)
        _, result, _ = await asyncio.wait_for(request, 2)
        assert result["status"] == "failed"
        assert result["error_code"] == "SYSTEM_ERROR"
        assert "Infrastructure failure" in result["error_message"]
        assert ("frozen" if frozen else "quarantined") in result["error_message"]
        assert stages == ["kernel"]
        assert await manager.get_task_result(child_id) is None
        assert await manager.redis.hgetall(key) == before
        assert await manager.redis.lrange(claim.inflight_queue, 0, -1) == inflight
        assert await read_gpu_quarantine(manager.redis, "gpu_worker", device="cuda:0", hostname="test-host")
        print(
            json.dumps(
                {
                    "result": result,
                    "fence": before[b"claim_recovery_state"].decode(),
                    "child_result": None,
                    "inflight_preserved": True,
                }
            )
        )

    run_case(scenario)


def test_workflow_renews_beyond_accelerated_old_ttl_and_remains_cancellable(run_case, monkeypatch):
    # Accelerate the former one-shot TTL to 0.3s; span four full TTL windows.
    monkeypatch.setattr(settings, "workflow_lease_seconds", 0.3)

    async def scenario(manager):
        record, _ = await accept(manager)
        task = asyncio.create_task(manager.workflows.run("parent", record, lambda: asyncio.sleep(10)))
        manager._workflow_tasks.add(task)
        await asyncio.sleep(1.2)
        status = await manager.get_task_status("parent")
        assert status["status"] == "processing"
        assert await manager.redis.pttl(manager.workflows.keys("parent")[2]) > 0
        assert await manager.cancel_task("parent")
        result = await manager.workflows.wait("parent", record["workflow_generation"])
        assert result["error_message"] == "Task cancelled"
        await asyncio.wait_for(task, 1)

    run_case(scenario)


def test_success_cache_compatible_but_explicit_cancel_blocks_cached_post(run_case, monkeypatch):
    from fastapi import HTTPException
    from kernelgym.server.api import server

    async def scenario(manager):
        calls = 0

        class Controller:
            async def handle_request(self, payload, scheduler):
                nonlocal calls
                calls += 1
                return success()

        monkeypatch.setattr(server, "get_workflow_controller", lambda name: Controller())
        first = await server._execute_workflow(manager, "kernelbench", {"task_id": "parent"})
        assert await server._execute_workflow(manager, "kernelbench", {"task_id": "parent"}) == first
        assert calls == 1
        await manager.cancel_task("parent")
        for force in (False, True):
            with pytest.raises(HTTPException) as error:
                await server._execute_workflow(manager, "kernelbench", {"task_id": "parent"}, force_refresh=force)
            assert error.value.status_code == 409
        assert calls == 1
        assert (await manager.get_task_result("parent"))["correctness"] is True

    run_case(scenario)


def test_cancelled_alias_of_queued_child_ends_parent(run_case):
    async def scenario(manager):
        record, scheduler = await accept(manager)
        await scheduler.submit(child("kernel"))
        await manager.cancel_task("parent_kernel")
        result = await asyncio.wait_for(manager.workflows.wait("parent", record["workflow_generation"]), 1)
        assert "cancelled" in result["error_message"]
        assert await manager.redis.llen(manager.resource_queues["gpu"]) == 0

    run_case(scenario)


def test_cancel_and_submit_race_never_leaves_dispatchable_work(run_case):
    async def scenario(manager):
        for index in range(20):
            task_id = f"race_{index}"
            submit = manager.submit_task({**child("kernel").payload, "task_id": task_id})
            cancel = manager.cancel_task(task_id)
            actions = [submit, cancel] if index % 2 else [cancel, submit]
            results = await asyncio.gather(*actions, return_exceptions=True)
            assert all(
                not isinstance(result, Exception) or isinstance(result, WorkflowStoppedError) for result in results
            )
            assert await manager.is_task_cancelled(task_id)
            assert await manager.redis.ttl(manager._tombstone_key(task_id)) == -1
        assert await manager.redis.llen(manager.resource_queues["gpu"]) == 0
        assert await manager._claim_gpu_task(manager.key_prefix, "gpu_worker", manager.resource_queues["gpu"]) is None

    run_case(scenario)


def test_cancel_storage_failure_is_not_acknowledged(run_case, monkeypatch):
    from fastapi import HTTPException
    from kernelgym.server.api import server
    from kernelgym.server.workflow_lifecycle import CANCEL_TASK_LUA

    async def scenario(manager):
        original = manager.redis.eval

        async def unavailable(script, *args):
            if script == CANCEL_TASK_LUA:
                raise redis.ConnectionError("isolated failure")
            return await original(script, *args)

        monkeypatch.setattr(manager.redis, "eval", unavailable)
        with pytest.raises(HTTPException) as error:
            await server.cancel_task("parent", manager)
        assert error.value.status_code == 500

    run_case(scenario)


def test_quarantine_business_read_never_waits_for_recovery_lock_or_rehydrates(run_case, monkeypatch):
    from kernelgym.utils import gpu_quarantine as quarantine

    async def scenario(manager):
        await quarantine.write_gpu_quarantine(
            manager.redis, "gpu_worker", device="cuda:0", hostname="test-host", reason="uncertain", fault_class="test"
        )
        keys = [
            quarantine.gpu_quarantine_key("gpu_worker"),
            quarantine.gpu_device_quarantine_key("test-host", "cuda:0"),
        ]
        await manager.redis.delete(*keys)  # Simulate missing Redis replicas, retain the isolated durable latch.

        def forbid_lock(*args, **kwargs):
            raise AssertionError("Business status must not acquire physical recovery lock")

        monkeypatch.setattr(quarantine, "_acquire_device_lock", forbid_lock)
        result = await quarantine.read_gpu_quarantine(
            manager.redis, "gpu_worker", device="cuda:0", hostname="test-host", read_only=True
        )
        assert result
        assert not await manager.redis.exists(*keys)

    run_case(scenario)
