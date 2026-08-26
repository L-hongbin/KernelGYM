"""GPU-worker quarantine and safety-latch tests."""

from __future__ import annotations

import asyncio
import fnmatch
import socket
import stat
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("redis")
pytest.importorskip("aiohttp")

from kernelgym.utils.page_user_notifier import PageUserNotificationOutcome
from kernelgym.utils import gpu_quarantine as quarantine_module
from kernelgym.utils import page_user_notifier as notifier_module
from kernelgym.worker import gpu_worker as gpu_worker_module
from kernelgym.worker.gpu_worker import GPUWorker
from kernelgym.config import settings
from kernelgym.utils.gpu_quarantine import (
    acquire_gpu_quarantine_notification_claim,
    clear_gpu_quarantine,
    finish_gpu_quarantine_notification_claim,
    gpu_device_quarantine_key,
    gpu_quarantine_generation,
    gpu_quarantine_key,
    read_gpu_quarantine,
    release_gpu_quarantine_notification_claim,
    update_gpu_quarantine_notification,
    write_gpu_quarantine,
)
from scripts import manage_gpu_quarantine


@pytest.fixture(autouse=True)
def _isolated_latch_dir(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setenv("KERNELGYM_SAFETY_LATCH_DIR", str(tmp_path / "safety_latches"))

    async def fake_page(record):  # noqa: ANN001, ARG001
        return PageUserNotificationOutcome(True, protocol_version="mock")

    # No quarantine test may ever send a real page.  Individual tests replace
    # this fake when they need to inspect delivery or failure behavior.
    monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", fake_page)


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.pipeline_error: Exception | None = None
        self.pipeline_executions = 0

    @staticmethod
    def _bytes(value):  # noqa: ANN001
        return value if isinstance(value, bytes) else str(value).encode()

    async def hgetall(self, key):  # noqa: ANN001
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping):  # noqa: ANN001
        target = self.hashes.setdefault(key, {})
        for field, value in mapping.items():
            target[self._bytes(field)] = self._bytes(value)
        return 1

    async def expire(self, key, ttl):  # noqa: ANN001, ARG002
        return key in self.hashes

    async def delete(self, *keys):  # noqa: ANN002
        return sum(int(self.hashes.pop(key, None) is not None) for key in keys)

    async def scan_iter(self, match=None, count=None):  # noqa: ANN001, ARG002
        for key in list(self.hashes):
            if match is None or fnmatch.fnmatch(key, match):
                yield key.encode()

    def pipeline(self, transaction=True):  # noqa: ANN001, ARG002
        return FakePipeline(self)

    async def aclose(self) -> None:
        return None


class BlockingHsetRedis(FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.block_hsets = False
        self.hset_started = asyncio.Event()
        self.allow_hset = asyncio.Event()

    async def hset(self, key, mapping):  # noqa: ANN001
        if self.block_hsets:
            self.block_hsets = False
            self.hset_started.set()
            await self.allow_hset.wait()
        return await super().hset(key, mapping)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.keys: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
        return False

    def hgetall(self, key):  # noqa: ANN001
        self.keys.append(key)
        return self

    async def execute(self):
        self.redis.pipeline_executions += 1
        if self.redis.pipeline_error is not None:
            raise self.redis.pipeline_error
        return [await self.redis.hgetall(key) for key in self.keys]


class QuarantinedPool:
    accepting_tasks = False

    @staticmethod
    def get_health_snapshot():
        return {
            "health_state": "quarantined",
            "accepting_tasks": False,
            "health_reason": "fresh context cuInit failed",
            "health_task_id": "fault-task",
        }


def test_pool_quarantine_is_persisted_and_admission_fails_closed() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        worker.running = True
        worker.worker_pool = QuarantinedPool()  # type: ignore[assignment]

        assert await worker._gpu_admission_allowed() is False
        latch = await redis.hgetall(gpu_quarantine_key("node21_gpu_0"))
        assert latch[b"state"] == b"quarantined"
        assert latch[b"manual_clear_required"] == b"true"
        assert latch[b"task_id"] == b"fault-task"
        device_latch = await redis.hgetall(gpu_device_quarantine_key(socket.gethostname(), "cuda:0"))
        assert device_latch[b"state"] == b"quarantined"

        heartbeat = await redis.hgetall(f"{settings.redis_key_prefix}:worker:node21_gpu_0")
        assert heartbeat[b"health_state"] == b"quarantined"
        assert heartbeat[b"accepting_tasks"] == b"false"

    asyncio.run(scenario())


def test_device_latch_blocks_a_different_worker_id_on_same_gpu() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        await redis.hset(
            gpu_device_quarantine_key(socket.gethostname(), "cuda:0"),
            mapping={"state": "quarantined", "reason": "physical GPU latch"},
        )
        worker = GPUWorker("replacement_name_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        worker.running = True

        assert await worker._gpu_admission_allowed() is False
        assert worker.health_state == "quarantined"
        assert worker.quarantine_reason == "physical GPU latch"

    asyncio.run(scenario())


def test_scope_less_redis_latch_is_materialized_as_physical_and_paged_once(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        pages: list[dict[str, str]] = []

        async def capture_page(record):  # noqa: ANN001
            pages.append(dict(record))
            return PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", capture_page)
        await redis.hset(
            gpu_quarantine_key("node21_gpu_0"),
            mapping={"state": "quarantined", "reason": "persisted driver failure"},
        )
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        worker.running = True
        worker.worker_pool = None

        assert await worker._gpu_admission_allowed() is False
        assert await worker._gpu_admission_allowed() is False
        assert worker.health_state == "quarantined"
        assert worker.quarantine_reason == "persisted driver failure"
        assert len(pages) == 1
        assert pages[0]["scope"] == "physical_gpu"
        assert pages[0]["worker_id"] == worker.worker_id
        assert pages[0]["hostname"] == socket.gethostname()
        assert pages[0]["device"] == worker.device

        stored = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert stored is not None
        assert stored["scope"] == "physical_gpu"
        assert stored["page_user_state"] == "sent"
        device_latch = await redis.hgetall(gpu_device_quarantine_key(socket.gethostname(), worker.device))
        assert device_latch[b"scope"] == b"physical_gpu"

    asyncio.run(scenario())


def test_durable_device_latch_survives_redis_nosave_restart() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        await write_gpu_quarantine(
            redis,
            "old_worker_id",
            device="cuda:0",
            reason="cuInit failed",
            fault_class="initialization_failure",
            hostname=hostname,
        )

        # Model the primary service's Redis SHUTDOWN NOSAVE behavior.
        redis.hashes.clear()
        recovered = await read_gpu_quarantine(
            redis,
            "new_worker_id",
            device="cuda:0",
            hostname=hostname,
        )
        assert recovered is not None
        assert recovered["reason"] == "cuInit failed"
        assert await redis.hgetall(gpu_quarantine_key("new_worker_id"))

        assert await clear_gpu_quarantine(
            redis,
            "new_worker_id",
            device="cuda:0",
            hostname=hostname,
        )
        assert (
            await read_gpu_quarantine(
                redis,
                "another_worker_id",
                device="cuda:0",
                hostname=hostname,
            )
            is None
        )
        assert (
            await read_gpu_quarantine(
                redis,
                "old_worker_id",
                device="cuda:0",
                hostname=hostname,
            )
            is None
        )

    asyncio.run(scenario())


def test_stale_read_cannot_rehydrate_after_manual_clear(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = "node21"
        device = "cuda:0"
        await write_gpu_quarantine(
            redis,
            "old_worker_id",
            device=device,
            reason="persistent device fault",
            fault_class="device_fault",
            hostname=hostname,
        )
        # Force the reader down the durable-to-Redis rehydration path.
        redis.hashes.clear()

        first_snapshot_read = asyncio.Event()
        allow_reader_to_continue = asyncio.Event()
        original_read = quarantine_module._read_merged_quarantine
        read_count = 0

        async def delayed_first_read(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal read_count
            result = await original_read(*args, **kwargs)
            read_count += 1
            if read_count == 1:
                first_snapshot_read.set()
                await allow_reader_to_continue.wait()
            return result

        monkeypatch.setattr(quarantine_module, "_read_merged_quarantine", delayed_first_read)
        read_task = asyncio.create_task(
            read_gpu_quarantine(
                redis,
                "replacement_worker_id",
                device=device,
                hostname=hostname,
            )
        )
        await asyncio.wait_for(first_snapshot_read.wait(), timeout=1)

        # Clear wins the physical-GPU mutation lock after the reader captured
        # its old snapshot but before that reader attempts any Redis HSET.
        assert await clear_gpu_quarantine(
            redis,
            "replacement_worker_id",
            device=device,
            hostname=hostname,
        )
        allow_reader_to_continue.set()

        assert await asyncio.wait_for(read_task, timeout=1) is None
        assert not redis.hashes
        assert read_count == 2

    asyncio.run(scenario())


def test_redis_read_failure_without_durable_latch_fails_closed() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        redis.pipeline_error = RuntimeError("redis unavailable")

        with pytest.raises(RuntimeError, match="redis unavailable"):
            await read_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )

    asyncio.run(scenario())


def test_empty_quarantine_reads_are_not_negatively_cached() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        assert (
            await read_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
            is None
        )
        await redis.hset(
            gpu_device_quarantine_key("node21", "cuda:0"),
            mapping={"state": "quarantined", "reason": "new fault"},
        )

        record = await read_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            hostname="node21",
        )

        assert record is not None
        assert record["reason"] == "new fault"
        assert record["state"] == "quarantined"
        assert record["manual_clear_required"] == "true"
        assert record["scope"] == "physical_gpu"
        # The positive second read joins the physical lock and rereads once
        # before publishing the missing worker alias.
        assert redis.pipeline_executions == 3

    asyncio.run(scenario())


def test_durable_latch_remains_authoritative_during_redis_read_failure() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason="persistent device fault",
            fault_class="device_fault",
            hostname="node21",
        )
        redis.pipeline_error = RuntimeError("redis unavailable")

        recovered = await read_gpu_quarantine(
            redis,
            "replacement_gpu_0",
            device="cuda:0",
            hostname="node21",
        )

        assert recovered is not None
        assert recovered["reason"] == "persistent device fault"
        assert recovered["scope"] == "physical_gpu"

    asyncio.run(scenario())


def test_redis_handoff_preserves_unlatched_page_after_writer_durable_failure(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        pages: list[dict[str, str]] = []

        def fail_durable_write(*args, **kwargs):  # noqa: ANN002, ANN003
            raise OSError("shared latch storage unavailable")

        async def capture_page(message, **kwargs):  # noqa: ANN001, ARG001
            pages.append({"message": message})
            return PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(quarantine_module, "_write_json_atomic", fail_durable_write)
        monkeypatch.setattr(notifier_module, "send_page_user_notification", capture_page)
        with pytest.raises(RuntimeError, match="durable safety-latch storage failed"):
            await write_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                reason="device fault",
                fault_class="device_fault",
                hostname="node21",
            )

        # Model the original writer dying before it can invoke the notifier.
        recovered = await read_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            hostname="node21",
        )
        assert recovered is not None
        assert recovered["notification_provenance"] == quarantine_module.UNLATCHED_NOTIFICATION_PROVENANCE

        outcome = await notifier_module.send_gpu_quarantine_page(recovered)

        assert outcome.success is True
        assert len(pages) == 1

    asyncio.run(scenario())


def test_clear_removes_all_redis_only_physical_aliases() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        await write_gpu_quarantine(
            redis,
            "origin_worker",
            device="cuda:0",
            reason="device fault",
            fault_class="device_fault",
            hostname=hostname,
        )
        for alias in ("replacement_a", "replacement_b", "replacement_c"):
            assert await read_gpu_quarantine(
                redis,
                alias,
                device="cuda:0",
                hostname=hostname,
            )
            assert await redis.hgetall(gpu_quarantine_key(alias))

        assert await clear_gpu_quarantine(
            redis,
            "replacement_a",
            device="cuda:0",
            hostname=hostname,
        )
        for alias in ("origin_worker", "replacement_a", "replacement_b", "replacement_c"):
            assert not await redis.hgetall(gpu_quarantine_key(alias))
        assert not await redis.hgetall(gpu_device_quarantine_key(hostname, "cuda:0"))

    asyncio.run(scenario())


def test_manual_clear_cannot_be_revived_between_durable_and_redis_write() -> None:
    async def scenario() -> None:
        redis = BlockingHsetRedis()
        redis.block_hsets = True
        write_task = asyncio.create_task(
            write_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                reason="device fault",
                fault_class="device_fault",
                hostname="node21",
            )
        )
        await asyncio.wait_for(redis.hset_started.wait(), timeout=1)

        clear_task = asyncio.create_task(
            clear_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
        )
        await asyncio.sleep(0.05)
        assert not clear_task.done()

        redis.allow_hset.set()
        await asyncio.wait_for(write_task, timeout=1)
        assert await asyncio.wait_for(clear_task, timeout=1)
        assert not redis.hashes
        assert (
            await read_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
            is None
        )

    asyncio.run(scenario())


def test_manual_clear_cannot_be_revived_between_notification_update_phases() -> None:
    async def scenario() -> None:
        redis = BlockingHsetRedis()
        record = await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason="device fault",
            fault_class="device_fault",
            hostname="node21",
        )
        redis.block_hsets = True
        update_task = asyncio.create_task(
            update_gpu_quarantine_notification(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
                expected_generation=gpu_quarantine_generation(record),
                state="sent",
            )
        )
        await asyncio.wait_for(redis.hset_started.wait(), timeout=1)

        clear_task = asyncio.create_task(
            clear_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
        )
        await asyncio.sleep(0.05)
        assert not clear_task.done()

        redis.allow_hset.set()
        updated = await asyncio.wait_for(update_task, timeout=1)
        assert updated is not None and updated["page_user_state"] == "sent"
        assert await asyncio.wait_for(clear_task, timeout=1)
        assert not redis.hashes
        assert (
            await read_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
            is None
        )

    asyncio.run(scenario())


def test_cancelled_clear_releases_late_acquired_device_lock(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        acquire_started = threading.Event()
        allow_acquire = threading.Event()
        original_acquire = quarantine_module._acquire_device_lock

        def delayed_acquire(hostname, device):  # noqa: ANN001
            acquire_started.set()
            if not allow_acquire.wait(timeout=2):
                raise AssertionError("test did not release lock acquisition")
            return original_acquire(hostname, device)

        monkeypatch.setattr(quarantine_module, "_acquire_device_lock", delayed_acquire)
        clear_task = asyncio.create_task(
            clear_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
        )
        while not acquire_started.is_set():
            await asyncio.sleep(0.01)
        clear_task.cancel()
        allow_acquire.set()

        with pytest.raises(asyncio.CancelledError):
            await clear_task
        probe_fd = await asyncio.wait_for(
            asyncio.to_thread(original_acquire, "node21", "cuda:0"),
            timeout=1,
        )
        quarantine_module._release_device_lock(probe_fd)

    asyncio.run(scenario())


def test_cancelled_clear_finishes_durable_mutation_before_unlock(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason="old fault",
            fault_class="device_fault",
            hostname="node21",
        )
        mutation_started = threading.Event()
        allow_mutation = threading.Event()
        original_clear = quarantine_module._clear_durable_latches

        def delayed_clear(*args, **kwargs):  # noqa: ANN002, ANN003
            mutation_started.set()
            if not allow_mutation.wait(timeout=2):
                raise AssertionError("test did not release durable clear")
            return original_clear(*args, **kwargs)

        monkeypatch.setattr(quarantine_module, "_clear_durable_latches", delayed_clear)
        clear_task = asyncio.create_task(
            clear_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
        )
        while not mutation_started.is_set():
            await asyncio.sleep(0.01)
        clear_task.cancel()
        replacement = asyncio.create_task(
            write_gpu_quarantine(
                redis,
                "replacement_gpu_0",
                device="cuda:0",
                reason="new fault",
                fault_class="device_fault",
                hostname="node21",
            )
        )
        await asyncio.sleep(0.05)

        assert not replacement.done()
        allow_mutation.set()
        with pytest.raises(asyncio.CancelledError):
            await clear_task
        new_record = await asyncio.wait_for(replacement, timeout=1)

        assert new_record["reason"] == "new fault"

    asyncio.run(scenario())


def test_worker_scope_latch_does_not_rehydrate_a_device_key() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        await write_gpu_quarantine(
            redis,
            "restart_limited_worker",
            device="cuda:0",
            reason="worker restart limit",
            fault_class="restart_limit",
            hostname=hostname,
            physical_scope=False,
        )

        redis.hashes.clear()
        recovered = await read_gpu_quarantine(
            redis,
            "restart_limited_worker",
            device="cuda:0",
            hostname=hostname,
        )

        assert recovered is not None
        assert recovered["scope"] == "worker_process"
        assert recovered["page_user_state"] == "pending"
        assert await redis.hgetall(gpu_quarantine_key("restart_limited_worker"))
        assert not await redis.hgetall(gpu_device_quarantine_key(hostname, "cuda:0"))

    asyncio.run(scenario())


def test_durable_sent_marker_overrides_stale_redis_pending() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        record = await write_gpu_quarantine(
            redis,
            "original_worker",
            device="cuda:0",
            reason="device fault",
            fault_class="device_fault",
            hostname=hostname,
        )
        await update_gpu_quarantine_notification(
            redis,
            "original_worker",
            device="cuda:0",
            hostname=hostname,
            expected_generation=gpu_quarantine_generation(record),
            state="sent",
        )
        device_key = gpu_device_quarantine_key(hostname, "cuda:0")
        redis.hashes[device_key][b"page_user_state"] = b"pending"
        redis.hashes[device_key].pop(b"page_user_sent_at", None)

        recovered = await read_gpu_quarantine(
            redis,
            "replacement_worker",
            device="cuda:0",
            hostname=hostname,
        )

        assert recovered is not None
        assert recovered["scope"] == "physical_gpu"
        assert recovered["page_user_state"] == "sent"
        assert (await redis.hgetall(device_key))[b"page_user_state"] == b"sent"

    asyncio.run(scenario())


def test_stale_redis_sent_marker_does_not_cross_latch_generation() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        worker_id = "node21_gpu_0"
        old = await write_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            reason="old fault",
            fault_class="device_fault",
            hostname=hostname,
        )
        await update_gpu_quarantine_notification(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
            expected_generation=gpu_quarantine_generation(old),
            state="sent",
        )
        device_key = gpu_device_quarantine_key(hostname, "cuda:0")
        stale_sent = dict(redis.hashes[device_key])

        assert await clear_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
        )
        replacement = await write_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            reason="new fault",
            fault_class="initialization_failure",
            hostname=hostname,
        )
        redis.hashes[device_key] = stale_sent

        recovered = await read_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
        )

        assert recovered is not None
        assert gpu_quarantine_generation(recovered) == gpu_quarantine_generation(replacement)
        assert recovered["reason"] == "new fault"
        assert recovered["page_user_state"] == "pending"
        assert (await redis.hgetall(device_key))[b"event_id"] == replacement["event_id"].encode()

    asyncio.run(scenario())


def test_mismatched_worker_alias_sent_does_not_suppress_current_gpu_page() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        await redis.hset(
            gpu_device_quarantine_key("node21", "cuda:0"),
            mapping={
                "state": "quarantined",
                "scope": "physical_gpu",
                "hostname": "node21",
                "device": "cuda:0",
                "worker_id": "shared_worker",
                "page_user_state": "pending",
                "reason": "current fault",
            },
        )
        await redis.hset(
            gpu_quarantine_key("shared_worker"),
            mapping={
                "state": "quarantined",
                "scope": "physical_gpu",
                "hostname": "old-node",
                "device": "cuda:7",
                "worker_id": "shared_worker",
                "page_user_state": "sent",
                "page_user_sent_at": "old",
                "reason": "old fault",
            },
        )

        recovered = await read_gpu_quarantine(
            redis,
            "shared_worker",
            device="cuda:0",
            hostname="node21",
        )

        assert recovered is not None
        assert recovered["hostname"] == "node21"
        assert recovered["device"] == "cuda:0"
        assert recovered["reason"] == "current fault"
        assert recovered["page_user_state"] == "pending"

    asyncio.run(scenario())


def test_mismatched_durable_worker_alias_does_not_replace_new_physical_fault() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        old_record = await write_gpu_quarantine(
            redis,
            "shared_worker",
            device="cuda:7",
            reason="old fault",
            fault_class="device_fault",
            hostname="old-node",
        )
        await update_gpu_quarantine_notification(
            redis,
            "shared_worker",
            device="cuda:7",
            hostname="old-node",
            expected_generation=gpu_quarantine_generation(old_record),
            state="sent",
        )

        current = await write_gpu_quarantine(
            redis,
            "shared_worker",
            device="cuda:0",
            reason="new fault",
            fault_class="initialization_failure",
            hostname="node21",
        )

        assert current["hostname"] == "node21"
        assert current["device"] == "cuda:0"
        assert current["reason"] == "new fault"
        assert current["page_user_state"] == "pending"
        old_device = await redis.hgetall(gpu_device_quarantine_key("old-node", "cuda:7"))
        assert old_device[b"page_user_state"] == b"sent"

    asyncio.run(scenario())


def test_durable_physical_latch_overrides_stale_worker_scope_redis() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        await write_gpu_quarantine(
            redis,
            "same_worker",
            device="cuda:0",
            reason="physical device fault",
            fault_class="device_fault",
            hostname=hostname,
        )
        redis.hashes.pop(gpu_device_quarantine_key(hostname, "cuda:0"), None)
        redis.hashes[gpu_quarantine_key("same_worker")] = {
            b"state": b"quarantined",
            b"scope": b"worker_process",
            b"worker_id": b"same_worker",
            b"device": b"cuda:0",
            b"hostname": hostname.encode(),
            b"reason": b"stale restart limit",
        }

        recovered = await read_gpu_quarantine(
            redis,
            "same_worker",
            device="cuda:0",
            hostname=hostname,
        )

        assert recovered is not None
        assert recovered["scope"] == "physical_gpu"
        assert recovered["reason"] == "physical device fault"
        assert await redis.hgetall(gpu_device_quarantine_key(hostname, "cuda:0"))

    asyncio.run(scenario())


def test_device_lock_rejects_symlink_without_changing_target(tmp_path) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        target = tmp_path / "unrelated-file"
        target.write_text("unchanged", encoding="utf-8")
        target.chmod(0o640)
        lock_path = tmp_path / "safety_latches" / "locks" / "gpus" / "node21" / "cuda_0.lock"
        lock_path.parent.mkdir(parents=True)
        lock_path.symlink_to(target)

        with pytest.raises(RuntimeError, match="durable safety-latch storage failed"):
            await write_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                reason="device fault",
                fault_class="device_fault",
                hostname="node21",
            )

        assert target.read_text(encoding="utf-8") == "unchanged"
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    asyncio.run(scenario())


def test_repeated_physical_quarantine_preserves_first_fault_and_page_marker() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        first = await write_gpu_quarantine(
            redis,
            "original_worker",
            device="cuda:0",
            reason="first cuInit failure",
            fault_class="initialization_failure",
            task_id="first-task",
            hostname=hostname,
        )
        sent = await update_gpu_quarantine_notification(
            redis,
            "original_worker",
            device="cuda:0",
            hostname=hostname,
            expected_generation=gpu_quarantine_generation(first),
            state="sent",
        )
        repeated = await write_gpu_quarantine(
            redis,
            "replacement_worker",
            device="cuda:0",
            reason="later local retry",
            fault_class="local_quarantine",
            hostname=hostname,
        )

        assert first["page_user_state"] == "pending"
        assert sent["page_user_state"] == "sent"
        assert repeated["reason"] == "first cuInit failure"
        assert repeated["fault_class"] == "initialization_failure"
        assert repeated["task_id"] == "first-task"
        assert repeated["created_at"] == first["created_at"]
        assert repeated["page_user_state"] == "sent"

        assert await clear_gpu_quarantine(
            redis,
            "replacement_worker",
            device="cuda:0",
            hostname=hostname,
        )
        assert not await redis.hgetall(gpu_quarantine_key("original_worker"))
        assert not await redis.hgetall(gpu_quarantine_key("replacement_worker"))
        assert not await redis.hgetall(gpu_device_quarantine_key(hostname, "cuda:0"))

    asyncio.run(scenario())


def test_repeated_worker_quarantine_preserves_generation_and_page_marker() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        first = await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason="first worker bootstrap failure",
            fault_class="worker_bootstrap_failure",
            task_id="first-task",
            hostname=hostname,
            physical_scope=False,
        )
        sent = await update_gpu_quarantine_notification(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            hostname=hostname,
            scope="worker_process",
            expected_generation=gpu_quarantine_generation(first),
            state="sent",
        )
        repeated = await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason="later heartbeat observation",
            fault_class="restart_limit",
            hostname=hostname,
            physical_scope=False,
        )

        assert sent is not None
        assert first["page_user_state"] == "pending"
        assert repeated["event_id"] == first["event_id"]
        assert repeated["created_at"] == first["created_at"]
        assert repeated["reason"] == "first worker bootstrap failure"
        assert repeated["fault_class"] == "worker_bootstrap_failure"
        assert repeated["task_id"] == "first-task"
        assert repeated["page_user_state"] == "sent"

    asyncio.run(scenario())


def test_physical_quarantine_pages_once_and_persists_delivery(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        pages: list[dict[str, str]] = []

        async def capture_page(record):  # noqa: ANN001
            pages.append(dict(record))
            return PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", capture_page)
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]

        await worker._quarantine_gpu(reason="first device failure", fault_class="device_fault")
        await worker._quarantine_gpu(reason="repeated heartbeat", fault_class="local_quarantine")

        assert len(pages) == 1
        stored = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert stored is not None
        assert stored["page_user_state"] == "sent"
        assert stored["reason"] == "first device failure"

    asyncio.run(scenario())


def test_failed_latch_write_preserves_existing_durable_notification_claim(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        existing = await write_gpu_quarantine(
            redis,
            "other_gpu_worker",
            device="cuda:0",
            reason="existing device fault",
            fault_class="device_fault",
            hostname=hostname,
        )
        observed: list[dict[str, str]] = []

        async def failed_write(*args, **kwargs):  # noqa: ANN002, ANN003
            raise OSError("new writer cannot persist")

        async def capture_page(record):  # noqa: ANN001
            observed.append(dict(record))
            return PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(gpu_worker_module, "write_gpu_quarantine", failed_write)
        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", capture_page)
        worker = GPUWorker("replacement_gpu_worker", "cuda:0", redis)  # type: ignore[arg-type]

        await worker._quarantine_gpu(reason="same fault observed", fault_class="device_fault")

        assert len(observed) == 1
        assert gpu_quarantine_generation(observed[0]) == gpu_quarantine_generation(existing)
        assert "notification_provenance" not in observed[0]

    asyncio.run(scenario())


def test_failed_page_retries_once_after_backoff_without_heartbeat_storm(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        monotonic_now = 100.0
        outcomes = iter(
            (
                PageUserNotificationOutcome(False, error_kind="transport_error", error="offline"),
                PageUserNotificationOutcome(True, protocol_version="mock"),
            )
        )
        calls = 0

        async def staged_page(record):  # noqa: ANN001, ARG001
            nonlocal calls
            calls += 1
            return next(outcomes)

        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", staged_page)
        monkeypatch.setattr(gpu_worker_module.time, "monotonic", lambda: monotonic_now)
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        worker.running = True
        await worker._quarantine_gpu(reason="device fault", fault_class="device_fault")

        # Repeated admission/heartbeat-adjacent observations during the one
        # minute backoff must not issue another network request.
        for _ in range(20):
            assert await worker._gpu_admission_allowed() is False
        assert calls == 1

        record = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert record is not None
        assert record["page_user_state"] == "failed"

        monotonic_now += gpu_worker_module._QUARANTINE_PAGE_RETRY_BACKOFF_SECONDS
        assert await worker._gpu_admission_allowed() is False
        assert calls == 2
        # Success and the per-process bound both make subsequent observations
        # permanent no-ops until an operator clears the latch.
        monotonic_now += 3600
        for _ in range(20):
            assert await worker._gpu_admission_allowed() is False
        assert calls == 2

        recovered = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert recovered is not None
        assert recovered["page_user_state"] == "sent"

    asyncio.run(scenario())


def test_worker_process_quarantine_pages_once_and_persists_delivery(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        pages: list[dict[str, str]] = []

        async def capture_worker_page(record):  # noqa: ANN001
            pages.append(dict(record))
            return PageUserNotificationOutcome(True, protocol_version="mock")

        async def unexpected_physical_page(record):  # noqa: ANN001, ARG001
            raise AssertionError("worker-scope quarantine must use the worker-exclusion page")

        monkeypatch.setattr(gpu_worker_module, "send_gpu_worker_exclusion_page", capture_worker_page)
        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", unexpected_physical_page)
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        await worker._quarantine_gpu(
            reason="warm-spare bootstrap failed",
            fault_class="worker_bootstrap_failure",
            physical_scope=False,
        )
        # A repeated observation must neither resend nor regress the durable
        # sent marker (worker-scope writes do not preserve the first record).
        await worker._quarantine_gpu(
            reason="same worker remains excluded",
            fault_class="worker_bootstrap_failure",
            physical_scope=False,
        )

        assert len(pages) == 1
        assert pages[0]["scope"] == "worker_process"
        stored = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert stored is not None
        assert stored["scope"] == "worker_process"
        assert stored["page_user_state"] == "sent"
        assert not await redis.hgetall(gpu_device_quarantine_key(socket.gethostname(), worker.device))

    asyncio.run(scenario())


def test_failed_worker_exclusion_page_retries_once_after_backoff(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        monotonic_now = 100.0
        outcomes = iter(
            (
                PageUserNotificationOutcome(False, error_kind="transport_error", error="offline"),
                PageUserNotificationOutcome(True, protocol_version="mock"),
            )
        )
        calls = 0

        async def staged_page(record):  # noqa: ANN001, ARG001
            nonlocal calls
            calls += 1
            return next(outcomes)

        monkeypatch.setattr(gpu_worker_module, "send_gpu_worker_exclusion_page", staged_page)
        monkeypatch.setattr(gpu_worker_module.time, "monotonic", lambda: monotonic_now)
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        worker.running = True
        await worker._quarantine_gpu(
            reason="warm-spare bootstrap failed",
            fault_class="worker_bootstrap_failure",
            physical_scope=False,
        )

        for _ in range(20):
            assert await worker._gpu_admission_allowed() is False
        assert calls == 1

        monotonic_now += gpu_worker_module._QUARANTINE_PAGE_RETRY_BACKOFF_SECONDS
        assert await worker._gpu_admission_allowed() is False
        assert calls == 2
        for _ in range(20):
            assert await worker._gpu_admission_allowed() is False
        assert calls == 2

        stored = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert stored is not None
        assert stored["scope"] == "worker_process"
        assert stored["page_user_state"] == "sent"

    asyncio.run(scenario())


def test_physical_escalation_is_not_suppressed_by_worker_exclusion_page(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        worker_pages = 0
        physical_pages = 0

        async def capture_worker_page(record):  # noqa: ANN001, ARG001
            nonlocal worker_pages
            worker_pages += 1
            return PageUserNotificationOutcome(True, protocol_version="mock")

        async def capture_physical_page(record):  # noqa: ANN001, ARG001
            nonlocal physical_pages
            physical_pages += 1
            return PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(gpu_worker_module, "send_gpu_worker_exclusion_page", capture_worker_page)
        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", capture_physical_page)
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]

        await worker._quarantine_gpu(
            reason="worker bootstrap failed",
            fault_class="worker_bootstrap_failure",
            physical_scope=False,
        )
        await worker._quarantine_gpu(
            reason="fresh CUDA probe failed",
            fault_class="cuda_probe_failure",
            physical_scope=True,
        )

        assert worker_pages == 1
        assert physical_pages == 1
        stored = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert stored is not None
        assert stored["scope"] == "physical_gpu"
        assert stored["page_user_state"] == "sent"

    asyncio.run(scenario())


def test_late_worker_notification_update_cannot_mark_physical_page_sent() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        worker_id = "node21_gpu_0"
        worker_record = await write_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            reason="worker bootstrap failed",
            fault_class="worker_bootstrap_failure",
            hostname=hostname,
            physical_scope=False,
        )
        physical = await write_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            reason="fresh CUDA probe failed",
            fault_class="cuda_probe_failure",
            hostname=hostname,
            physical_scope=True,
        )

        late_update = await update_gpu_quarantine_notification(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
            scope="worker_process",
            expected_generation=gpu_quarantine_generation(worker_record),
            state="sent",
        )

        assert physical["page_user_state"] == "pending"
        assert late_update is None
        stored = await read_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
        )
        assert stored is not None
        assert stored["scope"] == "physical_gpu"
        assert stored["page_user_state"] == "pending"

    asyncio.run(scenario())


def test_outer_delivery_update_cannot_mark_replacement_generation_sent() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        hostname = socket.gethostname()
        worker_id = "node21_gpu_0"
        old = await write_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            reason="old fault",
            fault_class="device_fault",
            hostname=hostname,
        )
        claim = acquire_gpu_quarantine_notification_claim(old)
        try:
            assert claim.should_send
            finish_gpu_quarantine_notification_claim(claim, state="sent")
        finally:
            release_gpu_quarantine_notification_claim(claim)

        assert await clear_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
        )
        replacement = await write_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            reason="new fault",
            fault_class="initialization_failure",
            hostname=hostname,
        )

        late_update = await update_gpu_quarantine_notification(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
            expected_generation=gpu_quarantine_generation(old),
            state="sent",
        )

        assert late_update is None
        stored = await read_gpu_quarantine(
            redis,
            worker_id,
            device="cuda:0",
            hostname=hostname,
        )
        assert stored is not None
        assert gpu_quarantine_generation(stored) == gpu_quarantine_generation(replacement)
        assert stored["page_user_state"] == "pending"

    asyncio.run(scenario())


def test_quarantine_cancellation_waits_for_page_and_delivery_marker(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        page_started = asyncio.Event()
        release_page = asyncio.Event()

        async def delayed_page(record):  # noqa: ANN001, ARG001
            page_started.set()
            await release_page.wait()
            return PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", delayed_page)
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        quarantine_task = asyncio.create_task(
            worker._quarantine_gpu(reason="device fault", fault_class="device_fault")
        )
        await page_started.wait()
        quarantine_task.cancel()
        release_page.set()

        with pytest.raises(asyncio.CancelledError):
            await quarantine_task
        stored = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert stored is not None
        assert stored["page_user_state"] == "sent"

    asyncio.run(scenario())


def test_quarantine_cancellation_during_latch_write_still_pages_and_persists(monkeypatch) -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        write_started = asyncio.Event()
        release_write = asyncio.Event()
        pages = 0
        original_write = gpu_worker_module.write_gpu_quarantine

        async def delayed_write(*args, **kwargs):  # noqa: ANN002, ANN003
            write_started.set()
            await release_write.wait()
            return await original_write(*args, **kwargs)

        async def capture_page(record):  # noqa: ANN001, ARG001
            nonlocal pages
            pages += 1
            return PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(gpu_worker_module, "write_gpu_quarantine", delayed_write)
        monkeypatch.setattr(gpu_worker_module, "send_gpu_quarantine_page", capture_page)
        worker = GPUWorker("node21_gpu_0", "cuda:0", redis)  # type: ignore[arg-type]
        quarantine_task = asyncio.create_task(
            worker._quarantine_gpu(reason="device fault", fault_class="device_fault")
        )

        await write_started.wait()
        quarantine_task.cancel()
        await asyncio.sleep(0)
        assert not quarantine_task.done()
        release_write.set()

        with pytest.raises(asyncio.CancelledError):
            await quarantine_task
        assert pages == 1
        stored = await read_gpu_quarantine(
            redis,
            worker.worker_id,
            device=worker.device,
            hostname=socket.gethostname(),
        )
        assert stored is not None
        assert stored["scope"] == "physical_gpu"
        assert stored["page_user_state"] == "sent"

    asyncio.run(scenario())


def test_manual_clear_detects_live_supervised_worker_pid(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        prefix = settings.redis_key_prefix
        await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason="worker leader exited before session drain was proven",
            fault_class="unsafe_process_group_shutdown",
            hostname="node21",
        )
        await redis.hset(
            f"{prefix}:worker_process:node21_gpu_0",
            mapping={"device": "cuda:0", "pid": "4242"},
        )
        await redis.hset(
            f"{prefix}:expected_worker:node21_gpu_0",
            mapping={"hostname": "node21"},
        )
        monkeypatch.setattr(manage_gpu_quarantine.os, "kill", lambda pid, signal: None)

        live = await manage_gpu_quarantine._live_worker_processes_on_device(redis, "cuda:0", "node21")

        assert live == ["node21_gpu_0(pid=4242)"]

    asyncio.run(scenario())


def test_manual_clear_refuses_retained_map_after_session_leader_exits(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        prefix = settings.redis_key_prefix
        await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason="worker session may still own a CUDA context",
            fault_class="unsafe_process_group_shutdown",
            hostname="node21",
        )
        await redis.hset(
            f"{prefix}:worker_process:node21_gpu_0",
            mapping={
                "device": "cuda:0",
                "pid": "4242",
                "proc_start_ticks": "88",
                "process_group": "4242",
                "session_id": "4242",
            },
        )
        await redis.hset(
            f"{prefix}:expected_worker:node21_gpu_0",
            mapping={"hostname": "node21"},
        )

        def leader_is_gone(pid: int, signal_number: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(manage_gpu_quarantine.os, "kill", leader_is_gone)

        retained = await manage_gpu_quarantine._live_worker_processes_on_device(redis, "cuda:0", "node21")

        assert retained == ["node21_gpu_0(pid=4242,leader-gone,map-retained)"]
        monkeypatch.setattr(manage_gpu_quarantine.redis, "from_url", lambda url: redis)
        args = SimpleNamespace(
            command="clear",
            worker_id="node21_gpu_0",
            device="cuda:0",
            hostname="node21",
            confirm="node21/cuda:0",
            confirm_unsafe_orphan="node21/cuda:0/NO_GPU_PROCESSES",
        )
        assert await manage_gpu_quarantine._run(args) == 4
        assert await read_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            hostname="node21",
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("fault_class", "reason"),
    [
        ("unsafe_pool_shutdown", "child could not be confirmed reaped"),
        (
            "unsafe_process_group_shutdown",
            "worker leader exited before process group drained",
        ),
    ],
)
def test_unsafe_orphan_clear_requires_stronger_exact_confirmation(
    monkeypatch,
    fault_class: str,
    reason: str,
) -> None:  # noqa: ANN001
    async def scenario() -> None:
        redis = FakeRedis()
        await write_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            reason=reason,
            fault_class=fault_class,
            hostname="node21",
        )
        monkeypatch.setattr(manage_gpu_quarantine.redis, "from_url", lambda url: redis)
        args = SimpleNamespace(
            command="clear",
            worker_id="node21_gpu_0",
            device="cuda:0",
            hostname="node21",
            confirm="node21/cuda:0",
            confirm_unsafe_orphan="",
        )

        assert await manage_gpu_quarantine._run(args) == 5
        assert await read_gpu_quarantine(
            redis,
            "node21_gpu_0",
            device="cuda:0",
            hostname="node21",
        )

        args.confirm_unsafe_orphan = "node21/cuda:0/NO_GPU_PROCESSES"
        assert await manage_gpu_quarantine._run(args) == 0
        assert (
            await read_gpu_quarantine(
                redis,
                "node21_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
            is None
        )

    asyncio.run(scenario())
