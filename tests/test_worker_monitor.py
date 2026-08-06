import asyncio
import importlib.util
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_latch_dir(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setenv("KERNELGYM_SAFETY_LATCH_DIR", str(tmp_path / "safety_latches"))

    async def forbid_real_page(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError("worker monitor test attempted a real page-user request")

    from kernelgym.utils import page_user_notifier

    monkeypatch.setattr(page_user_notifier, "send_page_user_notification", forbid_real_page)


def load_worker_monitor():
    spec = importlib.util.spec_from_file_location(
        "worker_monitor_script", ROOT / "kernelgym" / "worker" / "worker_monitor.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRedis:
    def __init__(self) -> None:
        self.hashes = {}

    async def hgetall(self, key):
        if isinstance(key, bytes):
            key = key.decode()
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping):
        if isinstance(key, bytes):
            key = key.decode()
        target = self.hashes.setdefault(key, {})
        target.update({str(field).encode(): str(value).encode() for field, value in mapping.items()})
        return 1

    async def hdel(self, key, *fields):
        if isinstance(key, bytes):
            key = key.decode()
        target = self.hashes.get(key, {})
        removed = 0
        for field in fields:
            field_bytes = str(field).encode()
            if field_bytes in target:
                removed += 1
                del target[field_bytes]
        return removed

    async def delete(self, *keys):
        removed = 0
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode()
            if key in self.hashes:
                removed += 1
                del self.hashes[key]
        return removed

    async def eval(self, script, numkeys, key, *args):  # noqa: ARG002
        if isinstance(key, bytes):
            key = key.decode()
        target = self.hashes.get(key, {})
        if "REGISTER_IF_EMPTY" in script:
            raise AssertionError("sentinel is not part of production script")
        if "current_pid and current_pid ~= ''" in script:
            if target.get(b"pid", b""):
                return 0
            pid, start_time, start_ticks, process_group, session_id, device = args
            await self.hset(
                key,
                mapping={
                    "pid": pid,
                    "start_time": start_time,
                    "proc_start_ticks": start_ticks,
                    "process_group": process_group,
                    "session_id": session_id,
                    "device": device,
                },
            )
            return 1

        pid, expected_ticks, expected_group, expected_session, has_ticks, has_group, has_session = args
        if target.get(b"pid", b"").decode() != str(pid):
            return 0
        current_ticks = target.get(b"proc_start_ticks")
        if has_ticks == "1":
            if current_ticks is None or current_ticks.decode() != expected_ticks:
                return 0
        elif current_ticks not in {None, b""}:
            return 0
        current_group = target.get(b"process_group")
        if has_group == "1":
            if current_group is None or current_group.decode() != expected_group:
                return 0
        elif current_group not in {None, b""}:
            return 0
        current_session = target.get(b"session_id")
        if has_session == "1":
            if current_session is None or current_session.decode() != expected_session:
                return 0
        elif current_session not in {None, b""}:
            return 0
        return await self.delete(key)


class FakeProcess:
    pid = 12345

    def __init__(self) -> None:
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):  # noqa: ARG002
        return self.returncode


class ReapableFakeProcess:
    pid = 12346

    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False
        self.reaped = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.reaped = True
        self.returncode = -15
        return self.returncode


def test_worker_monitor_restarts_cpu_worker_with_cpu_entrypoint(monkeypatch, tmp_path) -> None:
    worker_monitor = load_worker_monitor()
    commands = []
    popen_kwargs = []
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    monkeypatch.chdir(tmp_path)

    async def noop(*args, **kwargs):  # noqa: ARG001
        return True

    async def reset_must_not_run(*args):
        raise AssertionError("monitor must not reset a GPU during process restart")

    monkeypatch.setattr(monitor, "_kill_worker_process", noop)
    monkeypatch.setattr(monitor, "_reset_gpu_device", reset_must_not_run)
    monkeypatch.setattr(
        monitor,
        "_read_process_identity",
        lambda pid: worker_monitor.ProcessIdentity(pid, "100", "S", pid, pid),
    )
    monkeypatch.setattr(asyncio, "sleep", noop)

    def fake_popen(command, **kwargs):
        commands.append(command)
        popen_kwargs.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert asyncio.run(monitor._restart_worker("worker_cpu_0", "cpu"))
    assert commands == [
        [
            sys.executable,
            "-m",
            "kernelgym.worker.cpu_worker",
            "--worker-id",
            "worker_cpu_0",
        ]
    ]
    assert popen_kwargs[0]["start_new_session"] is True
    assert "preexec_fn" not in popen_kwargs[0]


def test_load_local_expected_ids_filters_by_hostname() -> None:
    """A monitor must only enforce expected workers registered for its own host;
    ids with no recorded hostname stay local for backward compatibility."""
    worker_monitor = load_worker_monitor()

    class HostFakeRedis(FakeRedis):
        async def smembers(self, key):
            return {b"local_gpu_0", b"remote_gpu_0", b"legacy_cpu_0"}

        async def hgetall(self, key):
            data = {
                "kernelgym:expected_worker:local_gpu_0": {b"hostname": b"host-a", b"device": b"cuda:0"},
                "kernelgym:expected_worker:remote_gpu_0": {b"hostname": b"host-b", b"device": b"cuda:0"},
            }
            return data.get(key, {})

    monitor = worker_monitor.WorkerMonitor(HostFakeRedis(), persistent=True)
    monitor.hostname = "host-a"
    local_ids = asyncio.run(monitor._load_local_expected_ids())
    assert local_ids == {"local_gpu_0", "legacy_cpu_0"}


def test_worker_monitor_refuses_to_restart_quarantined_gpu(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()

    class QuarantineRedis(FakeRedis):
        async def hgetall(self, key):
            if key == "kernelgym:quarantine:worker:worker_gpu_0":
                return {b"state": b"quarantined", b"reason": b"driver failed"}
            return {}

    monitor = worker_monitor.WorkerMonitor(QuarantineRedis(), persistent=True)
    killed = False

    async def must_not_kill(*args):
        nonlocal killed
        killed = True

    monkeypatch.setattr(monitor, "_kill_worker_process", must_not_kill)

    assert asyncio.run(monitor._restart_worker("worker_gpu_0", "cuda:0")) is False
    assert killed is False


def test_worker_monitor_quarantines_and_pages_before_fourth_gpu_restart(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    from kernelgym.utils import page_user_notifier

    class RecordingRedis(FakeRedis):
        def __init__(self):
            self.hashes = {}

        async def hgetall(self, key):
            return dict(self.hashes.get(key, {}))

        async def hset(self, key, mapping):
            self.hashes[key] = {str(field).encode(): str(value).encode() for field, value in mapping.items()}
            return 1

    redis = RecordingRedis()
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    monitor.hostname = "host-a"
    monitor.restart_attempts["worker_gpu_0"] = monitor.max_restart_attempts
    pages = []

    async def capture_page(message, **kwargs):  # noqa: ANN001
        pages.append((message, kwargs))
        return page_user_notifier.PageUserNotificationOutcome(True, protocol_version="test")

    monkeypatch.setattr(page_user_notifier, "send_page_user_notification", capture_page)

    allowed = asyncio.run(monitor._reserve_gpu_restart_attempt("worker_gpu_0", "cuda:0", "process died during init"))

    assert allowed is False
    latch = asyncio.run(redis.hgetall(f"{worker_monitor.KEY_PREFIX}:quarantine:worker:worker_gpu_0"))
    assert latch[b"fault_class"] == b"restart_limit"
    assert latch[b"scope"] == b"worker_process"
    assert latch[b"page_user_state"] == b"sent"
    assert latch[b"page_user_attempt_count"] == b"1"
    assert b"3 automatic restart attempts" in latch[b"reason"]
    assert not asyncio.run(redis.hgetall(f"{worker_monitor.KEY_PREFIX}:quarantine:gpu:host-a:cuda:0"))
    assert len(pages) == 1
    assert "GPU worker removed after restart limit" in pages[0][0]
    assert "physical GPU fault: not yet proven" in pages[0][0]
    assert pages[0][1]["title"] == "KernelGYM GPU worker excluded"


def test_restart_limit_page_failure_retries_once_after_backoff(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    from kernelgym.utils import page_user_notifier

    redis = FakeRedis()
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    monitor.hostname = "host-a"
    monitor.restart_attempts["worker_gpu_0"] = monitor.max_restart_attempts
    monotonic_now = 100.0
    outcomes = iter(
        (
            page_user_notifier.PageUserNotificationOutcome(False, error_kind="transport_error", error="offline"),
            page_user_notifier.PageUserNotificationOutcome(True, protocol_version="test"),
        )
    )
    calls = 0

    async def staged_page(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        nonlocal calls
        calls += 1
        return next(outcomes)

    monkeypatch.setattr(page_user_notifier, "send_page_user_notification", staged_page)
    monkeypatch.setattr(worker_monitor.time, "monotonic", lambda: monotonic_now)

    assert (
        asyncio.run(monitor._reserve_gpu_restart_attempt("worker_gpu_0", "cuda:0", "process died during init"))
        is False
    )
    assert calls == 1

    for _ in range(20):
        assert asyncio.run(monitor._is_worker_quarantined("worker_gpu_0", device="cuda:0")) is True
    assert calls == 1

    monotonic_now += worker_monitor._RESTART_LIMIT_PAGE_RETRY_BACKOFF_SECONDS
    assert asyncio.run(monitor._is_worker_quarantined("worker_gpu_0", device="cuda:0")) is True
    assert calls == 2
    for _ in range(20):
        assert asyncio.run(monitor._is_worker_quarantined("worker_gpu_0", device="cuda:0")) is True
    assert calls == 2

    latch = asyncio.run(redis.hgetall(f"{worker_monitor.KEY_PREFIX}:quarantine:worker:worker_gpu_0"))
    assert latch[b"page_user_state"] == b"sent"
    assert latch[b"page_user_attempt_count"] == b"2"


def test_restart_limit_persistence_failure_still_pages_and_keeps_budget_exhausted(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    from kernelgym.utils import page_user_notifier

    redis = FakeRedis()
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    monitor.hostname = "host-a"
    monitor.restart_attempts["worker_gpu_0"] = monitor.max_restart_attempts
    pages = []

    async def failed_persistence(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("Redis and shared latch unavailable")

    async def capture_page(message, **kwargs):  # noqa: ANN001
        pages.append((message, kwargs))
        return page_user_notifier.PageUserNotificationOutcome(True, protocol_version="test")

    monkeypatch.setattr(worker_monitor, "write_gpu_quarantine", failed_persistence)
    monkeypatch.setattr(page_user_notifier, "send_page_user_notification", capture_page)

    allowed = asyncio.run(monitor._reserve_gpu_restart_attempt("worker_gpu_0", "cuda:0", "init failed"))

    assert allowed is False
    assert monitor.restart_attempts["worker_gpu_0"] == monitor.max_restart_attempts
    assert len(pages) == 1
    assert "GPU worker removed after restart limit" in pages[0][0]
    assert pages[0][1]["title"] == "KernelGYM GPU worker excluded"


def test_restart_limit_page_budget_resets_for_new_latch_generation(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    monitor.hostname = "host-a"
    pages = []

    class Outcome:
        success = True
        protocol_version = "test"
        error_kind = None
        error = None

    async def capture_page(record):  # noqa: ANN001
        pages.append(record["created_at"])
        return Outcome()

    async def ignore_update(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return None

    monkeypatch.setattr(worker_monitor, "send_gpu_worker_exclusion_page", capture_page)
    monkeypatch.setattr(worker_monitor, "update_gpu_quarantine_notification", ignore_update)
    base = {
        "worker_id": "worker_gpu_0",
        "device": "cuda:0",
        "hostname": "host-a",
        "scope": "worker_process",
        "fault_class": "restart_limit",
        "page_user_state": "pending",
    }

    asyncio.run(monitor._ensure_restart_limit_notification({**base, "created_at": "generation-1"}))
    asyncio.run(monitor._ensure_restart_limit_notification({**base, "created_at": "generation-1"}))
    asyncio.run(monitor._ensure_restart_limit_notification({**base, "created_at": "generation-2"}))

    assert pages == ["generation-1", "generation-2"]


def test_monitor_retries_service_created_failed_physical_page(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    monitor.hostname = "node21"
    pages = []
    updates = []
    record = {
        "state": "quarantined",
        "scope": "physical_gpu",
        "worker_id": "worker_gpu_0",
        "device": "cuda:0",
        "hostname": "node21",
        "fault_class": "unsafe_process_group_shutdown",
        "reason": "service could not prove worker SID drained",
        "created_at": "service-generation-1",
        "page_user_state": "failed",
        "page_user_updated_at": (datetime.now() - timedelta(seconds=120)).isoformat(),
    }

    async def fake_read(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return dict(record)

    class Outcome:
        success = True
        protocol_version = "test"
        error_kind = None
        error = None

    async def capture_physical_page(current):  # noqa: ANN001
        pages.append(current)
        return Outcome()

    async def worker_page_must_not_run(current):  # noqa: ANN001, ARG001
        raise AssertionError("a physical latch must use the physical GPU page")

    async def capture_update(redis, worker_id, **kwargs):  # noqa: ANN001, ARG001
        updates.append((worker_id, kwargs))
        return record

    monkeypatch.setattr(worker_monitor, "read_gpu_quarantine", fake_read)
    monkeypatch.setattr(worker_monitor, "send_gpu_quarantine_page", capture_physical_page)
    monkeypatch.setattr(worker_monitor, "send_gpu_worker_exclusion_page", worker_page_must_not_run)
    monkeypatch.setattr(worker_monitor, "update_gpu_quarantine_notification", capture_update)

    assert (
        asyncio.run(
            monitor._is_worker_quarantined(
                "worker_gpu_0",
                device="cuda:0",
                hostname="node21",
            )
        )
        is True
    )
    assert [page["created_at"] for page in pages] == ["service-generation-1"]
    assert updates == [
        (
            "worker_gpu_0",
            {
                "device": "cuda:0",
                "hostname": "node21",
                "scope": "physical_gpu",
                "expected_generation": "created:service-generation-1",
                "state": "sent",
                "error": "",
            },
        )
    ]


def test_stale_queued_restart_does_not_kill_recovered_worker(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()

    class HealthyRedis(FakeRedis):
        async def hgetall(self, key):
            if key == f"{worker_monitor.KEY_PREFIX}:worker:worker_gpu_0":
                return {
                    b"online": b"true",
                    b"health_state": b"healthy",
                    b"accepting_tasks": b"true",
                    b"last_heartbeat": datetime.now().isoformat().encode(),
                }
            return {}

    monitor = worker_monitor.WorkerMonitor(HealthyRedis(), persistent=True)
    request = {
        "worker_id": "worker_gpu_0",
        "device": "cuda:0",
        "reason": "Heartbeat timeout",
        "timestamp": datetime.now(),
    }
    monitor.restart_in_progress.add("worker_gpu_0")
    monitor.running = True

    class OneShotQueue:
        async def get(self):
            monitor.running = False
            return request

        async def put(self, item):
            raise AssertionError(f"stale restart must not be requeued: {item}")

    monitor.restart_queue = OneShotQueue()

    async def must_not_kill(*args):
        raise AssertionError("a recovered worker must not be killed")

    monkeypatch.setattr(monitor, "_kill_worker_process", must_not_kill)

    asyncio.run(monitor._restart_loop())

    assert "worker_gpu_0" not in monitor.restart_in_progress
    assert "worker_gpu_0" not in monitor.restart_attempts


def test_kill_not_confirmed_prohibits_spawn(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)

    async def kill_not_confirmed(*args, **kwargs):  # noqa: ARG001
        return False

    def must_not_spawn(*args, **kwargs):
        raise AssertionError("Popen must not run when the old process may still be alive")

    monkeypatch.setattr(monitor, "_kill_worker_process", kill_not_confirmed)
    monkeypatch.setattr(subprocess, "Popen", must_not_spawn)

    assert asyncio.run(monitor._restart_worker("worker_gpu_0", "cuda:0")) is False


def test_unexpected_kill_error_quarantines_gpu_and_refuses_spawn(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()

    class UnexpectedRedis(FakeRedis):
        async def hgetall(self, key):  # noqa: ANN001, ARG002
            raise KeyError("unexpected decode failure")

    monitor = worker_monitor.WorkerMonitor(UnexpectedRedis(), persistent=True)
    quarantines = []

    async def not_quarantined(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return False

    async def not_recovered(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return False

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    def must_not_spawn(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Popen must not run after unproven GPU containment")

    monkeypatch.setattr(monitor, "_is_worker_quarantined", not_quarantined)
    monkeypatch.setattr(monitor, "_worker_admission_is_currently_open", not_recovered)
    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)
    monkeypatch.setattr(subprocess, "Popen", must_not_spawn)

    assert asyncio.run(monitor._restart_worker("worker_gpu_0", "cuda:0")) is False
    assert quarantines == [
        (
            "worker_gpu_0",
            "cuda:0",
            "unexpected error before session - drain proof: KeyError",
        )
    ]


def test_pid_registration_failure_reaps_spawned_process(monkeypatch, tmp_path) -> None:
    worker_monitor = load_worker_monitor()

    class FailingRedis(FakeRedis):
        async def eval(self, script, numkeys, key, *args):  # noqa: ARG002
            if "current_pid and current_pid ~= ''" in script:
                raise RuntimeError("redis unavailable")
            return await super().eval(script, numkeys, key, *args)

    monitor = worker_monitor.WorkerMonitor(FailingRedis(), persistent=True)
    process = ReapableFakeProcess()

    async def kill_confirmed(*args, **kwargs):  # noqa: ARG001
        return True

    async def no_sleep(*args):
        return None

    def fake_popen(*args, **kwargs):
        return process

    def fake_killpg(pid, signum):
        assert pid == process.pid
        if signum == 0:
            raise ProcessLookupError
        assert signum == worker_monitor.signal.SIGTERM
        process.terminated = True
        process.returncode = -15

    monkeypatch.setattr(monitor, "_kill_worker_process", kill_confirmed)
    monkeypatch.setattr(
        monitor,
        "_read_process_identity",
        lambda pid: worker_monitor.ProcessIdentity(pid, "200", "S", pid, pid),
    )
    monkeypatch.setattr(monitor, "_cmdline_matches_worker", lambda pid, worker_id: True)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(worker_monitor.os, "killpg", fake_killpg)
    monkeypatch.setattr(worker_monitor.settings, "log_dir", tmp_path)

    assert asyncio.run(monitor._restart_worker("worker_gpu_0", "cuda:0")) is False
    assert process.terminated is True
    assert process.reaped is True
    assert process.poll() is not None


def test_missing_initial_identity_with_live_group_is_tracked_and_quarantined(monkeypatch, tmp_path) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    process = ReapableFakeProcess()
    process.returncode = -15
    probes = []
    quarantines = []

    async def kill_confirmed(*args, **kwargs):  # noqa: ARG001
        return True

    async def no_sleep(*args):  # noqa: ARG001
        return None

    def group_still_exists(process_group, signum):
        probes.append((process_group, signum))

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    async def registration_must_not_run(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("an unauthenticated process generation must not be registered")

    monkeypatch.setattr(monitor, "_kill_worker_process", kill_confirmed)
    monkeypatch.setattr(monitor, "_read_process_identity", lambda pid: None)
    monkeypatch.setattr(monitor, "_register_spawned_process", registration_must_not_run)
    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(worker_monitor.os, "killpg", group_still_exists)
    monkeypatch.setattr(worker_monitor.settings, "log_dir", tmp_path)
    monkeypatch.setattr(worker_monitor, "prepare_core_dump_dir", lambda *args, **kwargs: None)

    assert asyncio.run(monitor._restart_worker("worker_gpu_0", "cuda:0")) is False
    assert process.reaped is True
    assert monitor.spawned_processes["worker_gpu_0"] is process
    assert probes == [(process.pid, 0)]
    assert quarantines == [
        (
            "worker_gpu_0",
            "cuda:0",
            f"worker leader PID {process.pid} exited but session {process.pid} was not drained: "
            "leader generation was never authenticated",
        )
    ]


def test_retained_unidentified_group_blocks_later_replacement(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    process = ReapableFakeProcess()
    process.returncode = -15
    monitor.spawned_processes["worker_gpu_0"] = process
    quarantines = []

    monkeypatch.setattr(worker_monitor.os, "killpg", lambda process_group, signum: None)

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)

    stopped = asyncio.run(
        monitor._kill_worker_process(
            "worker_gpu_0",
            expected_pid=0,
            device_hint="cuda:0",
        )
    )

    assert stopped is False
    assert monitor.spawned_processes["worker_gpu_0"] is process
    assert quarantines and quarantines[0][:2] == ("worker_gpu_0", "cuda:0")


def test_initial_identity_read_error_fails_closed_when_session_scan_is_unreadable(monkeypatch, tmp_path) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    process = ReapableFakeProcess()
    process.returncode = -15
    identity_reads = 0
    probes = []

    async def kill_confirmed(*args, **kwargs):  # noqa: ARG001
        return True

    async def no_sleep(*args):  # noqa: ARG001
        return None

    def unreadable_identity(pid):  # noqa: ARG001
        nonlocal identity_reads
        identity_reads += 1
        raise RuntimeError("malformed proc stat")

    def group_absent(process_group, signum):
        probes.append((process_group, signum))
        raise ProcessLookupError

    quarantines = []

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    monkeypatch.setattr(monitor, "_kill_worker_process", kill_confirmed)
    monkeypatch.setattr(monitor, "_read_process_identity", unreadable_identity)
    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(worker_monitor.os, "killpg", group_absent)
    monkeypatch.setattr(worker_monitor.settings, "log_dir", tmp_path)
    monkeypatch.setattr(worker_monitor, "prepare_core_dump_dir", lambda *args, **kwargs: None)

    assert asyncio.run(monitor._restart_worker("worker_gpu_0", "cuda:0")) is False
    assert identity_reads == 2
    assert process.reaped is True
    assert monitor.spawned_processes["worker_gpu_0"] is process
    assert probes == []
    assert quarantines and "session inspection failed" in quarantines[0][2]


def test_unidentified_spawn_cleanup_finishes_before_cancellation_propagates(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def delayed_stop(*args, **kwargs):  # noqa: ARG001
        cleanup_started.set()
        await release_cleanup.wait()
        cleanup_finished.set()
        return False

    monkeypatch.setattr(monitor, "_stop_spawned_process", delayed_stop)

    async def scenario() -> None:
        task = asyncio.create_task(
            monitor._finish_spawn_cleanup(
                FakeProcess(),
                "worker_gpu_0",
                None,
                "cuda:0",
            )
        )
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleanup_finished.is_set()

    asyncio.run(scenario())


def test_restart_budget_survives_monitor_recreation() -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()

    first = worker_monitor.WorkerMonitor(redis, persistent=True)
    first.hostname = "host-a"
    assert asyncio.run(first._reserve_gpu_restart_attempt("worker_gpu_0", "cuda:0", "first")) is True

    second = worker_monitor.WorkerMonitor(redis, persistent=True)
    second.hostname = "host-a"
    assert second.restart_attempts == {}
    assert asyncio.run(second._reserve_gpu_restart_attempt("worker_gpu_0", "cuda:0", "second")) is True
    assert second.restart_attempts["worker_gpu_0"] == 2

    asyncio.run(second._clear_gpu_restart_attempts("worker_gpu_0", "cuda:0"))
    third = worker_monitor.WorkerMonitor(redis, persistent=True)
    third.hostname = "host-a"
    assert asyncio.run(third._reserve_gpu_restart_attempt("worker_gpu_0", "cuda:0", "fresh")) is True
    assert third.restart_attempts["worker_gpu_0"] == 1


def test_default_restart_budget_uses_shared_repository_latch_dir(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monkeypatch.delenv("KERNELGYM_SAFETY_LATCH_DIR")

    assert worker_monitor._restart_budget_path("host-a", "worker_gpu_0") == (
        ROOT / "logs" / "safety_latches" / "restart_attempts" / "host-a" / "worker_gpu_0.json"
    )


def test_malformed_and_missing_heartbeats_fail_closed_per_worker(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()

    class HeartbeatRedis(FakeRedis):
        async def scan_iter(self, pattern, count=500):  # noqa: ARG002
            yield f"{worker_monitor.KEY_PREFIX}:worker:worker_gpu_bad".encode()
            yield f"{worker_monitor.KEY_PREFIX}:worker:worker_gpu_missing".encode()

    redis = HeartbeatRedis()
    redis.hashes[f"{worker_monitor.KEY_PREFIX}:worker:worker_gpu_bad"] = {
        b"device": b"cuda:0",
        b"online": b"true",
        b"health_state": b"healthy",
        b"accepting_tasks": b"true",
        b"last_heartbeat": b"not-a-timestamp",
    }
    redis.hashes[f"{worker_monitor.KEY_PREFIX}:worker:worker_gpu_missing"] = {
        b"device": b"cuda:1",
        b"online": b"true",
        b"health_state": b"healthy",
        b"accepting_tasks": b"true",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=False)

    async def not_quarantined(*args, **kwargs):  # noqa: ARG001
        return False

    monkeypatch.setattr(monitor, "_is_worker_quarantined", not_quarantined)

    asyncio.run(monitor._check_workers())

    assert monitor.restart_in_progress == {"worker_gpu_bad", "worker_gpu_missing"}
    first = monitor.restart_queue.get_nowait()
    second = monitor.restart_queue.get_nowait()
    assert {first["reason"], second["reason"]} == {"Invalid heartbeat", "Missing heartbeat"}
    assert {first["observed_pid"], second["observed_pid"]} == {0}


def test_stale_process_generation_request_is_dropped_before_budget_or_kill(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    redis.hashes[f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"] = {
        b"pid": b"222",
        b"proc_start_ticks": b"new-generation",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    monitor.restart_in_progress.add("worker_gpu_0")
    monitor.running = True
    request = {
        "worker_id": "worker_gpu_0",
        "device": "cuda:0",
        "reason": "Process dead",
        "timestamp": datetime.now(),
        "observed_pid": 111,
        "observed_start_ticks": "old-generation",
        "replacement_required": True,
    }

    class OneShotQueue:
        async def get(self):
            monitor.running = False
            return request

        async def put(self, item):
            raise AssertionError(f"stale request must not be requeued: {item}")

    monitor.restart_queue = OneShotQueue()

    async def must_not_run(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("stale generation must be rejected before budget reservation or kill")

    monkeypatch.setattr(monitor, "_reserve_gpu_restart_attempt", must_not_run)
    monkeypatch.setattr(monitor, "_kill_worker_process", must_not_run)

    asyncio.run(monitor._restart_loop())
    assert "worker_gpu_0" not in monitor.restart_in_progress


def test_successful_spawn_is_not_requeued_when_shutdown_flag_cleanup_fails(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()

    class HdelFailRedis(FakeRedis):
        async def hdel(self, key, *fields):  # noqa: ARG002
            raise RuntimeError("redis write failed after spawn")

    monitor = worker_monitor.WorkerMonitor(HdelFailRedis(), persistent=True)
    monitor.restart_in_progress.add("worker_cpu_0")
    monitor.running = True
    request = {
        "worker_id": "worker_cpu_0",
        "device": "cpu",
        "reason": "Heartbeat timeout",
        "timestamp": datetime.now(),
    }

    class OneShotQueue:
        async def get(self):
            monitor.running = False
            return request

        async def put(self, item):
            raise AssertionError(f"successful restart must never be replayed: {item}")

    monitor.restart_queue = OneShotQueue()

    async def restart_succeeds(*args, **kwargs):  # noqa: ARG001
        return True

    async def no_sleep(*args):  # noqa: ARG001
        return None

    monkeypatch.setattr(monitor, "_restart_worker", restart_succeeds)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    asyncio.run(monitor._restart_loop())
    assert "worker_cpu_0" not in monitor.restart_in_progress


def test_old_cuda_shutdown_flag_does_not_apply_to_replacement_when_cleanup_fails() -> None:
    worker_monitor = load_worker_monitor()

    class HdelFailRedis(FakeRedis):
        async def hdel(self, key, *fields):  # noqa: ARG002
            raise RuntimeError("redis write failed after spawn")

    redis = HdelFailRedis()
    redis.hashes[f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"] = {
        b"pid": b"222",
        b"proc_start_ticks": b"new-generation",
        b"start_time": b"2026-08-05T12:00:01",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    stale_worker_info = {
        "cuda_error_shutdown": "true",
        "shutdown_time": "2026-08-05T12:00:00",
    }

    assert asyncio.run(monitor._cuda_shutdown_flag_is_current("worker_gpu_0", stale_worker_info)) is False


def test_pid_start_ticks_are_rechecked_before_sigkill(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    process_key = f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"
    redis.hashes[process_key] = {
        b"pid": b"12345",
        b"proc_start_ticks": b"100",
        b"process_group": b"12345",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    old = worker_monitor.ProcessIdentity(12345, "100", "S", 12345, 12345)
    reused = worker_monitor.ProcessIdentity(12345, "200", "S", 12345, 12345)
    identities = iter((old, old, old, reused))
    signals = []

    monkeypatch.setattr(monitor, "_read_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(monitor, "_cmdline_matches_worker", lambda pid, worker_id: True)
    monkeypatch.setattr(worker_monitor.os, "killpg", lambda pid, signum: signals.append((pid, signum)))

    async def never_exits(*args, **kwargs):  # noqa: ARG001
        return False

    monkeypatch.setattr(monitor, "_wait_for_process_exit", never_exits)

    quarantines = []

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)

    result = asyncio.run(
        monitor._kill_worker_process(
            "worker_gpu_0",
            expected_pid=12345,
            expected_start_ticks="100",
        )
    )

    assert result is False
    assert signals == [(12345, worker_monitor.signal.SIGTERM)]
    assert redis.hashes[process_key][b"pid"] == b"12345"
    assert quarantines and quarantines[0][:2] == ("worker_gpu_0", "cuda:0")


def test_malformed_pid_map_is_quarantined_and_preserved(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    process_key = f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"
    redis.hashes[process_key] = {
        b"pid": b"not-a-pid",
        b"proc_start_ticks": b"100",
        b"process_group": b"12345",
        b"session_id": b"12345",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    quarantines = []

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)
    monkeypatch.setattr(
        worker_monitor.os,
        "killpg",
        lambda process_group, signum: (_ for _ in ()).throw(
            AssertionError("a malformed unauthenticated PID must not be signalled")
        ),
    )

    assert asyncio.run(monitor._kill_worker_process("worker_gpu_0")) is False
    assert process_key in redis.hashes
    assert quarantines == [("worker_gpu_0", "cuda:0", "worker process map contains a malformed PID")]


def test_exact_retained_zombie_is_reaped_and_map_deleted(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    process_key = f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"
    redis.hashes[process_key] = {
        b"pid": b"12346",
        b"proc_start_ticks": b"300",
        b"process_group": b"12346",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    process = ReapableFakeProcess()
    process.returncode = -9
    monitor.spawned_processes["worker_gpu_0"] = process
    monitor.spawned_identities["worker_gpu_0"] = worker_monitor.ProcessIdentity(12346, "300", "Z", 12346, 12346)
    monkeypatch.setattr(
        monitor,
        "_read_process_identity",
        lambda pid: worker_monitor.ProcessIdentity(pid, "300", "Z", pid, pid),
    )

    def must_not_signal(pid, signum):
        assert pid == 12346
        if signum == 0:
            raise ProcessLookupError
        raise AssertionError("an already-dead exact child should only be reaped")

    monkeypatch.setattr(worker_monitor.os, "killpg", must_not_signal)

    assert (
        asyncio.run(
            monitor._kill_worker_process(
                "worker_gpu_0",
                expected_pid=12346,
                expected_start_ticks="300",
            )
        )
        is True
    )
    assert process.reaped is True
    assert process_key not in redis.hashes
    assert "worker_gpu_0" not in monitor.spawned_processes


def test_compare_and_delete_preserves_new_process_generation() -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    process_key = f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"
    redis.hashes[process_key] = {
        b"pid": b"777",
        b"proc_start_ticks": b"new",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)

    deleted = asyncio.run(
        monitor._compare_and_delete_process_map(
            "worker_gpu_0",
            pid=777,
            map_start_ticks="old",
        )
    )

    assert deleted is False
    assert redis.hashes[process_key][b"proc_start_ticks"] == b"new"


def test_compare_and_delete_preserves_changed_session_generation() -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    process_key = f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"
    redis.hashes[process_key] = {
        b"pid": b"777",
        b"proc_start_ticks": b"same",
        b"process_group": b"777",
        b"session_id": b"888",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)

    deleted = asyncio.run(
        monitor._compare_and_delete_process_map(
            "worker_gpu_0",
            pid=777,
            map_start_ticks="same",
            map_process_group="777",
            map_session_id="777",
        )
    )

    assert deleted is False
    assert redis.hashes[process_key][b"session_id"] == b"888"


def test_process_group_drain_requires_atomic_kernel_absence(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    probes = []

    def group_still_exists(process_group, signum):
        probes.append((process_group, signum))

    # Even an empty /proc diagnostic snapshot is not an absence proof: a child
    # can fork between that snapshot and its parent's exit.
    monkeypatch.setattr(worker_monitor.os, "killpg", group_still_exists)
    monkeypatch.setattr(monitor, "_live_process_group_members", lambda process_group: [])
    assert monitor._process_group_is_drained(43210) is False
    assert probes == [(43210, 0)]

    def group_absent(process_group, signum):
        probes.append((process_group, signum))
        raise ProcessLookupError

    monkeypatch.setattr(worker_monitor.os, "killpg", group_absent)
    assert monitor._process_group_is_drained(43210) is True
    assert probes[-1] == (43210, 0)


def test_session_force_kill_freezes_and_kills_all_inner_process_groups(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)

    def member(pid, state, process_group):
        return worker_monitor.ProcessIdentity(pid, str(pid * 10), state, process_group, 100)

    snapshots = iter(
        [
            [member(100, "S", 100), member(201, "S", 200)],
            [member(100, "T", 100), member(201, "T", 200), member(301, "T", 300)],
            [member(100, "T", 100), member(201, "T", 200), member(301, "T", 300)],
            [member(100, "T", 100), member(201, "T", 200), member(301, "T", 300)],
        ]
    )
    monkeypatch.setattr(monitor, "_live_session_members", lambda session_id: next(snapshots))
    signals = []
    monkeypatch.setattr(worker_monitor.os, "killpg", lambda group, signum: signals.append((group, signum)))

    async def no_sleep(*args):  # noqa: ANN002, ARG001
        return None

    waited = []

    async def drained(worker_id, pid, ticks, timeout, **kwargs):  # noqa: ANN001, ARG001
        waited.append((worker_id, pid, ticks, kwargs["expected_session_id"], set(kwargs["observed_process_groups"])))
        return True

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(monitor, "_wait_for_process_exit", drained)

    stopped, reason = asyncio.run(
        monitor._force_kill_worker_session(
            "worker_gpu_0",
            pid=100,
            expected_leader_start_ticks="1000",
            session_id=100,
            observed_process_groups={100},
        )
    )

    assert stopped is True
    assert reason == ""
    assert signals == [
        (100, worker_monitor.signal.SIGSTOP),
        (200, worker_monitor.signal.SIGSTOP),
        (100, worker_monitor.signal.SIGSTOP),
        (200, worker_monitor.signal.SIGSTOP),
        (300, worker_monitor.signal.SIGSTOP),
        (100, worker_monitor.signal.SIGKILL),
        (200, worker_monitor.signal.SIGKILL),
        (300, worker_monitor.signal.SIGKILL),
    ]
    assert waited == [("worker_gpu_0", 100, "1000", 100, {100, 200, 300})]


def test_session_force_kill_rejects_non_leader_sid_without_signalling(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)

    monkeypatch.setattr(
        monitor,
        "_live_session_members",
        lambda session_id: (_ for _ in ()).throw(AssertionError("invalid SID must not be scanned")),
    )
    monkeypatch.setattr(
        worker_monitor.os,
        "killpg",
        lambda process_group, signum: (_ for _ in ()).throw(AssertionError("invalid SID must not be signalled")),
    )

    stopped, reason = asyncio.run(
        monitor._force_kill_worker_session(
            "worker_gpu_0",
            pid=100,
            expected_leader_start_ticks="1000",
            session_id=101,
            observed_process_groups={100},
        )
    )

    assert stopped is False
    assert "invalid authenticated session leader identity" in reason


def test_session_freeze_fails_closed_on_uninterruptible_member(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    member = worker_monitor.ProcessIdentity(100, "88", "D", 100, 100)
    monkeypatch.setattr(monitor, "_live_session_members", lambda session_id: [member])
    signals = []
    monkeypatch.setattr(worker_monitor.os, "killpg", lambda group, signum: signals.append((group, signum)))

    frozen, groups, reason = asyncio.run(
        monitor._freeze_worker_session(
            100,
            expected_leader_start_ticks="88",
            timeout=0.0,
        )
    )

    assert frozen is False
    assert groups == {100}
    assert "did not freeze" in reason
    assert signals == [(100, worker_monitor.signal.SIGSTOP)]


def test_spawn_cleanup_leader_exit_race_still_requires_session_drain(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    quarantines = []

    class ExitsBetweenPollAndIdentityRead:
        pid = 43211

        def __init__(self) -> None:
            self.polls = 0
            self.reaped = False

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else -15

        def wait(self, timeout=None):  # noqa: ARG002
            self.reaped = True
            return -15

    process = ExitsBetweenPollAndIdentityRead()
    monitor.spawned_processes["worker_gpu_0"] = process
    monkeypatch.setattr(monitor, "_read_process_identity", lambda pid: None)
    monkeypatch.setattr(worker_monitor.os, "killpg", lambda process_group, signum: None)

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)

    stopped = asyncio.run(monitor._stop_spawned_process(process, "worker_gpu_0", device="cuda:0"))

    assert stopped is False
    assert process.reaped is True
    assert "worker_gpu_0" in monitor.spawned_processes
    assert quarantines == [
        (
            "worker_gpu_0",
            "cuda:0",
            "worker leader PID 43211 exited during cleanup but unauthenticated session 43211 was not proven drained",
        )
    ]


def test_dead_leader_with_undrained_session_quarantines_and_preserves_pid_map(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    process_key = f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"
    redis.hashes[process_key] = {
        b"pid": b"22222",
        b"proc_start_ticks": b"500",
        b"process_group": b"22222",
        b"device": b"cuda:0",
    }
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    quarantines = []

    monkeypatch.setattr(monitor, "_read_process_identity", lambda pid: None)
    monkeypatch.setattr(monitor, "_process_group_is_drained", lambda process_group: False)

    async def capture_quarantine(worker_id, device, reason):  # noqa: ANN001
        quarantines.append((worker_id, device, reason))

    monkeypatch.setattr(monitor, "_quarantine_unsafe_process_group", capture_quarantine)

    stopped = asyncio.run(
        monitor._kill_worker_process(
            "worker_gpu_0",
            expected_pid=22222,
            expected_start_ticks="500",
            device_hint="cuda:0",
        )
    )

    assert stopped is False
    assert process_key in redis.hashes
    assert quarantines == [
        (
            "worker_gpu_0",
            "cuda:0",
            "worker leader PID 22222 exited but session 22222 was not drained: "
            "session 22222 or one of its observed process groups survived SIGKILL",
        )
    ]


def test_unsafe_group_quarantine_pages_and_records_delivery(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    monitor.hostname = "node21"
    writes = []
    updates = []

    async def fake_write(redis, worker_id, **kwargs):  # noqa: ANN001, ARG001
        writes.append((worker_id, kwargs))
        return {
            "state": "quarantined",
            "scope": "physical_gpu",
            "worker_id": worker_id,
            "device": kwargs["device"],
            "hostname": kwargs["hostname"],
            "fault_class": kwargs["fault_class"],
            "page_user_state": "pending",
        }

    class Outcome:
        success = True
        protocol_version = "test"
        error_kind = None
        error = None

    async def fake_page(record):  # noqa: ANN001
        assert record["scope"] == "physical_gpu"
        return Outcome()

    async def fake_update(redis, worker_id, **kwargs):  # noqa: ANN001, ARG001
        updates.append((worker_id, kwargs))

    monkeypatch.setattr(worker_monitor, "write_gpu_quarantine", fake_write)
    monkeypatch.setattr(worker_monitor, "send_gpu_quarantine_page", fake_page)
    monkeypatch.setattr(worker_monitor, "update_gpu_quarantine_notification", fake_update)

    asyncio.run(
        monitor._quarantine_unsafe_process_group(
            "worker_gpu_0",
            "cuda:0",
            "old CUDA child survived leader",
        )
    )

    assert writes[0][1]["physical_scope"] is True
    assert writes[0][1]["fault_class"] == "unsafe_process_group_shutdown"
    assert updates[0][1]["state"] == "sent"


def test_unsafe_group_page_finishes_before_cancellation_propagates(monkeypatch) -> None:
    worker_monitor = load_worker_monitor()
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    monitor.hostname = "node21"
    page_started = asyncio.Event()
    release_page = asyncio.Event()
    delivery_recorded = asyncio.Event()

    async def fake_write(redis, worker_id, **kwargs):  # noqa: ANN001, ARG001
        return {
            "state": "quarantined",
            "scope": "physical_gpu",
            "worker_id": worker_id,
            "device": kwargs["device"],
            "hostname": kwargs["hostname"],
            "page_user_state": "pending",
        }

    class Outcome:
        success = True
        protocol_version = "test"
        error_kind = None
        error = None

    async def delayed_page(record):  # noqa: ANN001, ARG001
        page_started.set()
        await release_page.wait()
        return Outcome()

    async def record_update(redis, worker_id, **kwargs):  # noqa: ANN001, ARG001
        assert kwargs["state"] == "sent"
        delivery_recorded.set()

    monkeypatch.setattr(worker_monitor, "write_gpu_quarantine", fake_write)
    monkeypatch.setattr(worker_monitor, "send_gpu_quarantine_page", delayed_page)
    monkeypatch.setattr(worker_monitor, "update_gpu_quarantine_notification", record_update)

    async def scenario() -> None:
        task = asyncio.create_task(
            monitor._quarantine_unsafe_process_group(
                "worker_gpu_0",
                "cuda:0",
                "old CUDA child survived leader",
            )
        )
        await page_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_page.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert delivery_recorded.is_set()

    asyncio.run(scenario())


def test_single_worker_leaves_process_map_for_monitor_group_proof() -> None:
    source = (ROOT / "kernelgym" / "worker" / "single_worker.py").read_text(encoding="utf-8")

    assert "redis_client.delete" not in source
    assert source.index("await worker.stop()") < source.index("await redis_client.aclose()")


def test_cancellation_after_spawn_registration_reaps_and_removes_exact_map(monkeypatch, tmp_path) -> None:
    worker_monitor = load_worker_monitor()
    redis = FakeRedis()
    monitor = worker_monitor.WorkerMonitor(redis, persistent=True)
    process = ReapableFakeProcess()
    startup_probe_entered = asyncio.Event()
    sleep_calls = 0

    async def kill_confirmed(*args, **kwargs):  # noqa: ARG001
        return True

    async def controlled_sleep(delay):  # noqa: ARG001
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            return
        startup_probe_entered.set()
        await asyncio.Event().wait()

    def fake_popen(*args, **kwargs):  # noqa: ARG001
        return process

    def fake_killpg(pid, signum):
        assert pid == process.pid
        if signum == 0:
            raise ProcessLookupError
        assert signum == worker_monitor.signal.SIGTERM
        process.terminated = True
        process.returncode = -15

    monkeypatch.setattr(monitor, "_kill_worker_process", kill_confirmed)
    monkeypatch.setattr(
        monitor,
        "_read_process_identity",
        lambda pid: worker_monitor.ProcessIdentity(pid, "400", "S", pid, pid),
    )
    monkeypatch.setattr(monitor, "_cmdline_matches_worker", lambda pid, worker_id: True)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(worker_monitor.os, "killpg", fake_killpg)
    monkeypatch.setattr(worker_monitor.settings, "log_dir", tmp_path)
    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    async def scenario() -> None:
        task = asyncio.create_task(monitor._restart_worker("worker_gpu_0", "cuda:0"))
        await startup_probe_entered.wait()
        process_key = f"{worker_monitor.KEY_PREFIX}:worker_process:worker_gpu_0"
        assert redis.hashes[process_key][b"proc_start_ticks"] == b"400"
        assert redis.hashes[process_key][b"session_id"] == str(process.pid).encode()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process_key not in redis.hashes

    asyncio.run(scenario())
    assert process.terminated is True
    assert process.reaped is True
