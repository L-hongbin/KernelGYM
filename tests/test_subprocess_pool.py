import asyncio
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest


SUBPROCESS_POOL_PATH = Path(__file__).resolve().parents[1] / "kernelgym" / "worker" / "subprocess_pool.py"
spec = importlib.util.spec_from_file_location("subprocess_pool_under_test", SUBPROCESS_POOL_PATH)
assert spec is not None and spec.loader is not None
subprocess_pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subprocess_pool)
SubprocessWorkerPool = subprocess_pool.SubprocessWorkerPool


class FakeWorker:
    def __init__(self, worker_id: str, alive: bool = True) -> None:
        self.worker_id = worker_id
        self.tasks_processed = 0
        self.shutdown_called = False
        self._alive = alive
        self.process = None

    def is_alive(self) -> bool:
        return self._alive

    def shutdown(self, timeout: int = 10, force: bool = False) -> bool:  # noqa: ARG002
        self.shutdown_called = True
        self._alive = False
        return True


class _NoOpThread:
    """Captures the target without actually starting a thread."""

    def __init__(self, target=None, daemon: bool = False, **_: object) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True


async def _drain_loop() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


async def _inline_to_thread(function, /, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    """Keep lifecycle tests deterministic when their replenisher Thread is stubbed."""

    return function(*args, **kwargs)


def _pool_without_processes(*, pool_size: int = 1) -> SubprocessWorkerPool:
    pool = SubprocessWorkerPool.__new__(SubprocessWorkerPool)
    pool.device_id = 0
    pool.pool_size = pool_size
    pool.worker_prefix = "test_worker"
    pool.max_tasks_per_worker = 1
    pool.workers = []
    pool.idle_workers = []
    pool.busy_workers = []
    pool.pending_replacements = 0
    pool.pending_retirements = 0
    pool._retiring_worker_ids = set()
    pool._top_up_sequence = 0
    pool._ticket_condition = threading.Condition()
    pool._active_replenishments = {}
    pool._closing = False
    pool._shutdown_task = None
    pool.unsafe_shutdown_reason = ""
    pool.health_state = subprocess_pool.POOL_HEALTHY
    pool.health_reason = ""
    pool.health_task_id = ""
    pool.health_fault_class = ""
    pool.health_scope = "gpu"
    pool.health_epoch = 0
    pool.pool_generation = 0
    pool.hard_recovery_epoch = 0
    pool.speculative_dispatches_remaining = 0
    pool.consecutive_replacement_failures = 0
    pool.max_replacement_failures = 1
    pool.total_tasks_processed = 0
    pool.total_workers_restarted = 0
    pool.pool_start_time = time.time()
    pool.lock = asyncio.Lock()
    pool._recovery_lock = asyncio.Lock()
    return pool


@pytest.fixture(autouse=True)
def _clear_unreaped_worker_registry():
    """Keep process-global fake handles isolated between unit tests."""

    with subprocess_pool._UNREAPED_WORKER_HANDLES_LOCK:
        subprocess_pool._UNREAPED_WORKER_HANDLES.clear()
    with subprocess_pool._ACTIVE_WORKER_IDENTITIES_LOCK:
        subprocess_pool._ACTIVE_WORKER_IDENTITIES.clear()
        subprocess_pool._STARTING_WORKER_IDENTITIES.clear()
    yield
    with subprocess_pool._UNREAPED_WORKER_HANDLES_LOCK:
        subprocess_pool._UNREAPED_WORKER_HANDLES.clear()
    with subprocess_pool._ACTIVE_WORKER_IDENTITIES_LOCK:
        subprocess_pool._ACTIVE_WORKER_IDENTITIES.clear()
        subprocess_pool._STARTING_WORKER_IDENTITIES.clear()


def _worker_with_failed_ready_handshake(get_result, monkeypatch):  # noqa: ANN001
    worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
    worker.worker_id = "handshake_worker"
    worker.device_id = 0
    worker.pool_size_info = "(test)"
    worker.max_tasks_per_worker = 1

    class FakeTaskQueue:
        values: list[str] = []

        def put(self, value):  # noqa: ANN001
            self.values.append(value)

    worker.task_queue = FakeTaskQueue()
    worker.is_alive_flag = True
    worker.tasks_processed = 0
    worker.start_time = time.time()
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = False
    worker._result_channel_closed = False
    worker._process_identity = None
    worker._starting_process_identity = None
    worker._expected_session_id = 90001
    worker.process = None

    class FakeChildResultChannel:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeResultQueue:
        calls = 0

        @classmethod
        def get(cls, timeout):  # noqa: ANN001, ARG004
            cls.calls += 1
            if cls.calls == 1:
                return {
                    "status": "CONTAINED",
                    "pid": 12345,
                    "start_ticks": 678,
                    "pgid": 12345,
                    "sid": 90001,
                }
            return get_result()

    class FakeProcess:
        pid = 12345

        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

    class FakeContext:
        def __init__(self) -> None:
            self.process = FakeProcess()

        def Process(self, **_kwargs):  # noqa: N802, ANN003
            return self.process

    worker.result_queue = FakeResultQueue()
    worker._child_result_channel = FakeChildResultChannel()
    worker.ctx = FakeContext()
    identity = subprocess_pool._LinuxProcessIdentity(
        pid=12345,
        start_ticks=678,
        ppid=os.getpid(),
        pgid=12345,
        sid=90001,
        state="S",
    )
    monkeypatch.setattr(
        subprocess_pool,
        "_read_linux_process_identity",
        lambda pid: identity if pid == identity.pid else None,
    )
    return worker


def test_json_result_channel_rejects_candidate_object_without_reduce_execution() -> None:
    reduce_called = False

    class HostileResult:
        def __reduce__(self):  # noqa: ANN204
            nonlocal reduce_called
            reduce_called = True
            return str, ("executed",)

    class UnexpectedConnection:
        @staticmethod
        def send_bytes(_payload):  # noqa: ANN001
            raise AssertionError("invalid payload must not cross the result pipe")

    channel = subprocess_pool._ChildJSONResultChannel(UnexpectedConnection())
    with pytest.raises(subprocess_pool.WorkerIPCProtocolError, match="non-primitive"):
        channel.send(
            "task_result",
            {
                "success": True,
                "result": {"hostile": HostileResult()},
                "worker_exiting": False,
            },
        )

    assert reduce_called is False


def test_json_result_channel_round_trip_uses_exact_primitives() -> None:
    encoded_messages: list[bytes] = []

    class WriteConnection:
        @staticmethod
        def send_bytes(payload):  # noqa: ANN001
            encoded_messages.append(payload)

    class ReadConnection:
        @staticmethod
        def poll(_timeout):  # noqa: ANN001
            return True

        @staticmethod
        def recv_bytes(maxlength):  # noqa: ANN001
            assert maxlength == subprocess_pool._MAX_RESULT_MESSAGE_BYTES
            return encoded_messages.pop(0)

    child = subprocess_pool._ChildJSONResultChannel(WriteConnection())
    parent = subprocess_pool._ParentJSONResultChannel(ReadConnection())
    child.send(
        "task_result",
        {
            "success": True,
            "result": {"value": 3, "nested": [True, None, 1.25]},
            "worker_exiting": False,
        },
    )

    decoded = parent.get(timeout=0.1, expected_kinds=frozenset({"task_result"}))

    assert decoded == {
        "success": True,
        "result": {"value": 3, "nested": [True, None, 1.25]},
        "worker_exiting": False,
    }
    assert type(decoded) is dict
    assert type(decoded["result"]) is dict
    assert type(decoded["result"]["nested"]) is list


def test_parent_json_result_channel_rejects_oversize_frame() -> None:
    observed_limits: list[int] = []

    class OversizeConnection:
        @staticmethod
        def poll(_timeout):  # noqa: ANN001
            return True

        @staticmethod
        def recv_bytes(maxlength):  # noqa: ANN001
            observed_limits.append(maxlength)
            raise OSError("bad message length")

    parent = subprocess_pool._ParentJSONResultChannel(OversizeConnection(), max_message_bytes=128)
    with pytest.raises(subprocess_pool.WorkerIPCProtocolError, match="bounded child result"):
        parent.get(timeout=0.1, expected_kinds=frozenset({"task_result"}))

    assert observed_limits == [128]


def test_parent_json_result_channel_converts_poll_oserror_to_protocol_error() -> None:
    class BrokenPollConnection:
        @staticmethod
        def poll(_timeout):  # noqa: ANN001
            raise OSError("poll fd failed")

    parent = subprocess_pool._ParentJSONResultChannel(BrokenPollConnection())
    with pytest.raises(subprocess_pool.WorkerIPCProtocolError, match="poll bounded child result"):
        parent.get(timeout=0.1, expected_kinds=frozenset({"task_result"}))


def test_parent_json_result_channel_rejects_wrong_phase_kind() -> None:
    ready_payload = {
        "status": "READY",
        "init_time": 1.0,
        "device": "cuda:0",
        "pid": 10,
        "start_ticks": 20,
        "pgid": 10,
        "sid": 1,
    }
    encoded = subprocess_pool.json.dumps([subprocess_pool._RESULT_PROTOCOL_MAGIC, "ready", ready_payload]).encode()

    class Connection:
        @staticmethod
        def poll(_timeout):  # noqa: ANN001
            return True

        @staticmethod
        def recv_bytes(_maxlength):  # noqa: ANN001
            return encoded

    parent = subprocess_pool._ParentJSONResultChannel(Connection())
    with pytest.raises(subprocess_pool.WorkerIPCProtocolError, match="unexpected child result kind"):
        parent.get(timeout=0.1, expected_kinds=frozenset({"task_result"}))


def test_parent_json_result_channel_rejects_invalid_schema() -> None:
    encoded = subprocess_pool.json.dumps(
        [
            subprocess_pool._RESULT_PROTOCOL_MAGIC,
            "task_result",
            {"success": "yes", "result": {}, "worker_exiting": False},
        ]
    ).encode()

    class Connection:
        @staticmethod
        def poll(_timeout):  # noqa: ANN001
            return True

        @staticmethod
        def recv_bytes(_maxlength):  # noqa: ANN001
            return encoded

    parent = subprocess_pool._ParentJSONResultChannel(Connection())
    with pytest.raises(subprocess_pool.WorkerIPCProtocolError, match="schema"):
        parent.get(timeout=0.1, expected_kinds=frozenset({"task_result"}))


def test_ready_handshake_arbitrary_exception_force_reaps_before_reraise(monkeypatch) -> None:
    def fail_get():
        raise OSError("broken result queue")

    worker = _worker_with_failed_ready_handshake(fail_get, monkeypatch)
    shutdown_calls: list[tuple[int, bool]] = []

    def shutdown(timeout: int, force: bool) -> bool:
        shutdown_calls.append((timeout, force))
        return True

    worker.shutdown = shutdown
    monkeypatch.setattr(subprocess_pool, "prepare_core_dump_dir", lambda *_args, **_kwargs: None)

    with pytest.raises(OSError, match="broken result queue"):
        worker._start_worker()

    assert worker.process.started is True
    assert shutdown_calls == [(5, True)]
    assert subprocess_pool._snapshot_unreaped_workers(0) == []


def test_ready_handshake_timeout_is_gpu_probe_failure(monkeypatch) -> None:
    def timeout_get():
        raise subprocess_pool.queue.Empty

    worker = _worker_with_failed_ready_handshake(timeout_get, monkeypatch)
    shutdown_calls: list[tuple[int, bool]] = []

    def shutdown(timeout: int, force: bool) -> bool:
        shutdown_calls.append((timeout, force))
        return True

    worker.shutdown = shutdown
    monkeypatch.setattr(subprocess_pool, "prepare_core_dump_dir", lambda *_args, **_kwargs: None)

    with pytest.raises(subprocess_pool.WorkerInitializationError) as error:
        worker._start_worker()

    assert error.value.init_stage == "handshake_timeout"
    assert error.value.cuda_probe_failure is True
    assert error.value.reap_confirmed is True
    assert shutdown_calls == [(5, True)]


def test_ready_handshake_unconfirmed_reap_retains_process_handle(monkeypatch) -> None:
    def malformed_get():
        return "not-a-ready-message"

    worker = _worker_with_failed_ready_handshake(malformed_get, monkeypatch)

    def failed_shutdown(timeout: int, force: bool) -> bool:  # noqa: ARG001
        return False

    worker.shutdown = failed_shutdown
    monkeypatch.setattr(subprocess_pool, "prepare_core_dump_dir", lambda *_args, **_kwargs: None)

    with pytest.raises(subprocess_pool.WorkerInitializationError) as error:
        worker._start_worker()

    assert error.value.reap_confirmed is False
    assert error.value.cuda_probe_failure is True
    assert subprocess_pool._snapshot_unreaped_workers(0) == [worker]


def test_worker_exits_when_parent_containment_ack_times_out(monkeypatch) -> None:
    messages = []
    real_pid = os.getpid()
    real_sid = os.getsid(0)
    identity = subprocess_pool._LinuxProcessIdentity(
        pid=real_pid,
        start_ticks=123,
        ppid=os.getppid(),
        pgid=real_pid,
        sid=real_sid,
        state="S",
    )

    class TaskQueue:
        @staticmethod
        def get(timeout):  # noqa: ANN001
            assert timeout == subprocess_pool._PARENT_CONTAINMENT_ACK_TIMEOUT_S
            raise subprocess_pool.queue.Empty

    class ResultQueue:
        @staticmethod
        def put(payload):  # noqa: ANN001
            messages.append(payload)

    monkeypatch.setattr(subprocess_pool, "prepare_core_dump_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(subprocess_pool, "_redirect_native_stderr_to_capture_file", lambda worker_id: None)
    monkeypatch.setattr(subprocess_pool.os, "setpgid", lambda pid, pgid: None)
    monkeypatch.setattr(subprocess_pool, "_read_linux_process_identity", lambda pid: identity)

    subprocess_pool._persistent_worker_loop("ack-timeout", 0, TaskQueue(), ResultQueue(), 1)

    assert [message["status"] for message in messages] == ["CONTAINED", "INIT_FAILED"]
    assert messages[-1]["init_stage"] == "parent_containment_ack"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process containment")
def test_persistent_worker_shutdown_reaps_forked_process_group_descendant() -> None:
    child_code = """
import os
import time
os.setpgid(0, 0)
if os.fork() == 0:
    time.sleep(60)
else:
    print('READY', flush=True)
    time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        text=True,
    )

    class ProcessAdapter:
        pid = process.pid

        @staticmethod
        def is_alive() -> bool:
            return process.poll() is None

        @staticmethod
        def join(timeout: float) -> None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

        @staticmethod
        def kill() -> None:
            process.kill()

        @staticmethod
        def close() -> None:
            return None

    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        identity = subprocess_pool._read_linux_process_identity(process.pid)
        assert identity is not None
        assert identity.pgid == process.pid
        snapshot = subprocess_pool._snapshot_linux_processes()
        descendants = subprocess_pool._descendants_from_snapshot(snapshot, process.pid)
        assert descendants

        worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
        worker.worker_id = "forking-worker"
        worker.device_id = 0
        worker.process = ProcessAdapter()
        worker._process_identity = identity
        worker._expected_session_id = os.getsid(0)
        worker.is_alive_flag = True
        worker.tasks_processed = 0
        worker.start_time = time.time()
        worker._shutdown_lock = threading.Lock()
        worker._shutdown_complete = False

        assert worker.shutdown(timeout=5, force=True) is True
        assert process.poll() is not None
        assert subprocess_pool._process_group_is_drained(identity.pgid) is True
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def test_shutdown_returns_false_when_process_group_proof_fails(monkeypatch) -> None:
    identity = subprocess_pool._LinuxProcessIdentity(
        pid=43123,
        start_ticks=99,
        ppid=os.getpid(),
        pgid=43123,
        sid=os.getsid(0),
        state="S",
    )

    class FakeProcess:
        pid = identity.pid

        def __init__(self) -> None:
            self.alive = True
            self.closed = False

        def join(self, timeout):  # noqa: ANN001, ARG002
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

        def kill(self) -> None:
            self.alive = False

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
    worker.worker_id = "unproven-group"
    worker.device_id = 0
    worker.process = process
    worker._process_identity = identity
    worker._expected_session_id = identity.sid
    worker.is_alive_flag = True
    worker.tasks_processed = 0
    worker.start_time = time.time()
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = False
    monkeypatch.setattr(subprocess_pool, "_kill_and_verify_worker_process_tree", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(subprocess_pool, "_wait_for_process_group_drain", lambda *_args, **_kwargs: False)

    assert worker.shutdown(timeout=1, force=True) is False
    assert process.alive is False
    assert process.closed is False
    assert subprocess_pool._snapshot_unreaped_workers(0) == [worker]


def test_shutdown_accepts_crashed_leader_only_after_join_and_group_drain(monkeypatch) -> None:
    identity = subprocess_pool._LinuxProcessIdentity(
        pid=43124,
        start_ticks=100,
        ppid=os.getpid(),
        pgid=43124,
        sid=os.getsid(0),
        state="S",
    )

    class FakeProcess:
        pid = identity.pid

        def __init__(self) -> None:
            self.joined = False
            self.closed = False

        def join(self, timeout):  # noqa: ANN001, ARG002
            self.joined = True

        @staticmethod
        def is_alive() -> bool:
            return False

        @staticmethod
        def kill() -> None:
            raise AssertionError("an exited PID generation must never be signalled")

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
    worker.worker_id = "crashed-before-shutdown"
    worker.device_id = 0
    worker.process = process
    worker._process_identity = identity
    worker._expected_session_id = identity.sid
    worker.is_alive_flag = True
    worker.tasks_processed = 0
    worker.start_time = time.time()
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = False
    monkeypatch.setattr(subprocess_pool, "_read_linux_process_identity", lambda _pid: None)
    monkeypatch.setattr(
        subprocess_pool,
        "_kill_and_verify_worker_process_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("there is no live leader generation to freeze or signal")
        ),
    )
    drain_calls = []
    monkeypatch.setattr(
        subprocess_pool,
        "_wait_for_process_group_drain_or_registered_reuse",
        lambda leader, timeout: drain_calls.append((leader, timeout)) or True,
    )
    session_calls = []
    monkeypatch.setattr(
        subprocess_pool,
        "_wait_for_worker_session_containment",
        lambda leader, timeout: session_calls.append((leader, timeout)) or True,
    )

    assert worker.shutdown(timeout=1, force=True) is True
    assert process.joined is True
    assert process.closed is True
    assert drain_calls == [(identity, 3.0)]
    assert session_calls == [(identity, 3.0)]
    assert subprocess_pool._snapshot_unreaped_workers(0) == []


def test_crashed_leader_session_proof_rejects_unknown_cross_pgid_member(monkeypatch) -> None:
    session_id = 99001
    outer = subprocess_pool._LinuxProcessIdentity(
        pid=os.getpid(),
        start_ticks=1,
        ppid=1,
        pgid=session_id,
        sid=session_id,
        state="S",
    )
    leader = subprocess_pool._LinuxProcessIdentity(
        pid=43125,
        start_ticks=101,
        ppid=outer.pid,
        pgid=43125,
        sid=session_id,
        state="S",
    )
    sibling = subprocess_pool._LinuxProcessIdentity(
        pid=43126,
        start_ticks=102,
        ppid=outer.pid,
        pgid=43126,
        sid=session_id,
        state="S",
    )
    sibling_child = subprocess_pool._LinuxProcessIdentity(
        pid=43127,
        start_ticks=103,
        ppid=sibling.pid,
        pgid=sibling.pgid,
        sid=session_id,
        state="S",
    )
    escaped_child = subprocess_pool._LinuxProcessIdentity(
        pid=43128,
        start_ticks=104,
        ppid=1,
        pgid=43128,
        sid=session_id,
        state="S",
    )
    joined_sibling_group = subprocess_pool._LinuxProcessIdentity(
        pid=43129,
        start_ticks=105,
        ppid=1,
        pgid=sibling.pgid,
        sid=session_id,
        state="S",
    )
    snapshot = {item.pid: item for item in (outer, sibling, sibling_child)}
    monkeypatch.setattr(subprocess_pool, "_snapshot_linux_processes", lambda: dict(snapshot))
    monkeypatch.setattr(subprocess_pool, "_multiprocessing_resource_tracker_pid", lambda: None)
    provisional_sibling = subprocess_pool._LinuxProcessIdentity(
        pid=sibling.pid,
        start_ticks=sibling.start_ticks,
        ppid=sibling.ppid,
        pgid=outer.pgid,
        sid=sibling.sid,
        state="S",
    )
    subprocess_pool._register_starting_worker_identity(provisional_sibling)
    subprocess_pool._promote_active_worker_identity(provisional_sibling, sibling)

    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is True

    snapshot[escaped_child.pid] = escaped_child
    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is False

    snapshot.pop(escaped_child.pid)
    snapshot[joined_sibling_group.pid] = joined_sibling_group
    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is False


def test_crashed_leader_session_proof_allows_only_exact_starting_worker(monkeypatch) -> None:
    session_id = 99002
    outer = subprocess_pool._LinuxProcessIdentity(
        pid=os.getpid(), start_ticks=1, ppid=1, pgid=session_id, sid=session_id, state="S"
    )
    leader = subprocess_pool._LinuxProcessIdentity(
        pid=43200, start_ticks=2, ppid=outer.pid, pgid=43200, sid=session_id, state="S"
    )
    starting = subprocess_pool._LinuxProcessIdentity(
        pid=43201, start_ticks=3, ppid=outer.pid, pgid=outer.pgid, sid=session_id, state="S"
    )
    unknown = subprocess_pool._LinuxProcessIdentity(
        pid=43202, start_ticks=4, ppid=outer.pid, pgid=outer.pgid, sid=session_id, state="S"
    )
    snapshot = {outer.pid: outer, starting.pid: starting}
    monkeypatch.setattr(subprocess_pool, "_snapshot_linux_processes", lambda: dict(snapshot))
    monkeypatch.setattr(subprocess_pool, "_multiprocessing_resource_tracker_pid", lambda: None)
    subprocess_pool._register_starting_worker_identity(starting)

    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is True

    snapshot[unknown.pid] = unknown
    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is False


def test_crashed_leader_session_proof_allows_authenticated_resource_tracker(monkeypatch) -> None:
    session_id = 99003
    outer = subprocess_pool._LinuxProcessIdentity(
        pid=os.getpid(), start_ticks=1, ppid=1, pgid=session_id, sid=session_id, state="S"
    )
    leader = subprocess_pool._LinuxProcessIdentity(
        pid=43300, start_ticks=2, ppid=outer.pid, pgid=43300, sid=session_id, state="S"
    )
    tracker = subprocess_pool._LinuxProcessIdentity(
        pid=43301, start_ticks=3, ppid=outer.pid, pgid=outer.pgid, sid=session_id, state="S"
    )
    monkeypatch.setattr(
        subprocess_pool,
        "_snapshot_linux_processes",
        lambda: {outer.pid: outer, tracker.pid: tracker},
    )
    monkeypatch.setattr(subprocess_pool, "_multiprocessing_resource_tracker_pid", lambda: tracker.pid)

    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is True


def test_failed_crash_containment_keeps_registration_until_successful_retry(monkeypatch) -> None:
    identity = subprocess_pool._LinuxProcessIdentity(
        pid=43400,
        start_ticks=5,
        ppid=os.getpid(),
        pgid=43400,
        sid=os.getsid(0),
        state="S",
    )
    provisional = subprocess_pool._LinuxProcessIdentity(
        pid=identity.pid,
        start_ticks=identity.start_ticks,
        ppid=identity.ppid,
        pgid=os.getpgrp(),
        sid=identity.sid,
        state="S",
    )

    class FakeProcess:
        pid = identity.pid

        def __init__(self) -> None:
            self.closed = False

        @staticmethod
        def join(timeout):  # noqa: ANN001, ARG004
            return None

        @staticmethod
        def is_alive() -> bool:
            return False

        @staticmethod
        def kill() -> None:
            raise AssertionError("an exited PID generation must never be signalled")

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
    worker.worker_id = "retry-crash-containment"
    worker.device_id = 0
    worker.process = process
    worker._process_identity = identity
    worker._starting_process_identity = provisional
    worker._expected_session_id = identity.sid
    worker.is_alive_flag = True
    worker.tasks_processed = 0
    worker.start_time = time.time()
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = False
    subprocess_pool._register_starting_worker_identity(provisional)
    subprocess_pool._promote_active_worker_identity(provisional, identity)
    monkeypatch.setattr(subprocess_pool, "_read_linux_process_identity", lambda _pid: None)
    monkeypatch.setattr(
        subprocess_pool,
        "_wait_for_process_group_drain_or_registered_reuse",
        lambda *_args: True,
    )
    containment_results = iter([False, True])
    monkeypatch.setattr(
        subprocess_pool,
        "_wait_for_worker_session_containment",
        lambda *_args: next(containment_results),
    )

    assert worker.shutdown(timeout=1, force=True) is False
    assert subprocess_pool._active_worker_identities(identity.sid) == [identity]
    assert process.closed is False

    assert worker.shutdown(timeout=1, force=True) is True
    assert subprocess_pool._active_worker_identities(identity.sid) == []
    assert process.closed is True


def test_shutdown_rejects_reused_pid_generation_and_preserves_new_worker(monkeypatch) -> None:
    session_id = 99004
    starting = subprocess_pool._LinuxProcessIdentity(
        pid=43500, start_ticks=10, ppid=os.getpid(), pgid=os.getpgrp(), sid=session_id, state="S"
    )
    reused = subprocess_pool._LinuxProcessIdentity(
        pid=starting.pid,
        start_ticks=11,
        ppid=os.getpid(),
        pgid=starting.pid,
        sid=session_id,
        state="S",
    )

    class FakeProcess:
        pid = starting.pid

        def __init__(self) -> None:
            self.closed = False

        @staticmethod
        def join(timeout):  # noqa: ANN001, ARG004
            return None

        @staticmethod
        def is_alive() -> bool:
            return False

        @staticmethod
        def kill() -> None:
            raise AssertionError("the stale Process handle must not signal a reused PID")

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
    worker.worker_id = "reused-startup-pid"
    worker.device_id = 0
    worker.process = process
    worker._process_identity = None
    worker._starting_process_identity = starting
    worker._expected_session_id = session_id
    worker.is_alive_flag = True
    worker.tasks_processed = 0
    worker.start_time = time.time()
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = False
    subprocess_pool._register_starting_worker_identity(starting)
    with subprocess_pool._ACTIVE_WORKER_IDENTITIES_LOCK:
        subprocess_pool._ACTIVE_WORKER_IDENTITIES.setdefault(session_id, {})[reused.pid] = reused
    monkeypatch.setattr(subprocess_pool, "_read_linux_process_identity", lambda pid: reused)
    monkeypatch.setattr(
        subprocess_pool,
        "_kill_and_verify_worker_process_tree",
        lambda *_args: (_ for _ in ()).throw(AssertionError("a reused generation must never be adopted")),
    )
    audits = []
    monkeypatch.setattr(
        subprocess_pool,
        "_wait_for_worker_session_containment",
        lambda leader, timeout: audits.append((leader, timeout)) or True,
    )

    assert worker.shutdown(timeout=1, force=True) is True
    assert worker._process_identity is None
    assert audits == [(starting, 3.0)]
    assert subprocess_pool._active_worker_identities(session_id) == [reused]
    assert process.closed is True


def test_uncontained_bootstrap_exit_recovers_after_clean_session_audit(monkeypatch) -> None:
    starting = subprocess_pool._LinuxProcessIdentity(
        pid=43600,
        start_ticks=12,
        ppid=os.getpid(),
        pgid=os.getpgrp(),
        sid=os.getsid(0),
        state="S",
    )

    class FakeProcess:
        pid = starting.pid

        def __init__(self) -> None:
            self.closed = False

        @staticmethod
        def join(timeout):  # noqa: ANN001, ARG004
            return None

        @staticmethod
        def is_alive() -> bool:
            return False

        @staticmethod
        def kill() -> None:
            raise AssertionError("the exited owned child must not be signalled")

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
    worker.worker_id = "bootstrap-exit"
    worker.device_id = 0
    worker.process = process
    worker._process_identity = None
    worker._starting_process_identity = starting
    worker._expected_session_id = starting.sid
    worker.is_alive_flag = True
    worker.tasks_processed = 0
    worker.start_time = time.time()
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = False
    subprocess_pool._register_starting_worker_identity(starting)
    monkeypatch.setattr(subprocess_pool, "_read_linux_process_identity", lambda pid: starting)
    monkeypatch.setattr(
        subprocess_pool,
        "_kill_and_verify_worker_process_tree",
        lambda *_args: (_ for _ in ()).throw(AssertionError("an unauthenticated outer PGID must not be signalled")),
    )
    monkeypatch.setattr(subprocess_pool, "_wait_for_worker_session_containment", lambda *_args: True)

    assert worker.shutdown(timeout=1, force=True) is True
    assert subprocess_pool._starting_worker_identities(starting.sid) == []
    assert process.closed is True


def test_reused_process_group_generation_is_allowed_only_when_registered(monkeypatch) -> None:
    leader = subprocess_pool._LinuxProcessIdentity(
        pid=43700, start_ticks=13, ppid=os.getpid(), pgid=43700, sid=99005, state="S"
    )
    reused = subprocess_pool._LinuxProcessIdentity(
        pid=leader.pid,
        start_ticks=14,
        ppid=os.getpid(),
        pgid=leader.pgid,
        sid=leader.sid,
        state="S",
    )
    with subprocess_pool._ACTIVE_WORKER_IDENTITIES_LOCK:
        subprocess_pool._ACTIVE_WORKER_IDENTITIES.setdefault(leader.sid, {})[reused.pid] = reused
    monkeypatch.setattr(subprocess_pool, "_read_linux_process_identity", lambda pid: reused)
    monkeypatch.setattr(subprocess_pool, "_process_group_is_drained", lambda pgid: False)
    outer = subprocess_pool._LinuxProcessIdentity(
        pid=os.getpid(), start_ticks=1, ppid=1, pgid=leader.sid, sid=leader.sid, state="S"
    )
    old_group_survivor = subprocess_pool._LinuxProcessIdentity(
        pid=43701,
        start_ticks=15,
        ppid=1,
        pgid=reused.pgid,
        sid=leader.sid,
        state="S",
    )
    snapshot = {outer.pid: outer, reused.pid: reused}
    monkeypatch.setattr(
        subprocess_pool,
        "_snapshot_linux_processes",
        lambda: dict(snapshot),
    )
    monkeypatch.setattr(subprocess_pool, "_multiprocessing_resource_tracker_pid", lambda: None)

    assert subprocess_pool._wait_for_process_group_drain_or_registered_reuse(leader, 0.01) is True
    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is True

    snapshot[old_group_survivor.pid] = old_group_survivor
    assert subprocess_pool._worker_session_is_contained_after_leader_exit(leader) is False

    with subprocess_pool._ACTIVE_WORKER_IDENTITIES_LOCK:
        subprocess_pool._ACTIVE_WORKER_IDENTITIES.clear()
    assert subprocess_pool._wait_for_process_group_drain_or_registered_reuse(leader, 0.01) is False


def test_shutdown_closes_result_read_fd_once_after_proven_exit() -> None:
    close_calls = 0

    class ResultChannel:
        @staticmethod
        def close() -> None:
            nonlocal close_calls
            close_calls += 1

    worker = subprocess_pool.PersistentWorker.__new__(subprocess_pool.PersistentWorker)
    worker.worker_id = "already-stopped"
    worker.device_id = 0
    worker.process = None
    worker.result_queue = ResultChannel()
    worker._result_channel_closed = False
    worker.is_alive_flag = False
    worker.tasks_processed = 0
    worker.start_time = time.time()
    worker._shutdown_lock = threading.Lock()
    worker._shutdown_complete = False

    assert worker.shutdown(timeout=1, force=True) is True
    assert worker.shutdown(timeout=1, force=True) is True
    assert close_calls == 1


def test_get_idle_worker_waits_for_pending_replacement(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes()
        pool.pending_replacements = 1

        def fail_if_emergency_worker_starts(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("emergency worker should not start while replacement is pending")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fail_if_emergency_worker_starts)

        assert await pool._get_idle_worker(timeout=0.2) is None
        assert pool.pending_replacements == 1

    asyncio.run(scenario())


def test_restart_worker_does_not_grow_pool_past_configured_size() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old_worker = FakeWorker("old", alive=False)
        existing_worker = FakeWorker("existing", alive=True)
        pool.workers = [existing_worker, old_worker]
        pool.idle_workers = [existing_worker]
        pool.busy_workers = [old_worker]

        await pool._restart_worker(old_worker)  # type: ignore[arg-type]

        assert pool.workers == [existing_worker]
        assert pool.idle_workers == [existing_worker]
        assert pool.busy_workers == []
        assert pool.pending_replacements == 0
        assert old_worker.shutdown_called is True

    asyncio.run(scenario())


def test_pool_size_2_recycle_leaves_spare_idle(monkeypatch) -> None:
    """When one of two workers recycles, the other stays idle and the
    replacement is scheduled in the background."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        spare = FakeWorker("spare", alive=True)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [spare, recycling]
        pool.idle_workers = [spare]
        pool.busy_workers = [recycling]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)
        monkeypatch.setattr(subprocess_pool.asyncio, "to_thread", _inline_to_thread)

        await pool._restart_worker(recycling)  # type: ignore[arg-type]

        # The live spare is still tracked and immediately available.
        assert pool.workers == [spare]
        assert pool.idle_workers == [spare]
        assert pool.busy_workers == []
        assert recycling.shutdown_called is True
        # One replacement is pending and a background thread was queued.
        assert pool.pending_replacements == 1
        assert len(threads) == 1 and threads[0].started is True
        # Invariant: workers + pending must never exceed pool_size.
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_normal_recycle_waits_for_reap_while_existing_spare_dispatches(monkeypatch) -> None:
    """Task return waits for old-child proof without blocking the warm spare."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        spare = FakeWorker("spare", alive=True)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [spare, recycling]
        pool.idle_workers = [spare]
        pool.busy_workers = [recycling]
        shutdown_started = threading.Event()
        allow_shutdown = threading.Event()
        replenishments: list[dict[str, object]] = []

        def blocking_shutdown(timeout: int = 10, force: bool = False) -> bool:  # noqa: ARG001
            shutdown_started.set()
            assert allow_shutdown.wait(timeout=5)
            recycling.shutdown_called = True
            recycling._alive = False
            return True

        recycling.shutdown = blocking_shutdown  # type: ignore[method-assign]
        monkeypatch.setattr(pool, "_start_replenishment_thread", lambda **kwargs: replenishments.append(kwargs))

        restart_task = asyncio.create_task(
            pool._restart_worker(recycling, task_id="normal-recycle")  # type: ignore[arg-type]
        )
        for _ in range(100):
            if shutdown_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert shutdown_started.is_set()
        assert restart_task.done() is False

        # Detach happens before the blocking reap, so the already-warm spare
        # remains independently dispatchable.
        assert await pool._get_idle_worker(timeout=0.2) is spare  # type: ignore[comparison-overlap]

        allow_shutdown.set()
        await asyncio.wait_for(restart_task, timeout=2)
        assert recycling.shutdown_called is True
        assert len(replenishments) == 1
        assert replenishments[0]["old_worker"] is None
        assert pool.pending_retirements == 0

    asyncio.run(scenario())


def test_normal_recycle_unconfirmed_reap_is_unsafe_and_never_replenishes(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        recycling = FakeWorker("unreaped-recycle", alive=False)
        recycling.shutdown = lambda *_args, **_kwargs: False  # type: ignore[method-assign]
        pool.workers = [recycling]
        pool.busy_workers = [recycling]
        replenishments: list[dict[str, object]] = []
        monkeypatch.setattr(pool, "_start_replenishment_thread", lambda **kwargs: replenishments.append(kwargs))

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._restart_worker(
                recycling,  # type: ignore[arg-type]
                task_id="unreaped-task",
            )

        assert error.value.task_id == "unreaped-task"
        assert error.value.worker_id == "unreaped-recycle"
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.unsafe_shutdown_reason
        assert pool.pending_retirements == 0
        assert replenishments == []
        assert subprocess_pool._snapshot_unreaped_workers(0) == [recycling]

    asyncio.run(scenario())


def test_pool_size_2_two_concurrent_recycles_respect_capacity(monkeypatch) -> None:
    """Even when both workers recycle in a tight window, workers+pending
    stays bounded by pool_size."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        a = FakeWorker("a", alive=False)
        b = FakeWorker("b", alive=False)
        pool.workers = [a, b]
        pool.idle_workers = []
        pool.busy_workers = [a, b]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)
        monkeypatch.setattr(subprocess_pool.asyncio, "to_thread", _inline_to_thread)

        await pool._restart_worker(a)  # type: ignore[arg-type]
        await pool._restart_worker(b)  # type: ignore[arg-type]

        # Both workers were removed; two replacements are queued.
        assert pool.workers == []
        assert pool.busy_workers == []
        assert pool.pending_replacements == 2
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size
        # Each recycle queued exactly one background thread.
        assert len(threads) == 2
        assert all(t.started for t in threads)

    asyncio.run(scenario())


def test_pool_size_2_recycle_at_capacity_shuts_down_extra() -> None:
    """If a worker recycles while a replacement is already pending and the
    remaining workers already meet pool_size, no extra replacement is
    scheduled and the recycled worker is shut down synchronously.

    No threading monkeypatch here — the synchronous-shutdown path uses
    asyncio.to_thread, which would deadlock if threading.Thread were stubbed.
    Instead we infer "no background thread spawned" from the fact that the
    `should_replenish=False` branch is mutually exclusive with the thread
    branch.
    """

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        live = FakeWorker("live", alive=True)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [live, recycling]
        pool.idle_workers = []
        pool.busy_workers = [live, recycling]
        # Simulate an in-flight replacement from a previous recycle.
        pool.pending_replacements = 1

        await pool._restart_worker(recycling)  # type: ignore[arg-type]

        # `live` stays tracked; recycling was removed.
        assert pool.workers == [live]
        # Pending unchanged: 1 (live) + 1 (in-flight) == pool_size=2 already.
        assert pool.pending_replacements == 1
        # The recycled worker was shut down synchronously instead.
        assert recycling.shutdown_called is True
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_pool_size_2_get_idle_returns_spare_during_recycle(monkeypatch) -> None:
    """While one worker is being replaced in the background, the warm spare
    is handed out immediately to the next request — no emergency path."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        spare = FakeWorker("spare", alive=True)
        pool.workers = [spare]
        pool.idle_workers = [spare]
        pool.busy_workers = []
        # Mark one replacement as in-flight (the previously-recycled worker).
        pool.pending_replacements = 1

        def fail_if_emergency_worker_starts(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("emergency worker must not start; spare is idle")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fail_if_emergency_worker_starts)

        worker = await pool._get_idle_worker(timeout=1.0)

        assert worker is spare
        assert pool.busy_workers == [spare]
        assert pool.idle_workers == []
        # pending unchanged — emergency must not have fired.
        assert pool.pending_replacements == 1

    asyncio.run(scenario())


def test_pool_size_2_get_idle_tops_up_when_only_one_worker_remains(monkeypatch) -> None:
    """Checking out the only idle worker must start a top-up immediately.

    Otherwise a busy worker can sit alone until it returns, which defeats the
    warm-spare invariant under sustained load.
    """

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        only_worker = FakeWorker("only", alive=True)
        pool.workers = [only_worker]
        pool.idle_workers = [only_worker]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)
        monkeypatch.setattr(subprocess_pool.asyncio, "to_thread", _inline_to_thread)

        worker = await pool._get_idle_worker(timeout=1.0)

        assert worker is only_worker
        assert pool.busy_workers == [only_worker]
        assert pool.idle_workers == []
        assert pool.pending_replacements == 1
        assert len(threads) == 1
        assert threads[0].started is True
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_pool_size_2_no_emergency_when_pending_already_in_flight(monkeypatch) -> None:
    """With pool_size=2 and pending_replacements=2 (both replacements in
    flight), _get_idle_worker waits without spawning an emergency."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        pool.pending_replacements = 2

        def fail_if_emergency_worker_starts(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("emergency worker should not start while replacements are pending")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fail_if_emergency_worker_starts)

        assert await pool._get_idle_worker(timeout=0.2) is None
        assert pool.pending_replacements == 2

    asyncio.run(scenario())


def test_pool_size_2_degraded_recycle_schedules_top_up(monkeypatch) -> None:
    """If the pool has already degraded to one worker, a recycle should
    schedule the direct replacement plus another top-up spare."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [recycling]
        pool.busy_workers = [recycling]

        created: list[FakeWorker] = []

        def fake_persistent_worker(worker_id, *_args, **_kwargs):  # noqa: ANN001
            worker = FakeWorker(worker_id, alive=True)
            created.append(worker)
            return worker

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fake_persistent_worker)
        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)
        monkeypatch.setattr(subprocess_pool.asyncio, "to_thread", _inline_to_thread)

        await pool._restart_worker(recycling)  # type: ignore[arg-type]
        assert pool.pending_replacements == 1
        assert len(threads) == 1

        threads[0].target()
        await _drain_loop()

        assert pool.workers == [created[0]]
        assert pool.idle_workers == [created[0]]
        assert pool.pending_replacements == 1
        assert len(threads) == 2
        assert threads[1].started is True
        assert len(pool.workers) + pool.pending_replacements == pool.pool_size

    asyncio.run(scenario())


def test_pool_size_2_failed_replacement_quarantines_without_retry_storm(monkeypatch) -> None:
    """A fresh-context init failure fails closed without recursive top-up."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        recycling = FakeWorker("recycling", alive=False)
        pool.workers = [recycling]
        pool.busy_workers = [recycling]

        def fake_persistent_worker(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("spawn failed")

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            t = _NoOpThread(*args, **kwargs)
            threads.append(t)
            return t

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fake_persistent_worker)
        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)
        monkeypatch.setattr(subprocess_pool.asyncio, "to_thread", _inline_to_thread)

        await pool._restart_worker(recycling)  # type: ignore[arg-type]
        assert pool.pending_replacements == 1
        assert len(threads) == 1

        threads[0].target()
        await _drain_loop()

        assert pool.workers == []
        assert pool.idle_workers == []
        assert pool.pending_replacements == 0
        assert len(threads) == 1
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.accepting_tasks is False

    asyncio.run(scenario())


def test_return_worker_does_not_revive_recycling_worker(monkeypatch) -> None:
    """The task finally block must not return an untracked recycled worker."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        recycling = FakeWorker("recycling", alive=True)
        pool.workers = [recycling]
        pool.busy_workers = [recycling]

        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            thread = _NoOpThread(*args, **kwargs)
            threads.append(thread)
            return thread

        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)
        monkeypatch.setattr(subprocess_pool.asyncio, "to_thread", _inline_to_thread)

        # _restart_worker removes the worker immediately but must prove the old
        # CUDA child is reaped before returning. Only replacement construction
        # remains captured in the background thread.
        await pool._restart_worker(recycling)  # type: ignore[arg-type]
        assert recycling.is_alive() is False
        assert recycling.shutdown_called is True
        assert recycling not in pool.workers
        assert len(threads) == 1 and threads[0].started is True

        # This is the execute_task finally path that used to resurrect it.
        await pool._return_worker(recycling)  # type: ignore[arg-type]

        assert recycling not in pool.workers
        assert recycling not in pool.idle_workers
        assert recycling not in pool.busy_workers
        assert pool.pending_replacements == 1

    asyncio.run(scenario())


def test_get_idle_worker_quarantines_live_untracked_processes(monkeypatch) -> None:
    """A live CUDA process may never be silently dropped from pool ownership."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        dead_idle = FakeWorker("dead-idle", alive=False)
        dead_busy = FakeWorker("dead-busy", alive=False)
        untracked_idle = FakeWorker("untracked-idle", alive=True)
        untracked_busy = FakeWorker("untracked-busy", alive=True)
        pool.workers = [dead_idle, dead_busy]
        pool.idle_workers = [dead_idle, untracked_idle]
        pool.busy_workers = [dead_busy, untracked_busy]

        replenishments: list[dict[str, object]] = []

        def capture_replenishment(**kwargs):  # noqa: ANN003
            replenishments.append(kwargs)

        # Keep the real thread factory available to ``asyncio.to_thread``;
        # this path must reap the live, untracked CUDA processes before it can
        # report quarantine.
        monkeypatch.setattr(pool, "_start_replenishment_thread", capture_replenishment)

        with pytest.raises(subprocess_pool.GPUQuarantinedError):
            await pool._get_idle_worker(timeout=0.2)
        assert pool.workers == []
        assert pool.idle_workers == []
        assert pool.busy_workers == []
        assert pool.pending_replacements == 0
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert all(item.shutdown_called for item in [dead_idle, dead_busy, untracked_idle, untracked_busy])
        assert replenishments == []

    asyncio.run(scenario())


def test_get_idle_worker_reaps_dead_idle_before_top_up(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        dead_idle = FakeWorker("dead-idle", alive=False)
        pool.workers = [dead_idle]
        pool.idle_workers = [dead_idle]

        replenishments: list[dict[str, object]] = []

        def capture_replenishment(**kwargs):  # noqa: ANN003
            replenishments.append(kwargs)

        # The dead worker reap itself runs through ``asyncio.to_thread``.
        # Capture only the pool's explicit replenishment hook rather than
        # replacing Python's global thread implementation.
        monkeypatch.setattr(pool, "_start_replenishment_thread", capture_replenishment)

        assert await pool._get_idle_worker(timeout=0.05) is None
        assert dead_idle.shutdown_called is True
        assert pool.health_state == subprocess_pool.POOL_HEALTHY
        assert pool.pending_replacements == 1
        assert len(replenishments) == 1

    asyncio.run(scenario())


def test_result_is_published_only_after_strict_cuda_sync() -> None:
    events: list[str] = []

    class FakeQueue:
        @staticmethod
        def put(_result):  # noqa: ANN001
            events.append("put")

    subprocess_pool._publish_task_result_after_sync(
        lambda: events.append("sync:cuda:0"), FakeQueue(), {"success": True}
    )
    assert events == ["sync:cuda:0", "put"]


def test_strict_cuda_sync_failure_never_publishes_success() -> None:
    put_calls = 0

    class FakeQueue:
        @staticmethod
        def put(_result):  # noqa: ANN001
            nonlocal put_calls
            put_calls += 1

    with pytest.raises(subprocess_pool.CudaFinalSyncError):
        subprocess_pool._publish_task_result_after_sync(
            lambda: (_ for _ in ()).throw(RuntimeError("an illegal memory access was encountered")),
            FakeQueue(),
            {"success": True},
        )
    assert put_calls == 0


def test_ordinary_task_exception_crosses_strict_cuda_barrier() -> None:
    sync_calls: list[str] = []

    details = subprocess_pool._synchronize_and_classify_task_error(
        lambda: sync_calls.append("cuda:0"),
        ValueError("ordinary Python failure"),
    )

    assert sync_calls == ["cuda:0"]
    assert details["error_type"] == "ValueError"
    assert details["fault_severity"] == subprocess_pool.FAULT_NONE
    assert details["final_sync_failed"] is False


def test_ordinary_task_exception_is_promoted_when_delayed_cuda_fault_surfaces() -> None:
    details = subprocess_pool._synchronize_and_classify_task_error(
        lambda: (_ for _ in ()).throw(RuntimeError("an illegal memory access was encountered")),
        ValueError("ordinary Python failure"),
    )

    assert details["error_type"] == "CudaFinalSyncError"
    assert details["fault_severity"] == subprocess_pool.FAULT_CONTEXT
    assert details["final_sync_failed"] is True
    assert details["is_cuda_error"] is True


def test_ordinary_failure_counts_toward_recycle_limit_before_publish() -> None:
    published: list[dict[str, object]] = []

    class FakeQueue:
        @staticmethod
        def put(result):  # noqa: ANN001
            published.append(dict(result))

    payload = {
        "success": False,
        "error_type": "ValueError",
        "error_message": "ordinary Python failure",
        "worker_exiting": False,
        "cuda_error": False,
    }
    tasks_processed, must_recycle = subprocess_pool._publish_non_cuda_failure_and_count_task(
        FakeQueue(),
        payload,
        tasks_processed=0,
        max_tasks_per_worker=1,
    )

    assert tasks_processed == 1
    assert must_recycle is True
    assert payload["worker_exiting"] is True
    assert published == [payload]


def test_reusable_worker_syncs_before_cleanup_and_commits_last() -> None:
    events: list[str] = []

    class FakeQueue:
        @staticmethod
        def put(_result):  # noqa: ANN001
            events.append("put")

    subprocess_pool._commit_task_result(
        lambda: events.append("sync:cuda:0"),
        FakeQueue(),
        {"success": True},
        prepare_for_reuse=lambda: events.append("cleanup"),
    )

    assert events == ["sync:cuda:0", "cleanup", "sync:cuda:0", "put"]


def test_reusable_worker_cleanup_failure_never_publishes() -> None:
    put_calls = 0

    class FakeQueue:
        @staticmethod
        def put(_result):  # noqa: ANN001
            nonlocal put_calls
            put_calls += 1

    def failed_cleanup() -> None:
        raise subprocess_pool.CudaFinalSyncError("empty_cache exposed a sticky fault")

    with pytest.raises(subprocess_pool.CudaFinalSyncError):
        subprocess_pool._commit_task_result(
            lambda: None,
            FakeQueue(),
            {"success": True},
            prepare_for_reuse=failed_cleanup,
        )
    assert put_calls == 0


def test_captured_low_level_barrier_survives_setattr_and_alias_monkeypatch() -> None:
    events: list[str] = []

    class FakeCExtension:
        @staticmethod
        def _cuda_setDevice(device_id):  # noqa: N802, ANN001
            events.append(f"set:{device_id}")

        @staticmethod
        def _cuda_synchronize():
            events.append("trusted-sync")

    class FakeCuda:
        @staticmethod
        def synchronize(*_args, **_kwargs):  # noqa: ANN002, ANN003
            events.append("dynamic-sync")

    class FakeTorch:
        _C = FakeCExtension()
        cuda = FakeCuda()

    class FakeQueue:
        @staticmethod
        def put(_result):  # noqa: ANN001
            events.append("put")

    barrier = subprocess_pool._capture_trusted_cuda_task_barrier(FakeTorch, 3)

    # This models generated Python/toolkit code mutating both public and
    # low-level module attributes after the trusted callable was captured.
    setattr(FakeTorch.cuda, "synchronize", lambda *_args, **_kwargs: events.append("poisoned-public"))
    original_alias = FakeTorch._C._cuda_synchronize
    setattr(FakeTorch._C, "_cuda_synchronize", lambda: events.append("poisoned-low-level"))
    assert original_alias is not FakeTorch._C._cuda_synchronize

    subprocess_pool._commit_task_result(
        barrier,
        FakeQueue(),
        {"success": True},
        prepare_for_reuse=lambda: events.append("cleanup"),
    )

    assert events == [
        "set:3",
        "trusted-sync",
        "cleanup",
        "set:3",
        "trusted-sync",
        "put",
    ]


def test_captured_commit_survives_module_helper_monkeypatch(monkeypatch) -> None:
    events: list[str] = []
    operations = subprocess_pool._capture_trusted_cuda_task_operations(
        lambda: events.append("trusted-sync"),
        lambda _result: events.append("send"),
        lambda: events.append("wait"),
    )

    monkeypatch.setattr(
        subprocess_pool,
        "_commit_task_result",
        lambda *_args, **_kwargs: events.append("poisoned-commit"),
    )
    monkeypatch.setattr(
        subprocess_pool,
        "_strict_cuda_task_barrier",
        lambda *_args, **_kwargs: events.append("poisoned-sync"),
    )

    operations.commit({"success": True}, lambda: events.append("cleanup"))

    assert events == ["trusted-sync", "cleanup", "trusted-sync", "send"]


def test_retiring_result_publication_waits_for_parent_without_cuda_calls() -> None:
    events: list[str] = []

    class ParentStoppedTest(Exception):
        pass

    def wait_for_parent() -> None:
        events.append("wait")
        raise ParentStoppedTest

    operations = subprocess_pool._capture_trusted_cuda_task_operations(
        lambda: events.append("unexpected-sync"),
        lambda _result: events.append("send"),
        wait_for_parent,
    )

    with pytest.raises(ParentStoppedTest):
        operations.publish_and_wait({"success": False})

    assert events == ["send", "wait"]


def test_max_task_success_commits_then_waits_for_parent_reap() -> None:
    events: list[str] = []

    class ParentStoppedTest(BaseException):
        pass

    def wait_for_parent() -> None:
        events.append("wait")
        raise ParentStoppedTest

    operations = subprocess_pool._capture_trusted_cuda_task_operations(
        lambda: events.append("sync"),
        lambda _result: events.append("send"),
        wait_for_parent,
    )

    with pytest.raises(ParentStoppedTest):
        operations.commit_and_wait(
            {"success": True, "result": {}, "worker_exiting": True},
            lambda: events.append("cleanup"),
        )

    assert events == ["sync", "cleanup", "sync", "send", "wait"]


def test_max_task_ordinary_failure_publishes_then_waits_for_parent_reap() -> None:
    events: list[str] = []
    payload = {
        "success": False,
        "error_type": "ValueError",
        "error_message": "ordinary failure",
        "worker_exiting": False,
    }

    class ParentStoppedTest(BaseException):
        pass

    def wait_for_parent() -> None:
        events.append("wait")
        raise ParentStoppedTest

    operations = subprocess_pool._capture_trusted_cuda_task_operations(
        lambda: events.append("unexpected-sync"),
        lambda _result: events.append("send"),
        wait_for_parent,
    )

    with pytest.raises(ParentStoppedTest):
        operations.publish_non_cuda_failure(payload, 0, 1)

    assert payload["worker_exiting"] is True
    assert events == ["send", "wait"]


@pytest.mark.parametrize(
    ("error_type", "message", "expected"),
    [
        ("CudaFinalSyncError", "illegal memory access", subprocess_pool.FAULT_CONTEXT),
        ("RuntimeError", "device-side assert triggered", subprocess_pool.FAULT_CONTEXT),
        ("TimeoutError", "task timeout", subprocess_pool.FAULT_DEVICE),
        ("WorkerProcessCrashed", "exitcode=-11", subprocess_pool.FAULT_DEVICE),
        ("RuntimeError", "CUDA driver initialization failed", subprocess_pool.FAULT_DEVICE),
        ("RuntimeError", "CUDA out of memory", subprocess_pool.FAULT_NONE),
        ("RuntimeError", "PROFILER_NO_CUDA_EVENTS", subprocess_pool.FAULT_NONE),
        ("RuntimeError", "ordinary Python failure", subprocess_pool.FAULT_NONE),
    ],
)
def test_cuda_fault_classifier(error_type: str, message: str, expected: str) -> None:
    assert subprocess_pool._classify_cuda_fault(error_type, message) == expected


def test_parent_strengthens_child_fault_classification() -> None:
    child_claim = subprocess_pool.FAULT_NONE
    parent_classification = subprocess_pool._classify_cuda_fault(
        "RuntimeError",
        "CUDA driver initialization failed",
    )

    assert subprocess_pool._strongest_cuda_fault(child_claim, parent_classification) == subprocess_pool.FAULT_DEVICE


def test_context_fault_consumes_exactly_one_prefault_spare() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        first = FakeWorker("first")
        second = FakeWorker("second")
        pool.workers = [first, second]
        pool.idle_workers = [first, second]

        generation = await pool._begin_context_fault_validation(task_id="fault", reason="illegal access")
        assert generation == 1
        assert pool.accepting_tasks is True

        assert await pool._get_idle_worker(timeout=0.1) is first
        assert pool.speculative_dispatches_remaining == 0
        assert pool.accepting_tasks is False
        assert await pool._get_idle_worker(timeout=0.05) is None
        assert pool.idle_workers == [second]

    asyncio.run(scenario())


def test_stale_context_fault_does_not_start_another_validation() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        pool.hard_recovery_epoch = 2
        stale_worker = FakeWorker("stale")
        stale_worker._pool_hard_recovery_epoch_at_checkout = 1

        generation = await pool._begin_context_fault_validation(
            task_id="stale-context",
            reason="late illegal access",
            worker=stale_worker,  # type: ignore[arg-type]
        )

        assert generation == subprocess_pool._STALE_HARD_RECOVERY_EPOCH
        assert pool.pool_generation == 0
        assert pool.health_state == subprocess_pool.POOL_HEALTHY
        assert pool.pending_replacements == 0

    asyncio.run(scenario())


def test_stale_device_fault_is_ignored_even_while_pool_is_degraded(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        pool.hard_recovery_epoch = 4
        pool.health_state = subprocess_pool.POOL_DEGRADED_CHECK
        stale_worker = FakeWorker("stale")
        stale_worker._pool_hard_recovery_epoch_at_checkout = 3
        recovery_calls = 0

        async def unexpected_recovery(*_args, **_kwargs):  # noqa: ANN002, ANN003
            nonlocal recovery_calls
            recovery_calls += 1

        monkeypatch.setattr(pool, "_recover_from_device_fault_once", unexpected_recovery)

        await pool._recover_from_device_fault(
            stale_worker,  # type: ignore[arg-type]
            task_id="stale-device",
            reason="late device lost",
        )

        assert recovery_calls == 0
        assert pool.hard_recovery_epoch == 4
        assert pool.health_state == subprocess_pool.POOL_DEGRADED_CHECK

    asyncio.run(scenario())


def test_post_context_fault_ready_reopens_pool(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        culprit = FakeWorker("culprit", alive=False)
        spare = FakeWorker("spare")
        pool.workers = [culprit, spare]
        pool.busy_workers = [culprit]
        pool.idle_workers = [spare]
        created: list[FakeWorker] = []

        def fresh_worker(worker_id, *_args, **_kwargs):  # noqa: ANN001
            assert culprit.shutdown_called is True
            worker = FakeWorker(worker_id)
            created.append(worker)
            return worker

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fresh_worker)

        generation = await pool._begin_context_fault_validation(task_id="fault", reason="illegal access")
        await pool._restart_worker(culprit, validation_generation=generation)  # type: ignore[arg-type]
        assert pool.health_state == subprocess_pool.POOL_DEGRADED_CHECK
        assert culprit.shutdown_called is True
        for _ in range(100):
            if pool.health_state == subprocess_pool.POOL_HEALTHY and not pool._active_replenishments:
                break
            await asyncio.sleep(0.01)

        assert pool.health_state == subprocess_pool.POOL_HEALTHY
        assert created[0] in pool.workers
        assert pool.accepting_tasks is True

    asyncio.run(scenario())


def test_context_probe_waits_for_and_reaps_stale_capacity_constructor(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        culprit = FakeWorker("culprit", alive=False)
        pool.workers = [culprit]
        pool.busy_workers = [culprit]
        old_constructor_started = threading.Event()
        release_old_constructor = threading.Event()
        validation_constructor_started = threading.Event()
        created: dict[str, FakeWorker] = {}

        def fresh_worker(worker_id, *_args, **_kwargs):  # noqa: ANN001
            worker = FakeWorker(worker_id)
            created[worker_id] = worker
            if worker_id.endswith("_topup_1"):
                old_constructor_started.set()
                assert release_old_constructor.wait(timeout=5)
            elif worker_id == "culprit":
                validation_constructor_started.set()
            return worker

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fresh_worker)

        async with pool.lock:
            pool._ensure_capacity_locked(asyncio.get_running_loop())
        for _ in range(100):
            if old_constructor_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert old_constructor_started.is_set()

        generation = await pool._begin_context_fault_validation(task_id="fault", reason="illegal access")
        await pool._restart_worker(culprit, validation_generation=generation)  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        assert validation_constructor_started.is_set() is False

        release_old_constructor.set()
        for _ in range(300):
            if pool.health_state == subprocess_pool.POOL_HEALTHY and not pool._active_replenishments:
                break
            await asyncio.sleep(0.01)

        stale_worker = created["test_worker_0_topup_1"]
        assert stale_worker.shutdown_called is True
        assert stale_worker not in pool.workers
        assert validation_constructor_started.is_set() is True
        assert pool.health_state == subprocess_pool.POOL_HEALTHY
        assert len(pool.workers) == pool.pool_size

    asyncio.run(scenario())


def test_quarantined_pool_fails_allocation_immediately() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes()
        pool.health_state = subprocess_pool.POOL_QUARANTINED
        pool.health_reason = "fresh context failed"
        started = time.monotonic()
        with pytest.raises(subprocess_pool.GPUQuarantinedError):
            await pool._get_idle_worker(timeout=30)
        assert time.monotonic() - started < 0.1

    asyncio.run(scenario())


def test_device_fault_reaps_old_workers_before_fresh_probe(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old = FakeWorker("old")
        pool.workers = [old]
        pool.busy_workers = [old]
        events: list[str] = []

        def shutdown(timeout: int = 10, force: bool = False) -> bool:  # noqa: ARG002
            events.append("old_shutdown")
            old.shutdown_called = True
            old._alive = False
            return True

        old.shutdown = shutdown  # type: ignore[method-assign]

        def fresh_worker(worker_id, *_args, **_kwargs):  # noqa: ANN001
            assert events == ["old_shutdown"]
            events.append("fresh_probe")
            return FakeWorker(worker_id)

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fresh_worker)
        await pool._recover_from_device_fault(old, task_id="t-hard", reason="device lost")  # type: ignore[arg-type]

        assert events == ["old_shutdown", "fresh_probe"]
        assert pool.health_state == subprocess_pool.POOL_HEALTHY
        assert len(pool.workers) == 1
        assert pool.accepting_tasks is True

    asyncio.run(scenario())


def test_concurrent_unsafe_quarantine_is_propagated_to_current_device_fault(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("racing-worker")

        async def raced_inner(*_args, **_kwargs):  # noqa: ANN002, ANN003
            # Simulate another containment path winning after the wrapper's
            # initial state check but before this recovery can take ownership.
            async with pool.lock:
                pool.health_state = subprocess_pool.POOL_QUARANTINED
                pool.unsafe_shutdown_reason = "another CUDA context could not be confirmed reaped"

        monkeypatch.setattr(pool, "_recover_from_device_fault_once", raced_inner)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._recover_from_device_fault(
                worker,  # type: ignore[arg-type]
                task_id="racing-device-fault",
                reason="device lost",
            )

        assert error.value.task_id == "racing-device-fault"
        assert error.value.worker_id == "racing-worker"
        assert "could not be confirmed reaped" in str(error.value)

    asyncio.run(scenario())


def test_stale_context_result_waits_for_real_concurrent_unsafe_recovery() -> None:
    """A stale-epoch result cannot outrun another task's failed hard reap."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        context_worker = FakeWorker("context-worker")
        device_worker = FakeWorker("device-worker")
        pool.workers = [context_worker, device_worker]
        pool.idle_workers = [context_worker, device_worker]
        context_execute_started = threading.Event()
        allow_context_result = threading.Event()
        shutdown_started = threading.Event()
        allow_shutdown = threading.Event()

        def context_execute(*_args, **_kwargs):  # noqa: ANN002, ANN003
            context_execute_started.set()
            assert allow_context_result.wait(timeout=5)
            return {
                "success": False,
                "error_type": "CudaFinalSyncError",
                "error_message": "illegal memory access",
                "fault_severity": subprocess_pool.FAULT_CONTEXT,
                "final_sync_failed": True,
                "worker_exiting": True,
            }

        def context_shutdown(*_args, **_kwargs):  # noqa: ANN002, ANN003
            shutdown_started.set()
            assert allow_shutdown.wait(timeout=5)
            context_worker._alive = False
            return True

        def device_shutdown(*_args, **_kwargs):  # noqa: ANN002, ANN003
            shutdown_started.set()
            assert allow_shutdown.wait(timeout=5)
            return False

        context_worker.execute_task = context_execute  # type: ignore[attr-defined]
        context_worker.shutdown = context_shutdown  # type: ignore[method-assign]
        device_worker.shutdown = device_shutdown  # type: ignore[method-assign]

        context_task = asyncio.create_task(pool.execute_task({"task_id": "stale-context-task"}, timeout=2))
        for _ in range(100):
            if context_execute_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert context_execute_started.is_set()

        device_recovery = asyncio.create_task(
            pool._recover_from_device_fault(  # type: ignore[arg-type]
                device_worker,
                task_id="device-fault-task",
                reason="device lost",
            )
        )
        for _ in range(100):
            if pool.hard_recovery_epoch == 1 and shutdown_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert pool.hard_recovery_epoch == 1
        assert shutdown_started.is_set()

        # The context result now observes a stale epoch. It must serialize
        # behind device_recovery rather than returning/ACKing immediately.
        allow_context_result.set()
        await asyncio.sleep(0.05)
        assert context_task.done() is False

        allow_shutdown.set()
        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError):
            await device_recovery
        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError):
            await context_task

        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.unsafe_shutdown_reason

    asyncio.run(scenario())


def test_inner_device_recovery_early_return_preserves_unsafe_signal() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("already-quarantined")
        pool.health_state = subprocess_pool.POOL_QUARANTINED
        pool.unsafe_shutdown_reason = "retained CUDA child survived containment"

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._recover_from_device_fault_once(
                worker,  # type: ignore[arg-type]
                task_id="already-unsafe",
                reason="device lost",
            )

        assert error.value.task_id == "already-unsafe"
        assert error.value.worker_id == "already-quarantined"

    asyncio.run(scenario())


@pytest.mark.parametrize("shutdown_safe", [True, False])
def test_device_recovery_waits_for_shared_shutdown_proof(shutdown_safe: bool) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("shutdown-owned")
        shutdown_started = asyncio.Event()
        release_shutdown = asyncio.Event()

        async def shutdown_proof() -> bool:
            shutdown_started.set()
            await release_shutdown.wait()
            if not shutdown_safe:
                pool.unsafe_shutdown_reason = "shutdown could not prove child reap"
            return shutdown_safe

        pool._closing = True
        pool._shutdown_task = asyncio.create_task(shutdown_proof())
        recovery = asyncio.create_task(
            pool._recover_from_device_fault(  # type: ignore[arg-type]
                worker,
                task_id="shutdown-owned-task",
                reason="device lost during shutdown",
            )
        )
        await asyncio.wait_for(shutdown_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert recovery.done() is False

        release_shutdown.set()
        expected_error = (
            subprocess_pool.PoolShutdownContainmentError
            if shutdown_safe
            else subprocess_pool.UnsafeGPUContainmentError
        )
        with pytest.raises(expected_error):
            await recovery

    asyncio.run(scenario())


def test_execute_task_preserves_unsafe_when_return_cleanup_raises(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("unsafe-primary")
        pool.workers = [worker]
        pool.idle_workers = [worker]

        def unsafe_execute(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise subprocess_pool.UnsafeGPUContainmentError(
                "primary unsafe reap signal",
                task_id="unsafe-cleanup-task",
                worker_id=worker.worker_id,
            )

        async def broken_return(_worker):  # noqa: ANN001
            raise RuntimeError("return cleanup exploded")

        worker.execute_task = unsafe_execute  # type: ignore[attr-defined]
        monkeypatch.setattr(pool, "_return_worker", broken_return)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError, match="primary unsafe reap signal"):
            await pool.execute_task({"task_id": "unsafe-cleanup-task"}, timeout=1)

    asyncio.run(scenario())


def test_execute_task_preserves_committed_success_when_return_cleanup_raises(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("successful-cleanup-error")
        pool.workers = [worker]
        pool.idle_workers = [worker]

        def successful_execute(*_args, **_kwargs):  # noqa: ANN002, ANN003
            return {"success": True, "result": {"status": "completed"}}

        async def broken_return(_worker):  # noqa: ANN001
            raise RuntimeError("return cleanup exploded")

        worker.execute_task = successful_execute  # type: ignore[attr-defined]
        monkeypatch.setattr(pool, "_return_worker", broken_return)

        result = await pool.execute_task({"task_id": "successful-cleanup-error"}, timeout=1)

        assert result["success"] is True
        assert result["result"] == {"status": "completed"}

    asyncio.run(scenario())


def test_execute_task_preserves_unsafe_over_cleanup_cancellation(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("unsafe-cancel")
        pool.workers = [worker]
        pool.idle_workers = [worker]

        def unsafe_execute(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise subprocess_pool.UnsafeGPUContainmentError(
                "unsafe beats cancellation",
                task_id="unsafe-cancel-task",
                worker_id=worker.worker_id,
            )

        original_complete = subprocess_pool._complete_despite_cancellation

        async def report_cleanup_cancellation(awaitable):  # noqa: ANN001
            if getattr(awaitable, "cr_code", None) is not None and awaitable.cr_code.co_name == "_return_worker":
                await awaitable
                return None, True
            return await original_complete(awaitable)

        worker.execute_task = unsafe_execute  # type: ignore[attr-defined]
        monkeypatch.setattr(subprocess_pool, "_complete_despite_cancellation", report_cleanup_cancellation)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError, match="unsafe beats cancellation"):
            await pool.execute_task({"task_id": "unsafe-cancel-task"}, timeout=1)

    asyncio.run(scenario())


def test_unexpected_worker_exception_runs_device_recovery_before_failure(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("unexpected-exception")
        pool.workers = [worker]
        pool.idle_workers = [worker]
        fresh_workers: list[FakeWorker] = []

        def broken_execute(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise ValueError("non-RuntimeError transport failure")

        def fresh_worker(worker_id, *_args, **_kwargs):  # noqa: ANN001
            fresh = FakeWorker(worker_id)
            fresh_workers.append(fresh)
            return fresh

        worker.execute_task = broken_execute  # type: ignore[attr-defined]
        monkeypatch.setattr(subprocess_pool, "PersistentWorker", fresh_worker)

        with pytest.raises(RuntimeError, match="device-level fault"):
            await pool.execute_task({"task_id": "unexpected-exception-task"}, timeout=1)

        assert worker.shutdown_called is True
        assert worker not in pool.workers
        assert worker not in pool.idle_workers
        assert len(fresh_workers) == 1
        assert pool.workers == fresh_workers
        assert pool.health_state == subprocess_pool.POOL_HEALTHY

    asyncio.run(scenario())


def test_native_crash_with_unprovable_process_group_suppresses_fresh_probe(monkeypatch) -> None:
    """A vanished leader loses the freeze-before-reparent proof by design."""

    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        crashed = FakeWorker("native-crash", alive=False)
        crashed.shutdown = lambda *_args, **_kwargs: False  # type: ignore[method-assign]
        pool.workers = [crashed]
        pool.busy_workers = [crashed]
        constructor_calls = 0

        def forbidden_probe(*_args, **_kwargs):  # noqa: ANN002, ANN003
            nonlocal constructor_calls
            constructor_calls += 1
            raise AssertionError("fresh CUDA probe must remain suppressed")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", forbidden_probe)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError):
            await pool._recover_from_device_fault(
                crashed,  # type: ignore[arg-type]
                task_id="native-crash",
                reason="WorkerProcessCrashed: exitcode=-11",
            )

        assert constructor_calls == 0
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.health_scope == "gpu"
        assert pool.health_fault_class == "pre_fault_reap_failure"
        assert pool.accepting_tasks is False

    asyncio.run(scenario())


def test_context_fault_culprit_must_be_reaped_before_background_validation(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        culprit = FakeWorker("context-culprit", alive=False)
        culprit.shutdown = lambda *_args, **_kwargs: False  # type: ignore[method-assign]
        pool.workers = [culprit]
        pool.busy_workers = [culprit]
        replenishments = []
        monkeypatch.setattr(pool, "_start_replenishment_thread", lambda **kwargs: replenishments.append(kwargs))

        generation = await pool._begin_context_fault_validation(
            task_id="context-task",
            reason="illegal memory access",
            worker=culprit,  # type: ignore[arg-type]
        )
        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._restart_worker(
                culprit,  # type: ignore[arg-type]
                validation_generation=generation,
            )

        assert error.value.task_id == "context-task"
        assert replenishments == []
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.health_fault_class == "pre_fault_reap_failure"
        assert pool.unsafe_shutdown_reason

    asyncio.run(scenario())


def test_device_recovery_replenishment_timeout_is_unsafe(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old = FakeWorker("old")
        pool.workers = [old]
        pool.busy_workers = [old]
        constructor_calls = 0

        def forbidden_constructor(*_args, **_kwargs):  # noqa: ANN002, ANN003
            nonlocal constructor_calls
            constructor_calls += 1
            raise AssertionError("fresh validation must be suppressed")

        monkeypatch.setattr(pool, "_wait_for_older_replenishments", lambda _generation: False)
        monkeypatch.setattr(subprocess_pool, "PersistentWorker", forbidden_constructor)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._recover_from_device_fault(
                old,  # type: ignore[arg-type]
                task_id="constructor-timeout",
                reason="device lost",
            )

        assert error.value.task_id == "constructor-timeout"
        assert constructor_calls == 0
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.health_fault_class == "pre_fault_reap_failure"
        assert "constructors" in pool.unsafe_shutdown_reason

    asyncio.run(scenario())


def test_unproven_fresh_validation_constructor_is_unsafe(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old = FakeWorker("old")
        pool.workers = [old]
        pool.busy_workers = [old]

        def unproven_constructor(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise subprocess_pool.WorkerInitializationError(
                "READY child survived shutdown",
                init_stage="cuda_sync",
                reap_confirmed=False,
            )

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", unproven_constructor)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._recover_from_device_fault(
                old,  # type: ignore[arg-type]
                task_id="unproven-validation",
                reason="device lost",
            )

        assert error.value.task_id == "unproven-validation"
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.health_fault_class == "pre_fault_reap_failure"
        assert "validation CUDA context" in pool.unsafe_shutdown_reason

    asyncio.run(scenario())


def test_stale_validation_shutdown_failure_is_unsafe(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old = FakeWorker("old")
        pool.workers = [old]
        pool.busy_workers = [old]
        stale_validation = FakeWorker("stale-validation")
        stale_validation.shutdown = lambda *_args, **_kwargs: False  # type: ignore[method-assign]

        def stale_constructor(*_args, **_kwargs):  # noqa: ANN002, ANN003
            # Make the freshly READY validation context stale before its event-
            # loop registration lock is acquired.
            pool.pool_generation += 1
            return stale_validation

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", stale_constructor)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._recover_from_device_fault(
                old,  # type: ignore[arg-type]
                task_id="stale-validation-task",
                reason="device lost",
            )

        assert error.value.worker_id == "stale-validation"
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.health_fault_class == "pre_fault_reap_failure"
        assert subprocess_pool._snapshot_unreaped_workers(0) == [stale_validation]

    asyncio.run(scenario())


def test_failed_device_probe_quarantines_without_retry(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old = FakeWorker("old")
        pool.workers = [old]
        calls = 0

        def failed_probe(*_args, **_kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            raise RuntimeError("CUDA_ERROR_NOT_INITIALIZED")

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", failed_probe)
        await pool._recover_from_device_fault(old, task_id="t-hard", reason="driver failure")  # type: ignore[arg-type]

        assert calls == 1
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.pending_replacements == 0
        pool._ensure_capacity_locked(asyncio.get_running_loop())
        assert calls == 1

    asyncio.run(scenario())


def test_execute_cancellation_waits_for_worker_return_bookkeeping() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("worker")

        def execute_task(*_args, **_kwargs):  # noqa: ANN002, ANN003
            return {"success": True, "result": {"status": "completed"}}

        worker.execute_task = execute_task
        pool.workers = [worker]
        pool.idle_workers = [worker]
        return_started = asyncio.Event()
        allow_return = asyncio.Event()
        original_return = pool._return_worker

        async def delayed_return(returning_worker):  # noqa: ANN001
            return_started.set()
            await allow_return.wait()
            await original_return(returning_worker)

        pool._return_worker = delayed_return
        task = asyncio.create_task(pool.execute_task({"task_id": "cancel-at-return"}, timeout=2))
        await asyncio.wait_for(return_started.wait(), timeout=2)

        task.cancel()
        await asyncio.sleep(0.05)
        assert task.done() is False

        allow_return.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert pool.busy_workers == []
        assert pool.idle_workers == [worker]

    asyncio.run(scenario())


def test_unconfirmed_reap_handle_is_retained_and_retried_by_shutdown() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        worker = FakeWorker("unreaped")
        shutdown_calls = 0

        def staged_shutdown(timeout: int = 10, force: bool = False) -> bool:  # noqa: ARG001
            nonlocal shutdown_calls
            shutdown_calls += 1
            if shutdown_calls == 1:
                return False
            worker._alive = False
            return True

        worker.shutdown = staged_shutdown
        pool.workers = [worker]
        pool.idle_workers = [worker]

        await pool._quarantine_unreaped_context("first reap could not be proven")

        assert pool.workers == []
        assert subprocess_pool._snapshot_unreaped_workers(0) == [worker]
        assert worker in pool._tracked_workers_locked()

        # unsafe_shutdown_reason remains sticky, but the later shutdown pass
        # must retry and release the retained Process handle after proof.
        assert await pool.shutdown(timeout=1) is False
        assert shutdown_calls == 2
        assert subprocess_pool._snapshot_unreaped_workers(0) == []

    asyncio.run(scenario())


def test_shutdown_cancellation_waits_for_busy_worker_force_reap() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        busy = FakeWorker("busy")
        pool.workers = [busy]
        pool.busy_workers = [busy]
        shutdown_started = threading.Event()
        allow_shutdown = threading.Event()
        force_values: list[bool] = []

        def blocking_shutdown(timeout: int = 10, force: bool = False) -> bool:  # noqa: ARG001
            force_values.append(force)
            shutdown_started.set()
            assert allow_shutdown.wait(timeout=5)
            busy._alive = False
            busy.shutdown_called = True
            return True

        busy.shutdown = blocking_shutdown  # type: ignore[method-assign]
        shutdown_task = asyncio.create_task(pool.shutdown(timeout=5))
        for _ in range(100):
            if shutdown_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert shutdown_started.is_set()

        shutdown_task.cancel()
        await asyncio.sleep(0.05)
        assert shutdown_task.done() is False

        allow_shutdown.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown_task

        assert force_values == [True]
        assert busy.shutdown_called is True
        assert pool.workers == []
        assert pool.busy_workers == []

    asyncio.run(scenario())


def test_healthy_shutdown_forces_busy_worker_but_allows_idle_graceful() -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=2)
        idle = FakeWorker("idle")
        busy = FakeWorker("busy")
        pool.workers = [idle, busy]
        pool.idle_workers = [idle]
        pool.busy_workers = [busy]
        force_by_worker: dict[str, bool] = {}

        def shutdown_for(worker: FakeWorker):
            def shutdown(timeout: int = 10, force: bool = False) -> bool:  # noqa: ARG001
                force_by_worker[worker.worker_id] = force
                worker._alive = False
                return True

            return shutdown

        idle.shutdown = shutdown_for(idle)  # type: ignore[method-assign]
        busy.shutdown = shutdown_for(busy)  # type: ignore[method-assign]

        assert await pool.shutdown(timeout=10) is True
        assert force_by_worker == {"idle": False, "busy": True}

    asyncio.run(scenario())


def test_shutdown_uses_full_replenishment_lifecycle_bound(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes()
        observed_timeouts: list[float] = []

        def wait_for_tickets(timeout: float) -> bool:
            observed_timeouts.append(timeout)
            return True

        monkeypatch.setattr(pool, "_wait_for_all_replenishments", wait_for_tickets)
        assert await pool.shutdown(timeout=1) is True
        assert observed_timeouts == [subprocess_pool._REPLENISHMENT_DRAIN_TIMEOUT_S]
        assert observed_timeouts[0] >= 210.0

    asyncio.run(scenario())


def test_cancelled_ready_registration_reaps_before_ticket_finishes(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes()
        pool.pending_replacements = 1
        ready_worker = FakeWorker("ready-but-unregistered")
        threads: list[_NoOpThread] = []

        def fake_thread(*args, **kwargs):  # noqa: ANN002, ANN003
            thread = _NoOpThread(*args, **kwargs)
            threads.append(thread)
            return thread

        class CancelledFuture:
            def __init__(self, coro):  # noqa: ANN001
                self.coro = coro
                self.callback = None

            def add_done_callback(self, callback):  # noqa: ANN001
                self.callback = callback

            @staticmethod
            def result():
                raise asyncio.CancelledError

            def cancel_before_start(self) -> None:
                self.coro.close()
                assert self.callback is not None
                self.callback(self)

        scheduled: list[CancelledFuture] = []

        def fake_schedule(coro, _loop):  # noqa: ANN001
            future = CancelledFuture(coro)
            scheduled.append(future)
            return future

        monkeypatch.setattr(subprocess_pool, "PersistentWorker", lambda *_args, **_kwargs: ready_worker)
        monkeypatch.setattr(subprocess_pool.threading, "Thread", fake_thread)
        monkeypatch.setattr(subprocess_pool.asyncio, "run_coroutine_threadsafe", fake_schedule)

        pool._start_replenishment_thread(
            old_worker=None,
            old_process=None,
            old_pid=None,
            worker_id="ready-but-unregistered",
            loop=asyncio.get_running_loop(),
            reason="test",
            generation=pool.pool_generation,
        )
        assert len(threads) == 1
        threads[0].target()
        assert subprocess_pool._snapshot_unreaped_workers(0) == [ready_worker]
        assert pool._active_replenishments == {0: 1}

        scheduled[0].cancel_before_start()
        # The Future being done starts a dedicated cleanup thread, but the
        # ticket remains live until that thread proves shutdown.
        assert len(threads) == 2
        assert pool._active_replenishments == {0: 1}
        assert ready_worker.shutdown_called is False

        threads[1].target(ready_worker)
        assert ready_worker.shutdown_called is True
        assert subprocess_pool._snapshot_unreaped_workers(0) == []
        assert pool._active_replenishments == {}

    asyncio.run(scenario())


def test_shutdown_second_snapshot_reaps_worker_created_during_ticket_drain(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes()
        late_worker = FakeWorker("late-ready")

        def finish_constructor(_timeout):  # noqa: ANN001
            subprocess_pool._retain_unreaped_worker(pool.device_id, late_worker)
            return True

        monkeypatch.setattr(pool, "_wait_for_all_replenishments", finish_constructor)

        assert await pool.shutdown(timeout=1) is True
        assert late_worker.shutdown_called is True
        assert subprocess_pool._snapshot_unreaped_workers(0) == []

    asyncio.run(scenario())


def test_unexpected_hard_recovery_error_quarantines_gpu_and_reaps_context(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old = FakeWorker("old")
        pool.workers = [old]
        pool.busy_workers = [old]

        def broken_ticket_drain(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("ticket condition failed")

        monkeypatch.setattr(pool, "_wait_for_older_replenishments", broken_ticket_drain)

        with pytest.raises(RuntimeError, match="ticket condition failed"):
            await pool._recover_from_device_fault(
                old,
                task_id="unexpected-hard-failure",
                reason="device lost",
            )  # type: ignore[arg-type]

        assert old.shutdown_called is True
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.health_scope == "gpu"
        assert pool.health_fault_class == "hard_recovery_failure"
        assert pool.accepting_tasks is False
        assert pool.pending_replacements == 0
        assert pool._active_replenishments == {}

    asyncio.run(scenario())


def test_unexpected_hard_recovery_with_unreaped_context_raises_unsafe(monkeypatch) -> None:
    async def scenario() -> None:
        pool = _pool_without_processes(pool_size=1)
        old = FakeWorker("old")
        old.shutdown = lambda *_args, **_kwargs: False  # type: ignore[method-assign]
        pool.workers = [old]
        pool.busy_workers = [old]

        async def broken_recovery(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("recovery bookkeeping exploded")

        monkeypatch.setattr(pool, "_recover_from_device_fault_once", broken_recovery)

        with pytest.raises(subprocess_pool.UnsafeGPUContainmentError) as error:
            await pool._recover_from_device_fault(
                old,  # type: ignore[arg-type]
                task_id="unsafe-hard-recovery",
                reason="device lost",
            )

        assert error.value.task_id == "unsafe-hard-recovery"
        assert error.value.worker_id == "old"
        assert pool.health_state == subprocess_pool.POOL_QUARANTINED
        assert pool.health_fault_class == "hard_recovery_failure"
        assert pool.unsafe_shutdown_reason
        assert subprocess_pool._snapshot_unreaped_workers(0) == [old]

    asyncio.run(scenario())
