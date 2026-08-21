"""Shutdown drain: SIGTERM-triggered stop() must let the in-flight task finish
(up to KERNELGYM_WORKER_SHUTDOWN_DRAIN_SEC) instead of failing it immediately,
and must still fail it once the drain window expires."""

import asyncio
import importlib.util
import signal
from pathlib import Path

import pytest

# gpu_worker imports torch/redis/aiohttp at module scope; skip where unavailable.
pytest.importorskip("torch")
pytest.importorskip("redis")
pytest.importorskip("aiohttp")

ROOT = Path(__file__).resolve().parents[2]


def _load_gpu_worker():
    spec = importlib.util.spec_from_file_location(
        "gpu_worker_drain_under_test", ROOT / "kernelgym" / "worker" / "gpu_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_worker = _load_gpu_worker()


class FakeRedis:
    def __getattr__(self, name):
        async def _noop(*args, **kwargs):
            return 1

        return _noop


class RecordingRedis(FakeRedis):
    def __init__(self):
        self.hsets = []

    async def hset(self, key, mapping):  # noqa: ANN001
        self.hsets.append((key, mapping))
        return 1


def _make_worker(monkeypatch, drain_sec: int):
    # Inject via __dict__: pydantic's __setattr__ rejects the field when running
    # against an installed kernelgym whose Settings predates it.
    monkeypatch.setitem(gpu_worker.settings.__dict__, "worker_shutdown_drain_sec", drain_sec)
    worker = gpu_worker.GPUWorker("draintest_gpu_0", "cuda:0", FakeRedis())
    failed = []

    async def record_fail(
        task_id,
        reason,
        *,
        adopt_current_claim=False,
        allow_frozen_claim=False,
    ):  # noqa: ANN001
        assert adopt_current_claim is False
        assert allow_frozen_claim is True
        failed.append((task_id, reason))

    monkeypatch.setattr(worker.task_manager, "fail_task", record_fail)
    return worker, failed


def test_stop_waits_for_inflight_task_to_finish(monkeypatch) -> None:
    worker, failed = _make_worker(monkeypatch, drain_sec=10)
    worker.current_task = "task_inflight"

    async def scenario():
        async def finish_task():
            await asyncio.sleep(0.6)
            worker.current_task = None

        finisher = asyncio.create_task(finish_task())
        await asyncio.wait_for(worker.stop(), timeout=8)
        await finisher

    asyncio.run(scenario())
    assert failed == []


def test_stop_fails_task_after_drain_expires(monkeypatch) -> None:
    worker, failed = _make_worker(monkeypatch, drain_sec=1)
    worker.current_task = "task_stuck"

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=8))
    assert failed == [("task_stuck", "Worker shutdown")]


def test_stop_immediate_when_drain_disabled(monkeypatch) -> None:
    worker, failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_now"

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=3))
    assert failed == [("task_now", "Worker shutdown")]


def test_stop_waits_for_processing_loop_even_without_current_task(monkeypatch) -> None:
    """A task can be popped from its queue before current_task is set; the
    drain must wait for the processing loop to exit, not just current_task."""
    worker, failed = _make_worker(monkeypatch, drain_sec=10)
    worker.current_task = None
    worker._processing_active = True

    async def scenario():
        async def loop_exits():
            await asyncio.sleep(0.6)
            worker._processing_active = False

        finisher = asyncio.create_task(loop_exits())
        await asyncio.wait_for(worker.stop(), timeout=8)
        await finisher

    asyncio.run(scenario())
    assert failed == []


def test_stop_skips_drain_on_error_shutdown(monkeypatch) -> None:
    """Eviction / error shutdowns (shutdown_due_to_error) must stay immediate."""
    worker, failed = _make_worker(monkeypatch, drain_sec=30)
    worker.current_task = "task_evicted"
    worker.shutdown_due_to_error = True

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=5))
    assert failed == [("task_evicted", "Worker shutdown")]


def test_stop_cancellation_waits_for_pool_containment_and_quarantines_unsafe_result(monkeypatch) -> None:
    worker, failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_unsafe_cancel"
    started = asyncio.Event()
    release = asyncio.Event()
    quarantines = []

    class UnsafePool:
        unsafe_shutdown_reason = "mock context could not be reaped"

        async def shutdown(self, timeout):  # noqa: ANN001, ARG002
            started.set()
            await release.wait()
            return False

        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    async def capture_quarantine(**kwargs):  # noqa: ANN003
        quarantines.append(kwargs)

    worker.worker_pool = UnsafePool()
    monkeypatch.setattr(worker, "_quarantine_gpu", capture_quarantine)

    async def scenario():
        stop_task = asyncio.create_task(worker.stop())
        await started.wait()
        stop_task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await stop_task

    asyncio.run(scenario())
    assert failed == []
    assert worker.current_task == "task_unsafe_cancel"
    assert isinstance(worker.worker_pool, UnsafePool)
    assert len(quarantines) == 1
    assert quarantines[0]["physical_scope"] is True
    assert quarantines[0]["fault_class"] == "unsafe_pool_shutdown"


def test_stop_stays_online_but_non_accepting_until_pool_is_reaped(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_contained_first"
    events = []
    pool_reaped = False

    class SafePool:
        unsafe_shutdown_reason = ""

        async def shutdown(self, timeout):  # noqa: ANN001, ARG002
            nonlocal pool_reaped
            events.append("pool_shutdown")
            pool_reaped = True
            return True

        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    async def capture_status(online):  # noqa: ANN001
        events.append(f"status:{str(online).lower()}:{worker.health_state}")
        if not online:
            assert pool_reaped is True

    async def capture_unregister():
        events.append("unregister")
        assert pool_reaped is True
        return True

    async def capture_fail(
        task_id,
        reason,
        *,
        adopt_current_claim=False,
        allow_frozen_claim=False,
    ):  # noqa: ANN001
        assert task_id == "task_contained_first"
        assert reason == "Worker shutdown"
        assert adopt_current_claim is False
        assert allow_frozen_claim is True
        assert pool_reaped is True
        assert worker.worker_pool is None
        events.append("fail_task")

    worker.health_state = "healthy"
    worker.worker_pool = SafePool()
    monkeypatch.setattr(worker, "_update_worker_status", capture_status)
    monkeypatch.setattr(worker, "_unregister_from_api", capture_unregister)
    monkeypatch.setattr(worker.task_manager, "fail_task", capture_fail)

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=3))

    assert events == [
        "status:true:stopping",
        "pool_shutdown",
        "fail_task",
        "unregister",
        "status:false:stopping",
    ]


def test_failed_nvidia_smi_then_failed_fresh_cuda_probe_quarantines_and_pages_path(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    quarantines = []

    async def failed_precheck():
        raise RuntimeError("nvidia-smi timeout")

    class FailedProbePool:
        def __init__(self, **kwargs):  # noqa: ANN003, ARG002
            raise gpu_worker.GPUProbeFailedError("handshake_timeout")

    async def capture_quarantine(**kwargs):  # noqa: ANN003
        quarantines.append(kwargs)

    monkeypatch.setattr(worker, "_initialize_gpu", failed_precheck)
    monkeypatch.setattr(gpu_worker, "SubprocessWorkerPool", FailedProbePool)
    monkeypatch.setattr(worker, "_quarantine_gpu", capture_quarantine)

    asyncio.run(worker._initialize_worker_pool())

    assert worker.worker_pool is None
    assert len(quarantines) == 1
    assert quarantines[0]["physical_scope"] is True
    assert quarantines[0]["fault_class"] == "initialization_failure"
    assert "nvidia-smi precheck also failed" in quarantines[0]["reason"]


def test_shutdown_interrupted_startup_probe_does_not_quarantine_gpu(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    quarantines = []

    class InterruptedProbePool:
        def __init__(self, **kwargs):  # noqa: ANN003, ARG002
            worker._signal_handler(signal.SIGTERM, None)
            raise gpu_worker.GPUProbeFailedError("handshake_timeout")

    async def capture_quarantine(**kwargs):  # noqa: ANN003
        quarantines.append(kwargs)

    async def completed_stop():
        return None

    worker.running = True
    monkeypatch.setattr(gpu_worker, "SubprocessWorkerPool", InterruptedProbePool)
    monkeypatch.setattr(worker, "_quarantine_gpu", capture_quarantine)
    monkeypatch.setattr(worker, "stop", completed_stop)

    with pytest.raises(RuntimeError, match="interrupted by shutdown"):
        asyncio.run(worker._initialize_worker_pool())

    assert quarantines == []
    assert worker.running is False
    assert worker._stopping is True


def test_failed_nvidia_smi_does_not_quarantine_when_fresh_cuda_probe_passes(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)

    async def failed_precheck():
        raise RuntimeError("nvidia-smi unavailable")

    class HealthyPool:
        accepting_tasks = True

        def __init__(self, **kwargs):  # noqa: ANN003, ARG002
            pass

    async def unexpected_quarantine(**kwargs):  # noqa: ANN003
        raise AssertionError(f"unexpected quarantine: {kwargs}")

    monkeypatch.setattr(worker, "_initialize_gpu", failed_precheck)
    monkeypatch.setattr(gpu_worker, "SubprocessWorkerPool", HealthyPool)
    monkeypatch.setattr(worker, "_quarantine_gpu", unexpected_quarantine)

    asyncio.run(worker._initialize_worker_pool())

    assert isinstance(worker.worker_pool, HealthyPool)
    assert worker.health_state == "healthy"


def test_worker_instance_id_is_unique_and_written_to_task_manager_and_status(monkeypatch) -> None:
    redis_client = RecordingRedis()
    worker = gpu_worker.GPUWorker("instance_gpu_0", "cuda:0", redis_client)
    other = gpu_worker.GPUWorker("instance_gpu_0", "cuda:0", FakeRedis())

    assert worker.worker_instance_id
    assert worker.worker_instance_id != other.worker_instance_id
    assert worker.task_manager.worker_instance_id == worker.worker_instance_id

    class HealthyPool:
        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    worker.worker_pool = HealthyPool()
    worker.health_state = "healthy"
    worker.running = True
    asyncio.run(worker._update_worker_status(online=True))

    assert redis_client.hsets[-1][1]["worker_instance_id"] == worker.worker_instance_id
    assert redis_client.hsets[-1][1]["accepting_tasks"] == "true"


def test_start_releases_frozen_inflight_only_after_fresh_gpu_gate(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    events = []

    class FakeSession:
        closed = False

        async def close(self):
            self.closed = True

    async def register():
        events.append("register")
        return True

    async def recover(worker_id, *, release_frozen_claims=False):  # noqa: ANN001
        assert worker_id == worker.worker_id
        assert worker.task_manager.worker_instance_id == worker.worker_instance_id
        assert release_frozen_claims is True
        events.append("recover")

    async def read_gate(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        events.append("gpu_gate")
        return None

    async def initialize_pool():
        events.append("pool_init")
        worker.worker_pool = object()
        worker.health_state = "healthy"

    async def noop(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return None

    monkeypatch.setitem(gpu_worker.settings.__dict__, "node_id", "test-node")
    monkeypatch.setattr(gpu_worker.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(gpu_worker, "read_gpu_quarantine", read_gate)
    monkeypatch.setattr(worker, "_register_with_api", register)
    monkeypatch.setattr(worker.task_manager, "recover_gpu_inflight", recover)
    monkeypatch.setattr(worker, "_initialize_worker_pool", initialize_pool)
    monkeypatch.setattr(worker, "_update_worker_status", noop)
    monkeypatch.setattr(worker, "_heartbeat_loop", noop)
    monkeypatch.setattr(worker, "_processing_loop", noop)
    monkeypatch.setattr(worker, "stop", noop)

    asyncio.run(worker.start())

    assert events == ["register", "gpu_gate", "pool_init", "recover"]


def test_start_does_not_recover_any_claim_while_quarantine_exists(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    events = []

    class FakeSession:
        closed = False

        async def close(self):
            self.closed = True

    async def register():
        events.append("register")
        return True

    async def read_gate(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        events.append("gpu_gate")
        return {
            "state": "quarantined",
            "scope": "physical_gpu",
            "reason": "unsafe child not reaped",
            "page_user_state": "sent",
        }

    async def notify(record):  # noqa: ANN001
        assert record["state"] == "quarantined"
        events.append("notify")

    async def unexpected_recover(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("quarantined replacement must not recover inflight claims")

    async def unexpected_pool_init():
        raise AssertionError("quarantined replacement must not initialize CUDA")

    async def noop(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return None

    monkeypatch.setitem(gpu_worker.settings.__dict__, "node_id", "test-node")
    monkeypatch.setattr(gpu_worker.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(gpu_worker, "read_gpu_quarantine", read_gate)
    monkeypatch.setattr(worker, "_register_with_api", register)
    monkeypatch.setattr(worker, "_ensure_quarantine_notification", notify)
    monkeypatch.setattr(worker.task_manager, "recover_gpu_inflight", unexpected_recover)
    monkeypatch.setattr(worker, "_initialize_worker_pool", unexpected_pool_init)
    monkeypatch.setattr(worker, "_update_worker_status", noop)
    monkeypatch.setattr(worker, "_heartbeat_loop", noop)
    monkeypatch.setattr(worker, "_processing_loop", noop)
    monkeypatch.setattr(worker, "stop", noop)

    asyncio.run(worker.start())

    assert events == ["register", "gpu_gate", "notify"]


def test_local_lifecycle_gate_fails_closed_without_touching_redis() -> None:
    class ExplodingRedis:
        def __getattr__(self, name):
            async def _unexpected(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
                raise AssertionError(f"Redis method {name} must not be called")

            return _unexpected

    worker = gpu_worker.GPUWorker("gate_gpu_0", "cuda:0", ExplodingRedis())
    assert asyncio.run(worker._gpu_admission_allowed()) is False

    worker.running = True
    worker._stopping = True
    assert asyncio.run(worker._gpu_admission_allowed()) is False


def test_local_lifecycle_gate_rechecks_after_redis_await(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.running = True
    started = asyncio.Event()
    release = asyncio.Event()

    class HealthyPool:
        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    async def delayed_gate(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        started.set()
        await release.wait()
        return None

    worker.worker_pool = HealthyPool()
    worker.health_state = "healthy"
    monkeypatch.setattr(gpu_worker, "read_gpu_quarantine", delayed_gate)

    async def scenario():
        admission = asyncio.create_task(worker._gpu_admission_allowed())
        await started.wait()
        worker._stopping = True
        release.set()
        return await admission

    assert asyncio.run(scenario()) is False


def test_processing_active_clears_when_loop_is_cancelled(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.running = True
    started = asyncio.Event()
    never = asyncio.Event()

    async def blocked_gate():
        started.set()
        await never.wait()
        return False

    monkeypatch.setattr(worker, "_gpu_admission_allowed", blocked_gate)

    async def scenario():
        processing = asyncio.create_task(worker._processing_loop())
        await started.wait()
        assert worker._processing_active is True
        processing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await processing

    asyncio.run(scenario())
    assert worker._processing_active is False


def test_pool_shutdown_exception_quarantines_and_retains_claim(monkeypatch) -> None:
    worker, failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_pool_exception"
    quarantines = []

    class RaisingPool:
        unsafe_shutdown_reason = ""

        async def shutdown(self, timeout):  # noqa: ANN001, ARG002
            raise RuntimeError("mock reap failure")

        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    async def capture_quarantine(**kwargs):  # noqa: ANN003
        quarantines.append(kwargs)

    worker.worker_pool = RaisingPool()
    monkeypatch.setattr(worker, "_quarantine_gpu", capture_quarantine)

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=3))

    assert failed == []
    assert worker.current_task == "task_pool_exception"
    assert isinstance(worker.worker_pool, RaisingPool)
    assert worker._shutdown_containment_safe is False
    assert quarantines[0]["fault_class"] == "unsafe_pool_shutdown"
    assert quarantines[0]["physical_scope"] is True


def test_unsafe_task_freezes_claim_before_slow_page_and_never_publishes_terminal(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.running = True
    events = []
    page_started = asyncio.Event()
    release_page = asyncio.Event()
    terminal_writes = []
    requeues = []

    async def unsafe_run(_task_data, _start_time):  # noqa: ANN001
        raise gpu_worker.UnsafeGPUContainmentError(
            "CUDA context reap could not be proven",
            task_id="task_slow_page",
            worker_id="pool-child-0",
        )

    async def freeze(task_id, reason):  # noqa: ANN001
        assert task_id == "task_slow_page"
        assert "automatic recovery" in reason
        events.append("freeze")
        return True

    async def slow_quarantine(**kwargs):  # noqa: ANN003
        assert events == ["freeze"]
        assert kwargs["fault_class"] == "pre_fault_reap_failure"
        assert kwargs["physical_scope"] is True
        events.append("page")
        page_started.set()
        await release_page.wait()

    async def unexpected_complete(task_id, result):  # noqa: ANN001
        terminal_writes.append((task_id, result))

    async def unexpected_requeue(task_data, reason):  # noqa: ANN001
        requeues.append((task_data, reason))

    monkeypatch.setattr(worker, "_process_toolkit_task", unsafe_run)
    monkeypatch.setattr(worker.task_manager, "freeze_task_claim", freeze)
    monkeypatch.setattr(worker, "_quarantine_gpu", slow_quarantine)
    monkeypatch.setattr(worker.task_manager, "complete_task", unexpected_complete)
    monkeypatch.setattr(worker.task_manager, "requeue_unstarted_task", unexpected_requeue)

    async def scenario():
        processing = asyncio.create_task(worker._process_task({"task_id": "task_slow_page"}))
        await asyncio.wait_for(page_started.wait(), timeout=1)

        assert processing.done() is False
        assert events == ["freeze", "page"]
        assert worker.running is False
        assert worker._stopping is True
        assert worker.shutdown_due_to_error is True
        assert worker.current_task == "task_slow_page"
        assert worker._shutdown_retained_task_id == "task_slow_page"
        assert terminal_writes == []
        assert requeues == []

        release_page.set()
        await asyncio.wait_for(processing, timeout=1)

    asyncio.run(scenario())

    assert worker.current_task == "task_slow_page"
    assert worker._shutdown_retained_task_id == "task_slow_page"
    assert terminal_writes == []
    assert requeues == []


def test_real_toolkit_path_preserves_unsafe_before_admission_page_or_cleanup_error(monkeypatch) -> None:
    """Exercise _run_toolkit_task instead of injecting above its finally block."""

    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.running = True
    events = []
    terminal_writes = []

    class UnsafePool:
        async def execute_task(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            await asyncio.sleep(0.01)
            events.append("pool_unsafe")
            raise gpu_worker.UnsafeGPUContainmentError(
                "real toolkit unsafe containment",
                task_id="task_real_unsafe",
                worker_id="pool-child-real",
            )

    async def noisy_watcher(*_args, **_kwargs):  # noqa: ANN002, ANN003
        await asyncio.sleep(0)
        raise RuntimeError("watcher cleanup must not mask unsafe")

    async def unexpected_admission():
        events.append("unexpected_admission")
        raise AssertionError("unsafe path must freeze before admission/page checks")

    async def freeze(task_id, reason):  # noqa: ANN001
        assert task_id == "task_real_unsafe"
        assert "automatic recovery" in reason
        events.append("freeze")
        return True

    async def quarantine(**kwargs):  # noqa: ANN003
        assert kwargs["physical_scope"] is True
        events.append("page")

    async def unexpected_complete(task_id, result):  # noqa: ANN001
        terminal_writes.append((task_id, result))

    worker.worker_pool = UnsafePool()
    monkeypatch.setattr(worker, "_cancellation_watcher", noisy_watcher)
    monkeypatch.setattr(worker, "_gpu_admission_allowed", unexpected_admission)
    monkeypatch.setattr(worker.task_manager, "freeze_task_claim", freeze)
    monkeypatch.setattr(worker, "_quarantine_gpu", quarantine)
    monkeypatch.setattr(worker.task_manager, "complete_task", unexpected_complete)

    asyncio.run(
        worker._process_task(
            {
                "task_id": "task_real_unsafe",
                "toolkit": "test-toolkit",
                "backend_adapter": "test-backend",
            }
        )
    )

    assert events == ["pool_unsafe", "freeze", "page"]
    assert terminal_writes == []
    assert worker.current_task == "task_real_unsafe"
    assert worker._shutdown_retained_task_id == "task_real_unsafe"


def test_toolkit_success_survives_post_result_admission_error(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)

    class SuccessfulPool:
        async def execute_task(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return {
                "success": True,
                "result": {"compiled": True, "metadata": {}},
                "pool_timing": {"pool_total_s": 0.1},
            }

    async def unavailable_admission():
        raise ConnectionError("Redis unavailable after result commit")

    worker.worker_pool = SuccessfulPool()
    monkeypatch.setattr(worker, "_gpu_admission_allowed", unavailable_admission)

    result = asyncio.run(worker._run_toolkit_task({"task_id": "successful-post-check"}))

    assert result["compiled"] is True
    assert result["metadata"]["wg_pool_total_s"] == 0.1


def test_shutdown_freezes_claim_before_unsafe_pool_reap(monkeypatch) -> None:
    worker, failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_freeze_before_reap"
    events = []

    class UnsafePool:
        unsafe_shutdown_reason = "child survived SIGKILL"

        async def shutdown(self, timeout):  # noqa: ANN001, ARG002
            assert events == ["freeze:task_freeze_before_reap"]
            events.append("pool_shutdown")
            return False

        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    async def freeze(task_id, reason):  # noqa: ANN001
        assert "automatic recovery is unsafe" in reason
        events.append(f"freeze:{task_id}")
        return True

    async def quarantine(**kwargs):  # noqa: ANN003
        assert kwargs["fault_class"] == "unsafe_pool_shutdown"
        assert kwargs["task_id"] == "task_freeze_before_reap"
        events.append("quarantine")

    worker.worker_pool = UnsafePool()
    monkeypatch.setattr(worker.task_manager, "freeze_task_claim", freeze)
    monkeypatch.setattr(worker, "_quarantine_gpu", quarantine)

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=3))

    assert events == ["freeze:task_freeze_before_reap", "pool_shutdown", "quarantine"]
    assert failed == []
    assert worker.current_task == "task_freeze_before_reap"
    assert worker._shutdown_containment_safe is False


def test_safe_containment_redis_failure_retains_claim_for_recovery(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_redis_failure"

    class SafePool:
        unsafe_shutdown_reason = ""

        async def shutdown(self, timeout):  # noqa: ANN001, ARG002
            return True

        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    async def failed_commit(
        task_id,
        reason,
        *,
        adopt_current_claim=False,
        allow_frozen_claim=False,
    ):  # noqa: ANN001, ARG001
        assert adopt_current_claim is False
        assert allow_frozen_claim is True
        raise ConnectionError("redis unavailable")

    worker.worker_pool = SafePool()
    monkeypatch.setattr(worker.task_manager, "fail_task", failed_commit)

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=3))

    assert worker.worker_pool is None
    assert worker._shutdown_containment_safe is True
    assert worker.current_task == "task_redis_failure"
    assert worker._shutdown_retained_task_id == "task_redis_failure"


def test_status_redis_failure_does_not_prevent_containment(monkeypatch) -> None:
    worker, failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_status_redis_failure"

    class FailingStatusRedis(FakeRedis):
        async def hset(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            raise ConnectionError("status Redis unavailable")

    class SafePool:
        unsafe_shutdown_reason = ""

        async def shutdown(self, timeout):  # noqa: ANN001, ARG002
            return True

        @staticmethod
        def get_health_snapshot():
            return {"health_state": "healthy", "accepting_tasks": True}

    worker.redis = FailingStatusRedis()
    worker.worker_pool = SafePool()

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=3))

    assert worker.worker_pool is None
    assert worker.current_task is None
    assert failed == [("task_status_redis_failure", "Worker shutdown")]


def test_stale_shutdown_claim_is_treated_as_handled_by_new_owner(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_superseded"

    async def stale_fail(
        task_id,
        reason,
        *,
        adopt_current_claim=False,
        allow_frozen_claim=False,
    ):  # noqa: ANN001, ARG001
        assert adopt_current_claim is False
        assert allow_frozen_claim is True
        raise gpu_worker.StaleTaskClaimError("replacement owns token")

    monkeypatch.setattr(worker.task_manager, "fail_task", stale_fail)

    asyncio.run(asyncio.wait_for(worker.stop(), timeout=3))

    assert worker.current_task is None
    assert worker._shutdown_retained_task_id is None


def test_shutdown_containment_exception_does_not_publish_ordinary_task_failure(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker._shutdown_retained_task_id = "task_reaping"
    completions = []

    async def interrupted(task_data, start_time):  # noqa: ANN001, ARG001
        raise RuntimeError("pool pipe closed by reap")

    async def unexpected_complete(task_id, result):  # noqa: ANN001
        completions.append((task_id, result))

    monkeypatch.setattr(worker, "_process_toolkit_task", interrupted)
    monkeypatch.setattr(worker.task_manager, "complete_task", unexpected_complete)

    asyncio.run(worker._process_task({"task_id": "task_reaping"}))

    assert completions == []
    assert worker.current_task == "task_reaping"


def test_shutdown_gpu_gate_exception_does_not_requeue_frozen_claim(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker._shutdown_retained_task_id = "task_gate_during_reap"
    requeues = []

    async def gate_closed(task_data, start_time):  # noqa: ANN001, ARG001
        raise gpu_worker.GPUQuarantinedError("pool is stopping")

    async def unexpected_requeue(task_data, reason):  # noqa: ANN001
        requeues.append((task_data, reason))

    monkeypatch.setattr(worker, "_process_toolkit_task", gate_closed)
    monkeypatch.setattr(worker.task_manager, "requeue_unstarted_task", unexpected_requeue)

    asyncio.run(worker._process_task({"task_id": "task_gate_during_reap"}))

    assert requeues == []
    assert worker.current_task == "task_gate_during_reap"


def test_normal_completion_stale_claim_does_not_retry_as_failure_or_count_cuda(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    commits = []

    async def successful_run(task_data):  # noqa: ANN001, ARG001
        return {"status": "completed"}

    async def stale_complete(task_id, result):  # noqa: ANN001
        commits.append((task_id, result["status"]))
        raise gpu_worker.StaleTaskClaimError("new owner")

    monkeypatch.setattr(worker, "_run_toolkit_task", successful_run)
    monkeypatch.setattr(worker.task_manager, "complete_task", stale_complete)

    task = {"task_id": "task_stale_complete", "toolkit": "mock", "backend_adapter": "mock"}
    asyncio.run(worker._process_task(task))

    assert commits == [("task_stale_complete", "completed")]
    assert worker.current_task is None
    assert worker.cuda_error_count == 0
    assert worker.stats["tasks_failed"] == 0


def test_normal_completion_redis_write_failure_is_not_retried_as_task_failure(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.running = True
    commits = []

    async def successful_run(task_data):  # noqa: ANN001, ARG001
        return {"status": "completed"}

    async def failed_complete(task_id, result):  # noqa: ANN001
        commits.append((task_id, result["status"]))
        raise ConnectionError("terminal write failed")

    monkeypatch.setattr(worker, "_run_toolkit_task", successful_run)
    monkeypatch.setattr(worker.task_manager, "complete_task", failed_complete)

    task = {"task_id": "task_commit_failure", "toolkit": "mock", "backend_adapter": "mock"}
    asyncio.run(worker._process_task(task))

    assert commits == [("task_commit_failure", "completed")]
    assert worker.current_task == "task_commit_failure"
    assert worker.running is False
    assert worker.shutdown_due_to_error is True


def test_heartbeat_rejection_keeps_current_task_until_stop(monkeypatch) -> None:
    worker, _failed = _make_worker(monkeypatch, drain_sec=0)
    worker.current_task = "task_heartbeat_rejected"
    worker.running = True
    stop_calls = []

    class FakeResponse:
        def __init__(self, status):  # noqa: ANN001
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def __await__(self):
            async def resolve():
                return self

            return resolve().__await__()

    class RejectingSession:
        closed = False

        def post(self, url, params):  # noqa: ANN001, ARG002
            return FakeResponse(409 if url.endswith("/worker/heartbeat") else 200)

    async def capture_stop():
        stop_calls.append(worker.current_task)

    worker.http_session = RejectingSession()
    monkeypatch.setattr(worker, "stop", capture_stop)

    allowed_to_continue = asyncio.run(worker._send_heartbeat_to_api())

    assert allowed_to_continue is False
    assert stop_calls == ["task_heartbeat_rejected"]
    assert worker.current_task == "task_heartbeat_rejected"
    assert worker.running is False
