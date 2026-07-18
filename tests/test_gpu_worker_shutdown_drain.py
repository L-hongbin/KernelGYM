"""Shutdown drain: SIGTERM-triggered stop() must let the in-flight task finish
(up to KERNELGYM_WORKER_SHUTDOWN_DRAIN_SEC) instead of failing it immediately,
and must still fail it once the drain window expires."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

# gpu_worker imports torch/redis/aiohttp at module scope; skip where unavailable.
pytest.importorskip("torch")
pytest.importorskip("redis")
pytest.importorskip("aiohttp")

ROOT = Path(__file__).resolve().parents[1]


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


def _make_worker(monkeypatch, drain_sec: int):
    # Inject via __dict__: pydantic's __setattr__ rejects the field when running
    # against an installed kernelgym whose Settings predates it.
    monkeypatch.setitem(gpu_worker.settings.__dict__, "worker_shutdown_drain_sec", drain_sec)
    worker = gpu_worker.GPUWorker("draintest_gpu_0", "cuda:0", FakeRedis())
    failed = []

    async def record_fail(task_id, reason):
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
