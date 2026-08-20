"""CPU-worker heartbeat must not starve during blocking compiles.

The worker heartbeats from an asyncio task on the same event loop that
processes tasks. If toolkit.evaluate runs synchronously on the loop thread, a
compile longer than WORKER_MONITOR_HEARTBEAT_TIMEOUT (120 s) stops all
heartbeats and the worker monitor kills the worker mid-build, surfacing as
"Manual ninja build finished but .so file was not generated" compile failures.
"""

import asyncio
import time

import pytest


def _make_worker():
    pytest.importorskip("torch")
    from kernelgym.worker.cpu_worker import CPUCompileWorker

    class StubRedis:
        async def hset(self, *args, **kwargs):
            return 1

    class StubTaskManager:
        def __init__(self):
            self.completed = []

        async def complete_task(self, task_id, result):
            self.completed.append((task_id, result))

    class BlockingToolkit:
        def evaluate(self, task_data, backend=None):
            time.sleep(0.5)
            return {"task_id": task_data["task_id"], "metadata": {}}

    worker = CPUCompileWorker.__new__(CPUCompileWorker)
    worker.worker_id = "worker_cpu_test"
    worker.redis = StubRedis()
    worker.task_manager = StubTaskManager()
    worker.running = True
    worker.current_task = None
    worker.toolkit_cache = {"blocking_toolkit": BlockingToolkit()}
    worker.backend_cache = {"fake_backend": object()}
    worker.hostname = "test-host"
    worker.node_id = "test-node"
    return worker


def test_process_task_keeps_event_loop_responsive_during_blocking_evaluate() -> None:
    worker = _make_worker()
    ticks = []

    async def scenario():
        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        ticker_task = asyncio.create_task(ticker())
        try:
            await worker._process_task(
                {
                    "task_id": "hb_probe_task",
                    "toolkit": "blocking_toolkit",
                    "backend_adapter": "fake_backend",
                }
            )
        finally:
            ticker_task.cancel()
            try:
                await ticker_task
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())

    # The 0.5 s blocking evaluate must not freeze the loop: the 20 ms ticker
    # (standing in for the 10 s heartbeat loop) has to keep firing throughout.
    assert len(ticks) >= 10, f"event loop was starved during evaluate: {len(ticks)} ticks"
    completed = worker.task_manager.completed
    assert len(completed) == 1 and completed[0][0] == "hb_probe_task"
    assert worker.current_task is None
