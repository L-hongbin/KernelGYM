"""CPU compile worker for split KernelBench CUDA-Agent tasks."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time
from typing import Any, Dict, Optional

import redis.asyncio as redis

from kernelgym.backend import get_backend
from kernelgym.config import settings, setup_logging
from kernelgym.server.task_manager import TaskManager
from kernelgym.toolkit import get_toolkit
from kernelgym.utils.error_classifier import classify_error
from kernelgym.utils.task_status import task_status_from_result_payload

logger = logging.getLogger("kernelgym.cpu_worker")


class CPUCompileWorker:
    """Worker that consumes CPU resource tasks and runs compile-only payloads."""

    def __init__(self, worker_id: str, redis_client: redis.Redis):
        self.worker_id = worker_id
        self.redis = redis_client
        self.task_manager = TaskManager(redis_client)
        self.running = False
        self.current_task: Optional[str] = None
        self.toolkit_cache: Dict[str, Any] = {}
        self.backend_cache: Dict[str, Any] = {}

    def _signal_handler(self, signum, frame):
        logger.info("CPU worker %s received signal %s", self.worker_id, signum)
        self.running = False

    async def start(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.running = True
        await self.task_manager.register_worker(self.worker_id, "cpu")
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            while self.running:
                task_data = await self.task_manager.get_next_task(self.worker_id, resources=["cpu"])
                if task_data is None:
                    await asyncio.sleep(0.1)
                    continue
                await self._process_task(task_data)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self.task_manager.unregister_worker(self.worker_id)

    async def _heartbeat_loop(self) -> None:
        while self.running:
            try:
                await self.task_manager.update_worker_heartbeat(self.worker_id)
            except Exception as exc:
                logger.warning("CPU worker heartbeat failed: %s", exc)
            await asyncio.sleep(10)

    async def _process_task(self, task_data: Dict[str, Any]) -> None:
        task_id = task_data["task_id"]
        self.current_task = task_id
        started = time.time()
        try:
            task_data.setdefault("task_stage", "compile")
            task_data.setdefault("required_resource", "cpu")
            task_data.setdefault("pure_compile_task", True)
            task_data.setdefault("return_internal_compile_artifact", True)

            toolkit_name = task_data.get("toolkit") or settings.default_toolkit
            backend_name = task_data.get("backend_adapter") or settings.default_backend_adapter
            if toolkit_name not in self.toolkit_cache:
                self.toolkit_cache[toolkit_name] = get_toolkit(toolkit_name)
            if backend_name not in self.backend_cache:
                self.backend_cache[backend_name] = get_backend(backend_name)

            result = self.toolkit_cache[toolkit_name].evaluate(task_data, backend=self.backend_cache[backend_name])
            metadata = result.setdefault("metadata", {})
            metadata["cpu_worker_id"] = self.worker_id
            metadata["cpu_worker_run_s"] = time.time() - started
            await self.task_manager.complete_task(task_id, result)
        except Exception as exc:
            error_code = classify_error(str(exc), "runtime")
            result = {
                "task_id": task_id,
                "base_task_id": task_data.get("base_task_id", task_id),
                "compiled": False,
                "correctness": False,
                "decoy_kernel": False,
                "kernel_runtime": -1.0,
                "metadata": {"error": str(exc), "cpu_worker_id": self.worker_id},
                "status": task_status_from_result_payload({"status": "failed", "error_code": error_code}).value,
                "error_message": f"CPU compile task failed: {exc}",
                "error_code": error_code,
            }
            await self.task_manager.complete_task(task_id, result)
        finally:
            self.current_task = None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Start a CPU compile worker")
    parser.add_argument("--worker-id", required=True)
    _args = parser.parse_args()
    setup_logging(f"cpu_worker_{_args.worker_id}")
    redis_client = redis.from_url(settings.redis_url)
    await redis_client.ping()
    worker = CPUCompileWorker(_args.worker_id, redis_client)
    try:
        await worker.start()
    finally:
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
