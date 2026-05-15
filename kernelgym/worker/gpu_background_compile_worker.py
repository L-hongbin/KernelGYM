"""GPU worker variant with background split-compile scheduling."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict

from kernelgym.config import settings
from kernelgym.worker.gpu_worker import GPUWorker, logger
from kernelgym.worker.subprocess_pool import SubprocessWorkerPool


class BackgroundCompileGPUWorker(GPUWorker):
    """GPU worker that background-schedules split compile tasks."""

    def __init__(self, worker_id: str, device: str, redis_client):
        super().__init__(worker_id, device, redis_client)
        self.split_compile_mode = "background_compile"
        self.background_compile_limit = max(
            1, int(getattr(settings, "worker_background_compile_limit", 2))
        )
        self._background_compile_semaphore = asyncio.Semaphore(self.background_compile_limit)
        self._background_tasks: set[asyncio.Task] = set()
        self._background_task_ids: set[str] = set()
        self._background_split_compile_enabled = True
        self._warned_triton_non_split_tasks = False
        self.compile_worker_pool: SubprocessWorkerPool | None = None
        self.compile_pool_size = max(
            1, int(getattr(settings, "worker_compile_pool_size", self.background_compile_limit))
        )
        self.compile_max_tasks_per_worker = max(
            1, int(getattr(settings, "worker_compile_max_tasks_per_worker", 1))
        )

    async def start(self):
        logger.info(
            "Starting BackgroundCompileGPUWorker (GPU worker variant) %s on device %s (limit=%s)",
            self.worker_id,
            self.device,
            self.background_compile_limit,
        )
        await super().start()

    async def _initialize_gpu_worker_pools(self):
        await super()._initialize_gpu_worker_pools()
        logger.info(
            "Initializing compile worker pool for %s (device=%s, pool_size=%s, max_tasks_per_worker=%s)",
            self.worker_id,
            self.device,
            self.compile_pool_size,
            self.compile_max_tasks_per_worker,
        )
        try:
            self.compile_worker_pool = SubprocessWorkerPool(
                device_id=self.device_id,
                pool_size=self.compile_pool_size,
                worker_prefix=f"{self.worker_id}_compile_pool",
                max_tasks_per_worker=self.compile_max_tasks_per_worker,
            )
            logger.info(
                "Compile worker pool initialized successfully for %s with %s subprocess workers "
                "(max %s tasks per worker)",
                self.worker_id,
                self.compile_pool_size,
                self.compile_max_tasks_per_worker,
            )
        except Exception as exc:
            logger.error(f"Failed to initialize compile worker pool for {self.worker_id}: {exc}")
            await super()._shutdown_gpu_worker_pools()
            raise

    async def _shutdown_gpu_worker_pools(self):
        if self.compile_worker_pool:
            try:
                logger.info(f"Shutting down compile worker pool for {self.worker_id}...")
                await self.compile_worker_pool.shutdown(timeout=30)
                logger.info(f"Compile worker pool shut down successfully for {self.worker_id}")
            except Exception as exc:
                logger.error(f"Error shutting down compile worker pool: {exc}")
            finally:
                self.compile_worker_pool = None
        await super()._shutdown_gpu_worker_pools()

    async def _processing_loop(self):
        """Main processing loop with background compile scheduling."""
        logger.info(f"Worker {self.worker_id} processing loop started")

        while self.running:
            try:
                task_data = await self.task_manager.get_next_task(
                    self.worker_id,
                    worker_resource=self.worker_resource,
                )

                if task_data:
                    self._warn_if_background_compile_is_ineffective(task_data)
                    if self._should_background_compile_task(task_data):
                        await self._enqueue_background_compile_task(task_data)
                    else:
                        await self._process_task(task_data)
                else:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in processing loop for worker {self.worker_id}: {e}")

                from kernelgym.server.code_retry_manager import CodeRetryManager

                if CodeRetryManager(self.redis)._is_memory_error(str(e)):
                    logger.info(
                        f"[SUBPROCESS-ISOLATION] CUDA error detected in loop for worker {self.worker_id}, "
                        "but isolated in subprocess"
                    )
                else:
                    self.main_process_error_count += 1
                    logger.warning(
                        f"Main process error in worker {self.worker_id}: "
                        f"{self.main_process_error_count}/{self.max_main_process_errors}"
                    )

                    if self.main_process_error_count >= self.max_main_process_errors:
                        logger.error(
                            f"Worker {self.worker_id} main process has too many errors. Shutting down for restart."
                        )
                        await self.redis.hset(
                            f"{settings.redis_key_prefix}:worker:{self.worker_id}",
                            mapping={
                                "cuda_error_shutdown": "true",
                                "shutdown_reason": "main_process_errors",
                                "shutdown_time": datetime.now().isoformat(),
                            },
                        )
                        self.running = False
                        break

                await asyncio.sleep(5)

    def _warn_if_background_compile_is_ineffective(self, task_data: Dict[str, Any]) -> None:
        backend_name = str(task_data.get("backend") or "").strip().lower()
        task_stage = str(task_data.get("task_stage") or "").strip().lower()
        split_requested = bool(task_data.get("execute_after_compile_on_gpu")) or bool(
            task_data.get("split_compile_and_execute")
        )
        if self._warned_triton_non_split_tasks:
            return
        if backend_name != "triton":
            return
        if task_stage == "compile" or split_requested:
            return

        logger.warning(
            "BackgroundCompileGPUWorker is enabled, but task %s uses backend=triton without split compile; "
            "it will follow the legacy path and gain no background compile benefit.",
            task_data.get("task_id", "unknown"),
        )
        self._warned_triton_non_split_tasks = True

    async def _run_toolkit_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run task payload with background compile followed by serialized GPU execution."""
        per_task_timeout_sec = self.per_task_timeout_sec
        if "timeout" in task_data:
            logger.info(
                f"[Worker] Load per_task_timeout from payload: {task_data['timeout']}"
            )
            per_task_timeout_sec = task_data["timeout"]

        if self._is_compile_only_task(task_data):
            result_data = await self._run_compile_stage(task_data, per_task_timeout_sec)
            inline_execute = (
                self.worker_resource == "gpu"
                and self.worker_pool is not None
                and self.running
                and not getattr(self, "_stopping", False)
                and bool(task_data.get("execute_after_compile_on_gpu"))
                and not bool(task_data.get("pure_compile_task"))
            )
            compile_result = result_data.get("result", {})
            compile_metadata = compile_result.get("metadata") or {}
            compile_artifact = compile_metadata.get("_internal_compile_artifact") or compile_metadata.get(
                "compile_artifact"
            )
            if inline_execute and compile_result.get("compiled") is True and isinstance(compile_artifact, dict):
                result_data = await self._run_execute_after_compile(
                    task_data,
                    compile_artifact,
                    per_task_timeout_sec,
                )
        elif self.worker_resource == "cpu":
            result_data = await asyncio.wait_for(
                asyncio.to_thread(self._run_cpu_toolkit_task, task_data),
                timeout=per_task_timeout_sec,
            )
        else:
            result_data = await self.worker_pool.execute_task(
                task_data,
                timeout=per_task_timeout_sec,
                max_retries=2,
            )

        if not result_data.get("success", False):
            error_type = result_data.get("error_type", "Unknown")
            error_message = result_data.get("error_message", "Unknown error")
            raise RuntimeError(f"{error_type}: {error_message}")

        return result_data["result"]

    async def _run_compile_stage(self, task_data: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
        if self.compile_worker_pool is None:
            return await super()._run_compile_stage(task_data, timeout_sec)
        return await self.compile_worker_pool.execute_task(
            task_data,
            timeout=timeout_sec,
            max_retries=2,
        )
