"""
Subprocess Worker Pool with CUDA Error Auto-Restart

核心特性：
1. 预先启动一组 worker 进程，复用处理多个任务
2. torch 和 CUDA 只在启动时初始化一次
3. **第一次遇到 CUDA error 时立即关闭 worker 进程**
4. 主进程自动重启新的 worker 进程
5. 大幅降低 spawn 开销（从每任务 2.5s 降至几乎为 0）

Author: KernelGym Team
Date: 2025-10-30
Version: v0.3.3-rc
"""

import os
import signal
import sys
import time
import logging
import traceback
import multiprocessing as mp
import queue
import asyncio
import threading
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("kernelgym.subprocess_pool")


_STAGE_METADATA_PATH_ENV = "KERNELGYM_STAGE_METADATA_PATH"
_STAGE_METADATA_DIR_ENV = "KERNELGYM_STAGE_METADATA_DIR"
_STAGE_METADATA_DEFAULT_DIR = "/dev/shm/kernelgym/stage_metadata"
_FAST_RW_ROOT = "/dev/shm"


def _require_fast_rw_dir(path_value: str, *, label: str) -> str:
    path = os.path.abspath(path_value)
    root = os.path.abspath(_FAST_RW_ROOT)
    if path != root and not path.startswith(root + os.sep):
        raise ValueError(f"{label} must be under /dev/shm for fast local I/O: {path}")
    os.makedirs(path, exist_ok=True)
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"{label} is not writable/executable: {path}")
    return path


def _aggressive_gpu_cleanup(device_id: int):
    """
    强力清理 GPU 显存

    这个函数会尝试多种方法来清理显存：
    1. 清空 PyTorch 缓存
    2. 收集 Python 垃圾
    3. 重置 CUDA 峰值内存统计
    4. 清空 Triton 缓存（如果有）
    5. 同步 CUDA 操作

    Args:
        device_id: GPU 设备 ID
    """
    import torch
    import gc

    # 1. 同步所有 CUDA 操作
    try:
        torch.cuda.synchronize(device_id)
    except Exception:
        pass

    # 2. 清空 PyTorch 缓存
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 3. Python 垃圾回收（释放未引用的张量）
    gc.collect()

    # 4. 再次清空缓存
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 5. 重置内存统计（帮助下次分配）
    try:
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.reset_accumulated_memory_stats(device_id)
    except Exception:
        pass

    # 6. Triton 编译的 kernel 缓存可能残留，但 Triton 没有公开的清理 API；
    #    进程退出时会自动清理。

    # 7. 最终同步
    try:
        torch.cuda.synchronize(device_id)
    except Exception:
        pass


@dataclass
class WorkerMetrics:
    """Worker 执行指标"""

    task_execution_time: float  # 任务执行时间
    total_time: float  # 总时间（包括 queue 等待）
    success: bool = True
    error_type: Optional[str] = None


class PersistentWorker:
    """
    持久化的 worker 进程

    特性：
    - 启动时一次性初始化 torch 和 CUDA
    - 通过 Queue 接收任务并返回结果
    - **遇到 CUDA error 立即退出（通过特殊标记）**
    - 主进程检测到退出后会重启新的 worker
    """

    def __init__(self, worker_id: str, device_id: int, pool_size_info: str = "", max_tasks_per_worker: int = 100):
        """
        Args:
            worker_id: Worker 标识符（如 "worker_0"）
            device_id: GPU 设备 ID（如 0-7）
            pool_size_info: 用于日志的 pool 大小信息
            max_tasks_per_worker: 每个 worker 最多处理的任务数（防止显存累积）
        """
        self.worker_id = worker_id
        self.device_id = device_id
        self.pool_size_info = pool_size_info
        self.max_tasks_per_worker = max_tasks_per_worker
        self.process: Optional[mp.Process] = None

        # 使用 spawn context 确保完全隔离
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue(maxsize=10)  # 限制队列大小，避免内存爆炸
        self.result_queue = self.ctx.Queue(maxsize=10)

        self.is_alive_flag = True
        self.tasks_processed = 0
        self.start_time = time.time()

        # 启动 worker 进程
        self._start_worker()

    def _start_worker(self):
        """启动 worker 进程"""
        logger.info(f"[{self.worker_id}] Starting persistent worker for GPU {self.device_id} {self.pool_size_info}")

        self.process = self.ctx.Process(
            target=_persistent_worker_loop,
            args=(self.worker_id, self.device_id, self.task_queue, self.result_queue),
            daemon=False,  # 不使用 daemon，确保可以正常清理
        )

        self.process.start()

        # 等待初始化完成
        try:
            init_msg = self.result_queue.get(timeout=120)  # 给足够时间加载 torch (increased from 60s)
            if init_msg.get("status") == "READY":
                logger.info(
                    f"[{self.worker_id}] Worker initialized successfully "
                    f"(init_time={init_msg.get('init_time', 0):.2f}s)"
                )
                self.is_alive_flag = True
            else:
                raise RuntimeError(f"Worker failed to initialize: {init_msg}")
        except queue.Empty:
            self.process.terminate()
            self.process.join(timeout=5)
            raise RuntimeError(f"[{self.worker_id}] Worker initialization timeout (>120s)")

    def execute_task(self, task_data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        执行任务

        Args:
            task_data: 任务数据字典
            timeout: 超时时间（秒）

        Returns:
            结果字典，包含 success, result/error_type/error_message

        Raises:
            RuntimeError: Worker 已死亡或任务执行失败
            TimeoutError: 任务超时
        """
        if not self.is_alive():
            raise RuntimeError(f"[{self.worker_id}] Worker is not alive")

        # 发送任务
        try:
            self.task_queue.put(task_data, timeout=5)
        except queue.Full:
            raise RuntimeError(f"[{self.worker_id}] Task queue is full")

        # 等待结果
        try:
            result = self.result_queue.get(timeout=timeout)

            # 检查 worker 是否报告 CUDA error 并准备退出
            if result.get("worker_exiting") is True:
                logger.warning(
                    f"[{self.worker_id}] Worker encountered CUDA error and will exit. "
                    f"Error: {result.get('error_type', 'Unknown')}: {result.get('error_message', 'N/A')}"
                )
                self.is_alive_flag = False
                # 标记进程将要退出，主进程会重启

            # 更新统计
            self.tasks_processed += 1

            # **关键：检查是否达到任务上限（防止显存累积）**
            if self.tasks_processed >= self.max_tasks_per_worker:
                logger.info(
                    f"[{self.worker_id}] Reached max tasks limit ({self.max_tasks_per_worker}). "
                    f"Marking for restart to prevent memory accumulation."
                )
                self.is_alive_flag = False
                # 注意：我们不立即关闭进程，而是让它在下次检查时被重启
                # 这样可以先返回当前任务的结果

            return result

        except queue.Empty:
            # 超时
            logger.error(
                f"[{self.worker_id}] Task timeout after {timeout}s, task_id={task_data.get('task_id', 'unknown')}"
            )
            # 标记 worker 为不可用（可能卡死了）
            self.is_alive_flag = False
            raise TimeoutError(
                f"[{self.worker_id}] Task {task_data.get('task_id', 'unknown')} timeout after {timeout}s"
            )

    def is_alive(self) -> bool:
        """检查 worker 是否存活"""
        return self.is_alive_flag and self.process is not None and self.process.is_alive()

    def shutdown(self, timeout: int = 10):
        """关闭 worker 进程

        Ensures the child process is fully reaped (joined) so that the
        CUDA driver releases its GPU memory.  Every kill path is followed
        by ``process.join()`` and ``process.close()``.
        """
        logger.info(f"[{self.worker_id}] Shutting down worker...")

        try:
            # Send graceful shutdown sentinel -- worker will run GPU
            # cleanup and exit on its own.
            self.task_queue.put({"command": "GRACEFUL_SHUTDOWN"}, timeout=2)
        except Exception:
            # Queue full / broken -- fall through to SHUTDOWN then kill.
            pass

        try:
            # Fallback: legacy SHUTDOWN command (immediate exit).
            try:
                self.task_queue.put({"command": "SHUTDOWN"}, timeout=1)
            except Exception:
                pass

            # Always join to reap zombie and release CUDA context,
            # even if process already exited (is_alive() == False).
            if self.process:
                self.process.join(timeout=timeout)

                # Escalate: SIGTERM (only if still alive after join)
                if self.process.is_alive():
                    logger.warning(f"[{self.worker_id}] Force terminating worker")
                    self.process.terminate()
                    self.process.join(timeout=3)

                    # Escalate: SIGKILL
                    if self.process.is_alive():
                        logger.error(f"[{self.worker_id}] Force killing worker")
                        self.process.kill()
                        # CRITICAL: must join after kill to reap zombie
                        # and release CUDA driver context
                        self.process.join(timeout=10)

        except Exception as e:
            logger.error(f"[{self.worker_id}] Error during shutdown: {e}")
            if self.process:
                try:
                    self.process.kill()
                    # CRITICAL: reap the zombie even in error path
                    self.process.join(timeout=10)
                except Exception as kill_err:
                    logger.error(f"[{self.worker_id}] Failed to kill/join in error handler: {kill_err}")

        # Release multiprocessing.Process internal resources (fds, etc.)
        if self.process is not None:
            try:
                self.process.close()
            except (ValueError, Exception) as close_err:
                # close() raises ValueError if process is still alive
                logger.warning(f"[{self.worker_id}] process.close() failed: {close_err}")

        self.is_alive_flag = False
        logger.info(
            f"[{self.worker_id}] Worker shut down "
            f"(processed {self.tasks_processed} tasks in "
            f"{time.time() - self.start_time:.1f}s)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取 worker 统计信息"""
        return {
            "worker_id": self.worker_id,
            "device_id": self.device_id,
            "is_alive": self.is_alive(),
            "tasks_processed": self.tasks_processed,
            "uptime": time.time() - self.start_time,
            "pid": self.process.pid if self.process else None,
        }


class SubprocessWorkerPool:
    """
    Worker Pool 管理器

    职责：
    1. 管理多个 PersistentWorker
    2. 分配任务到空闲的 worker
    3. **自动重启遇到 CUDA error 的 worker**
    4. 负载均衡
    """

    def __init__(
        self, device_id: int, pool_size: int = 2, worker_prefix: str = "pool_worker", max_tasks_per_worker: int = 100
    ):
        """
        Args:
            device_id: GPU 设备 ID
            pool_size: Worker 进程数量（建议 2-4，根据内存大小调整）
            worker_prefix: Worker ID 前缀
            max_tasks_per_worker: 每个 worker 最多处理的任务数（防止显存累积，默认100）
        """
        self.device_id = device_id
        self.pool_size = pool_size
        self.worker_prefix = worker_prefix
        self.max_tasks_per_worker = max_tasks_per_worker

        # Workers 列表
        self.workers: List[PersistentWorker] = []
        self.idle_workers: List[PersistentWorker] = []
        self.busy_workers: List[PersistentWorker] = []

        # 统计
        self.total_tasks_processed = 0
        self.total_workers_restarted = 0
        self.pool_start_time = time.time()

        # 同步锁
        self.lock = asyncio.Lock()

        # 初始化 workers
        self._init_workers()

        # Start background zombie reaper thread
        self._reaper_stop = threading.Event()
        self._reaper_thread = threading.Thread(
            target=self._zombie_reaper_loop,
            daemon=True,
            name=f"zombie-reaper-gpu{device_id}",
        )
        self._reaper_thread.start()

        logger.info(f"[GPU {device_id}] Worker pool initialized with {pool_size} workers")

    def _init_workers(self):
        """初始化所有 workers"""
        pool_info = f"(pool_size={self.pool_size}, max_tasks={self.max_tasks_per_worker})"

        for i in range(self.pool_size):
            worker_id = f"{self.worker_prefix}_{self.device_id}_{i}"
            try:
                worker = PersistentWorker(
                    worker_id, self.device_id, pool_info, max_tasks_per_worker=self.max_tasks_per_worker
                )
                self.workers.append(worker)
                self.idle_workers.append(worker)
            except Exception as e:
                logger.error(f"[GPU {self.device_id}] Failed to start worker {worker_id}: {e}")
                # 如果启动失败，尝试继续启动其他 workers
                # 至少要有一个 worker 成功启动
                if len(self.workers) == 0 and i == self.pool_size - 1:
                    raise RuntimeError(f"[GPU {self.device_id}] Failed to start any worker in pool")

    async def execute_task(self, task_data: Dict[str, Any], timeout: int = 60, max_retries: int = 2) -> Dict[str, Any]:
        """
        执行任务（自动选择空闲 worker）

        Args:
            task_data: 任务数据
            timeout: 超时时间
            max_retries: 最大重试次数（用于 worker 重启后重试）
                        注意：timeout错误不会重试，以避免阻塞队列

        Returns:
            结果字典
        """
        retry_count = 0
        last_error = None
        is_timeout_error = False  # Track if error was timeout
        task_id = task_data.get("task_id", "unknown")
        request_start = time.time()
        total_idle_wait_s = 0.0
        total_restart_s = 0.0
        last_execute_s = 0.0
        last_return_s = 0.0
        stage_dir = _require_fast_rw_dir(
            os.environ.get(_STAGE_METADATA_DIR_ENV, _STAGE_METADATA_DEFAULT_DIR),
            label=_STAGE_METADATA_DIR_ENV,
        )
        stage_metadata_path = os.path.join(
            stage_dir,
            f"kernelgym_stage_{os.getpid()}_{self.device_id}_{uuid.uuid4().hex}.json",
        )
        task_data["_stage_metadata_path"] = stage_metadata_path

        def _build_pool_timing(total_s: Optional[float] = None) -> Dict[str, Any]:
            final_total = time.time() - request_start if total_s is None else total_s
            return {
                "pool_idle_wait_s": total_idle_wait_s,
                "pool_execute_s": last_execute_s,
                "pool_restart_s": total_restart_s,
                "pool_return_s": last_return_s,
                "pool_total_s": final_total,
                "pool_retry_count": retry_count,
            }

        while retry_count <= max_retries:
            # 获取空闲 worker
            idle_wait_start = time.time()
            worker = await self._get_idle_worker(timeout=timeout)
            total_idle_wait_s += time.time() - idle_wait_start

            if worker is None:
                # 所有 workers 都忙，等待一下再试
                await asyncio.sleep(0.5)
                retry_count += 1
                continue

            try:
                # 执行任务（在线程池中执行，避免阻塞 asyncio）
                loop = asyncio.get_event_loop()
                execute_start = time.time()
                result = await loop.run_in_executor(None, worker.execute_task, task_data, timeout)
                last_execute_s = time.time() - execute_start

                # 任务完成
                self.total_tasks_processed += 1

                # 检查 worker 是否需要重启
                if not worker.is_alive():
                    logger.warning(f"[{worker.worker_id}] Worker needs restart after task")
                    restart_start = time.time()
                    await self._restart_worker(worker)
                    total_restart_s += time.time() - restart_start

                pool_timing = _build_pool_timing()
                result["pool_timing"] = pool_timing
                logger.info(
                    f"[PoolTiming] device=cuda:{self.device_id} worker={worker.worker_id} "
                    f"task={task_id} status=success idle_wait_s={pool_timing['pool_idle_wait_s']:.2f} "
                    f"execute_s={pool_timing['pool_execute_s']:.2f} restart_s={pool_timing['pool_restart_s']:.2f} "
                    f"return_s={pool_timing['pool_return_s']:.2f} total_s={pool_timing['pool_total_s']:.2f} "
                    f"retries={pool_timing['pool_retry_count']}"
                )

                return result

            except (RuntimeError, TimeoutError) as e:
                # Worker 可能已死亡或超时
                logger.error(f"[{worker.worker_id}] Task execution failed: {e}")
                last_error = e

                # Check if this is a timeout error
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "Task timeout after" in error_msg:
                    is_timeout_error = True
                    task_id = task_data.get("task_id", "unknown")
                    logger.warning(
                        f"[{worker.worker_id}] Task {task_id} timeout detected "
                        f"(timeout={timeout}s) - will NOT retry to avoid blocking queue"
                    )

                # 尝试重启 worker
                restart_start = time.time()
                await self._restart_worker(worker)
                total_restart_s += time.time() - restart_start

                # Don't retry if timeout - exit immediately to free up worker queue
                if is_timeout_error:
                    pool_timing = _build_pool_timing()
                    logger.info(
                        f"[PoolTiming] device=cuda:{self.device_id} worker={worker.worker_id} "
                        f"task={task_id} status=timeout idle_wait_s={pool_timing['pool_idle_wait_s']:.2f} "
                        f"execute_s={pool_timing['pool_execute_s']:.2f} restart_s={pool_timing['pool_restart_s']:.2f} "
                        f"return_s={pool_timing['pool_return_s']:.2f} total_s={pool_timing['pool_total_s']:.2f} "
                        f"retries={pool_timing['pool_retry_count']}"
                    )
                    logger.error(
                        f"[{worker.worker_id}] Task failed due to timeout, not retrying to free up worker queue"
                    )
                    break  # Exit retry loop immediately

                # 重试（仅对非timeout错误）
                retry_count += 1

            finally:
                # 归还 worker 到 idle pool（如果还存活）
                return_start = time.time()
                await self._return_worker(worker)
                last_return_s = time.time() - return_start

        # Failed after all retries (or timeout)
        if is_timeout_error:
            task_id = task_data.get("task_id", "unknown")
            raise TimeoutError(
                f"[GPU {self.device_id}] Task {task_id} timeout after {timeout}s. "
                f"Not retried to avoid blocking worker queue."
            )
        else:
            pool_timing = _build_pool_timing()
            logger.info(
                f"[PoolTiming] device=cuda:{self.device_id} task={task_id} status=failed "
                f"idle_wait_s={pool_timing['pool_idle_wait_s']:.2f} "
                f"execute_s={pool_timing['pool_execute_s']:.2f} restart_s={pool_timing['pool_restart_s']:.2f} "
                f"return_s={pool_timing['pool_return_s']:.2f} total_s={pool_timing['pool_total_s']:.2f} "
                f"retries={pool_timing['pool_retry_count']}"
            )
            raise RuntimeError(
                f"[GPU {self.device_id}] Task failed after {max_retries} retries. Last error: {last_error}"
            )

    async def _get_idle_worker(self, timeout: int = 60) -> Optional[PersistentWorker]:
        """
        获取一个空闲的 worker

        如果所有 workers 都忙，会等待直到有 worker 空闲
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            async with self.lock:
                # 清理已死亡的 workers
                self.idle_workers = [w for w in self.idle_workers if w.is_alive()]

                # Emergency recovery: if pool has no workers at all, try to create one
                if not self.workers and not self.idle_workers and not self.busy_workers:
                    logger.warning(f"[GPU {self.device_id}] Pool has no workers! Attempting emergency recovery...")
                    try:
                        # Wait a bit for GPU resources to be released
                        await asyncio.sleep(3.0)

                        emergency_worker = PersistentWorker(
                            f"worker_gpu_{self.device_id}_pool_{self.device_id}_emergency",
                            self.device_id,
                            "(emergency recovery)",
                            max_tasks_per_worker=self.max_tasks_per_worker,
                        )
                        self.workers.append(emergency_worker)
                        self.idle_workers.append(emergency_worker)
                        logger.info(f"[GPU {self.device_id}] Emergency worker created successfully")
                    except Exception as e:
                        logger.error(f"[GPU {self.device_id}] Emergency recovery failed: {e}")

                if self.idle_workers:
                    worker = self.idle_workers.pop(0)
                    self.busy_workers.append(worker)
                    return worker

            # 没有空闲 worker，等待一下
            await asyncio.sleep(0.1)

        # 超时
        logger.error(f"[GPU {self.device_id}] No idle worker available after {timeout}s")
        return None

    async def _return_worker(self, worker: PersistentWorker):
        """归还 worker 到 idle pool"""
        async with self.lock:
            if worker in self.busy_workers:
                self.busy_workers.remove(worker)

            # 只有存活的 worker 才归还到 idle pool
            if worker.is_alive():
                if worker not in self.idle_workers:
                    self.idle_workers.append(worker)

    async def _restart_worker(self, worker: PersistentWorker):
        """
        重启一个 worker (non-blocking, warm-spare promotion).

        Fast path (under lock):
        1. Remove the old worker from all tracking lists immediately.
        2. The existing idle spare(s) in the pool are already available
           for ``_get_idle_worker`` to hand out on the next task.

        Slow path (background thread — does NOT block the caller):
        3. Shut down the old worker process (synchronous, ~0.5-5 s).
        4. Sleep 2 s so GPU resources are fully released.
        5. Spawn a fresh ``PersistentWorker`` as a replacement spare and
           register it back into the pool (under lock).
        """
        async with self.lock:
            logger.info(
                f"[{worker.worker_id}] Recycling worker "
                f"(processed {worker.tasks_processed} tasks) — "
                f"spare replenishment will happen in background"
            )

            # --- fast: remove the dead worker from every list -----------
            if worker in self.workers:
                self.workers.remove(worker)
            if worker in self.idle_workers:
                self.idle_workers.remove(worker)
            if worker in self.busy_workers:
                self.busy_workers.remove(worker)

            spare_count = len(self.idle_workers)
            logger.info(
                f"[{worker.worker_id}] Pool state after removal: "
                f"workers={len(self.workers)} idle={spare_count} "
                f"busy={len(self.busy_workers)}"
            )

        # --- slow: fire-and-forget background replenishment ------------
        # CRITICAL: capture the actual Process object and PID eagerly,
        # BEFORE entering the background thread.  The old code captured
        # ``old_worker`` (an alias for the *same* PersistentWorker object)
        # and dereferenced ``old_worker.process`` lazily inside the
        # thread.  By that time, ``shutdown()`` may have called
        # ``.close()`` on the Process, putting it in an invalid state
        # where ``.kill()`` / ``.is_alive()`` / ``.join()`` raise
        # ValueError.  Capturing the raw objects here avoids that class
        # of bugs entirely.
        old_worker = worker
        old_process = worker.process  # multiprocessing.Process – may be None
        old_pid = old_process.pid if old_process is not None else None
        old_wid = worker.worker_id  # immutable str, safe to read later
        _loop = asyncio.get_running_loop()

        def _background_replenish():
            """Run in a daemon thread — shuts down old worker, waits for
            GPU release, creates a new spare, and re-registers it.

            CRITICAL: uses only ``old_process`` / ``old_pid`` captured
            before the thread started. Never touches ``old_worker.process``
            which may have been replaced by a new spare by now.
            """
            # 1. Shut down the old process directly (not via old_worker.shutdown()
            #    which would operate on old_worker.process — potentially replaced)
            if old_process is not None:
                try:
                    # Send graceful shutdown via the old worker's queue
                    old_worker.task_queue.put({"command": "GRACEFUL_SHUTDOWN"}, timeout=2)
                except Exception:
                    pass
                try:
                    old_worker.task_queue.put({"command": "SHUTDOWN"}, timeout=1)
                except Exception:
                    pass

                # Wait for exit, then escalate
                try:
                    old_process.join(timeout=5)
                except Exception:
                    pass

                if old_process.is_alive():
                    try:
                        old_process.terminate()
                        old_process.join(timeout=3)
                    except Exception:
                        pass

                if old_process.is_alive():
                    try:
                        old_process.kill()
                        old_process.join(timeout=5)
                    except Exception:
                        pass

                # Fallback: direct os.kill
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

                # Always try to reap
                try:
                    old_process.join(timeout=5)
                except Exception:
                    pass
                try:
                    os.waitpid(old_pid, 0)
                except Exception:
                    pass
                try:
                    old_process.close()
                except Exception:
                    pass

                logger.info(f"[{old_wid}] Old process pid={old_pid} cleanup done")

            # 2. Verify the old process is actually dead; escalate if not.
            #    All checks use ``old_process`` / ``old_pid`` captured
            #    before the thread started — never ``old_worker.process``.
            def _process_still_alive() -> bool:
                """Check whether the old worker subprocess is still running."""
                if old_pid is None:
                    return False
                try:
                    os.kill(old_pid, 0)
                    return True
                except ProcessLookupError:
                    return False
                except OSError:
                    # Permission error etc — assume alive to be safe
                    return True

            if _process_still_alive():
                logger.warning(
                    f"[{old_wid}] Worker pid={old_pid} still alive after shutdown(timeout=5); escalating to SIGKILL"
                )
                # Force kill via the captured Process handle
                try:
                    if old_process is not None:
                        old_process.kill()  # sends SIGKILL for mp.Process
                        logger.warning(f"[{old_wid}] Sent process.kill() to pid={old_pid}")
                except Exception as kill_exc:
                    logger.warning(f"[{old_wid}] process.kill() failed: {kill_exc}")

                # Fallback: os.kill with SIGKILL (works even if the
                # Process object is in a bad state after close())
                if old_pid is not None and _process_still_alive():
                    try:
                        os.kill(old_pid, signal.SIGKILL)
                        logger.warning(f"[{old_wid}] Sent os.kill(SIGKILL) to pid={old_pid}")
                    except ProcessLookupError:
                        logger.warning(f"[{old_wid}] pid={old_pid} already gone before os.kill")
                    except Exception as os_kill_exc:
                        logger.warning(f"[{old_wid}] os.kill(SIGKILL) failed for pid={old_pid}: {os_kill_exc}")

                # Reap the zombie to avoid pid table leak and release CUDA VRAM
                try:
                    if old_process is not None:
                        old_process.join(timeout=10)
                        logger.warning(f"[{old_wid}] Reaped killed worker pid={old_pid}")
                except Exception as reap_exc:
                    logger.warning(f"[{old_wid}] Failed to reap pid={old_pid}: {reap_exc}")
                    # Last resort: try os.waitpid directly
                    if old_pid is not None:
                        try:
                            os.waitpid(old_pid, os.WNOHANG)
                            logger.warning(f"[{old_wid}] os.waitpid fallback for pid={old_pid}")
                        except ChildProcessError:
                            pass  # already reaped
                        except Exception:
                            pass

                # Verification: confirm the kill actually worked
                if old_pid is not None:
                    try:
                        os.kill(old_pid, 0)
                        logger.error(
                            f"[{old_wid}] KILL VERIFICATION FAILED: pid={old_pid} still alive after SIGKILL + join"
                        )
                    except ProcessLookupError:
                        logger.info(f"[{old_wid}] KILL VERIFIED: pid={old_pid} confirmed dead")
                    except OSError as verify_exc:
                        logger.warning(f"[{old_wid}] KILL VERIFY INCONCLUSIVE for pid={old_pid}: {verify_exc}")

            # Release multiprocessing.Process internal resources
            if old_process is not None:
                try:
                    old_process.close()
                except (ValueError, Exception):
                    pass

            # Safety-net check: log if the process is somehow still alive
            if old_pid is not None:
                try:
                    os.kill(old_pid, 0)
                    # Still alive — log error but don't block replacement
                    logger.error(
                        f"[{old_wid}] VRAM LEAK RISK: pid={old_pid} still "
                        f"alive after shutdown + SIGKILL + reap escalation"
                    )
                except (ProcessLookupError, OSError):
                    pass  # confirmed dead — good

            # 3. Wait for GPU driver to reclaim VRAM from the dead process
            time.sleep(2.0)

            # 4. Create a replacement spare worker
            try:
                new_worker = PersistentWorker(
                    old_wid,
                    self.device_id,
                    f"(pool_size={self.pool_size}, max_tasks={self.max_tasks_per_worker}, restart)",
                    max_tasks_per_worker=self.max_tasks_per_worker,
                )
            except Exception as e:
                logger.error(
                    f"[{old_wid}] Failed to create replacement spare: {e}. Pool now has {len(self.workers)} workers"
                )
                return

            # 5. Register the new spare back into the pool (thread-safe
            #    via the asyncio event-loop).
            async def _register():
                async with self.lock:
                    self.workers.append(new_worker)
                    self.idle_workers.append(new_worker)
                    self.total_workers_restarted += 1
                    logger.info(
                        f"[{new_worker.worker_id}] Background spare ready — "
                        f"pool: workers={len(self.workers)} idle={len(self.idle_workers)} "
                        f"busy={len(self.busy_workers)} "
                        f"(total restarts: {self.total_workers_restarted})"
                    )

            # Schedule the coroutine on the event loop from this thread.
            asyncio.run_coroutine_threadsafe(_register(), _loop)

        t = threading.Thread(target=_background_replenish, daemon=True)

        t.start()

    def _zombie_reaper_loop(self):
        """Periodically reap zombie child processes.

        The CUDA driver keeps GPU memory allocated for a process until the
        parent calls waitpid().  If any worker-shutdown path missed the
        join(), the zombie lingers and VRAM leaks.  This thread acts as a
        safety net by periodically calling ``multiprocessing.active_children()``
        (which internally reaps finished children) and explicitly waiting on
        any known dead processes.
        """
        INTERVAL = 30  # seconds between sweeps
        while not self._reaper_stop.wait(timeout=INTERVAL):
            try:
                # active_children() calls waitpid(WNOHANG) for every
                # child Process that multiprocessing knows about.
                # This is the cheapest way to reap zombies.
                mp.active_children()
                # Also try a blanket waitpid to catch anything
                # multiprocessing doesn't track.
                try:
                    while True:
                        pid, status = os.waitpid(-1, os.WNOHANG)
                        if pid == 0:
                            break
                        logger.info(f"[GPU {self.device_id}] Zombie reaper: reaped pid={pid} status={status}")
                except ChildProcessError:
                    pass  # no more children
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"[GPU {self.device_id}] Zombie reaper error: {e}")

    async def shutdown(self, timeout: int = 30):
        """关闭整个 worker pool"""
        logger.info(f"[GPU {self.device_id}] Shutting down worker pool...")

        # Stop the zombie reaper thread
        self._reaper_stop.set()

        # 关闭所有 workers
        for worker in self.workers:
            try:
                worker.shutdown(timeout=timeout // len(self.workers) if self.workers else 5)
            except Exception as e:
                logger.error(f"Error shutting down {worker.worker_id}: {e}")

        self.workers.clear()
        self.idle_workers.clear()
        self.busy_workers.clear()

        logger.info(
            f"[GPU {self.device_id}] Worker pool shut down "
            f"(processed {self.total_tasks_processed} tasks, "
            f"restarted {self.total_workers_restarted} workers in "
            f"{time.time() - self.pool_start_time:.1f}s)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取 pool 统计信息"""
        return {
            "device_id": self.device_id,
            "pool_size": self.pool_size,
            "workers_alive": len([w for w in self.workers if w.is_alive()]),
            "idle_workers": len(self.idle_workers),
            "busy_workers": len(self.busy_workers),
            "total_tasks_processed": self.total_tasks_processed,
            "total_workers_restarted": self.total_workers_restarted,
            "uptime": time.time() - self.pool_start_time,
            "workers": [w.get_stats() for w in self.workers],
        }


# ============================================================================
# Worker Loop (在 subprocess 中运行)
# ============================================================================


def _persistent_worker_loop(worker_id: str, device_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """
    持久化 worker 的主循环

    这个函数在 subprocess 中运行：
    1. 启动时一次性初始化 torch 和 CUDA
    2. 循环处理任务
    3. **遇到 CUDA error 立即退出**
    4. 每次任务后清理 GPU 内存
    """
    init_start = time.time()

    try:
        # ====================================================================
        # 第1步：一次性初始化（只执行一次！）
        # ====================================================================

        # Import 依赖
        import torch
        import torch.cuda
        from kernelgym.backend import get_backend
        from kernelgym.toolkit import get_toolkit

        # 初始化 CUDA
        torch.cuda.init()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)

        # 预热（确保 CUDA 完全初始化）
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()

        toolkit_cache: Dict[str, Any] = {}
        backend_cache: Dict[str, Any] = {}

        init_time = time.time() - init_start

        # 通知主进程：初始化成功
        result_queue.put({"status": "READY", "init_time": init_time, "device": str(device)})

        # 日志
        print(f"[{worker_id}] Initialized successfully (device={device}, init_time={init_time:.2f}s)", file=sys.stderr)

        # ====================================================================
        # 第2步：任务处理循环
        # ====================================================================

        tasks_processed = 0

        while True:
            try:
                # 获取任务
                task_data = task_queue.get()

                # 检查是否是 shutdown 命令
                if isinstance(task_data, dict) and task_data.get("command") in ("SHUTDOWN", "GRACEFUL_SHUTDOWN"):
                    cmd = task_data.get("command")
                    print(f"[{worker_id}] Received {cmd} command", file=sys.stderr)
                    if cmd == "GRACEFUL_SHUTDOWN":
                        # Perform thorough GPU cleanup before exiting so
                        # that the CUDA context is released cleanly without
                        # needing SIGKILL.
                        print(
                            f"[{worker_id}] Graceful shutdown: cleaning up GPU...",
                            file=sys.stderr,
                        )
                        try:
                            _aggressive_gpu_cleanup(device_id)
                        except Exception as e:
                            print(
                                f"[{worker_id}] GPU cleanup during graceful shutdown failed: {e}",
                                file=sys.stderr,
                            )
                    break

                # 执行任务
                result = _execute_task_in_worker(
                    task_data,
                    device,
                    toolkit_cache,
                    backend_cache,
                    get_toolkit,
                    get_backend,
                )

                # 返回结果
                result_queue.put(result)

                tasks_processed += 1

                # GPU 内存清理（每次任务后）
                try:
                    # 强制清理显存
                    _aggressive_gpu_cleanup(device_id)
                except Exception as cleanup_error:
                    print(f"[{worker_id}] GPU cleanup warning: {cleanup_error}", file=sys.stderr)

            except Exception as task_error:
                # 任务执行失败
                error_type = type(task_error).__name__
                error_message = str(task_error)

                # **关键：检查是否是 CUDA error**
                is_cuda_error = (
                    "CUDA" in error_type
                    or "CUDA" in error_message
                    or "cuda" in error_message.lower()
                    or error_type in ["RuntimeError", "CudaError"]
                )
                is_profiler_error = "PROFILER_NO_CUDA_EVENTS" in error_message

                if is_cuda_error or is_profiler_error:
                    # CUDA error / profiler dropout！准备退出
                    print(
                        f"[{worker_id}] CUDA/profiler error detected! Worker will exit. "
                        f"Error: {error_type}: {error_message}",
                        file=sys.stderr,
                    )

                    # 返回错误结果，并标记 worker 将退出
                    result_queue.put(
                        {
                            "success": False,
                            "error_type": error_type,
                            "error_message": error_message,
                            "traceback": traceback.format_exc(),
                            "worker_exiting": True,  # 关键标记！
                            "cuda_error": is_cuda_error,
                            "profiling_error": is_profiler_error,
                        }
                    )

                    # **关键：CUDA error 退出前强制清理显存**
                    print(f"[{worker_id}] Performing aggressive GPU cleanup before exit...", file=sys.stderr)
                    try:
                        _aggressive_gpu_cleanup(device_id)
                        print(f"[{worker_id}] GPU cleanup completed", file=sys.stderr)
                    except Exception as cleanup_err:
                        print(
                            f"[{worker_id}] GPU cleanup failed (expected after CUDA error): {cleanup_err}",
                            file=sys.stderr,
                        )

                    # 尝试最终同步（可能失败，但尝试一下）
                    try:
                        torch.cuda.synchronize()
                        print(f"[{worker_id}] Final CUDA sync before exit", file=sys.stderr)
                    except:
                        pass

                    # 立即退出循环
                    break

                else:
                    # 非 CUDA error，返回错误但继续运行
                    print(f"[{worker_id}] Task error (non-CUDA): {error_type}: {error_message}", file=sys.stderr)

                    result_queue.put(
                        {
                            "success": False,
                            "error_type": error_type,
                            "error_message": error_message,
                            "traceback": traceback.format_exc(),
                            "worker_exiting": False,
                            "cuda_error": False,
                        }
                    )

        # 正常退出 - 清理显存
        print(f"[{worker_id}] Worker exiting normally (processed {tasks_processed} tasks)", file=sys.stderr)

        # **关键：正常退出时也要清理显存**
        print(f"[{worker_id}] Performing final GPU cleanup...", file=sys.stderr)
        try:
            _aggressive_gpu_cleanup(device_id)
            print(f"[{worker_id}] Final GPU cleanup completed", file=sys.stderr)
        except Exception as cleanup_err:
            print(f"[{worker_id}] Final GPU cleanup failed: {cleanup_err}", file=sys.stderr)

        # **额外：尝试重置 CUDA 上下文（确保进程退出时完全释放）**
        try:
            import torch

            # 这会在进程退出时自动调用 CUDA cleanup
            # 但我们显式调用以确保
            torch.cuda.synchronize()
            print(f"[{worker_id}] CUDA context synchronized before exit", file=sys.stderr)
        except Exception as cuda_cleanup_err:
            print(f"[{worker_id}] CUDA synchronize failed: {cuda_cleanup_err}", file=sys.stderr)

    except Exception as init_error:
        # 初始化失败
        print(f"[{worker_id}] Initialization failed: {init_error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        result_queue.put({"status": "INIT_FAILED", "error": str(init_error), "traceback": traceback.format_exc()})


def _execute_task_in_worker(
    task_data: Dict[str, Any],
    device: Any,  # torch.device
    toolkit_cache: Dict[str, Any],
    backend_cache: Dict[str, Any],
    get_toolkit: Any,
    get_backend: Any,
) -> Dict[str, Any]:
    """
    在 worker 中执行单个任务

    Args:
        task_data: 任务数据字典
        device: torch.device
        toolkit: KernelBench integration 模块

    Returns:
        结果字典
    """

    def _has_no_cuda_events(result_obj: Any) -> bool:
        """Detect profiler dropouts where no CUDA events were captured."""
        try:
            metadata = None
            if isinstance(result_obj, dict):
                metadata = result_obj.get("metadata")
            else:
                metadata = getattr(result_obj, "metadata", None)
            if not isinstance(metadata, dict):
                return False
            profiling = metadata.get("profiling")
            if not isinstance(profiling, dict):
                return False
            return profiling.get("profiling_warning") == "no_cuda_events"
        except Exception:
            return False

    try:
        toolkit_name = task_data.get("toolkit")
        backend_adapter = task_data.get("backend_adapter")
        if not toolkit_name:
            raise ValueError("Task payload missing required 'toolkit'")
        if not backend_adapter:
            raise ValueError("Task payload missing required 'backend_adapter'")

        if toolkit_name not in toolkit_cache:
            toolkit_cache[toolkit_name] = get_toolkit(toolkit_name)
        if backend_adapter not in backend_cache:
            backend_cache[backend_adapter] = get_backend(backend_adapter)

        task_data["device"] = str(device)
        stage_metadata_path = task_data.get("_stage_metadata_path")
        previous_stage_metadata_path = os.environ.get(_STAGE_METADATA_PATH_ENV)
        if stage_metadata_path:
            os.environ[_STAGE_METADATA_PATH_ENV] = str(stage_metadata_path)

        toolkit = toolkit_cache[toolkit_name]
        backend = backend_cache[backend_adapter]
        try:
            result = toolkit.evaluate(task_data, backend=backend)
        finally:
            if previous_stage_metadata_path is None:
                os.environ.pop(_STAGE_METADATA_PATH_ENV, None)
            else:
                os.environ[_STAGE_METADATA_PATH_ENV] = previous_stage_metadata_path

        if isinstance(result, dict):
            status = result.get("status")
            error_msg = result.get("error_message")
        else:
            status = getattr(result, "status", None)
            error_msg = getattr(result, "error_message", None)

        if status == "failed" and error_msg:
            if (
                "CUDA" in error_msg
                or "cuda" in error_msg.lower()
                or "illegal memory access" in error_msg.lower()
                or "device-side assert" in error_msg.lower()
            ):
                raise RuntimeError(f"CUDA error detected: {error_msg}")

        if _has_no_cuda_events(result):
            raise RuntimeError("PROFILER_NO_CUDA_EVENTS")

        return {
            "success": True,
            "result": result,
            "worker_exiting": False,
        }

    except Exception:
        # 这里的异常会被上层捕获并判断是否是 CUDA error
        raise
