"""
GPU Worker for KernelGym - with Worker Pool Architecture.

Modified: 2025-10-30
Version: v0.3.3-rc - Worker Pool for performance optimization with CUDA error isolation
"""

import asyncio
import json
import logging
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import redis.asyncio as redis
import aiohttp
import torch

from kernelgym.config import settings

KEY_PREFIX = settings.redis_key_prefix
from kernelgym.config import setup_logging
from kernelgym.server.task_manager import FrozenTaskClaimError, StaleTaskClaimError, TaskManager
from kernelgym.utils.error_classifier import classify_error
from kernelgym.utils.gpu_quarantine import (
    UNLATCHED_NOTIFICATION_PROVENANCE,
    gpu_quarantine_generation,
    read_gpu_quarantine,
    update_gpu_quarantine_notification,
    write_gpu_quarantine,
)
from kernelgym.utils.page_user_notifier import send_gpu_quarantine_page, send_gpu_worker_exclusion_page
from kernelgym.utils.task_status import task_status_from_result_payload
from aiohttp import ClientConnectorError, ClientResponseError

# Import Worker Pool for persistent subprocess workers
from kernelgym.worker.subprocess_pool import (
    GPUProbeFailedError,
    GPUQuarantinedError,
    PoolShutdownContainmentError,
    SubprocessWorkerPool,
    TaskCancelledError,
    UnsafeGPUContainmentError,
    _complete_despite_cancellation,
)

logger = logging.getLogger("kernelgym.worker")

_QUARANTINE_PAGE_MAX_ATTEMPTS_PER_PROCESS = 2
_QUARANTINE_PAGE_RETRY_BACKOFF_SECONDS = 60.0


class _TerminalTaskWriteError(RuntimeError):
    """A terminal result could not be durably committed to Redis."""


class GPUWorker:
    """GPU worker for processing evaluation tasks."""

    def __init__(self, worker_id: str, device: str, redis_client: redis.Redis):
        self.worker_id = worker_id
        self.device = device
        self.redis = redis_client
        self.task_manager = TaskManager(redis_client)
        self.worker_instance_id = uuid.uuid4().hex
        self.task_manager.worker_instance_id = self.worker_instance_id
        self.running = False
        self.current_task: Optional[str] = None
        self._processing_active = False
        self.tasks_processed = 0
        self.last_heartbeat = None
        self.health_state = "initializing"
        self.quarantine_reason = ""
        self.quarantine_physical_scope = True
        # Worker-process exclusion and physical-GPU quarantine are distinct
        # notification events.  A later physical escalation must not be hidden
        # by an earlier successful worker-only page.
        self._quarantine_page_attempts: Dict[str, int] = {}
        self._quarantine_page_sent: set[str] = set()
        self._quarantine_page_retry_not_before: Dict[str, float] = {}
        self._quarantine_page_lock = asyncio.Lock()
        self._stopping = False
        # Once shutdown starts forcefully containing an in-flight task, only
        # stop() may finalize that claim.  The execution coroutine can be
        # unwound by the pool reap and must not publish an ordinary failure.
        self._shutdown_retained_task_id: Optional[str] = None
        self._shutdown_containment_safe: Optional[bool] = None

        # Worker statistics
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_task_time": 0.0,
        }

        # CUDA error tracking (for monitoring, worker pool handles auto-restart)
        self.cuda_error_count = 0
        self.max_cuda_errors_for_alert = 50  # Alert threshold (worker pool auto-restarts on CUDA errors)
        self.cuda_errors_window = []  # Track recent CUDA errors with timestamps
        self.last_cuda_error_time = None
        self.shutdown_due_to_error = False

        # Main process health tracking
        self.main_process_error_count = 0
        self.max_main_process_errors = 3  # If main process itself has errors, we need restart
        # Per-task timeout (seconds).
        # Set to 35s to account for any overhead
        # This prevents false positives for tasks completing at ~30.00x seconds
        self.per_task_timeout_sec = 35

        # Worker Pool (NEW!)
        # Each GPU worker maintains a pool of subprocess workers
        # Pool size and per-worker task limit are configurable to enforce isolation.
        self.worker_pool: Optional[SubprocessWorkerPool] = None
        self.pool_size = getattr(settings, "worker_pool_size", 2)
        self.max_tasks_per_worker = getattr(settings, "max_tasks_per_worker", 1)

        # GPU device setup (主进程不使用CUDA，只存储device_id)
        # 从"cuda:N"提取device_id
        if device.startswith("cuda:"):
            self.device_id = int(device.split(":")[1])
        else:
            raise ValueError(f"Invalid device format: {device}, expected 'cuda:N'")

        # GPU信息缓存（用于_get_worker_info）
        self.gpu_info = {"name": "Unknown", "total_memory": 0}

        # API server URL - handle IPv6 addresses properly
        if ":" in settings.api_host and not settings.api_host.startswith("["):
            # IPv6 address needs brackets in URL
            self.api_url = f"http://[{settings.api_host}]:{settings.api_port}"
        else:
            self.api_url = f"http://{settings.api_host}:{settings.api_port}"

        # HTTP session for API calls
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.node_id: Optional[str] = None
        self._stop_task: Optional[asyncio.Task[None]] = None

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}")
        # Stop consuming new tasks ASAP and begin shutdown
        self.running = False
        asyncio.create_task(self.stop())

    async def start(self):
        """Start the worker."""
        try:
            self.running = True
            logger.info(f"Starting GPU worker {self.worker_id} on device {self.device}")

            # Write initial heartbeat immediately to prevent monitor from detecting missing key
            # This happens before any potentially slow operations (API registration, GPU init)
            try:
                worker_key = f"{KEY_PREFIX}:worker:{self.worker_id}"
                await self.redis.hset(
                    worker_key,
                    mapping={
                        "online": "initializing",
                        "last_heartbeat": datetime.now().isoformat(),
                        "device": self.device,
                        "current_task": "",
                        "tasks_processed": "0",
                        "worker_instance_id": self.worker_instance_id,
                        "health_state": "initializing",
                        "accepting_tasks": "false",
                    },
                )
                await self.redis.expire(worker_key, 120)
                logger.info(f"Worker {self.worker_id} wrote initial heartbeat during startup")
            except Exception as e:
                logger.warning(f"Failed to write initial heartbeat: {e}")

            # Create HTTP session
            self.http_session = aiohttp.ClientSession()

            # Obtain/allocate node_id from server if not configured
            import socket

            hostname = socket.gethostname()
            if not settings.node_id:
                try:
                    url = f"{self.api_url}/node/allocate"
                    async with self.http_session.post(url, params={"hostname": hostname}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.node_id = data.get("node_id")
                            logger.info(f"Obtained server-assigned node_id={self.node_id} for hostname={hostname}")
                        else:
                            logger.warning(f"Failed to allocate node_id from server: {resp.status}")
                except Exception as e:
                    logger.warning(f"Allocate node_id error: {e}")
            else:
                self.node_id = settings.node_id

            # Register with API server
            registered = await self._register_with_api()
            if not registered:
                logger.error(f"Failed to register worker {self.worker_id}")
                raise RuntimeError("Worker registration failed")

            import socket

            existing_quarantine = await read_gpu_quarantine(
                self.redis,
                self.worker_id,
                device=self.device,
                hostname=socket.gethostname(),
            )
            if existing_quarantine:
                self.health_state = "quarantined"
                self.quarantine_reason = existing_quarantine.get("reason", "persistent GPU quarantine")
                self.quarantine_physical_scope = existing_quarantine.get("scope", "physical_gpu") == "physical_gpu"
                await self._ensure_quarantine_notification(existing_quarantine)
                logger.error(
                    f"Worker {self.worker_id} remains QUARANTINED; CUDA initialization and task dequeue "
                    f"are disabled until the Redis latch is manually cleared: {self.quarantine_reason}"
                )
            else:
                await self._initialize_worker_pool()
                if self.worker_pool is not None and self.health_state == "healthy":
                    # This is the only release path for claims frozen by an
                    # unsafe predecessor: the durable quarantine is absent and
                    # a fresh CUDA context has initialized successfully.  A
                    # quarantined or failed replacement leaves every retained
                    # claim token/inflight entry untouched.
                    await self.task_manager.recover_gpu_inflight(
                        self.worker_id,
                        release_frozen_claims=True,
                    )

            # Send initial heartbeat immediately after registration
            await self._update_worker_status(online=True)
            logger.info(f"Worker {self.worker_id} sent initial heartbeat")

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Start main processing loop
            processing_task = asyncio.create_task(self._processing_loop())

            # Wait for either task to complete
            await asyncio.gather(heartbeat_task, processing_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in worker {self.worker_id}: {e}")
            raise
        finally:
            try:
                await self.stop()
            finally:
                if self.http_session and not self.http_session.closed:
                    await self.http_session.close()
                self.http_session = None

    async def stop(self):
        """Stop once, completing CUDA containment before cancellation escapes."""

        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_once())
        cancellation_requested = False
        while not self._stop_task.done():
            try:
                await asyncio.shield(self._stop_task)
            except asyncio.CancelledError:
                cancellation_requested = True
        self._stop_task.result()
        if cancellation_requested:
            raise asyncio.CancelledError

    async def _stop_once(self) -> None:
        """Perform the non-cancellable, idempotent worker shutdown body."""

        self._stopping = True

        # Ensure loops observe shutdown
        self.running = False

        logger.info(f"Stopping GPU worker {self.worker_id}")

        # Close scheduler admission before draining, but keep the worker
        # visibly online until every CUDA child has been proven reaped.  The
        # manual-clear command refuses online workers, so publishing offline
        # earlier would create a window in which an operator could clear the
        # physical latch while an unsafe CUDA context still existed.
        if self.health_state != "quarantined":
            self.health_state = "stopping"
        await self._update_worker_status(online=True)

        # Drain: give an in-flight task a chance to finish before failing it.
        # After self.running=False the processing loop still completes the task
        # it already popped (posting its result and clearing current_task), so
        # a drained shutdown produces zero spurious "Worker shutdown" failures.
        # Wait on the LOOP, not just current_task: a task can already be popped
        # from its queue before current_task is set. Error/eviction shutdowns
        # (shutdown_due_to_error) stay immediate.
        drain_sec = max(0, int(getattr(settings, "worker_shutdown_drain_sec", 120)))

        def _draining() -> bool:
            return bool(self.current_task or getattr(self, "_processing_active", False))

        if drain_sec and not self.shutdown_due_to_error and _draining():
            logger.info(
                f"Worker {self.worker_id} draining before shutdown: "
                f"waiting up to {drain_sec}s (task: {self.current_task or 'being dequeued'})"
            )
            deadline = time.monotonic() + drain_sec
            while _draining() and time.monotonic() < deadline:
                await asyncio.sleep(0.5)

        # From this point on, a pool reap may unwind the task coroutine.  Keep
        # its claim under stop()'s authority so that cancellation/pipe errors
        # caused by containment cannot publish an ordinary retryable failure.
        retained_task_id = self.current_task
        if retained_task_id:
            self._shutdown_retained_task_id = retained_task_id
            logger.warning(
                f"Worker {self.worker_id} drain expired; containing CUDA before failing task {retained_task_id}"
            )
            try:
                await self.task_manager.freeze_task_claim(
                    retained_task_id,
                    "worker shutdown is containing CUDA; automatic recovery is unsafe until a fresh safe startup",
                )
            except Exception as exc:
                # The physical quarantine below is the second durable fence.
                # Keep stopping/containing even if this task-scoped write fails.
                logger.critical(
                    "Failed to freeze recovery for shutdown task %s; relying on the GPU quarantine latch: %s",
                    retained_task_id,
                    exc,
                )

        # No pool means there is no child CUDA context to contain.  Otherwise
        # safety is established only by an affirmative pool shutdown result.
        shutdown_safe = self.worker_pool is None
        if self.worker_pool:
            pool = self.worker_pool
            try:
                logger.info(f"Shutting down worker pool for {self.worker_id}...")
                shutdown_safe = await pool.shutdown(timeout=30)
                if shutdown_safe:
                    logger.info(f"Worker pool shut down successfully for {self.worker_id}")
                else:
                    unsafe_reason = pool.unsafe_shutdown_reason or "CUDA context reap failed"
                    await self._quarantine_gpu(
                        reason=f"unsafe worker-pool shutdown: {unsafe_reason}",
                        fault_class="unsafe_pool_shutdown",
                        task_id=retained_task_id or "",
                        physical_scope=True,
                        update_status=False,
                    )
            except asyncio.CancelledError:
                # _stop_once normally runs behind shield(), but also fail safe
                # if its pool coroutine reports cancellation itself: a clean
                # reap was not proven, so retain the claim and quarantine.
                shutdown_safe = False
                unsafe_reason = getattr(pool, "unsafe_shutdown_reason", "") or "pool shutdown was cancelled"
                await self._quarantine_gpu(
                    reason=f"unsafe worker-pool shutdown: {unsafe_reason}",
                    fault_class="unsafe_pool_shutdown",
                    task_id=retained_task_id or "",
                    physical_scope=True,
                    update_status=False,
                )
            except Exception as e:
                shutdown_safe = False
                logger.error(f"Error shutting down worker pool: {e}")
                await self._quarantine_gpu(
                    reason=f"worker-pool shutdown raised {type(e).__name__}; safe reap was not proven",
                    fault_class="unsafe_pool_shutdown",
                    task_id=retained_task_id or "",
                    physical_scope=True,
                    update_status=False,
                )
            finally:
                # Never discard the last Python handle to a child whose reap
                # was not proven.  The pool also keeps a process-level orphan
                # registry; retaining this reference permits another explicit
                # containment pass while the parent process is still alive.
                if shutdown_safe:
                    self.worker_pool = None

        self._shutdown_containment_safe = shutdown_safe

        # Publishing "Worker shutdown" also acknowledges the inflight claim,
        # so it is legal only after every CUDA child is known to be gone.  Do
        # not adopt an unknown Redis token: that token may belong to a newer
        # process instance using the same stable worker_id.
        if shutdown_safe and retained_task_id and self.current_task == retained_task_id:
            try:
                await self.task_manager.fail_task(
                    retained_task_id,
                    "Worker shutdown",
                    adopt_current_claim=False,
                    allow_frozen_claim=True,
                )
            except StaleTaskClaimError:
                logger.info(
                    "Shutdown claim for task %s was superseded; a newer owner has already handled it",
                    retained_task_id,
                )
                self.current_task = None
                self._shutdown_retained_task_id = None
            except Exception as exc:
                # A Redis failure must leave the durable inflight entry as the
                # recovery authority.  The replacement worker will recover it
                # before opening its GPU gate.
                logger.critical(
                    "Failed to finalize safely contained task %s during shutdown; retaining claim for recovery: %s",
                    retained_task_id,
                    exc,
                )
            else:
                self.current_task = None
                self._shutdown_retained_task_id = None

        # Only expose the worker as offline after CUDA containment has
        # completed (or a persistent physical quarantine has been recorded).
        if not self.shutdown_due_to_error:
            await self._unregister_from_api()
        await self._update_worker_status(online=False)

        # Close HTTP session
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        self.http_session = None

        # GPU cleanup不再需要（主进程不使用CUDA，worker pool已清理）
        logger.info("GPU cleanup handled by worker pool shutdown")

        # Log final statistics
        logger.info(f"Worker {self.worker_id} processed {self.tasks_processed} tasks")

    async def _initialize_gpu(self):
        """
        验证GPU可用性（不在主进程中初始化CUDA）

        使用nvidia-smi验证GPU，不会触发CUDA初始化。
        GPU信息缓存用于后续的worker info查询。
        """
        try:
            from kernelgym.utils.gpu_diagnostics import GPUDiagnostics

            logger.info(f"Verifying GPU {self.device_id} availability (no CUDA init in main process)")

            # 使用nvidia-smi验证GPU（不初始化CUDA）
            health = GPUDiagnostics.test_gpu_health_nvidia_smi(self.device_id)

            if not health.healthy:
                raise RuntimeError(f"GPU {self.device_id} not healthy: {health.error_message}")

            # 缓存GPU信息
            self.gpu_info = {
                "name": health.device_name or "Unknown",
                "total_memory": int(health.total_memory_gb * 1024**3) if health.total_memory_gb else 0,
            }

            logger.info(f"GPU {self.device_id} verified successfully")
            logger.info(f"GPU Name: {health.device_name}")
            logger.info(f"GPU Memory: {health.total_memory_gb:.1f}GB")
            logger.info("Main process will NOT use CUDA (subprocess isolation enabled)")

        except Exception as e:
            logger.error(f"Failed to verify GPU {self.device_id}: {e}")
            raise

    async def _initialize_worker_pool(self) -> None:
        """Run an advisory precheck, then the authoritative fresh CUDA probe."""

        precheck_error = ""
        try:
            # nvidia-smi is useful diagnostics but cannot distinguish a missing
            # binary/transient NVML issue from a broken device.  Never restart
            # repeatedly on this signal alone; the child READY probe below is
            # authoritative because it performs CUDA init/alloc/synchronize.
            await self._initialize_gpu()
        except Exception as exc:
            precheck_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "GPU %s nvidia-smi precheck failed; continuing to one fresh CUDA context probe: %s",
                self.device,
                precheck_error,
            )

        try:
            logger.info(
                f"Initializing worker pool for {self.worker_id} "
                f"(device={self.device}, pool_size={self.pool_size}, "
                f"max_tasks_per_worker={self.max_tasks_per_worker})"
            )
            self.worker_pool = SubprocessWorkerPool(
                device_id=self.device_id,
                pool_size=self.pool_size,
                worker_prefix=f"{self.worker_id}_pool",
                max_tasks_per_worker=self.max_tasks_per_worker,
            )
        except GPUProbeFailedError as exc:
            details = f"fresh CUDA context initialization failed: {exc}"
            if precheck_error:
                details = f"{details}; nvidia-smi precheck also failed: {precheck_error}"
            logger.error(f"Failed to initialize worker pool for {self.worker_id}: {details}")
            await self._quarantine_gpu(
                reason=details,
                fault_class="initialization_failure",
                physical_scope=True,
            )
            return
        except Exception:
            # Import/queue/fd/PID/bootstrap errors are worker-process failures,
            # not evidence that the physical GPU is bad.
            logger.exception(f"Worker infrastructure failed during startup for {self.worker_id}")
            raise

        self.health_state = "healthy"
        logger.info(
            f"Worker pool initialized successfully for {self.worker_id} "
            f"with {self.pool_size} subprocess workers "
            f"(max {self.max_tasks_per_worker} tasks per worker)"
        )

    async def _quarantine_gpu(
        self,
        *,
        reason: str,
        fault_class: str,
        task_id: str = "",
        physical_scope: bool = True,
        update_status: bool = True,
    ) -> None:
        """Finish the full latch/page transaction before cancellation escapes."""

        operation = asyncio.create_task(
            self._quarantine_gpu_to_completion(
                reason=reason,
                fault_class=fault_class,
                task_id=task_id,
                physical_scope=physical_scope,
                update_status=update_status,
            )
        )
        cancellation_requested = False
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                cancellation_requested = True
        operation.result()
        if cancellation_requested:
            raise asyncio.CancelledError

    async def _quarantine_gpu_to_completion(
        self,
        *,
        reason: str,
        fault_class: str,
        task_id: str,
        physical_scope: bool,
        update_status: bool,
    ) -> None:
        """Persist a non-expiring, manually cleared GPU admission latch."""

        import socket

        hostname = socket.gethostname()
        node_id = self.node_id or settings.node_id or hostname
        self.health_state = "quarantined"
        self.quarantine_reason = reason
        self.quarantine_physical_scope = physical_scope
        record: Optional[Dict[str, str]] = None
        try:
            record = await write_gpu_quarantine(
                self.redis,
                self.worker_id,
                device=self.device,
                reason=reason,
                fault_class=fault_class,
                task_id=task_id,
                node_id=node_id,
                hostname=hostname,
                physical_scope=physical_scope,
            )
        except Exception as exc:
            # Local admission remains closed.  _gpu_admission_allowed retries
            # persistence once Redis is reachable again.
            logger.critical(f"Failed to persist GPU quarantine latch for {self.worker_id}: {exc}")
            try:
                record = await read_gpu_quarantine(
                    self.redis,
                    self.worker_id,
                    device=self.device,
                    hostname=hostname,
                )
            except Exception:
                record = None
        if record is None:
            # Paging must not depend on Redis/durable-latch availability.
            # In-memory attempt tracking prevents a retry storm; a worker
            # restart retries because no durable "sent" marker exists.
            record = {
                "state": "quarantined",
                "scope": "physical_gpu" if physical_scope else "worker_process",
                "worker_id": self.worker_id,
                "device": self.device,
                "reason": reason,
                "fault_class": fault_class,
                "task_id": task_id,
                "node_id": node_id,
                "hostname": hostname,
                "page_user_state": "pending",
                "notification_provenance": UNLATCHED_NOTIFICATION_PROVENANCE,
            }
        record_scope = str(record.get("scope") or "worker_process")
        effective_reason = str(record.get("reason") or reason)
        self.quarantine_physical_scope = record_scope == "physical_gpu"
        self.quarantine_reason = effective_reason
        await self._ensure_quarantine_notification(record)
        if record_scope == "physical_gpu":
            logger.error(
                f"Worker {self.worker_id} GPU is QUARANTINED (manual clear required); "
                f"new tasks will not be dequeued: {effective_reason}"
            )
        else:
            logger.error(
                f"Worker {self.worker_id} is QUARANTINED (physical GPU fault not proven; manual clear required); "
                f"new tasks will not be dequeued: {effective_reason}"
            )
        if update_status:
            await self._update_worker_status(online=True)

    async def _ensure_quarantine_notification(self, record: Dict[str, str]) -> None:
        """Page any CUDA scheduling exclusion with durable dedupe/backoff."""

        scope = str(record.get("scope") or "")
        if scope not in {"physical_gpu", "worker_process"} or not self.device.startswith("cuda:"):
            return
        notification_key = f"{scope}:{gpu_quarantine_generation(record)}"
        unlatched_best_effort = record.get("notification_provenance") == UNLATCHED_NOTIFICATION_PROVENANCE

        async def _restore_confirmed_delivery_marker() -> None:
            """Do not let a repeated worker-scope latch write regress ``sent``."""

            if record.get("page_user_state") == "sent":
                return
            if unlatched_best_effort:
                return
            try:
                await update_gpu_quarantine_notification(
                    self.redis,
                    self.worker_id,
                    device=self.device,
                    hostname=str(record.get("hostname") or ""),
                    scope=scope,
                    expected_generation=gpu_quarantine_generation(record),
                    state="sent",
                )
            except Exception as exc:
                logger.critical(
                    f"Failed to restore confirmed page-user state for quarantined GPU worker {self.worker_id}: {exc}"
                )

        if record.get("page_user_state") == "sent":
            self._quarantine_page_sent.add(notification_key)
            return
        if notification_key in self._quarantine_page_sent:
            await _restore_confirmed_delivery_marker()
            return

        # A replacement process may read a failure written only moments ago.
        # Respect that durable timestamp so repeated process starts cannot turn
        # an MCP outage into a page storm.
        attempts = self._quarantine_page_attempts.get(notification_key, 0)
        if record.get("page_user_state") == "failed" and attempts == 0:
            try:
                failed_at = datetime.fromisoformat(str(record.get("page_user_updated_at") or ""))
                now = datetime.now(tz=failed_at.tzinfo)
                remaining = _QUARANTINE_PAGE_RETRY_BACKOFF_SECONDS - max(
                    0.0,
                    (now - failed_at).total_seconds(),
                )
                if remaining > 0:
                    self._quarantine_page_retry_not_before[notification_key] = max(
                        self._quarantine_page_retry_not_before.get(notification_key, 0.0),
                        time.monotonic() + remaining,
                    )
            except (TypeError, ValueError):
                pass

        if attempts >= _QUARANTINE_PAGE_MAX_ATTEMPTS_PER_PROCESS:
            return
        if time.monotonic() < self._quarantine_page_retry_not_before.get(notification_key, 0.0):
            return

        # _quarantine_gpu(), the admission loop, and heartbeat-related paths can
        # all observe the same latch. Serialize their checks so only one attempt
        # starts and waiters see the outcome/backoff chosen by that attempt.
        async with self._quarantine_page_lock:
            if notification_key in self._quarantine_page_sent:
                await _restore_confirmed_delivery_marker()
                return
            attempts = self._quarantine_page_attempts.get(notification_key, 0)
            if attempts >= _QUARANTINE_PAGE_MAX_ATTEMPTS_PER_PROCESS:
                return
            if time.monotonic() < self._quarantine_page_retry_not_before.get(notification_key, 0.0):
                return
            self._quarantine_page_attempts[notification_key] = attempts + 1

            async def _deliver_and_record() -> None:
                try:
                    if scope == "physical_gpu":
                        outcome = await send_gpu_quarantine_page(record)
                    else:
                        outcome = await send_gpu_worker_exclusion_page(record)
                    success = outcome.success
                    protocol_version = outcome.protocol_version or "unknown"
                    superseded = protocol_version == "superseded"
                    error = "" if success else f"{outcome.error_kind or 'unknown'}: {outcome.error or ''}"
                except Exception as exc:  # pragma: no cover - notifier is specified never to raise
                    success = False
                    superseded = False
                    protocol_version = "unknown"
                    error = f"unexpected_error: {type(exc).__name__}"
                state = "sent" if success else "failed"
                if not superseded and not unlatched_best_effort:
                    try:
                        await update_gpu_quarantine_notification(
                            self.redis,
                            self.worker_id,
                            device=self.device,
                            hostname=str(record.get("hostname") or ""),
                            scope=scope,
                            expected_generation=gpu_quarantine_generation(record),
                            state=state,
                            error=error,
                        )
                    except Exception as exc:
                        logger.critical(
                            f"Failed to persist page-user delivery state for quarantined GPU {self.worker_id}: {exc}"
                        )
                if success:
                    self._quarantine_page_sent.add(notification_key)
                    if superseded:
                        logger.info(
                            f"Skipped superseded worker-exclusion notification for {self.worker_id}; "
                            "a physical GPU quarantine now owns the alert"
                        )
                    else:
                        logger.warning(
                            f"Sent page-user notification for quarantined GPU worker {self.worker_id} "
                            f"(scope={scope}) via MCP {protocol_version}"
                        )
                else:
                    self._quarantine_page_retry_not_before[notification_key] = (
                        time.monotonic() + _QUARANTINE_PAGE_RETRY_BACKOFF_SECONDS
                    )
                    logger.critical(
                        f"Failed to send page-user notification for quarantined GPU {self.worker_id}: {error}"
                    )

            # Once a CUDA worker is removed from scheduling, caller
            # cancellation must not abandon the page half-sent or lose its
            # durable dedupe marker.
            notification_task = asyncio.create_task(_deliver_and_record())
            cancellation_requested = False
            while not notification_task.done():
                try:
                    await asyncio.shield(notification_task)
                except asyncio.CancelledError:
                    cancellation_requested = True
            notification_task.result()
            if cancellation_requested:
                raise asyncio.CancelledError

    async def _gpu_admission_allowed(self) -> bool:
        """Synchronize local pool health with Redis before any queue pop."""

        # Redis heartbeat state can be stale or temporarily unwritable.  The
        # in-process lifecycle is therefore the first and last authority at
        # every admission check.
        if self._stopping or not self.running:
            return False

        try:
            import socket

            quarantine = await read_gpu_quarantine(
                self.redis,
                self.worker_id,
                device=self.device,
                hostname=socket.gethostname(),
            )
        except Exception as exc:
            logger.error(f"Unable to read GPU quarantine latch for {self.worker_id}; failing closed: {exc}")
            return False

        if quarantine:
            self.health_state = "quarantined"
            self.quarantine_reason = quarantine.get("reason", "persistent GPU quarantine")
            self.quarantine_physical_scope = quarantine.get("scope", "physical_gpu") == "physical_gpu"
            await self._ensure_quarantine_notification(quarantine)
            return False

        if self.health_state == "quarantined":
            # Clearing the latch does not mutate a live CUDA process.  A
            # deliberate worker restart is required to create fresh contexts.
            await self._quarantine_gpu(
                reason=self.quarantine_reason or "local GPU quarantine",
                fault_class="local_quarantine",
                physical_scope=self.quarantine_physical_scope,
            )
            return False
        if self.worker_pool is None:
            return False

        pool_health = self.worker_pool.get_health_snapshot()
        self.health_state = str(pool_health["health_state"])
        if self.health_state == "quarantined":
            await self._quarantine_gpu(
                reason=str(pool_health.get("health_reason") or "fresh-context validation failed"),
                fault_class=str(pool_health.get("health_fault_class") or "pool_validation_failure"),
                task_id=str(pool_health.get("health_task_id") or ""),
                physical_scope=str(pool_health.get("health_scope") or "gpu") == "gpu",
            )
            return False
        return bool(pool_health["accepting_tasks"]) and self.running and not self._stopping

    async def _processing_loop(self):
        """Main processing loop."""
        logger.info(f"Worker {self.worker_id} processing loop started")
        # Read by stop()'s drain: current_task alone is not enough, because a
        # task may already be popped from its queue before current_task is set.
        self._processing_active = True
        try:
            while self.running:
                try:
                    # Note: In subprocess isolation architecture, CUDA error count is no longer used
                    # as errors are contained in subprocesses and don't affect the main worker

                    # Gate before RPOP/BRPOP.  This keeps queued tasks untouched
                    # while the physical GPU is being validated or quarantined.
                    if not await self._gpu_admission_allowed():
                        await self._update_worker_status(online=True)
                        await asyncio.sleep(2)
                        continue

                    # Get next task
                    task_data = await self.task_manager.get_next_task(self.worker_id, resources=["gpu"])

                    if task_data:
                        if not await self._gpu_admission_allowed():
                            await self.task_manager.requeue_unstarted_task(
                                task_data,
                                reason="gpu_admission_closed_after_dequeue",
                                release_execution_fence=True,
                            )
                            logger.warning(
                                f"Worker {self.worker_id} requeued task {task_data.get('task_id')} "
                                "because GPU admission closed during dequeue"
                            )
                            continue
                        await self._process_task(task_data)
                    else:
                        # No tasks available. get_next_task 已 BRPOP(1s)，此处仅做极短休眠避免忙等
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error in processing loop for worker {self.worker_id}: {e}")

                    # Distinguish between subprocess errors and main process errors
                    from kernelgym.server.code_retry_manager import CodeRetryManager

                    if CodeRetryManager(self.redis)._is_memory_error(str(e)):
                        # This is likely from a subprocess, no need to restart main worker
                        logger.info(
                            f"[SUBPROCESS-ISOLATION] CUDA error detected in loop for worker {self.worker_id}, but isolated in subprocess"
                        )
                    else:
                        # This is a main process error, track it
                        self.main_process_error_count += 1
                        logger.warning(
                            f"Main process error in worker {self.worker_id}: {self.main_process_error_count}/{self.max_main_process_errors}"
                        )

                        # If too many main process errors, shutdown for restart
                        if self.main_process_error_count >= self.max_main_process_errors:
                            logger.error(
                                f"Worker {self.worker_id} main process has too many errors. Shutting down for restart."
                            )
                            await self.redis.hset(
                                f"{KEY_PREFIX}:worker:{self.worker_id}",
                                mapping={
                                    "cuda_error_shutdown": "true",  # Reuse this flag for any critical shutdown
                                    "shutdown_reason": "main_process_errors",
                                    "shutdown_time": datetime.now().isoformat(),
                                },
                            )
                            self.running = False
                            break

                    await asyncio.sleep(5)  # Sleep longer on error
        finally:
            self._processing_active = False
            logger.info(f"Worker {self.worker_id} processing loop exited")

    async def _process_task(self, task_data: Dict[str, Any]):
        """Process a single task."""
        task_id = task_data["task_id"]
        self.current_task = task_id
        start_time = datetime.now()
        terminal_resolved = False

        try:
            logger.info(f"Worker {self.worker_id} processing task {task_id}")

            await self._process_toolkit_task(task_data, start_time)
            terminal_resolved = True

            # Reset CUDA error count on successful completion
            self.cuda_error_count = 0

            # Clear retry history if this was a retry
            if "_retry" in task_id:
                # Extract original task ID
                original_task_id = task_id.rsplit("_retry", 1)[0]
                await self.task_manager.retry_manager.clear_retry_history(original_task_id)

        except StaleTaskClaimError:
            # The terminal Lua transaction rejected this process's token.  A
            # cancellation or newer worker instance now owns the outcome; do
            # not translate fencing into a second ordinary task failure.
            terminal_resolved = True
            logger.info(
                "Task %s completion was superseded; a newer owner has already handled it",
                task_id,
            )

        except UnsafeGPUContainmentError as containment_error:
            await self._retain_claim_for_unsafe_containment(task_id, containment_error)

        except FrozenTaskClaimError:
            # A concurrent shutdown/unsafe-containment path atomically
            # upgraded the normal execution fence before this terminal write.
            # Preserve the exact token; only that containment owner may ACK.
            self.running = False
            self._stopping = True
            self._shutdown_retained_task_id = task_id
            logger.critical(
                "Task %s terminal commit was blocked by its containment fence; retaining claim",
                task_id,
            )

        except PoolShutdownContainmentError:
            # stop() froze this exact execution claim before it closed the
            # pool. It alone may publish a terminal shutdown outcome after the
            # shared pool-shutdown proof succeeds.
            self.running = False
            self._stopping = True
            self._shutdown_retained_task_id = task_id
            logger.warning(
                "Task %s transferred terminal ownership to worker shutdown containment",
                task_id,
            )

        except GPUQuarantinedError:
            # The pool gate can close in the narrow interval after the outer
            # post-pop check but before checkout.  No subprocess received this
            # task, so restore it to pending instead of recording a failure.
            if self._shutdown_retained_task_id == task_id:
                logger.warning(
                    "Task %s hit the GPU gate during shutdown containment; retaining its frozen claim",
                    task_id,
                )
            else:
                logger.warning(f"Worker {self.worker_id} requeueing unstarted task {task_id}: GPU gate closed")
                try:
                    await self.task_manager.requeue_unstarted_task(
                        task_data,
                        reason="gpu_admission_closed_before_execution",
                        release_execution_fence=True,
                    )
                except StaleTaskClaimError:
                    terminal_resolved = True
                    logger.info(
                        "Unstarted task %s was superseded; a newer owner has already handled it",
                        task_id,
                    )
                except Exception as commit_err:
                    self._retain_claim_after_terminal_write_failure(task_id, commit_err)
                else:
                    terminal_resolved = True

        except TaskCancelledError:
            # Task was cancelled mid-flight: the CUDA subprocess has been killed.
            # Record a terminal cancelled result so any waiter (e.g. the workflow
            # controller blocked in scheduler.wait on this sub-task) returns
            # promptly instead of hanging until the task timeout.
            if self._shutdown_retained_task_id == task_id:
                logger.warning(
                    "Task %s was interrupted by shutdown containment; retaining its claim until reap is proven safe",
                    task_id,
                )
            else:
                from kernelgym.common import ErrorCode

                logger.info(f"Worker {self.worker_id} task {task_id} cancelled; recording cancelled result")
                cancelled_result = self._build_failed_result(
                    task_data,
                    "Task cancelled",
                    ErrorCode.SYSTEM_ERROR.value,
                )
                try:
                    await self.task_manager.complete_task(task_id, cancelled_result)
                except StaleTaskClaimError:
                    terminal_resolved = True
                    logger.info(
                        "Cancelled task %s was superseded; a newer owner has already handled it",
                        task_id,
                    )
                except Exception as commit_err:
                    self._retain_claim_after_terminal_write_failure(task_id, commit_err)
                else:
                    terminal_resolved = True
                    self.stats["tasks_failed"] += 1

        except _TerminalTaskWriteError as e:
            self._retain_claim_after_terminal_write_failure(task_id, e)

        except Exception as e:
            if terminal_resolved:
                # Ancillary bookkeeping (for example retry-history cleanup)
                # failed after the fenced terminal commit already succeeded.
                logger.warning(f"Post-completion bookkeeping failed for task {task_id}: {e}")
            elif self._shutdown_retained_task_id == task_id:
                logger.warning(
                    "Task %s unwound during shutdown containment (%s); ordinary failure publication is suppressed",
                    task_id,
                    e,
                )
            else:
                # Task failed
                error_message = f"Task processing failed: {str(e)}"
                logger.error(f"Worker {self.worker_id} failed task {task_id}: {error_message}")

                # Track CUDA errors for monitoring, but don't auto-restart in subprocess isolation mode
                from kernelgym.server.code_retry_manager import CodeRetryManager

                if CodeRetryManager(self.redis)._is_memory_error(str(e)):
                    # Try to print code content from task_data for debugging
                    try:
                        if task_data.get("reference_code"):
                            logger.error(
                                f"[MEMORY-ERROR] Task {task_id} reference_code below:\n{task_data['reference_code']}"
                            )
                        if task_data.get("kernel_code"):
                            logger.error(
                                f"[MEMORY-ERROR] Task {task_id} kernel_code below:\n{task_data['kernel_code']}"
                            )
                    except Exception:
                        pass

                    # Track CUDA errors for monitoring
                    self._track_cuda_error()
                    logger.info(
                        f"[SUBPROCESS-ISOLATION] CUDA error contained in subprocess for task {task_id}, worker continues normally"
                    )

                error_code = classify_error(str(e), "runtime")
                failed_result = self._build_failed_result(task_data, error_message, error_code)
                try:
                    await self.task_manager.complete_task(task_id, failed_result)
                except StaleTaskClaimError:
                    terminal_resolved = True
                    logger.info(
                        "Failed task %s was superseded; a newer owner has already handled it",
                        task_id,
                    )
                except Exception as commit_err:
                    self._retain_claim_after_terminal_write_failure(task_id, commit_err)
                else:
                    terminal_resolved = True
                    self.stats["tasks_failed"] += 1

        finally:
            # complete_task/fail_task atomically publish terminal state + result
            # and acknowledge the Redis inflight claim.  Requeue does the same
            # conditionally.  Never ACK here: either operation may have failed,
            # in which case the claim is the only crash-recovery authority.
            # GPU清理由subprocess自动处理
            if terminal_resolved:
                if self.current_task == task_id:
                    self.current_task = None
                if self._shutdown_retained_task_id == task_id:
                    self._shutdown_retained_task_id = None
            self.tasks_processed += 1

    async def _retain_claim_for_unsafe_containment(
        self,
        task_id: str,
        error: UnsafeGPUContainmentError,
    ) -> None:
        """Persist physical quarantine and freeze this exact inflight attempt."""

        reason = str(error) or "CUDA context reap could not be proven"
        # Close every local admission path before the first external await.  A
        # later stop() retry owns this retained task and may finalize it only if
        # the full pool eventually reports a safe reap.
        self.running = False
        self._stopping = True
        self.shutdown_due_to_error = True
        self._shutdown_retained_task_id = task_id

        async def _freeze_and_persist() -> None:
            try:
                frozen = await self.task_manager.freeze_task_claim(
                    task_id,
                    "CUDA context reap is unproven; automatic recovery and force-refresh are unsafe",
                )
            except Exception as freeze_error:
                frozen = False
                logger.critical(
                    "Failed to freeze exact claim for unsafe CUDA task %s; retaining local claim and stopping: %s",
                    task_id,
                    freeze_error,
                )
            if not frozen:
                logger.critical(
                    "Unsafe CUDA task %s was not durably frozen; worker remains stopped and must not ACK it",
                    task_id,
                )

            # Paging may block on an external endpoint for tens of seconds.
            # Freeze the attempt token first so no terminal writer or reclaim
            # path can race that notification latency.
            try:
                await self._quarantine_gpu(
                    reason=reason,
                    fault_class="pre_fault_reap_failure",
                    task_id=task_id,
                    physical_scope=True,
                )
            except Exception as quarantine_error:
                logger.critical(
                    "Failed to persist/page unsafe GPU containment for task %s: %s",
                    task_id,
                    quarantine_error,
                )

        containment_task = asyncio.create_task(_freeze_and_persist())
        cancellation_requested = False
        while not containment_task.done():
            try:
                await asyncio.shield(containment_task)
            except asyncio.CancelledError:
                cancellation_requested = True
        containment_task.result()
        if cancellation_requested:
            raise asyncio.CancelledError

    def _retain_claim_after_terminal_write_failure(self, task_id: str, error: Exception) -> None:
        """Stop admission and preserve a claim whose terminal Redis write failed."""

        self.shutdown_due_to_error = True
        self.running = False
        logger.critical(
            "Terminal Redis write failed for task %s; stopping worker and retaining inflight claim: %s",
            task_id,
            error,
        )

    async def _process_toolkit_task(self, task_data: Dict[str, Any], start_time: datetime):
        """Process task via toolkit/backend abstractions."""
        task_id = task_data["task_id"]
        timing_start = time.time()

        task_data["device"] = self.device
        if "toolkit" not in task_data:
            raise ValueError("Task payload missing required 'toolkit'")
        if "backend_adapter" not in task_data:
            raise ValueError("Task payload missing required 'backend_adapter'")

        run_toolkit_start = time.time()
        result_dict = await self._run_toolkit_task(task_data)
        run_toolkit_s = time.time() - run_toolkit_start
        if self._shutdown_retained_task_id == task_id:
            raise PoolShutdownContainmentError(f"worker shutdown retained task {task_id} before terminal publication")

        status = result_dict.get("status")
        error_message = result_dict.get("error_message") or "Task failed"
        error_code = result_dict.get("error_code")

        if status != "completed":
            if error_code is None:
                error_code = classify_error(error_message, "runtime")
                result_dict["error_code"] = error_code
            result_dict["status"] = task_status_from_result_payload(result_dict).value
            result_dict["error_message"] = error_message

        metadata = result_dict.setdefault("metadata", {})
        metadata["wg_run_toolkit_s"] = run_toolkit_s

        complete_task_start = time.time()
        complete_task_start_mono_ns = time.monotonic_ns()
        try:
            await self.task_manager.complete_task(task_id, result_dict)
        except (FrozenTaskClaimError, StaleTaskClaimError):
            raise
        except Exception as exc:
            raise _TerminalTaskWriteError(f"task {task_id}: {exc}") from exc
        complete_task_s = time.time() - complete_task_start
        metadata["wg_complete_task_s"] = complete_task_s
        metadata["wg_total_s"] = time.time() - timing_start

        tm_enter_mono_ns = metadata.get("tm_enter_monotonic_ns")
        tm_exit_mono_ns = metadata.get("tm_exit_monotonic_ns")
        if isinstance(tm_enter_mono_ns, int) and isinstance(tm_exit_mono_ns, int):
            metadata["wg_before_tm_enter_s"] = max(0.0, (tm_enter_mono_ns - complete_task_start_mono_ns) / 1e9)
            metadata["wg_after_tm_exit_s"] = max(0.0, (time.monotonic_ns() - tm_exit_mono_ns) / 1e9)

        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_task_stats(processing_time, status == "completed")

        logger.info(
            f"[WorkerTiming] worker={self.worker_id} task={task_id} status={result_dict.get('status')} "
            f"run_toolkit_s={run_toolkit_s:.2f} complete_task_s={complete_task_s:.2f} "
            f"total_s={metadata['wg_total_s']:.2f}"
        )
        if "wg_before_tm_enter_s" in metadata and "wg_after_tm_exit_s" in metadata:
            logger.info(
                f"[WorkerCompleteBreakdown] worker={self.worker_id} task={task_id} "
                f"before_tm_enter_s={metadata['wg_before_tm_enter_s']:.4f} "
                f"tm_complete_task_s={metadata.get('tm_complete_task_s', -1.0):.4f} "
                f"after_tm_exit_s={metadata['wg_after_tm_exit_s']:.4f}"
            )
        logger.info(f"Worker {self.worker_id} completed task {task_id} in {processing_time:.2f}s")

    def _build_failed_result(
        self,
        task_data: Dict[str, Any],
        error_message: str,
        error_code: Any,
    ) -> Dict[str, Any]:
        from kernelgym.schema import (
            EvaluationResult,
            KernelEvaluationResult,
            ReferenceTimingResult,
        )

        task_id = task_data.get("task_id", "unknown")
        base_task_id = task_data.get("base_task_id", task_id)
        task_type = task_data.get("task_type", "evaluation")

        metadata = {"error": error_message}
        stage_metadata = self._load_stage_metadata(task_data)
        metadata.update(stage_metadata)
        # A failure that happens after compile+load (timeout or crash during
        # execution) must still report compiled=True; otherwise a post-compile
        # timeout/crash is indistinguishable from a genuine compilation error.
        compiled_before_failure = self._compiled_from_stage_metadata(stage_metadata)
        result_status = task_status_from_result_payload({"status": "failed", "error_code": error_code}).value

        if task_type == "reference_timing":
            result = ReferenceTimingResult(
                task_id=task_id,
                base_task_id=base_task_id,
                reference_runtime=-1.0,
                metadata=metadata,
                status=result_status,
                error_message=error_message,
                error_code=error_code,
            )
            return result.to_dict()

        if task_type == "kernel_evaluation":
            result = KernelEvaluationResult(
                task_id=task_id,
                base_task_id=base_task_id,
                compiled=compiled_before_failure,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata=metadata,
                status=result_status,
                error_message=error_message,
                error_code=error_code,
            )
            return result.to_dict()

        result = EvaluationResult(
            task_id=task_id,
            compiled=compiled_before_failure,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=-1.0,
            kernel_runtime=-1.0,
            speedup=0.0,
            metadata=metadata,
            status=result_status,
            error_message=error_message,
            error_code=error_code,
        )
        return result.to_dict()

    @staticmethod
    def _compiled_from_stage_metadata(stage_metadata: Dict[str, Any]) -> bool:
        """Infer whether compile+load already finished before a failure.

        A task that times out (or whose subprocess crashes) during execution has
        already passed the ``kernel.compile_and_load`` stage, so reporting
        ``compiled=False`` for it is wrong. The eval pipeline records completed
        stages in ``kg_stage_completed_s`` and the active stage in
        ``kg_stage_current``; use those to recover the real compiled state.
        """
        completed = stage_metadata.get("kg_stage_completed_s")
        if isinstance(completed, dict) and any(
            str(stage).endswith("compile_and_load") or str(stage).endswith("compile_only") for stage in completed
        ):
            return True
        # Fall back to the current/last stage being something that only runs
        # after compile+load (custom model build, correctness, perf, ...).
        post_compile_markers = ("build_custom_model", "correctness", "triton_detect", "performance")
        for key in ("kg_stage_last_completed", "kg_stage_current", "kg_stage_current_prefix"):
            value = stage_metadata.get(key)
            if isinstance(value, str) and any(marker in value for marker in post_compile_markers):
                return True
        return False

    @staticmethod
    def _load_stage_metadata(task_data: Dict[str, Any]) -> Dict[str, Any]:
        path_value = task_data.get("_stage_metadata_path") or task_data.get("stage_metadata_path")
        if not path_value:
            return {}
        path = Path(str(path_value))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"kg_stage_metadata_path": str(path)}
        if not isinstance(payload, dict):
            return {"kg_stage_metadata_path": str(path)}

        now_mono = time.monotonic_ns()
        current_start = payload.get("kg_stage_current_started_monotonic_ns")
        total_start = payload.get("kg_stage_total_started_monotonic_ns")
        if payload.get("kg_stage_is_active", True) and isinstance(current_start, int):
            payload["kg_stage_current_elapsed_s"] = max(0.0, (now_mono - current_start) / 1e9)
        if isinstance(total_start, int):
            payload["kg_stage_total_elapsed_s"] = max(0.0, (now_mono - total_start) / 1e9)
        payload["kg_stage_metadata_path"] = str(path)
        return payload

    async def _cancellation_watcher(self, task_id: str, cancel_event: threading.Event, base_task_id: str = "") -> None:
        """Poll Redis for a cancellation marker and signal the pool to abort.

        Runs concurrently with the in-flight task. The server may mark either
        this sub-task's id or its workflow parent (``base_task_id``, set when an
        ``/evaluate`` request is cancelled) as cancelled. On either signal,
        ``cancel_event`` is set, which makes the worker pool kill the CUDA
        subprocess and raise ``TaskCancelledError``.
        """
        interval = max(0.25, float(getattr(settings, "cancel_poll_interval_sec", 1.0)))
        watch_ids = [task_id]
        if base_task_id and base_task_id != task_id:
            watch_ids.append(base_task_id)
        try:
            while not cancel_event.is_set():
                try:
                    for watch_id in watch_ids:
                        if await self.task_manager.is_task_cancelled(watch_id):
                            logger.warning(
                                f"Worker {self.worker_id} detected cancellation "
                                f"(marker={watch_id}) for task {task_id}; aborting in-flight execution"
                            )
                            cancel_event.set()
                            return
                except Exception as e:  # pragma: no cover - polling is best effort
                    logger.debug(f"Cancellation watcher error for task {task_id}: {e}")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

    async def _run_toolkit_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run task payload through worker pool."""
        per_task_timeout_sec = self.per_task_timeout_sec
        if "timeout" in task_data:
            logger.info(f"[Worker] Load per_task_timeout from payload: {task_data['timeout']}")
            per_task_timeout_sec = task_data["timeout"]

        task_id = task_data.get("task_id", "unknown")
        base_task_id = str(task_data.get("base_task_id") or "")
        cancel_event = threading.Event()
        watcher = asyncio.create_task(self._cancellation_watcher(task_id, cancel_event, base_task_id))
        primary_error: Optional[BaseException] = None
        try:
            result_data = await self.worker_pool.execute_task(
                task_data,
                timeout=per_task_timeout_sec,
                max_retries=2,
                cancel_event=cancel_event,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:

            async def _stop_watcher() -> None:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

            watcher_cancellation_requested = False
            try:
                _, watcher_cancellation_requested = await _complete_despite_cancellation(_stop_watcher())
            except BaseException as cleanup_error:
                if isinstance(primary_error, (UnsafeGPUContainmentError, PoolShutdownContainmentError)):
                    logger.critical(
                        "Cancellation-watcher cleanup raised %s while containment was propagating; "
                        "preserving the primary containment signal",
                        type(cleanup_error).__name__,
                    )
                    raise primary_error
                if primary_error is not None:
                    logger.error(
                        "Cancellation-watcher cleanup raised %s; preserving the primary %s",
                        type(cleanup_error).__name__,
                        type(primary_error).__name__,
                    )
                    raise primary_error
                if not isinstance(cleanup_error, Exception):
                    raise
                logger.exception(
                    "Cancellation-watcher cleanup failed after the CUDA result was committed; "
                    "preserving the completed result"
                )
            if watcher_cancellation_requested and not isinstance(
                primary_error,
                (UnsafeGPUContainmentError, PoolShutdownContainmentError),
            ):
                raise asyncio.CancelledError

            # Unsafe containment must reach _process_task immediately: it
            # closes local admission and reinforces the already-durable
            # execution fence before any potentially slow quarantine page.
            # Ordinary paths still mirror pool health before returning.
            if primary_error is None:
                try:
                    await self._gpu_admission_allowed()
                except Exception as admission_error:
                    # This post-result check controls only the next dequeue.
                    # The just-observed child result has already passed the
                    # CUDA commit barrier and must not be rewritten as a
                    # failure because Redis/page coordination is unavailable.
                    logger.error(
                        "Post-result GPU admission synchronization raised %s; "
                        "preserving the completed result and leaving the next dequeue gated",
                        type(admission_error).__name__,
                    )

        if not result_data.get("success", False):
            error_type = result_data.get("error_type", "Unknown")
            error_message = result_data.get("error_message", "Unknown error")
            raise RuntimeError(f"{error_type}: {error_message}")

        result = result_data["result"]
        metadata = result.setdefault("metadata", {})
        pool_timing = result_data.get("pool_timing") or {}
        for key, value in pool_timing.items():
            metadata[f"wg_{key}"] = value

        return result

    def _update_task_stats(self, processing_time: float, success: bool):
        """Update task statistics."""
        if success:
            self.stats["tasks_completed"] += 1
        else:
            self.stats["tasks_failed"] += 1

        self.stats["total_processing_time"] += processing_time
        completed_tasks = self.stats["tasks_completed"]
        if completed_tasks > 0:
            self.stats["average_processing_time"] = self.stats["total_processing_time"] / completed_tasks
        self.stats["last_task_time"] = processing_time

    def _track_cuda_error(self):
        """
        Track CUDA errors for monitoring purposes.

        In subprocess isolation architecture, CUDA errors don't require worker restart,
        but we still track them to detect anomalies and potential issues.
        """
        from datetime import datetime, timedelta

        now = datetime.now()
        self.cuda_error_count += 1
        self.last_cuda_error_time = now
        self.cuda_errors_window.append(now)

        # Keep only errors from last 5 minutes
        cutoff = now - timedelta(minutes=5)
        self.cuda_errors_window = [t for t in self.cuda_errors_window if t > cutoff]

        # Log warning if too many errors in short time
        if len(self.cuda_errors_window) >= self.max_cuda_errors_for_alert:
            logger.warning(
                f"[MONITORING] Worker {self.worker_id} has {len(self.cuda_errors_window)} CUDA errors in last 5 minutes. "
                f"Total: {self.cuda_error_count}. This is high but subprocess isolation is handling them."
            )

    async def _heartbeat_loop(self):
        """Send periodic heartbeat to indicate worker is alive."""
        while self.running:
            try:
                # 先发 API 心跳；仅当服务端明确拒绝（409/410，已在内部停机）才退出循环。
                # 瞬时故障（5xx/网络抖动）仍继续刷新 Redis 心跳，防止 monitor 误杀。
                ok = await self._send_heartbeat_to_api()
                if not ok:
                    # _send_heartbeat_to_api 内已处理停机/剔除
                    break
                # Update Redis status
                await self._update_worker_status(online=True)

                await asyncio.sleep(10)  # Heartbeat every 10 seconds

            except Exception as e:
                logger.error(f"Error in heartbeat loop for worker {self.worker_id}: {e}")
                await asyncio.sleep(20)  # Sleep on error, then retry

    async def _update_worker_status(self, online: bool):
        """Update worker status in Redis."""
        try:
            worker_key = f"{KEY_PREFIX}:worker:{self.worker_id}"
            pool_health = self.worker_pool.get_health_snapshot() if self.worker_pool is not None else {}
            if self.health_state == "quarantined":
                health_state = "quarantined"
            elif self._stopping:
                health_state = "stopping"
            else:
                health_state = str(pool_health.get("health_state") or self.health_state)
            accepting_tasks = (
                online
                and self.running
                and not self._stopping
                and bool(pool_health.get("accepting_tasks", False))
                and health_state in {"healthy", "degraded_check"}
            )
            health_reason = self.quarantine_reason or str(pool_health.get("health_reason") or "")

            if online:
                await self.redis.hset(
                    worker_key,
                    mapping={
                        "online": "true",
                        "last_heartbeat": datetime.now().isoformat(),
                        "current_task": self.current_task or "",
                        "tasks_processed": str(self.tasks_processed),
                        "device": self.device,
                        "worker_instance_id": self.worker_instance_id,
                        "stats": str(self.stats),
                        "health_state": health_state,
                        "accepting_tasks": str(accepting_tasks).lower(),
                        "health_reason": health_reason,
                    },
                )
                # Set expiration for heartbeat (120s). Monitor handles persistence for expected workers.
                await self.redis.expire(worker_key, 120)
            else:
                await self.redis.hset(
                    worker_key,
                    mapping={
                        "online": "false",
                        "last_heartbeat": datetime.now().isoformat(),
                        "current_task": "",
                        "tasks_processed": str(self.tasks_processed),
                        "device": self.device,
                        "worker_instance_id": self.worker_instance_id,
                        "stats": str(self.stats),
                        "health_state": health_state,
                        "accepting_tasks": "false",
                        "health_reason": health_reason,
                    },
                )
                # Ensure offline records expire to avoid long-term residue
                await self.redis.expire(worker_key, 120)

        except Exception as e:
            logger.error(f"Failed to update worker status for {self.worker_id}: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "worker_instance_id": self.worker_instance_id,
            "device": self.device,
            "running": self.running,
            "current_task": self.current_task,
            "tasks_processed": self.tasks_processed,
            "stats": self.stats,
            "gpu_info": {
                "name": self.gpu_info.get("name", "Unknown"),
                "memory_total": self.gpu_info.get("total_memory", 0),
                # 主进程不使用CUDA，无法获取实时内存使用
                "memory_allocated": 0,
                "memory_reserved": 0,
            },
            "health_state": self.health_state,
            "accepting_tasks": bool(self.worker_pool and self.worker_pool.accepting_tasks)
            and self.running
            and not self._stopping
            and self.health_state in {"healthy", "degraded_check"},
            "quarantine_reason": self.quarantine_reason,
        }

    async def _register_with_api(self) -> bool:
        """Register worker with the API server."""
        try:
            if not self.http_session:
                logger.error("HTTP session not initialized")
                return False

            url = f"{self.api_url}/worker/register"
            logger.debug("Worker register URL: %s", url)
            import socket

            hostname = socket.gethostname()
            node_id = self.node_id or settings.node_id or hostname
            params = {"worker_id": self.worker_id, "device": self.device, "node_id": node_id, "hostname": hostname}

            # Retry register until API ready (e.g., when server just started)
            retry_deadline = asyncio.get_event_loop().time() + 60.0
            last_err = None
            while True:
                try:
                    async with self.http_session.post(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Successfully registered with API server: {data}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to register with API server: {response.status} - {error_text}")
                            last_err = RuntimeError(f"HTTP {response.status}")
                except (ClientConnectorError, ClientResponseError) as e:
                    last_err = e
                    logger.warning(f"API not ready for worker register: {e}. Retrying...")
                except Exception as e:
                    last_err = e
                    logger.warning(f"Register error: {e}. Retrying...")
                if asyncio.get_event_loop().time() > retry_deadline:
                    logger.error(f"Worker register timeout: {last_err}")
                    return False
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error registering with API server: {e}")
            return False

    async def _unregister_from_api(self) -> bool:
        """Unregister worker from the API server."""
        try:
            if not self.http_session:
                return True

            url = f"{self.api_url}/worker/unregister"
            params = {"worker_id": self.worker_id}

            async with self.http_session.post(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully unregistered from API server: {data}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to unregister from API server: {response.status} - {error_text}")
                    return False

        except Exception as e:
            logger.error(f"Error unregistering from API server: {e}")
            return False

    async def _send_heartbeat_to_api(self) -> bool:
        """Send heartbeat to API server.

        Returns False only when the server deliberately rejected this worker
        (409/410) and shutdown has been initiated; transient failures return
        True so the heartbeat loop keeps running and Redis stays fresh.
        """
        try:
            if not self.http_session:
                return True

            url = f"{self.api_url}/worker/heartbeat"
            import socket

            hostname = socket.gethostname()
            node_id = self.node_id or settings.node_id or hostname
            params = {"worker_id": self.worker_id, "device": self.device, "node_id": node_id, "hostname": hostname}

            async with self.http_session.post(url, params=params) as response:
                if response.status == 200:
                    return True
                # 仅当服务端明确拒绝该 worker（409/410）时才主动停机，避免“幽灵心跳”
                if response.status in (409, 410):
                    logger.warning(
                        f"Heartbeat rejected: HTTP {response.status}; shutting down worker {self.worker_id}"
                    )
                    # 标记，避免监控误判
                    self.shutdown_due_to_error = True
                    # 尝试从LB剔除，防止残留
                    try:
                        evict_url = f"{self.api_url}/worker/evict_from_lb"
                        await self.http_session.post(evict_url, params={"worker_id": self.worker_id})
                    except Exception:
                        pass
                    # 主动停止
                    self.running = False
                    await self.stop()
                    return False
                # 其他状态码（如 500，多为 API 侧 Redis 抖动）视为瞬时故障：
                # 继续心跳循环并照常刷新 Redis 心跳，防止 monitor 误杀 worker
                logger.warning(
                    f"Heartbeat got HTTP {response.status}; treating as transient, worker {self.worker_id} stays up"
                )
                return True

        except Exception as e:
            logger.error(f"Error sending heartbeat to API server (transient, worker stays up): {e}")
            return True


class WorkerManager:
    """Manages multiple GPU workers."""

    def __init__(self):
        self.workers: Dict[str, GPUWorker] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.running = False

    async def start(self):
        """Start all workers."""
        try:
            self.running = True

            # Initialize Redis connection
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established for worker manager")

            # Create workers for each GPU device
            worker_tasks = []
            import socket

            node_id = settings.node_id or socket.gethostname()
            for device in settings.gpu_devices:
                device_name = f"cuda:{device}"
                worker_id = f"{node_id}_gpu_{device}"
                worker = GPUWorker(worker_id, device_name, self.redis_client)
                self.workers[worker_id] = worker

                # Start worker in background
                worker_task = asyncio.create_task(worker.start())
                worker_tasks.append(worker_task)

            logger.info(f"Started {len(self.workers)} GPU workers")

            # Wait for all workers to complete
            await asyncio.gather(*worker_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in worker manager: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop all workers."""
        if not self.running:
            return

        logger.info("Stopping worker manager")
        self.running = False

        # Stop all workers
        stop_tasks = []
        for worker in self.workers.values():
            stop_tasks.append(worker.stop())

        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        logger.info("Worker manager stopped")

    async def get_workers_status(self) -> Dict[str, Any]:
        """Get status of all workers."""
        status = {}
        for device, worker in self.workers.items():
            status[device] = await worker.get_stats()
        return status


async def main():
    """Main entry point for GPU workers."""
    # Configure logging with file support
    logger = setup_logging("worker")

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)

    available_devices = torch.cuda.device_count()
    required_devices = max(settings.gpu_devices) + 1 if settings.gpu_devices else 1

    if available_devices < required_devices:
        logger.error(f"Not enough GPUs available. Required: {required_devices}, Available: {available_devices}")
        sys.exit(1)

    # Start worker manager
    worker_manager = WorkerManager()

    try:
        await worker_manager.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker manager error: {e}")
        sys.exit(1)
    finally:
        await worker_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
