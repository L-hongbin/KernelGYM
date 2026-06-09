"""Cached system/GPU stats with a per-process background refresher.

The detailed ``/health`` and ``/metrics`` endpoints must never block the event
loop. ``nvidia-smi`` can take seconds (or stall) when the GPUs are saturated,
and ``psutil.cpu_percent(interval=...)`` sleeps synchronously. Both are pulled
off the request hot path here: a single background task per worker process
refreshes a snapshot on an interval (running the blocking work in a thread), and
request handlers read the latest snapshot instantly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil

from .utils import compute_gpu_info_sync, format_timestamp

logger = logging.getLogger("kernelgym.api")

DEFAULT_REFRESH_INTERVAL = 5.0
_MIN_REFRESH_INTERVAL = 0.5


class SystemStatsCache:
    """Holds the latest system snapshot and refreshes it in the background."""

    def __init__(self, refresh_interval: float = DEFAULT_REFRESH_INTERVAL) -> None:
        self._refresh_interval = max(_MIN_REFRESH_INTERVAL, float(refresh_interval))
        self._snapshot: Optional[Dict[str, Any]] = None
        self._task: Optional[asyncio.Task] = None
        # Single-flight guard for the cold-start path so concurrent first callers
        # don't each spawn a redundant _compute() thread.
        self._refresh_lock = asyncio.Lock()

    def set_interval(self, refresh_interval: float) -> None:
        self._refresh_interval = max(_MIN_REFRESH_INTERVAL, float(refresh_interval))

    @property
    def snapshot(self) -> Optional[Dict[str, Any]]:
        return self._snapshot

    @property
    def refresh_lock(self) -> asyncio.Lock:
        return self._refresh_lock

    def _compute(self) -> Dict[str, Any]:
        """Blocking snapshot computation; always invoked via ``asyncio.to_thread``."""
        # interval=None is non-blocking: it reports usage since the previous
        # call, so the refresher's cadence defines the sampling window.
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        gpu_status = compute_gpu_info_sync()
        return {
            "timestamp": format_timestamp(datetime.now()),
            "monotonic": time.monotonic(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "gpu_status": gpu_status,
        }

    async def refresh_once(self) -> Dict[str, Any]:
        snapshot = await asyncio.to_thread(self._compute)
        self._snapshot = snapshot
        return snapshot

    async def _run(self) -> None:
        while True:
            try:
                await self.refresh_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep serving the last good snapshot
                logger.warning("System stats refresh failed: %s", exc)
            await asyncio.sleep(self._refresh_interval)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        # Prime psutil so the first real cpu sample is meaningful, then populate
        # an initial snapshot before the server starts serving requests.
        psutil.cpu_percent(interval=None)
        try:
            await self.refresh_once()
        except Exception as exc:
            logger.warning("Initial system stats refresh failed: %s", exc)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None


_cache = SystemStatsCache()


async def start(refresh_interval: Optional[float] = None) -> None:
    """Start the background refresher (idempotent)."""
    if refresh_interval is not None:
        _cache.set_interval(refresh_interval)
    await _cache.start()


async def stop() -> None:
    """Stop the background refresher."""
    await _cache.stop()


async def get_snapshot() -> Dict[str, Any]:
    """Return the latest snapshot, computing one on demand if none exists yet.

    The on-demand path still runs the blocking work in a thread, so callers never
    stall the event loop even before the refresher has produced a first snapshot.
    A single-flight lock collapses concurrent cold-start callers onto one refresh.
    """
    snap = _cache.snapshot
    if snap is not None:
        return snap
    async with _cache.refresh_lock:
        snap = _cache.snapshot
        if snap is None:
            snap = await _cache.refresh_once()
        return snap
