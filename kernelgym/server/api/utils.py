"""Utility functions for KernelGym API server."""

import asyncio
import logging
import shutil
import subprocess
import time
from datetime import datetime
from typing import Dict, Any

import torch

from kernelgym.config import settings

logger = logging.getLogger(__name__)


def format_timestamp(dt: datetime) -> str:
    return dt.isoformat() + "Z"


def compute_gpu_info_sync() -> Dict[str, Any]:
    """Synchronous GPU info collection (nvidia-smi, falling back to torch).

    This blocks (subprocess + driver calls), so callers on the event loop must
    invoke it via a thread (see ``get_gpu_info``) or the cached snapshot in
    ``system_stats``.
    """
    nvidia_smi_info = _get_gpu_info_from_nvidia_smi()
    if nvidia_smi_info is not None:
        return nvidia_smi_info
    return _get_gpu_info_from_torch()


async def get_gpu_info() -> Dict[str, Any]:
    # nvidia-smi can take seconds under GPU saturation; keep it off the loop.
    return await asyncio.to_thread(compute_gpu_info_sync)


def _get_gpu_info_from_nvidia_smi() -> Dict[str, Any] | None:
    """Read whole-device GPU memory/utilization instead of this process only."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi GPU query failed: %s", exc)
        return None
    if completed.returncode != 0:
        logger.warning("nvidia-smi GPU query failed: %s", completed.stderr.strip())
        return None

    gpu_info: Dict[str, Any] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            memory_used_mib = float(parts[2])
            memory_total_mib = float(parts[3])
            utilization = float(parts[4])
        except ValueError:
            continue
        if index not in settings.gpu_devices:
            continue
        used_gib = memory_used_mib / 1024
        total_gib = memory_total_mib / 1024
        used_percent = (memory_used_mib / memory_total_mib) * 100 if memory_total_mib else 0.0
        gpu_info[f"cuda:{index}"] = {
            "name": parts[1],
            "memory_total": f"{total_gib:.1f}GB",
            "memory_used": f"{used_gib:.1f}GB",
            "memory_used_percent": f"{used_percent:.1f}%",
            "utilization_gpu_percent": f"{utilization:.1f}%",
            "available": True,
            "source": "nvidia-smi",
        }
    return gpu_info or None


def _get_gpu_info_from_torch() -> Dict[str, Any]:
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            if i in settings.gpu_devices:
                device = f"cuda:{i}"
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)

                    memory_used_percent = (memory_allocated / memory_total) * 100

                    gpu_info[device] = {
                        "name": gpu_name,
                        "memory_total": f"{memory_total / (1024**3):.1f}GB",
                        "memory_allocated": f"{memory_allocated / (1024**3):.1f}GB",
                        "memory_reserved": f"{memory_reserved / (1024**3):.1f}GB",
                        "memory_used_percent": f"{memory_used_percent:.1f}%",
                        "available": True,
                        "source": "torch",
                    }
                except Exception as exc:
                    gpu_info[device] = {"error": str(exc), "available": False}

        return gpu_info

    except Exception as exc:
        logger.error(f"Error getting GPU info: {exc}")
        return {"error": str(exc)}


async def get_system_health() -> Dict[str, Any]:
    try:
        # Served from the cached snapshot so /health never samples CPU or forks
        # nvidia-smi on the event loop (see kernelgym.server.api.system_stats).
        from . import system_stats

        snap = await system_stats.get_snapshot()

        queue_status = {"pending": 0, "processing": 0, "completed": 0}
        snapshot_age = None
        if "monotonic" in snap:
            snapshot_age = round(max(0.0, time.monotonic() - snap["monotonic"]), 3)

        return {
            "status": "healthy",
            "timestamp": snap["timestamp"],
            "gpu_status": snap["gpu_status"],
            "queue_status": queue_status,
            "memory_usage": {
                "cpu_percent": snap["cpu_percent"],
                "memory_percent": snap["memory_percent"],
                "memory_available": f"{snap['memory_available_gb']:.1f}GB",
                "memory_total": f"{snap['memory_total_gb']:.1f}GB",
            },
            "active_tasks": 0,
            "total_processed": 0,
            "uptime": 0.0,
            "snapshot_age_s": snapshot_age,
        }

    except Exception as exc:
        logger.error(f"Error getting system health: {exc}")
        # Return a payload that still satisfies SystemHealthResponse so the route
        # serves a 200 "unhealthy" instead of a 500 from response-model validation.
        return {
            "status": "unhealthy",
            "timestamp": format_timestamp(datetime.now()),
            "gpu_status": {"error": str(exc)},
            "queue_status": {"pending": 0, "processing": 0, "completed": 0},
            "memory_usage": {},
            "active_tasks": 0,
            "total_processed": 0,
            "uptime": 0.0,
            "snapshot_age_s": None,
        }


async def get_system_metrics() -> Dict[str, Any]:
    try:
        from . import system_stats

        snap = await system_stats.get_snapshot()

        gpu_info = snap["gpu_status"]
        avg_gpu_utilization = 0.0
        if gpu_info and "error" not in gpu_info:
            utilizations = []
            for info in gpu_info.values():
                if isinstance(info, dict) and "memory_used_percent" in info:
                    utilizations.append(float(info["memory_used_percent"].rstrip("%")))
            if utilizations:
                avg_gpu_utilization = sum(utilizations) / len(utilizations)

        return {
            "timestamp": snap["timestamp"],
            "performance_metrics": {
                "avg_processing_time": 0.0,
                "throughput_per_hour": 0.0,
                "success_rate": 0.0,
            },
            "resource_metrics": {
                "avg_gpu_utilization": avg_gpu_utilization,
                "memory_usage_percent": snap["memory_percent"],
                "cpu_usage_percent": snap["cpu_percent"],
            },
            "queue_metrics": {"pending_tasks": 0, "active_tasks": 0, "completed_tasks": 0},
            "error_metrics": {"compilation_errors": 0, "runtime_errors": 0, "timeout_errors": 0},
        }

    except Exception as exc:
        logger.error(f"Error getting system metrics: {exc}")
        # Keep all MetricsResponse-required keys so the route returns 200, not 500.
        return {
            "timestamp": format_timestamp(datetime.now()),
            "performance_metrics": {},
            "resource_metrics": {"error": str(exc)},
            "queue_metrics": {},
            "error_metrics": {},
        }


async def validate_gpu_availability() -> bool:
    try:
        if not torch.cuda.is_available():
            return False

        available_devices = torch.cuda.device_count()
        required_devices = max(settings.gpu_devices) + 1 if settings.gpu_devices else 1
        return available_devices >= required_devices

    except Exception as exc:
        logger.error(f"Error validating GPU availability: {exc}")
        return False


async def cleanup_old_tasks(redis_client, max_age_hours: int = 24) -> int:
    try:
        return 0
    except Exception as exc:
        logger.error(f"Error cleaning up old tasks: {exc}")
        return 0


async def get_task_statistics(redis_client) -> Dict[str, Any]:
    try:
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0,
        }
    except Exception as exc:
        logger.error(f"Error getting task statistics: {exc}")
        return {"error": str(exc)}
