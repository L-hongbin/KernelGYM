"""Shared compile-stage helpers for GPU workers and subprocess pools."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from kernelgym.common import ErrorCode
from kernelgym.schema import KernelEvaluationResult
from kernelgym.utils.device_info import get_cuda_device_info


_COMPILE_ARTIFACT_CACHE: dict[str, Dict[str, Any]] = {}


def _compile_cache_key(task_data: Dict[str, Any]) -> str:
    payload = {
        "backend_adapter": task_data.get("backend_adapter"),
        "backend": task_data.get("backend"),
        "entry_point": task_data.get("entry_point"),
        "kernel_code": task_data.get("kernel_code"),
        "cuda_sources": task_data.get("cuda_sources", {}),
        "source_mode": task_data.get("source_mode", "files"),
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _cache_work_dir(worker_id: str, cache_key: str) -> str:
    root = Path(tempfile.gettempdir()) / "kernelgym_compile_cache" / worker_id
    root.mkdir(parents=True, exist_ok=True)
    work_dir = root / cache_key
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def _compile_artifact_cache_index() -> str:
    index = os.environ.get("COMPILE_ARTIFACT_CACHE_INDEX") or os.environ.get(
        "CACHE_INDEX", "memory"
    )
    index = index.strip().lower()
    if index in {"", "process", "worker", "local", "fs"}:
        return "memory"
    if index not in {"memory", "redis"}:
        raise ValueError("COMPILE_ARTIFACT_CACHE_INDEX must be one of {memory, redis}")
    return index


def _redis_client() -> Any | None:
    try:
        import redis
    except Exception as exc:
        print(f"[compile_stage] Redis compile artifact cache disabled: import failed: {exc}")
        return None
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    db = int(os.environ.get("REDIS_DB", "0"))
    password = os.environ.get("REDIS_PASSWORD") or None
    try:
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
        )
        client.ping()
        return client
    except Exception as exc:
        print(f"[compile_stage] Redis compile artifact cache disabled: connection failed: {exc}")
        return None


def _redis_compile_artifact_key(cache_key: str) -> str:
    prefix = os.environ.get("REDIS_KEY_PREFIX", "kernelgym")
    node_id = os.environ.get("NODE_ID") or socket.gethostname()
    return f"{prefix}:compile_artifact_cache:{node_id}:{cache_key}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _redis_get_compile_artifact(cache_key: str) -> Dict[str, Any] | None:
    client = _redis_client()
    if client is None:
        return None
    redis_key = _redis_compile_artifact_key(cache_key)
    try:
        raw = client.get(redis_key)
        if not raw:
            return None
        entry = json.loads(raw)
        if not isinstance(entry, dict):
            return None
        artifact = entry.get("artifact")
        if not isinstance(artifact, dict):
            return None
        if not _artifact_is_usable(artifact):
            client.delete(redis_key)
            return None
        entry["hit_count"] = int(entry.get("hit_count") or 0) + 1
        entry["last_used_at"] = time.time()
        client.set(redis_key, json.dumps(_json_safe(entry), sort_keys=True))
        artifact["compile_artifact_cache_index"] = "redis"
        artifact["compile_artifact_cache_hit_source"] = "redis"
        return artifact
    except Exception as exc:
        print(f"[compile_stage] Redis compile artifact cache get failed for {redis_key}: {exc}")
        return None


def _redis_set_compile_artifact(cache_key: str, artifact: Dict[str, Any]) -> None:
    client = _redis_client()
    if client is None:
        return
    redis_key = _redis_compile_artifact_key(cache_key)
    try:
        entry = {
            "status": "ready",
            "cache_key": cache_key,
            "node_id": os.environ.get("NODE_ID") or socket.gethostname(),
            "created_at": time.time(),
            "last_used_at": time.time(),
            "hit_count": 0,
            "compiled": bool(artifact.get("compiled")),
            "work_dir": artifact.get("work_dir"),
            "so_path": artifact.get("so_path"),
            "artifact": artifact,
        }
        client.set(redis_key, json.dumps(_json_safe(entry), sort_keys=True))
    except Exception as exc:
        print(f"[compile_stage] Redis compile artifact cache set failed for {redis_key}: {exc}")


def _artifact_is_usable(artifact: Dict[str, Any]) -> bool:
    if not artifact.get("compiled"):
        return True
    so_path = artifact.get("so_path")
    work_dir = artifact.get("work_dir")
    if so_path and not Path(so_path).exists():
        return False
    if work_dir and not Path(work_dir).exists():
        return False
    return True


def sanitize_compile_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    skipped = {
        "cache_hit_source",
        "cache_index",
        "cache_key",
        "cache_scope",
        "code",
        "compile_artifact_cache_hit_source",
        "compile_artifact_cache_index",
        "module",
        "model_class",
        "session",
        "so_path",
        "tempfile_handle",
        "work_dir",
    }

    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: _sanitize(item)
                for key, item in value.items()
                if key not in skipped
            }
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [_sanitize(item) for item in value]
        return value

    return _sanitize(artifact)


def compile_kernel_artifact(
    task_data: Dict[str, Any],
    worker_id: str,
) -> tuple[Dict[str, Any], bool]:
    from kernelgym.backend import get_backend

    cache_enabled = bool(task_data.get("enable_compile_artifact_cache"))
    cache_key = _compile_cache_key(task_data)
    cache_index = _compile_artifact_cache_index() if cache_enabled else "disabled"
    if cache_enabled:
        lookup_started = time.perf_counter()
        cached_artifact = (
            _redis_get_compile_artifact(cache_key)
            if cache_index == "redis"
            else _COMPILE_ARTIFACT_CACHE.get(cache_key)
        )
        if cached_artifact is not None and _artifact_is_usable(cached_artifact):
            cached_artifact["compile_artifact_cache_enabled"] = True
            cached_artifact["compile_artifact_cache_index"] = cache_index
            cached_artifact.setdefault("compile_artifact_cache_hit_source", cache_index)
            cached_artifact["compile_timing"] = {
                "total_wall_sec": round(time.perf_counter() - lookup_started, 6),
                "cache_hit": True,
                "cache_hit_source": cached_artifact.get(
                    "compile_artifact_cache_hit_source"
                ),
            }
            return copy.deepcopy(cached_artifact), True

    backend = get_backend(task_data["backend_adapter"])
    entry_point = task_data.get("entry_point") or "Model"
    if not entry_point.endswith("New"):
        entry_point = f"{entry_point}New"

    compile_kwargs = {
        "device": task_data.get("device"),
        "backend": task_data.get("backend"),
        "entry_point": entry_point,
        "source_mode": task_data.get("source_mode", "files"),
        "cuda_sources": task_data.get("cuda_sources", {}),
        "enable_compile_artifact_cache": cache_enabled,
    }
    build_dir = task_data.get("build_dir")
    work_dir = task_data.get("work_dir")
    if build_dir is not None:
        compile_kwargs["build_dir"] = build_dir
    if work_dir is not None:
        compile_kwargs["work_dir"] = work_dir
    elif cache_enabled:
        compile_kwargs["work_dir"] = _cache_work_dir(worker_id, cache_key)

    artifact = backend.compile(task_data["kernel_code"], **compile_kwargs)
    artifact["compile_artifact_cache_enabled"] = cache_enabled
    artifact["compile_artifact_cache_index"] = cache_index
    if cache_enabled:
        artifact["cache_scope"] = "worker_process"
        artifact["cache_key"] = cache_key
        if cache_index == "redis":
            _redis_set_compile_artifact(cache_key, artifact)
        else:
            _COMPILE_ARTIFACT_CACHE[cache_key] = copy.deepcopy(artifact)
    return artifact, False


def run_compile_only_task(
    task_data: Dict[str, Any],
    worker_id: str,
    worker_device: str,
) -> Dict[str, Any]:
    """Compile a kernel without running correctness/performance on CUDA."""
    try:
        from kernelgym.schema.serialization import make_json_safe

        artifact, cache_hit = compile_kernel_artifact(task_data, worker_id)
        compiled = bool(artifact.get("compiled"))
        error_message = None
        error_code = None
        if not compiled:
            error = artifact.get("error") or "Unknown compile error"
            error_message = f"Kernel compilation failed: {error}"
            error_code = ErrorCode.COMPILATION_ERROR

        metadata = {
            "compile_only": True,
            "worker_id": worker_id,
            "worker_device": worker_device,
            "device_info": get_cuda_device_info(
                task_data.get("device") or worker_device
            ),
            "required_resource": task_data.get("required_resource"),
            "task_stage": task_data.get("task_stage"),
            "compile_artifact": sanitize_compile_artifact(artifact),
            "compile_artifact_cache_hit": cache_hit,
            "compile_artifact_cache_enabled": bool(
                task_data.get("enable_compile_artifact_cache")
            ),
        }
        if compiled and task_data.get("return_internal_compile_artifact"):
            metadata["_internal_compile_artifact"] = artifact
        result = KernelEvaluationResult(
            task_id=task_data.get("task_id", "unknown"),
            base_task_id=task_data.get("base_task_id", task_data.get("task_id", "unknown")),
            compiled=compiled,
            correctness=False,
            decoy_kernel=False,
            kernel_runtime=-1.0,
            metadata=make_json_safe(metadata),
            status="completed" if compiled else "failed",
            error_message=error_message,
            error_code=error_code,
        )
        return {"success": True, "result": result.to_dict()}
    except Exception as exc:
        return {
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
