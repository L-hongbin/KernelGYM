"""Persistent fail-closed latches for GPU workers.

Heartbeat hashes expire and are rewritten by the API, so quarantine state must
live in a separate non-TTL Redis key.  Recovery is deliberately manual: after
the host/device has been repaired, an operator deletes the latch and restarts
the worker through the normal service workflow.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from kernelgym.config import settings


logger = logging.getLogger(__name__)
_LATCH_DIR_ENV = "KERNELGYM_SAFETY_LATCH_DIR"
UNLATCHED_NOTIFICATION_PROVENANCE = "unlatched_best_effort"


@dataclass
class GPUQuarantineNotificationClaim:
    """An exclusive quarantine page claim held across one delivery attempt."""

    lock_fd: Optional[int]
    record: Dict[str, str]
    worker_id: str
    should_send: bool
    superseded: bool = False


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _latch_root() -> Path:
    configured = os.environ.get(_LATCH_DIR_ENV)
    return Path(configured) if configured else _REPOSITORY_ROOT / "logs" / "safety_latches"


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _worker_latch_path(worker_id: str) -> Path:
    return _latch_root() / "workers" / f"{_safe_component(worker_id)}.json"


def _device_latch_path(hostname: str, device: str) -> Path:
    return _latch_root() / "gpus" / _safe_component(hostname) / f"{_safe_component(device)}.json"


def _device_lock_path(hostname: str, device: str) -> Path:
    return _latch_root() / "locks" / "gpus" / _safe_component(hostname) / f"{_safe_component(device)}.lock"


def _acquire_device_lock(hostname: str, device: str) -> int:
    if not hostname or not device:
        raise ValueError("Physical GPU lock requires hostname and device")
    path = _device_lock_path(hostname, device)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(path.parent, parent_flags)
    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode) or parent_stat.st_mode & 0o022:
            raise PermissionError(f"GPU lock directory is not private: {path.parent}")
        lock_flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path.name, lock_flags, 0o600, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1 or lock_stat.st_mode & 0o077:
            raise PermissionError(f"GPU lock file is not private: {path}")
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _release_device_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


async def _run_in_thread_to_completion(function: Any, /, *args: Any, **kwargs: Any) -> tuple[Any, bool]:
    """Await a lock operation without ever abandoning its worker thread."""

    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancellation_requested = False
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancellation_requested = True
    try:
        return operation.result(), cancellation_requested
    except BaseException:
        if cancellation_requested:
            raise asyncio.CancelledError from None
        raise


async def _run_awaitable_to_completion(awaitable: Any) -> tuple[Any, bool]:
    """Finish an issued Redis mutation before releasing its device lock."""

    operation = asyncio.create_task(awaitable)
    cancellation_requested = False
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancellation_requested = True
    try:
        return operation.result(), cancellation_requested
    except BaseException:
        if cancellation_requested:
            raise asyncio.CancelledError from None
        raise


async def _release_device_lock_to_completion(fd: Optional[int]) -> None:
    if fd is None:
        return
    _, cancellation_requested = await _run_in_thread_to_completion(_release_device_lock, fd)
    if cancellation_requested:
        raise asyncio.CancelledError


def _write_json_atomic(path: Path, record: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        # Keep the rename durable across host/container restarts where the
        # underlying shared filesystem honors directory fsync.
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> Optional[Dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict) or payload.get("state") != "quarantined":
        raise RuntimeError(f"Invalid GPU safety latch: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def _existing_durable_record(
    worker_id: str,
    *,
    device: str,
    hostname: str,
) -> Optional[Dict[str, str]]:
    """Return the strongest durable record without consulting Redis."""

    if hostname and device:
        device_record = _read_json(_device_latch_path(hostname, device))
        if device_record is not None:
            return _normalize_device_record(device_record, hostname=hostname, device=device)
    return _read_json(_worker_latch_path(worker_id))


def _records_for_durable_paths(
    record: Dict[str, str],
    worker_id: str,
) -> list[tuple[Path, Dict[str, str]]]:
    records: list[tuple[Path, Dict[str, str]]] = []
    if record.get("scope") == "physical_gpu" and record.get("hostname") and record.get("device"):
        records.append((_device_latch_path(record["hostname"], record["device"]), record))
    alias_record = dict(record)
    if worker_id != alias_record.get("worker_id"):
        alias_record.setdefault("origin_worker_id", alias_record.get("worker_id", ""))
        alias_record["worker_id"] = worker_id
    records.append((_worker_latch_path(worker_id), alias_record))
    return records


def gpu_quarantine_key(worker_id: str, *, key_prefix: Optional[str] = None) -> str:
    prefix = key_prefix or settings.redis_key_prefix
    return f"{prefix}:quarantine:worker:{worker_id}"


def gpu_device_quarantine_key(hostname: str, device: str, *, key_prefix: Optional[str] = None) -> str:
    prefix = key_prefix or settings.redis_key_prefix
    return f"{prefix}:quarantine:gpu:{hostname}:{device}"


def _decode_hash(data: Dict[Any, Any]) -> Dict[str, str]:
    decoded: Dict[str, str] = {}
    for key, value in data.items():
        key_text = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
        value_text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        decoded[key_text] = value_text
    return decoded


def gpu_quarantine_generation(record: Mapping[str, Any]) -> str:
    """Return the immutable identity of one quarantine-latch generation.

    New records carry a random ``event_id``. ``created_at`` keeps records from
    the first deployment generation fenceable, while the stable legacy digest
    avoids treating mutable page-delivery fields as event identity.
    """

    event_id = str(record.get("event_id") or "").strip()
    if event_id:
        return f"event:{event_id}"
    created_at = str(record.get("created_at") or "").strip()
    if created_at:
        return f"created:{created_at}"
    scope = str(record.get("scope") or "")
    identity = {
        "scope": scope,
        "hostname": str(record.get("hostname") or ""),
        "device": str(record.get("device") or ""),
        "worker_id": str(record.get("worker_id") or "") if scope != "physical_gpu" else "",
        "fault_class": str(record.get("fault_class") or ""),
        "reason": str(record.get("reason") or ""),
        "task_id": str(record.get("task_id") or ""),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return f"legacy:{digest}"


def _normalize_device_record(
    record: Optional[Dict[str, str]],
    *,
    hostname: str,
    device: str,
) -> Optional[Dict[str, str]]:
    if record is None:
        return None
    normalized = dict(record)
    # The durable path / Redis device key is the authoritative identity. A
    # stale payload copied from another alias must not move the physical latch.
    normalized["scope"] = "physical_gpu"
    normalized["hostname"] = hostname
    normalized["device"] = device
    return normalized


def _normalize_read_record(
    record: Optional[Dict[str, str]],
    *,
    worker_id: str,
    hostname: str,
    device: str,
) -> tuple[Optional[Dict[str, str]], bool]:
    """Upgrade legacy/incomplete latch identity without weakening admission.

    Records predating explicit scopes represented physical GPU exclusions.
    Unknown scopes therefore fail closed to ``physical_gpu``.  The returned
    boolean records whether durable state must be rewritten before a notifier
    may safely take its cross-process delivery claim.
    """

    if record is None:
        return None, False
    normalized = dict(record)
    original = dict(record)
    # Any positive latch source is fail-closed.  Canonicalize incomplete Redis
    # hashes before materializing them so a later durable read cannot reject
    # the migrated record as malformed.
    normalized["state"] = "quarantined"
    normalized["manual_clear_required"] = "true"
    normalized.setdefault("page_user_state", "pending")
    scope = str(normalized.get("scope") or "")
    if scope not in {"physical_gpu", "worker_process"}:
        normalized["scope"] = "physical_gpu"
        # A scope-less record could not have completed a valid generation-
        # scoped notification claim.  Force one canonical delivery attempt.
        for key in tuple(normalized):
            if key.startswith("page_user_"):
                normalized.pop(key, None)
        normalized["page_user_state"] = "pending"
    normalized["worker_id"] = str(normalized.get("worker_id") or worker_id)
    normalized["hostname"] = str(normalized.get("hostname") or hostname)
    normalized["device"] = str(normalized.get("device") or device)
    return normalized, normalized != original


def _same_physical_gpu(record: Mapping[str, str], *, hostname: str, device: str) -> bool:
    return (
        record.get("scope") == "physical_gpu"
        and bool(hostname)
        and bool(device)
        and record.get("hostname") == hostname
        and record.get("device") == device
    )


def _constrain_worker_record_to_device(
    record: Optional[Dict[str, str]],
    *,
    hostname: str,
    device: str,
) -> Optional[Dict[str, str]]:
    """Prevent an old worker alias from impersonating the queried GPU."""

    if record is None or record.get("scope") != "physical_gpu" or not hostname or not device:
        return record
    if _same_physical_gpu(record, hostname=hostname, device=device):
        return record
    constrained = dict(record)
    constrained["scope"] = "worker_process"
    for key in tuple(constrained):
        if key.startswith("page_user_"):
            constrained.pop(key, None)
    constrained["page_user_state"] = "not_applicable"
    return constrained


def _read_durable_quarantine_records(
    worker_id: str,
    *,
    hostname: str,
    device: str,
) -> tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """Read both durable aliases in one blocking-thread handoff."""

    durable_device = None
    if hostname and device:
        durable_device = _read_json(_device_latch_path(hostname, device))
    durable_worker = _read_json(_worker_latch_path(worker_id))
    return durable_device, durable_worker


async def _read_redis_quarantine_records(
    redis_client: Any,
    worker_id: str,
    *,
    hostname: str,
    device: str,
) -> tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    """Fetch device and worker aliases in one Redis pipeline when supported."""

    keys: list[str] = []
    if hostname and device:
        keys.append(gpu_device_quarantine_key(hostname, device))
    keys.append(gpu_quarantine_key(worker_id))

    raw_records: list[Dict[Any, Any]]
    pipeline_factory = getattr(redis_client, "pipeline", None)
    if callable(pipeline_factory):
        pipeline = pipeline_factory(transaction=False)
        async with pipeline as pipe:
            pipeline_hgetall = getattr(pipe, "hgetall", None)
            if callable(pipeline_hgetall):
                for key in keys:
                    pipeline_hgetall(key)
                raw_records = list(await pipe.execute())
            else:
                # A few lightweight Redis-compatible clients implement only
                # write pipelines. Keep them usable without weakening the
                # production path, which always pipelines these reads.
                raw_records = list(await asyncio.gather(*(redis_client.hgetall(key) for key in keys)))
    else:
        raw_records = list(await asyncio.gather(*(redis_client.hgetall(key) for key in keys)))

    raw_worker = raw_records[-1]
    redis_worker = _decode_hash(raw_worker) if raw_worker else None
    redis_device = None
    if hostname and device:
        raw_device = raw_records[0]
        if raw_device:
            redis_device = _decode_hash(raw_device)
    return redis_device, redis_worker


def _strongest_quarantine_record(
    *,
    durable_device: Optional[Dict[str, str]],
    durable_worker: Optional[Dict[str, str]],
    redis_device: Optional[Dict[str, str]],
    redis_worker: Optional[Dict[str, str]],
    hostname: str = "",
    device: str = "",
) -> Optional[Dict[str, str]]:
    """Merge latches without allowing Redis to weaken durable physical state."""

    ordered = [
        durable_device,
        _constrain_worker_record_to_device(durable_worker, hostname=hostname, device=device),
        redis_device,
        _constrain_worker_record_to_device(redis_worker, hostname=hostname, device=device),
    ]
    physical = [record for record in ordered if record and record.get("scope") == "physical_gpu"]
    candidates = physical or [record for record in ordered if record]
    if not candidates:
        return None

    merged = dict(candidates[0])
    # Delivery success is monotonic until explicit manual clear.  In
    # particular, a stale Redis ``pending`` value must never overwrite the
    # durable marker written after the page actually succeeded.
    sent_record = next(
        (
            record
            for record in ordered
            if record and record.get("scope") == "physical_gpu" and record.get("page_user_state") == "sent"
        ),
        None,
    )
    if sent_record is not None and gpu_quarantine_generation(sent_record) == gpu_quarantine_generation(merged):
        for key, value in sent_record.items():
            if key.startswith("page_user_"):
                merged[key] = value
        merged["page_user_state"] = "sent"
    return merged


async def _read_merged_quarantine(
    redis_client: Any,
    worker_id: str,
    *,
    device: str,
    hostname: str,
) -> tuple[Optional[Dict[str, str]], Optional[Dict[str, str]], Optional[Dict[str, str]], bool]:
    """Read and merge all sources, preserving fail-closed Redis semantics."""

    raw_durable_device, raw_durable_worker = await asyncio.to_thread(
        _read_durable_quarantine_records,
        worker_id,
        hostname=hostname,
        device=device,
    )

    redis_error: Optional[Exception] = None
    redis_device: Optional[Dict[str, str]] = None
    redis_worker: Optional[Dict[str, str]] = None
    try:
        raw_redis_device, raw_redis_worker = await _read_redis_quarantine_records(
            redis_client,
            worker_id,
            hostname=hostname,
            device=device,
        )
    except Exception as exc:
        redis_error = exc
        raw_redis_device = None
        raw_redis_worker = None

    durable_device, durable_device_changed = _normalize_read_record(
        _normalize_device_record(raw_durable_device, hostname=hostname, device=device),
        worker_id=worker_id,
        hostname=hostname,
        device=device,
    )
    durable_worker, durable_worker_changed = _normalize_read_record(
        raw_durable_worker,
        worker_id=worker_id,
        hostname=hostname,
        device=device,
    )
    redis_device, _ = _normalize_read_record(
        _normalize_device_record(raw_redis_device, hostname=hostname, device=device),
        worker_id=worker_id,
        hostname=hostname,
        device=device,
    )
    redis_worker, _ = _normalize_read_record(
        raw_redis_worker,
        worker_id=worker_id,
        hostname=hostname,
        device=device,
    )

    record = _strongest_quarantine_record(
        durable_device=durable_device,
        durable_worker=durable_worker,
        redis_device=redis_device,
        redis_worker=redis_worker,
        hostname=hostname,
        device=device,
    )
    if record is None:
        if redis_error is not None:
            raise redis_error
        return None, redis_device, redis_worker, True

    if record.get("scope") == "physical_gpu":
        durable_ready = bool(
            durable_device
            and not durable_device_changed
            and gpu_quarantine_generation(durable_device) == gpu_quarantine_generation(record)
        )
    else:
        durable_ready = bool(
            durable_worker
            and not durable_worker_changed
            and durable_worker.get("scope") == "worker_process"
            and gpu_quarantine_generation(durable_worker) == gpu_quarantine_generation(record)
        )
    return record, redis_device, redis_worker, durable_ready


def _write_durable_aliases_locked(record: Dict[str, str], worker_id: str) -> None:
    """Materialize one canonical latch while the caller owns its device lock."""

    for path, durable_record in _records_for_durable_paths(record, worker_id):
        _write_json_atomic(path, durable_record)


def _record_contains(actual: Optional[Mapping[str, str]], expected: Mapping[str, str]) -> bool:
    return actual is not None and all(actual.get(key) == value for key, value in expected.items())


def _redis_rehydration_needed(
    record: Dict[str, str],
    worker_id: str,
    *,
    hostname: str,
    device: str,
    redis_device: Optional[Dict[str, str]],
    redis_worker: Optional[Dict[str, str]],
) -> tuple[bool, bool, Dict[str, str]]:
    alias_record = _records_for_durable_paths(record, worker_id)[-1][1]
    write_device = bool(
        record.get("scope") == "physical_gpu" and hostname and device and not _record_contains(redis_device, record)
    )
    write_worker = not _record_contains(redis_worker, alias_record)
    return write_device, write_worker, alias_record


async def _rehydrate_redis_to_completion(
    redis_client: Any,
    record: Dict[str, str],
    worker_id: str,
    *,
    hostname: str,
    device: str,
    redis_device: Optional[Dict[str, str]],
    redis_worker: Optional[Dict[str, str]],
) -> None:
    write_device, write_worker, alias_record = _redis_rehydration_needed(
        record,
        worker_id,
        hostname=hostname,
        device=device,
        redis_device=redis_device,
        redis_worker=redis_worker,
    )
    cancellation_requested = False
    try:
        if write_device:
            _, cancelled = await _run_awaitable_to_completion(
                redis_client.hset(gpu_device_quarantine_key(hostname, device), mapping=record)
            )
            cancellation_requested = cancellation_requested or cancelled
        if write_worker:
            _, cancelled = await _run_awaitable_to_completion(
                redis_client.hset(gpu_quarantine_key(worker_id), mapping=alias_record)
            )
            cancellation_requested = cancellation_requested or cancelled
    except Exception:
        # Redis is coordination state; an authoritative durable latch still
        # blocks admission while Redis is unavailable.
        pass
    if cancellation_requested:
        raise asyncio.CancelledError


async def read_gpu_quarantine(
    redis_client: Any,
    worker_id: str,
    *,
    device: str = "",
    hostname: str = "",
) -> Optional[Dict[str, str]]:
    # Read durable state even when Redis has a value.  A successful page is
    # recorded durably before Redis is updated, so returning Redis early could
    # regress ``sent`` to ``pending`` after a transient Redis failure.
    record, redis_device, redis_worker, durable_ready = await _read_merged_quarantine(
        redis_client,
        worker_id,
        hostname=hostname,
        device=device,
    )
    if record is None:
        return None

    write_device, write_worker, _ = _redis_rehydration_needed(
        record,
        worker_id,
        hostname=hostname,
        device=device,
        redis_device=redis_device,
        redis_worker=redis_worker,
    )
    if durable_ready and not write_device and not write_worker:
        return record

    # Redis is coordination state and may be restarted NOSAVE. Rehydrate it
    # best-effort from the merged authoritative record. A positive read that
    # needs mutation joins the same physical lock as manual clear, then rereads
    # all sources; a clear that won the lock can therefore never be revived by
    # a stale pre-clear snapshot.
    if not hostname or not device:
        if not durable_ready:
            record = dict(record)
            record["notification_provenance"] = UNLATCHED_NOTIFICATION_PROVENANCE
        await _rehydrate_redis_to_completion(
            redis_client,
            record,
            worker_id,
            hostname=hostname,
            device=device,
            redis_device=redis_device,
            redis_worker=redis_worker,
        )
        return record

    lock_fd: Optional[int] = None
    try:
        lock_fd, cancellation_requested = await _run_in_thread_to_completion(
            _acquire_device_lock,
            hostname,
            device,
        )
        if cancellation_requested:
            acquired_fd = lock_fd
            lock_fd = None
            await _release_device_lock_to_completion(acquired_fd)
            raise asyncio.CancelledError
        record, redis_device, redis_worker, durable_ready = await _read_merged_quarantine(
            redis_client,
            worker_id,
            hostname=hostname,
            device=device,
        )
        if record is None:
            return None
        if not durable_ready:
            try:
                _, migration_cancelled = await _run_in_thread_to_completion(
                    _write_durable_aliases_locked,
                    record,
                    worker_id,
                )
                logger.warning(
                    "Materialized legacy/Redis-only GPU quarantine for %s as scope=%s",
                    worker_id,
                    record.get("scope"),
                )
                if migration_cancelled:
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A positive latch still closes admission.  Explicitly bypass
                # durable notification dedupe so storage failure cannot turn
                # the exclusion into a silent event.
                logger.critical(
                    "Failed to materialize GPU quarantine for %s before paging: %s",
                    worker_id,
                    exc,
                )
                record = dict(record)
                record["notification_provenance"] = UNLATCHED_NOTIFICATION_PROVENANCE
        await _rehydrate_redis_to_completion(
            redis_client,
            record,
            worker_id,
            hostname=hostname,
            device=device,
            redis_device=redis_device,
            redis_worker=redis_worker,
        )
        return record
    finally:
        await _release_device_lock_to_completion(lock_fd)


def _new_quarantine_record(
    worker_id: str,
    *,
    device: str,
    reason: str,
    fault_class: str,
    task_id: str,
    node_id: str,
    hostname: str,
    physical_scope: bool,
) -> Dict[str, str]:
    return {
        "state": "quarantined",
        "worker_id": worker_id,
        "device": device,
        "reason": reason,
        "fault_class": fault_class,
        "task_id": task_id,
        "node_id": node_id,
        "hostname": hostname,
        "event_id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(),
        "manual_clear_required": "true",
        "scope": "physical_gpu" if physical_scope else "worker_process",
    }


def _write_quarantine_durable_locked(
    worker_id: str,
    *,
    device: str,
    reason: str,
    fault_class: str,
    task_id: str,
    node_id: str,
    hostname: str,
    physical_scope: bool,
) -> tuple[Dict[str, str], list[tuple[Path, Dict[str, str]]], Optional[Exception]]:
    """Write durable aliases while the async caller holds the device lock."""

    record: Optional[Dict[str, str]] = None
    durable_records: list[tuple[Path, Dict[str, str]]] = []
    durable_error: Optional[Exception] = None
    try:
        existing = _existing_durable_record(worker_id, device=device, hostname=hostname)
        preserve_physical = bool(
            existing
            and existing.get("scope") == "physical_gpu"
            and (not hostname or not device or _same_physical_gpu(existing, hostname=hostname, device=device))
        )
        preserve_worker = bool(
            existing
            and not physical_scope
            and existing.get("scope") == "worker_process"
            and existing.get("worker_id") == worker_id
            and existing.get("hostname") == hostname
            and existing.get("device") == device
        )
        preserve_existing = preserve_physical or preserve_worker
        record = (
            dict(existing)
            if preserve_existing
            else _new_quarantine_record(
                worker_id,
                device=device,
                reason=reason,
                fault_class=fault_class,
                task_id=task_id,
                node_id=node_id,
                hostname=hostname,
                physical_scope=physical_scope,
            )
        )
        record.setdefault(
            "page_user_state",
            "pending",
        )
        durable_records = _records_for_durable_paths(record, worker_id)
        for path, durable_record in durable_records:
            _write_json_atomic(path, durable_record)
    except Exception as exc:
        durable_error = exc
        if record is None:
            record = _new_quarantine_record(
                worker_id,
                device=device,
                reason=reason,
                fault_class=fault_class,
                task_id=task_id,
                node_id=node_id,
                hostname=hostname,
                physical_scope=physical_scope,
            )
            record["page_user_state"] = "pending"
        if not durable_records:
            durable_records = _records_for_durable_paths(record, worker_id)
    return record, durable_records, durable_error


async def write_gpu_quarantine(
    redis_client: Any,
    worker_id: str,
    *,
    device: str,
    reason: str,
    fault_class: str,
    task_id: str = "",
    node_id: str = "",
    hostname: str = "",
    physical_scope: bool = True,
) -> Dict[str, str]:
    """Persist one latch without allowing a concurrent clear to be revived.

    Durable aliases and their Redis mirrors are mutated while holding the same
    physical-device lock used by manual clear. The public coroutine shields the
    complete transaction so cancellation cannot release that lock between the
    filesystem and Redis phases.
    """

    operation = asyncio.create_task(
        _write_gpu_quarantine_to_completion(
            redis_client,
            worker_id,
            device=device,
            reason=reason,
            fault_class=fault_class,
            task_id=task_id,
            node_id=node_id,
            hostname=hostname,
            physical_scope=physical_scope,
        )
    )
    cancellation_requested = False
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancellation_requested = True
    result = operation.result()
    if cancellation_requested:
        raise asyncio.CancelledError
    return result


async def _write_gpu_quarantine_to_completion(
    redis_client: Any,
    worker_id: str,
    *,
    device: str,
    reason: str,
    fault_class: str,
    task_id: str,
    node_id: str,
    hostname: str,
    physical_scope: bool,
) -> Dict[str, str]:
    lock_fd: Optional[int] = None
    try:
        if hostname and device:
            try:
                lock_fd, cancellation_requested = await _run_in_thread_to_completion(
                    _acquire_device_lock,
                    hostname,
                    device,
                )
            except Exception as exc:
                raise RuntimeError(f"GPU quarantine durable safety-latch storage failed: {exc}") from exc
            if cancellation_requested:
                raise asyncio.CancelledError
        (record, durable_records, durable_error), cancellation_requested = await _run_in_thread_to_completion(
            _write_quarantine_durable_locked,
            worker_id,
            device=device,
            reason=reason,
            fault_class=fault_class,
            task_id=task_id,
            node_id=node_id,
            hostname=hostname,
            physical_scope=physical_scope,
        )
        if cancellation_requested:
            raise asyncio.CancelledError

        if durable_error is not None:
            # Redis may be the only surviving handoff to another monitor when
            # durable storage fails. Persist the explicit best-effort marker
            # there before raising so a writer crash cannot permanently
            # suppress the page by leaving an apparently normal Redis latch.
            record = dict(record)
            record["notification_provenance"] = UNLATCHED_NOTIFICATION_PROVENANCE
            durable_records = _records_for_durable_paths(record, worker_id)

        redis_error: Optional[Exception] = None
        if record.get("scope") == "physical_gpu" and record.get("hostname") and record.get("device"):
            try:
                await redis_client.hset(
                    gpu_device_quarantine_key(record["hostname"], record["device"]),
                    mapping=record,
                )
                worker_record = durable_records[-1][1]
                await redis_client.hset(gpu_quarantine_key(worker_id), mapping=worker_record)
            except Exception as exc:
                redis_error = exc
        else:
            try:
                worker_record = durable_records[-1][1]
                await redis_client.hset(gpu_quarantine_key(worker_id), mapping=worker_record)
            except Exception as exc:
                redis_error = exc

        if durable_error is not None and redis_error is not None:
            raise RuntimeError(
                f"Failed to persist GPU quarantine in durable storage ({durable_error}) and Redis ({redis_error})"
            ) from durable_error
        if durable_error is not None:
            raise RuntimeError(
                f"GPU quarantine reached Redis but durable safety-latch storage failed: {durable_error}"
            ) from durable_error
        return record
    finally:
        await _release_device_lock_to_completion(lock_fd)


def _update_notification_durable_locked(
    worker_id: str,
    *,
    device: str,
    hostname: str,
    scope: str,
    expected_generation: str,
    state: str,
    error: str,
) -> Optional[tuple[Dict[str, str], list[tuple[Path, Dict[str, str]]]]]:
    """CAS one notification outcome while the async caller owns the lock."""

    if scope == "worker_process":
        record = _read_json(_worker_latch_path(worker_id))
    else:
        record = _existing_durable_record(
            worker_id,
            device=device,
            hostname=hostname,
        )
    if record is None or record.get("scope") != scope or gpu_quarantine_generation(record) != expected_generation:
        # Manual clear, a stronger physical escalation, or a new latch generation
        # won the lock. Never recreate or mutate that event from a stale outcome.
        return None

    now = datetime.now().isoformat()
    # Delivery success is monotonic within this exact latch generation. A late
    # failure from another alias must not regress an already-confirmed page.
    if record.get("page_user_state") != "sent" or state == "sent":
        record["page_user_state"] = state
        record["page_user_updated_at"] = now
        record["page_user_error"] = str(error)[:500] if error else ""
        if state == "sent":
            record["page_user_sent_at"] = now

    durable_records = _records_for_durable_paths(record, worker_id)
    for path, durable_record in durable_records:
        _write_json_atomic(path, durable_record)
    return record, durable_records


async def update_gpu_quarantine_notification(
    redis_client: Any,
    worker_id: str,
    *,
    device: str,
    hostname: str,
    scope: str = "physical_gpu",
    expected_generation: str,
    state: str,
    error: str = "",
) -> Optional[Dict[str, str]]:
    """CAS page delivery state for one exact latch generation.

    Missing, cleared, superseded, and replacement generations are explicit
    no-ops. The public coroutine completes the durable+Redis transaction before
    propagating cancellation.
    """

    if state not in {"pending", "sent", "failed"}:
        raise ValueError(f"Invalid page-user notification state: {state}")
    if scope not in {"physical_gpu", "worker_process"}:
        raise ValueError(f"Invalid page-user notification scope: {scope}")
    if not expected_generation:
        raise ValueError("Notification update requires an expected latch generation")

    operation = asyncio.create_task(
        _update_gpu_quarantine_notification_to_completion(
            redis_client,
            worker_id,
            device=device,
            hostname=hostname,
            scope=scope,
            expected_generation=expected_generation,
            state=state,
            error=error,
        )
    )
    cancellation_requested = False
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancellation_requested = True
    result = operation.result()
    if cancellation_requested:
        raise asyncio.CancelledError
    return result


async def _update_gpu_quarantine_notification_to_completion(
    redis_client: Any,
    worker_id: str,
    *,
    device: str,
    hostname: str,
    scope: str,
    expected_generation: str,
    state: str,
    error: str,
) -> Optional[Dict[str, str]]:
    lock_fd: Optional[int] = None
    try:
        lock_fd, cancellation_requested = await _run_in_thread_to_completion(
            _acquire_device_lock,
            hostname,
            device,
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        updated, cancellation_requested = await _run_in_thread_to_completion(
            _update_notification_durable_locked,
            worker_id,
            device=device,
            hostname=hostname,
            scope=scope,
            expected_generation=expected_generation,
            state=state,
            error=error,
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        if updated is None:
            return None
        record, durable_records = updated

        if record.get("scope") == "physical_gpu" and record.get("hostname") and record.get("device"):
            await redis_client.hset(
                gpu_device_quarantine_key(record["hostname"], record["device"]),
                mapping=record,
            )
        await redis_client.hset(gpu_quarantine_key(worker_id), mapping=durable_records[-1][1])
        return record
    finally:
        await _release_device_lock_to_completion(lock_fd)


def acquire_gpu_quarantine_notification_claim(
    input_record: Mapping[str, Any],
) -> GPUQuarantineNotificationClaim:
    """Claim one quarantine page attempt across all worker processes.

    The advisory lock is held by the returned object until ``release``. A
    crashed sender releases the kernel lock automatically; the next process may
    then retry a durable ``sending`` marker.
    """

    record = {str(key): "" if value is None else str(value) for key, value in input_record.items()}
    scope = record.get("scope")
    if scope not in {"physical_gpu", "worker_process"}:
        raise ValueError("Notification claims require a GPU quarantine latch")
    hostname = record.get("hostname", "")
    device = record.get("device", "")
    worker_id = record.get("worker_id", "")
    if not hostname or not device or not worker_id:
        raise ValueError("GPU notification claim requires hostname, device, and worker_id")

    input_generation = gpu_quarantine_generation(record)
    lock_fd = _acquire_device_lock(hostname, device)
    try:
        durable_worker = _read_json(_worker_latch_path(worker_id))
        durable_device = _normalize_device_record(
            _read_json(_device_latch_path(hostname, device)),
            hostname=hostname,
            device=device,
        )
        if scope == "physical_gpu":
            current = durable_device
        else:
            # A worker-only event that was persisted before a physical latch
            # won the same device lock is stale.  Suppress that weaker page;
            # the physical-latch owner is responsible for the stronger alert.
            # Conversely, if the worker page already owns this lock, a later
            # physical escalation waits and both distinct events may be sent.
            if durable_device is not None:
                return GPUQuarantineNotificationClaim(
                    lock_fd,
                    durable_device,
                    worker_id,
                    False,
                    superseded=True,
                )
            # A worker notification is scoped to this worker generation, not
            # proof that every alias of the physical device is defective.
            current = durable_worker if durable_worker and durable_worker.get("scope") == scope else None

        # The durable latch is authoritative. The input is only a generation
        # fence and must never recreate a manually-cleared event or replace a
        # newer event that won this lock first.
        if current is None or gpu_quarantine_generation(current) != input_generation:
            return GPUQuarantineNotificationClaim(
                lock_fd,
                dict(current or record),
                worker_id,
                False,
                superseded=True,
            )

        if current.get("page_user_state") == "sent":
            return GPUQuarantineNotificationClaim(lock_fd, current, worker_id, False)

        current["page_user_state"] = "sending"
        current["page_user_claimed_at"] = datetime.now().isoformat()
        current["page_user_claim_owner"] = f"{os.getpid()}"
        try:
            prior_attempts = max(0, int(current.get("page_user_attempt_count", "0")))
        except (TypeError, ValueError):
            prior_attempts = 0
        current["page_user_attempt_count"] = str(prior_attempts + 1)
        for path, durable_record in _records_for_durable_paths(current, worker_id):
            _write_json_atomic(path, durable_record)
        return GPUQuarantineNotificationClaim(lock_fd, current, worker_id, True)
    except BaseException:
        _release_device_lock(lock_fd)
        raise


def finish_gpu_quarantine_notification_claim(
    claim: GPUQuarantineNotificationClaim,
    *,
    state: str,
    error: str = "",
) -> Dict[str, str]:
    """Finish a held claim durably before another process may send."""

    if claim.lock_fd is None:
        raise RuntimeError("GPU quarantine notification claim is no longer held")
    if state not in {"sent", "failed"}:
        raise ValueError(f"Invalid claim completion state: {state}")

    record = dict(claim.record)
    now = datetime.now().isoformat()
    record["page_user_state"] = state
    record["page_user_updated_at"] = now
    record["page_user_error"] = str(error)[:500] if error else ""
    record.pop("page_user_claim_owner", None)
    if state == "sent":
        record["page_user_sent_at"] = now
    for path, durable_record in _records_for_durable_paths(record, claim.worker_id):
        _write_json_atomic(path, durable_record)
    claim.record = record
    return record


def release_gpu_quarantine_notification_claim(claim: GPUQuarantineNotificationClaim) -> None:
    fd = claim.lock_fd
    claim.lock_fd = None
    _release_device_lock(fd)


def _clear_durable_latches(worker_id: str, *, device: str, hostname: str) -> tuple[bool, set[str]]:
    """Delete durable state first and return all physical worker aliases."""

    removed = False
    aliases = {worker_id}
    explicit_worker_path = _worker_latch_path(worker_id)

    if hostname and device:
        workers_dir = _latch_root() / "workers"
        try:
            worker_paths = list(workers_dir.glob("*.json"))
        except FileNotFoundError:
            worker_paths = []
        for path in worker_paths:
            try:
                record = _read_json(path)
            except (OSError, ValueError, RuntimeError):
                continue
            if (
                record
                and record.get("scope") == "physical_gpu"
                and record.get("hostname") == hostname
                and record.get("device") == device
            ):
                aliases.add(record.get("worker_id") or "")
                try:
                    path.unlink()
                    removed = True
                except FileNotFoundError:
                    pass

        try:
            _device_latch_path(hostname, device).unlink()
            removed = True
        except FileNotFoundError:
            pass

    try:
        explicit_worker_path.unlink()
        removed = True
    except FileNotFoundError:
        pass
    aliases.discard("")
    return removed, aliases


async def _redis_physical_aliases(redis_client: Any, *, hostname: str, device: str) -> set[str]:
    aliases: set[str] = set()
    prefix = f"{settings.redis_key_prefix}:quarantine:worker:"
    async for raw_key in redis_client.scan_iter(f"{prefix}*", count=500):
        key = raw_key.decode("utf-8", errors="replace") if isinstance(raw_key, bytes) else str(raw_key)
        if not key.startswith(prefix):
            continue
        data = await redis_client.hgetall(key)
        if not data:
            continue
        record = _decode_hash(data)
        if (
            record.get("scope") in {"", "physical_gpu"}
            and record.get("hostname") == hostname
            and record.get("device") == device
        ):
            aliases.add(key[len(prefix) :])
    return aliases


async def clear_gpu_quarantine(
    redis_client: Any,
    worker_id: str,
    *,
    device: str = "",
    hostname: str = "",
) -> bool:
    """Explicit recovery primitive; never called automatically.

    Operators must stop the affected worker before clearing.  Durable state is
    removed before Redis coordination state so a NOSAVE restart cannot revive
    an old worker alias after a normal manual-clear operation.
    """

    aliases = {worker_id}
    if device and hostname:
        # Redis may contain aliases created while rehydrating a NOSAVE restart;
        # those aliases need not have their own durable worker file.
        aliases.update(await _redis_physical_aliases(redis_client, hostname=hostname, device=device))

    lock_fd: Optional[int] = None
    try:
        if device and hostname:
            lock_fd, cancellation_requested = await _run_in_thread_to_completion(
                _acquire_device_lock,
                hostname,
                device,
            )
            if cancellation_requested:
                acquired_fd = lock_fd
                lock_fd = None
                await _release_device_lock_to_completion(acquired_fd)
                raise asyncio.CancelledError
            # Re-scan after acquiring the mutation lock so a page/write that was
            # already in flight cannot leave a newly published alias behind.
            aliases.update(await _redis_physical_aliases(redis_client, hostname=hostname, device=device))
        (file_removed, durable_aliases), cancellation_requested = await _run_in_thread_to_completion(
            _clear_durable_latches,
            worker_id,
            device=device,
            hostname=hostname,
        )
        aliases.update(durable_aliases)
        keys = [gpu_quarantine_key(alias) for alias in sorted(aliases)]
        if device and hostname:
            keys.append(gpu_device_quarantine_key(hostname, device))
        redis_result, redis_cancellation_requested = await _run_awaitable_to_completion(redis_client.delete(*keys))
        redis_removed = bool(redis_result)
        if cancellation_requested or redis_cancellation_requested:
            raise asyncio.CancelledError
        return redis_removed or file_removed
    finally:
        await _release_device_lock_to_completion(lock_fd)


__all__ = [
    "GPUQuarantineNotificationClaim",
    "UNLATCHED_NOTIFICATION_PROVENANCE",
    "acquire_gpu_quarantine_notification_claim",
    "clear_gpu_quarantine",
    "finish_gpu_quarantine_notification_claim",
    "gpu_device_quarantine_key",
    "gpu_quarantine_generation",
    "gpu_quarantine_key",
    "read_gpu_quarantine",
    "release_gpu_quarantine_notification_claim",
    "update_gpu_quarantine_notification",
    "write_gpu_quarantine",
]
