#!/usr/bin/env python3
"""Format a reward-node check report from reward-node status JSON.

The companion bash script `check_node.sh` does the HTTP probing and pipes a
combined ``{"health": ..., "queue": ..., "workers": ...}`` JSON document to
this program on stdin. We keep the rendering here so the bash side stays a thin
wrapper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime


DEFAULT_MAX_HEARTBEAT_AGE_S = 180
DEFAULT_REDIS_TIMEOUT_S = 5.0
REDIS_WORKER_SCAN_TIMEOUT_S = 15.0
REDIS_PROCESSING_SCAN_TIMEOUT_S = 15.0


def render_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print rows as a `+--+--+` ASCII table with left-aligned cells."""
    str_rows = [list(map(str, r)) for r in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print()
    print(f"=== {title} ===")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in str_rows:
        print(fmt.format(*row))
    print(sep)


def _device_key(name: str) -> tuple[int, str]:
    """Sort key for `cuda:N` device strings and `worker_gpu_N` worker ids."""
    tail = name.rsplit(":", 1)[-1] if ":" in name else name.rsplit("_", 1)[-1]
    return (int(tail) if tail.isdigit() else 10**9, name)


def _short_gpu_name(name: object) -> str:
    text = str(name or "?")
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _short_hostname(hostname: object) -> str:
    text = str(hostname or "")
    if text.startswith("ai-"):
        return text[len("ai-") :]
    return text


def _short_task_id(task_id: object) -> str:
    text = str(task_id or "")
    prefix = "parallel_task_"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _short_reason(reason: object, limit: int = 72) -> str:
    text = " ".join(str(reason or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_timestamp(value: object) -> datetime | None:
    """Parse service timestamps as naive local datetimes for age checks."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _is_gpu_worker(worker_id: str, info: object) -> bool:
    if isinstance(info, dict):
        device = str(info.get("device", ""))
        if device.startswith("cuda:"):
            return True
    return "_gpu_" in worker_id


def _is_cpu_worker(worker_id: str, info: object) -> bool:
    if isinstance(info, dict) and str(info.get("device", "")) == "cpu":
        return True
    return "_cpu_" in worker_id


def _worker_is_busy(info: object) -> bool:
    if not isinstance(info, dict):
        return False
    if info.get("busy") is True:
        return True
    if str(info.get("status", "")).lower() in {"busy", "processing", "running"}:
        return True
    current_task = str(info.get("current_task") or info.get("task_id") or "").strip()
    return current_task not in {"", "none", "null"}


def _worker_has_busy_signal(info: object) -> bool:
    if not isinstance(info, dict):
        return False
    return any(key in info for key in ("busy", "current_task", "task_id")) or str(info.get("status", "")).lower() in {
        "busy",
        "processing",
        "running",
    }


def _busy_count(workers: dict[str, object]) -> int | str:
    if workers and not any(_worker_has_busy_signal(info) for info in workers.values()):
        return "unknown"
    return sum(1 for info in workers.values() if _worker_is_busy(info))


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _queue_count(queue: dict) -> int:
    explicit_count = _int_or_none(queue.get("queue_count"))
    if explicit_count is not None:
        return explicit_count
    count = _int_or_none(queue.get("pending")) or 0
    worker_queues = queue.get("worker_queues", {})
    if isinstance(worker_queues, dict):
        for value in worker_queues.values():
            count += _int_or_none(value) or 0
    return count


def _run_redis_cli(
    redis_host: str,
    redis_port: int,
    *args: str,
    timeout_s: float = DEFAULT_REDIS_TIMEOUT_S,
) -> str | None:
    """Run redis-cli for local container checks; return stdout on success."""
    redis_cli = shutil.which("redis-cli")
    if not redis_cli:
        return None
    try:
        completed = subprocess.run(
            [redis_cli, "-h", redis_host, "-p", str(redis_port), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse_hgetall(output: str | None) -> dict[str, str]:
    if not output:
        return {}
    lines = output.splitlines()
    parsed: dict[str, str] = {}
    for i in range(0, len(lines) - 1, 2):
        parsed[lines[i]] = lines[i + 1]
    return parsed


def _redis_snapshot(redis_host: str, redis_port: int, redis_prefix: str) -> dict[str, object] | None:
    """Read queue lengths and worker hashes directly from Redis when local."""
    if (_run_redis_cli(redis_host, redis_port, "ping") or "").strip() != "PONG":
        return None

    gpu_queue = _int_or_none(_run_redis_cli(redis_host, redis_port, "llen", f"{redis_prefix}:queue:resource:gpu")) or 0
    cpu_queue = _int_or_none(_run_redis_cli(redis_host, redis_port, "llen", f"{redis_prefix}:queue:resource:cpu")) or 0
    cpu_processing, gpu_processing = _redis_processing_counts(redis_host, redis_port, redis_prefix)

    worker_ids_out = _run_redis_cli(redis_host, redis_port, "smembers", f"{redis_prefix}:workers")
    worker_keys = [f"{redis_prefix}:worker:{worker_id}" for worker_id in (worker_ids_out or "").splitlines()]
    if not worker_keys:
        keys_out = _run_redis_cli(
            redis_host,
            redis_port,
            "--scan",
            "--pattern",
            f"{redis_prefix}:worker:*",
            timeout_s=REDIS_WORKER_SCAN_TIMEOUT_S,
        )
        worker_keys = (keys_out or "").splitlines()
    workers: dict[str, dict[str, str]] = {}
    for key in sorted(worker_keys):
        worker_id = key.rsplit(":", 1)[-1]
        info = _parse_hgetall(_run_redis_cli(redis_host, redis_port, "hgetall", key))
        if info:
            if info.get("online") in {"true", "1", "yes", "initializing"} and not info.get("status"):
                info["status"] = "online"
            workers[worker_id] = info

    expected_ids_out = _run_redis_cli(redis_host, redis_port, "smembers", f"{redis_prefix}:expected_workers")
    expected_workers: dict[str, dict[str, str]] = {}
    for worker_id in (expected_ids_out or "").splitlines():
        info = _parse_hgetall(
            _run_redis_cli(redis_host, redis_port, "hgetall", f"{redis_prefix}:expected_worker:{worker_id}")
        )
        expected_workers[worker_id] = info

    quarantine_keys_out = _run_redis_cli(
        redis_host,
        redis_port,
        "--scan",
        "--pattern",
        f"{redis_prefix}:quarantine:*",
        timeout_s=REDIS_WORKER_SCAN_TIMEOUT_S,
    )
    quarantines: list[dict[str, str]] = []
    quarantine_checked = quarantine_keys_out is not None
    for key in (quarantine_keys_out or "").splitlines():
        record_output = _run_redis_cli(redis_host, redis_port, "hgetall", key)
        if record_output is None:
            quarantine_checked = False
            continue
        record = _parse_hgetall(record_output)
        if record.get("state") == "quarantined":
            quarantines.append(record)

    return {
        "queue": {
            "queue_count": gpu_queue + cpu_queue,
            "pending": gpu_queue + cpu_queue,
            "processing": cpu_processing + gpu_processing,
            "resource_queues": {"gpu": gpu_queue, "cpu": cpu_queue},
            "processing_by_resource": {"gpu": gpu_processing, "cpu": cpu_processing},
            "source": "redis",
        },
        "workers": workers,
        "expected_workers": expected_workers,
        "quarantines": quarantines,
        "quarantine_checked": quarantine_checked,
    }


def _redis_processing_counts(redis_host: str, redis_port: int, redis_prefix: str) -> tuple[int, int]:
    """Count processing CPU/GPU tasks from Redis task hashes in one Lua scan."""
    script = r"""
local cursor = "0"
local cpu = 0
local gpu = 0
repeat
  local result = redis.call("SCAN", cursor, "MATCH", ARGV[1] .. ":task:*", "COUNT", 1000)
  cursor = result[1]
  for _, key in ipairs(result[2]) do
    local status = redis.call("HGET", key, "status")
    if status == "processing" then
      local data = redis.call("HGET", key, "data") or ""
      if string.find(data, '"required_resource": "cpu"', 1, true)
          or string.find(data, '"required_resource":"cpu"', 1, true) then
        cpu = cpu + 1
      else
        gpu = gpu + 1
      end
    end
  end
until cursor == "0"
return {cpu, gpu}
"""
    output = _run_redis_cli(
        redis_host,
        redis_port,
        "eval",
        script,
        "0",
        redis_prefix,
        timeout_s=REDIS_PROCESSING_SCAN_TIMEOUT_S,
    )
    values = [_int_or_none(line) for line in (output or "").splitlines()]
    if len(values) >= 2 and values[0] is not None and values[1] is not None:
        return values[0], values[1]
    return 0, 0


def _nvidia_smi_gpu_snapshot() -> dict[str, dict[str, object]] | None:
    """Read whole-device GPU memory/utilization for local container checks."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None

    gpus: dict[str, dict[str, object]] = {}
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
        used_percent = (memory_used_mib / memory_total_mib) * 100 if memory_total_mib else 0.0
        gpus[f"cuda:{index}"] = {
            "name": parts[1],
            "memory_total": f"{memory_total_mib / 1024:.1f}GB",
            "memory_used": f"{memory_used_mib / 1024:.1f}GB",
            "memory_used_percent": f"{used_percent:.1f}%",
            "utilization_gpu_percent": f"{utilization:.1f}%",
            "available": True,
            "source": "nvidia-smi",
        }
    return gpus or None


def _merge_gpu_status(health: dict, local_gpus: dict[str, dict[str, object]] | None) -> dict:
    if not local_gpus:
        return health
    merged = dict(health)
    api_gpus = health.get("gpu_status")
    if not isinstance(api_gpus, dict):
        merged["gpu_status"] = local_gpus
        return merged
    merged_gpus = dict(api_gpus)
    for device, info in local_gpus.items():
        api_info = merged_gpus.get(device)
        if isinstance(api_info, dict):
            merged_gpus[device] = {**api_info, **info}
        else:
            merged_gpus[device] = info
    merged["gpu_status"] = merged_gpus
    return merged


def _merge_workers(api_workers: dict[str, object], redis_workers: dict[str, object]) -> dict[str, object]:
    """Overlay Redis worker hashes on API worker info so current_task is fresh."""
    merged: dict[str, object] = dict(api_workers)
    for worker_id, redis_info in redis_workers.items():
        api_info = merged.get(worker_id)
        if isinstance(api_info, dict) and isinstance(redis_info, dict):
            merged[worker_id] = {**api_info, **redis_info}
        else:
            merged[worker_id] = redis_info
    return merged


def _merge_expected_workers(workers: dict[str, object], expected_workers: dict[str, object]) -> dict[str, object]:
    """Merge topology metadata and add missing workers so denominators cannot shrink."""

    merged: dict[str, object] = dict(workers)
    for worker_id, expected_info in expected_workers.items():
        if not isinstance(expected_info, dict):
            continue
        current = merged.get(worker_id)
        if isinstance(current, dict):
            combined = dict(current)
            for field in ("device", "hostname", "node_id"):
                if not str(combined.get(field) or "").strip() and str(expected_info.get(field) or "").strip():
                    combined[field] = expected_info[field]
            merged[worker_id] = combined
            continue
        merged[worker_id] = {
            **expected_info,
            "status": "missing",
            "online": "false",
            "health_state": "missing",
            "accepting_tasks": "false",
            "_expected_missing": True,
        }
    return merged


def _merge_quarantines(
    workers: dict[str, object],
    quarantine_records: list[dict[str, str]],
) -> dict[str, object]:
    """Overlay worker aliases and host/device physical latches without cross-node leakage."""

    worker_records: dict[str, dict[str, str]] = {}
    physical_records: dict[tuple[str, str], dict[str, str]] = {}
    for record in quarantine_records:
        if record.get("state") != "quarantined":
            continue
        scope = record.get("scope", "worker_process")
        worker_id = record.get("worker_id", "")
        if worker_id:
            current = worker_records.get(worker_id)
            if current is None or (scope == "physical_gpu" and current.get("scope") != "physical_gpu"):
                worker_records[worker_id] = record
        if scope == "physical_gpu":
            hostname = record.get("hostname", "")
            device = record.get("device", "")
            if hostname and device:
                physical_records[(hostname, device)] = record

    merged: dict[str, object] = dict(workers)
    for worker_id, info in workers.items():
        if not isinstance(info, dict):
            continue
        hostname = str(info.get("hostname") or "")
        device = str(info.get("device") or "")
        record = physical_records.get((hostname, device)) or worker_records.get(worker_id)
        if record is None:
            continue
        merged[worker_id] = {
            **info,
            "quarantine_state": record.get("state", "quarantined"),
            "quarantine_scope": record.get("scope", "worker_process"),
            "quarantine_fault_class": record.get("fault_class", ""),
            "quarantine_reason": record.get("reason", ""),
            "quarantine_event_id": record.get("event_id", ""),
        }
    return merged


def _expected_worker_contract_is_valid(expected_workers: dict[str, object]) -> bool:
    """Require every expected entry to identify one host-owned CPU/GPU slot."""

    if not expected_workers:
        return False
    gpu_count = 0
    for info in expected_workers.values():
        if not isinstance(info, dict) or not str(info.get("hostname") or "").strip():
            return False
        device = str(info.get("device") or "")
        if device.startswith("cuda:"):
            gpu_count += 1
        elif device == "cpu":
            continue
        else:
            return False
    return gpu_count > 0


def _worker_heartbeat_age_s(info: object, now: datetime) -> float | None:
    if not isinstance(info, dict):
        return None
    heartbeat = _parse_timestamp(info.get("last_heartbeat"))
    if heartbeat is None:
        return None
    return max(0.0, (now - heartbeat).total_seconds())


def _worker_is_fresh(info: object, now: datetime, max_heartbeat_age_s: int) -> bool:
    if not isinstance(info, dict):
        return False
    if info.get("status") != "online":
        return False
    age_s = _worker_heartbeat_age_s(info, now)
    return age_s is not None and age_s <= max_heartbeat_age_s


def _gpu_worker_is_ready(info: object, now: datetime, max_heartbeat_age_s: int) -> bool:
    if not _worker_is_fresh(info, now, max_heartbeat_age_s) or not isinstance(info, dict):
        return False
    return (
        str(info.get("health_state") or "").lower() == "healthy"
        and str(info.get("accepting_tasks") or "").lower() == "true"
    )


def _worker_is_quarantined(info: object) -> bool:
    if not isinstance(info, dict):
        return False
    return (
        str(info.get("health_state") or "").lower() == "quarantined"
        or str(info.get("quarantine_state") or "").lower() == "quarantined"
    )


def _worker_node_aliases(info: object) -> set[str]:
    if not isinstance(info, dict):
        return set()
    return {str(info.get(field) or "").strip() for field in ("hostname", "node_id")} - {""}


def _expected_worker_is_missing(info: object) -> bool:
    return isinstance(info, dict) and info.get("_expected_missing") is True


def _gpu_display_status(info: object, now: datetime, max_heartbeat_age_s: int) -> str:
    """Collapse internal worker state into the three operator-facing states."""

    if _worker_is_quarantined(info):
        return "quarantine"
    if not isinstance(info, dict):
        return "checking"
    if (
        _worker_is_fresh(info, now, max_heartbeat_age_s)
        and str(info.get("health_state") or "").lower() == "healthy"
        and str(info.get("accepting_tasks") or "").lower() == "true"
    ):
        return "online"
    return "checking"


def render_summary(
    base: str,
    health: dict,
    workers: dict,
    max_heartbeat_age_s: int,
    queue_status: dict | None = None,
    expected_workers: dict[str, object] | None = None,
    quarantine_checked: bool = True,
) -> int:
    """Print vertical key:value summary; return the process exit code."""
    status = health.get("status", "?")
    gpus = health.get("gpu_status", {}) or {}
    # /health.gpu_status entries are dicts on a healthy node but degrade to a
    # plain error string when torch.cuda fails to query the device. Guard the
    # .get() call so a degraded node still renders instead of crashing.
    gpus_ok = sum(1 for v in gpus.values() if isinstance(v, dict) and v.get("available"))
    queue_raw = queue_status or health.get("queue_status", {}) or {}
    queue = queue_raw if isinstance(queue_raw, dict) else {}
    now = _parse_timestamp(health.get("timestamp")) or datetime.now()
    gpu_workers = {worker_id: info for worker_id, info in workers.items() if _is_gpu_worker(worker_id, info)}
    cpu_workers = {worker_id: info for worker_id, info in workers.items() if _is_cpu_worker(worker_id, info)}
    online = sum(1 for w in gpu_workers.values() if isinstance(w, dict) and w.get("status") == "online")
    fresh = sum(1 for w in gpu_workers.values() if _worker_is_fresh(w, now, max_heartbeat_age_s))
    ready = sum(1 for w in gpu_workers.values() if _gpu_worker_is_ready(w, now, max_heartbeat_age_s))
    quarantined = sum(1 for w in gpu_workers.values() if _worker_is_quarantined(w))
    stale = len(gpu_workers) - fresh
    cpu_online = sum(1 for w in cpu_workers.values() if isinstance(w, dict) and w.get("status") == "online")
    cpu_fresh = sum(1 for w in cpu_workers.values() if _worker_is_fresh(w, now, max_heartbeat_age_s))
    cpu_stale = len(cpu_workers) - cpu_fresh
    expected_contract_ok = None if expected_workers is None else _expected_worker_contract_is_valid(expected_workers)
    expected_cpu_ok = expected_workers is None or cpu_stale == 0
    quarantine_contract_ok = expected_workers is None or quarantine_checked
    # Readiness is a worker property, not an expected-contract property. Keep
    # enforcing it when Redis is unavailable and workers came from the API.
    expected_gpu_ok = ready == len(gpu_workers)
    node_ok = (
        status == "healthy"
        and len(gpu_workers) > 0
        and stale == 0
        and expected_contract_ok is not False
        and expected_cpu_ok
        and expected_gpu_ok
        and quarantine_contract_ok
        and quarantined == 0
    )
    top = "UP" if node_ok else "WARN"
    processing_by_resource = queue.get("processing_by_resource") or {}
    gpu_busy = _busy_count(gpu_workers)
    cpu_workers_busy = sum(1 for info in cpu_workers.values() if _worker_is_busy(info))
    if gpu_busy == "unknown" and isinstance(processing_by_resource, dict):
        gpu_busy = processing_by_resource.get("gpu", "unknown")
    fields: list[tuple[str, object]] = [
        ("status", top),
        ("url", base),
        ("api_status", status),
        ("gpus_available", f"{gpus_ok}/{len(gpus)}"),
        ("gpu_workers_online", f"{online}/{len(gpu_workers)}"),
        ("gpu_workers_fresh", f"{fresh}/{len(gpu_workers)}"),
        ("gpu_workers_ready", f"{ready}/{len(gpu_workers)}"),
        ("gpu_workers_quarantined", quarantined if quarantine_checked or quarantined else "unknown"),
        (
            "quarantine_scan",
            "not_checked" if expected_workers is None else ("complete" if quarantine_checked else "incomplete"),
        ),
        ("stale_gpu_workers", stale),
        ("cpu_workers_online", f"{cpu_online}/{len(cpu_workers)}"),
        ("cpu_workers_fresh", f"{cpu_fresh}/{len(cpu_workers)}"),
        ("stale_cpu_workers", cpu_stale),
        (
            "expected_worker_contract",
            "not_checked"
            if expected_contract_ok is None
            else ("ok" if expected_contract_ok else "missing_or_invalid"),
        ),
        ("max_heartbeat_age_s", max_heartbeat_age_s),
        ("queue_processing", queue.get("processing", 0)),
        (
            "queue_gpu_pending",
            (queue.get("resource_queues") or {}).get("gpu", "?")
            if isinstance(queue.get("resource_queues"), dict)
            else "?",
        ),
        (
            "queue_cpu_pending",
            (queue.get("resource_queues") or {}).get("cpu", "?")
            if isinstance(queue.get("resource_queues"), dict)
            else "?",
        ),
        ("gpu_busy", gpu_busy),
        ("cpu_workers_busy", f"{cpu_workers_busy}/{len(cpu_workers)}"),
    ]
    label_w = max(len(k) for k, _ in fields) + 1
    for key, value in fields:
        print(f"{key + ':':<{label_w + 1}}{value}")
    return 0 if node_ok else 1


def render_verbose(health: dict, workers: dict, max_heartbeat_age_s: int) -> None:
    """Render `/health.gpu_status` and `/workers/status` as ASCII tables."""
    now = _parse_timestamp(health.get("timestamp")) or datetime.now()
    gpus = health.get("gpu_status", {}) or {}
    active_node_aliases: set[str] = set()
    for info in workers.values():
        if _expected_worker_is_missing(info):
            continue
        if _worker_is_fresh(info, now, max_heartbeat_age_s):
            active_node_aliases.update(_worker_node_aliases(info))
    gpu_workers = [
        (wid, info)
        for wid, info in workers.items()
        if isinstance(info, dict)
        and _is_gpu_worker(wid, info)
        and info.get("device")
        and not (
            _expected_worker_is_missing(info)
            and _worker_node_aliases(info)
            and not (_worker_node_aliases(info) & active_node_aliases)
            and not _worker_is_quarantined(info)
        )
    ]
    gpu_rows: list[list[str]] = []
    quarantine_rows: list[list[str]] = []

    def add_gpu_row(dev: str, worker_id: str = "", worker_info: dict | None = None) -> None:
        raw = gpus.get(dev)
        if isinstance(raw, dict):
            memory_used = raw.get("memory_used")
            memory_display = str(memory_used) if memory_used not in (None, "", "?") else "?"
            name = _short_gpu_name(raw.get("name", "?"))
        else:
            # Degraded entries (typically an error string from the server when
            # torch.cuda can't query the device) — surface as ERROR with the
            # raw message in the name column so it's still visible.
            name = _short_gpu_name(raw) if raw is not None else "?"
            memory_display = "?"

        display_status = "checking"
        worker_age = ""
        current_task = ""
        node_id = ""
        hostname = ""
        if worker_info:
            age_s = _worker_heartbeat_age_s(worker_info, now)
            display_status = _gpu_display_status(worker_info, now, max_heartbeat_age_s)
            worker_age = "?" if age_s is None else f"{age_s:.0f}"
            current_task = _short_task_id(worker_info.get("current_task", ""))
            node_id = str(worker_info.get("node_id", ""))
            hostname = _short_hostname(worker_info.get("hostname", ""))
            if _worker_is_quarantined(worker_info):
                quarantine_rows.append(
                    [
                        worker_id,
                        dev,
                        str(worker_info.get("quarantine_scope") or "unknown"),
                        str(worker_info.get("quarantine_fault_class") or ""),
                        _short_reason(worker_info.get("quarantine_reason") or worker_info.get("health_reason") or ""),
                        str(worker_info.get("hostname") or ""),
                    ]
                )

        gpu_rows.append(
            [
                dev,
                name,
                memory_display,
                display_status,
                worker_age,
                current_task,
                node_id,
                hostname,
            ]
        )

    def gpu_worker_sort_key(item: tuple[str, dict]) -> tuple[str, tuple[int, str], str]:
        worker_id, info = item
        node = str(info.get("hostname") or info.get("node_id") or "")
        return (node, _device_key(str(info.get("device", ""))), worker_id)

    worker_devices: set[str] = set()
    for worker_id, worker_info in sorted(gpu_workers, key=gpu_worker_sort_key):
        dev = str(worker_info.get("device"))
        worker_devices.add(dev)
        add_gpu_row(dev, worker_id, worker_info)

    for dev in sorted(set(gpus) - worker_devices, key=_device_key):
        add_gpu_row(dev)

    render_table(
        "gpu status",
        [
            "device",
            "name",
            "memory_used",
            "status",
            "age_s",
            "current_task",
            "node_id",
            "hostname",
        ],
        gpu_rows,
    )
    if quarantine_rows:
        render_table(
            "gpu quarantine",
            ["worker_id", "device", "scope", "fault_class", "reason", "hostname"],
            quarantine_rows,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format a reward-node check report from reward-node status JSON on stdin.",
    )
    parser.add_argument("--base", required=True, help="API base URL for display only")
    parser.add_argument("--verbose", action="store_true", help="Also render ASCII tables")
    parser.add_argument(
        "--max-heartbeat-age",
        type=int,
        default=DEFAULT_MAX_HEARTBEAT_AGE_S,
        help="Maximum accepted GPU worker heartbeat age in seconds",
    )
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host for direct in-container status reads")
    parser.add_argument(
        "--redis-port", type=int, default=20110, help="Redis port for direct in-container status reads"
    )
    parser.add_argument("--redis-prefix", default="kernelgym", help="Redis key prefix")
    parser.add_argument("--no-redis", action="store_true", help="Disable direct Redis status reads")
    parser.add_argument("--no-gpu-smi", action="store_true", help="Disable direct nvidia-smi GPU status reads")
    args = parser.parse_args()

    payload = json.load(sys.stdin)
    health = payload.get("health") or {}
    queue = payload.get("queue") or None
    workers = payload.get("workers") or {}
    # When Redis contract checks are enabled, fail closed until a valid
    # snapshot is read. Only explicit --no-redis permits an unchecked report.
    expected_workers: dict[str, object] | None = None if args.no_redis else {}
    quarantine_checked = False

    if not args.no_redis:
        redis_status = _redis_snapshot(args.redis_host, args.redis_port, args.redis_prefix)
        if redis_status:
            redis_workers = redis_status.get("workers")
            if isinstance(redis_workers, dict):
                workers = _merge_workers(workers, redis_workers)
            redis_expected_workers = redis_status.get("expected_workers")
            if isinstance(redis_expected_workers, dict):
                expected_workers = redis_expected_workers
                workers = _merge_expected_workers(workers, expected_workers)
            redis_quarantines = redis_status.get("quarantines")
            if isinstance(redis_quarantines, list):
                workers = _merge_quarantines(workers, redis_quarantines)
            quarantine_checked = redis_status.get("quarantine_checked") is True
            redis_queue = redis_status.get("queue")
            if isinstance(redis_queue, dict):
                queue = redis_queue
    if not args.no_gpu_smi:
        health = _merge_gpu_status(health, _nvidia_smi_gpu_snapshot())

    exit_code = render_summary(
        args.base,
        health,
        workers,
        args.max_heartbeat_age,
        queue_status=queue,
        expected_workers=expected_workers,
        quarantine_checked=quarantine_checked,
    )
    if args.verbose:
        render_verbose(health, workers, args.max_heartbeat_age)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
