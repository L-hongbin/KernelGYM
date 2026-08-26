"""Service management CLI for reward-only KernelGym.

The repository keeps shell entrypoints for compatibility, but operational logic
lives here so it can be tested and maintained as Python code.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kernelgym.deployment_profiles import (
    API_PORT,
    API_RELOAD,
    API_WORKERS,
    METRICS_PORT,
    REDIS_DB,
    REDIS_KEY_PREFIX,
    REDIS_PASSWORD,
    REDIS_PORT,
    bool_env,
    get_profile,
    profile_names,
)
from kernelgym.utils.device_info import DEVICE_INFO_ENV, detect_device_info, encode_device_info
from kernelgym.utils.core_dumps import (
    CORE_DUMP_DIR_ENV,
    CORE_DUMP_KEEP_ENV,
    DEFAULT_CORE_DUMP_DIR,
    DEFAULT_CORE_DUMP_KEEP,
    ensure_core_dump_dir,
    prune_core_dumps,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
TORCH_CUDA_ARCH_LIST_ENV = "TORCH_CUDA_ARCH_LIST"
GPU_DEVICES_AUTO = "auto"
_CUDA_ARCH_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_GPU_DEVICE_INDEX_PATTERN = re.compile(r"^[0-9]+$")
_PROCESS_GROUP_KILL_GRACE_SECONDS = 10.0


@dataclass(frozen=True)
class _ProcessIdentity:
    """Immutable-enough Linux identity used to fence PID/session reuse."""

    pid: int
    start_ticks: str
    state: str
    process_group: int
    session_id: int


_LAUNCHED_IDENTITIES: dict[int, _ProcessIdentity] = {}


_REGISTER_EXPECTED_WORKER_IF_EMPTY = """
local current_pid = redis.call('HGET', KEYS[1], 'pid')
if current_pid and current_pid ~= '' then
    return 0
end
redis.call('DEL', KEYS[1])
redis.call(
    'HSET', KEYS[1],
    'pid', ARGV[1],
    'start_time', ARGV[2],
    'proc_start_ticks', ARGV[3],
    'process_group', ARGV[4],
    'session_id', ARGV[5],
    'device', ARGV[6]
)
redis.call('DEL', KEYS[2])
redis.call(
    'HSET', KEYS[2],
    'device', ARGV[6],
    'hostname', ARGV[7],
    'node_id', ARGV[8],
    'pid', ARGV[1],
    'proc_start_ticks', ARGV[3],
    'session_id', ARGV[5]
)
redis.call('SADD', KEYS[3], ARGV[9])
return 1
"""


_DELETE_WORKER_REGISTRATION_IF_CURRENT = """
local current_pid = redis.call('HGET', KEYS[1], 'pid')
if not current_pid or current_pid ~= ARGV[1] then
    return 0
end
local current_ticks = redis.call('HGET', KEYS[1], 'proc_start_ticks')
if ARGV[5] == '1' then
    if not current_ticks or current_ticks ~= ARGV[2] then
        return 0
    end
else
    if current_ticks and current_ticks ~= '' then
        return 0
    end
end
local current_group = redis.call('HGET', KEYS[1], 'process_group')
if ARGV[6] == '1' then
    if not current_group or current_group ~= ARGV[3] then
        return 0
    end
else
    if current_group and current_group ~= '' then
        return 0
    end
end
local current_session = redis.call('HGET', KEYS[1], 'session_id')
if ARGV[7] == '1' then
    if not current_session or current_session ~= ARGV[4] then
        return 0
    end
else
    if current_session and current_session ~= '' then
        return 0
    end
end
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
redis.call('SREM', KEYS[3], ARGV[8])
return 1
"""


def _hostname() -> str:
    # Workers publish socket.gethostname() in their Redis/API registrations.
    # Do not use the inherited HOSTNAME environment variable as an ownership
    # identity: a stale or poisoned value could name a different cluster node.
    return socket.gethostname() or "local"


def _profile_values(profile_name: str) -> dict[str, str]:
    try:
        return get_profile(profile_name).env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _worker_profile_values(profile_name: str, master_addr: str, node_rank: str | None = None) -> dict[str, str]:
    values = _profile_values(profile_name)
    base_node_id = values["NODE_ID"]
    if node_rank is not None:
        node_id = f"{base_node_id}-worker-{node_rank}"
    else:
        # No rank given (e.g. `deploy_node.sh --join <primary>`): leave NODE_ID empty
        # so cmd_start_worker_node omits node_name and the server auto-allocates a
        # stable per-hostname node id. This makes the node count fully dynamic — no
        # rank/nnodes ceremony — while staying idempotent across re-joins of a host.
        node_id = ""
    log_base = node_id or base_node_id
    values.update(
        {
            "API_HOST": master_addr,
            "REDIS_HOST": master_addr,
            "NODE_ID": node_id,
            "WORKER_NAME_PREFIX": node_id,
            "LOG_DIR": f"logs/{log_base}-worker",
            "PY_LOG_DIR": f"py_logs/{log_base}-worker",
        }
    )
    if node_rank is not None:
        values["KERNELGYM_NODE_RANK"] = str(node_rank)
    return values


def _with_hostname_log_dirs(values: dict[str, str]) -> dict[str, str]:
    """Nest log output under a per-host subdirectory.

    Nodes share this repo over NFS, so writing straight to ``logs/<profile>``
    makes every host clobber the same files. Each host instead gets its own
    ``logs/<profile>/<hostname>/`` subtree. Idempotent: paths already ending in
    the current hostname are left untouched.
    """
    host = _hostname()
    updated = dict(values)
    for key in ("LOG_DIR", "PY_LOG_DIR", CORE_DUMP_DIR_ENV):
        base = updated.get(key)
        if base and Path(base).name != host:
            updated[key] = str(Path(base) / host)
    eval_path = updated.get("EVAL_RESULTS_PATH")
    if eval_path:
        path = Path(eval_path)
        if path.parent.name != host:
            updated["EVAL_RESULTS_PATH"] = str(path.parent / host / path.name)
    return updated


def _default_env_file() -> Path:
    host_env = ROOT_DIR / f".env.{_hostname()}"
    return host_env if host_env.exists() else ROOT_DIR / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _format_torch_cuda_arch_list(values: list[str]) -> str:
    arches: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        for item in re.split(r"[;\s,]+", raw_value.strip().strip('"').strip("'")):
            arch = item.strip()
            if not arch or not _CUDA_ARCH_PATTERN.match(arch) or arch in seen:
                continue
            seen.add(arch)
            arches.append(arch)
    return ";".join(arches)


def _detect_torch_cuda_arch_list_with_torch() -> str:
    try:
        import torch
    except Exception:
        return ""
    try:
        if not torch.cuda.is_available():
            return ""
        arches = []
        for device_index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_index)
            arches.append(f"{major}.{minor}")
        return _format_torch_cuda_arch_list(arches)
    except Exception:
        return ""


def _detect_torch_cuda_arch_list_with_nvidia_smi() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    for query_field in ("compute_cap", "compute_capability"):
        proc = subprocess.run(
            [nvidia_smi, f"--query-gpu={query_field}", "--format=csv,noheader"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            arch_list = _format_torch_cuda_arch_list(proc.stdout.splitlines())
            if arch_list:
                return arch_list
    return ""


def _detect_visible_torch_cuda_arch_list() -> str:
    return _detect_torch_cuda_arch_list_with_torch() or _detect_torch_cuda_arch_list_with_nvidia_smi()


def _with_torch_cuda_arch_list(values: dict[str, str]) -> dict[str, str]:
    if values.get(TORCH_CUDA_ARCH_LIST_ENV):
        return values
    configured = os.environ.get(TORCH_CUDA_ARCH_LIST_ENV, "").strip()
    arch_list = configured or _detect_visible_torch_cuda_arch_list()
    if not arch_list:
        return values
    updated = dict(values)
    updated[TORCH_CUDA_ARCH_LIST_ENV] = arch_list
    return updated


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    groups = [
        (
            "Deployment",
            (
                "KERNELGYM_DEPLOYMENT_PROFILE",
                "KERNELGYM_SSH_RUNTIME",
                "KERNELGYM_CONTAINER_REQUIRED",
                "KERNELGYM_LOCK_GPU_CLOCKS",
            ),
        ),
        ("Network", ("API_HOST", "API_PORT", "API_WORKERS", "API_RELOAD")),
        ("GPU", ("GPU_DEVICES", "NODE_ID", "KERNELGYM_DEVICE_INFO")),
        (
            "Redis",
            (
                "REDIS_HOST",
                "REDIS_PORT",
                "REDIS_DB",
                "REDIS_PASSWORD",
                "REDIS_KEY_PREFIX",
                "KERNELGYM_REDIS_REMOTE_ACCESS",
            ),
        ),
        (
            "Worker pool",
            (
                "WORKER_POOL_SIZE",
                "MAX_TASKS_PER_WORKER",
                "KERNELGYM_WORKER_SPAWN_CONCURRENCY",
                "KERNELGYM_WORKER_SPAWN_SLOT_TIMEOUT",
                "KERNELGYM_WORKER_CONTAINMENT_TIMEOUT",
                "KERNELGYM_WORKER_READY_TIMEOUT",
                "CPU_COMPILE_WORKERS",
            ),
        ),
        ("Defaults", ("DEFAULT_TOOLKIT", "DEFAULT_BACKEND_ADAPTER", "DEFAULT_BACKEND")),
        ("Logging", ("LOG_LEVEL", "LOG_DIR", "PY_LOG_DIR")),
        ("Core dumps", ("KERNELGYM_CORE_DUMP_DIR", "KERNELGYM_CORE_DUMP_KEEP")),
        ("Metrics", ("ENABLE_METRICS", "METRICS_PORT")),
        ("Profiling", ("ENABLE_PROFILING",)),
        ("Errors", ("VERBOSE_ERROR_TRACEBACK",)),
        ("Result persistence", ("SAVE_EVAL_RESULTS", "EVAL_RESULTS_PATH")),
        (
            "CUDA build",
            (
                "TORCH_CUDA_ARCH_LIST",
                "KERNELGYM_NVCC_THREADS",
                "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE",
                "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE_INDEX",
                "KERNELGYM_COMPILE_ARTIFACT_CACHE",
            ),
        ),
    ]
    emitted: set[str] = set()
    lines = ["# KernelGym reward-only configuration", f"# Generated on: {time.ctime()}"]
    for title, keys in groups:
        lines.extend(["", f"# {title}"])
        for key in keys:
            if key in values:
                lines.append(f"{key}={values[key]}")
                emitted.add(key)
    extra_keys = sorted(set(values) - emitted)
    if extra_keys:
        lines.extend(["", "# Extra"])
        lines.extend(f"{key}={values[key]}" for key in extra_keys)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_runtime_overrides(values: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    updated = dict(values)
    gpu_devices = getattr(args, "gpu_devices", None)
    if gpu_devices is not None:
        updated["GPU_DEVICES"] = str(gpu_devices)
    cpu_compile_workers = getattr(args, "cpu_compile_workers", None)
    if cpu_compile_workers is not None:
        if cpu_compile_workers < 0:
            raise SystemExit("--cpu-compile-workers must be >= 0")
        updated["CPU_COMPILE_WORKERS"] = str(cpu_compile_workers)
    if getattr(args, "redis_remote_access", False):
        updated["KERNELGYM_REDIS_REMOTE_ACCESS"] = "true"
    return updated


def _with_device_info(values: dict[str, str]) -> dict[str, str]:
    if values.get(DEVICE_INFO_ENV):
        return values
    configured = os.environ.get(DEVICE_INFO_ENV, "").strip()
    updated = dict(values)
    updated[DEVICE_INFO_ENV] = configured or encode_device_info(detect_device_info())
    return updated


def _update_env_file(path: Path, updates: dict[str, str]) -> None:
    values = _read_env_file(path)
    values.update(updates)
    _write_env_file(path, values)


def _parse_gpu_devices(raw: str | None) -> list[str]:
    value = str(raw or "").strip()
    if not value or value.lower() == GPU_DEVICES_AUTO:
        raise ValueError("GPU device selection must be resolved before workers are launched")
    try:
        parsed = json.loads(value)
        items = parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        items = [item.strip() for item in value.split(",") if item.strip()]

    devices: list[str] = []
    for item in items:
        text = str(item).strip()
        if isinstance(item, bool) or not _GPU_DEVICE_INDEX_PATTERN.fullmatch(text):
            raise ValueError(f"invalid CUDA device index: {item!r}")
        normalized = str(int(text))
        if normalized in devices:
            raise ValueError(f"duplicate CUDA device index: {normalized}")
        devices.append(normalized)
    return devices


def _detect_visible_gpu_count_with_torch() -> int | None:
    try:
        import torch
    except Exception:
        return None
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return None


def _detect_visible_gpu_count() -> int:
    count = _detect_visible_gpu_count_with_torch()
    if count is None:
        # nvidia-smi is deliberately not a fallback: it may ignore a
        # CUDA_VISIBLE_DEVICES-only restriction and report host-wide devices.
        # PyTorch is the runtime that will execute tasks, so its logical CUDA
        # namespace is the authoritative worker namespace.
        raise SystemExit("Cannot detect container-visible CUDA devices with the configured PyTorch runtime")
    if count <= 0:
        raise SystemExit(
            "No CUDA devices are visible to PyTorch inside this container; "
            "check container GPU passthrough, CUDA_VISIBLE_DEVICES, and driver health"
        )
    return count


def _resolve_gpu_devices(values: dict[str, str]) -> dict[str, str]:
    """Resolve auto or validate an explicit logical CUDA device list."""
    configured = values.get("GPU_DEVICES")
    raw = GPU_DEVICES_AUTO if configured is None else str(configured).strip()
    if not raw:
        raise SystemExit("GPU_DEVICES cannot be blank; use 'auto' or at least one CUDA device index")
    automatic = raw.lower() == GPU_DEVICES_AUTO
    if automatic:
        visible_count = _detect_visible_gpu_count()
        devices = [str(index) for index in range(visible_count)]
        source = "auto"
    else:
        try:
            devices = _parse_gpu_devices(raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid GPU_DEVICES={raw!r}: {exc}") from exc
        if not devices:
            raise SystemExit("GPU_DEVICES cannot be empty; this deployment requires at least one visible GPU")
        visible_count = _detect_visible_gpu_count()
        out_of_range = [device for device in devices if int(device) >= visible_count]
        if out_of_range:
            raise SystemExit(
                f"GPU_DEVICES selects unavailable logical device(s) {out_of_range}; "
                f"container-visible range is 0..{visible_count - 1}"
            )
        source = "explicit"

    updated = dict(values)
    updated["GPU_DEVICES"] = json.dumps([int(device) for device in devices], separators=(",", ":"))
    print(f"GPU device selection: source={source} visible_count={visible_count} devices={updated['GPU_DEVICES']}")
    return updated


def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _with_cupti_tsc_shim(env: dict[str, str]) -> dict[str, str]:
    """Inject the CUPTI TSC shim when KERNELGYM_CUPTI_TSC_SHIM is enabled.

    Builds the LD_PRELOAD shim and, on success, declares the Kineto TSC fix so
    profiling drops to a single forward (see
    docs/design-doc/PROFILER_EMPTY_CAPTURE.md). On any failure nothing is
    injected: the service starts normally and the legacy multi-forward
    profiling workaround stays active (fail-open).
    """
    from kernelgym.utils import cupti_tsc_shim

    # The operator's ambient environment wins over the profile default so
    # `export KERNELGYM_CUPTI_TSC_SHIM=false` is a working emergency off switch.
    flag_value = os.environ.get(cupti_tsc_shim.SHIM_FLAG_ENV) or env.get(cupti_tsc_shim.SHIM_FLAG_ENV) or ""
    if flag_value.strip().lower() not in {"1", "true", "yes", "on"}:
        return env
    shim_path = cupti_tsc_shim.ensure_shim_built()
    if shim_path is None:
        print(
            "WARNING: KERNELGYM_CUPTI_TSC_SHIM=true but the shim build failed; "
            "keeping the legacy multi-forward profiling workaround"
        )
        return env
    preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{shim_path}:{preload}" if preload else str(shim_path)
    env["KINETO_TSC_FIXED"] = "true"
    env[cupti_tsc_shim.SHIM_EXPECTED_ENV] = str(shim_path)
    print(f"CUPTI TSC shim enabled: {shim_path}")
    return env


def _service_env(values: dict[str, str]) -> dict[str, str]:
    values = _with_torch_cuda_arch_list(values)
    values = _with_device_info(values)
    env = os.environ.copy()
    env.update(values)
    env = _with_cupti_tsc_shim(env)
    env["API_PORT"] = str(API_PORT)
    env["API_WORKERS"] = str(API_WORKERS)
    env["API_RELOAD"] = bool_env(API_RELOAD)
    env["REDIS_PORT"] = str(REDIS_PORT)
    env["REDIS_DB"] = str(REDIS_DB)
    env["REDIS_PASSWORD"] = REDIS_PASSWORD
    env["REDIS_KEY_PREFIX"] = REDIS_KEY_PREFIX
    env["METRICS_PORT"] = str(METRICS_PORT)
    env.setdefault(CORE_DUMP_DIR_ENV, str(Path(DEFAULT_CORE_DUMP_DIR) / _hostname()))
    env.setdefault(CORE_DUMP_KEEP_ENV, str(DEFAULT_CORE_DUMP_KEEP))
    for directory_key in ("LOG_DIR", "PY_LOG_DIR", CORE_DUMP_DIR_ENV):
        if env.get(directory_key):
            path = Path(env[directory_key]).expanduser()
            if not path.is_absolute():
                path = ROOT_DIR / path
            env[directory_key] = str(path)
    if env.get("EVAL_RESULTS_PATH"):
        path = Path(env["EVAL_RESULTS_PATH"]).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        env["EVAL_RESULTS_PATH"] = str(path)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT_DIR) if not pythonpath else f"{ROOT_DIR}:{pythonpath}"
    return env


def _reap_exact_launch_handle(process: Any) -> tuple[bool, str]:
    """Best-effort rollback using the exact Popen child, never a bare PID scope."""

    try:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=_PROCESS_GROUP_KILL_GRACE_SECONDS)
        if process.poll() is None:
            return False, "Popen child remained live after SIGKILL"
        return True, ""
    except Exception as exc:
        return False, f"exact Popen child rollback failed: {type(exc).__name__}"


def _launch_background(command: list[str], log_file: Path, env: dict[str, str]) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    core_dir = ROOT_DIR
    try:
        core_dir = ensure_core_dump_dir(env.get(CORE_DUMP_DIR_ENV))
        prune_core_dumps(core_dir, env.get(CORE_DUMP_KEEP_ENV))
    except Exception:
        core_dir = ROOT_DIR
    handle = log_file.open("ab")
    proc = subprocess.Popen(
        command,
        cwd=core_dir,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    pid = int(proc.pid)
    try:
        identity = _read_process_identity(pid)
    except Exception as exc:
        reaped, reap_reason = _reap_exact_launch_handle(proc)
        worker_id = _command_option(command, "--worker-id")
        device = _command_option(command, "--device")
        if worker_id and device.startswith("cuda:"):
            _quarantine_unsafe_worker_group(
                None,
                worker_id,
                device,
                "launch identity read failed before session authentication; "
                f"exact leader reaped={reaped}: {reap_reason or 'descendant scope is unproven'}",
            )
        raise RuntimeError(f"Could not record launch identity for background PID {pid}") from exc
    if identity is None:
        reaped, reap_reason = _reap_exact_launch_handle(proc)
        worker_id = _command_option(command, "--worker-id")
        device = _command_option(command, "--device")
        if worker_id and device.startswith("cuda:"):
            _quarantine_unsafe_worker_group(
                None,
                worker_id,
                device,
                "background leader disappeared before session authentication; "
                f"exact leader reaped={reaped}: {reap_reason or 'descendant scope is unproven'}",
            )
        raise RuntimeError(f"Background process PID {pid} exited before its launch identity was recorded")
    if identity.process_group != pid or identity.session_id != pid:
        reaped, reap_reason = _reap_exact_launch_handle(proc)
        worker_id = _command_option(command, "--worker-id")
        device = _command_option(command, "--device")
        if worker_id and device.startswith("cuda:"):
            _quarantine_unsafe_worker_group(
                None,
                worker_id,
                device,
                f"new background PID {pid} did not establish its own session; "
                f"exact leader reaped={reaped}: {reap_reason or 'descendant scope is unproven'}",
            )
        raise RuntimeError(f"Launched PID {pid} without a provable new session")
    if identity.state == "Z":
        drained, drain_reason = _force_kill_worker_session(
            pid,
            expected_leader_start_ticks=identity.start_ticks,
            observed_process_groups={pid},
        )
        if not drained:
            worker_id = _command_option(command, "--worker-id")
            device = _command_option(command, "--device")
            if worker_id and device.startswith("cuda:"):
                _quarantine_unsafe_worker_group(
                    None,
                    worker_id,
                    device,
                    f"new background session {pid} could not be authenticated or drained: {drain_reason}",
                )
            raise RuntimeError(f"Launched PID {pid} without a provable new session")
        raise RuntimeError(f"Background process PID {pid} exited before its launch identity was recorded")
    _LAUNCHED_IDENTITIES[pid] = identity
    return pid


def _redis_client(values: dict[str, str]) -> Any | None:
    try:
        import redis
    except Exception:
        return None
    return redis.Redis(
        host=values.get("REDIS_HOST", "localhost"),
        port=REDIS_PORT,
        db=REDIS_DB,
        password=None,
        decode_responses=True,
    )


def _ensure_redis(values: dict[str, str]) -> None:
    host = values.get("REDIS_HOST", "localhost")
    port = REDIS_PORT
    remote_access = str(values.get("KERNELGYM_REDIS_REMOTE_ACCESS", "")).strip().lower() in {"1", "true", "yes", "on"}
    if _port_is_open(host, port):
        if remote_access:
            _configure_redis_remote_access(values)
        return
    if host not in {"localhost", "127.0.0.1"}:
        raise SystemExit(f"Redis is not reachable at {host}:{port}. Start it before launching workers.")
    redis_server = shutil.which("redis-server")
    if not redis_server:
        raise SystemExit("redis-server not found; install Redis or set REDIS_HOST/REDIS_PORT to an existing server.")
    # Persist to a NODE-LOCAL data dir, never the launch cwd. Deployments share
    # this checkout over NFS; with no explicit --dir, redis defaults its dir to
    # the cwd, so every node would read/write the same dump.rdb and cross-load
    # each other's data on restart. A per-node dir (and node-tagged dbfilename)
    # keeps each deployment's Redis isolated.
    data_dir = values.get("KERNELGYM_REDIS_DATA_DIR") or "/tmp/kernelgym-redis"
    node_tag = (values.get("NODE_ID") or socket.gethostname() or "node").strip() or "node"
    dbfilename = f"dump-{node_tag}-{port}.rdb"
    os.makedirs(data_dir, exist_ok=True)
    command = [
        redis_server,
        "--port",
        str(port),
        "--daemonize",
        "yes",
        "--dir",
        data_dir,
        "--dbfilename",
        dbfilename,
    ]
    if remote_access:
        command.extend(["--bind", "0.0.0.0", "--protected-mode", "no"])
    subprocess.run(command, check=True)
    time.sleep(1)
    if remote_access:
        _configure_redis_remote_access(values)


def _configure_redis_remote_access(values: dict[str, str]) -> None:
    client = _redis_client(values)
    if client is None:
        raise SystemExit("redis Python package is required to configure multi-node Redis access.")
    try:
        client.config_set("protected-mode", "no")
    except Exception as exc:
        raise SystemExit(f"Failed to disable Redis protected-mode for multi-node access: {exc}") from exc
    try:
        client.config_set("bind", "0.0.0.0")
    except Exception:
        pass


def _api_base(values: dict[str, str]) -> str:
    host = values.get("API_HOST", "127.0.0.1")
    port = str(API_PORT)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _http_get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_post_json(url: str, timeout: float = 5.0) -> Any:
    request = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _decode_redis_hash(data: Any) -> dict[str, str]:
    if not data:
        return {}
    return {
        (key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)): (
            value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        )
        for key, value in data.items()
    }


def _decode_redis_text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _read_process_identity(pid: int) -> _ProcessIdentity | None:
    """Read the Linux generation, state, process group, and session for one PID."""

    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    closing_paren = raw_stat.rfind(")")
    if closing_paren < 0:
        raise RuntimeError(f"Malformed /proc/{pid}/stat")
    fields = raw_stat[closing_paren + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError(f"Incomplete /proc/{pid}/stat")
    return _ProcessIdentity(
        pid=pid,
        state=fields[0],
        process_group=int(fields[2]),
        session_id=int(fields[3]),
        start_ticks=fields[19],
    )


def _read_process_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _cmdline_matches_worker(pid: int, worker_id: str) -> bool:
    argv = _read_process_argv(pid)
    worker_module = any(part in {"kernelgym.worker.single_worker", "kernelgym.worker.cpu_worker"} for part in argv)
    return worker_module and worker_id in argv


def _process_group_is_drained(process_group: int) -> bool:
    """Only kernel ESRCH proves that an old CUDA process group is absent."""

    if process_group <= 1 or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return True
        raise
    return False


def _wait_for_process_group_drain(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if _process_group_is_drained(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _live_session_members(session_id: int) -> list[_ProcessIdentity]:
    """Return one fail-closed snapshot of a Linux session.

    A worker's pool children may lead their own process groups, so probing only
    the outer PGID is insufficient.  Session membership is inherited by an
    ordinary fork and remains stable across ``setpgid``.  Any unreadable
    numeric ``/proc`` entry fails the scan closed; node deployments therefore
    need permission to read process stat records across the container PID
    namespace.
    """

    if session_id <= 1:
        raise RuntimeError(f"Invalid worker session id: {session_id}")
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise RuntimeError(f"Could not enumerate /proc for session {session_id}") from exc

    members: list[_ProcessIdentity] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            identity = _read_process_identity(int(entry.name))
        except OSError as exc:
            raise RuntimeError(f"Could not inspect PID {entry.name} in session {session_id}") from exc
        if identity is not None and identity.session_id == session_id:
            members.append(identity)
    return sorted(members, key=lambda item: (item.pid, item.start_ticks))


def _session_is_drained(session_id: int, observed_process_groups: set[int]) -> bool:
    """Require both an empty SID snapshot and ESRCH for every observed PGID."""

    members = _live_session_members(session_id)
    observed_process_groups.update(member.process_group for member in members)
    if members:
        return False
    return all(_process_group_is_drained(process_group) for process_group in observed_process_groups)


def _wait_for_session_drain(session_id: int, observed_process_groups: set[int], timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    last_scan_error: Exception | None = None
    while True:
        try:
            if _session_is_drained(session_id, observed_process_groups):
                return True
            last_scan_error = None
        except (OSError, RuntimeError, ValueError) as exc:
            # Numeric /proc entries can disappear or become temporarily
            # unreadable while a process exits.  One incomplete snapshot is
            # not a containment failure and is not an absence proof either:
            # retry until a complete scan proves the SID empty or the existing
            # shutdown deadline expires.
            last_scan_error = exc
        if time.monotonic() >= deadline:
            if last_scan_error is not None:
                raise RuntimeError(f"Could not complete session {session_id} drain scan") from last_scan_error
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _freeze_worker_session(
    session_id: int,
    *,
    expected_leader_start_ticks: str,
    timeout: float,
) -> tuple[bool, set[int], str]:
    """Stop a session to a stable fixed point before destructive signalling.

    Stopping only the session leader leaves an existing descendant able to
    fork while ``/proc`` is being scanned.  Repeatedly stop every observed
    PGID and require two identical snapshots whose non-zombie members are all
    stopped.  Once that fixed point is reached, ordinary descendants cannot
    create another process before the subsequent SIGKILL sweep.
    """

    deadline = time.monotonic() + max(0.0, timeout)
    observed_process_groups: set[int] = set()
    previous_signature: tuple[tuple[int, str, int], ...] | None = None
    stable_passes = 0

    while True:
        try:
            members = _live_session_members(session_id)
        except Exception as exc:
            return False, observed_process_groups, f"session /proc scan failed: {type(exc).__name__}"

        for member in members:
            if (
                member.pid == session_id
                and expected_leader_start_ticks
                and member.start_ticks != expected_leader_start_ticks
            ):
                return False, observed_process_groups, f"session leader PID {session_id} generation changed"
            if member.process_group <= 1:
                return False, observed_process_groups, f"invalid PGID {member.process_group} in session {session_id}"
            observed_process_groups.add(member.process_group)

        if not members:
            return True, observed_process_groups, ""

        for process_group in sorted({member.process_group for member in members}):
            try:
                os.killpg(process_group, signal.SIGSTOP)
            except ProcessLookupError:
                continue
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    continue
                return (
                    False,
                    observed_process_groups,
                    f"SIGSTOP failed for session {session_id} PGID {process_group}: {type(exc).__name__}",
                )

        try:
            confirmation = _live_session_members(session_id)
        except Exception as exc:
            return False, observed_process_groups, f"session confirmation scan failed: {type(exc).__name__}"
        observed_process_groups.update(member.process_group for member in confirmation)
        for member in confirmation:
            if (
                member.pid == session_id
                and expected_leader_start_ticks
                and member.start_ticks != expected_leader_start_ticks
            ):
                return False, observed_process_groups, f"session leader PID {session_id} generation changed"

        signature = tuple((member.pid, member.start_ticks, member.process_group) for member in confirmation)
        all_frozen = all(member.state in {"T", "t", "Z"} for member in confirmation)
        if all_frozen and signature == previous_signature:
            stable_passes += 1
        elif all_frozen:
            stable_passes = 1
        else:
            stable_passes = 0
        previous_signature = signature
        if stable_passes >= 2:
            return True, observed_process_groups, ""
        if time.monotonic() >= deadline:
            states = ",".join(f"{member.pid}:{member.state}" for member in confirmation)
            return (
                False,
                observed_process_groups,
                f"session {session_id} did not freeze to a stable snapshot ({states})",
            )
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _force_kill_worker_session(
    session_id: int,
    *,
    expected_leader_start_ticks: str,
    observed_process_groups: set[int] | None = None,
) -> tuple[bool, str]:
    """Freeze, kill, and prove an entire worker session absent."""

    known_groups = set(observed_process_groups or ())
    frozen, frozen_groups, reason = _freeze_worker_session(
        session_id,
        expected_leader_start_ticks=expected_leader_start_ticks,
        timeout=_PROCESS_GROUP_KILL_GRACE_SECONDS,
    )
    known_groups.update(frozen_groups)
    if not frozen:
        return False, reason

    # Only signal groups that the frozen SID snapshot authenticated.  A stale
    # Redis PGID that is no longer in this session may already have been reused
    # elsewhere; it still remains in ``known_groups`` for the final ESRCH gate,
    # but must never be signalled by number alone.
    for process_group in sorted(frozen_groups):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                continue
            return False, f"SIGKILL failed for session {session_id} PGID {process_group}: {type(exc).__name__}"

    try:
        drained = _wait_for_session_drain(session_id, known_groups, _PROCESS_GROUP_KILL_GRACE_SECONDS)
    except Exception as exc:
        return False, f"session drain proof failed: {type(exc).__name__}"
    if drained:
        return True, ""
    return False, f"session {session_id} or one of its observed process groups survived SIGKILL"


def _stop_authenticated_worker_group(
    worker_id: str,
    *,
    pid: int,
    expected_start_ticks: str,
    process_group: int,
    session_id: int | None = None,
    graceful_seconds: float,
) -> tuple[bool, str]:
    """Stop one worker generation and prove its complete session gone."""

    worker_session = session_id if session_id is not None else pid
    if pid <= 1 or process_group <= 1 or worker_session <= 1 or process_group != pid or worker_session != pid:
        return (
            False,
            f"invalid new-session identity pid={pid} process_group={process_group} session_id={worker_session}",
        )
    observed_process_groups = {process_group}

    identity = _read_process_identity(pid)
    if identity is None:
        try:
            if _session_is_drained(worker_session, observed_process_groups):
                return True, ""
        except Exception as exc:
            return False, f"session drain proof failed: {type(exc).__name__}"
        if not expected_start_ticks:
            return False, f"leader PID {pid} is absent but session {worker_session} is not authenticated"
        return _force_kill_worker_session(
            worker_session,
            expected_leader_start_ticks=expected_start_ticks,
            observed_process_groups=observed_process_groups,
        )
    if expected_start_ticks and identity.start_ticks != expected_start_ticks:
        return False, f"PID {pid} generation changed before shutdown"
    if identity.process_group != process_group:
        return False, f"PID {pid} moved from process group {process_group} to {identity.process_group}"
    if identity.session_id != worker_session:
        return False, f"PID {pid} moved from session {worker_session} to {identity.session_id}"
    if identity.state == "Z":
        if not expected_start_ticks:
            return False, f"legacy zombie PID {pid} cannot be generation-authenticated"
    elif not _cmdline_matches_worker(pid, worker_id):
        latest = _read_process_identity(pid)
        if latest is not None and latest.state != "Z":
            return False, f"PID {pid} command line no longer belongs to worker {worker_id}"
        try:
            if _wait_for_session_drain(worker_session, observed_process_groups, graceful_seconds):
                return True, ""
        except Exception as exc:
            return False, f"session drain proof failed: {type(exc).__name__}"
        return _force_kill_worker_session(
            worker_session,
            expected_leader_start_ticks=expected_start_ticks or identity.start_ticks,
            observed_process_groups=observed_process_groups,
        )

    # Legacy maps did not carry start ticks. A live matching worker can be
    # authenticated from /proc; a zombie cannot because its cmdline is empty.
    authenticated_ticks = expected_start_ticks or identity.start_ticks
    current = _read_process_identity(pid)
    if current is None:
        try:
            if _session_is_drained(worker_session, observed_process_groups):
                return True, ""
        except Exception as exc:
            return False, f"session drain proof failed: {type(exc).__name__}"
        # The authenticated leader exited between validation and the final
        # signal fence.  Never signal its bare numeric PGID: it may already be
        # reusable.  Freeze and clean only groups rediscovered as members of
        # the recorded SID, then require the normal complete drain proof.
        return _force_kill_worker_session(
            worker_session,
            expected_leader_start_ticks=authenticated_ticks,
            observed_process_groups=observed_process_groups,
        )
    if (
        current.start_ticks != authenticated_ticks
        or current.process_group != process_group
        or current.session_id != worker_session
        or (current.state != "Z" and not _cmdline_matches_worker(pid, worker_id))
    ):
        latest = _read_process_identity(pid)
        if latest is not None and latest.state != "Z":
            return False, f"PID {pid} generation changed immediately before SIGTERM"
        try:
            if _wait_for_session_drain(worker_session, observed_process_groups, graceful_seconds):
                return True, ""
        except Exception as exc:
            return False, f"session drain proof failed: {type(exc).__name__}"
        return _force_kill_worker_session(
            worker_session,
            expected_leader_start_ticks=authenticated_ticks,
            observed_process_groups=observed_process_groups,
        )

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            return False, f"SIGTERM failed for process group {process_group}: {type(exc).__name__}"
    try:
        if _wait_for_session_drain(worker_session, observed_process_groups, graceful_seconds):
            return True, ""
    except Exception as exc:
        return False, f"session drain proof failed: {type(exc).__name__}"

    return _force_kill_worker_session(
        worker_session,
        expected_leader_start_ticks=authenticated_ticks,
        observed_process_groups=observed_process_groups,
    )


class _AsyncRedisFacade:
    """Narrow awaitable wrapper used by the synchronous service command."""

    def __init__(self, client: Any | None):
        self._client = client

    async def hset(self, *args: Any, **kwargs: Any) -> Any:
        if self._client is None:
            raise RuntimeError("Redis is unavailable")
        return self._client.hset(*args, **kwargs)


def _quarantine_unsafe_worker_group(client: Any | None, worker_id: str, device: str, reason: str) -> None:
    """Persist a physical latch and page for an uncontained GPU process group."""

    if not device.startswith("cuda:"):
        print(f"ERROR: worker {worker_id} containment is unproven: {reason}")
        return

    from kernelgym.utils.gpu_quarantine import (
        UNLATCHED_NOTIFICATION_PROVENANCE,
        gpu_quarantine_generation,
        update_gpu_quarantine_notification,
        write_gpu_quarantine,
    )
    from kernelgym.utils.page_user_notifier import send_gpu_quarantine_page

    async def persist_and_page() -> None:
        redis_facade = _AsyncRedisFacade(client)
        try:
            record = await write_gpu_quarantine(
                redis_facade,
                worker_id,
                device=device,
                reason=reason,
                fault_class="unsafe_process_group_shutdown",
                node_id=_hostname(),
                hostname=_hostname(),
                physical_scope=True,
            )
        except Exception as exc:
            print(f"ERROR: could not persist physical GPU quarantine for {worker_id}: {exc}")
            record = {
                "state": "quarantined",
                "scope": "physical_gpu",
                "worker_id": worker_id,
                "device": device,
                "reason": reason,
                "fault_class": "unsafe_process_group_shutdown",
                "node_id": _hostname(),
                "hostname": _hostname(),
                "page_user_state": "pending",
                "notification_provenance": UNLATCHED_NOTIFICATION_PROVENANCE,
            }
        try:
            outcome = await send_gpu_quarantine_page(record)
            state = "sent" if outcome.success else "failed"
            superseded = outcome.protocol_version == "superseded"
            error = "" if outcome.success else f"{outcome.error_kind or 'unknown'}: {outcome.error or ''}"
        except Exception as exc:
            state = "failed"
            superseded = False
            error = f"unexpected_error: {type(exc).__name__}"
        unlatched = record.get("notification_provenance") == UNLATCHED_NOTIFICATION_PROVENANCE
        if not superseded and not unlatched:
            try:
                await update_gpu_quarantine_notification(
                    redis_facade,
                    worker_id,
                    device=device,
                    hostname=_hostname(),
                    expected_generation=gpu_quarantine_generation(record),
                    state=state,
                    error=error,
                )
            except Exception as exc:
                print(f"ERROR: could not persist page-user state for {worker_id}: {exc}")

    asyncio.run(persist_and_page())
    print(f"ERROR: quarantined {worker_id} on {device}: {reason}")


def _delete_worker_registration_if_current(
    client: Any,
    worker_id: str,
    *,
    pid: int,
    map_start_ticks: str | None,
    map_process_group: str | None,
    map_session_id: str | None,
) -> bool:
    prefix = REDIS_KEY_PREFIX
    deleted = client.eval(
        _DELETE_WORKER_REGISTRATION_IF_CURRENT,
        3,
        f"{prefix}:worker_process:{worker_id}",
        f"{prefix}:expected_worker:{worker_id}",
        f"{prefix}:expected_workers",
        str(pid),
        map_start_ticks or "",
        map_process_group or "",
        map_session_id or "",
        "1" if map_start_ticks is not None else "0",
        "1" if map_process_group is not None else "0",
        "1" if map_session_id is not None else "0",
        worker_id,
    )
    return bool(deleted)


def _drain_registered_worker(
    client: Any,
    worker_id: str,
    *,
    graceful_seconds: float,
) -> bool:
    prefix = REDIS_KEY_PREFIX
    process_info = _decode_redis_hash(client.hgetall(f"{prefix}:worker_process:{worker_id}"))
    if not process_info:
        client.srem(f"{prefix}:expected_workers", worker_id)
        client.delete(f"{prefix}:expected_worker:{worker_id}")
        return True

    device = process_info.get("device", "")
    try:
        pid = int(process_info.get("pid") or 0)
        process_group = int(process_info.get("process_group") or pid)
        session_id = int(process_info.get("session_id") or pid)
    except (TypeError, ValueError):
        reason = "invalid PID, process-group, or session identity in worker process map"
        _quarantine_unsafe_worker_group(client, worker_id, device, reason)
        return False
    map_start_ticks = process_info.get("proc_start_ticks") if "proc_start_ticks" in process_info else None
    map_process_group = process_info.get("process_group") if "process_group" in process_info else None
    map_session_id = process_info.get("session_id") if "session_id" in process_info else None
    stopped, reason = _stop_authenticated_worker_group(
        worker_id,
        pid=pid,
        expected_start_ticks=map_start_ticks or "",
        process_group=process_group,
        session_id=session_id,
        graceful_seconds=graceful_seconds,
    )
    if not stopped:
        _quarantine_unsafe_worker_group(client, worker_id, device, reason)
        return False
    try:
        deleted = _delete_worker_registration_if_current(
            client,
            worker_id,
            pid=pid,
            map_start_ticks=map_start_ticks,
            map_process_group=map_process_group,
            map_session_id=map_session_id,
        )
    except Exception as exc:
        print(f"ERROR: worker {worker_id} drained but its generation map could not be removed: {exc}")
        return False
    if not deleted:
        print(f"ERROR: worker {worker_id} process map changed during drain; refusing replacement")
        return False
    return True


def _clear_expected_workers_for_host(client: Any, hostname: str, *, graceful_seconds: float = 1.0) -> bool:
    """Drain and remove this host's registrations without erasing live generations."""

    prefix = REDIS_KEY_PREFIX
    success = True
    try:
        worker_ids = list(client.smembers(f"{prefix}:expected_workers"))
    except Exception as exc:
        print(f"ERROR: could not load expected workers before replacement: {exc}")
        return False
    for raw_worker_id in worker_ids:
        worker_id = _decode_redis_text(raw_worker_id)
        try:
            owner = client.hget(f"{prefix}:expected_worker:{worker_id}", "hostname") or ""
            owner = _decode_redis_text(owner) if owner else ""
            if not owner or owner == hostname:
                success = (
                    _drain_registered_worker(
                        client,
                        worker_id,
                        graceful_seconds=graceful_seconds,
                    )
                    and success
                )
        except Exception as exc:
            print(f"ERROR: could not safely retire expected worker {worker_id}: {exc}")
            success = False
    return success


def _abort_unregistered_launch(
    client: Any | None,
    worker_id: str,
    device: str,
    worker_pid: int,
    reason: str,
) -> bool:
    """Contain a just-launched process before propagating a registration error."""

    identity = _LAUNCHED_IDENTITIES.pop(worker_pid, None)
    if identity is None:
        identity = _read_process_identity(worker_pid)
    if identity is None:
        observed_process_groups = {worker_pid}
        try:
            stopped = _session_is_drained(worker_pid, observed_process_groups)
        except Exception:
            stopped = False
        stop_reason = (
            "" if stopped else f"unregistered session {worker_pid} is not drained and has no authenticated start_ticks"
        )
    else:
        stopped, stop_reason = _stop_authenticated_worker_group(
            worker_id,
            pid=identity.pid,
            expected_start_ticks=identity.start_ticks,
            process_group=identity.process_group,
            session_id=identity.session_id,
            graceful_seconds=1.0,
        )
    if not stopped:
        _quarantine_unsafe_worker_group(
            client,
            worker_id,
            device,
            f"{reason}; cleanup failed: {stop_reason}",
        )
    return stopped


def _assert_worker_process_slot_empty(client: Any, worker_id: str) -> None:
    """Refuse to launch while any prior generation map remains."""

    key = f"{REDIS_KEY_PREFIX}:worker_process:{worker_id}"
    try:
        process_info = _decode_redis_hash(client.hgetall(key))
    except Exception as exc:
        raise SystemExit(f"Cannot preflight worker {worker_id} process map: {exc}") from exc
    if process_info.get("pid"):
        raise SystemExit(f"Cannot launch worker {worker_id}: an older process generation still owns {key}")


def _register_expected_worker(
    client: Any, worker_id: str, device: str, hostname: str, node_id: str, worker_pid: int
) -> None:
    """Register a worker for monitor supervision.

    Hashes are written BEFORE set membership: an id must never be visible in
    expected_workers without its owning hostname recorded, or every host's
    monitor would claim it.
    """
    if client is None:
        _abort_unregistered_launch(client, worker_id, device, worker_pid, "Redis client is unavailable")
        raise SystemExit(f"Cannot register worker {worker_id}: Redis client is unavailable")
    prefix = REDIS_KEY_PREFIX
    launch_identity = _LAUNCHED_IDENTITIES.get(worker_pid)
    try:
        identity = _read_process_identity(worker_pid)
    except Exception as exc:
        _abort_unregistered_launch(client, worker_id, device, worker_pid, "process identity read failed")
        raise SystemExit(f"Cannot register worker {worker_id}: process identity read failed") from exc
    if identity is None or identity.state == "Z":
        _abort_unregistered_launch(client, worker_id, device, worker_pid, "process exited before registration")
        raise SystemExit(f"Cannot register worker {worker_id}: PID {worker_pid} exited before registration")
    if identity.process_group != worker_pid or identity.session_id != worker_pid:
        _abort_unregistered_launch(client, worker_id, device, worker_pid, "worker is not a session leader")
        raise SystemExit(f"Cannot register worker {worker_id}: PID {worker_pid} is not a new-session leader")
    if launch_identity is not None and (
        identity.start_ticks != launch_identity.start_ticks
        or identity.process_group != launch_identity.process_group
        or identity.session_id != launch_identity.session_id
    ):
        _abort_unregistered_launch(client, worker_id, device, worker_pid, "PID generation changed before registration")
        raise SystemExit(f"Cannot register worker {worker_id}: PID generation changed before registration")
    if not _cmdline_matches_worker(worker_pid, worker_id):
        _abort_unregistered_launch(client, worker_id, device, worker_pid, "worker command identity mismatch")
        raise SystemExit(f"Cannot register worker {worker_id}: PID {worker_pid} identity does not match its command")
    try:
        registered = client.eval(
            _REGISTER_EXPECTED_WORKER_IF_EMPTY,
            3,
            f"{prefix}:worker_process:{worker_id}",
            f"{prefix}:expected_worker:{worker_id}",
            f"{prefix}:expected_workers",
            str(worker_pid),
            datetime.now().isoformat(),
            identity.start_ticks,
            str(identity.process_group),
            str(identity.session_id),
            device,
            hostname,
            node_id,
            worker_id,
        )
        if registered:
            _LAUNCHED_IDENTITIES.pop(worker_pid, None)
            return
    except Exception as exc:
        registration_error = f"Redis registration failed: {type(exc).__name__}"
    else:
        registration_error = "an older process generation still owns the worker map"

    stopped = _abort_unregistered_launch(client, worker_id, device, worker_pid, registration_error)
    if stopped:
        try:
            _delete_worker_registration_if_current(
                client,
                worker_id,
                pid=identity.pid,
                map_start_ticks=identity.start_ticks,
                map_process_group=str(identity.process_group),
                map_session_id=str(identity.session_id),
            )
        except Exception:
            # An ambiguous Redis write may have committed. The dead generation
            # remains fenced and will be CAS-cleaned by the next safe start.
            pass
    raise SystemExit(f"Cannot register worker {worker_id}: {registration_error}; replacement launch aborted")


def _collect_pids(pattern: str) -> list[int]:
    pgrep = shutil.which("pgrep")
    if not pgrep:
        return []
    proc = subprocess.run([pgrep, "-f", pattern], text=True, stdout=subprocess.PIPE, check=False)
    own_pid = os.getpid()
    return [int(line) for line in proc.stdout.splitlines() if line.strip().isdigit() and int(line) != own_pid]


def _cmdline_matches_pattern(pid: int, pattern: str) -> bool:
    return re.search(pattern, " ".join(_read_process_argv(pid))) is not None


def _command_option(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def _worker_metadata_from_argv(pid: int) -> tuple[str, str]:
    argv = _read_process_argv(pid)
    return _command_option(argv, "--worker-id"), _command_option(argv, "--device")


def _stop_discovered_process_group(pid: int, pattern: str, graceful_seconds: float) -> tuple[bool, str]:
    """Stop a service root found by pgrep and prove its complete session gone."""

    identity = _read_process_identity(pid)
    if identity is None:
        return True, ""
    if identity.process_group != pid or identity.session_id != pid:
        return False, f"matched PID {pid} is not a new-session leader"
    if identity.state == "Z" or not _cmdline_matches_pattern(pid, pattern):
        latest = _read_process_identity(pid)
        if latest is not None and latest.state != "Z":
            return False, f"PID {pid} no longer matches service pattern {pattern}"
        observed_process_groups = {identity.process_group}
        try:
            if _wait_for_session_drain(identity.session_id, observed_process_groups, graceful_seconds):
                return True, ""
        except Exception as exc:
            return False, f"session drain proof failed: {type(exc).__name__}"
        return _force_kill_worker_session(
            identity.session_id,
            expected_leader_start_ticks=identity.start_ticks,
            observed_process_groups=observed_process_groups,
        )
    observed_process_groups = {identity.process_group}
    current = _read_process_identity(pid)
    if current is None:
        try:
            if _session_is_drained(identity.session_id, observed_process_groups):
                return True, ""
        except Exception as exc:
            return False, f"session drain proof failed: {type(exc).__name__}"
        return _force_kill_worker_session(
            identity.session_id,
            expected_leader_start_ticks=identity.start_ticks,
            observed_process_groups=observed_process_groups,
        )
    if (
        current.start_ticks != identity.start_ticks
        or current.process_group != identity.process_group
        or current.session_id != identity.session_id
        or current.state == "Z"
        or not _cmdline_matches_pattern(pid, pattern)
    ):
        latest = _read_process_identity(pid)
        if latest is not None and latest.state != "Z":
            return False, f"PID {pid} generation changed immediately before SIGTERM"
        try:
            if _wait_for_session_drain(identity.session_id, observed_process_groups, graceful_seconds):
                return True, ""
        except Exception as exc:
            return False, f"session drain proof failed: {type(exc).__name__}"
        return _force_kill_worker_session(
            identity.session_id,
            expected_leader_start_ticks=identity.start_ticks,
            observed_process_groups=observed_process_groups,
        )
    try:
        os.killpg(identity.process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            return False, f"SIGTERM failed for process group {identity.process_group}: {type(exc).__name__}"
    try:
        if _wait_for_session_drain(identity.session_id, observed_process_groups, graceful_seconds):
            return True, ""
    except Exception as exc:
        return False, f"session drain proof failed: {type(exc).__name__}"
    return _force_kill_worker_session(
        identity.session_id,
        expected_leader_start_ticks=identity.start_ticks,
        observed_process_groups=observed_process_groups,
    )


def _kill_processes(
    pattern: str,
    description: str,
    grace_seconds: float = 1.0,
    *,
    quarantine_client: Any | None = None,
) -> bool:
    print(f"Stopping {description}...")
    pids = _collect_pids(pattern)
    if not pids:
        print(f"No {description} processes found.")
        return True
    success = True
    for pid in pids:
        try:
            identity = _read_process_identity(pid)
            if identity is not None and identity.session_id != pid:
                # A child (including an inner PGID leader) is contained by its
                # outer session leader.  It must be absent in the final pgrep
                # below before this operation passes.
                continue
            worker_id, device = _worker_metadata_from_argv(pid)
            stopped, reason = _stop_discovered_process_group(pid, pattern, grace_seconds)
        except Exception as exc:
            worker_id, device = _worker_metadata_from_argv(pid)
            stopped = False
            reason = f"process inspection failed: {type(exc).__name__}"
        if not stopped:
            print(f"ERROR: could not safely stop {description} PID {pid}: {reason}")
            if worker_id and device.startswith("cuda:"):
                _quarantine_unsafe_worker_group(quarantine_client, worker_id, device, reason)
            success = False
    leftovers = _collect_pids(pattern)
    if leftovers:
        print(f"ERROR: {description} still has live matching PIDs: {','.join(map(str, leftovers))}")
        success = False
    return success


def _default_stop_grace_seconds() -> float:
    raw = os.environ.get("KERNELGYM_WORKER_SHUTDOWN_DRAIN_SEC", "120")
    try:
        drain_seconds = max(0.0, float(raw))
    except ValueError:
        drain_seconds = 120.0
    return max(1.0, drain_seconds + 30.0)


def cmd_stop(args: argparse.Namespace) -> int:
    values = _profile_values(args.profile)
    requested_grace = getattr(args, "graceful_seconds", None)
    grace = max(1.0, float(requested_grace)) if requested_grace is not None else _default_stop_grace_seconds()

    redis_host = values.get("REDIS_HOST", "localhost")
    is_local_redis = redis_host in {"localhost", "127.0.0.1", "::1", ""}
    redis_is_up = bool(is_local_redis and _port_is_open(redis_host, REDIS_PORT))
    client = _redis_client(values) if redis_is_up else None

    # Close the public admission point before any potentially slow monitor
    # notification or worker drain.  Otherwise a request can enter after the
    # queue was observed empty but before the workers are stopped, leaving a
    # half-shutdown deployment executing an orphaned task.
    safe = _kill_processes(
        "kernelgym.server.api.server",
        "KernelGym API server",
        min(grace, 15.0),
    )
    if not safe:
        print("KernelGym stop is INCOMPLETE; API admission is still live, preserving all process maps.")
        return 1

    # Stop the supervisor first so it cannot race this command by replacing a
    # worker while the old process group is draining.
    safe = _kill_processes(
        "kernelgym.worker.worker_monitor",
        "KernelGym worker monitor",
        min(grace, 15.0),
    )
    if not safe:
        print("KernelGym stop is INCOMPLETE; worker monitor still live, preserving all process maps.")
        return 1

    # Registered workers retain an immutable PID generation, PGID, and SID.
    # Their map is generation-CAS deleted only after the session is empty and
    # every observed process group returns ESRCH.
    if safe and client is not None:
        safe = _clear_expected_workers_for_host(
            client,
            _hostname(),
            graceful_seconds=grace,
        )

    # A worker-only node's process map lives in the primary Redis, which this
    # host-local stop intentionally does not contact. Authenticate local
    # new-session roots from /proc and drain every PGID in each complete SID;
    # the next join will generation-CAS-clean the retained remote maps.
    for pattern, description in [
        ("kernelgym.worker.single_worker", "KernelGym single workers"),
        ("kernelgym.worker.cpu_worker", "KernelGym CPU compile workers"),
        ("kernelgym.worker.gpu_worker", "KernelGym worker manager"),
    ]:
        if not _kill_processes(
            pattern,
            description,
            grace,
            quarantine_client=client,
        ):
            safe = False

    # Pool children remain inside a worker's session even when each warm worker
    # leads its own PGID. Any survivor means the SID was not proven absent;
    # never erase Redis or permit a replacement in that state.
    for pattern, description in [
        ("multiprocessing.spawn", "multiprocessing spawn workers"),
        ("multiprocessing.resource_tracker", "multiprocessing resource tracker"),
    ]:
        leftovers = _collect_pids(pattern)
        if leftovers:
            print(f"ERROR: {description} survived group shutdown: {','.join(map(str, leftovers))}")
            safe = False

    if not safe:
        print("KernelGym stop is INCOMPLETE; preserving Redis process maps and refusing replacement startup.")
        return 1

    # Shut down the LOCAL redis WITHOUT saving, instead of only clearing keys.
    # nosave avoids writing redis's dataset back to its (possibly NFS-shared)
    # dump file, and freeing the process lets the next start relaunch redis with
    # the configured --dir. cmd_stop always targets REDIS_HOST=localhost (plain
    # profile), so on a worker-only node — which has no local redis — there is
    # nothing to stop and the primary's remote redis is never touched.
    if client is not None and values.get("REDIS_PORT") and is_local_redis:
        if redis_is_up:
            try:
                client.execute_command("SHUTDOWN", "NOSAVE")
            except Exception:
                pass  # SHUTDOWN closes the socket on success -> expected error
            port_freed = False
            for _ in range(50):
                if not _port_is_open(redis_host, REDIS_PORT):
                    port_freed = True
                    break
                time.sleep(0.1)
            if port_freed:
                print("Stopped local Redis (SHUTDOWN NOSAVE).")
            else:
                # SHUTDOWN did not take: fall back to clearing our keyspace so a
                # reused redis at least restarts from a clean slate.
                try:
                    keys = list(client.scan_iter(f"{REDIS_KEY_PREFIX}:*"))
                    if keys:
                        client.delete(*keys)
                    print(f"Redis still up after SHUTDOWN; cleared {len(keys)} keys instead.")
                except Exception as exc:
                    print(f"Skipping Redis cleanup: {exc}")
        # else: no local redis (e.g. worker-only node) -> nothing to stop
    elif not is_local_redis:
        print(f"REDIS_HOST={redis_host} is remote; leaving its Redis untouched on stop.")
    print("KernelGym stopped.")
    return 0


def cmd_start_local(args: argparse.Namespace) -> int:
    values = _apply_runtime_overrides(_profile_values(args.profile), args)
    values = _resolve_gpu_devices(values)

    # ``--no-stop-first`` is retained for CLI compatibility, but it is not a
    # safety bypass: a no-op verified stop is cheap after --clear-cache and is
    # still required to prove that no prior process group survived.
    if cmd_stop(argparse.Namespace(profile=args.profile)) != 0:
        raise SystemExit("Refusing start-local because the previous process groups were not safely drained")

    if args.log_dir:
        values["LOG_DIR"] = args.log_dir
    if args.eval_results_path:
        values["EVAL_RESULTS_PATH"] = args.eval_results_path
    values = _with_hostname_log_dirs(values)
    env = _service_env(values)
    log_dir = ROOT_DIR / values.get("LOG_DIR", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    _ensure_redis(values)
    client = _redis_client(values)
    if client is None:
        raise SystemExit("Redis Python package is required for generation-fenced worker startup")
    if not _clear_expected_workers_for_host(
        client,
        _hostname(),
        graceful_seconds=_default_stop_grace_seconds(),
    ):
        raise SystemExit("Refusing start-local because an old worker process group was not safely drained")

    gpu_ids = [str(gpu) for gpu in _parse_gpu_devices(values.get("GPU_DEVICES"))]
    cpu_workers = int(env.get("CPU_COMPILE_WORKERS", "2"))
    worker_specs = [*(f"worker_gpu_{gpu}" for gpu in gpu_ids), *(f"worker_cpu_{i}" for i in range(cpu_workers))]
    for worker_id in worker_specs:
        _assert_worker_process_slot_empty(client, worker_id)

    api_pid = _launch_background(
        [sys.executable, "-m", "kernelgym.server.api.server"], log_dir / "api_server.log", env
    )
    print(f"API server PID: {api_pid}")
    monitor_pid = _launch_background(
        [sys.executable, "-m", "kernelgym.worker.worker_monitor", "--persistent"],
        log_dir / "worker_monitor.log",
        env,
    )
    print(f"Worker monitor PID: {monitor_pid}")

    for gpu in gpu_ids:
        worker_id = f"worker_gpu_{gpu}"
        pid = _launch_background(
            [
                sys.executable,
                "-m",
                "kernelgym.worker.single_worker",
                "--worker-id",
                worker_id,
                "--device",
                f"cuda:{gpu}",
                "--persistent",
            ],
            log_dir / f"worker_gpu_{gpu}.log",
            env,
        )
        print(f"{worker_id} PID: {pid}")
        _register_expected_worker(client, worker_id, f"cuda:{gpu}", _hostname(), values.get("NODE_ID", ""), pid)
    for index in range(max(0, cpu_workers)):
        worker_id = f"worker_cpu_{index}"
        pid = _launch_background(
            [
                sys.executable,
                "-m",
                "kernelgym.worker.cpu_worker",
                "--worker-id",
                worker_id,
            ],
            log_dir / f"worker_cpu_{index}.log",
            env,
        )
        print(f"{worker_id} PID: {pid}")
        _register_expected_worker(client, worker_id, "cpu", _hostname(), values.get("NODE_ID", ""), pid)
    print(f"KernelGym processes launched; readiness checks are still pending. Logs: {log_dir}")
    return 0


def _check_worker_connectivity(values: dict[str, str]) -> None:
    client = _redis_client(values)
    if client is None:
        raise SystemExit("redis Python package is required for worker-node startup.")
    try:
        client.ping()
    except Exception as exc:
        raise SystemExit(f"Cannot connect to Redis: {exc}") from exc
    health_url = f"{_api_base(values)}/health"
    try:
        _http_get_json(health_url)
    except Exception as exc:
        raise SystemExit(f"Cannot reach API health endpoint {health_url}: {exc}") from exc


def cmd_start_worker_node(args: argparse.Namespace) -> int:
    server_env = Path(args.server_env) if getattr(args, "server_env", None) else None
    if server_env is not None:
        if not server_env.exists():
            raise SystemExit(f"server.env not found: {server_env}")
        values = _read_env_file(server_env)
        for required in ("API_HOST", "REDIS_HOST"):
            if not values.get(required):
                raise SystemExit(f"Missing required env var in {server_env}: {required}")
    else:
        master_addr = getattr(args, "master_addr", None)
        if not master_addr:
            raise SystemExit("--master-addr is required when server_env is not provided")
        values = _worker_profile_values(
            getattr(args, "profile", "auto"), master_addr, getattr(args, "node_rank", None)
        )
    values = _apply_runtime_overrides(values, args)
    values = _resolve_gpu_devices(values)
    values = _with_torch_cuda_arch_list(values)
    _check_worker_connectivity(values)

    api_base = _api_base(values)
    hostname = _hostname()
    params = {"hostname": hostname}
    if values.get("NODE_ID"):
        params["node_name"] = values["NODE_ID"]
    allocate_url = f"{api_base}/node/allocate?{urllib.parse.urlencode(params)}"
    try:
        allocation = _http_post_json(allocate_url)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Node allocation failed: HTTP {exc.code}") from exc
    except Exception as exc:
        raise SystemExit(f"Node allocation failed: {exc}") from exc
    updates: dict[str, str] = {}
    if allocation.get("node_id"):
        updates["NODE_ID"] = str(allocation["node_id"])
    if allocation.get("hostname"):
        updates["WORKER_NAME_PREFIX"] = str(allocation["hostname"])
    if updates and server_env is not None:
        _update_env_file(server_env, updates)
    if updates:
        values.update(updates)

    values.setdefault("LOG_DIR", "logs")
    values.setdefault("PY_LOG_DIR", "py_logs")
    values = _with_hostname_log_dirs(values)
    env = _service_env(values)
    env.pop("GPU_ARCH", None)
    log_dir = ROOT_DIR / values.get("LOG_DIR", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Same supervised layout as start-local: a per-host worker_monitor plus one
    # process per worker, all registered in the (cluster-shared) expected set so
    # the LOCAL monitor restarts them. The monitor only enforces expected ids
    # whose recorded hostname matches its own host.
    node_prefix = values.get("NODE_ID") or values.get("WORKER_NAME_PREFIX") or hostname
    client = _redis_client(values)
    if client is None:
        raise SystemExit("Redis Python package is required for generation-fenced worker startup")
    if not _kill_processes(
        "kernelgym.worker.worker_monitor",
        "KernelGym worker monitor",
        min(_default_stop_grace_seconds(), 15.0),
    ):
        raise SystemExit("Refusing worker-node start because the old worker monitor did not stop safely")
    for pattern, description in [
        ("kernelgym.worker.single_worker", "KernelGym single workers"),
        ("kernelgym.worker.cpu_worker", "KernelGym CPU compile workers"),
        ("kernelgym.worker.gpu_worker", "KernelGym worker manager"),
    ]:
        if not _kill_processes(
            pattern,
            description,
            _default_stop_grace_seconds(),
            quarantine_client=client,
        ):
            raise SystemExit(f"Refusing worker-node start because {description} did not drain safely")
    for pattern in ("multiprocessing.spawn", "multiprocessing.resource_tracker"):
        if _collect_pids(pattern):
            raise SystemExit(
                "Refusing worker-node start because a worker subprocess survived its authenticated process group"
            )
    # Drop stale registrations owned by this host from a previous join (the
    # worker count or node id may have changed); never touch other hosts'.
    if not _clear_expected_workers_for_host(
        client,
        hostname,
        graceful_seconds=_default_stop_grace_seconds(),
    ):
        raise SystemExit("Refusing worker-node start because an old process group was not safely drained")

    gpu_ids = [str(gpu) for gpu in _parse_gpu_devices(values.get("GPU_DEVICES"))]
    cpu_worker_count = max(0, int(env.get("CPU_COMPILE_WORKERS", "2")))
    target_worker_ids = [
        *(f"{node_prefix}_gpu_{gpu}" for gpu in gpu_ids),
        *(f"{node_prefix}_cpu_{index}" for index in range(cpu_worker_count)),
    ]
    for worker_id in target_worker_ids:
        _assert_worker_process_slot_empty(client, worker_id)

    monitor_pid = _launch_background(
        [sys.executable, "-m", "kernelgym.worker.worker_monitor", "--persistent"],
        log_dir / "worker_monitor.log",
        env,
    )
    print(f"Worker monitor PID: {monitor_pid}")

    worker_ids: list[str] = []
    for gpu in gpu_ids:
        worker_id = f"{node_prefix}_gpu_{gpu}"
        pid = _launch_background(
            [
                sys.executable,
                "-m",
                "kernelgym.worker.single_worker",
                "--worker-id",
                worker_id,
                "--device",
                f"cuda:{gpu}",
                "--persistent",
            ],
            log_dir / f"worker_gpu_{gpu}.log",
            env,
        )
        print(f"{worker_id} PID: {pid}")
        _register_expected_worker(client, worker_id, f"cuda:{gpu}", hostname, values.get("NODE_ID", ""), pid)
        worker_ids.append(worker_id)

    cpu_pids: list[str] = []
    for index in range(cpu_worker_count):
        cpu_worker_id = f"{node_prefix}_cpu_{index}"
        cpu_pid = _launch_background(
            [sys.executable, "-m", "kernelgym.worker.cpu_worker", "--worker-id", cpu_worker_id],
            log_dir / f"worker_cpu_{index}.log",
            env,
        )
        print(f"{cpu_worker_id} PID: {cpu_pid}")
        _register_expected_worker(client, cpu_worker_id, "cpu", hostname, values.get("NODE_ID", ""), cpu_pid)
        cpu_pids.append(str(cpu_pid))
    if cpu_pids:
        (log_dir / "cpu_worker.pids").write_text("\n".join(cpu_pids) + "\n", encoding="utf-8")

    (log_dir / "worker_ids.list").write_text("\n".join(worker_ids) + "\n", encoding="utf-8")
    try:
        status = _http_get_json(f"{api_base}/workers/status")
        print(json.dumps(status, indent=2, sort_keys=True))
    except Exception:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage reward-only KernelGym services")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_local = subparsers.add_parser("start-local", help="start local API, monitor, and GPU workers")
    start_local.add_argument("--profile", default="auto", help=f"auto or known profile: {', '.join(profile_names())}")
    start_local.add_argument("--log-dir", default=None)
    start_local.add_argument("--eval-results-path", default=None)
    start_local.add_argument("--cpu-compile-workers", "--cpu-workers", type=int, default=None)
    start_local.add_argument(
        "--gpu-devices",
        default=None,
        help="auto (default) or a comma/JSON list of container-logical CUDA device indices",
    )
    start_local.add_argument("--redis-remote-access", action="store_true")
    start_local.add_argument("--no-stop-first", action="store_true")
    start_local.set_defaults(func=cmd_start_local)

    worker_node = subparsers.add_parser("start-worker-node", help="start a worker-only node")
    worker_node.add_argument("server_env", nargs="?")
    worker_node.add_argument("--profile", default="auto", help=f"auto or known profile: {', '.join(profile_names())}")
    worker_node.add_argument("--master-addr", default=None)
    worker_node.add_argument("--node-rank", default=None)
    worker_node.add_argument("--cpu-compile-workers", "--cpu-workers", type=int, default=None)
    worker_node.add_argument(
        "--gpu-devices",
        default=None,
        help="auto (default) or a comma/JSON list of container-logical CUDA device indices",
    )
    worker_node.set_defaults(func=cmd_start_worker_node)

    stop = subparsers.add_parser("stop", help="stop local KernelGym processes and clear Redis keys")
    stop.add_argument("--profile", default="auto", help=f"auto or known profile: {', '.join(profile_names())}")
    stop.add_argument(
        "--graceful-seconds",
        type=float,
        default=None,
        help=(
            "how long to wait after SIGTERM before SIGKILL; defaults to "
            "KERNELGYM_WORKER_SHUTDOWN_DRAIN_SEC + 30 seconds"
        ),
    )
    stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
