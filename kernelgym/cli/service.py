"""Service management CLI for reward-only KernelGym.

The repository keeps shell entrypoints for compatibility, but operational logic
lives here so it can be tested and maintained as Python code.
"""

from __future__ import annotations

import argparse
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

ROOT_DIR = Path(__file__).resolve().parents[2]
TORCH_CUDA_ARCH_LIST_ENV = "TORCH_CUDA_ARCH_LIST"
_CUDA_ARCH_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def _hostname() -> str:
    return os.environ.get("HOSTNAME") or socket.gethostname() or "local"


def _local_host_addresses() -> set[str]:
    addresses = {"localhost", "127.0.0.1", _hostname(), socket.gethostname()}
    try:
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        addresses.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return {address for address in addresses if address}


def _profile_values(profile_name: str) -> dict[str, str]:
    if profile_name != "auto":
        try:
            return get_profile(profile_name).env()
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    addresses = _local_host_addresses()
    for name in profile_names():
        profile = get_profile(name)
        if profile.host in addresses:
            return profile.env()
    choices = ", ".join(("auto", *profile_names()))
    raise SystemExit(f"Could not auto-detect reward profile from local addresses {sorted(addresses)}. Use: {choices}")


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
        ("GPU", ("GPU_DEVICES", "NODE_ID")),
        ("Redis", ("REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_PASSWORD", "REDIS_KEY_PREFIX")),
        ("Worker pool", ("WORKER_POOL_SIZE", "MAX_TASKS_PER_WORKER", "CPU_COMPILE_WORKERS")),
        ("Defaults", ("DEFAULT_TOOLKIT", "DEFAULT_BACKEND_ADAPTER", "DEFAULT_BACKEND")),
        ("Logging", ("LOG_LEVEL", "LOG_DIR", "PY_LOG_DIR")),
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


def _update_env_file(path: Path, updates: dict[str, str]) -> None:
    values = _read_env_file(path)
    values.update(updates)
    _write_env_file(path, values)


def _parse_gpu_devices(raw: str | None) -> list[str]:
    if not raw:
        return ["0"]
    value = raw.strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    except Exception:
        return [item.strip() for item in value.split(",") if item.strip()]


def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _service_env(values: dict[str, str]) -> dict[str, str]:
    values = _with_torch_cuda_arch_list(values)
    env = os.environ.copy()
    env.update(values)
    env["API_PORT"] = str(API_PORT)
    env["API_WORKERS"] = str(API_WORKERS)
    env["API_RELOAD"] = bool_env(API_RELOAD)
    env["REDIS_PORT"] = str(REDIS_PORT)
    env["REDIS_DB"] = str(REDIS_DB)
    env["REDIS_PASSWORD"] = REDIS_PASSWORD
    env["REDIS_KEY_PREFIX"] = REDIS_KEY_PREFIX
    env["METRICS_PORT"] = str(METRICS_PORT)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT_DIR) if not pythonpath else f"{ROOT_DIR}:{pythonpath}"
    return env


def _launch_background(command: list[str], log_file: Path, env: dict[str, str]) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("ab")
    proc = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return int(proc.pid)


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
    if _port_is_open(host, port):
        return
    if host not in {"localhost", "127.0.0.1"}:
        raise SystemExit(f"Redis is not reachable at {host}:{port}. Start it before launching workers.")
    redis_server = shutil.which("redis-server")
    if not redis_server:
        raise SystemExit("redis-server not found; install Redis or set REDIS_HOST/REDIS_PORT to an existing server.")
    command = [redis_server, "--port", str(port), "--daemonize", "yes"]
    subprocess.run(command, check=True)
    time.sleep(1)


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


def _kill_processes(pattern: str, description: str) -> None:
    print(f"Stopping {description}...")
    pgrep = shutil.which("pgrep")
    if not pgrep:
        return
    proc = subprocess.run([pgrep, "-f", pattern], text=True, stdout=subprocess.PIPE, check=False)
    pids = [int(line) for line in proc.stdout.splitlines() if line.strip().isdigit()]
    own_pid = os.getpid()
    pids = [pid for pid in pids if pid != own_pid]
    if not pids:
        print(f"No {description} processes found.")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(1)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def cmd_stop(args: argparse.Namespace) -> int:
    values = _profile_values(args.profile)
    patterns = [
        ("kernelgym.server.api.server", "KernelGym API server"),
        ("kernelgym.worker.worker_monitor", "KernelGym worker monitor"),
        ("kernelgym.worker.single_worker", "KernelGym single workers"),
        ("kernelgym.worker.cpu_worker", "KernelGym CPU compile workers"),
        ("kernelgym.worker.gpu_worker", "KernelGym worker manager"),
        ("uvicorn.*kernelgym", "Uvicorn server"),
        ("multiprocessing.spawn", "multiprocessing spawn workers"),
        ("multiprocessing.resource_tracker", "multiprocessing resource tracker"),
    ]
    for pattern, description in patterns:
        _kill_processes(pattern, description)

    client = _redis_client(values)
    if client is not None and values.get("REDIS_PORT"):
        prefix = REDIS_KEY_PREFIX
        try:
            keys = list(client.scan_iter(f"{prefix}:*"))
            if keys:
                client.delete(*keys)
            print(f"Cleared {len(keys)} Redis keys with prefix {prefix}:")
        except Exception as exc:
            print(f"Skipping Redis cleanup: {exc}")
    print("KernelGym stopped.")
    return 0


def cmd_start_local(args: argparse.Namespace) -> int:
    if not args.no_stop_first:
        cmd_stop(argparse.Namespace(profile=args.profile))

    values = _profile_values(args.profile)
    if args.log_dir:
        values["LOG_DIR"] = args.log_dir
    if args.eval_results_path:
        values["EVAL_RESULTS_PATH"] = args.eval_results_path
    env = _service_env(values)
    log_dir = ROOT_DIR / values.get("LOG_DIR", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    _ensure_redis(values)
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

    client = _redis_client(values)
    prefix = REDIS_KEY_PREFIX
    if client is not None:
        try:
            client.delete(f"{prefix}:expected_workers")
        except Exception:
            pass

    for gpu in _parse_gpu_devices(values.get("GPU_DEVICES")):
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
        if client is not None:
            try:
                client.sadd(f"{prefix}:expected_workers", worker_id)
                client.hset(
                    f"{prefix}:expected_worker:{worker_id}",
                    mapping={"device": f"cuda:{gpu}", "hostname": _hostname(), "node_id": values.get("NODE_ID", "")},
                )
                client.hset(
                    f"{prefix}:worker_process:{worker_id}",
                    mapping={"pid": str(pid), "start_time": time.ctime(), "device": f"cuda:{gpu}"},
                )
            except Exception:
                pass
    cpu_workers = int(env.get("CPU_COMPILE_WORKERS", "2"))
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
        if client is not None:
            try:
                client.sadd(f"{prefix}:expected_workers", worker_id)
                client.hset(
                    f"{prefix}:expected_worker:{worker_id}",
                    mapping={"device": "cpu", "hostname": _hostname(), "node_id": values.get("NODE_ID", "")},
                )
                client.hset(
                    f"{prefix}:worker_process:{worker_id}",
                    mapping={"pid": str(pid), "start_time": time.ctime(), "device": "cpu"},
                )
            except Exception:
                pass
    print(f"KernelGym started. Logs: {log_dir}")
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
    server_env = Path(args.server_env)
    env_file = ROOT_DIR / ".env"
    if not server_env.exists():
        raise SystemExit(f"server.env not found: {server_env}")
    if not env_file.exists():
        shutil.copyfile(server_env, env_file)

    values = _read_env_file(env_file)
    for required in ("API_HOST", "REDIS_HOST"):
        if not values.get(required):
            raise SystemExit(f"Missing required env var in {env_file}: {required}")
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
    if updates:
        _update_env_file(env_file, updates)
        values.update(updates)

    env = _service_env(values)
    env.pop("GPU_ARCH", None)
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pid = _launch_background(
        [sys.executable, "-m", "kernelgym.worker.gpu_worker"], log_dir / "worker_manager.log", env
    )
    (log_dir / "worker_manager.pid").write_text(f"{pid}\n", encoding="utf-8")
    cpu_pids: list[str] = []
    for index in range(max(0, int(env.get("CPU_COMPILE_WORKERS", "2")))):
        cpu_worker_id = f"{updates.get('NODE_ID') or values.get('NODE_ID') or hostname}_cpu_{index}"
        cpu_pid = _launch_background(
            [sys.executable, "-m", "kernelgym.worker.cpu_worker", "--worker-id", cpu_worker_id],
            log_dir / f"worker_cpu_{index}.log",
            env,
        )
        cpu_pids.append(str(cpu_pid))
    if cpu_pids:
        (log_dir / "cpu_worker.pids").write_text("\n".join(cpu_pids) + "\n", encoding="utf-8")

    prefix = values.get("NODE_ID") or values.get("WORKER_NAME_PREFIX") or hostname
    worker_ids = [f"{prefix}_gpu_{gpu}" for gpu in _parse_gpu_devices(values.get("GPU_DEVICES"))]
    (log_dir / "worker_ids.list").write_text("\n".join(worker_ids) + "\n", encoding="utf-8")
    print(f"WorkerManager PID: {pid}")
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
    start_local.add_argument("--profile", choices=("auto", *profile_names()), default="auto")
    start_local.add_argument("--log-dir", default=None)
    start_local.add_argument("--eval-results-path", default=None)
    start_local.add_argument("--no-stop-first", action="store_true")
    start_local.set_defaults(func=cmd_start_local)

    worker_node = subparsers.add_parser("start-worker-node", help="start a worker-only node")
    worker_node.add_argument("server_env", nargs="?", default=str(ROOT_DIR / "server.env"))
    worker_node.set_defaults(func=cmd_start_worker_node)

    stop = subparsers.add_parser("stop", help="stop local KernelGym processes and clear Redis keys")
    stop.add_argument("--profile", choices=("auto", *profile_names()), default="auto")
    stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
