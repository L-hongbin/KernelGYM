"""Service management CLI for reward-only KernelGym.

The repository keeps shell entrypoints for compatibility, but operational logic
lives here so it can be tested and maintained as Python code.
"""

from __future__ import annotations

import argparse
import json
import os
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


ROOT_DIR = Path(__file__).resolve().parents[2]


def _hostname() -> str:
    return os.environ.get("HOSTNAME") or socket.gethostname() or "local"


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
        ("GPU", ("GPU_DEVICES", "GPU_MEMORY_LIMIT", "NODE_ID")),
        ("Redis", ("REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_PASSWORD", "REDIS_KEY_PREFIX")),
        ("Worker pool", ("WORKER_POOL_SIZE", "MAX_TASKS_PER_WORKER")),
        ("Defaults", ("DEFAULT_TOOLKIT", "DEFAULT_BACKEND_ADAPTER", "DEFAULT_BACKEND")),
        ("Logging", ("LOG_LEVEL", "LOG_DIR", "PY_LOG_DIR")),
        ("Metrics", ("ENABLE_METRICS", "METRICS_PORT")),
        ("Profiling", ("ENABLE_PROFILING",)),
        ("Errors", ("VERBOSE_ERROR_TRACEBACK",)),
        ("Result persistence", ("SAVE_EVAL_RESULTS", "EVAL_RESULTS_PATH")),
        (
            "CUDA build",
            (
                "CUDA_HOME",
                "KERNELGYM_CUDA_AGENT_NVCC_THREADS",
                "KERNELGYM_TVM_FFI_NVCC_THREADS",
                "KERNELGYM_CUDA_AGENT_TMPDIR",
                "KERNELGYM_TVM_FFI_TMPDIR",
                "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DIR",
                "KERNELGYM_TVM_FFI_COMPILE_CACHE_DIR",
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


def _detect_gpus_json() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "[0]"
    try:
        proc = subprocess.run(
            [nvidia_smi, "-L"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "[0]"
    count = len([line for line in proc.stdout.splitlines() if line.strip()])
    if count <= 0:
        return "[0]"
    return json.dumps(list(range(count)))


def _host_ip() -> str:
    arnold_role = os.environ.get("ARNOLD_ROLE", "")
    arnold_id = os.environ.get("ARNOLD_ID", "")
    arnold_host = os.environ.get(f"ARNOLD_{arnold_role.upper()}_{arnold_id}_HOST")
    if arnold_host:
        return arnold_host
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        address = probe.getsockname()[0]
        probe.close()
        return address
    except OSError:
        return "127.0.0.1"


def _available_ports(use_indexed_ports: bool) -> list[int]:
    if use_indexed_ports:
        ports = []
    else:
        role = os.environ.get("ARNOLD_ROLE", "")
        worker_id = os.environ.get("ARNOLD_ID", "")
        raw = os.environ.get(f"ARNOLD_{role.upper()}_{worker_id}_PORT", "")
        ports = [int(part.strip()) for part in raw.split(",") if part.strip().isdigit()]

    if not ports:
        index = 0
        while True:
            raw = os.environ.get(f"PORT{index}")
            if raw is None:
                break
            if raw.isdigit():
                ports.append(int(raw))
            index += 1

    return ports or list(range(8000, 8010))


def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _select_ports(candidates: list[int], count: int = 3) -> list[int]:
    selected: list[int] = []
    for port in candidates:
        if len(selected) >= count:
            break
        if not _port_is_open("127.0.0.1", port, timeout=0.2):
            selected.append(port)
    if len(selected) < count:
        raise SystemExit(f"Could not find {count} available ports from candidates: {candidates}")
    return selected


def _service_env(values: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(values)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT_DIR) if not pythonpath else f"{ROOT_DIR}:{pythonpath}"
    cuda_home = env.get("CUDA_HOME")
    if cuda_home:
        cuda_bin = str(Path(cuda_home) / "bin")
        cuda_lib = str(Path(cuda_home) / "lib64")
        env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
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
    password = values.get("REDIS_PASSWORD") or None
    return redis.Redis(
        host=values.get("REDIS_HOST", "localhost"),
        port=int(values.get("REDIS_PORT", "6379")),
        db=int(values.get("REDIS_DB", "0")),
        password=password,
        decode_responses=True,
    )


def _ensure_redis(values: dict[str, str]) -> None:
    host = values.get("REDIS_HOST", "localhost")
    port = int(values.get("REDIS_PORT", "6379"))
    if _port_is_open(host, port):
        return
    if host not in {"localhost", "127.0.0.1"}:
        raise SystemExit(f"Redis is not reachable at {host}:{port}. Start it before launching workers.")
    redis_server = shutil.which("redis-server")
    if not redis_server:
        raise SystemExit("redis-server not found; install Redis or set REDIS_HOST/REDIS_PORT to an existing server.")
    command = [redis_server, "--port", str(port), "--daemonize", "yes"]
    password = values.get("REDIS_PASSWORD")
    if password:
        command.extend(["--requirepass", password])
    subprocess.run(command, check=True)
    time.sleep(1)


def _api_base(values: dict[str, str]) -> str:
    host = values.get("API_HOST", "127.0.0.1")
    port = values.get("API_PORT", "10907")
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


def cmd_auto_configure(args: argparse.Namespace) -> int:
    env_file = Path(args.env_file) if args.env_file else _default_env_file()
    if env_file.exists() and not args.force:
        print(f"Found existing env file at {env_file}. Use --force to overwrite.")
        return 0

    redis_port, api_port, metrics_port = _select_ports(_available_ports(args.use_indexed_ports))
    tmp_root = os.environ.get("KERNELGYM_TMP_ROOT", "/dev/shm/kernelgym")
    values = {
        "API_HOST": os.environ.get("API_HOST", _host_ip()),
        "API_PORT": str(api_port),
        "API_WORKERS": os.environ.get("API_WORKERS", "4"),
        "API_RELOAD": os.environ.get("API_RELOAD", "false"),
        "GPU_DEVICES": os.environ.get("GPU_DEVICES", _detect_gpus_json()),
        "GPU_MEMORY_LIMIT": os.environ.get("GPU_MEMORY_LIMIT", "16GB"),
        "NODE_ID": os.environ.get("NODE_ID", _hostname()),
        "REDIS_HOST": os.environ.get("REDIS_HOST", "localhost"),
        "REDIS_PORT": str(redis_port),
        "REDIS_DB": os.environ.get("REDIS_DB", "0"),
        "REDIS_PASSWORD": os.environ.get("REDIS_PASSWORD", ""),
        "REDIS_KEY_PREFIX": os.environ.get("REDIS_KEY_PREFIX", "kernelgym"),
        "WORKER_POOL_SIZE": os.environ.get("WORKER_POOL_SIZE", "1"),
        "MAX_TASKS_PER_WORKER": os.environ.get("MAX_TASKS_PER_WORKER", "1"),
        "DEFAULT_TOOLKIT": os.environ.get("DEFAULT_TOOLKIT", "kernelbench"),
        "DEFAULT_BACKEND_ADAPTER": os.environ.get("DEFAULT_BACKEND_ADAPTER", "kernelbench"),
        "DEFAULT_BACKEND": os.environ.get("DEFAULT_BACKEND", "triton"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "LOG_DIR": os.environ.get("LOG_DIR", f"logs/{_hostname()}"),
        "PY_LOG_DIR": os.environ.get("PY_LOG_DIR", f"py_logs/{_hostname()}"),
        "ENABLE_METRICS": os.environ.get("ENABLE_METRICS", "true"),
        "METRICS_PORT": str(metrics_port),
        "ENABLE_PROFILING": os.environ.get("ENABLE_PROFILING", "true"),
        "VERBOSE_ERROR_TRACEBACK": os.environ.get("VERBOSE_ERROR_TRACEBACK", "true"),
        "SAVE_EVAL_RESULTS": os.environ.get("SAVE_EVAL_RESULTS", "true" if args.save_eval_results else "false"),
        "EVAL_RESULTS_PATH": os.environ.get("EVAL_RESULTS_PATH", f"logs/{_hostname()}/eval_results.jsonl"),
        "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda-12.9"),
        "KERNELGYM_CUDA_AGENT_NVCC_THREADS": os.environ.get("KERNELGYM_CUDA_AGENT_NVCC_THREADS", "4"),
        "KERNELGYM_TVM_FFI_NVCC_THREADS": os.environ.get("KERNELGYM_TVM_FFI_NVCC_THREADS", "4"),
        "KERNELGYM_CUDA_AGENT_TMPDIR": os.environ.get("KERNELGYM_CUDA_AGENT_TMPDIR", f"{tmp_root}/work/cuda_agent"),
        "KERNELGYM_TVM_FFI_TMPDIR": os.environ.get("KERNELGYM_TVM_FFI_TMPDIR", f"{tmp_root}/work/tvm_ffi"),
        "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DIR": os.environ.get(
            "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DIR", f"{tmp_root}/compile_cache/cuda_agent"
        ),
        "KERNELGYM_TVM_FFI_COMPILE_CACHE_DIR": os.environ.get(
            "KERNELGYM_TVM_FFI_COMPILE_CACHE_DIR", f"{tmp_root}/compile_cache/tvm_ffi"
        ),
    }
    _write_env_file(env_file, values)
    print(f"Wrote configuration to {env_file}")
    return 0


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
    env_file = Path(args.env_file) if args.env_file else _default_env_file()
    values = _read_env_file(env_file)
    patterns = [
        ("kernelgym.server.api.server", "KernelGym API server"),
        ("kernelgym.worker.worker_monitor", "KernelGym worker monitor"),
        ("kernelgym.worker.single_worker", "KernelGym single workers"),
        ("kernelgym.worker.gpu_worker", "KernelGym worker manager"),
        ("uvicorn.*kernelgym", "Uvicorn server"),
        ("multiprocessing.spawn", "multiprocessing spawn workers"),
        ("multiprocessing.resource_tracker", "multiprocessing resource tracker"),
    ]
    for pattern, description in patterns:
        _kill_processes(pattern, description)

    client = _redis_client(values)
    if client is not None and values.get("REDIS_PORT"):
        prefix = values.get("REDIS_KEY_PREFIX", "kernelgym")
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
    env_file = Path(args.env_file) if args.env_file else _default_env_file()
    if args.force_config or not env_file.exists():
        cmd_auto_configure(
            argparse.Namespace(
                env_file=str(env_file),
                force=True,
                use_indexed_ports=args.use_indexed_ports,
                save_eval_results=False,
            )
        )
    if not args.no_stop_first:
        cmd_stop(argparse.Namespace(env_file=str(env_file)))

    values = _read_env_file(env_file)
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
    prefix = values.get("REDIS_KEY_PREFIX", "kernelgym")
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
    for required in ("API_HOST", "API_PORT", "REDIS_HOST", "REDIS_PORT"):
        if not values.get(required):
            raise SystemExit(f"Missing required env var in {env_file}: {required}")
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

    auto_configure = subparsers.add_parser("auto-configure", help="generate a KernelGym .env file")
    auto_configure.add_argument("--env-file", default=None)
    auto_configure.add_argument("--force", action="store_true")
    auto_configure.add_argument("--use-indexed-ports", action="store_true")
    auto_configure.add_argument("--save-eval-results", action="store_true")
    auto_configure.set_defaults(func=cmd_auto_configure)

    start_local = subparsers.add_parser("start-local", help="start local API, monitor, and GPU workers")
    start_local.add_argument("--env-file", default=None)
    start_local.add_argument("--force-config", action="store_true")
    start_local.add_argument("--use-indexed-ports", action="store_true")
    start_local.add_argument("--log-dir", default=None)
    start_local.add_argument("--eval-results-path", default=None)
    start_local.add_argument("--no-stop-first", action="store_true")
    start_local.set_defaults(func=cmd_start_local)

    worker_node = subparsers.add_parser("start-worker-node", help="start a worker-only node")
    worker_node.add_argument("server_env", nargs="?", default=str(ROOT_DIR / "server.env"))
    worker_node.set_defaults(func=cmd_start_worker_node)

    stop = subparsers.add_parser("stop", help="stop local KernelGym processes and clear Redis keys")
    stop.add_argument("--env-file", default=None)
    stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
