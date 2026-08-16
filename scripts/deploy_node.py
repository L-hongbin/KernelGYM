#!/usr/bin/env python
"""Container-only single/multi-node reward deployment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path


API_PORT = 20111
DEFAULT_STARTUP_WARMUP_TIMEOUT = 1800
NODE_WORKER_READY_TIMEOUT = 600
NODE_WORKER_HEARTBEAT_MAX_AGE = 120
WARMUP_CLIENT_TIMEOUT_GRACE = 60
ROOT_DIR = Path(__file__).resolve().parents[1]
REDIS_DATA_DIR = Path("/tmp/kernelgym-redis")
DEFAULT_CACHE_PATHS = (
    Path("/dev/shm/kernelgym/compile_cache"),
    Path("/dev/shm/kernelgym/work"),
)
CACHE_PATH_ENV_VARS = (
    "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE_DIR",
    "KERNELGYM_COMPILE_ARTIFACT_CACHE_DIR",
    "KERNELGYM_TVM_FFI_COMPILE_ARTIFACT_CACHE_DIR",
    "KERNELGYM_MODEL_TMPDIR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a KernelGym reward node inside an existing container")
    # Preferred, count-free hot-plug interface: the cluster has no fixed size.
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="start the primary so worker nodes can join later (enables remote Redis); no node count",
    )
    parser.add_argument(
        "--join",
        default="",
        metavar="PRIMARY_ADDR",
        help="join an already-running cluster as a worker node, connecting to PRIMARY_ADDR",
    )
    # Legacy torchrun-style flags (deprecated; kept working). --cluster/--join above
    # supersede these and need no nnodes/node-rank.
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=None)
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", type=int, default=API_PORT)
    parser.add_argument("--cpu-compile-workers", "--cpu-workers", type=int, default=None)
    parser.add_argument(
        "--gpu-devices",
        default=None,
        help="auto (default) or a comma/JSON list of container-logical CUDA device indices",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help=(
            "after stopping this node, delete local Redis persistence and KernelGym compile/work caches before start"
        ),
    )
    parser.add_argument(
        "--block-terminal",
        action="store_true",
        help=(
            "after startup succeeds, remain in the foreground; Ctrl-C, SIGTERM, or a terminal hangup "
            "stops this node's KernelGym services"
        ),
    )
    parser.add_argument(
        "--no-startup-warmup",
        action="store_true",
        help="skip the node-affined correctness and CUDA profiling warmup after workers become ready",
    )
    parser.add_argument(
        "--startup-warmup-timeout",
        type=int,
        default=DEFAULT_STARTUP_WARMUP_TIMEOUT,
        help="HTTP wall timeout for the startup warmup, including cold CPU compilation",
    )
    return parser.parse_args()


def local_ids() -> set[str]:
    # Gather hostnames and IPs visible from inside the container for master detection.
    ids = {"localhost", "127.0.0.1"}
    hostname = socket.gethostname()
    if hostname:
        ids.add(hostname)
    fqdn = socket.getfqdn()
    if fqdn:
        ids.add(fqdn)
    try:
        ids.update(socket.gethostbyname_ex(hostname)[2])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("1.1.1.1", 80))
        ids.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        proc = subprocess.run(["hostname", "-I"], check=False, text=True, stdout=subprocess.PIPE)
        ids.update(item for item in proc.stdout.split() if item)
    except OSError:
        pass
    return {item for item in ids if item}


def run(command: list[str], *, allow_failure: bool = False) -> None:
    # Keep subprocess handling explicit so deployment failures surface directly.
    proc = subprocess.run(command, check=False)
    if proc.returncode and not allow_failure:
        raise SystemExit(proc.returncode)


def _remove_path(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
            print(f"Removed cache file: {path}")
        elif path.is_dir():
            shutil.rmtree(path)
            print(f"Removed cache directory: {path}")
    except FileNotFoundError:
        return


def clear_local_caches() -> None:
    """Delete local runtime caches after workers and Redis have been stopped."""
    paths: list[Path] = [*DEFAULT_CACHE_PATHS, REDIS_DATA_DIR, ROOT_DIR / "dump.rdb"]
    for env_var in CACHE_PATH_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            paths.append(Path(value).expanduser())

    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        _remove_path(resolved)


def wait_api(master_addr: str) -> None:
    # Worker nodes should not start until the primary API is reachable.
    url = f"http://{master_addr}:{API_PORT}/health"
    deadline = time.time() + 180
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    print(f"API ready: {url}")
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(3)
    raise SystemExit(f"API did not become ready at {url}: {last_error}")


def _http_get_json(url: str, timeout: float = 10) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _redis_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _worker_registration_is_current(worker: dict[str, object]) -> bool:
    if str(worker.get("status") or "").strip().lower() != "online":
        return False
    try:
        heartbeat = datetime.fromisoformat(str(worker.get("last_heartbeat") or ""))
    except ValueError:
        return False
    now = datetime.now(tz=heartbeat.tzinfo)
    return (now - heartbeat).total_seconds() <= NODE_WORKER_HEARTBEAT_MAX_AGE


def wait_node_workers(api_host: str, target_hostname: str, timeout: int = NODE_WORKER_READY_TIMEOUT) -> None:
    """Wait until every current GPU/CPU worker registration on this host is usable."""
    url = f"http://{api_host}:{API_PORT}/workers/status"
    deadline = time.monotonic() + timeout
    last_state = "no matching workers registered"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status = _http_get_json(url)
            matching = [value for value in status.values() if value.get("hostname") == target_hostname]
            current = [value for value in matching if _worker_registration_is_current(value)]
            stale_count = len(matching) - len(current)
            gpu_workers = [value for value in current if str(value.get("device") or "").startswith("cuda:")]
            cpu_workers = [value for value in current if value.get("device") == "cpu"]
            gpu_ready = [
                value
                for value in gpu_workers
                # degraded_check is a transient post-CUDA-fault state. It may
                # be scheduler-admissible for one pre-fault spare, but a node
                # is not deployment-ready until fresh-context validation has
                # restored every launched GPU worker to fully healthy.
                if _redis_bool(value.get("online"))
                and value.get("health_state") == "healthy"
                and _redis_bool(value.get("accepting_tasks"))
            ]
            cpu_ready = [value for value in cpu_workers if _redis_bool(value.get("online"))]
            last_state = (
                f"gpu={len(gpu_ready)}/{len(gpu_workers)} ready, cpu={len(cpu_ready)}/{len(cpu_workers)} online, "
                f"stale_or_offline={stale_count}"
            )
            if (
                gpu_workers
                and cpu_workers
                and len(gpu_ready) == len(gpu_workers)
                and len(cpu_ready) == len(cpu_workers)
            ):
                print(f"Node workers ready: hostname={target_hostname} {last_state}")
                return
        except Exception as exc:  # noqa: BLE001 - surface the last readiness failure
            last_error = exc
        time.sleep(3)
    suffix = f"; last_error={last_error}" if last_error else ""
    raise SystemExit(f"Node workers did not become ready at {url}: {last_state}{suffix}")


def run_startup_warmup(api_host: str, request_timeout: int) -> None:
    target_hostname = socket.gethostname()
    section(f"Startup profiling warmup hostname={target_hostname}")
    wait_node_workers(api_host, target_hostname)
    safe_hostname = "".join(char if char.isalnum() else "-" for char in target_hostname).strip("-") or "node"
    task_id = f"startup_warmup_{safe_hostname}_{uuid.uuid4().hex[:12]}"
    # Leave room for the HTTP response while allowing the configurable wall
    # timeout to expand the cold compile budget beyond the old fixed 600s cap.
    task_timeout = max(60, request_timeout - WARMUP_CLIENT_TIMEOUT_GRACE)
    run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "test_reward.py"),
            "--host",
            api_host,
            "--port",
            str(API_PORT),
            "--timeout",
            str(task_timeout),
            "--client-timeout",
            str(request_timeout),
            "--task-id",
            task_id,
            "--target-hostname",
            target_hostname,
            "--require-correct",
        ]
    )


def section(title: str) -> None:
    """Print a visual section break so deploy phases are easy to scan in the log."""
    print()
    print(f"=== {title} ===")


def wait_for_shutdown_signal() -> signal.Signals:
    """Wait without a signal-delivery race and return the requested shutdown signal."""
    stop_requested = threading.Event()
    received_signal = signal.SIGTERM
    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}

    def request_shutdown(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signal.Signals(signum)
        stop_requested.set()

    for signum in handled_signals:
        signal.signal(signum, request_shutdown)
    try:
        while not stop_requested.wait(timeout=3600):
            pass
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
    return received_signal


def block_terminal() -> None:
    """Keep the deploy command in the foreground and own local service cleanup."""
    section("Block terminal")
    print("KernelGym is ready. Waiting in the foreground; press Ctrl-C to stop this node.", flush=True)
    received_signal = wait_for_shutdown_signal()
    print(f"Received {received_signal.name}; stopping this node's KernelGym services.", flush=True)
    run([sys.executable, "-m", "kernelgym.cli.service", "stop", "--profile", "v1"])


def finish_deployment(args: argparse.Namespace, *, api_host: str | None = None) -> int:
    if api_host and not bool(getattr(args, "no_startup_warmup", False)):
        run_startup_warmup(api_host, int(getattr(args, "startup_warmup_timeout", DEFAULT_STARTUP_WARMUP_TIMEOUT)))
    if bool(getattr(args, "block_terminal", False)):
        block_terminal()
    return 0


def _append_runtime_overrides(
    command: list[str],
    cpu_compile_workers: int | None,
    gpu_devices: str | None,
) -> list[str]:
    updated = list(command)
    if cpu_compile_workers is not None:
        updated += ["--cpu-compile-workers", str(cpu_compile_workers)]
    if gpu_devices is not None:
        updated += ["--gpu-devices", gpu_devices]
    return updated


def start_primary(
    node_rank: int | None,
    cpu_compile_workers: int | None = None,
    gpu_devices: str | None = None,
    *,
    redis_remote_access: bool = False,
    clear_cache: bool = False,
) -> None:
    # Rank 0 runs API, Redis, monitor, and local workers through the service CLI.
    section(f"Start primary node rank={node_rank if node_rank is not None else 'auto'}")
    if clear_cache:
        run([sys.executable, "-m", "kernelgym.cli.service", "stop", "--profile", "v1"], allow_failure=True)
        clear_local_caches()
    command = [sys.executable, "-m", "kernelgym.cli.service", "start-local", "--profile", "v1"]
    if redis_remote_access:
        command.append("--redis-remote-access")
    if clear_cache:
        command.append("--no-stop-first")
    run(
        _append_runtime_overrides(
            command,
            cpu_compile_workers,
            gpu_devices,
        )
    )
    section("Wait for API")
    wait_api("127.0.0.1")


def start_worker(
    master_addr: str,
    node_rank: int | None,
    cpu_compile_workers: int | None = None,
    gpu_devices: str | None = None,
    *,
    clear_cache: bool = False,
) -> None:
    # Worker configuration is generated by kernelgym.cli.service from deployment_profiles.py.
    rank_label = node_rank if node_rank is not None else "auto"
    section(f"Start worker node rank={rank_label} master={master_addr}:{API_PORT}")
    run([sys.executable, "-m", "kernelgym.cli.service", "stop", "--profile", "v1"], allow_failure=True)
    if clear_cache:
        clear_local_caches()
    section("Wait for primary API")
    wait_api(master_addr)
    section("Launch worker-only services")
    command = [
        sys.executable,
        "-m",
        "kernelgym.cli.service",
        "start-worker-node",
        "--profile",
        "v1",
        "--master-addr",
        master_addr,
    ]
    # No rank -> the server auto-allocates a stable per-hostname node id.
    if node_rank is not None:
        command += ["--node-rank", str(node_rank)]
    run(_append_runtime_overrides(command, cpu_compile_workers, gpu_devices))


def validate(args: argparse.Namespace) -> None:
    # Validate topology before touching running services.
    if args.master_port != API_PORT:
        raise SystemExit(f"--master-port must be {API_PORT}; service ports are fixed in this repo")
    cpu_compile_workers = getattr(args, "cpu_compile_workers", None)
    if cpu_compile_workers is not None and cpu_compile_workers < 0:
        raise SystemExit("--cpu-compile-workers must be >= 0")
    if cpu_compile_workers == 0 and not bool(getattr(args, "no_startup_warmup", False)):
        raise SystemExit("--cpu-compile-workers 0 requires --no-startup-warmup")
    startup_warmup_timeout = int(getattr(args, "startup_warmup_timeout", DEFAULT_STARTUP_WARMUP_TIMEOUT))
    if startup_warmup_timeout < 60:
        raise SystemExit("--startup-warmup-timeout must be >= 60 seconds")

    # Preferred count-free interface.
    cluster = bool(getattr(args, "cluster", False))
    join = str(getattr(args, "join", "") or "")
    if cluster and join:
        raise SystemExit("--cluster and --join are mutually exclusive")
    if cluster or join:
        legacy = args.nnodes != 1 or args.node_rank is not None or args.master_addr
        if legacy:
            raise SystemExit("--cluster/--join supersede --nnodes/--node-rank/--master-addr; do not combine them")
        if join and join in local_ids():
            raise SystemExit(
                "--join expects the PRIMARY's address; run `deploy_node.sh --cluster` on the primary instead"
            )
        return

    # Legacy torchrun-style path.
    if args.nnodes < 1:
        raise SystemExit("--nnodes must be a positive integer")
    if args.nnodes == 1:
        return
    if not args.master_addr:
        raise SystemExit("--master-addr is required when --nnodes > 1")
    if args.node_rank is None:
        raise SystemExit("--node-rank is required when --nnodes > 1")
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        raise SystemExit("--node-rank must be an integer in [0, nnodes)")


def main() -> int:
    args = parse_args()
    validate(args)
    cluster = bool(getattr(args, "cluster", False))
    join = str(getattr(args, "join", "") or "")
    clear_cache = bool(getattr(args, "clear_cache", False))
    gpu_devices = getattr(args, "gpu_devices", None)

    # Preferred count-free interface: no nnodes, no node-rank.
    if cluster:
        start_primary(
            None,
            args.cpu_compile_workers,
            gpu_devices,
            redis_remote_access=True,
            clear_cache=clear_cache,
        )
        return finish_deployment(args, api_host="127.0.0.1")
    if join:
        start_worker(join, None, args.cpu_compile_workers, gpu_devices, clear_cache=clear_cache)
        return finish_deployment(args, api_host=join)

    # Legacy torchrun-style path (deprecated).
    if args.nnodes != 1 or args.node_rank is not None or args.master_addr:
        print(
            "[deploy_node] note: --nnodes/--node-rank are deprecated; prefer --cluster (primary) and --join <addr> (worker)."
        )
    if args.nnodes == 1:
        start_primary(args.node_rank, args.cpu_compile_workers, gpu_devices, clear_cache=clear_cache)
        return finish_deployment(args, api_host="127.0.0.1")

    # Role is determined by whether this container can see itself as --master-addr.
    is_master = args.master_addr in local_ids()
    if is_master and args.node_rank != 0:
        raise SystemExit("The node matching --master-addr must use --node-rank 0")
    if not is_master and args.node_rank == 0:
        raise SystemExit("--node-rank 0 must run on the node matching --master-addr")
    if is_master:
        start_primary(
            args.node_rank,
            args.cpu_compile_workers,
            gpu_devices,
            redis_remote_access=True,
            clear_cache=clear_cache,
        )
    else:
        start_worker(
            args.master_addr,
            args.node_rank,
            args.cpu_compile_workers,
            gpu_devices,
            clear_cache=clear_cache,
        )
    return finish_deployment(args, api_host="127.0.0.1" if is_master else args.master_addr)


if __name__ == "__main__":
    raise SystemExit(main())
