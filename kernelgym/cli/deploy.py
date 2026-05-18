"""Deployment helpers for reward-only KernelGym.

This module intentionally keeps deployment logic in Python. Shell wrappers may
delegate here, but host preparation, CUDA environment creation, and Docker
command construction should stay testable here.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
from pathlib import Path

from kernelgym.cli import service


ROOT_DIR = Path(__file__).resolve().parents[2]

CUDA129_INDEX_URL = "https://download.pytorch.org/whl/cu129"
TORCH_CUDA129_PACKAGES = (
    "torch==2.11.0+cu129",
    "torchvision==0.26.0+cu129",
)
DEFAULT_CUDA_HOME = "/usr/local/cuda-12.9"
DEFAULT_CONTAINER_IMAGE = "192.168.14.129:80/fm/llmc:v1.1"
DEFAULT_SHM_SIZE = "256g"
DEFAULT_MARKER_PATH = "/ms"
DEFAULT_PROXY = "http://192.168.28.186:7897"
PROFILE_INTERNAL = "internal"
PROFILE_EXTERNAL = "external"
PROFILE_AUTO = "auto"


def _host_ip() -> str:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        address = probe.getsockname()[0]
        probe.close()
        return address
    except OSError:
        return "127.0.0.1"


def _detect_network_profile(marker_path: str | Path = DEFAULT_MARKER_PATH) -> str:
    marker = Path(marker_path)
    if marker.is_symlink():
        return PROFILE_EXTERNAL
    if marker.exists():
        return PROFILE_INTERNAL
    return PROFILE_EXTERNAL


def _resolve_profile(profile: str, marker_path: str | Path = DEFAULT_MARKER_PATH) -> str:
    if profile == PROFILE_AUTO:
        return _detect_network_profile(marker_path)
    if profile not in {PROFILE_INTERNAL, PROFILE_EXTERNAL}:
        raise SystemExit(f"Unknown deployment profile: {profile}")
    return profile


def _profile_base_env(profile: str) -> dict[str, str]:
    common = {
        "KERNELGYM_DEPLOYMENT_PROFILE": profile,
        "CUDA_HOME": DEFAULT_CUDA_HOME,
        "GPU_DEVICES": os.environ.get("GPU_DEVICES", service._detect_gpus_json()),
        "GPU_MEMORY_LIMIT": os.environ.get("GPU_MEMORY_LIMIT", "16GB"),
        "API_WORKERS": os.environ.get("API_WORKERS", "4"),
        "API_RELOAD": os.environ.get("API_RELOAD", "false"),
        "REDIS_DB": os.environ.get("REDIS_DB", "0"),
        "REDIS_PASSWORD": os.environ.get("REDIS_PASSWORD", ""),
        "WORKER_POOL_SIZE": os.environ.get("WORKER_POOL_SIZE", "1"),
        "MAX_TASKS_PER_WORKER": os.environ.get("MAX_TASKS_PER_WORKER", "1"),
        "DEFAULT_TOOLKIT": os.environ.get("DEFAULT_TOOLKIT", "kernelbench"),
        "DEFAULT_BACKEND_ADAPTER": os.environ.get("DEFAULT_BACKEND_ADAPTER", "kernelbench"),
        "DEFAULT_BACKEND": os.environ.get("DEFAULT_BACKEND", "triton"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "ENABLE_METRICS": os.environ.get("ENABLE_METRICS", "true"),
        "ENABLE_PROFILING": os.environ.get("ENABLE_PROFILING", "true"),
        "VERBOSE_ERROR_TRACEBACK": os.environ.get("VERBOSE_ERROR_TRACEBACK", "true"),
        "SAVE_EVAL_RESULTS": os.environ.get("SAVE_EVAL_RESULTS", "false"),
        "KERNELGYM_CUDA_AGENT_NVCC_THREADS": os.environ.get("KERNELGYM_CUDA_AGENT_NVCC_THREADS", "4"),
        "KERNELGYM_TVM_FFI_NVCC_THREADS": os.environ.get("KERNELGYM_TVM_FFI_NVCC_THREADS", "4"),
        "KERNELGYM_CUDA_AGENT_TMPDIR": os.environ.get(
            "KERNELGYM_CUDA_AGENT_TMPDIR", "/dev/shm/kernelgym/work/cuda_agent"
        ),
        "KERNELGYM_TVM_FFI_TMPDIR": os.environ.get("KERNELGYM_TVM_FFI_TMPDIR", "/dev/shm/kernelgym/work/tvm_ffi"),
        "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DIR": os.environ.get(
            "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DIR", "/dev/shm/kernelgym/compile_cache/cuda_agent"
        ),
        "KERNELGYM_TVM_FFI_COMPILE_CACHE_DIR": os.environ.get(
            "KERNELGYM_TVM_FFI_COMPILE_CACHE_DIR", "/dev/shm/kernelgym/compile_cache/tvm_ffi"
        ),
    }
    if profile == PROFILE_EXTERNAL:
        common.update(
            {
                "KERNELGYM_SSH_RUNTIME": "physical_host",
                "KERNELGYM_CONTAINER_REQUIRED": "true",
                "KERNELGYM_LOCK_GPU_CLOCKS": "true",
                "API_PORT": os.environ.get("API_PORT", "8111"),
                "REDIS_PORT": os.environ.get("REDIS_PORT", "8110"),
                "METRICS_PORT": os.environ.get("METRICS_PORT", "8112"),
                "REDIS_KEY_PREFIX": os.environ.get("REDIS_KEY_PREFIX", "kernelgym_external"),
            }
        )
    else:
        common.update(
            {
                "KERNELGYM_SSH_RUNTIME": "container",
                "KERNELGYM_CONTAINER_REQUIRED": "false",
                "KERNELGYM_LOCK_GPU_CLOCKS": "false",
                "API_PORT": os.environ.get("API_PORT", "10907"),
                "REDIS_PORT": os.environ.get("REDIS_PORT", "10906"),
                "METRICS_PORT": os.environ.get("METRICS_PORT", "10908"),
                "REDIS_KEY_PREFIX": os.environ.get("REDIS_KEY_PREFIX", "kernelgym_internal"),
            }
        )
    return common


def _deployment_env(
    profile: str,
    *,
    role: str,
    server_host: str | None = None,
    api_host: str | None = None,
    node_id: str | None = None,
) -> dict[str, str]:
    if role not in {"api", "worker"}:
        raise SystemExit(f"Unknown env role: {role}")
    values = _profile_base_env(profile)
    detected_host = _host_ip()
    if role == "api":
        values["API_HOST"] = api_host or os.environ.get("API_HOST", detected_host)
        values["REDIS_HOST"] = os.environ.get("REDIS_HOST", "localhost")
    else:
        remote_host = server_host or os.environ.get("KERNELGYM_SERVER_HOST") or detected_host
        values["API_HOST"] = api_host or os.environ.get("API_HOST", remote_host)
        values["REDIS_HOST"] = os.environ.get("REDIS_HOST", remote_host)
    values["NODE_ID"] = node_id or os.environ.get("NODE_ID", f"reward-{profile}-{role}-{socket.gethostname()}")
    values["LOG_DIR"] = os.environ.get("LOG_DIR", f"logs/{values['NODE_ID']}")
    values["PY_LOG_DIR"] = os.environ.get("PY_LOG_DIR", f"py_logs/{values['NODE_ID']}")
    values["EVAL_RESULTS_PATH"] = os.environ.get("EVAL_RESULTS_PATH", f"logs/{values['NODE_ID']}/eval_results.jsonl")
    return values


def _run(
    command: list[str],
    *,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    print("+ " + " ".join(command))
    if dry_run:
        return None
    return subprocess.run(command, check=check, env=env, text=True)


def _with_proxy(env: dict[str, str], proxy: str | None) -> dict[str, str]:
    if not proxy:
        return env
    updated = env.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        updated[key] = proxy
    return updated


def _uv_command(uv: str, python: str, *, dry_run: bool = False, env: dict[str, str] | None = None) -> list[str]:
    if shutil.which(uv):
        return [uv]
    if uv != "uv":
        return [uv]
    _run([python, "-m", "pip", "install", "uv"], dry_run=dry_run, env=env)
    return [python, "-m", "uv"]


def _cuda_env(cuda_home: str, *, proxy: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    cuda_bin = str(Path(cuda_home) / "bin")
    cuda_lib = str(Path(cuda_home) / "lib64")
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return _with_proxy(env, proxy)


def _venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def _validate_cuda129_environment(
    venv: Path,
    cuda_home: str,
    *,
    dry_run: bool = False,
    proxy: str | None = None,
) -> None:
    script = """
import shutil
import subprocess
import sys

import torch

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
if torch.version.cuda != "12.9":
    raise SystemExit(f"expected torch.version.cuda == 12.9, got {torch.version.cuda!r}")

nvcc = shutil.which("nvcc")
print(f"nvcc={nvcc}")
if not nvcc:
    raise SystemExit("nvcc not found on PATH")
out = subprocess.check_output([nvcc, "--version"], text=True)
print(out.strip().splitlines()[-1])
if "12.9" not in out:
    raise SystemExit("expected nvcc from CUDA 12.9")
"""
    command = [str(_venv_python(venv)), "-c", script]
    _run(command, dry_run=dry_run, env=_cuda_env(cuda_home, proxy=proxy))


def cmd_create_venv(args: argparse.Namespace) -> int:
    venv = Path(args.venv)
    cuda_home = args.cuda_home
    proxy = args.proxy or os.environ.get("KERNELGYM_PROXY") or None
    fallback_proxy = args.fallback_proxy or os.environ.get("KERNELGYM_FALLBACK_PROXY") or DEFAULT_PROXY
    uv_command = _uv_command(args.uv, args.python, dry_run=args.dry_run, env=_cuda_env(cuda_home, proxy=proxy))
    if args.recreate and venv.exists():
        print(f"Removing existing venv: {venv}")
        if not args.dry_run:
            shutil.rmtree(venv)

    if not venv.exists():
        _run(
            [*uv_command, "venv", "--python", args.python, str(venv)],
            dry_run=args.dry_run,
        )

    _run(
        [
            *uv_command,
            "pip",
            "install",
            "--python",
            str(_venv_python(venv)),
            "-e",
            ".[dev]",
        ],
        dry_run=args.dry_run,
        env=_cuda_env(cuda_home, proxy=proxy),
    )
    torch_install_command = [
        *uv_command,
        "pip",
        "install",
        "--python",
        str(_venv_python(venv)),
        "--index-url",
        CUDA129_INDEX_URL,
        *TORCH_CUDA129_PACKAGES,
    ]
    try:
        _run(torch_install_command, dry_run=args.dry_run, env=_cuda_env(cuda_home, proxy=proxy))
    except subprocess.CalledProcessError:
        if proxy or not fallback_proxy:
            raise
        print(f"PyTorch CUDA 12.9 install failed; retrying with proxy {fallback_proxy}")
        _run(torch_install_command, dry_run=args.dry_run, env=_cuda_env(cuda_home, proxy=fallback_proxy))
        proxy = fallback_proxy
    if not args.skip_validate:
        _validate_cuda129_environment(venv, cuda_home, dry_run=args.dry_run, proxy=proxy)
    return 0


def _sudo_prefix(use_sudo: bool) -> list[str]:
    return ["sudo"] if use_sudo else []


def cmd_lock_gpu_clocks(args: argparse.Namespace) -> int:
    nvidia_smi = args.nvidia_smi
    prefix = _sudo_prefix(args.sudo)
    if args.persistence:
        _run([*prefix, nvidia_smi, "-pm", "1"], dry_run=args.dry_run)
    if args.gpu_clock:
        clock = str(args.gpu_clock)
        _run([*prefix, nvidia_smi, "-lgc", f"{clock},{clock}"], dry_run=args.dry_run)
    if args.power_limit:
        _run([*prefix, nvidia_smi, "-pl", str(args.power_limit)], dry_run=args.dry_run)
    return 0


def _docker_run_command(args: argparse.Namespace) -> list[str]:
    repo_dir = str(Path(args.repo_dir).resolve())
    cuda_home = str(Path(args.cuda_home).resolve())
    command = [
        args.docker,
        "run",
        "-d",
        "--name",
        args.name,
        "--gpus",
        args.gpus,
        "--network",
        "host",
        "--privileged",
        "-v",
        "/nfs:/nfs",
        "-v",
        f"{cuda_home}:{cuda_home}:ro",
        "-e",
        f"CUDA_HOME={cuda_home}",
        "-e",
        f"PATH={cuda_home}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "-e",
        f"LD_LIBRARY_PATH={cuda_home}/lib64",
        "-w",
        repo_dir,
    ]
    if args.exec_shm:
        command.extend(["--tmpfs", f"/dev/shm:rw,nosuid,nodev,exec,size={args.shm_size}"])
    else:
        command.extend(["--shm-size", args.shm_size])
    for mount in args.mount:
        command.extend(["-v", mount])
    for env_var in args.env:
        command.extend(["-e", env_var])
    command.extend([args.image, "sleep", "infinity"])
    return command


def cmd_host_container(args: argparse.Namespace) -> int:
    prefix = _sudo_prefix(args.sudo)
    if args.lock_gpu_clocks:
        cmd_lock_gpu_clocks(
            argparse.Namespace(
                nvidia_smi=args.nvidia_smi,
                sudo=args.sudo,
                persistence=True,
                gpu_clock=args.gpu_clock,
                power_limit=args.power_limit,
                dry_run=args.dry_run,
            )
        )
    if args.replace:
        _run([*prefix, args.docker, "rm", "-f", args.name], dry_run=args.dry_run, check=False)
    _run([*prefix, *_docker_run_command(args)], dry_run=args.dry_run)
    print(f"Container {args.name} prepared. Enter it with:")
    print(f"  {args.docker} exec -it {args.name} bash")
    return 0


def cmd_detect_profile(args: argparse.Namespace) -> int:
    profile = _detect_network_profile(args.marker_path)
    print(profile)
    return 0


def cmd_write_env(args: argparse.Namespace) -> int:
    profile = _resolve_profile(args.profile, args.marker_path)
    env_file = Path(args.env_file)
    if env_file.exists() and not args.force:
        raise SystemExit(f"Env file already exists: {env_file}. Use --force to overwrite.")
    values = _deployment_env(
        profile,
        role=args.role,
        server_host=args.server_host,
        api_host=args.api_host,
        node_id=args.node_id,
    )
    service._write_env_file(env_file, values)
    print(f"Wrote {profile} {args.role} env to {env_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare reward-only KernelGym deployments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_venv = subparsers.add_parser("create-venv", help="create a uv .venv with PyTorch CUDA 12.9")
    create_venv.add_argument("--venv", default=str(ROOT_DIR / ".venv"))
    create_venv.add_argument("--python", default="python3.10")
    create_venv.add_argument("--uv", default="uv")
    create_venv.add_argument("--cuda-home", default=DEFAULT_CUDA_HOME)
    create_venv.add_argument("--proxy", default=None)
    create_venv.add_argument("--fallback-proxy", default=DEFAULT_PROXY)
    create_venv.add_argument("--recreate", action="store_true")
    create_venv.add_argument("--skip-validate", action="store_true")
    create_venv.add_argument("--dry-run", action="store_true")
    create_venv.set_defaults(func=cmd_create_venv)

    detect_profile = subparsers.add_parser(
        "detect-profile",
        help="detect internal/external profile from the real /ms path",
    )
    detect_profile.add_argument("--marker-path", default=DEFAULT_MARKER_PATH)
    detect_profile.set_defaults(func=cmd_detect_profile)

    write_env = subparsers.add_parser("write-env", help="write an env file for the detected deployment profile")
    write_env.add_argument("--env-file", default=str(ROOT_DIR / ".env"))
    write_env.add_argument(
        "--profile", choices=[PROFILE_AUTO, PROFILE_INTERNAL, PROFILE_EXTERNAL], default=PROFILE_AUTO
    )
    write_env.add_argument("--marker-path", default=DEFAULT_MARKER_PATH)
    write_env.add_argument("--role", choices=["api", "worker"], default="api")
    write_env.add_argument("--server-host", default=None)
    write_env.add_argument("--api-host", default=None)
    write_env.add_argument("--node-id", default=None)
    write_env.add_argument("--force", action="store_true")
    write_env.set_defaults(func=cmd_write_env)

    lock_clocks = subparsers.add_parser("lock-gpu-clocks", help="lock GPU clocks on a physical host")
    lock_clocks.add_argument("--nvidia-smi", default="nvidia-smi")
    lock_clocks.add_argument("--sudo", action="store_true")
    lock_clocks.add_argument("--no-persistence", dest="persistence", action="store_false")
    lock_clocks.add_argument("--gpu-clock", type=int, default=2700)
    lock_clocks.add_argument("--power-limit", type=int, default=400)
    lock_clocks.add_argument("--dry-run", action="store_true")
    lock_clocks.set_defaults(func=cmd_lock_gpu_clocks, persistence=True)

    host_container = subparsers.add_parser(
        "host-container",
        help="start a reward container from a physical host",
    )
    host_container.add_argument("--name", required=True)
    host_container.add_argument("--image", default=DEFAULT_CONTAINER_IMAGE)
    host_container.add_argument("--repo-dir", default=str(ROOT_DIR))
    host_container.add_argument("--cuda-home", default=DEFAULT_CUDA_HOME)
    host_container.add_argument("--docker", default="docker")
    host_container.add_argument("--nvidia-smi", default="nvidia-smi")
    host_container.add_argument("--gpus", default="all")
    host_container.add_argument("--shm-size", default=DEFAULT_SHM_SIZE)
    host_container.add_argument("--no-exec-shm", dest="exec_shm", action="store_false")
    host_container.add_argument("--mount", action="append", default=[])
    host_container.add_argument("--env", action="append", default=[])
    host_container.add_argument("--sudo", action="store_true")
    host_container.add_argument("--replace", action="store_true")
    host_container.add_argument("--lock-gpu-clocks", action="store_true")
    host_container.add_argument("--gpu-clock", type=int, default=2700)
    host_container.add_argument("--power-limit", type=int, default=400)
    host_container.add_argument("--dry-run", action="store_true")
    host_container.set_defaults(func=cmd_host_container, exec_shm=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
