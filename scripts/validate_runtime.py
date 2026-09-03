"""Validate the runtime stack (CUDA toolchain + torch + redis-server).

Both the system toolchain (nvcc, used for compiling CUDA C++ extensions) and
the bundled CUDA runtime that torch ships with must be CUDA 12.9. The intranet
mirror serves the cu129-suffixed wheel, and the deployed GPU driver line is
sized for CUDA 12.9; mixing in a 13.x torch wheel against this driver silently
breaks at first CUDA touch, so the version check is strict.

The redis-server binary is also required because every reward node spawns a
local redis on REDIS_PORT for task coordination. We verify it's installed and
opportunistically ping the configured port — a successful ping is reported,
a failure is informational only because validate is typically run before the
service launches (and starts the daemon).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REQUIRED_CUDA = (12, 9)
PREFERRED_NVCC = Path("/usr/local/cuda-12.9/bin/nvcc")
PREFERRED_COMPUTE_SANITIZER = Path(
    os.environ.get("COMPUTE_SANITIZER_PATH", "/usr/local/cuda-12.9/bin/compute-sanitizer")
)
ENABLE_COMPUTE_SANITIZER = os.environ.get("ENABLE_COMPUTE_SANITIZER", "false").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_RELEASE_RE = re.compile(r"release (\d+)\.(\d+)")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "20110"))


def _parse_version(text: str) -> tuple[int, int] | None:
    """Pull (major, minor) out of a string like '12.9' or 'release 12.9, V12.9.86'."""
    match = _RELEASE_RE.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    parts = text.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def _check_torch_cuda() -> None:
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    torch_cuda = _parse_version(torch.version.cuda or "")
    if torch_cuda is None:
        raise SystemExit(f"could not parse torch.version.cuda={torch.version.cuda!r}")
    if torch_cuda != REQUIRED_CUDA:
        raise SystemExit(
            f"expected torch.version.cuda == {REQUIRED_CUDA[0]}.{REQUIRED_CUDA[1]}, got {torch.version.cuda!r}"
        )


def _check_nvcc() -> tuple[str, tuple[int, int]]:
    if PREFERRED_NVCC.exists():
        nvcc = str(PREFERRED_NVCC)
    else:
        located = shutil.which("nvcc")
        if not located:
            raise SystemExit(f"nvcc not found at {PREFERRED_NVCC} or on PATH")
        nvcc = located
    print(f"nvcc={nvcc}")
    out = subprocess.check_output([nvcc, "--version"], text=True)
    print(out.strip().splitlines()[-1])
    version = _parse_version(out)
    if version is None:
        raise SystemExit(f"could not parse nvcc release from:\n{out}")
    if version != REQUIRED_CUDA:
        raise SystemExit(
            f"expected nvcc release == {REQUIRED_CUDA[0]}.{REQUIRED_CUDA[1]}, got {version[0]}.{version[1]} at {nvcc}"
        )
    return nvcc, version


def _check_compute_sanitizer() -> str | None:
    if not ENABLE_COMPUTE_SANITIZER:
        print("compute_sanitizer=skipped (ENABLE_COMPUTE_SANITIZER=false)")
        return None
    if PREFERRED_COMPUTE_SANITIZER.exists():
        executable = str(PREFERRED_COMPUTE_SANITIZER)
    else:
        located = shutil.which("compute-sanitizer")
        if not located:
            raise SystemExit(
                "ENABLE_COMPUTE_SANITIZER=true but compute-sanitizer was not found at "
                f"{PREFERRED_COMPUTE_SANITIZER} or on PATH"
            )
        executable = located
    try:
        output = subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"compute-sanitizer --version failed at {executable}: {exc.output or exc}")
    version_line = next((line.strip() for line in reversed(output.splitlines()) if line.strip()), "unknown")
    print(f"compute_sanitizer={executable}")
    print(f"compute_sanitizer_version={version_line}")
    return version_line


def _check_cuda_init() -> int:
    # Driver vs torch-bundled-runtime mismatch only surfaces when something
    # actually touches CUDA. Force lazy init here so the validator fails fast
    # with the real driver error instead of letting a broken install pass.
    try:
        torch.cuda.init()
    except RuntimeError as exc:
        raise SystemExit(
            f"torch cannot initialize CUDA (torch built for {torch.version.cuda}, likely driver too old): {exc}"
        )
    device_count = torch.cuda.device_count()
    print(f"torch_cuda_device_count={device_count}")
    if device_count <= 0:
        raise SystemExit("torch.cuda reports zero devices after init")
    return device_count


def _check_cuda_math_libs() -> None:
    """Fail fast if a *required* CUDA math library the TVM-FFI build links is gone.

    Model-generated TVM-FFI extensions call cuBLAS/cuBLASLt/cuDNN/cuFFT/cuSPARSE/
    cuSOLVER/cuRAND/NVRTC host APIs directly, and the backend links those wheels
    by full path. If one is absent the produced ``.so`` carries undefined symbols
    (e.g. ``cublasCreate_v2``) that the dynamic linker only discovers at the first
    call inside ``ModelNew.forward`` — crashing the GPU worker, which the pool
    misreports as a task timeout. A missing *required* library aborts deploy; a
    missing *optional* one (proactive coverage, e.g. cuSPARSELt which not every
    torch build ships) only warns so we never block deploy on an optional wheel.

    Note: this validates the active deploy venv — i.e. the same interpreter the
    GPU workers launch from in the normal ``deploy_node.sh`` flow. A worker
    started with a different interpreter would not be covered by this check.
    """
    from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend

    report = KernelBenchTvmFfiBackend._resolve_cuda_math_libs()
    for package_name, lib_base, required, path in report:
        tier = "required" if required else "optional"
        print(f"cuda_math_lib {lib_base} ({package_name}, {tier})={path or 'MISSING'}")

    missing_optional = [lib_base for _pkg, lib_base, required, path in report if not required and path is None]
    if missing_optional:
        print(
            "cuda_math_libs: WARNING optional libraries not found (kernels that call them will "
            f"crash at runtime): {', '.join(missing_optional)}"
        )

    missing_required = [
        (package_name, lib_base) for package_name, lib_base, required, path in report if required and path is None
    ]
    if missing_required:
        detail = ", ".join(f"{lib_base} (from {package_name})" for package_name, lib_base in missing_required)
        raise SystemExit(
            "required CUDA math libraries not found for TVM-FFI linking: "
            f"{detail}. Install the matching nvidia-*-cu12 wheels into the venv."
        )

    resolved = sum(1 for _pkg, _lib_base, _required, path in report if path is not None)
    print(f"cuda_math_libs=OK ({resolved}/{len(report)} libraries resolved)")


def _check_redis() -> str:
    """Verify redis-server is installed; opportunistically ping the daemon.

    The binary check is fatal — we can't run reward without it. The ping is
    informational because validate is usually invoked before the service has
    started the daemon; if redis is already up, we report the version, else
    we note it's not yet running.
    """
    redis_server = shutil.which("redis-server")
    if not redis_server:
        raise SystemExit("redis-server not found on PATH")
    try:
        version_line = subprocess.check_output([redis_server, "--version"], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"redis-server --version failed: {exc}")
    print(f"redis_server={redis_server}")
    print(version_line)

    # Try to ping the configured port. Don't fail on connection refused.
    try:
        import redis  # noqa: WPS433 — imported lazily so the script still works without the pkg
    except ImportError:
        print("redis_ping=skipped (python redis package not installed)")
        return version_line
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        info = client.info("server")
        print(f"redis_ping=ok host={REDIS_HOST} port={REDIS_PORT} server_version={info.get('redis_version', '?')}")
    except Exception as exc:  # noqa: BLE001 — ping is best-effort
        print(f"redis_ping=not running host={REDIS_HOST} port={REDIS_PORT} ({type(exc).__name__})")
    return version_line


_SHIM_PROBE_SNIPPET = """
import ctypes, json, torch
device = torch.device("cuda:0")
x = torch.randn(256, 256, device=device)
import torch.profiler as profiler
with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
    y = x @ x
    torch.cuda.synchronize()
kernels = [e for e in prof.key_averages() if getattr(e, "device_time_total", 0) > 0]
state = int(ctypes.CDLL(None).kernelgym_cupti_tsc_shim_state())
version = int(ctypes.CDLL(None).kernelgym_cupti_tsc_shim_cupti_version())
print(json.dumps({"state": state, "cupti_version": version, "kernel_count": len(kernels)}))
"""


def _check_cupti_tsc_shim(device_count: int) -> None:
    """Build the CUPTI TSC shim and probe that it engages under LD_PRELOAD.

    Diagnostic gate only: a failure prints a WARNING and never blocks the
    deploy — at runtime, workers fall back to the legacy multi-forward
    profiling workaround whenever the shim is expected but not engaged.
    """
    print("\n=== Validate CUPTI TSC shim ===")
    from kernelgym.utils import cupti_tsc_shim

    shim_path = cupti_tsc_shim.ensure_shim_built()
    if shim_path is None:
        print("WARNING: CUPTI TSC shim build failed; legacy profiling workaround will stay active")
        return
    print(f"shim_artifact={shim_path}")
    if device_count <= 0:
        print("shim_probe=skipped (no CUDA devices)")
        return
    env = os.environ.copy()
    env["LD_PRELOAD"] = f"{shim_path}:{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else str(shim_path)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _SHIM_PROBE_SNIPPET],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        report = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.returncode == 0 else None
    except Exception as exc:
        print(f"WARNING: CUPTI TSC shim probe failed to run: {exc}")
        return
    if report is None:
        print(f"WARNING: CUPTI TSC shim probe exited {completed.returncode}: {completed.stderr[-800:]}")
        return
    print(
        f"shim_state={report['state']} shim_cupti_version={report['cupti_version']} "
        f"probe_kernels={report['kernel_count']}"
    )
    if cupti_tsc_shim.shim_state_healthy(report["state"]) and report["kernel_count"] > 0:
        print("shim_probe=OK")
    else:
        print(
            "WARNING: CUPTI TSC shim did not engage cleanly; workers will fall back "
            "to the legacy multi-forward profiling workaround at runtime"
        )


def main() -> int:
    print("\n=== Validate CUDA 12.9 + torch ===")
    _check_torch_cuda()
    nvcc, nvcc_version = _check_nvcc()
    _check_compute_sanitizer()
    device_count = _check_cuda_init()
    _check_cuda_math_libs()
    _check_redis()
    _check_cupti_tsc_shim(device_count)
    print(
        f"validate_runtime: OK — torch {torch.__version__} (cuda {torch.version.cuda}), "
        f"nvcc {nvcc_version[0]}.{nvcc_version[1]} at {nvcc}, {device_count} cuda device(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
