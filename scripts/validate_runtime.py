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

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REQUIRED_CUDA = (12, 9)
PREFERRED_NVCC = Path("/usr/local/cuda-12.9/bin/nvcc")
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


def main() -> int:
    print("\n=== Validate CUDA 12.9 + torch ===")
    _check_torch_cuda()
    nvcc, nvcc_version = _check_nvcc()
    device_count = _check_cuda_init()
    _check_redis()
    print(
        f"validate_runtime: OK — torch {torch.__version__} (cuda {torch.version.cuda}), "
        f"nvcc {nvcc_version[0]}.{nvcc_version[1]} at {nvcc}, {device_count} cuda device(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
