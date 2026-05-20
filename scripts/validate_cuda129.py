"""Validate that the active Python environment uses CUDA 12.9 exactly.

Both the system toolchain (nvcc, used for compiling CUDA C++ extensions) and
the bundled CUDA runtime that torch ships with must be CUDA 12.9. The intranet
mirror serves the cu129-suffixed wheel, and the deployed GPU driver line is
sized for CUDA 12.9; mixing in a 13.x torch wheel against this driver
silently breaks at first CUDA touch, so the version check is strict.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REQUIRED_VERSION = (12, 9)
PREFERRED_NVCC = Path("/usr/local/cuda-12.9/bin/nvcc")
_RELEASE_RE = re.compile(r"release (\d+)\.(\d+)")


def _parse_version(text: str) -> tuple[int, int] | None:
    """Pull (major, minor) out of a string like '12.9' or 'release 12.9, V12.9.86'."""
    match = _RELEASE_RE.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    parts = text.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def main() -> int:
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    torch_cuda = _parse_version(torch.version.cuda or "")
    if torch_cuda is None:
        raise SystemExit(f"could not parse torch.version.cuda={torch.version.cuda!r}")
    if torch_cuda != REQUIRED_VERSION:
        raise SystemExit(
            f"expected torch.version.cuda == {REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]}, got {torch.version.cuda!r}"
        )

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
    nvcc_version = _parse_version(out)
    if nvcc_version is None:
        raise SystemExit(f"could not parse nvcc release from:\n{out}")
    if nvcc_version != REQUIRED_VERSION:
        raise SystemExit(
            f"expected nvcc release == {REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]}, got {nvcc_version[0]}.{nvcc_version[1]} at {nvcc}"
        )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
