"""Validate that the active Python environment uses CUDA 12.9."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch


CUDA129_NVCC = Path("/usr/local/cuda-12.9/bin/nvcc")


def main() -> int:
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.version.cuda != "12.9":
        raise SystemExit(f"expected torch.version.cuda == 12.9, got {torch.version.cuda!r}")

    print(f"nvcc={CUDA129_NVCC}")
    if not CUDA129_NVCC.exists():
        raise SystemExit(f"nvcc not found at {CUDA129_NVCC}")

    out = subprocess.check_output([str(CUDA129_NVCC), "--version"], text=True)
    print(out.strip().splitlines()[-1])
    if "12.9" not in out:
        raise SystemExit(f"expected CUDA 12.9 nvcc at {CUDA129_NVCC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
