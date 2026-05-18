"""Validate that the active Python environment uses CUDA 12.9."""

from __future__ import annotations

import shutil
import subprocess
import sys

import torch


def main() -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
