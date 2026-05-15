"""CUDA device information helpers."""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional

import torch


def _normalize_arch_token(token: str) -> Optional[str]:
    token = token.strip()
    if not token:
        return None

    token = token.upper().removesuffix("+PTX")
    token = token.lower().removeprefix("sm_").removeprefix("compute_")

    if re.fullmatch(r"\d+\.\d+", token):
        return token

    if token.isdigit() and len(token) >= 2:
        return f"{int(token[:-1])}.{int(token[-1])}"

    return token


def get_requested_compute_capability() -> Optional[str]:
    """Return normalized TORCH_CUDA_ARCH_LIST, if it was configured."""

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if not arch_list:
        return None

    tokens = [
        normalized
        for raw in re.split(r"[;,\s]+", arch_list)
        if (normalized := _normalize_arch_token(raw))
    ]
    if not tokens:
        return arch_list

    return ";".join(tokens)


@lru_cache(maxsize=1)
def get_cuda_driver_version() -> Optional[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            versions = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip() and "not supported" not in line.lower()
            ]
            if versions:
                return versions[0]
    except Exception:
        pass

    try:
        raw_version = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
        major = raw_version // 1000
        minor = (raw_version % 1000) // 10
        return f"{major}.{minor}"
    except Exception:
        return None


def get_cuda_device_info(device: Any = None) -> Dict[str, Any]:
    """Return stable CUDA device information for task results."""

    device_info: Dict[str, Any] = {
        "gpu_name": None,
        "compute_capability": get_requested_compute_capability(),
        "cuda_version": torch.version.cuda,
        "driver_version": get_cuda_driver_version(),
    }

    try:
        if torch.cuda.is_available():
            cuda_device = device
            if isinstance(device, str):
                parsed = torch.device(device)
                cuda_device = parsed if parsed.type == "cuda" else None
            if isinstance(device, torch.device) and device.type != "cuda":
                cuda_device = None
            if cuda_device is None:
                cuda_device = torch.cuda.current_device()

            device_info["gpu_name"] = torch.cuda.get_device_name(cuda_device)
            if device_info["compute_capability"] is None:
                major, minor = torch.cuda.get_device_capability(cuda_device)
                device_info["compute_capability"] = f"{major}.{minor}"
    except Exception as exc:
        device_info["error"] = str(exc)

    return device_info
