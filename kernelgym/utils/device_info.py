"""Device metadata detection and serialization helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Any


DEVICE_INFO_ENV = "KERNELGYM_DEVICE_INFO"
DEVICE_INFO_KEYS = (
    "gpu_name",
    "compute_capability",
    "cuda_version",
    "driver_version",
    "nvcc_version",
)
_CUDA_ARCH_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def _unknown_device_info() -> dict[str, str]:
    return {key: "unknown" for key in DEVICE_INFO_KEYS}


def _run_text(command: list[str], timeout: float = 3.0) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        value = line.strip()
        if value:
            return value
    return ""


def _format_cuda_arch_list(values: list[str]) -> str:
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


def _detect_with_nvidia_smi() -> dict[str, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}

    info: dict[str, str] = {}
    gpu_name = _first_nonempty_line(_run_text([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"]))
    if gpu_name:
        info["gpu_name"] = gpu_name

    for query_field in ("compute_cap", "compute_capability"):
        arch_list = _format_cuda_arch_list(
            _run_text([nvidia_smi, f"--query-gpu={query_field}", "--format=csv,noheader"]).splitlines()
        )
        if arch_list:
            info["compute_capability"] = arch_list
            break

    driver_version = _first_nonempty_line(
        _run_text([nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"])
    )
    if driver_version:
        info["driver_version"] = driver_version

    smi_output = _run_text([nvidia_smi])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", smi_output)
    if cuda_match:
        info["cuda_version"] = cuda_match.group(1)
    return info


def _detect_with_torch() -> dict[str, str]:
    try:
        import torch
    except Exception:
        return {}

    info: dict[str, str] = {}
    try:
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_version:
            info["cuda_version"] = str(cuda_version)
        if not torch.cuda.is_available():
            return info
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        arches = []
        for device_index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_index)
            arches.append(f"{major}.{minor}")
        arch_list = _format_cuda_arch_list(arches)
        if arch_list:
            info["compute_capability"] = arch_list
    except Exception:
        return info
    return info


def _detect_nvcc_version() -> str:
    candidates = [
        shutil.which("nvcc"),
        "/usr/local/cuda-12.9/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ]
    for nvcc in candidates:
        if not nvcc:
            continue
        output = _run_text([nvcc, "--version"])
        match = re.search(r"release\s+([0-9.]+)", output)
        if match:
            return match.group(1)
    return ""


def _normalize_device_info(raw: dict[str, Any]) -> dict[str, str]:
    normalized = _unknown_device_info()
    for key in DEVICE_INFO_KEYS:
        value = raw.get(key)
        if value is not None and str(value).strip():
            normalized[key] = str(value).strip()
    return normalized


def detect_device_info() -> dict[str, str]:
    detected: dict[str, Any] = {}
    detected.update(_detect_with_nvidia_smi())
    detected.update(_detect_with_torch())
    nvcc_version = _detect_nvcc_version()
    if nvcc_version:
        detected["nvcc_version"] = nvcc_version
    return _normalize_device_info(detected)


def encode_device_info(info: dict[str, Any]) -> str:
    return json.dumps(_normalize_device_info(info), sort_keys=True, separators=(",", ":"))


def decode_device_info(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _normalize_device_info(parsed)


def _existing_device_info(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict):
        return _normalize_device_info(value)
    if isinstance(value, str):
        return decode_device_info(value)
    return None


@lru_cache(maxsize=1)
def current_device_info() -> dict[str, str]:
    return decode_device_info(os.environ.get(DEVICE_INFO_ENV)) or detect_device_info()


def with_device_info(metadata: dict[str, Any] | None) -> dict[str, Any]:
    updated: dict[str, Any] = dict(metadata or {})
    updated["device_info"] = _existing_device_info(updated.get("device_info")) or dict(current_device_info())
    return updated
