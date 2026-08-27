"""Device metadata detection and serialization helpers."""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Any


DEVICE_INFO_ENV = "KERNELGYM_DEVICE_INFO"
_CUDA_ARCH_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def _unknown_device_info() -> dict[str, Any]:
    return {
        "gpu_name": "unknown",
        "cuda_arch": "unknown",
        "compute_capability": "unknown",
        "sm_count": None,
        "warp_size": None,
        "thread_limits": {
            "max_threads_per_block": None,
            "max_threads_per_sm": None,
            "max_warps_per_sm": None,
            "max_blocks_per_sm": None,
            "max_block_dimensions": None,
            "max_grid_dimensions": None,
        },
        "shared_memory": {
            "per_block_default": None,
            "per_block_optin": None,
            "per_sm": None,
        },
        "register_limits": {
            "per_sm": None,
            "per_block": None,
        },
        "l2_cache": None,
        "device_memory": None,
        "theoretical_memory_bandwidth": None,
        "software": {
            "cuda_version": "unknown",
            "driver_version": "unknown",
            "nvcc_version": "unknown",
        },
    }


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


def _detect_with_nvidia_smi() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}

    info: dict[str, Any] = {}
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

    return info


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _theoretical_memory_bandwidth_gbps(memory_clock_rate_khz: Any, memory_bus_width_bits: Any) -> float | None:
    """Calculate peak DDR bandwidth from CUDA device properties."""
    clock_rate = _positive_int(memory_clock_rate_khz)
    bus_width = _positive_int(memory_bus_width_bits)
    if clock_rate is None or bus_width is None:
        return None
    # CUDA reports the memory clock in kHz. The factor of two accounts for DDR.
    return round(clock_rate * 1000 * bus_width / 8 * 2 / 1_000_000_000, 3)


def _detect_with_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}

    info: dict[str, Any] = {}
    try:
        if not torch.cuda.is_available():
            return info
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        properties = torch.cuda.get_device_properties(0)
        property_mapping = {
            "total_memory_bytes": "total_memory",
            "sm_count": "multi_processor_count",
            "warp_size": "warp_size",
            "max_threads_per_block": "max_threads_per_block",
            "max_threads_per_sm": "max_threads_per_multi_processor",
            "shared_memory_per_block_bytes": "shared_memory_per_block",
            "shared_memory_per_block_optin_bytes": "shared_memory_per_block_optin",
            "shared_memory_per_sm_bytes": "shared_memory_per_multiprocessor",
            "registers_per_sm": "regs_per_multiprocessor",
            "l2_cache_bytes": "L2_cache_size",
            "memory_bus_width_bits": "memory_bus_width",
            "memory_clock_rate_khz": "memory_clock_rate",
        }
        for output_key, property_name in property_mapping.items():
            value = _positive_int(getattr(properties, property_name, None))
            if value is not None:
                info[output_key] = value
        bandwidth = _theoretical_memory_bandwidth_gbps(
            info.get("memory_clock_rate_khz"),
            info.get("memory_bus_width_bits"),
        )
        if bandwidth is not None:
            info["theoretical_memory_bandwidth_gbps"] = bandwidth
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


def _load_cuda_runtime() -> Any | None:
    for library_name in ("libcudart.so", "libcudart.so.12"):
        try:
            return ctypes.CDLL(library_name)
        except OSError:
            continue
    return None


def _detect_cuda_runtime_version() -> str:
    """Query the loaded CUDA Runtime API version without relying on framework build metadata."""
    runtime = _load_cuda_runtime()
    if runtime is None:
        return ""
    try:
        query_version = runtime.cudaRuntimeGetVersion
        query_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query_version.restype = ctypes.c_int
        encoded_version = ctypes.c_int()
        if query_version(ctypes.byref(encoded_version)) != 0:
            return ""
    except AttributeError:
        return ""

    version = _positive_int(encoded_version.value)
    if version is not None:
        return f"{version // 1000}.{version % 1000 // 10}"
    return ""


def _detect_cuda_device_attributes(device_index: int = 0) -> dict[str, Any]:
    """Query CUDA limits not exposed by ``torch.cuda.DeviceProperties``."""
    runtime = _load_cuda_runtime()
    if runtime is None:
        return {}
    try:
        query_attribute = runtime.cudaDeviceGetAttribute
        query_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        query_attribute.restype = ctypes.c_int
    except AttributeError:
        return {}

    def query(attribute: int) -> int | None:
        value = ctypes.c_int()
        if query_attribute(ctypes.byref(value), attribute, device_index) != 0:
            return None
        return _positive_int(value.value)

    attributes = {
        "max_block_dimensions": [query(attribute) for attribute in (2, 3, 4)],
        "max_grid_dimensions": [query(attribute) for attribute in (5, 6, 7)],
        "registers_per_block": query(12),
        "max_blocks_per_sm": query(106),
    }
    return {
        key: value
        for key, value in attributes.items()
        if value is not None and (not isinstance(value, list) or all(item is not None for item in value))
    }


def _nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nested_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _positive_int_triplet(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    parsed = [_positive_int(item) for item in value]
    if any(item is None for item in parsed):
        return None
    return [int(item) for item in parsed]


def _format_binary_size(value: Any, divisor: int, unit: str, decimals: int = 0) -> str | None:
    size_bytes = _positive_int(value)
    if size_bytes is None:
        return None
    return f"{size_bytes / divisor:.{decimals}f} {unit}"


def _formatted_or_binary_size(
    direct_value: Any,
    raw_bytes: Any,
    divisor: int,
    unit: str,
    decimals: int = 0,
) -> str | None:
    return _nonempty_string(direct_value) or _format_binary_size(raw_bytes, divisor, unit, decimals)


def _format_theoretical_memory_bandwidth(raw: dict[str, Any]) -> str | None:
    direct_value = _nonempty_string(raw.get("theoretical_memory_bandwidth"))
    if direct_value:
        return direct_value
    try:
        bandwidth_gbps = float(raw.get("theoretical_memory_bandwidth_gbps"))
    except (TypeError, ValueError, OverflowError):
        return None
    if bandwidth_gbps <= 0:
        return None
    if bandwidth_gbps >= 1000:
        return f"{bandwidth_gbps / 1000:.3f} TB/s"
    return f"{bandwidth_gbps:.2f} GB/s"


def _cuda_arch(compute_capability: str, configured_arch: Any) -> str:
    direct_value = _nonempty_string(configured_arch)
    if direct_value:
        return direct_value
    first_arch = compute_capability.split(";", maxsplit=1)[0]
    if not _CUDA_ARCH_PATTERN.fullmatch(first_arch):
        return "unknown"
    major, _, minor = first_arch.partition(".")
    return f"sm_{major}{minor or '0'}"


def _normalize_device_info(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = _unknown_device_info()
    compute_capability = _nonempty_string(raw.get("compute_capability")) or "unknown"
    normalized["gpu_name"] = _nonempty_string(raw.get("gpu_name")) or "unknown"
    normalized["compute_capability"] = compute_capability
    normalized["cuda_arch"] = _cuda_arch(compute_capability, raw.get("cuda_arch"))

    for key in ("sm_count", "warp_size"):
        value = _positive_int(raw.get(key))
        if value is not None:
            normalized[key] = value

    raw_thread_limits = _nested_mapping(raw, "thread_limits")
    max_threads_per_block = _positive_int(
        raw_thread_limits.get("max_threads_per_block", raw.get("max_threads_per_block"))
    )
    max_threads_per_sm = _positive_int(raw_thread_limits.get("max_threads_per_sm", raw.get("max_threads_per_sm")))
    max_warps_per_sm = _positive_int(raw_thread_limits.get("max_warps_per_sm", raw.get("max_warps_per_sm")))
    if max_warps_per_sm is None and max_threads_per_sm is not None and normalized["warp_size"] is not None:
        max_warps_per_sm = max_threads_per_sm // normalized["warp_size"]
    normalized["thread_limits"] = {
        "max_threads_per_block": max_threads_per_block,
        "max_threads_per_sm": max_threads_per_sm,
        "max_warps_per_sm": max_warps_per_sm,
        "max_blocks_per_sm": _positive_int(
            raw_thread_limits.get("max_blocks_per_sm", raw.get("max_blocks_per_sm"))
        ),
        "max_block_dimensions": _positive_int_triplet(
            raw_thread_limits.get("max_block_dimensions", raw.get("max_block_dimensions"))
        ),
        "max_grid_dimensions": _positive_int_triplet(
            raw_thread_limits.get("max_grid_dimensions", raw.get("max_grid_dimensions"))
        ),
    }

    raw_register_limits = _nested_mapping(raw, "register_limits")
    normalized["register_limits"] = {
        "per_sm": _positive_int(raw_register_limits.get("per_sm", raw.get("registers_per_sm"))),
        "per_block": _positive_int(raw_register_limits.get("per_block", raw.get("registers_per_block"))),
    }

    raw_shared_memory = _nested_mapping(raw, "shared_memory")
    normalized["shared_memory"] = {
        "per_block_default": _formatted_or_binary_size(
            raw_shared_memory.get("per_block_default"),
            raw.get("shared_memory_per_block_bytes"),
            1024,
            "KiB",
        ),
        "per_block_optin": _formatted_or_binary_size(
            raw_shared_memory.get("per_block_optin"),
            raw.get("shared_memory_per_block_optin_bytes"),
            1024,
            "KiB",
        ),
        "per_sm": _formatted_or_binary_size(
            raw_shared_memory.get("per_sm"),
            raw.get("shared_memory_per_sm_bytes"),
            1024,
            "KiB",
        ),
    }
    normalized["l2_cache"] = _formatted_or_binary_size(
        raw.get("l2_cache"),
        raw.get("l2_cache_bytes"),
        1024**2,
        "MiB",
    )
    normalized["device_memory"] = _formatted_or_binary_size(
        raw.get("device_memory"),
        raw.get("total_memory_bytes"),
        1024**3,
        "GiB",
        decimals=2,
    )
    normalized["theoretical_memory_bandwidth"] = _format_theoretical_memory_bandwidth(raw)

    raw_software = _nested_mapping(raw, "software")
    normalized["software"] = {
        "cuda_version": _nonempty_string(
            raw_software.get("cuda_version", raw_software.get("cuda", raw.get("cuda_version")))
        )
        or "unknown",
        "driver_version": _nonempty_string(
            raw_software.get("driver_version", raw_software.get("driver", raw.get("driver_version")))
        )
        or "unknown",
        "nvcc_version": _nonempty_string(
            raw_software.get("nvcc_version", raw_software.get("nvcc", raw.get("nvcc_version")))
        )
        or "unknown",
    }
    return normalized


def detect_device_info() -> dict[str, Any]:
    detected: dict[str, Any] = {}
    detected.update(_detect_with_nvidia_smi())
    detected.update(_detect_with_torch())
    detected.update(_detect_cuda_device_attributes())
    cuda_runtime_version = _detect_cuda_runtime_version()
    if cuda_runtime_version:
        detected["cuda_version"] = cuda_runtime_version
    nvcc_version = _detect_nvcc_version()
    if nvcc_version:
        detected["nvcc_version"] = nvcc_version
    return _normalize_device_info(detected)


def encode_device_info(info: dict[str, Any]) -> str:
    return json.dumps(_normalize_device_info(info), sort_keys=True, separators=(",", ":"))


def decode_device_info(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _normalize_device_info(parsed)


def _existing_device_info(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return _normalize_device_info(value)
    if isinstance(value, str):
        return decode_device_info(value)
    return None


@lru_cache(maxsize=1)
def current_device_info() -> dict[str, Any]:
    return decode_device_info(os.environ.get(DEVICE_INFO_ENV)) or detect_device_info()


def with_device_info(metadata: dict[str, Any] | None) -> dict[str, Any]:
    updated: dict[str, Any] = dict(metadata or {})
    updated["device_info"] = _existing_device_info(updated.get("device_info")) or dict(current_device_info())
    return updated
