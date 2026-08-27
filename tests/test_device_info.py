import sys
from types import SimpleNamespace

import pytest

from kernelgym.utils import device_info


def test_torch_detection_collects_static_device_capabilities(monkeypatch) -> None:
    properties = SimpleNamespace(
        total_memory=85_019_328_512,
        multi_processor_count=114,
        warp_size=32,
        max_threads_per_block=1024,
        max_threads_per_multi_processor=2048,
        shared_memory_per_block=49_152,
        shared_memory_per_block_optin=232_448,
        shared_memory_per_multiprocessor=233_472,
        regs_per_multiprocessor=65_536,
        L2_cache_size=52_428_800,
        memory_bus_width=5120,
        memory_clock_rate=1_593_000,
        clock_rate=1_755_000,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "NVIDIA H100 PCIe",
            get_device_properties=lambda _index: properties,
            device_count=lambda: 1,
            get_device_capability=lambda _index: (9, 0),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    detected = device_info._normalize_device_info(device_info._detect_with_torch())

    assert detected == {
        "gpu_name": "NVIDIA H100 PCIe",
        "cuda_arch": "sm_90",
        "compute_capability": "9.0",
        "sm_count": 114,
        "warp_size": 32,
        "thread_limits": {
            "max_threads_per_block": 1024,
            "max_threads_per_sm": 2048,
            "max_warps_per_sm": 64,
            "max_blocks_per_sm": None,
            "max_block_dimensions": None,
            "max_grid_dimensions": None,
        },
        "shared_memory": {
            "per_block_default": "48 KiB",
            "per_block_optin": "227 KiB",
            "per_sm": "228 KiB",
        },
        "register_limits": {
            "per_sm": 65_536,
            "per_block": None,
        },
        "l2_cache": "50 MiB",
        "device_memory": "79.18 GiB",
        "theoretical_memory_bandwidth": "2.039 TB/s",
        "software": {
            "cuda_version": "unknown",
            "driver_version": "unknown",
            "nvcc_version": "unknown",
        },
    }


def test_device_info_codec_preserves_model_friendly_structure_and_fills_unknowns() -> None:
    decoded = device_info.decode_device_info(
        device_info.encode_device_info(
            {
                "gpu_name": "GPU",
                "sm_count": "114",
                "theoretical_memory_bandwidth_gbps": "2039.04",
            }
        )
    )

    assert decoded is not None
    assert decoded["gpu_name"] == "GPU"
    assert decoded["cuda_arch"] == "unknown"
    assert decoded["compute_capability"] == "unknown"
    assert decoded["sm_count"] == 114
    assert decoded["warp_size"] is None
    assert decoded["theoretical_memory_bandwidth"] == "2.039 TB/s"
    assert "theoretical_memory_bandwidth_gbps" not in decoded


def test_memory_bandwidth_requires_positive_clock_and_bus_width() -> None:
    assert device_info._theoretical_memory_bandwidth_gbps(1_593_000, 5120) == pytest.approx(2039.04)
    assert device_info._theoretical_memory_bandwidth_gbps(0, 5120) is None
    assert device_info._theoretical_memory_bandwidth_gbps(1_593_000, None) is None


def test_cuda_version_is_queried_from_the_runtime_api(monkeypatch) -> None:
    class FakeRuntimeVersionQuery:
        argtypes = None
        restype = None

        def __call__(self, version_pointer) -> int:
            version_pointer._obj.value = 12_090
            return 0

    fake_runtime = SimpleNamespace(cudaRuntimeGetVersion=FakeRuntimeVersionQuery())
    monkeypatch.setattr(device_info.ctypes, "CDLL", lambda _library_name: fake_runtime)

    assert device_info._detect_cuda_runtime_version() == "12.9"


def test_cuda_device_limits_are_queried_from_runtime_attributes(monkeypatch) -> None:
    values = {
        2: 1024,
        3: 1024,
        4: 64,
        5: 2_147_483_647,
        6: 65_535,
        7: 65_535,
        12: 65_536,
        106: 32,
    }

    class FakeDeviceAttributeQuery:
        argtypes = None
        restype = None

        def __call__(self, value_pointer, attribute, device_index) -> int:
            assert device_index == 0
            value_pointer._obj.value = values[attribute]
            return 0

    fake_runtime = SimpleNamespace(cudaDeviceGetAttribute=FakeDeviceAttributeQuery())
    monkeypatch.setattr(device_info, "_load_cuda_runtime", lambda: fake_runtime)

    assert device_info._detect_cuda_device_attributes() == {
        "max_block_dimensions": [1024, 1024, 64],
        "max_grid_dimensions": [2_147_483_647, 65_535, 65_535],
        "registers_per_block": 65_536,
        "max_blocks_per_sm": 32,
    }
