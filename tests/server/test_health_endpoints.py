"""Server tests for /health, /health/live and /metrics payloads.

The bug Codex flagged was the unhealthy/error fallback dicts missing required
fields, which makes FastAPI's response_model validation raise -> the client gets
a 500 instead of the intended "unhealthy" body. FastAPI validates the returned
dict with the response model, so constructing the model from the payload here
reproduces exactly that validation step (httpx/TestClient is not available in
this env, so we assert at the model layer rather than over HTTP).
"""

import asyncio
import time

from kernelgym.server.api import system_stats, utils
from kernelgym.server.api.models import DeviceInfoResponse, MetricsResponse, SystemHealthResponse


def _run(coro):
    return asyncio.run(coro)


def _raising_snapshot():
    async def _boom():
        raise RuntimeError("snapshot boom")

    return _boom


def test_health_ok_is_model_valid():
    payload = _run(utils.get_system_health())
    assert payload["status"] == "healthy"
    model = SystemHealthResponse(**payload)  # must not raise (== route returns 200)
    assert model.status == "healthy"
    assert model.snapshot_age_s is None or model.snapshot_age_s >= 0


def test_health_unhealthy_fallback_is_model_valid(monkeypatch):
    monkeypatch.setattr(system_stats, "get_snapshot", _raising_snapshot())
    payload = _run(utils.get_system_health())
    assert payload["status"] == "unhealthy"
    # Previously broken: missing required fields -> 500. Now must validate cleanly.
    SystemHealthResponse(**payload)


def test_metrics_unhealthy_fallback_is_model_valid(monkeypatch):
    monkeypatch.setattr(system_stats, "get_snapshot", _raising_snapshot())
    payload = _run(utils.get_system_metrics())
    MetricsResponse(**payload)
    for key in ("performance_metrics", "resource_metrics", "queue_metrics", "error_metrics"):
        assert key in payload


def test_health_and_liveness_routes_registered():
    from kernelgym.server.api import server

    paths = {r.path for r in server.app.routes if hasattr(r, "path")}
    assert "/health" in paths
    assert "/health/live" in paths
    assert "/device-info" in paths


def test_device_info_endpoint_returns_the_local_detected_schema(monkeypatch):
    from kernelgym.server.api import server

    detected = {
        "gpu_name": "Local GPU",
        "cuda_arch": "sm_90",
        "compute_capability": "9.0",
        "sm_count": 114,
        "warp_size": 32,
        "thread_limits": {
            "max_threads_per_block": 1024,
            "max_threads_per_sm": 2048,
            "max_warps_per_sm": 64,
            "max_blocks_per_sm": 32,
            "max_block_dimensions": [1024, 1024, 64],
            "max_grid_dimensions": [2_147_483_647, 65_535, 65_535],
        },
        "shared_memory": {
            "per_block_default": "48 KiB",
            "per_block_optin": "227 KiB",
            "per_sm": "228 KiB",
        },
        "register_limits": {"per_sm": 65_536, "per_block": 65_536},
        "l2_cache": "50 MiB",
        "device_memory": "79.18 GiB",
        "theoretical_memory_bandwidth": "2.039 TB/s",
        "software": {"cuda_version": "12.9", "driver_version": "590.1", "nvcc_version": "12.9"},
    }
    monkeypatch.setattr(server, "current_device_info", lambda: detected)

    response = _run(server.get_device_info())

    assert isinstance(response, DeviceInfoResponse)
    assert response.gpu_name == "Local GPU"
    assert response.sm_count == 114
    assert response.thread_limits.max_warps_per_sm == 64
    assert response.register_limits.per_block == 65_536
    assert response.shared_memory.per_block_optin == "227 KiB"
    assert "peak_compute_tflops" not in response.model_dump()


def test_get_snapshot_is_single_flight(monkeypatch):
    """Concurrent cold-start get_snapshot() callers must trigger exactly one compute.

    Regression guard for the single-flight asyncio.Lock in system_stats.get_snapshot:
    without it, every cold caller would spawn its own _compute() (one nvidia-smi each).
    Everything runs in one event loop with a fresh cache so the lock binds to a single
    loop (avoids the cross-asyncio.run() lock-binding trap).
    """

    async def scenario():
        cache = system_stats.SystemStatsCache()
        calls = 0

        def counting_compute():
            nonlocal calls
            calls += 1
            time.sleep(0.05)  # widen the race window so all callers reach the lock first
            return {
                "timestamp": "t",
                "monotonic": 0.0,
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_available_gb": 0.0,
                "memory_total_gb": 0.0,
                "gpu_status": {},
            }

        cache._compute = counting_compute
        monkeypatch.setattr(system_stats, "_cache", cache)

        results = await asyncio.gather(*[system_stats.get_snapshot() for _ in range(20)])

        assert calls == 1  # single-flight: only one refresh ran for 20 concurrent callers
        assert all(r is results[0] for r in results)  # all callers share the one snapshot

    asyncio.run(scenario())
