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
from kernelgym.server.api.models import MetricsResponse, SystemHealthResponse


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
