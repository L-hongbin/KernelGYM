import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_worker_monitor():
    spec = importlib.util.spec_from_file_location(
        "worker_monitor_script", ROOT / "kernelgym" / "worker" / "worker_monitor.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRedis:
    async def hgetall(self, key):
        return {}

    async def hset(self, key, mapping):
        return 1

    async def delete(self, key):
        return 1


class FakeProcess:
    pid = 12345

    def poll(self):
        return None


def test_worker_monitor_restarts_cpu_worker_with_cpu_entrypoint(monkeypatch, tmp_path) -> None:
    worker_monitor = load_worker_monitor()
    commands = []
    monitor = worker_monitor.WorkerMonitor(FakeRedis(), persistent=True)
    monkeypatch.chdir(tmp_path)

    async def noop(*args):
        return None

    monkeypatch.setattr(monitor, "_kill_worker_process", noop)
    monkeypatch.setattr(monitor, "_reset_gpu_device", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    def fake_popen(command, **kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert asyncio.run(monitor._restart_worker("worker_cpu_0", "cpu"))
    assert commands == [
        [
            sys.executable,
            "-m",
            "kernelgym.worker.cpu_worker",
            "--worker-id",
            "worker_cpu_0",
        ]
    ]


def test_load_local_expected_ids_filters_by_hostname() -> None:
    """A monitor must only enforce expected workers registered for its own host;
    ids with no recorded hostname stay local for backward compatibility."""
    worker_monitor = load_worker_monitor()

    class HostFakeRedis(FakeRedis):
        async def smembers(self, key):
            return {b"local_gpu_0", b"remote_gpu_0", b"legacy_cpu_0"}

        async def hgetall(self, key):
            data = {
                "kernelgym:expected_worker:local_gpu_0": {b"hostname": b"host-a", b"device": b"cuda:0"},
                "kernelgym:expected_worker:remote_gpu_0": {b"hostname": b"host-b", b"device": b"cuda:0"},
            }
            return data.get(key, {})

    monitor = worker_monitor.WorkerMonitor(HostFakeRedis(), persistent=True)
    monitor.hostname = "host-a"
    local_ids = asyncio.run(monitor._load_local_expected_ids())
    assert local_ids == {"local_gpu_0", "legacy_cpu_0"}
