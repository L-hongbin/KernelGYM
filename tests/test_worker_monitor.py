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
