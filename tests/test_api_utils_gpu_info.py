import subprocess
import sys
from types import SimpleNamespace


def load_utils(monkeypatch):
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    from kernelgym.server.api import utils

    return utils


def test_nvidia_smi_gpu_info_reports_whole_device_memory(monkeypatch) -> None:
    utils = load_utils(monkeypatch)
    monkeypatch.setattr(utils.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(utils.settings, "gpu_devices", [0, 2])

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "0, NVIDIA GeForce RTX 4090, 923, 24564, 87\n"
                "1, NVIDIA GeForce RTX 4090, 911, 24564, 0\n"
                "2, NVIDIA GeForce RTX 4090, 19343, 24564, 99\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    info = utils._get_gpu_info_from_nvidia_smi()

    assert info is not None
    assert set(info) == {"cuda:0", "cuda:2"}
    assert info["cuda:0"]["memory_used"] == "0.9GB"
    assert info["cuda:0"]["memory_used_percent"] == "3.8%"
    assert info["cuda:2"]["memory_used"] == "18.9GB"
    assert info["cuda:2"]["memory_used_percent"] == "78.7%"
    assert info["cuda:2"]["utilization_gpu_percent"] == "99.0%"
    assert info["cuda:2"]["source"] == "nvidia-smi"


def test_nvidia_smi_gpu_info_returns_none_when_unavailable(monkeypatch) -> None:
    utils = load_utils(monkeypatch)
    monkeypatch.setattr(utils.shutil, "which", lambda _name: None)

    assert utils._get_gpu_info_from_nvidia_smi() is None
