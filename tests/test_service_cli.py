import os
from pathlib import Path

from kernelgym.cli import service


def test_service_parser_exposes_expected_commands() -> None:
    parser = service.build_parser()
    help_text = parser.format_help()
    assert "auto-configure" in help_text
    assert "start-local" in help_text
    assert "start-worker-node" in help_text
    assert "stop" in help_text


def test_auto_configure_writes_reward_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / "kernelgym.env"
    monkeypatch.setenv("PORT0", "19080")
    monkeypatch.setenv("PORT1", "19081")
    monkeypatch.setenv("PORT2", "19082")
    monkeypatch.setenv("GPU_DEVICES", "[0]")

    rc = service.main(
        [
            "auto-configure",
            "--env-file",
            str(env_file),
            "--force",
            "--use-indexed-ports",
        ]
    )

    assert rc == 0
    values = service._read_env_file(env_file)
    assert values["API_PORT"] == "19081"
    assert values["REDIS_PORT"] == "19080"
    assert values["METRICS_PORT"] == "19082"
    assert values["DEFAULT_BACKEND_ADAPTER"] == "kernelbench"
    assert values["KERNELGYM_CUDA_AGENT_NVCC_THREADS"] == os.environ.get("KERNELGYM_CUDA_AGENT_NVCC_THREADS", "4")
