from pathlib import Path

from kernelgym.cli import deploy


def test_deploy_parser_exposes_expected_commands() -> None:
    parser = deploy.build_parser()
    help_text = parser.format_help()
    assert "detect-profile" in help_text
    assert "write-env" in help_text
    assert "create-venv" in help_text
    assert "lock-gpu-clocks" in help_text
    assert "host-container" in help_text


def test_detect_network_profile_uses_ms_path_and_treats_symlink_as_external(tmp_path: Path) -> None:
    missing = tmp_path / "missing-ms"
    real_ms = tmp_path / "real-ms"
    linked_ms = tmp_path / "linked-ms"
    target = tmp_path / "target"

    real_ms.mkdir()
    target.mkdir()
    linked_ms.symlink_to(target, target_is_directory=True)

    assert deploy._detect_network_profile(missing) == deploy.PROFILE_EXTERNAL
    assert deploy._detect_network_profile(real_ms) == deploy.PROFILE_INTERNAL
    assert deploy._detect_network_profile(linked_ms) == deploy.PROFILE_EXTERNAL


def test_write_env_uses_external_defaults_when_ms_is_symlink(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target"
    marker = tmp_path / "ms"
    env_file = tmp_path / "kernelgym.env"
    target.mkdir()
    marker.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(deploy, "_host_ip", lambda: "192.168.16.39")

    rc = deploy.main(
        [
            "write-env",
            "--env-file",
            str(env_file),
            "--marker-path",
            str(marker),
            "--role",
            "api",
            "--force",
        ]
    )

    assert rc == 0
    values = deploy.service._read_env_file(env_file)
    assert values["KERNELGYM_DEPLOYMENT_PROFILE"] == deploy.PROFILE_EXTERNAL
    assert values["KERNELGYM_SSH_RUNTIME"] == "physical_host"
    assert values["KERNELGYM_CONTAINER_REQUIRED"] == "true"
    assert values["KERNELGYM_LOCK_GPU_CLOCKS"] == "true"
    assert values["API_HOST"] == "192.168.16.39"
    assert values["API_PORT"] == "8111"
    assert values["REDIS_PORT"] == "8110"
    assert values["REDIS_KEY_PREFIX"] == "kernelgym_external"


def test_write_env_uses_internal_defaults_when_ms_is_real_directory(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "ms"
    env_file = tmp_path / "kernelgym.env"
    marker.mkdir()
    monkeypatch.setattr(deploy, "_host_ip", lambda: "10.0.0.8")

    rc = deploy.main(
        [
            "write-env",
            "--env-file",
            str(env_file),
            "--marker-path",
            str(marker),
            "--role",
            "worker",
            "--server-host",
            "10.0.0.1",
            "--force",
        ]
    )

    assert rc == 0
    values = deploy.service._read_env_file(env_file)
    assert values["KERNELGYM_DEPLOYMENT_PROFILE"] == deploy.PROFILE_INTERNAL
    assert values["KERNELGYM_SSH_RUNTIME"] == "container"
    assert values["KERNELGYM_CONTAINER_REQUIRED"] == "false"
    assert values["KERNELGYM_LOCK_GPU_CLOCKS"] == "false"
    assert values["API_HOST"] == "10.0.0.1"
    assert values["REDIS_HOST"] == "10.0.0.1"
    assert values["API_PORT"] == "10907"
    assert values["REDIS_PORT"] == "10906"
    assert values["REDIS_KEY_PREFIX"] == "kernelgym_internal"


def test_host_container_command_mounts_cuda129_and_nfs(tmp_path: Path) -> None:
    parser = deploy.build_parser()
    args = parser.parse_args(
        [
            "host-container",
            "--name",
            "kernelgym-reward-test",
            "--image",
            "example/cuda129:latest",
            "--repo-dir",
            str(tmp_path),
            "--cuda-home",
            "/usr/local/cuda-12.9",
            "--env",
            "EXAMPLE=1",
        ]
    )

    command = deploy._docker_run_command(args)
    joined = " ".join(command)

    assert "--gpus all" in joined
    assert "--network host" in joined
    assert "--tmpfs /dev/shm:rw,nosuid,nodev,exec,size=256g" in joined
    assert "--shm-size 256g" not in joined
    assert "-v /nfs:/nfs" in joined
    assert "-v /usr/local/cuda-12.9:/usr/local/cuda-12.9:ro" in joined
    assert "-e CUDA_HOME=/usr/local/cuda-12.9" in joined
    assert "-e EXAMPLE=1" in joined
    assert "example/cuda129:latest sleep infinity" in joined


def test_create_venv_dry_run_uses_cuda129_packages(tmp_path: Path, capsys) -> None:
    rc = deploy.main(
        [
            "create-venv",
            "--venv",
            str(tmp_path / ".venv"),
            "--python",
            "python3.10",
            "--dry-run",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "uv venv --python python3.10" in output
    assert deploy.CUDA129_INDEX_URL in output
    assert "torch==2.11.0+cu129" in output
    assert "torchvision==0.26.0+cu129" in output


def test_cuda_env_adds_proxy_when_requested() -> None:
    env = deploy._cuda_env("/usr/local/cuda-12.9", proxy=deploy.DEFAULT_PROXY)

    assert env["CUDA_HOME"] == "/usr/local/cuda-12.9"
    assert env["PATH"].startswith("/usr/local/cuda-12.9/bin:")
    assert env["LD_LIBRARY_PATH"].startswith("/usr/local/cuda-12.9/lib64:")
    assert env["HTTP_PROXY"] == deploy.DEFAULT_PROXY
    assert env["HTTPS_PROXY"] == deploy.DEFAULT_PROXY
    assert env["ALL_PROXY"] == deploy.DEFAULT_PROXY


def test_lock_gpu_clocks_dry_run_uses_host_level_nvidia_smi(capsys) -> None:
    rc = deploy.main(["lock-gpu-clocks", "--sudo", "--gpu-clock", "2700", "--power-limit", "400", "--dry-run"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "sudo nvidia-smi -pm 1" in output
    assert "sudo nvidia-smi -lgc 2700,2700" in output
    assert "sudo nvidia-smi -pl 400" in output
