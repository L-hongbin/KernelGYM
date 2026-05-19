from kernelgym.cli import service


def test_service_parser_exposes_expected_commands() -> None:
    parser = service.build_parser()
    help_text = parser.format_help()
    assert "auto-configure" not in help_text
    assert "start-local" in help_text
    assert "start-worker-node" in help_text
    assert "stop" in help_text
    start_args = parser.parse_args(["start-local", "--profile", "v1", "--no-stop-first"])
    worker_args = parser.parse_args(
        ["start-worker-node", "--profile", "v1", "--master-addr", "192.168.16.40", "--node-rank", "1"]
    )
    stop_args = parser.parse_args(["stop", "--profile", "v1"])
    assert start_args.profile == "v1"
    assert worker_args.master_addr == "192.168.16.40"
    assert worker_args.node_rank == "1"
    assert stop_args.profile == "v1"


def test_service_profile_values_load_python_profiles() -> None:
    values = service._profile_values("v1")
    assert values["KERNELGYM_DEPLOYMENT_PROFILE"] == "v1"
    assert values["NODE_ID"] == "v1"
    assert values["API_HOST"] == "0.0.0.0"
    assert values["API_PORT"] == "20111"
    assert values["REDIS_PORT"] == "20110"


def test_service_auto_profile_uses_default_functional_profile() -> None:
    values = service._profile_values("auto")

    assert values["API_HOST"] == "0.0.0.0"
    assert values["NODE_ID"] == "v1"


def test_worker_profile_values_reuses_deployment_profile() -> None:
    values = service._worker_profile_values("auto", "192.168.16.40", "1")

    assert values["API_HOST"] == "192.168.16.40"
    assert values["REDIS_HOST"] == "192.168.16.40"
    assert values["GPU_DEVICES"] == "[0,1,2,3,4,5,6,7]"
    assert values["NODE_ID"] == "v1-worker-1"
    assert values["WORKER_NAME_PREFIX"] == "v1-worker-1"
    assert values["LOG_DIR"] == "logs/v1-worker-1-worker"
    assert values["KERNELGYM_NODE_RANK"] == "1"


def test_service_env_respects_configured_torch_cuda_arch_list(monkeypatch) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    monkeypatch.setattr(
        service,
        "_detect_visible_torch_cuda_arch_list",
        lambda: (_ for _ in ()).throw(AssertionError("should not auto-detect")),
    )

    env = service._service_env({"TORCH_CUDA_ARCH_LIST": "8.9"})

    assert env["TORCH_CUDA_ARCH_LIST"] == "8.9"


def test_service_env_detects_torch_cuda_arch_list(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(service, "_detect_visible_torch_cuda_arch_list", lambda: "8.9")

    env = service._service_env({})

    assert env["TORCH_CUDA_ARCH_LIST"] == "8.9"


def test_write_env_file_groups_torch_cuda_arch_list(tmp_path) -> None:
    env_file = tmp_path / ".env"

    service._write_env_file(
        env_file,
        {
            "API_HOST": "127.0.0.1",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "KERNELGYM_NVCC_THREADS": "4",
        },
    )

    text = env_file.read_text(encoding="utf-8")
    cuda_build_index = text.index("# CUDA build")
    assert text.index("TORCH_CUDA_ARCH_LIST=8.9") > cuda_build_index
    assert text.index("KERNELGYM_NVCC_THREADS=4") > cuda_build_index


def test_start_worker_node_uses_explicit_server_env(tmp_path, monkeypatch) -> None:
    server_env = tmp_path / "server.env"
    server_env.write_text(
        "\n".join(
            [
                "API_HOST=192.168.16.40",
                "REDIS_HOST=192.168.16.40",
                "GPU_DEVICES=[0]",
                "NODE_ID=worker-node",
                "CPU_COMPILE_WORKERS=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "_check_worker_connectivity", lambda values: None)
    monkeypatch.setattr(service, "_http_post_json", lambda url: {"node_id": "worker-node", "hostname": "worker-node"})
    monkeypatch.setattr(service, "_http_get_json", lambda url: {})
    monkeypatch.setattr(service, "_launch_background", lambda command, log_file, env: 12345)

    assert service.cmd_start_worker_node(type("Args", (), {"server_env": str(server_env)})()) == 0

    text = server_env.read_text(encoding="utf-8")
    assert "API_HOST=192.168.16.40" in text
    assert "REDIS_HOST=192.168.16.40" in text
    assert "NODE_ID=worker-node" in text


def test_start_worker_node_generates_values_from_profile(monkeypatch) -> None:
    captured_envs = []
    monkeypatch.setattr(service, "_check_worker_connectivity", lambda values: None)
    monkeypatch.setattr(
        service,
        "_http_post_json",
        lambda url: {"node_id": "v1-worker-1", "hostname": "v1-worker-1"},
    )
    monkeypatch.setattr(service, "_http_get_json", lambda url: {})

    def fake_launch(command, log_file, env):
        captured_envs.append(env)
        return 12345

    monkeypatch.setattr(service, "_launch_background", fake_launch)

    args = type(
        "Args",
        (),
        {"server_env": None, "profile": "auto", "master_addr": "192.168.16.40", "node_rank": "1"},
    )()
    assert service.cmd_start_worker_node(args) == 0

    assert captured_envs[0]["API_HOST"] == "192.168.16.40"
    assert captured_envs[0]["REDIS_HOST"] == "192.168.16.40"
    assert captured_envs[0]["NODE_ID"] == "v1-worker-1"


def test_format_torch_cuda_arch_list_deduplicates_and_filters() -> None:
    assert service._format_torch_cuda_arch_list([" 8.9 ", "8.9", "9.0,invalid", "10.0;9.0"]) == "8.9;9.0;10.0"


def test_settings_hardcode_api_and_redis_runtime_knobs(monkeypatch) -> None:
    monkeypatch.setenv("API_PORT", "19081")
    monkeypatch.setenv("API_WORKERS", "99")
    monkeypatch.setenv("API_RELOAD", "true")
    monkeypatch.setenv("REDIS_PORT", "19080")
    monkeypatch.setenv("REDIS_DB", "9")
    monkeypatch.setenv("REDIS_PASSWORD", "secret")
    monkeypatch.setenv("REDIS_KEY_PREFIX", "custom")
    monkeypatch.setenv("REDIS_KEY_PREFIX_LEGACY", "legacy")
    monkeypatch.setenv("METRICS_PORT", "19082")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://bad:1/9")

    from kernelgym.config.settings import Settings

    settings = Settings()
    assert settings.api_port == 20111
    assert settings.api_workers == 4
    assert settings.api_reload is False
    assert settings.redis_port == 20110
    assert settings.redis_db == 0
    assert settings.redis_password == ""
    assert settings.redis_key_prefix == "kernelgym"
    assert settings.redis_key_prefix_legacy == "kernelserver"
    assert settings.metrics_port == 20112
    assert settings.celery_broker_url == "redis://localhost:20110/0"
