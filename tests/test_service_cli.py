from kernelgym.cli import service


def test_service_parser_exposes_expected_commands() -> None:
    parser = service.build_parser()
    help_text = parser.format_help()
    assert "auto-configure" not in help_text
    assert "start-local" in help_text
    assert "start-worker-node" in help_text
    assert "stop" in help_text
    start_args = parser.parse_args(["start-local", "--profile", "reward-40", "--no-stop-first"])
    stop_args = parser.parse_args(["stop", "--profile", "reward-39"])
    assert start_args.profile == "reward-40"
    assert stop_args.profile == "reward-39"


def test_service_profile_values_load_python_profiles() -> None:
    values = service._profile_values("reward-40")
    assert values["KERNELGYM_DEPLOYMENT_PROFILE"] == "reward-40"
    assert values["NODE_ID"] == "reward-40"
    assert values["API_HOST"] == "192.168.16.40"
    assert values["API_PORT"] == "20111"
    assert values["REDIS_PORT"] == "20110"


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
