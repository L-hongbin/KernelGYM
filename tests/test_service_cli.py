from kernelgym.cli import service


def test_service_parser_exposes_expected_commands() -> None:
    parser = service.build_parser()
    help_text = parser.format_help()
    assert "auto-configure" not in help_text
    assert "start-local" in help_text
    assert "start-worker-node" in help_text
    assert "stop" in help_text
    start_args = parser.parse_args(
        ["start-local", "--profile", "v1", "--cpu-workers", "4", "--redis-remote-access", "--no-stop-first"]
    )
    worker_args = parser.parse_args(
        [
            "start-worker-node",
            "--profile",
            "v1",
            "--master-addr",
            "192.168.16.40",
            "--node-rank",
            "1",
            "--cpu-compile-workers",
            "6",
        ]
    )
    stop_args = parser.parse_args(["stop", "--profile", "v1"])
    assert start_args.profile == "v1"
    assert start_args.cpu_compile_workers == 4
    assert start_args.redis_remote_access is True
    assert worker_args.master_addr == "192.168.16.40"
    assert worker_args.node_rank == "1"
    assert worker_args.cpu_compile_workers == 6
    assert stop_args.profile == "v1"


def test_service_profile_values_load_python_profiles() -> None:
    values = service._profile_values("v1")
    assert values["KERNELGYM_DEPLOYMENT_PROFILE"] == "v1"
    assert values["NODE_ID"] == "v1"
    assert values["API_HOST"] == "0.0.0.0"
    assert values["API_PORT"] == "20111"
    assert values["REDIS_PORT"] == "20110"
    assert values["KERNELGYM_CORE_DUMP_DIR"] == "logs/core_dumps"
    assert values["KERNELGYM_CORE_DUMP_KEEP"] == "10"


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


def test_with_hostname_log_dirs_nests_under_host(monkeypatch) -> None:
    monkeypatch.setattr(service, "_hostname", lambda: "node7")

    values = service._with_hostname_log_dirs(
        {
            "LOG_DIR": "logs/v1",
            "PY_LOG_DIR": "py_logs/v1",
            "EVAL_RESULTS_PATH": "logs/v1/eval_results.jsonl",
            "KERNELGYM_CORE_DUMP_DIR": "logs/core_dumps",
        }
    )

    assert values["LOG_DIR"] == "logs/v1/node7"
    assert values["PY_LOG_DIR"] == "py_logs/v1/node7"
    assert values["EVAL_RESULTS_PATH"] == "logs/v1/node7/eval_results.jsonl"
    assert values["KERNELGYM_CORE_DUMP_DIR"] == "logs/core_dumps/node7"


def test_with_hostname_log_dirs_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(service, "_hostname", lambda: "node7")

    once = service._with_hostname_log_dirs(
        {
            "LOG_DIR": "logs/v1",
            "EVAL_RESULTS_PATH": "logs/v1/eval_results.jsonl",
            "KERNELGYM_CORE_DUMP_DIR": "logs/core_dumps",
        }
    )
    twice = service._with_hostname_log_dirs(once)

    assert twice == once


def test_with_hostname_log_dirs_skips_missing_keys(monkeypatch) -> None:
    monkeypatch.setattr(service, "_hostname", lambda: "node7")

    values = service._with_hostname_log_dirs({"NODE_ID": "v1"})

    assert "LOG_DIR" not in values
    assert "EVAL_RESULTS_PATH" not in values


def test_service_env_respects_configured_torch_cuda_arch_list(monkeypatch) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    monkeypatch.setattr(
        service,
        "_detect_visible_torch_cuda_arch_list",
        lambda: (_ for _ in ()).throw(AssertionError("should not auto-detect")),
    )
    monkeypatch.setattr(service, "detect_device_info", lambda: {"gpu_name": "Detected GPU"})

    env = service._service_env({"TORCH_CUDA_ARCH_LIST": "8.9"})

    assert env["TORCH_CUDA_ARCH_LIST"] == "8.9"


def test_service_env_detects_torch_cuda_arch_list(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(service, "_detect_visible_torch_cuda_arch_list", lambda: "8.9")
    monkeypatch.setattr(service, "detect_device_info", lambda: {"gpu_name": "Detected GPU"})

    env = service._service_env({})

    assert env["TORCH_CUDA_ARCH_LIST"] == "8.9"


def test_service_env_detects_device_info(monkeypatch) -> None:
    monkeypatch.delenv("KERNELGYM_CORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("KERNELGYM_CORE_DUMP_KEEP", raising=False)
    detected = {
        "gpu_name": "Detected GPU",
        "compute_capability": "8.0",
        "cuda_version": "12.9",
        "driver_version": "575.1",
        "nvcc_version": "12.9",
    }
    monkeypatch.setattr(service, "_detect_visible_torch_cuda_arch_list", lambda: "")
    monkeypatch.setattr(service, "detect_device_info", lambda: detected)

    env = service._service_env({})

    assert service.json.loads(env["KERNELGYM_DEVICE_INFO"]) == detected
    assert env["KERNELGYM_CORE_DUMP_KEEP"] == "10"
    assert env["KERNELGYM_CORE_DUMP_DIR"].endswith("logs/core_dumps/" + service._hostname())


def test_runtime_overrides_can_set_cpu_compile_workers() -> None:
    values = service._apply_runtime_overrides(
        {"CPU_COMPILE_WORKERS": "24"},
        type("Args", (), {"cpu_compile_workers": 3, "redis_remote_access": True})(),
    )

    assert values["CPU_COMPILE_WORKERS"] == "3"
    assert values["KERNELGYM_REDIS_REMOTE_ACCESS"] == "true"


def test_ensure_redis_configures_remote_access_for_existing_redis(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(service, "_port_is_open", lambda host, port: True)
    monkeypatch.setattr(service, "_configure_redis_remote_access", lambda values: calls.append(values))

    service._ensure_redis({"KERNELGYM_REDIS_REMOTE_ACCESS": "true"})

    assert calls == [{"KERNELGYM_REDIS_REMOTE_ACCESS": "true"}]


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
        {
            "server_env": None,
            "profile": "auto",
            "master_addr": "192.168.16.40",
            "node_rank": "1",
            "cpu_compile_workers": 5,
        },
    )()
    assert service.cmd_start_worker_node(args) == 0

    assert captured_envs[0]["API_HOST"] == "192.168.16.40"
    assert captured_envs[0]["REDIS_HOST"] == "192.168.16.40"
    assert captured_envs[0]["NODE_ID"] == "v1-worker-1"
    assert captured_envs[0]["CPU_COMPILE_WORKERS"] == "5"


def test_start_worker_node_writes_logs_under_hostname_subdir(monkeypatch) -> None:
    captured_logs = []
    monkeypatch.setattr(service, "_hostname", lambda: "node7")
    monkeypatch.setattr(service, "_check_worker_connectivity", lambda values: None)
    monkeypatch.setattr(
        service,
        "_http_post_json",
        lambda url: {"node_id": "v1-worker-1", "hostname": "v1-worker-1"},
    )
    monkeypatch.setattr(service, "_http_get_json", lambda url: {})

    def fake_launch(command, log_file, env):
        captured_logs.append((log_file, env))
        return 12345

    monkeypatch.setattr(service, "_launch_background", fake_launch)

    args = type(
        "Args",
        (),
        {
            "server_env": None,
            "profile": "auto",
            "master_addr": "192.168.16.40",
            "node_rank": "1",
            "cpu_compile_workers": 0,
        },
    )()
    assert service.cmd_start_worker_node(args) == 0

    log_file, env = captured_logs[0]
    assert log_file.parent.name == "node7"
    assert log_file.parent.parent.name == "v1-worker-1-worker"
    assert env["LOG_DIR"].endswith("/node7")
    assert env["KERNELGYM_CORE_DUMP_DIR"].endswith("logs/core_dumps/node7")


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


def test_clear_expected_workers_for_host_preserves_other_hosts() -> None:
    """A primary restart must only clear its own host's expected-worker
    registrations; wiping the shared set would strip worker nodes of
    supervision. Unowned (legacy) ids are cleared so no monitor double-claims."""

    class FakeClient:
        def __init__(self):
            self.sets = {"kernelgym:expected_workers": {"worker_gpu_0", "node-1_gpu_0", "legacy_cpu_0"}}
            self.hashes = {
                "kernelgym:expected_worker:worker_gpu_0": {"hostname": "host-a"},
                "kernelgym:expected_worker:node-1_gpu_0": {"hostname": "host-b"},
            }

        def smembers(self, key):
            return set(self.sets.get(key, set()))

        def hget(self, key, field):
            return self.hashes.get(key, {}).get(field)

        def srem(self, key, member):
            self.sets.get(key, set()).discard(member)

        def delete(self, *keys):
            for key in keys:
                self.hashes.pop(key, None)

    client = FakeClient()
    service._clear_expected_workers_for_host(client, "host-a")
    assert client.sets["kernelgym:expected_workers"] == {"node-1_gpu_0"}
    assert "kernelgym:expected_worker:node-1_gpu_0" in client.hashes
    assert "kernelgym:expected_worker:worker_gpu_0" not in client.hashes
