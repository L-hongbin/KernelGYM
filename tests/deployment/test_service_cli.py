"""Service lifecycle CLI tests."""

import signal

import pytest

from kernelgym.cli import service


def _isolate_worker_start(monkeypatch) -> None:
    client = object()
    monkeypatch.setattr(
        service,
        "_resolve_gpu_devices",
        lambda values: {**values, "GPU_DEVICES": "[0]"},
    )
    monkeypatch.setattr(service, "_with_torch_cuda_arch_list", lambda values: values)
    monkeypatch.setattr(service, "detect_device_info", lambda: {})
    monkeypatch.setattr(service, "_redis_client", lambda values: client)
    monkeypatch.setattr(service, "_kill_processes", lambda *args, **kwargs: True)
    monkeypatch.setattr(service, "_collect_pids", lambda pattern: [])
    monkeypatch.setattr(service, "_clear_expected_workers_for_host", lambda *args, **kwargs: True)
    monkeypatch.setattr(service, "_assert_worker_process_slot_empty", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_register_expected_worker", lambda *args, **kwargs: None)


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
            "--gpu-devices",
            "0,1,2,3",
        ]
    )
    stop_args = parser.parse_args(["stop", "--profile", "v1"])
    assert start_args.profile == "v1"
    assert start_args.cpu_compile_workers == 4
    assert start_args.redis_remote_access is True
    assert worker_args.master_addr == "192.168.16.40"
    assert worker_args.node_rank == "1"
    assert worker_args.cpu_compile_workers == 6
    assert worker_args.gpu_devices == "0,1,2,3"
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


def test_hostname_prefers_worker_registration_source(monkeypatch) -> None:
    monkeypatch.setenv("HOSTNAME", "configured-alias")
    monkeypatch.setattr(service.socket, "gethostname", lambda: "runtime-host")

    assert service._hostname() == "runtime-host"


def test_service_auto_profile_uses_default_functional_profile() -> None:
    values = service._profile_values("auto")

    assert values["API_HOST"] == "0.0.0.0"
    assert values["NODE_ID"] == "v1"


def test_worker_profile_values_reuses_deployment_profile() -> None:
    values = service._worker_profile_values("auto", "192.168.16.40", "1")

    assert values["API_HOST"] == "192.168.16.40"
    assert values["REDIS_HOST"] == "192.168.16.40"
    assert values["GPU_DEVICES"] == "auto"
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

    assert service.json.loads(env["KERNELGYM_DEVICE_INFO"]) == service.json.loads(
        service.encode_device_info(detected)
    )
    assert env["KERNELGYM_CORE_DUMP_KEEP"] == "10"
    assert env["KERNELGYM_CORE_DUMP_DIR"].endswith("logs/core_dumps/" + service._hostname())


def test_runtime_overrides_can_set_worker_capacity() -> None:
    values = service._apply_runtime_overrides(
        {"CPU_COMPILE_WORKERS": "24", "GPU_DEVICES": "auto"},
        type(
            "Args",
            (),
            {"cpu_compile_workers": 3, "gpu_devices": "0,2", "redis_remote_access": True},
        )(),
    )

    assert values["CPU_COMPILE_WORKERS"] == "3"
    assert values["GPU_DEVICES"] == "0,2"
    assert values["KERNELGYM_REDIS_REMOTE_ACCESS"] == "true"


def test_resolve_gpu_devices_auto_uses_container_visible_count(monkeypatch, capsys) -> None:
    monkeypatch.setattr(service, "_detect_visible_gpu_count", lambda: 4)

    values = service._resolve_gpu_devices({"GPU_DEVICES": "auto"})

    assert values["GPU_DEVICES"] == "[0,1,2,3]"
    assert "source=auto visible_count=4 devices=[0,1,2,3]" in capsys.readouterr().out


def test_resolve_gpu_devices_validates_explicit_logical_subset(monkeypatch) -> None:
    monkeypatch.setattr(service, "_detect_visible_gpu_count", lambda: 4)

    values = service._resolve_gpu_devices({"GPU_DEVICES": "0, 2"})

    assert values["GPU_DEVICES"] == "[0,2]"


@pytest.mark.parametrize("configured", ["", "4", "0,0", "gpu0", "[true]", "[]"])
def test_resolve_gpu_devices_rejects_invalid_or_unavailable_selection(monkeypatch, configured) -> None:
    monkeypatch.setattr(service, "_detect_visible_gpu_count", lambda: 4)

    with pytest.raises(SystemExit):
        service._resolve_gpu_devices({"GPU_DEVICES": configured})


def test_detect_visible_gpu_count_requires_working_torch(monkeypatch) -> None:
    monkeypatch.setattr(service, "_detect_visible_gpu_count_with_torch", lambda: None)

    with pytest.raises(SystemExit, match="configured PyTorch runtime"):
        service._detect_visible_gpu_count()


def test_detect_visible_gpu_count_rejects_empty_container(monkeypatch) -> None:
    monkeypatch.setattr(service, "_detect_visible_gpu_count_with_torch", lambda: 0)

    with pytest.raises(SystemExit, match="No CUDA devices are visible"):
        service._detect_visible_gpu_count()


@pytest.mark.parametrize("configured", ["auto", "[]", "0,0", "gpu0", "[true]"])
def test_settings_gpu_devices_parser_fails_loudly(configured) -> None:
    from kernelgym.config.settings import _parse_gpu_devices_value

    with pytest.raises(ValueError):
        _parse_gpu_devices_value(configured)


def test_settings_gpu_devices_parser_accepts_explicit_subset() -> None:
    from kernelgym.config.settings import _parse_gpu_devices_value

    assert _parse_gpu_devices_value("0,2") == [0, 2]


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
    _isolate_worker_start(monkeypatch)
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
    _isolate_worker_start(monkeypatch)
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
    _isolate_worker_start(monkeypatch)
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


def test_clear_expected_workers_for_host_ignores_poisoned_hostname_env(monkeypatch) -> None:
    """A primary restart must only clear its own host's expected-worker
    registrations; wiping the shared set would strip worker nodes of
    supervision. HOSTNAME is untrusted process input and cannot add an owner."""

    class FakeClient:
        def __init__(self):
            self.sets = {
                "kernelgym:expected_workers": {
                    "host-a_gpu_0",
                    "host-b_gpu_0",
                    "legacy_cpu_0",
                }
            }
            self.hashes = {
                "kernelgym:expected_worker:host-a_gpu_0": {"hostname": "host-a"},
                "kernelgym:expected_worker:host-b_gpu_0": {"hostname": "host-b"},
            }

        def smembers(self, key):
            return set(self.sets.get(key, set()))

        def hget(self, key, field):
            return self.hashes.get(key, {}).get(field)

        def hgetall(self, key):
            return dict(self.hashes.get(key, {}))

        def srem(self, key, member):
            self.sets.get(key, set()).discard(member)

        def delete(self, *keys):
            for key in keys:
                self.hashes.pop(key, None)

    # Simulate host B inheriting host A's identity from its environment.
    monkeypatch.setenv("HOSTNAME", "host-a")
    client = FakeClient()
    assert service._clear_expected_workers_for_host(client, "host-b") is True
    assert client.sets["kernelgym:expected_workers"] == {"host-a_gpu_0"}
    assert "kernelgym:expected_worker:host-a_gpu_0" in client.hashes
    assert "kernelgym:expected_worker:host-b_gpu_0" not in client.hashes


def test_default_stop_grace_exceeds_worker_drain(monkeypatch) -> None:
    monkeypatch.setenv("KERNELGYM_WORKER_SHUTDOWN_DRAIN_SEC", "17")

    assert service._default_stop_grace_seconds() == 47
    assert service.build_parser().parse_args(["stop"]).graceful_seconds is None


def test_stop_closes_api_admission_before_waiting_on_monitor(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(service, "_profile_values", lambda _profile: {"REDIS_HOST": "192.0.2.1"})
    monkeypatch.setattr(
        service,
        "_kill_processes",
        lambda pattern, description, *args, **kwargs: calls.append((pattern, description)) or True,
    )
    monkeypatch.setattr(service, "_collect_pids", lambda _pattern: [])

    args = service.build_parser().parse_args(["stop", "--profile", "v1"])
    assert service.cmd_stop(args) == 0
    assert calls[:2] == [
        ("kernelgym.server.api.server", "KernelGym API server"),
        ("kernelgym.worker.worker_monitor", "KernelGym worker monitor"),
    ]
    assert [pattern for pattern, _description in calls].count("kernelgym.server.api.server") == 1


def test_stop_aborts_before_monitor_when_api_admission_cannot_close(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(service, "_profile_values", lambda _profile: {"REDIS_HOST": "192.0.2.1"})

    def stop_processes(pattern, description, *args, **kwargs):  # noqa: ARG001
        calls.append(pattern)
        return pattern != "kernelgym.server.api.server"

    monkeypatch.setattr(service, "_kill_processes", stop_processes)

    args = service.build_parser().parse_args(["stop", "--profile", "v1"])
    assert service.cmd_stop(args) == 1
    assert calls == ["kernelgym.server.api.server"]


def test_authenticated_worker_stop_requires_complete_session_drain(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=1234, start_ticks="88", state="S", process_group=1234, session_id=1234)
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: identity)
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: True)
    observed_by_wait = []
    monkeypatch.setattr(
        service,
        "_wait_for_session_drain",
        lambda session_id, groups, timeout: observed_by_wait.append((session_id, set(groups), timeout)) or False,
    )
    forced = []
    monkeypatch.setattr(
        service,
        "_force_kill_worker_session",
        lambda session_id, **kwargs: forced.append((session_id, kwargs)) or (True, ""),
    )
    signals = []
    monkeypatch.setattr(service.os, "killpg", lambda process_group, signum: signals.append((process_group, signum)))

    stopped, reason = service._stop_authenticated_worker_group(
        "worker_gpu_0",
        pid=1234,
        expected_start_ticks="88",
        process_group=1234,
        graceful_seconds=5,
    )

    assert stopped is True
    assert reason == ""
    assert signals == [(1234, signal.SIGTERM)]
    assert observed_by_wait == [(1234, {1234}, 5)]
    assert forced == [
        (
            1234,
            {
                "expected_leader_start_ticks": "88",
                "observed_process_groups": {1234},
            },
        )
    ]


def test_session_drain_retries_transient_proc_scan_error(monkeypatch) -> None:
    scans = iter([RuntimeError("transient incomplete stat"), True])

    def scan(session_id, groups):  # noqa: ARG001
        result = next(scans)
        if isinstance(result, Exception):
            raise result
        return result

    monotonic_values = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr(service, "_session_is_drained", scan)
    monkeypatch.setattr(service.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)

    assert service._wait_for_session_drain(1234, {1234}, 1.0) is True


def test_session_drain_fails_closed_when_proc_scans_never_complete(monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "_session_is_drained",
        lambda session_id, groups: (_ for _ in ()).throw(RuntimeError("incomplete proc scan")),
    )
    monkeypatch.setattr(service.time, "monotonic", lambda: 0.0)

    with pytest.raises(RuntimeError, match="Could not complete session 1234 drain scan"):
        service._wait_for_session_drain(1234, {1234}, 0)


def test_authenticated_worker_exit_during_cmdline_check_uses_session_drain_proof(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=1234, start_ticks="88", state="S", process_group=1234, session_id=1234)
    identities = iter([identity, None])
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: False)
    drains = []
    monkeypatch.setattr(
        service,
        "_wait_for_session_drain",
        lambda session_id, groups, timeout: drains.append((session_id, set(groups), timeout)) or True,
    )
    monkeypatch.setattr(
        service.os,
        "killpg",
        lambda *args: (_ for _ in ()).throw(AssertionError("an identity mismatch must never be signalled")),
    )

    stopped, reason = service._stop_authenticated_worker_group(
        "worker_gpu_0",
        pid=1234,
        expected_start_ticks="88",
        process_group=1234,
        graceful_seconds=5,
    )

    assert stopped is True
    assert reason == ""
    assert drains == [(1234, {1234}, 5)]


def test_authenticated_worker_exit_during_cmdline_check_escalates_after_grace(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=1234, start_ticks="88", state="S", process_group=1234, session_id=1234)
    identities = iter([identity, None])
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: False)
    monkeypatch.setattr(service, "_wait_for_session_drain", lambda *_args: False)
    cleanups = []
    monkeypatch.setattr(
        service,
        "_force_kill_worker_session",
        lambda session_id, **kwargs: cleanups.append((session_id, kwargs)) or (True, ""),
    )

    stopped, reason = service._stop_authenticated_worker_group(
        "worker_gpu_0",
        pid=1234,
        expected_start_ticks="88",
        process_group=1234,
        graceful_seconds=5,
    )

    assert stopped is True
    assert reason == ""
    assert cleanups == [
        (
            1234,
            {
                "expected_leader_start_ticks": "88",
                "observed_process_groups": {1234},
            },
        )
    ]


def test_authenticated_generation_exit_before_sigterm_escalates_after_grace(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=1234, start_ticks="88", state="S", process_group=1234, session_id=1234)
    changed = service._ProcessIdentity(pid=1234, start_ticks="99", state="Z", process_group=1234, session_id=1234)
    identities = iter([identity, changed, None])
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: True)
    monkeypatch.setattr(service, "_wait_for_session_drain", lambda *_args: False)
    cleanups = []
    monkeypatch.setattr(
        service,
        "_force_kill_worker_session",
        lambda session_id, **kwargs: cleanups.append((session_id, kwargs)) or (True, ""),
    )

    stopped, reason = service._stop_authenticated_worker_group(
        "worker_gpu_0",
        pid=1234,
        expected_start_ticks="88",
        process_group=1234,
        graceful_seconds=5,
    )

    assert stopped is True
    assert reason == ""
    assert cleanups[0][0] == 1234
    assert cleanups[0][1]["expected_leader_start_ticks"] == "88"


def test_authenticated_live_cmdline_mismatch_fails_without_drain_wait(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=1234, start_ticks="88", state="S", process_group=1234, session_id=1234)
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: identity)
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: False)
    monkeypatch.setattr(
        service,
        "_wait_for_session_drain",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live mismatch must fail immediately")),
    )

    stopped, reason = service._stop_authenticated_worker_group(
        "worker_gpu_0",
        pid=1234,
        expected_start_ticks="88",
        process_group=1234,
        graceful_seconds=150,
    )

    assert stopped is False
    assert "command line no longer belongs" in reason


def test_authenticated_worker_exit_before_final_revalidation_uses_session_cleanup(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=1234, start_ticks="88", state="S", process_group=1234, session_id=1234)
    identities = iter([identity, None])
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: True)
    monkeypatch.setattr(service, "_session_is_drained", lambda session_id, groups: False)
    cleanups = []

    def session_cleanup(session_id, *, expected_leader_start_ticks, observed_process_groups):
        cleanups.append((session_id, expected_leader_start_ticks, set(observed_process_groups)))
        return True, ""

    monkeypatch.setattr(service, "_force_kill_worker_session", session_cleanup)
    monkeypatch.setattr(
        service.os,
        "killpg",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("a disappeared leader's bare PGID must never be signalled")
        ),
    )

    stopped, reason = service._stop_authenticated_worker_group(
        "worker_gpu_0",
        pid=1234,
        expected_start_ticks="88",
        process_group=1234,
        graceful_seconds=5,
    )

    assert stopped is True
    assert reason == ""
    assert cleanups == [(1234, "88", {1234})]


def test_session_force_kill_freezes_newly_discovered_inner_groups_to_fixed_point(monkeypatch) -> None:
    def member(pid, state, process_group):
        return service._ProcessIdentity(
            pid=pid,
            start_ticks=str(pid * 10),
            state=state,
            process_group=process_group,
            session_id=100,
        )

    snapshots = iter(
        [
            [member(100, "S", 100), member(201, "S", 200)],
            [member(100, "T", 100), member(201, "T", 200), member(301, "T", 300)],
            [member(100, "T", 100), member(201, "T", 200), member(301, "T", 300)],
            [member(100, "T", 100), member(201, "T", 200), member(301, "T", 300)],
        ]
    )
    monkeypatch.setattr(service, "_live_session_members", lambda session_id: next(snapshots))
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    signals = []
    monkeypatch.setattr(service.os, "killpg", lambda process_group, signum: signals.append((process_group, signum)))
    drained = []
    monkeypatch.setattr(
        service,
        "_wait_for_session_drain",
        lambda session_id, groups, timeout: drained.append((session_id, set(groups))) or True,
    )

    stopped, reason = service._force_kill_worker_session(
        100,
        expected_leader_start_ticks="1000",
        observed_process_groups={100},
    )

    assert stopped is True
    assert reason == ""
    assert signals == [
        (100, signal.SIGSTOP),
        (200, signal.SIGSTOP),
        (100, signal.SIGSTOP),
        (200, signal.SIGSTOP),
        (300, signal.SIGSTOP),
        (100, signal.SIGKILL),
        (200, signal.SIGKILL),
        (300, signal.SIGKILL),
    ]
    assert drained == [(100, {100, 200, 300})]


def test_session_freeze_fails_closed_when_member_never_stops(monkeypatch) -> None:
    member = service._ProcessIdentity(
        pid=100,
        start_ticks="88",
        state="D",
        process_group=100,
        session_id=100,
    )
    monkeypatch.setattr(service, "_live_session_members", lambda session_id: [member])
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(service.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    signals = []
    monkeypatch.setattr(service.os, "killpg", lambda process_group, signum: signals.append((process_group, signum)))

    frozen, groups, reason = service._freeze_worker_session(
        100,
        expected_leader_start_ticks="88",
        timeout=0.5,
    )

    assert frozen is False
    assert groups == {100}
    assert "did not freeze" in reason
    assert signals == [(100, signal.SIGSTOP)]


def test_session_drain_requires_empty_sid_and_esrch_for_every_observed_group(monkeypatch) -> None:
    monkeypatch.setattr(service, "_live_session_members", lambda session_id: [])
    outcomes = {100: True, 200: False}
    monkeypatch.setattr(service, "_process_group_is_drained", lambda process_group: outcomes[process_group])

    assert service._session_is_drained(100, {100, 200}) is False


def test_discovered_service_root_escalates_its_complete_session(monkeypatch) -> None:
    identity = service._ProcessIdentity(
        pid=4321,
        start_ticks="77",
        state="S",
        process_group=4321,
        session_id=4321,
    )
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: identity)
    monkeypatch.setattr(service, "_cmdline_matches_pattern", lambda pid, pattern: True)
    monkeypatch.setattr(service, "_wait_for_session_drain", lambda *args, **kwargs: False)
    forced = []
    monkeypatch.setattr(
        service,
        "_force_kill_worker_session",
        lambda session_id, **kwargs: forced.append((session_id, kwargs)) or (True, ""),
    )
    signals = []
    monkeypatch.setattr(service.os, "killpg", lambda process_group, signum: signals.append((process_group, signum)))

    stopped, reason = service._stop_discovered_process_group(4321, "worker-pattern", 3)

    assert stopped is True
    assert reason == ""
    assert signals == [(4321, signal.SIGTERM)]
    assert forced == [
        (
            4321,
            {
                "expected_leader_start_ticks": "77",
                "observed_process_groups": {4321},
            },
        )
    ]


def test_discovered_service_root_exit_during_cmdline_check_escalates_after_grace(monkeypatch) -> None:
    identity = service._ProcessIdentity(
        pid=4321,
        start_ticks="77",
        state="S",
        process_group=4321,
        session_id=4321,
    )
    identities = iter([identity, None])
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(service, "_cmdline_matches_pattern", lambda pid, pattern: False)
    monkeypatch.setattr(service, "_wait_for_session_drain", lambda *args, **kwargs: False)
    forced = []
    monkeypatch.setattr(
        service,
        "_force_kill_worker_session",
        lambda session_id, **kwargs: forced.append((session_id, kwargs)) or (True, ""),
    )

    stopped, reason = service._stop_discovered_process_group(4321, "worker-pattern", 3)

    assert stopped is True
    assert reason == ""
    assert forced == [
        (
            4321,
            {
                "expected_leader_start_ticks": "77",
                "observed_process_groups": {4321},
            },
        )
    ]


def test_safe_registered_session_is_generation_cas_deleted(monkeypatch) -> None:
    class FakeClient:
        def __init__(self):
            self.eval_args = None

        def hgetall(self, key):
            return {
                "pid": "1234",
                "proc_start_ticks": "88",
                "process_group": "1234",
                "session_id": "1234",
                "device": "cuda:0",
            }

        def eval(self, *args):
            self.eval_args = args
            return 1

    stopped_with = []
    monkeypatch.setattr(
        service,
        "_stop_authenticated_worker_group",
        lambda *args, **kwargs: stopped_with.append(kwargs) or (True, ""),
    )
    client = FakeClient()

    assert service._drain_registered_worker(client, "worker_gpu_0", graceful_seconds=5) is True
    assert stopped_with[0]["session_id"] == 1234
    assert client.eval_args is not None
    assert "session_id" in client.eval_args[0]
    assert client.eval_args[-2:] == ("1", "worker_gpu_0")


def test_unsafe_registered_group_is_quarantined_without_deleting_map(monkeypatch) -> None:
    class FakeClient:
        def __init__(self):
            self.eval_called = False

        def hgetall(self, key):
            return {
                "pid": "1234",
                "proc_start_ticks": "88",
                "process_group": "1234",
                "session_id": "1234",
                "device": "cuda:0",
            }

        def eval(self, *args):
            self.eval_called = True
            return 1

    client = FakeClient()
    monkeypatch.setattr(
        service,
        "_stop_authenticated_worker_group",
        lambda *args, **kwargs: (False, "group survived"),
    )
    quarantines = []
    monkeypatch.setattr(
        service,
        "_quarantine_unsafe_worker_group",
        lambda client, worker_id, device, reason: quarantines.append((worker_id, device, reason)),
    )

    assert service._drain_registered_worker(client, "worker_gpu_0", graceful_seconds=5) is False
    assert client.eval_called is False
    assert quarantines == [("worker_gpu_0", "cuda:0", "group survived")]


def test_register_expected_worker_writes_full_identity_with_if_empty_lua(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=2345, start_ticks="99", state="S", process_group=2345, session_id=2345)
    service._LAUNCHED_IDENTITIES[2345] = identity
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: identity)
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: True)

    class FakeClient:
        def __init__(self):
            self.args = None

        def eval(self, *args):
            self.args = args
            return 1

    client = FakeClient()
    service._register_expected_worker(client, "worker_gpu_0", "cuda:0", "node21", "v1", 2345)

    assert client.args is not None
    assert client.args[1] == 3
    assert "proc_start_ticks" in client.args[0]
    assert "process_group" in client.args[0]
    assert "session_id" in client.args[0]
    assert "2345" in client.args
    assert "99" in client.args
    assert 2345 not in service._LAUNCHED_IDENTITIES


def test_register_rejection_drains_just_spawned_group_and_aborts(monkeypatch) -> None:
    identity = service._ProcessIdentity(pid=3456, start_ticks="101", state="S", process_group=3456, session_id=3456)
    service._LAUNCHED_IDENTITIES[3456] = identity
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: identity)
    monkeypatch.setattr(service, "_cmdline_matches_worker", lambda pid, worker_id: True)
    aborts = []
    monkeypatch.setattr(
        service,
        "_abort_unregistered_launch",
        lambda client, worker_id, device, pid, reason: aborts.append((worker_id, device, pid, reason)) or True,
    )

    class FakeClient:
        def eval(self, *args):
            return 0

    with pytest.raises(SystemExit, match="replacement launch aborted"):
        service._register_expected_worker(FakeClient(), "worker_gpu_0", "cuda:0", "node21", "v1", 3456)

    assert aborts == [("worker_gpu_0", "cuda:0", 3456, "an older process generation still owns the worker map")]
    service._LAUNCHED_IDENTITIES.pop(3456, None)


def test_launch_identity_read_error_reaps_exact_handle_and_quarantines(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        pid = 4567

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(service, "ensure_core_dump_dir", lambda value: tmp_path)
    monkeypatch.setattr(service, "prune_core_dumps", lambda *args: None)
    monkeypatch.setattr(
        service,
        "_read_process_identity",
        lambda pid: (_ for _ in ()).throw(RuntimeError("bad proc stat")),
    )
    quarantines = []
    monkeypatch.setattr(
        service,
        "_quarantine_unsafe_worker_group",
        lambda client, worker_id, device, reason: quarantines.append((worker_id, device, reason)),
    )

    with pytest.raises(RuntimeError, match="Could not record launch identity"):
        service._launch_background(
            ["python", "-m", "kernelgym.worker.single_worker", "--worker-id", "w", "--device", "cuda:0"],
            tmp_path / "worker.log",
            {},
        )

    assert process.returncode == -signal.SIGKILL
    assert quarantines[0][:2] == ("w", "cuda:0")
    assert "before session authentication" in quarantines[0][2]


def test_unregistered_launch_without_start_ticks_never_signals_bare_scope(monkeypatch) -> None:
    monkeypatch.setattr(service, "_read_process_identity", lambda pid: None)
    monkeypatch.setattr(service, "_session_is_drained", lambda session_id, groups: False)
    monkeypatch.setattr(
        service,
        "_force_kill_worker_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not signal a bare SID")),
    )
    quarantines = []
    monkeypatch.setattr(
        service,
        "_quarantine_unsafe_worker_group",
        lambda client, worker_id, device, reason: quarantines.append((worker_id, device, reason)),
    )

    stopped = service._abort_unregistered_launch(None, "worker_gpu_0", "cuda:0", 4567, "registration failed")

    assert stopped is False
    assert quarantines[0][:2] == ("worker_gpu_0", "cuda:0")
    assert "no authenticated start_ticks" in quarantines[0][2]


def test_start_local_aborts_before_launch_when_stop_is_incomplete(monkeypatch) -> None:
    monkeypatch.setattr(service, "_resolve_gpu_devices", lambda values: {**values, "GPU_DEVICES": "[0]"})
    monkeypatch.setattr(service, "cmd_stop", lambda args: 1)
    monkeypatch.setattr(
        service,
        "_ensure_redis",
        lambda values: (_ for _ in ()).throw(AssertionError("must not start Redis")),
    )
    args = type("Args", (), {"profile": "v1", "no_stop_first": True})()

    with pytest.raises(SystemExit, match="not safely drained"):
        service.cmd_start_local(args)


def test_start_local_resolves_gpu_devices_before_stopping(monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "_resolve_gpu_devices",
        lambda values: (_ for _ in ()).throw(SystemExit("no visible GPUs")),
    )
    monkeypatch.setattr(
        service,
        "cmd_stop",
        lambda args: (_ for _ in ()).throw(AssertionError("must not stop before GPU preflight")),
    )

    with pytest.raises(SystemExit, match="no visible GPUs"):
        service.cmd_start_local(type("Args", (), {"profile": "v1"})())


def test_worker_node_start_stops_monitor_and_aborts_on_undrained_local_worker(tmp_path, monkeypatch) -> None:
    server_env = tmp_path / "server.env"
    server_env.write_text(
        "API_HOST=192.0.2.1\nREDIS_HOST=192.0.2.1\nGPU_DEVICES=[0]\nCPU_COMPILE_WORKERS=0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "_check_worker_connectivity", lambda values: None)
    monkeypatch.setattr(service, "_http_post_json", lambda url: {"node_id": "node-test", "hostname": "node-test"})
    monkeypatch.setattr(service, "_redis_client", lambda values: object())
    calls = []

    def stop_processes(pattern, description, *args, **kwargs):
        calls.append(pattern)
        return pattern != "kernelgym.worker.single_worker"

    monkeypatch.setattr(service, "_kill_processes", stop_processes)
    monkeypatch.setattr(
        service,
        "_clear_expected_workers_for_host",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not clear process maps")),
    )

    with pytest.raises(SystemExit, match="did not drain safely"):
        service.cmd_start_worker_node(type("Args", (), {"server_env": str(server_env)})())

    assert calls == ["kernelgym.worker.worker_monitor", "kernelgym.worker.single_worker"]
