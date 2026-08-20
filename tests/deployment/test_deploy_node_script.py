import importlib.util
from argparse import Namespace
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEPLOY_NODE = ROOT / "scripts" / "deploy_node.py"


def load_deploy_node():
    spec = importlib.util.spec_from_file_location("deploy_node_script", DEPLOY_NODE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deploy_node_validate_requires_rank_for_multi_node() -> None:
    deploy_node = load_deploy_node()
    args = Namespace(nnodes=2, node_rank=None, master_addr="192.168.16.40", master_port=20111)

    try:
        deploy_node.validate(args)
    except SystemExit as exc:
        assert "--node-rank is required" in str(exc)
    else:
        raise AssertionError("validate should reject missing node rank")


def test_deploy_node_parser_exposes_runtime_options() -> None:
    deploy_node = load_deploy_node()
    monkeypatch_args = [
        "--cluster",
        "--clear-cache",
        "--block-terminal",
        "--cpu-compile-workers",
        "3",
        "--gpu-devices",
        "0,1,2,3",
    ]

    old_argv = deploy_node.sys.argv
    try:
        deploy_node.sys.argv = ["deploy_node.py", *monkeypatch_args]
        args = deploy_node.parse_args()
    finally:
        deploy_node.sys.argv = old_argv

    assert args.cluster is True
    assert args.clear_cache is True
    assert args.block_terminal is True
    assert args.cpu_compile_workers == 3
    assert args.gpu_devices == "0,1,2,3"
    assert args.no_startup_warmup is False
    assert args.startup_warmup_timeout == 1800


def test_deploy_node_block_terminal_stops_local_services(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(deploy_node, "wait_for_shutdown_signal", lambda: deploy_node.signal.SIGTERM)
    monkeypatch.setattr(deploy_node, "run", lambda command: calls.append(command))

    deploy_node.block_terminal()

    assert calls == [
        [
            deploy_node.sys.executable,
            "-m",
            "kernelgym.cli.service",
            "stop",
            "--profile",
            "v1",
        ]
    ]


def test_deploy_node_finish_only_blocks_when_requested(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(deploy_node, "block_terminal", lambda: calls.append("block"))

    assert deploy_node.finish_deployment(Namespace(block_terminal=False)) == 0
    assert calls == []
    assert deploy_node.finish_deployment(Namespace(block_terminal=True)) == 0
    assert calls == ["block"]


def test_deploy_node_finish_runs_node_warmup_before_blocking(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(
        deploy_node,
        "run_startup_warmup",
        lambda host, timeout: calls.append(("warmup", host, timeout)),
    )
    monkeypatch.setattr(deploy_node, "block_terminal", lambda: calls.append(("block",)))

    args = Namespace(block_terminal=True, no_startup_warmup=False, startup_warmup_timeout=1234)
    assert deploy_node.finish_deployment(args, api_host="192.168.16.21") == 0
    assert calls == [("warmup", "192.168.16.21", 1234), ("block",)]


def test_deploy_node_waits_for_all_registered_local_workers(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    heartbeat = datetime.now().isoformat()
    snapshots = iter(
        [
            {
                "gpu": {
                    "hostname": "host-a",
                    "device": "cuda:0",
                    "status": "online",
                    "last_heartbeat": heartbeat,
                    "online": "true",
                    "health_state": "initializing",
                    "accepting_tasks": "false",
                },
                "cpu": {
                    "hostname": "host-a",
                    "device": "cpu",
                    "status": "online",
                    "last_heartbeat": heartbeat,
                    "online": "true",
                },
            },
            {
                "gpu": {
                    "hostname": "host-a",
                    "device": "cuda:0",
                    "status": "online",
                    "last_heartbeat": heartbeat,
                    "online": "true",
                    "health_state": "healthy",
                    "accepting_tasks": "true",
                },
                "cpu": {
                    "hostname": "host-a",
                    "device": "cpu",
                    "status": "online",
                    "last_heartbeat": heartbeat,
                    "online": "true",
                },
            },
        ]
    )
    monkeypatch.setattr(deploy_node, "_http_get_json", lambda _url: next(snapshots))
    monkeypatch.setattr(deploy_node.time, "sleep", lambda _seconds: None)

    deploy_node.wait_node_workers("192.168.16.21", "host-a", timeout=30)


def test_deploy_node_ignores_stale_or_offline_worker_registrations(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    heartbeat = datetime.now().isoformat()
    snapshot = {
        "gpu": {
            "hostname": "host-a",
            "device": "cuda:0",
            "status": "online",
            "last_heartbeat": heartbeat,
            "online": "true",
            "health_state": "healthy",
            "accepting_tasks": "true",
        },
        "cpu": {
            "hostname": "host-a",
            "device": "cpu",
            "status": "online",
            "last_heartbeat": heartbeat,
            "online": "true",
        },
        "old_gpu": {
            "hostname": "host-a",
            "device": "cuda:1",
            "status": "offline",
            "last_heartbeat": "2000-01-01T00:00:00",
            "online": "initializing",
            "health_state": "initializing",
            "accepting_tasks": "false",
        },
        "stale_gpu": {
            "hostname": "host-a",
            "device": "cuda:2",
            "status": "online",
            "last_heartbeat": "2000-01-01T00:00:00",
            "online": "true",
            "health_state": "healthy",
            "accepting_tasks": "true",
        },
    }
    monkeypatch.setattr(deploy_node, "_http_get_json", lambda _url: snapshot)

    deploy_node.wait_node_workers("192.168.16.21", "host-a", timeout=30)


def test_deploy_node_startup_warmup_is_targeted_and_strict(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(deploy_node.socket, "gethostname", lambda: "host-a")
    monkeypatch.setattr(
        deploy_node,
        "wait_node_workers",
        lambda host, hostname: calls.append(("wait", host, hostname)),
    )
    monkeypatch.setattr(deploy_node, "run", lambda command: calls.append(("run", command)))

    deploy_node.run_startup_warmup("192.168.16.21", 1800)

    assert calls[0] == ("wait", "192.168.16.21", "host-a")
    command = calls[1][1]
    assert command[:2] == [deploy_node.sys.executable, str(deploy_node.ROOT_DIR / "scripts" / "test_reward.py")]
    assert command[command.index("--target-hostname") + 1] == "host-a"
    assert command[command.index("--client-timeout") + 1] == "1800"
    assert command[command.index("--timeout") + 1] == "1740"
    assert "--require-correct" in command


def test_deploy_node_rejects_zero_cpu_workers_with_default_warmup() -> None:
    deploy_node = load_deploy_node()
    args = Namespace(
        nnodes=1,
        node_rank=None,
        master_addr="",
        master_port=20111,
        cpu_compile_workers=0,
        no_startup_warmup=False,
        startup_warmup_timeout=1800,
        cluster=False,
        join="",
    )

    try:
        deploy_node.validate(args)
    except SystemExit as exc:
        assert "requires --no-startup-warmup" in str(exc)
    else:
        raise AssertionError("zero CPU workers must fail before a required startup warmup")

    args.no_startup_warmup = True
    deploy_node.validate(args)


def test_deploy_node_main_rejects_master_with_nonzero_rank(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    monkeypatch.setattr(
        deploy_node,
        "parse_args",
        lambda: Namespace(nnodes=2, node_rank=1, master_addr="192.168.16.40", master_port=20111),
    )
    monkeypatch.setattr(deploy_node, "local_ids", lambda: {"192.168.16.40"})

    try:
        deploy_node.main()
    except SystemExit as exc:
        assert "--master-addr must use --node-rank 0" in str(exc)
    else:
        raise AssertionError("main should reject master node with nonzero rank")


def test_deploy_node_start_primary_waits_for_health(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(deploy_node, "run", lambda command: calls.append(("run", command)))
    monkeypatch.setattr(deploy_node, "wait_api", lambda addr: calls.append(("wait_api", addr)))

    deploy_node.start_primary(None)

    assert calls == [
        (
            "run",
            [
                deploy_node.sys.executable,
                "-m",
                "kernelgym.cli.service",
                "start-local",
                "--profile",
                "v1",
            ],
        ),
        ("wait_api", "127.0.0.1"),
    ]


def test_deploy_node_start_primary_enables_redis_remote_access(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(deploy_node, "run", lambda command: calls.append(("run", command)))
    monkeypatch.setattr(deploy_node, "wait_api", lambda addr: calls.append(("wait_api", addr)))

    deploy_node.start_primary(0, redis_remote_access=True)

    assert calls[0] == (
        "run",
        [
            deploy_node.sys.executable,
            "-m",
            "kernelgym.cli.service",
            "start-local",
            "--profile",
            "v1",
            "--redis-remote-access",
        ],
    )


def test_deploy_node_start_primary_clear_cache_stops_then_skips_second_stop(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(
        deploy_node,
        "run",
        lambda command, allow_failure=False: calls.append(("run", command, allow_failure)),
    )
    monkeypatch.setattr(deploy_node, "wait_api", lambda addr: calls.append(("wait_api", addr)))
    monkeypatch.setattr(deploy_node, "clear_local_caches", lambda: calls.append(("clear_local_caches",)))

    deploy_node.start_primary(None, clear_cache=True)

    assert calls == [
        (
            "run",
            [
                deploy_node.sys.executable,
                "-m",
                "kernelgym.cli.service",
                "stop",
                "--profile",
                "v1",
            ],
            True,
        ),
        ("clear_local_caches",),
        (
            "run",
            [
                deploy_node.sys.executable,
                "-m",
                "kernelgym.cli.service",
                "start-local",
                "--profile",
                "v1",
                "--no-stop-first",
            ],
            False,
        ),
        ("wait_api", "127.0.0.1"),
    ]


def test_deploy_node_start_worker_passes_runtime_overrides(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(
        deploy_node,
        "run",
        lambda command, allow_failure=False: calls.append(("run", command, allow_failure)),
    )
    monkeypatch.setattr(deploy_node, "wait_api", lambda addr: calls.append(("wait_api", addr)))

    deploy_node.start_worker("192.168.16.40", 1, cpu_compile_workers=7, gpu_devices="0,1,2,3")

    assert calls[-1] == (
        "run",
        [
            deploy_node.sys.executable,
            "-m",
            "kernelgym.cli.service",
            "start-worker-node",
            "--profile",
            "v1",
            "--master-addr",
            "192.168.16.40",
            "--node-rank",
            "1",
            "--cpu-compile-workers",
            "7",
            "--gpu-devices",
            "0,1,2,3",
        ],
        False,
    )
