import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def test_deploy_node_start_worker_passes_cpu_compile_workers(monkeypatch) -> None:
    deploy_node = load_deploy_node()
    calls = []
    monkeypatch.setattr(
        deploy_node,
        "run",
        lambda command, allow_failure=False: calls.append(("run", command, allow_failure)),
    )
    monkeypatch.setattr(deploy_node, "wait_api", lambda addr: calls.append(("wait_api", addr)))

    deploy_node.start_worker("192.168.16.40", 1, cpu_compile_workers=7)

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
        ],
        False,
    )
