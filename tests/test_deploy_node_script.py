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
