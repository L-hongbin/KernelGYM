from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_drkernel_tree_is_not_present() -> None:
    assert not (ROOT / "drkernel").exists()
    assert not (ROOT / "scripts" / "check_offline_eval_prereqs.sh").exists()


def test_reward_only_entrypoints_are_present() -> None:
    assert (ROOT / "kernelgym" / "server" / "api" / "server.py").exists()
    assert (ROOT / "kernelgym" / "worker" / "single_worker.py").exists()
    assert (ROOT / "create_venv.sh").exists()
    assert (ROOT / "kernelgym" / "deployment_profiles.py").exists()
    assert (ROOT / "scripts" / "start_container.sh").exists()
    assert (ROOT / "scripts" / "lock_gpu_clocks.sh").exists()
    assert not (ROOT / "setup.sh").exists()
    assert not (ROOT / "start_all_with_monitor.sh").exists()
    assert not (ROOT / "stop_all.sh").exists()
    assert not (ROOT / "kernelgym" / "cli" / "deploy.py").exists()


def test_source_lineage_docs_name_both_sources() -> None:
    docs = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "SOURCE_LINEAGE.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "IMPLEMENTATION_DIFFERENCES.md").read_text(encoding="utf-8"),
        ]
    )
    assert "KernelGYM-vllm018-cuda-agent" in docs
    assert "KernelGYM-lhb" in docs
    assert "drkernel" in docs


def test_precommit_uses_ruff_format_and_not_black() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "ruff-check" in config
    assert "ruff-format" in config
    assert "psf/black" not in config
    assert " id: black" not in config


def test_no_pure_python_forwarder_shell_entrypoints_remain() -> None:
    removed_wrappers = [
        ROOT / "setup.sh",
        ROOT / "start_all_with_monitor.sh",
        ROOT / "start_worker_node.sh",
        ROOT / "start_worker_multinode.sh",
        ROOT / "stop_all.sh",
        ROOT / "scripts" / "auto_configure.sh",
        ROOT / "scripts" / "detect_profile.sh",
    ]
    for wrapper in removed_wrappers:
        assert not wrapper.exists()
