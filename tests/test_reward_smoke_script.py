import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_reward.py"


def load_script():
    spec = importlib.util.spec_from_file_location("test_reward_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_targeted_reward_smoke_enables_real_profile_and_split_affinity() -> None:
    script = load_script()

    payload = script._build_request(  # noqa: SLF001
        "startup",
        timeout=600,
        run_performance=True,
        target_hostname="host-a",
    )

    assert payload["force_refresh"] is True
    assert payload["run_performance"] is True
    assert payload["enable_profiling"] is True
    assert payload["split_compile_and_execute"] is True
    assert payload["target_hostname"] == "host-a"


def test_strict_smoke_rejects_no_performance_mode(monkeypatch) -> None:
    script = load_script()
    monkeypatch.setattr(sys, "argv", ["test_reward.py", "--require-correct", "--no-perf"])

    try:
        script.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("contradictory strict/no-perf flags must be rejected")


def test_strict_smoke_requires_completed_status(monkeypatch) -> None:
    script = load_script()
    monkeypatch.setattr(sys, "argv", ["test_reward.py", "--require-correct"])
    monkeypatch.setattr(script, "_http_get_json", lambda *_args, **_kwargs: {"status": "healthy"})
    monkeypatch.setattr(
        script,
        "_http_post_json",
        lambda *_args, **_kwargs: (
            200,
            {
                "status": "timeout",
                "compiled": True,
                "correctness": True,
                "reference_runtime": 1.0,
                "kernel_runtime": 1.0,
            },
        ),
    )

    assert script.main() == 1
