"""Server request-default tests."""

from kernelgym.server.request_defaults import apply_runtime_defaults


def test_split_compile_profile_default_overrides_client_false() -> None:
    payload = {
        "task_id": "task",
        "split_compile_and_execute": False,
    }

    result = apply_runtime_defaults(
        payload,
        workflow_name="kernelbench",
        split_compile_and_execute=True,
    )

    assert result["split_compile_and_execute"] is True
    assert result["resources"] is None


def test_split_compile_profile_default_does_not_touch_internal_stage() -> None:
    payload = {
        "task_id": "task_compile",
        "split_compile_and_execute": False,
        "task_stage": "compile",
    }

    result = apply_runtime_defaults(
        payload,
        workflow_name="kernelbench",
        split_compile_and_execute=True,
    )

    assert result["split_compile_and_execute"] is False
