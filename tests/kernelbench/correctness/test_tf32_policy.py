"""KernelBench shared TF32 execution-policy tests."""

import pytest


def _get_execution_policy_module():
    pytest.importorskip("torch")
    from kernelgym.toolkit.kernelbench import execution_policy

    return execution_policy


def _backend_state(execution_policy, torch):  # noqa: ANN001
    state = {}
    for label, path in (
        ("cudnn.conv.fp32_precision", ("cudnn", "conv", "fp32_precision")),
        ("cudnn.allow_tf32", ("cudnn", "allow_tf32")),
        ("cuda.matmul.fp32_precision", ("cuda", "matmul", "fp32_precision")),
        ("cuda.matmul.allow_tf32", ("cuda", "matmul", "allow_tf32")),
    ):
        exists, value = execution_policy._read_backend_attr(torch.backends, path)
        if exists:
            state[label] = value
    if hasattr(torch, "get_float32_matmul_precision"):
        state["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    return state


def test_tf32_execution_context_forces_and_restores_backend_state() -> None:
    execution_policy = _get_execution_policy_module()
    torch = pytest.importorskip("torch")

    targets = []
    for label, path in (
        ("cudnn.conv.fp32_precision", ("cudnn", "conv", "fp32_precision")),
        ("cudnn.allow_tf32", ("cudnn", "allow_tf32")),
        ("cuda.matmul.fp32_precision", ("cuda", "matmul", "fp32_precision")),
        ("cuda.matmul.allow_tf32", ("cuda", "matmul", "allow_tf32")),
    ):
        exists, value = execution_policy._read_backend_attr(torch.backends, path)
        if exists:
            targets.append((label, path, value))

    metadata = {}
    with execution_policy.tf32_execution_context(metadata, stage="test"):
        assert metadata["test_tf32_enabled"] is True
        forced = metadata["test_tf32_state_forced"]
        if "cudnn.conv.fp32_precision" in forced:
            assert forced["cudnn.conv.fp32_precision"] == "tf32"
            assert (
                execution_policy._read_backend_attr(
                    torch.backends,
                    ("cudnn", "conv", "fp32_precision"),
                )[1]
                == "tf32"
            )
        elif "cudnn.allow_tf32" in forced:
            assert forced["cudnn.allow_tf32"] == "True"
            assert execution_policy._read_backend_attr(torch.backends, ("cudnn", "allow_tf32"))[1] is True
        if "cuda.matmul.fp32_precision" in forced:
            assert forced["cuda.matmul.fp32_precision"] == "tf32"
            assert (
                execution_policy._read_backend_attr(
                    torch.backends,
                    ("cuda", "matmul", "fp32_precision"),
                )[1]
                == "tf32"
            )
        elif "cuda.matmul.allow_tf32" in forced:
            assert forced["cuda.matmul.allow_tf32"] == "True"
            assert (
                execution_policy._read_backend_attr(
                    torch.backends,
                    ("cuda", "matmul", "allow_tf32"),
                )[1]
                is True
            )

    for _, path, old_value in targets:
        assert execution_policy._read_backend_attr(torch.backends, path)[1] == old_value


def test_tf32_execution_context_restores_after_exception() -> None:
    execution_policy = _get_execution_policy_module()
    torch = pytest.importorskip("torch")
    before = _backend_state(execution_policy, torch)

    with pytest.raises(RuntimeError, match="test failure"):
        with execution_policy.tf32_execution_context(stage="test"):
            raise RuntimeError("test failure")

    assert _backend_state(execution_policy, torch) == before


def test_tf32_new_api_restores_legacy_matmul_precision_view() -> None:
    execution_policy = _get_execution_policy_module()
    torch = pytest.importorskip("torch")
    has_matmul_precision, _ = execution_policy._read_backend_attr(
        torch.backends,
        ("cuda", "matmul", "fp32_precision"),
    )
    if not has_matmul_precision or not hasattr(torch, "get_float32_matmul_precision"):
        pytest.skip("PyTorch new fp32_precision API is unavailable")

    before = torch.get_float32_matmul_precision()
    with execution_policy.tf32_execution_context(stage="test"):
        assert (
            execution_policy._read_backend_attr(
                torch.backends,
                ("cuda", "matmul", "fp32_precision"),
            )[1]
            == "tf32"
        )

    assert torch.get_float32_matmul_precision() == before


def test_tf32_execution_context_can_be_explicitly_disabled() -> None:
    execution_policy = _get_execution_policy_module()

    metadata = {}
    with execution_policy.tf32_execution_context(metadata, stage="test", enabled=False):
        assert metadata["test_tf32_enabled"] is False

    assert "test_tf32_state_forced" not in metadata


def test_kernelbench_fp32_tolerance_is_1e_3() -> None:
    torch = pytest.importorskip("torch")
    from kernelgym.toolkit.kernelbench.correctness import get_tolerance_for_dtype

    assert get_tolerance_for_dtype(torch.float32) == 1e-3
    assert get_tolerance_for_dtype(torch.float16) == 1e-2
    assert get_tolerance_for_dtype(torch.bfloat16) == 1e-2
