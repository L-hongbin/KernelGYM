"""KernelBench true-FP32 correctness-policy tests."""

import pytest


def _get_correctness_module():
    pytest.importorskip("torch")
    from kernelgym.toolkit.kernelbench import correctness

    return correctness


def test_true_fp32_correctness_context_forces_and_restores_backend_state(monkeypatch) -> None:
    correctness = _get_correctness_module()
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("KERNELGYM_CORRECTNESS_DISABLE_TF32", raising=False)

    targets = []
    for label, path in (
        ("cudnn.conv.fp32_precision", ("cudnn", "conv", "fp32_precision")),
        ("cudnn.allow_tf32", ("cudnn", "allow_tf32")),
        ("cuda.matmul.fp32_precision", ("cuda", "matmul", "fp32_precision")),
        ("cuda.matmul.allow_tf32", ("cuda", "matmul", "allow_tf32")),
    ):
        exists, value = correctness._read_backend_attr(torch.backends, path)
        if exists:
            targets.append((label, path, value))

    metadata = {}
    with correctness._true_fp32_correctness_context(metadata):
        assert metadata["correctness_tf32_disabled"] is True
        forced = metadata["correctness_tf32_state_forced"]
        if "cudnn.conv.fp32_precision" in forced:
            assert forced["cudnn.conv.fp32_precision"] == "ieee"
            assert correctness._read_backend_attr(torch.backends, ("cudnn", "conv", "fp32_precision"))[1] == "ieee"
        elif "cudnn.allow_tf32" in forced:
            assert forced["cudnn.allow_tf32"] == "False"
            assert correctness._read_backend_attr(torch.backends, ("cudnn", "allow_tf32"))[1] is False
        if "cuda.matmul.fp32_precision" in forced:
            assert forced["cuda.matmul.fp32_precision"] == "ieee"
            assert correctness._read_backend_attr(torch.backends, ("cuda", "matmul", "fp32_precision"))[1] == "ieee"
        elif "cuda.matmul.allow_tf32" in forced:
            assert forced["cuda.matmul.allow_tf32"] == "False"
            assert correctness._read_backend_attr(torch.backends, ("cuda", "matmul", "allow_tf32"))[1] is False

    for _, path, old_value in targets:
        assert correctness._read_backend_attr(torch.backends, path)[1] == old_value


def test_true_fp32_correctness_context_can_be_disabled(monkeypatch) -> None:
    correctness = _get_correctness_module()
    monkeypatch.setenv("KERNELGYM_CORRECTNESS_DISABLE_TF32", "0")

    metadata = {}
    with correctness._true_fp32_correctness_context(metadata):
        assert metadata["correctness_tf32_disabled"] is False

    assert "correctness_tf32_state_forced" not in metadata
