from __future__ import annotations

import torch

from kernelgym.toolkit.kernelbench import correctness


def _get_attr(path: tuple[str, ...]):
    obj = torch.backends
    for name in path[:-1]:
        try:
            obj = getattr(obj, name)
        except Exception:
            return None, False
        if obj is None:
            return None, False
    attr = path[-1]
    try:
        return getattr(obj, attr), True
    except Exception:
        return None, False


def _set_attr(path: tuple[str, ...], value):
    obj = torch.backends
    for name in path[:-1]:
        try:
            obj = getattr(obj, name)
        except Exception:
            return
        if obj is None:
            return
    attr = path[-1]
    try:
        setattr(obj, attr, value)
    except Exception:
        return


def test_true_fp32_correctness_context_forces_and_restores_tf32_state(monkeypatch):
    monkeypatch.delenv("KERNELGYM_CORRECTNESS_DISABLE_TF32", raising=False)
    paths = [
        ("cudnn", "allow_tf32"),
        ("cuda", "matmul", "allow_tf32"),
        ("fp32_precision",),
        ("cudnn", "fp32_precision"),
        ("cudnn", "conv", "fp32_precision"),
        ("cuda", "matmul", "fp32_precision"),
    ]
    original = {path: value for path in paths for value, exists in [_get_attr(path)] if exists}
    original_matmul_precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else None
    )

    try:
        # Put the process into a TF32-friendly state before entering the
        # correctness context; the context should force true fp32 temporarily.
        has_new_cudnn = ("cudnn", "conv", "fp32_precision") in original
        has_new_matmul = ("cuda", "matmul", "fp32_precision") in original
        if has_new_cudnn:
            _set_attr(("cudnn", "conv", "fp32_precision"), "tf32")
        elif ("cudnn", "allow_tf32") in original:
            _set_attr(("cudnn", "allow_tf32"), True)
        if has_new_matmul:
            _set_attr(("cuda", "matmul", "fp32_precision"), "tf32")
        elif ("cuda", "matmul", "allow_tf32") in original:
            _set_attr(("cuda", "matmul", "allow_tf32"), True)

        before_context = {path: _get_attr(path)[0] for path in original}
        metadata = {}
        with correctness._true_fp32_correctness_context(metadata):
            forced = metadata["correctness_tf32_state_forced"]
            if not has_new_cudnn and ("cudnn", "allow_tf32") in original:
                assert _get_attr(("cudnn", "allow_tf32"))[0] is False
            if not has_new_matmul and ("cuda", "matmul", "allow_tf32") in original:
                assert _get_attr(("cuda", "matmul", "allow_tf32"))[0] is False
            if ("cudnn", "conv", "fp32_precision") in original:
                assert _get_attr(("cudnn", "conv", "fp32_precision"))[0] == "ieee"
            if ("cuda", "matmul", "fp32_precision") in original:
                assert _get_attr(("cuda", "matmul", "fp32_precision"))[0] == "ieee"
            assert metadata["correctness_tf32_disabled"] is True
            assert forced

        after_context = {path: _get_attr(path)[0] for path in original}
        assert after_context == before_context
    finally:
        for path, value in original.items():
            _set_attr(path, value)
        if original_matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(original_matmul_precision)
