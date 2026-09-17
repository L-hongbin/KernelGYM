from __future__ import annotations

import pytest
import torch

from kernelgym.toolkit.kernelbench import pipeline
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult


def _correct_result() -> KernelExecResult:
    return KernelExecResult(compiled=True, correctness=True, metadata={})


def _incorrect_completed_result(metadata: dict[str, object]) -> KernelExecResult:
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def test_performance_probe_propagates_actionable_cuda_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError, match="illegal memory access"):
        pipeline._run_performance_step(
            kernel_exec_result=_correct_result(),
            custom_model=object(),
            get_inputs=lambda: [],
            metadata={},
            num_perf_trials=1,
            verbose=False,
            seed_num=42,
            device=0,
            enable_profiling=False,
            enable_triton_detection=False,
            detect_decoy_kernel=False,
            backend="tvm_ffi",
            backend_profiling_hints=None,
        )


def test_performance_probe_keeps_ordinary_diagnostic_failure_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("profiler unavailable")),
    )
    metadata: dict[str, object] = {}
    result = _correct_result()

    pipeline._run_performance_step(
        kernel_exec_result=result,
        custom_model=object(),
        get_inputs=lambda: [],
        metadata=metadata,
        num_perf_trials=1,
        verbose=False,
        seed_num=42,
        device=0,
        enable_profiling=False,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend="tvm_ffi",
        backend_profiling_hints=None,
    )

    assert str(result.metadata["error_during_performance"]) == "profiler unavailable"


def test_memory_probe_propagates_actionable_cuda_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("CUDA error: misaligned address")
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda **_kwargs: object())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError, match="misaligned address"):
        pipeline._run_memory_step(
            kernel_exec_result=_correct_result(),
            model=object(),
            get_inputs=lambda: [],
            source="",
            metadata={},
            allocator_check_metadata_key="allocator_check",
            seed_num=42,
            environment_floor={},
            device=0,
            verbose=False,
        )


def test_incorrect_output_probe_propagates_actionable_cuda_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda **_kwargs: (_ for _ in ()).throw(error))
    metadata: dict[str, object] = {
        "correctness_candidate_forward_completed": True,
        "correctness_output_mismatch": True,
    }

    with pytest.raises(RuntimeError, match="illegal memory access"):
        pipeline._run_incorrect_backend_usage_probe(
            kernel_exec_result=_incorrect_completed_result(metadata),
            custom_model=object(),
            get_inputs=lambda: [],
            metadata=metadata,
            seed_num=42,
            device=0,
            backend="tvm_ffi",
            backend_profiling_hints={"custom_kernel_names": ["candidate_kernel"]},
            detect_decoy_kernel=True,
        )
