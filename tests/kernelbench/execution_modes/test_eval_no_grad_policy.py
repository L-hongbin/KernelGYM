"""Regression coverage for the adopted eval plus no-grad execution policy."""

from __future__ import annotations

from contextlib import contextmanager
import json

import pytest


torch = pytest.importorskip("torch")


def test_execution_policy_recursively_sets_eval_and_records_metadata() -> None:
    from kernelgym.toolkit.kernelbench.execution_policy import (
        EXECUTION_POLICY_VERSION,
        prepare_model_for_execution,
        record_execution_policy,
    )

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5)).train()
    prepared = prepare_model_for_execution(model)
    metadata = {}
    record_execution_policy(metadata)

    assert prepared is model
    assert all(module.training is False for module in model.modules())
    assert metadata == {
        "execution_policy": EXECUTION_POLICY_VERSION,
        "model_mode": "eval",
        "grad_mode": "no_grad",
    }


def test_correctness_runs_reference_and_candidate_in_eval_no_grad(monkeypatch) -> None:
    from kernelgym.toolkit.kernelbench import correctness

    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, device=None: self)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    observed: dict[str, list[tuple[bool, bool, bool]]] = {"reference": [], "candidate": []}

    class ObservedModel(torch.nn.Module):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def forward(self, x):
            observed[self.label].append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            return x + 1

    result = correctness.run_and_check_correctness(
        ObservedModel("reference"),
        ObservedModel("candidate"),
        lambda: [torch.ones(4)],
        metadata={},
        num_correct_trials=2,
        device=None,
    )

    assert result.correctness is True
    assert observed == {
        "reference": [(False, False, False), (False, False, False)],
        "candidate": [(False, False, False), (False, False, False)],
    }
    assert result.metadata["model_mode"] == "eval"
    assert result.metadata["grad_mode"] == "no_grad"


def test_candidate_performance_prepares_model_in_eval(monkeypatch) -> None:
    from kernelgym.toolkit.kernelbench import pipeline
    from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    class TrainingModel(torch.nn.Module):
        def cuda(self, device=None):
            return self

    model = TrainingModel().train()
    observed_training_states = []

    def fake_timing(kernel_fn, *args, **kwargs):
        observed_training_states.append(kernel_fn.training)
        timing_info = {
            "warmup_wall_s": 0.0,
            "measure_wall_s": 0.0,
            "profiling_wall_s": 0.0,
            "timed_trials_cuda_event_s": 0.001,
            "num_warmup": 1,
            "num_trials": 1,
            "num_profiling_trials": 0,
            "total_wall_s": 0.0,
        }
        return [1.0], {}, timing_info

    monkeypatch.setattr(pipeline, "time_execution_with_cuda_event", fake_timing)
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    pipeline._run_performance_step(
        kernel_exec_result=result,
        custom_model=model,
        get_inputs=lambda: [],
        metadata={},
        num_perf_trials=1,
        num_warmup=1,
        perf_trim_count=0,
        verbose=False,
        seed_num=42,
        device=None,
        enable_profiling=False,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        backend="cuda_agent",
        backend_profiling_hints=None,
    )

    assert observed_training_states == [False]
    assert result.runtime == 1.0


def test_timing_window_preserves_eval_and_disables_grad(monkeypatch) -> None:
    from kernelgym.toolkit.kernelbench import timing
    from kernelgym.toolkit.kernelbench.execution_policy import prepare_model_for_execution

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "test-device")

    class FakeEvent:
        def record(self) -> None:
            return None

        def elapsed_time(self, other) -> float:
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: FakeEvent())

    observed = []

    class ObservedModel(torch.nn.Module):
        def forward(self, x):
            observed.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            return x + 1

    model = prepare_model_for_execution(ObservedModel().train())
    timing.time_execution_with_cuda_event(
        model,
        torch.ones(2),
        num_warmup=1,
        num_trials=2,
        verbose=False,
        device=0,
        enable_profiling=False,
    )

    assert observed == [(False, False, False), (False, False, False), (False, False, False)]


def test_reference_only_timing_prepares_model_in_eval(monkeypatch) -> None:
    from kernelgym.toolkit.kernelbench import pipeline

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "test-device")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, device=None: self)
    monkeypatch.setattr(pipeline, "graceful_eval_cleanup", lambda *args, **kwargs: None)

    observed = []

    def fake_timing(model, *args, **kwargs):
        observed.append(model.training)
        timing_info = {
            "warmup_wall_s": 0.0,
            "measure_wall_s": 0.0,
            "profiling_wall_s": 0.0,
            "timed_trials_cuda_event_s": 0.001,
            "num_warmup": 1,
            "num_trials": 1,
            "num_profiling_trials": 0,
            "total_wall_s": 0.0,
        }
        return [1.0], {}, timing_info

    monkeypatch.setattr(pipeline, "time_execution_with_cuda_event", fake_timing)
    result = pipeline.eval_reference_only(
        """
import torch

class Model(torch.nn.Module):
    def forward(self):
        return torch.ones(1)

def get_init_inputs():
    return []

def get_inputs():
    return []
""",
        num_perf_trials=1,
        num_warmup=1,
        device=0,
    )

    assert observed == [False]
    assert result.correctness is True
    assert result.runtime == 1.0
    assert result.metadata["model_mode"] == "eval"
    assert result.metadata["grad_mode"] == "no_grad"


def test_profiling_only_retry_disables_grad(monkeypatch) -> None:
    from kernelgym.toolkit.kernelbench import timing

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    @contextmanager
    def fake_profiling_context(enabled):
        yield object()

    monkeypatch.setattr(timing, "profiling_context", fake_profiling_context)
    monkeypatch.setattr(timing, "extract_profiling_metrics", lambda prof: {})

    observed_grad_states = []
    timing.run_profiling_only(
        lambda: observed_grad_states.append(torch.is_grad_enabled()),
        num_trials=3,
        verbose=False,
        device=0,
    )

    assert observed_grad_states == [False, False, False]


def test_triton_detection_primary_path_uses_eval_no_grad() -> None:
    from kernelgym.toolkit.kernelbench import triton_detect

    observed = []

    class ObservedModel(torch.nn.Module):
        def forward(self):
            observed.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))

    model = ObservedModel().train()
    triton_detect._call_no_grad(model.eval())

    assert observed == [(False, False, False)]


def test_triton_detection_has_one_eval_no_grad_execution_path(monkeypatch) -> None:
    import sys
    import types

    from kernelgym.toolkit.kernelbench import triton_detect

    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))

    class FakeHook:
        def __init__(self) -> None:
            self.captured = ["fake_kernel"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(triton_detect, "TritonKernelLaunchHook", FakeHook)
    observed = []

    class ObservedModel(torch.nn.Module):
        def forward(self):
            observed.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))

    used, matches = triton_detect.detect_triton_usage(
        ObservedModel().train(),
        warmup=1,
        steps=2,
        use_cuda=False,
        return_matches=True,
    )

    assert observed == [(False, False, False)] * 3
    assert used is True
    assert matches == ["fake_kernel"]


def test_reference_runtime_cache_rejects_legacy_execution_policy(tmp_path) -> None:
    from kernelgym.toolkit.kernelbench.execution_policy import EXECUTION_POLICY_VERSION
    from kernelgym.workflow.reference_cache import ReferenceRuntimeCache

    cache = ReferenceRuntimeCache()
    cache.put("problem-1", "reference", False, 1.25)

    assert EXECUTION_POLICY_VERSION in cache._entry_key("problem-1", False)
    assert cache.get("problem-1", "reference", False) == 1.25

    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text(
        '{"uuid":"legacy","reference_runtime":2.5,"reference_code":"reference"}\n',
        encoding="utf-8",
    )
    assert cache.preload(str(legacy_path), is_valid=False) == 0
    assert cache.get("legacy", "reference", False) is None


def test_reference_runtime_cache_preloads_current_nested_policy(tmp_path) -> None:
    from kernelgym.toolkit.kernelbench.execution_policy import EXECUTION_POLICY_VERSION
    from kernelgym.workflow.reference_cache import ReferenceRuntimeCache

    current_path = tmp_path / "current.jsonl"
    current_path.write_text(
        json.dumps(
            {
                "result": {
                    "runtime": 2.5,
                    "reference_code": "reference",
                    "metadata": {
                        "uuid": "current",
                        "execution_policy": EXECUTION_POLICY_VERSION,
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cache = ReferenceRuntimeCache()
    assert cache.preload(str(current_path), is_valid=True) == 1
    assert cache.get("current", "reference", True) == 2.5
    assert cache.describe()["preloaded_entries"] == 1


def test_request_hash_is_fenced_by_execution_policy(monkeypatch) -> None:
    from kernelgym.server import request_hash as request_hash_module

    payload = {"reference_code": "reference", "kernel_code": "candidate"}
    current_hash = request_hash_module.request_hash("kernelbench", payload)
    kernel_simple_hash = request_hash_module.request_hash("kernel_simple", payload)
    monkeypatch.setattr(request_hash_module, "EXECUTION_POLICY_VERSION", "next-policy")

    assert request_hash_module.request_hash("kernelbench", payload) != current_hash
    assert request_hash_module.request_hash("kernel_simple", payload) == kernel_simple_hash
