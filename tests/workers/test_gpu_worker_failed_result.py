"""Bug #3 regression: a failure that happens AFTER compile+load (timeout or
crash during execution) must report ``compiled=True``, so it is no longer
indistinguishable from a genuine compilation error.

Covers ``GPUWorker._compiled_from_stage_metadata`` and ``_build_failed_result``.
"""

import importlib.util
import json
from pathlib import Path

import pytest

# gpu_worker imports torch/redis/aiohttp at module scope; skip where unavailable.
pytest.importorskip("torch")
pytest.importorskip("redis")
pytest.importorskip("aiohttp")

ROOT = Path(__file__).resolve().parents[2]


def _load_gpu_worker():
    spec = importlib.util.spec_from_file_location(
        "gpu_worker_under_test", ROOT / "kernelgym" / "worker" / "gpu_worker.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_worker = _load_gpu_worker()
GPUWorker = gpu_worker.GPUWorker

from kernelgym.common import ErrorCode  # noqa: E402


def test_compiled_from_stage_metadata_inference() -> None:
    infer = GPUWorker._compiled_from_stage_metadata
    # No / unusable metadata -> safe default of "not compiled".
    assert infer({}) is False
    assert infer({"kg_stage_metadata_path": "/dev/shm/x.json"}) is False
    assert infer({"kg_stage_completed_s": {"kernel.load_original_src": 0.01}}) is False
    # Compile still in progress -> not yet compiled.
    assert infer({"kg_stage_current": "kernel.compile_and_load"}) is False
    # Compile/load finished -> compiled.
    assert infer({"kg_stage_completed_s": {"kernel.compile_and_load": 0.1}}) is True
    assert infer({"kg_stage_completed_s": {"kernel.compile_only": 0.1}}) is True
    # Current/last stage only reachable after compile -> compiled.
    assert infer({"kg_stage_current": "kernel.correctness.custom_forward"}) is True
    assert infer({"kg_stage_last_completed": "kernel.build_custom_model"}) is True
    assert infer({"kg_stage_current": "kernel.performance"}) is True


def _failed_result(tmp_path, stage_payload, task_type):
    stage_file = tmp_path / "stage.json"
    stage_file.write_text(json.dumps(stage_payload), encoding="utf-8")
    worker = GPUWorker.__new__(GPUWorker)
    task_data = {
        "task_id": "k1",
        "task_type": task_type,
        "_stage_metadata_path": str(stage_file),
    }
    return worker._build_failed_result(task_data, "Task timeout after 90s", ErrorCode.TIMEOUT_ERROR.value)


def test_kernel_evaluation_timeout_after_compile_is_compiled(tmp_path) -> None:
    result = _failed_result(
        tmp_path,
        {
            "kg_stage_completed_s": {"kernel.compile_and_load": 0.2},
            "kg_stage_current": "kernel.correctness.custom_forward",
        },
        "kernel_evaluation",
    )
    assert result["compiled"] is True


def test_kernel_evaluation_timeout_during_compile_is_not_compiled(tmp_path) -> None:
    result = _failed_result(
        tmp_path,
        {"kg_stage_current": "kernel.compile_and_load"},
        "kernel_evaluation",
    )
    assert result["compiled"] is False


def test_evaluation_timeout_after_compile_is_compiled(tmp_path) -> None:
    result = _failed_result(
        tmp_path,
        {"kg_stage_completed_s": {"kernel.compile_and_load": 0.2}},
        "evaluation",
    )
    assert result["compiled"] is True


def test_missing_stage_file_defaults_to_not_compiled() -> None:
    worker = GPUWorker.__new__(GPUWorker)
    task_data = {"task_id": "k2", "task_type": "kernel_evaluation"}  # no stage metadata path
    result = worker._build_failed_result(task_data, "boom", ErrorCode.RUNTIME_ERROR.value)
    assert result["compiled"] is False
