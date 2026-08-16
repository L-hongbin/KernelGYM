from __future__ import annotations

from typing import Any

import pytest
import torch

from kernelgym.backend.kernelbench import cuda_agent_backend, tvm_ffi_backend
from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend
from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend
from kernelgym.schema.task import EvaluationTask
from kernelgym.server.api.models import EvaluationRequest
from kernelgym.server.request_hash import request_hash
from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref
from kernelgym.toolkit.kernelbench.static_checker import validate_kernel_static
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks


REFERENCE_CODE = "class Model:\n    pass"
KERNEL_CODE = "class ModelNew:\n    pass"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "fp32"),
        ("float32", "fp32"),
        ("torch.float32", "fp32"),
        ("half", "fp16"),
        ("torch.float16", "fp16"),
        ("bfloat16", "bf16"),
        ("torch.bfloat16", "bf16"),
    ],
)
def test_evaluation_request_normalizes_precision(value: str | None, expected: str) -> None:
    request = EvaluationRequest(
        task_id="precision",
        reference_code=REFERENCE_CODE,
        kernel_code=KERNEL_CODE,
        precision=value,
    )

    assert request.precision == expected


def test_evaluation_request_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        EvaluationRequest(
            task_id="precision",
            reference_code=REFERENCE_CODE,
            kernel_code=KERNEL_CODE,
            precision="int8",
        )


def test_precision_policy_allows_internal_fp16_for_fp16_and_bf16_only() -> None:
    code = "output = output.to(torch.float16)"

    fp32 = validate_kernel_static(code, precision="fp32")
    fp16 = validate_kernel_static(code, precision="fp16")
    bf16 = validate_kernel_static(code, precision="bf16")
    unknown = validate_kernel_static(code, precision="unexpected")

    assert fp32.valid is False
    assert fp16.valid is True
    assert bf16.valid is True
    assert unknown.valid is False
    assert fp32.precision == "fp32"
    assert bf16.precision == "bf16"
    assert unknown.precision == "fp32"


def test_precision_survives_api_task_and_paired_kernel_task() -> None:
    request = EvaluationRequest(
        task_id="precision",
        reference_code=REFERENCE_CODE,
        kernel_code=KERNEL_CODE,
        precision="torch.bfloat16",
    )
    task = EvaluationTask.from_dict(request.dict())
    _reference_task, kernel_task = _create_paired_tasks(task)

    assert task.precision == "bf16"
    assert kernel_task.precision == "bf16"
    assert kernel_task.to_dict()["precision"] == "bf16"


def test_request_hash_distinguishes_precision() -> None:
    base = {
        "reference_code": REFERENCE_CODE,
        "kernel_code": KERNEL_CODE,
        "precision": "fp32",
    }

    assert request_hash("kernelbench", base) != request_hash(
        "kernelbench",
        {**base, "precision": "fp16"},
    )


@pytest.mark.parametrize(
    ("backend_module", "backend_class", "extension_name", "precheck_name"),
    [
        (
            cuda_agent_backend,
            KernelBenchCudaAgentBackend,
            "cuda_extension",
            "precheck_cuda_agent_submission",
        ),
        (
            tvm_ffi_backend,
            KernelBenchTvmFfiBackend,
            "tvm_ffi_extension",
            "precheck_tvm_ffi_submission",
        ),
    ],
)
def test_backend_compile_forwards_precision_to_precheck(
    monkeypatch,
    backend_module: Any,
    backend_class: type,
    extension_name: str,
    precheck_name: str,
) -> None:
    observed: dict[str, Any] = {}

    def fake_precheck(model_code, cuda_sources, *, entry_point, precision):
        observed.update(
            model_code=model_code,
            cuda_sources=cuda_sources,
            entry_point=entry_point,
            precision=precision,
        )
        return "intentional stop", None, {"passed": False}

    monkeypatch.setattr(backend_module, precheck_name, fake_precheck)
    backend = backend_class()
    model_code = f"""
import torch
import {extension_name}

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return {extension_name}.identity(x)
"""
    result = backend.compile(
        model_code,
        device="cpu",
        precision="bf16",
        cuda_sources={"kernels/generated.cu": "__global__ void identity_kernel(float* x) {}"},
    )

    assert result["compiled"] is False
    assert observed["precision"] == "bf16"
    assert observed["entry_point"] == "ModelNew"


def test_pipeline_forwards_precision_to_backend_compile() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def compile(self, _code: str, **kwargs: Any) -> dict[str, Any]:
            self.kwargs = kwargs
            return {"compiled": True, "backend": "cuda_agent"}

    backend = RecordingBackend()
    result = eval_kernel_against_ref(
        original_model_src=REFERENCE_CODE,
        custom_model_src=KERNEL_CODE,
        device=torch.device("cpu"),
        backend="cuda_agent",
        precision="bf16",
        backend_adapter=backend,
        compile_only=True,
    )

    assert result.compiled is True
    assert result.metadata["precision"] == "bf16"
    assert backend.kwargs["precision"] == "bf16"
