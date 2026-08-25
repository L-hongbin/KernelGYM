"""Triton and TileLang backend contract tests."""

import pytest

from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
from kernelgym.backend.kernelbench.python_dsl_backend import (
    KernelBenchTileLangBackend,
    KernelBenchTritonBackend,
)
from kernelgym.common import Backend
from kernelgym.schema.task import EvaluationTask
from kernelgym.server.api.models import EvaluationRequest
from kernelgym.toolkit.kernelbench.binding_detection import resolve_kernel_backend
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks


TRITON_MODEL = """
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def add_one_kernel(x, y, n: tl.constexpr):
    pass

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""


TILELANG_MODEL = """
import torch.nn as nn
import tilelang
import tilelang.language as T

@tilelang.jit()
def make_kernel(n):
    @T.prim_func
    def add_one_kernel():
        pass
    return add_one_kernel

class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""


def test_dispatcher_routes_triton_and_tilelang_without_cuda_fallback() -> None:
    backend = KernelBenchBackend()

    assert isinstance(backend._select("triton"), KernelBenchTritonBackend)
    assert isinstance(backend._select("tilelang"), KernelBenchTileLangBackend)
    with pytest.raises(ValueError, match="Unsupported KernelBench backend"):
        backend._select("unknown-dsl")


@pytest.mark.parametrize(
    ("backend", "code", "kernel_name"),
    [
        (KernelBenchTritonBackend(), TRITON_MODEL, "add_one_kernel"),
        (KernelBenchTileLangBackend(), TILELANG_MODEL, "add_one_kernel"),
    ],
)
def test_python_dsl_compile_returns_reusable_source_artifact(backend, code, kernel_name) -> None:
    artifact = backend.compile(
        code,
        device="cuda:0",
        entry_point="ModelNew",
        compiler_options={"threads": 128},
        enable_compile_artifact_cache=True,
    )

    assert artifact["compiled"] is True
    assert artifact["artifact_type"] == "python_jit_source"
    assert artifact["jit_compile_on_execute"] is True
    assert artifact["compiler_options"] == {"threads": 128}
    assert kernel_name in artifact["profiling_hints"]["custom_kernel_names"]
    assert artifact["compile_artifact_cache_enabled"] is True


@pytest.mark.parametrize("backend", [KernelBenchTritonBackend(), KernelBenchTileLangBackend()])
def test_python_dsl_compile_rejects_torch_only_submission(backend) -> None:
    artifact = backend.compile(
        "import torch\nclass ModelNew:\n    pass\n",
        entry_point="ModelNew",
    )

    assert artifact["compiled"] is False
    assert "must import its runtime" in artifact["error"]


def test_auto_backend_detection_understands_python_dsls() -> None:
    assert resolve_kernel_backend(TRITON_MODEL, Backend.AUTO) == "triton"
    assert resolve_kernel_backend(TILELANG_MODEL, Backend.AUTO) == "tilelang"


def test_compiler_options_survive_api_and_paired_task_conversion() -> None:
    request = EvaluationRequest(
        task_id="dsl",
        reference_code="class Model: pass",
        kernel_code=TRITON_MODEL,
        backend=Backend.TRITON,
        compiler_options={"num_warps": 4},
    )
    task = EvaluationTask.from_dict(request.dict())
    _, kernel_task = _create_paired_tasks(task)

    assert kernel_task.compiler_options == {"num_warps": 4}
