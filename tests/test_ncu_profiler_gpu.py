import os

import pytest

from kernelgym.config import settings
from kernelgym.toolkit.kernelbench.ncu_profiler import run_ncu_profile

REFERENCE_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def get_inputs():
    return [torch.randn(4096, device="cuda")]

def get_init_inputs():
    return []
"""

KERNEL_CODE = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return torch.sin(x)
"""


@pytest.mark.gpu
def test_real_ncu_collects_a_kernel_metric() -> None:
    if os.environ.get("RUN_NCU_INTEGRATION") != "1":
        pytest.skip("set RUN_NCU_INTEGRATION=1 to run the real Nsight Compute integration")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")

    result = run_ncu_profile(
        original_model_src=REFERENCE_CODE,
        custom_model_src=KERNEL_CODE,
        artifact=None,
        backend="cuda",
        entry_point="Model",
        device="cuda:0",
        kernel_names=[],
        ncu_path=settings.ncu_path,
        metrics=["gpu__time_duration.sum"],
        timeout_s=settings.ncu_timeout_s,
        max_kernels=2,
        warmup=1,
        profile_version=settings.ncu_profile_version,
    )

    if result["status"] == "permission_denied":
        pytest.skip("GPU performance counters are disabled for this process (ERR_NVGPUCTRPERM)")

    assert result["status"] == "ok", result
    assert result["profiled_kernel_count"] >= 1
    assert any("gpu__time_duration.sum" in kernel["metrics"] for kernel in result["kernels"])
