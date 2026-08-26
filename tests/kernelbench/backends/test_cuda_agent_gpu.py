"""KernelBench CUDA-Agent GPU integration tests."""

import os
import shutil
from pathlib import Path

import pytest

from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend
from kernelgym.toolkit.kernelbench.correctness import run_and_check_correctness


def _require_cuda_agent_toolchain() -> object:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.utils.cpp_extension")

    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is not available")

    nvcc = Path("/usr/local/cuda-12.9/bin/nvcc")
    if not nvcc.exists():
        pytest.skip("CUDA 12.9 nvcc is not available")

    if shutil.which(os.environ.get("CXX", "c++")) is None and shutil.which("g++") is None:
        pytest.skip("C++ compiler is not available")

    shm_root = Path("/dev/shm")
    if not shm_root.exists():
        pytest.skip("/dev/shm is not available")
    if shutil.disk_usage(shm_root).free < 1024 * 1024 * 1024:
        pytest.skip("/dev/shm does not have enough free space for CUDA-Agent compilation")
    if KernelBenchCudaAgentBackend._path_has_noexec_mount(shm_root):
        pytest.skip("/dev/shm is mounted noexec")

    return torch


@pytest.mark.gpu
def test_cuda_agent_compile_load_and_run_on_gpu(monkeypatch) -> None:
    torch = _require_cuda_agent_toolchain()
    monkeypatch.setenv("KERNELGYM_NVCC_THREADS", "1")

    backend = KernelBenchCudaAgentBackend()
    model_code = """
import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return cuda_extension.identity(x.float())
"""
    cuda_sources = {
        "kernels/generated.cu": """
#include <cuda_runtime.h>
#include <cstdint>

__global__ void identity_kernel(const float* input, float* output, int64_t size) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = input[index];
    }
}

extern "C" void launch_identity(const float* input, float* output, int64_t size) {
    constexpr int threads = 256;
    int blocks = static_cast<int>((size + threads - 1) / threads);
    identity_kernel<<<blocks, threads>>>(input, output, size);
}
""",
        "kernels/generated_binding.cpp": """
#include <torch/extension.h>
#include <cstdint>

extern "C" void launch_identity(const float* input, float* output, int64_t size);

torch::Tensor identity(torch::Tensor x) {
    auto output = torch::empty_like(x);
    launch_identity(x.data_ptr<float>(), output.data_ptr<float>(), x.numel());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("identity", &identity);
}
""",
    }

    handle = None
    artifact = None
    try:
        artifact = backend.compile(
            model_code,
            cuda_sources=cuda_sources,
            device="cuda:0",
            entry_point="ModelNew",
        )
        assert artifact["compiled"], artifact.get("error")
        assert Path(artifact["so_path"]).exists()
        assert artifact["profiling_hints"]["custom_kernel_names"] == ["identity_kernel"]

        handle = backend.load(artifact, device="cuda:0")
        x = torch.randn(8, device="cuda", dtype=torch.float16)
        output = backend.run(handle, {"init_inputs": [], "inputs": [x]}, device="cuda:0")["output"]

        assert torch.allclose(output, x.float())

        class Reference(torch.nn.Module):
            def forward(self, value):
                return value.float()

        result = run_and_check_correctness(
            Reference(),
            handle["model_cls"](),
            lambda: [torch.randn(128, device="cuda", dtype=torch.float16)],
            metadata={},
            num_correct_trials=1,
            seed=1234,
            device=torch.device("cuda:0"),
            detect_aten_fallback=True,
        )

        assert result.correctness is True
        assert result.decoy_kernel is False
        assert result.metadata["forbidden_aten_op_names"] == []
        allowed_names = {item["name"] for item in result.metadata["allowed_aten_ops"]}
        assert {"aten::to", "aten::_to_copy", "aten::empty_strided", "aten::copy_"} <= allowed_names
    finally:
        if handle is not None:
            backend.cleanup(handle)
        if artifact is not None and artifact.get("work_dir"):
            shutil.rmtree(artifact["work_dir"], ignore_errors=True)
