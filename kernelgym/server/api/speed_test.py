"""Fixed end-to-end speed-test case for the public API."""

from __future__ import annotations

from typing import Any, Dict


CASE_NAME = "gemm_rmsnorm_fp32"
BACKEND = "tvm_ffi"
REPEAT_COUNT = 3
INPUT_SHAPES = {"lhs": [512, 512], "rhs": [512, 512]}


REFERENCE_CODE = r"""
import torch


class Model(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, lhs, rhs):
        value = torch.matmul(lhs, rhs)
        return value * torch.rsqrt(torch.mean(value * value, dim=-1, keepdim=True) + self.eps)


def get_init_inputs():
    return [1e-5]


def get_inputs():
    # Keep this fixed health-check case comfortably inside the fp32 tolerance
    # even when the reference matmul uses TF32 while the custom kernel performs
    # scalar fp32 accumulation. Values do not affect the measured kernel path.
    return [torch.randn(512, 512) * 0.001, torch.randn(512, 512) * 0.001]
"""


KERNEL_CODE = r"""
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void gemm_kernel(
    const float* lhs, const float* rhs, float* output, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        sum += lhs[row * k + inner] * rhs[inner * n + col];
    }
    output[row * n + col] = sum;
}

__global__ void rms_norm_kernel(
    const float* input, float* output, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float scratch[256];
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        sum_sq += value * value;
    }
    scratch[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(scratch[0] / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    float inv_rms = scratch[0];
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row * cols + col] = input[row * cols + col] * inv_rms;
    }
}

extern "C" void gemm_launcher(
    const float* lhs, const float* rhs, float* output,
    int m, int n, int k, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    gemm_kernel<<<grid, block, 0, stream>>>(lhs, rhs, output, m, n, k);
}

extern "C" void rms_norm_launcher(
    const float* input, float* output, int rows, int cols,
    float eps, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    rms_norm_kernel<<<rows, 256, 0, stream>>>(input, output, rows, cols, eps);
}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void gemm_launcher(
    const float* lhs, const float* rhs, float* output,
    int m, int n, int k, void* stream_handle);
extern "C" void rms_norm_launcher(
    const float* input, float* output, int rows, int cols,
    float eps, void* stream_handle);

void gemm_forward(
    tvm::ffi::Tensor lhs, tvm::ffi::Tensor rhs, tvm::ffi::Tensor output) {
    auto lhs_shape = lhs.shape();
    auto rhs_shape = rhs.shape();
    void* stream = TVMFFIEnvGetStream(lhs.device().device_type, lhs.device().device_id);
    gemm_launcher(
        static_cast<const float*>(lhs.data_ptr()),
        static_cast<const float*>(rhs.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int>(lhs_shape[0]),
        static_cast<int>(rhs_shape[1]),
        static_cast<int>(lhs_shape[1]),
        stream);
}

void rms_norm_forward(
    tvm::ffi::Tensor input, tvm::ffi::Tensor output, double eps) {
    auto shape = input.shape();
    int cols = static_cast<int>(shape[shape.size() - 1]);
    int rows = static_cast<int>(input.numel() / cols);
    void* stream = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    rms_norm_launcher(
        static_cast<const float*>(input.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        rows,
        cols,
        static_cast<float>(eps),
        stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_forward, gemm_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rms_norm_forward, rms_norm_forward);
```

### MODEL_NEW
```python
import torch
import tvm_ffi_extension


class ModelNew(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, lhs, rhs):
        intermediate = torch.empty(
            (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=lhs.dtype
        )
        output = torch.empty_like(intermediate)
        tvm_ffi_extension.gemm_forward(lhs, rhs, intermediate)
        tvm_ffi_extension.rms_norm_forward(intermediate, output, float(self.eps))
        return output
```
"""


def build_payload(task_id: str, run_token: str) -> Dict[str, Any]:
    """Build one cold, uncached evaluation request for the fixed case."""
    kernel_code = KERNEL_CODE.replace(
        "__global__ void gemm_kernel",
        f"// speed-test-run: {run_token}\n__global__ void gemm_kernel",
        1,
    )
    return {
        "task_id": task_id,
        "reference_code": REFERENCE_CODE,
        "kernel_code": kernel_code,
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": BACKEND,
        "precision": "fp32",
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "num_warmup": 3,
        "perf_trim_count": 0,
        "adaptive_perf_trials": False,
        "timeout": 300,
        "priority": "normal",
        "entry_point": "Model",
        "force_refresh": True,
        "use_reference_cache": False,
        "enable_compile_artifact_cache": False,
        "enable_ncu": False,
        "enable_compute_sanitizer": False,
        "enable_correctness_input_perturbations": False,
        "run_correctness": True,
        "run_performance": True,
        "workflow": "kernelbench",
    }


STAGE_TIMING_FIELDS = {
    "reference_total_s": "kg_reference_total_s",
    "kernel_total_s": "kg_kernel_total_s",
    "kernel_compile_s": "kg_kernel_backend_compile_s",
    "kernel_load_s": "kg_kernel_backend_load_s",
    "kernel_correctness_s": "kg_kernel_correctness_s",
    "kernel_performance_s": "kg_kernel_performance_step_s",
    "worker_pool_s": "wg_pool_total_s",
}


def extract_stage_timings(metadata: Any, compile_metadata: Any = None) -> Dict[str, float]:
    """Expose stable, human-readable timing names from workflow metadata."""
    if not isinstance(metadata, dict):
        return {}
    timings: Dict[str, float] = {}
    for public_name, metadata_name in STAGE_TIMING_FIELDS.items():
        value = metadata.get(metadata_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timings[public_name] = round(float(value), 6)
    if isinstance(compile_metadata, dict):
        compile_s = compile_metadata.get("kg_kernel_backend_compile_s")
        if isinstance(compile_s, (int, float)) and not isinstance(compile_s, bool):
            timings["kernel_compile_s"] = round(float(compile_s), 6)
        compile_worker_s = compile_metadata.get("cpu_worker_run_s")
        if isinstance(compile_worker_s, (int, float)) and not isinstance(compile_worker_s, bool):
            timings["compile_worker_total_s"] = round(float(compile_worker_s), 6)
    return timings
