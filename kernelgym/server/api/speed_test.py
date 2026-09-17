"""Fixed end-to-end speed-test case for the public API."""

from __future__ import annotations

from typing import Any, Dict


CASE_NAME = "gemm_rmsnorm_fp32"
BACKEND = "tvm_ffi"
REPEAT_COUNT = 3
INPUT_SHAPES = {"lhs": [4096, 16], "rhs": [16, 512]}


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
    # while both the reference matmul and custom Tensor Core kernel use TF32.
    # Values do not affect the measured kernel path.
    return [torch.randn(4096, 16) * 0.001, torch.randn(16, 512) * 0.001]
"""


KERNEL_CODE = r"""
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>
#include <mma.h>

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
constexpr int OUTPUT_COLS = 512;
constexpr int WARPS_PER_BLOCK = OUTPUT_COLS / WMMA_N;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void gemm_rmsnorm_kernel(
    const float* lhs, const float* rhs, float* output,
    int m, int n, int k, float eps) {
    using namespace nvcuda;
    __shared__ __align__(128) float block_output[WMMA_M][OUTPUT_COLS];
    __shared__ float inv_rms[WMMA_M];

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * WMMA_M;
    const int col = warp_id * WMMA_N;
    if (row + WMMA_M > m || n != OUTPUT_COLS || (k % WMMA_K) != 0) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> lhs_fragment;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   wmma::precision::tf32, wmma::row_major> rhs_fragment;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (int inner = 0; inner < k; inner += WMMA_K) {
        wmma::load_matrix_sync(lhs_fragment, lhs + row * k + inner, k);
        wmma::load_matrix_sync(rhs_fragment, rhs + inner * n + col, n);
        wmma::mma_sync(accumulator, lhs_fragment, rhs_fragment, accumulator);
    }
    wmma::store_matrix_sync(&block_output[0][col], accumulator, OUTPUT_COLS, wmma::mem_row_major);
    __syncthreads();

    for (int local_row = warp_id; local_row < WMMA_M; local_row += WARPS_PER_BLOCK) {
        float sum_sq = 0.0f;
#pragma unroll
        for (int output_col = lane; output_col < OUTPUT_COLS; output_col += 32) {
            const float value = block_output[local_row][output_col];
            sum_sq = fmaf(value, value, sum_sq);
        }
        sum_sq = warp_sum(sum_sq);
        if (lane == 0) {
            inv_rms[local_row] = rsqrtf(sum_sq / static_cast<float>(OUTPUT_COLS) + eps);
        }
    }
    __syncthreads();

    for (int index = threadIdx.x; index < WMMA_M * OUTPUT_COLS; index += blockDim.x) {
        const int local_row = index / OUTPUT_COLS;
        const int output_col = index % OUTPUT_COLS;
        output[(row + local_row) * n + output_col] =
            block_output[local_row][output_col] * inv_rms[local_row];
    }
}

extern "C" void gemm_rmsnorm_launcher(
    const float* lhs, const float* rhs, float* output,
    int m, int n, int k, float eps, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    gemm_rmsnorm_kernel<<<m / WMMA_M, WARPS_PER_BLOCK * 32, 0, stream>>>(
        lhs, rhs, output, m, n, k, eps);
}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void gemm_rmsnorm_launcher(
    const float* lhs, const float* rhs, float* output,
    int m, int n, int k, float eps, void* stream_handle);

void gemm_rmsnorm_forward(
    tvm::ffi::Tensor lhs, tvm::ffi::Tensor rhs, tvm::ffi::Tensor output, double eps) {
    auto lhs_shape = lhs.shape();
    auto rhs_shape = rhs.shape();
    void* stream = TVMFFIEnvGetStream(lhs.device().device_type, lhs.device().device_id);
    gemm_rmsnorm_launcher(
        static_cast<const float*>(lhs.data_ptr()),
        static_cast<const float*>(rhs.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int>(lhs_shape[0]),
        static_cast<int>(rhs_shape[1]),
        static_cast<int>(lhs_shape[1]),
        static_cast<float>(eps),
        stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_rmsnorm_forward, gemm_rmsnorm_forward);
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
        output = torch.empty(
            (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=lhs.dtype
        )
        tvm_ffi_extension.gemm_rmsnorm_forward(lhs, rhs, output, float(self.eps))
        return output
```
"""


def build_payload(task_id: str, run_token: str) -> Dict[str, Any]:
    """Build one cold, uncached evaluation request for the fixed case."""
    kernel_code = KERNEL_CODE.replace(
        "__global__ void gemm_rmsnorm_kernel",
        f"// speed-test-run: {run_token}\n__global__ void gemm_rmsnorm_kernel",
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
        "num_perf_trials": 300,
        "num_warmup": 3,
        "perf_trim_count": 0,
        "adaptive_perf_trials": False,
        "timeout": 300,
        "priority": "normal",
        "entry_point": "Model",
        "force_refresh": True,
        "use_reference_cache": False,
        "enable_compile_artifact_cache": False,
        "enable_ncu": True,
        "enable_compute_sanitizer": False,
        "return_detail_correctness": False,
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
    "ncu_profile_s": "kg_kernel_ncu_profile_s",
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
