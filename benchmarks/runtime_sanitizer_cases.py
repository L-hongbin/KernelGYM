"""TVM-FFI fixtures that exercise NVIDIA Compute Sanitizer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

REFERENCE_TEMPLATE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return x.clone()


def get_init_inputs():
    return [{mode}]


def get_inputs():
    return [torch.randn(1000, dtype=torch.float32)]
"""


KERNEL_CODE = r"""
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void sanitizer_safe_kernel(const float* input, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = input[index];
    }
}

__global__ void sanitizer_oob_kernel(const float* input, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = index < n ? input[index] : 0.0f;
}

__global__ void sanitizer_race_kernel(const float* input, float* output, int n) {
    __shared__ volatile float scratch;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    scratch = index < n ? input[index] : 0.0f;
    __syncthreads();
    if (index < n) {
        output[index] = input[index];
    }
}

__global__ void sanitizer_sync_kernel(const float* input, float* output, int n) {
    __shared__ float scratch[16];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    if (lane < 16) {
        scratch[lane] = input[index];
    }
    unsigned int active = __ballot_sync(0xffffffff, lane < 16);
    if (lane <= 16) {
        __syncwarp(active);
    }
    if (lane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) sum += scratch[i];
        output[index] = sum;
    } else if (index < n) {
        output[index] = input[index];
    }
}

__global__ void sanitizer_init_kernel(
    const float* input, const float* uninitialized, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = input[index] + uninitialized[index];
    }
}

extern "C" void sanitizer_launcher(
    const float* input, float* output, int n, int mode, void* stream_handle) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    int grid = (n + 255) / 256;
    if (mode == 0) {
        sanitizer_safe_kernel<<<grid, 256, 0, stream>>>(input, output, n);
    } else if (mode == 1) {
        float* exact_allocation = nullptr;
        cudaMalloc(&exact_allocation, static_cast<size_t>(n) * sizeof(float));
        sanitizer_oob_kernel<<<grid, 256, 0, stream>>>(input, exact_allocation, n);
        cudaMemcpyAsync(
            output, exact_allocation, static_cast<size_t>(n) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
        cudaFree(exact_allocation);
    } else if (mode == 2) {
        sanitizer_race_kernel<<<grid, 256, 0, stream>>>(input, output, n);
    } else if (mode == 3) {
        sanitizer_sync_kernel<<<grid, 256, 0, stream>>>(input, output, n);
    } else {
        float* uninitialized = nullptr;
        cudaMalloc(&uninitialized, static_cast<size_t>(n) * sizeof(float));
        sanitizer_init_kernel<<<grid, 256, 0, stream>>>(input, uninitialized, output, n);
        cudaFree(uninitialized);
    }
}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void sanitizer_launcher(
    const float* input, float* output, int n, int mode, void* stream_handle);

void sanitizer_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output, int64_t mode) {
    void* stream =
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    sanitizer_launcher(
        static_cast<const float*>(input.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int>(input.numel()),
        static_cast<int>(mode),
        stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sanitizer_forward, sanitizer_forward);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import tvm_ffi_extension


class ModelNew(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        output = torch.empty_like(x)
        tvm_ffi_extension.sanitizer_forward(x, output, int(self.mode))
        return output
```
"""


@dataclass(frozen=True)
class SanitizerCase:
    name: str
    mode: int
    tool: str
    kernel_name: str
    expected_status: str
    expected_hazard_fragment: str | None = None

    @property
    def reference_code(self) -> str:
        return REFERENCE_TEMPLATE.format(mode=self.mode)


CASES = (
    SanitizerCase("safe", 0, "memcheck", "sanitizer_safe_kernel", "clean"),
    SanitizerCase(
        "global_oob",
        1,
        "memcheck",
        "sanitizer_oob_kernel",
        "issues_found",
        "invalid_global_write",
    ),
    SanitizerCase("shared_race", 2, "racecheck", "sanitizer_race_kernel", "issues_found", "race"),
    SanitizerCase(
        "divergent_sync",
        3,
        "synccheck",
        "sanitizer_sync_kernel",
        "issues_found",
        "synchronization",
    ),
    SanitizerCase(
        "uninitialized_read",
        4,
        "initcheck",
        "sanitizer_init_kernel",
        "issues_found",
        "uninitialized",
    ),
)
