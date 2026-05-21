"""TVM-FFI vector-add fixture (matches the binding pattern used in production).

Binding rules enforced by ``precheck_tvm_ffi_submission``:
- binding .cpp must NOT include <cuda_runtime.h> or use cudaStream_t
- binding .cpp must include a tvm/ffi/* header
- functions exported with TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, fn)
- model_new must import tvm_ffi_extension and call one of the exported names
"""

from ._reference import REFERENCE_CODE as REFERENCE_CODE

BACKEND = "tvm_ffi"

KERNEL_CODE = '''
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" void add_launcher(const float* a, const float* b, float* out,
                             int n, void* stream_handle) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    constexpr int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void add_launcher(const float* a, const float* b, float* out,
                             int n, void* stream_handle);

void add(tvm::ffi::Tensor a, tvm::ffi::Tensor b, tvm::ffi::Tensor out) {
    TVM_FFI_ICHECK(a.device().device_type == kDLCUDA) << "a must be CUDA";
    TVM_FFI_ICHECK(b.device().device_type == kDLCUDA) << "b must be CUDA";
    TVM_FFI_ICHECK(out.device().device_type == kDLCUDA) << "out must be CUDA";

    DLDataType f32_dtype{kDLFloat, 32, 1};
    TVM_FFI_ICHECK(a.dtype() == f32_dtype) << "a must be float32";
    TVM_FFI_ICHECK(b.dtype() == f32_dtype) << "b must be float32";
    TVM_FFI_ICHECK(out.dtype() == f32_dtype) << "out must be float32";

    TVM_FFI_ICHECK(a.IsContiguous()) << "a must be contiguous";
    TVM_FFI_ICHECK(b.IsContiguous()) << "b must be contiguous";
    TVM_FFI_ICHECK(out.IsContiguous()) << "out must be contiguous";

    TVM_FFI_ICHECK(a.numel() == b.numel() && a.numel() == out.numel())
        << "shape mismatch";

    void* stream_handle =
        TVMFFIEnvGetStream(a.device().device_type, a.device().device_id);

    add_launcher(static_cast<const float*>(a.data_ptr()),
                 static_cast<const float*>(b.data_ptr()),
                 static_cast<float*>(out.data_ptr()),
                 static_cast<int>(a.numel()),
                 stream_handle);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, add);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import tvm_ffi_extension


class ModelNew(nn.Module):
    """TVM-FFI-backed element-wise add."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        out = torch.empty_like(a)
        tvm_ffi_extension.add(a, b, out)
        return out
```
'''
