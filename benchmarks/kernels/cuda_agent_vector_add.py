"""CUDA-Agent (torch/extension.h + pybind11) vector-add fixture."""

from ._reference import REFERENCE_CODE as REFERENCE_CODE

BACKEND = "cuda_agent"

KERNEL_CODE = '''
### CUDA_KERNELS
```cpp
#include <torch/extension.h>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void launch_add_kernel(const float* a, const float* b, float* out, int n) {
    constexpr int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(a, b, out, n);
}
```

### APPLY_BINDINGS
```cpp
#include <torch/extension.h>

void launch_add_kernel(const float* a, const float* b, float* out, int n);

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "input shapes must match");
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty_like(a_c);
    launch_add_kernel(a_c.data_ptr<float>(),
                      b_c.data_ptr<float>(),
                      out.data_ptr<float>(),
                      static_cast<int>(a_c.numel()));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "elementwise add (CUDA)");
}
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    """CUDA-backed element-wise add."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return cuda_extension.add(a, b)
```
'''
