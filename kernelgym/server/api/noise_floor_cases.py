"""Fixed CUDA TVM-FFI cases for speedup noise-floor calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .speed_test import INPUT_SHAPES as GEMM_RMSNORM_INPUT_SHAPES
from .speed_test import KERNEL_CODE as GEMM_RMSNORM_KERNEL_CODE
from .speed_test import REFERENCE_CODE as GEMM_RMSNORM_REFERENCE_CODE


SOURCE_DATASET = "/nfs/FM/lihongbin/datasets/KernelData/KernelBench/data/level_1-00000-of-00001.parquet"


@dataclass(frozen=True)
class NoiseFloorCase:
    case_id: str
    name: str
    runtime_class: str
    reference_code: str
    kernel_code: str
    input_shapes: Dict[str, List[int]]
    dataset_problem_id: Optional[int] = None
    dataset_name: Optional[str] = None
    shape_policy: str = "fixed_calibration"

    def public_metadata(self) -> dict:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "runtime_class": self.runtime_class,
            "backend": "tvm_ffi",
            "precision": "fp32",
            "input_shapes": self.input_shapes,
            "dataset_path": SOURCE_DATASET if self.dataset_problem_id is not None else None,
            "dataset_problem_id": self.dataset_problem_id,
            "dataset_name": self.dataset_name,
            "shape_policy": self.shape_policy,
        }


def _package(cuda_source: str, binding_source: str, model_source: str) -> str:
    return (
        "### CUDA_KERNELS\n```cpp\n"
        + cuda_source.strip()
        + "\n```\n\n### APPLY_BINDINGS\n```cpp\n"
        + binding_source.strip()
        + "\n```\n\n### MODEL_NEW\n```python\n"
        + model_source.strip()
        + "\n```\n"
    )


_UNARY_BINDING = r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void unary_launcher(
    const float* input, float* output, int64_t count, void* stream_handle);

void unary_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output) {
    void* stream = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    unary_launcher(
        static_cast<const float*>(input.data_ptr()),
        static_cast<float*>(output.data_ptr()), input.numel(), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(unary_forward, unary_forward);
"""


def _unary_kernel(expression: str) -> str:
    return _package(
        f"""
#include <cuda_runtime.h>
#include <math.h>

__global__ void unary_kernel(const float* input, float* output, int64_t count) {{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {{
        float value = input[index];
        output[index] = {expression};
    }}
}}

extern "C" void unary_launcher(
    const float* input, float* output, int64_t count, void* stream_handle) {{
    auto stream = static_cast<cudaStream_t>(stream_handle);
    unary_kernel<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream>>>(input, output, count);
}}
""",
        _UNARY_BINDING,
        r"""
import torch
import tvm_ffi_extension

class ModelNew(torch.nn.Module):
    def forward(self, x):
        output = torch.empty_like(x)
        tvm_ffi_extension.unary_forward(x, output)
        return output
""",
    )


RELU_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)
def get_inputs():
    return [torch.rand(4096, 256) * 2 - 1]
def get_init_inputs():
    return []
"""
RELU_KERNEL = _unary_kernel("fmaxf(value, 0.0f)")


GELU_REFERENCE = r"""
import torch
import math
class Model(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
def get_inputs():
    return [torch.rand(8192, 8192) * 2 - 1]
def get_init_inputs():
    return []
"""
GELU_KERNEL = _unary_kernel(
    "0.5f * value * (1.0f + tanhf(0.7978845608028654f * "
    "(value + 0.044715f * value * value * value)))"
)


DIAGONAL_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def forward(self, diagonal, matrix):
        return torch.diag(diagonal) @ matrix
def get_inputs():
    return [torch.rand(4096), torch.rand(4096, 4096)]
def get_init_inputs():
    return []
"""
DIAGONAL_KERNEL = _package(
    r"""
#include <cuda_runtime.h>
__global__ void diagonal_kernel(
    const float* diagonal, const float* matrix, float* output, int64_t rows, int64_t cols) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t count = rows * cols;
    if (index < count) output[index] = diagonal[index / cols] * matrix[index];
}
extern "C" void diagonal_launcher(
    const float* diagonal, const float* matrix, float* output,
    int64_t rows, int64_t cols, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    int64_t count = rows * cols;
    diagonal_kernel<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream>>>(
        diagonal, matrix, output, rows, cols);
}
""",
    r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
extern "C" void diagonal_launcher(
    const float*, const float*, float*, int64_t, int64_t, void*);
void diagonal_forward(tvm::ffi::Tensor diagonal, tvm::ffi::Tensor matrix, tvm::ffi::Tensor output) {
    auto shape = matrix.shape();
    void* stream = TVMFFIEnvGetStream(matrix.device().device_type, matrix.device().device_id);
    diagonal_launcher(static_cast<const float*>(diagonal.data_ptr()),
        static_cast<const float*>(matrix.data_ptr()), static_cast<float*>(output.data_ptr()),
        shape[0], shape[1], stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(diagonal_forward, diagonal_forward);
""",
    r"""
import torch
import tvm_ffi_extension
class ModelNew(torch.nn.Module):
    def forward(self, diagonal, matrix):
        output = torch.empty_like(matrix)
        tvm_ffi_extension.diagonal_forward(diagonal, matrix, output)
        return output
""",
)


SUM_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.sum(x, dim=self.dim, keepdim=True)
def get_inputs():
    return [torch.rand(128, 512, 1024)]
def get_init_inputs():
    return [1]
"""
SUM_KERNEL = _package(
    r"""
#include <cuda_runtime.h>
__global__ void sum_dim1_kernel(
    const float* input, float* output, int batch, int reduce, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = batch * width;
    if (index < count) {
        int b = index / width;
        int w = index - b * width;
        float sum = 0.0f;
        for (int r = 0; r < reduce; ++r) sum += input[(b * reduce + r) * width + w];
        output[index] = sum;
    }
}
extern "C" void sum_dim1_launcher(
    const float* input, float* output, int batch, int reduce, int width, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    int count = batch * width;
    sum_dim1_kernel<<<(count + 255) / 256, 256, 0, stream>>>(input, output, batch, reduce, width);
}
""",
    r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
extern "C" void sum_dim1_launcher(const float*, float*, int, int, int, void*);
void sum_dim1_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output) {
    auto shape = input.shape();
    void* stream = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    sum_dim1_launcher(static_cast<const float*>(input.data_ptr()), static_cast<float*>(output.data_ptr()),
        static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]), stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sum_dim1_forward, sum_dim1_forward);
""",
    r"""
import torch
import tvm_ffi_extension
class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        output = torch.empty((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype)
        tvm_ffi_extension.sum_dim1_forward(x, output)
        return output
""",
)


AVG_POOL_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        return self.pool(x)
def get_inputs():
    return [torch.rand(32, 64, 8192)]
def get_init_inputs():
    return [8, 2, 3]
"""
AVG_POOL_KERNEL = _package(
    r"""
#include <cuda_runtime.h>
__global__ void avg_pool1d_kernel(
    const float* input, float* output, int channels, int length, int out_length,
    int kernel_size, int stride, int padding, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        int out_x = index % out_length;
        int nc = index / out_length;
        int start = out_x * stride - padding;
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int x = start + k;
            if (x >= 0 && x < length) sum += input[nc * length + x];
        }
        output[index] = sum / static_cast<float>(kernel_size);
    }
}
extern "C" void avg_pool1d_launcher(
    const float* input, float* output, int batch, int channels, int length, int out_length,
    int kernel_size, int stride, int padding, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    int count = batch * channels * out_length;
    avg_pool1d_kernel<<<(count + 255) / 256, 256, 0, stream>>>(input, output, channels,
        length, out_length, kernel_size, stride, padding, count);
}
""",
    r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
extern "C" void avg_pool1d_launcher(
    const float*, float*, int, int, int, int, int, int, int, void*);
void avg_pool1d_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output,
                        int64_t kernel_size, int64_t stride, int64_t padding) {
    auto in_shape = input.shape(); auto out_shape = output.shape();
    void* stream = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    avg_pool1d_launcher(static_cast<const float*>(input.data_ptr()), static_cast<float*>(output.data_ptr()),
        static_cast<int>(in_shape[0]), static_cast<int>(in_shape[1]), static_cast<int>(in_shape[2]),
        static_cast<int>(out_shape[2]), static_cast<int>(kernel_size), static_cast<int>(stride),
        static_cast<int>(padding), stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(avg_pool1d_forward, avg_pool1d_forward);
""",
    r"""
import torch
import tvm_ffi_extension
class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
    def forward(self, x):
        out_length = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty((x.shape[0], x.shape[1], out_length), device=x.device, dtype=x.dtype)
        tvm_ffi_extension.avg_pool1d_forward(
            x, output, self.kernel_size, self.stride, self.padding)
        return output
""",
)


LAYER_NORM_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (self.normalized_shape,))
def get_inputs():
    return [torch.rand(4096, 1024)]
def get_init_inputs():
    return [1024]
"""
LAYER_NORM_KERNEL = _package(
    r"""
#include <cuda_runtime.h>
#include <math.h>
__global__ void layer_norm_kernel(const float* input, float* output, int cols, float eps) {
    int row = blockIdx.x;
    float sum = 0.0f, square_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float value = input[row * cols + col]; sum += value; square_sum += value * value;
    }
    __shared__ float shared_sum[256], shared_square[256];
    shared_sum[threadIdx.x] = sum; shared_square[threadIdx.x] = square_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_square[threadIdx.x] += shared_square[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = shared_sum[0] / cols;
    float variance = fmaxf(shared_square[0] / cols - mean * mean, 0.0f);
    float inv_std = rsqrtf(variance + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        output[row * cols + col] = (input[row * cols + col] - mean) * inv_std;
}
extern "C" void layer_norm_launcher(
    const float* input, float* output, int rows, int cols, float eps, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    layer_norm_kernel<<<rows, 256, 0, stream>>>(input, output, cols, eps);
}
""",
    r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
extern "C" void layer_norm_launcher(const float*, float*, int, int, float, void*);
void layer_norm_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output, double eps) {
    auto shape = input.shape();
    void* stream = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    layer_norm_launcher(static_cast<const float*>(input.data_ptr()), static_cast<float*>(output.data_ptr()),
        static_cast<int>(input.numel() / shape[1]), static_cast<int>(shape[1]), static_cast<float>(eps), stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(layer_norm_forward, layer_norm_forward);
""",
    r"""
import torch
import tvm_ffi_extension
class ModelNew(torch.nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
    def forward(self, x):
        output = torch.empty_like(x)
        tvm_ffi_extension.layer_norm_forward(x, output, 1e-5)
        return output
""",
)


BMM_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def forward(self, a, b):
        return torch.bmm(a, b)
def get_inputs():
    return [torch.rand(32, 128, 64), torch.rand(32, 64, 128)]
def get_init_inputs():
    return []
"""
BMM_KERNEL = _package(
    r"""
#include <cuda_runtime.h>
__global__ void bmm_kernel(
    const float* a, const float* b, float* output, int batch, int m, int k, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = batch * m * n;
    if (index < count) {
        int col = index % n; int row = (index / n) % m; int batch_index = index / (m * n);
        float sum = 0.0f;
        for (int inner = 0; inner < k; ++inner)
            sum = fmaf(a[(batch_index * m + row) * k + inner],
                       b[(batch_index * k + inner) * n + col], sum);
        output[index] = sum;
    }
}
extern "C" void bmm_launcher(const float* a, const float* b, float* output,
    int batch, int m, int k, int n, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle); int count = batch * m * n;
    bmm_kernel<<<(count + 255) / 256, 256, 0, stream>>>(a, b, output, batch, m, k, n);
}
""",
    r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
extern "C" void bmm_launcher(const float*, const float*, float*, int, int, int, int, void*);
void bmm_forward(tvm::ffi::Tensor a, tvm::ffi::Tensor b, tvm::ffi::Tensor output) {
    auto ashape = a.shape(); auto bshape = b.shape();
    void* stream = TVMFFIEnvGetStream(a.device().device_type, a.device().device_id);
    bmm_launcher(static_cast<const float*>(a.data_ptr()), static_cast<const float*>(b.data_ptr()),
        static_cast<float*>(output.data_ptr()), static_cast<int>(ashape[0]), static_cast<int>(ashape[1]),
        static_cast<int>(ashape[2]), static_cast<int>(bshape[2]), stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_forward, bmm_forward);
""",
    r"""
import torch
import tvm_ffi_extension
class ModelNew(torch.nn.Module):
    def forward(self, a, b):
        output = torch.empty((a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype)
        tvm_ffi_extension.bmm_forward(a, b, output)
        return output
""",
)


BATCH_NORM_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def forward(self, x, running_mean, running_var, weight, bias):
        return torch.nn.functional.batch_norm(
            x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5)
def get_inputs():
    channels = 64
    return [torch.rand(64, channels, 128, 128), torch.zeros(channels),
            torch.ones(channels), torch.ones(channels), torch.zeros(channels)]
def get_init_inputs():
    return []
"""
BATCH_NORM_KERNEL = _package(
    r"""
#include <cuda_runtime.h>
#include <math.h>
__global__ void batch_norm_kernel(const float* input, const float* mean, const float* variance,
    const float* weight, const float* bias, float* output, int channels, int spatial, int64_t count, float eps) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        int channel = static_cast<int>((index / spatial) % channels);
        output[index] = (input[index] - mean[channel]) * rsqrtf(variance[channel] + eps)
            * weight[channel] + bias[channel];
    }
}
extern "C" void batch_norm_launcher(const float* input, const float* mean, const float* variance,
    const float* weight, const float* bias, float* output, int channels, int spatial,
    int64_t count, float eps, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    batch_norm_kernel<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream>>>(
        input, mean, variance, weight, bias, output, channels, spatial, count, eps);
}
""",
    r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
extern "C" void batch_norm_launcher(const float*, const float*, const float*, const float*, const float*,
    float*, int, int, int64_t, float, void*);
void batch_norm_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor mean, tvm::ffi::Tensor variance,
    tvm::ffi::Tensor weight, tvm::ffi::Tensor bias, tvm::ffi::Tensor output, double eps) {
    auto shape = input.shape();
    void* stream = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    batch_norm_launcher(static_cast<const float*>(input.data_ptr()), static_cast<const float*>(mean.data_ptr()),
        static_cast<const float*>(variance.data_ptr()), static_cast<const float*>(weight.data_ptr()),
        static_cast<const float*>(bias.data_ptr()), static_cast<float*>(output.data_ptr()),
        static_cast<int>(shape[1]), static_cast<int>(shape[2] * shape[3]), input.numel(),
        static_cast<float>(eps), stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_norm_forward, batch_norm_forward);
""",
    r"""
import torch
import tvm_ffi_extension
class ModelNew(torch.nn.Module):
    def forward(self, x, running_mean, running_var, weight, bias):
        output = torch.empty_like(x)
        tvm_ffi_extension.batch_norm_forward(
            x, running_mean, running_var, weight, bias, output, 1e-5)
        return output
""",
)


CONV_REFERENCE = r"""
import torch
class Model(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.nn.functional.conv2d(x, weight, bias, stride=4, padding=2)
def get_inputs():
    return [torch.rand(256, 3, 224, 224), torch.rand(96, 3, 11, 11), torch.rand(96)]
def get_init_inputs():
    return []
"""
CONV_KERNEL = _package(
    r"""
#include <cuda_runtime.h>
__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
    int in_channels, int input_height, int input_width, int out_channels,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride, int padding, int64_t count) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        int x = index % output_width;
        int y = (index / output_width) % output_height;
        int oc = (index / (static_cast<int64_t>(output_width) * output_height)) % out_channels;
        int n = index / (static_cast<int64_t>(output_width) * output_height * out_channels);
        float sum = bias[oc];
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ky = 0; ky < kernel_height; ++ky) {
                for (int kx = 0; kx < kernel_width; ++kx) {
                    int iy = y * stride + ky - padding;
                    int ix = x * stride + kx - padding;
                    if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                        sum = fmaf(
                            input[((n * in_channels + ic) * input_height + iy) * input_width + ix],
                            weight[((oc * in_channels + ic) * kernel_height + ky) * kernel_width + kx],
                            sum);
                    }
                }
            }
        }
        output[index] = sum;
    }
}
extern "C" void conv2d_launcher(const float* input, const float* weight, const float* bias, float* output,
    int in_channels, int input_height, int input_width, int out_channels,
    int output_height, int output_width, int kernel_height, int kernel_width,
    int stride, int padding, int64_t count, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    conv2d_kernel<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream>>>(
        input, weight, bias, output, in_channels, input_height, input_width, out_channels,
        output_height, output_width, kernel_height, kernel_width, stride, padding, count);
}
""",
    r"""
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
extern "C" void conv2d_launcher(const float*, const float*, const float*, float*,
    int, int, int, int, int, int, int, int, int, int, int64_t, void*);
void conv2d_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor weight, tvm::ffi::Tensor bias,
                    tvm::ffi::Tensor output, int64_t stride, int64_t padding) {
    auto shape = input.shape(); auto weight_shape = weight.shape();
    auto output_shape = output.shape();
    void* stream = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    conv2d_launcher(static_cast<const float*>(input.data_ptr()), static_cast<const float*>(weight.data_ptr()),
        static_cast<const float*>(bias.data_ptr()), static_cast<float*>(output.data_ptr()),
        static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3]),
        static_cast<int>(weight_shape[0]), static_cast<int>(output_shape[2]),
        static_cast<int>(output_shape[3]), static_cast<int>(weight_shape[2]),
        static_cast<int>(weight_shape[3]), static_cast<int>(stride), static_cast<int>(padding),
        output.numel(), stream);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(conv2d_forward, conv2d_forward);
""",
    r"""
import torch
import tvm_ffi_extension
class ModelNew(torch.nn.Module):
    def forward(self, x, weight, bias):
        output_height = (x.shape[2] + 4 - weight.shape[2]) // 4 + 1
        output_width = (x.shape[3] + 4 - weight.shape[3]) // 4 + 1
        output = torch.empty((x.shape[0], weight.shape[0], output_height, output_width),
                             device=x.device, dtype=x.dtype)
        tvm_ffi_extension.conv2d_forward(x, weight, bias, output, 4, 2)
        return output
""",
)


CASES = (
    NoiseFloorCase("gemm_rmsnorm", "GEMM + RMSNorm", "short", GEMM_RMSNORM_REFERENCE_CODE,
                   GEMM_RMSNORM_KERNEL_CODE, GEMM_RMSNORM_INPUT_SHAPES, shape_policy="existing_speed_test"),
    NoiseFloorCase("batch_norm", "BatchNorm inference", "medium", BATCH_NORM_REFERENCE,
                   BATCH_NORM_KERNEL, {"x": [64, 64, 128, 128]}),
    NoiseFloorCase("conv2d", "Conv2D 11x11 stride 4", "long", CONV_REFERENCE, CONV_KERNEL,
                   {"x": [256, 3, 224, 224], "weight": [96, 3, 11, 11], "bias": [96]}),
    NoiseFloorCase("kb_19_relu", "ReLU", "short", RELU_REFERENCE, RELU_KERNEL,
                   {"x": [4096, 256]}, 19, "19_ReLU"),
    NoiseFloorCase("kb_12_diagonal_matmul", "Diagonal matrix multiplication", "medium",
                   DIAGONAL_REFERENCE, DIAGONAL_KERNEL, {"diagonal": [4096], "matrix": [4096, 4096]},
                   12, "12_Matmul_with_diagonal_matrices_"),
    NoiseFloorCase("kb_88_mingpt_gelu", "MinGPT GELU", "long", GELU_REFERENCE, GELU_KERNEL,
                   {"x": [8192, 8192]}, 88, "88_MinGPTNewGelu"),
    NoiseFloorCase("kb_47_sum_dim1", "Sum reduction dim=1", "long", SUM_REFERENCE, SUM_KERNEL,
                   {"x": [128, 512, 1024]}, 47, "47_Sum_reduction_over_a_dimension"),
    NoiseFloorCase("kb_44_avg_pool1d", "Average Pooling 1D", "medium", AVG_POOL_REFERENCE,
                   AVG_POOL_KERNEL, {"x": [32, 64, 8192]}, 44, "44_Average_Pooling_1D"),
    NoiseFloorCase("kb_40_layer_norm", "LayerNorm", "medium", LAYER_NORM_REFERENCE,
                   LAYER_NORM_KERNEL, {"x": [4096, 1024]}, 40, "40_LayerNorm"),
    NoiseFloorCase("kb_3_batched_matmul", "Batched matrix multiplication", "medium", BMM_REFERENCE,
                   BMM_KERNEL, {"a": [32, 128, 64], "b": [32, 64, 128]},
                   3, "3_Batched_matrix_multiplication"),
)

CASE_BY_ID = {case.case_id: case for case in CASES}


def get_noise_floor_cases() -> tuple[NoiseFloorCase, ...]:
    return CASES
