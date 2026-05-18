from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend
from kernelgym.server.api.models import EvaluationRequest
from kernelgym.toolkit.validation import precheck_cuda_agent_submission


def _model_fields(model_cls: type) -> set[str]:
    fields = getattr(model_cls, "model_fields", None)
    if fields is None:
        fields = getattr(model_cls, "__fields__", {})
    return set(fields)


def test_schema_matches_current_reward_api_not_lhb_split_api() -> None:
    fields = _model_fields(EvaluationRequest)
    assert "num_warmup" in fields
    assert "perf_trim_count" in fields
    assert "split_compile_and_execute" not in fields
    assert "pure_compile_task" not in fields
    assert "enable_compile_artifact_cache" not in fields


def test_cuda_agent_parser_strips_think_blocks_and_uses_last_complete_group() -> None:
    code = """
<think>
### CUDA_KERNELS
```cpp
__global__ void ignored_kernel(float* x) {}
```
</think>

### CUDA_KERNELS
```cpp
__global__ void first_kernel(float* x) {}
```
### APPLY_BINDINGS
```cpp
void bind_first(pybind11::module& m) {}
```
### MODEL_NEW
```python
class First:
    pass
```

### CUDA_KERNELS
```cpp
__global__ void second_kernel(float* x) {}
```
### APPLY_BINDINGS
```cpp
void bind_second(pybind11::module& m) {}
```
### MODEL_NEW
```python
class ModelNew:
    pass
```
"""
    sources, model_code = KernelBenchCudaAgentBackend._parse_embedded_sources(code)

    assert "ignored_kernel" not in sources["kernels/generated.cu"]
    assert "first_kernel" not in sources["kernels/generated.cu"]
    assert "second_kernel" in sources["kernels/generated.cu"]
    assert "bind_second" in sources["kernels/generated_binding.cpp"]
    assert "class ModelNew" in model_code


def test_cuda_agent_precheck_accepts_pybind11_module_binding() -> None:
    model_code = """
import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return cuda_extension.identity(x)
"""
    cuda_sources = {
        "kernels/generated.cu": """
#include <torch/extension.h>
__global__ void identity_kernel(float* x) {}
""",
        "kernels/generated_binding.cpp": """
#include <torch/extension.h>

torch::Tensor identity(torch::Tensor x) {
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("identity", &identity);
}
""",
    }

    error, error_code, info = precheck_cuda_agent_submission(model_code, cuda_sources, entry_point="ModelNew")

    assert error == ""
    assert error_code is None
    assert info["passed"] is True
    assert info["binding_mode"] == "pybind11_module"


def test_cuda_agent_precheck_rejects_missing_register_binding_semicolon() -> None:
    model_code = """
import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return cuda_extension.identity(x)
"""
    cuda_sources = {
        "kernels/generated.cu": """
#include <torch/extension.h>
__global__ void identity_kernel(float* x) {}
""",
        "kernels/generated_binding.cpp": """
#include "../binding_registry.h"

void bind_identity(pybind11::module& m) {
    m.def("identity", [](torch::Tensor x) { return x; });
}

REGISTER_BINDING(identity, bind_identity)
""",
    }

    error, error_code, info = precheck_cuda_agent_submission(model_code, cuda_sources, entry_point="ModelNew")

    assert "without a trailing ';'" in error
    assert error_code is not None
    assert info["passed"] is False
