from kernelgym.common import Backend
from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend
from kernelgym.backend.kernelbench import tvm_ffi_backend
from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend
from kernelgym.server.api.models import EvaluationRequest
from kernelgym.toolkit.validation import precheck_cuda_agent_submission


def _model_fields(model_cls: type) -> set[str]:
    fields = getattr(model_cls, "model_fields", None)
    if fields is None:
        fields = getattr(model_cls, "__fields__", {})
    return set(fields)


def test_schema_exposes_compile_acceleration_fields() -> None:
    fields = _model_fields(EvaluationRequest)
    assert "num_warmup" in fields
    assert "perf_trim_count" in fields
    assert "split_compile_and_execute" in fields
    assert "pure_compile_task" in fields
    assert "enable_compile_artifact_cache" in fields
    assert "task_stage" in fields
    assert "required_resource" in fields
    assert "compile_artifact" in fields


def test_evaluation_request_defaults_to_auto_backend() -> None:
    request = EvaluationRequest(
        task_id="t",
        reference_code="class Model: pass",
        kernel_code="class ModelNew: pass",
    )

    assert request.backend == Backend.AUTO


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


def test_cuda_agent_ninja_object_edge_parser() -> None:
    build_text = """
ninja_required_version = 1.3
cxx = c++

build generated.cuda.o: cuda_compile /tmp/work/kernels/generated.cu
build generated_binding.o: compile /tmp/work/kernels/generated_binding.cpp
build extension.so: link generated.cuda.o generated_binding.o
"""

    _header, edges = KernelBenchCudaAgentBackend._ninja_header_and_object_edges(build_text)

    assert edges == [
        {
            "output": "generated.cuda.o",
            "rule": "cuda_compile",
            "source": "/tmp/work/kernels/generated.cu",
        },
        {
            "output": "generated_binding.o",
            "rule": "compile",
            "source": "/tmp/work/kernels/generated_binding.cpp",
        },
    ]


def test_cuda_agent_object_reuse_skips_module_bound_sources(tmp_path) -> None:
    source = tmp_path / "binding.cpp"
    source.write_text(
        """
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
""",
        encoding="utf-8",
    )

    reusable, reason = KernelBenchCudaAgentBackend._source_is_reusable_object(source)

    assert reusable is False
    assert "module name" in str(reason)


def test_cuda_agent_rewrites_ninja_link_inputs_for_cached_objects(tmp_path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    build_ninja = build_dir / "build.ninja"
    build_ninja.write_text(
        """
build generated.cuda.o: cuda_compile /tmp/work/generated.cu
build generated_binding.o: compile /tmp/work/generated_binding.cpp
build extension.so: link generated.cuda.o generated_binding.o binding.o
""".lstrip(),
        encoding="utf-8",
    )

    KernelBenchCudaAgentBackend._rewrite_manual_ninja_for_cached_objects(
        build_dir,
        {"generated_binding.o": "/cache/generated_binding.o"},
    )

    rewritten = build_ninja.read_text(encoding="utf-8")
    assert "build generated_binding.o:" not in rewritten
    assert "generated.cuda.o /cache/generated_binding.o binding.o" in rewritten


def test_tvm_ffi_compile_artifact_cache_reuses_built_work_dir(tmp_path, monkeypatch) -> None:
    calls = {"build": 0}

    def fake_precheck(model_code, cuda_sources, *, entry_point):
        return (
            "",
            None,
            {
                "passed": True,
                "detected_extension_calls": ["identity"],
            },
        )

    def fake_build(work_dir, cpp_files, cuda_files):
        calls["build"] += 1
        assert cpp_files
        assert cuda_files
        build_dir = work_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        so_path = build_dir / f"{work_dir.name}.so"
        so_path.write_bytes(b"fake-so")
        return {
            "compiled": True,
            "so_path": str(so_path),
            "module_name": work_dir.name,
            "build_backend": "fake",
        }

    monkeypatch.setattr(tvm_ffi_backend, "precheck_tvm_ffi_submission", fake_precheck)
    monkeypatch.setattr(KernelBenchTvmFfiBackend, "_build_extension", staticmethod(fake_build))
    monkeypatch.setattr(KernelBenchTvmFfiBackend, "_compile_artifact_cache_root", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(KernelBenchTvmFfiBackend, "_tvm_ffi_version", staticmethod(lambda: "test"))
    monkeypatch.setattr(KernelBenchCudaAgentBackend, "_cuda_arch_fingerprint", staticmethod(lambda: "sm_test"))
    monkeypatch.setattr(KernelBenchTvmFfiBackend, "_normalize_device", staticmethod(lambda device: "cuda:0"))
    monkeypatch.setenv("KERNELGYM_COMPILE_ARTIFACT_CACHE", "true")

    backend = KernelBenchTvmFfiBackend()
    code = """
import torch
import tvm_ffi_extension

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return tvm_ffi_extension.identity(x)
"""
    cuda_sources = {
        "kernels/generated.cpp": "TVM_FFI_DLL_EXPORT_TYPED_FUNC(identity, identity);",
        "kernels/generated.cu": "__global__ void identity_kernel(float* x) {}",
    }

    first = backend.compile(code, cuda_sources=cuda_sources)
    second = backend.compile(code, cuda_sources=cuda_sources)

    assert first["compiled"] is True
    assert first["compile_artifact_cache_enabled"] is True
    assert first["compile_artifact_cache_hit"] is False
    assert second["compiled"] is True
    assert second["compile_artifact_cache_hit"] is True
    assert second["persistent_work_dir"] is True
    assert calls["build"] == 1
