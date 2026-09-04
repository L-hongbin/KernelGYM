"""Regression tests for language-aware static KernelBench checks."""

from __future__ import annotations

import pytest

from kernelgym.toolkit.kernelbench.static_checker import detect_extension_calls, validate_kernel_static
from kernelgym.toolkit.validation import precheck_cuda_agent_submission, precheck_tvm_ffi_submission


@pytest.mark.parametrize(
    "code",
    [
        "import math\nclass ModelNew:\n    scale = math.sqrt(64)\n",
        "import math as m\nclass ModelNew:\n    scale = m.sqrt(64)\n",
        "from math import sqrt as scalar_sqrt\nclass ModelNew:\n    scale = scalar_sqrt(64)\n",
        '"""Documentation mentioning x.sqrt() and torch.sum(x)."""\nclass ModelNew:\n    def __init__(self):\n        pass\n',
    ],
)
def test_python_scalar_math_literals_and_benign_pass_do_not_fail(code: str) -> None:
    assert validate_kernel_static(code).valid is True


def test_provable_torch_compute_and_exception_fallback_remain_blocked() -> None:
    unknown_receiver = "class ModelNew:\n    def forward(self, x): return x.sqrt()\n"
    torch_alias = "import torch as t\nclass ModelNew:\n    def forward(self, x): return t.sum(x)\n"
    functional_alias = (
        "from torch.nn import functional as F\nclass ModelNew:\n    def forward(self, x): return F.gelu(x)\n"
    )
    fallback = "class ModelNew:\n    def forward(self, x):\n        try: return custom(x)\n        except Exception: return x.sqrt()\n"
    for code in (torch_alias, functional_alias, fallback):
        assert validate_kernel_static(code).valid is False
    assert validate_kernel_static(unknown_receiver).valid is True
    assert "framework_compute" in validate_kernel_static(torch_alias).errors[0]
    assert "code_bypass" in validate_kernel_static(fallback).errors[0]


@pytest.mark.parametrize(
    ("module_name", "arguments"),
    [
        ("ReLU", ""),
        ("GELU", ""),
        ("Softmax", "dim=-1"),
        ("LogSoftmax", "dim=-1"),
        ("LayerNorm", "16"),
        ("BatchNorm1d", "16"),
        ("BatchNorm2d", "16"),
        ("BatchNorm3d", "16"),
    ],
)
def test_torch_module_construction_in_model_init_is_allowed(module_name: str, arguments: str) -> None:
    code = f"""import torch
import torch.nn as nn
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.operation = nn.{module_name}({arguments})

    def forward(self, x):
        return x
"""

    assert validate_kernel_static(code).valid is True


@pytest.mark.parametrize(
    "body",
    [
        "self.operations = [nn.ReLU(), nn.GELU()]",
        "(self.relu, self.gelu) = (nn.ReLU(), nn.GELU())",
        "self.operation = (operation := nn.ReLU())",
        "self.operation = nn.ReLU() if enabled else nn.GELU()",
        "self.operations = nn.ModuleList([nn.ReLU(), nn.GELU()])",
        "self.operations = [nn.ReLU() for _ in range(2)]",
        'setattr(self, "operation", nn.ReLU())',
    ],
)
def test_passively_stored_torch_modules_in_model_init_are_allowed(body: str) -> None:
    code = f"""import torch.nn as nn
class ModelNew:
    def __init__(self, enabled=True):
        {body}

    def forward(self, x):
        return x
"""

    assert validate_kernel_static(code).valid is True


@pytest.mark.parametrize("module_name", ["ReLU", "GELU", "Softmax", "LayerNorm", "BatchNorm1d"])
def test_torch_module_construction_outside_model_init_remains_blocked(module_name: str) -> None:
    code = f"""import torch.nn as nn
class ModelNew:
    def forward(self, x):
        operation = nn.{module_name}()
        return operation(x)
"""

    result = validate_kernel_static(code)

    assert result.valid is False
    assert any(error.startswith("framework_compute:") for error in result.errors)


@pytest.mark.parametrize(
    "body",
    [
        "self.cached = nn.ReLU()(x)",
        "self.operation = nn.ReLU()\n        self.cached = self.operation(x)",
        "self.operation = nn.ReLU()\n        self.cached = self.operation.forward(x)",
        "self.operation = nn.ReLU()\n        self.cached = self.operation.__call__(x)",
        'self.operation = nn.ReLU()\n        self.cached = getattr(self.operation, "forward")(x)',
        "operation = nn.ReLU()\n        alias = operation\n        self.cached = alias(x)",
        "self.operation = nn.ReLU()\n        apply = self.operation.forward\n        self.cached = apply(x)",
        "self.cached = (operation := nn.ReLU())(x)",
        ("self.operation = (operation := nn.ReLU())\n        self.cached = self.operation(x)"),
        "(operation,) = (nn.ReLU(),)\n        self.cached = operation(x)",
        "operations = [nn.ReLU()]\n        self.cached = operations[0](x)",
        "self.operations = nn.Sequential(nn.ReLU())\n        self.cached = self.operations(x)",
        "self.operation = nn.ReLU()\n        operations = [self.operation]\n        self.cached = operations[0](x)",
        (
            "self.operations = [None]\n"
            "        self.operations[0] = nn.ReLU()\n"
            "        self.cached = self.operations.__getitem__(0)(x)"
        ),
        ('setattr(self, "operation", nn.ReLU())\n        self.cached = getattr(self, "operation")(x)'),
        "self.operation = nn.ReLU()\n        self.cached = self.__dict__['operation'](x)",
        'self.operation = nn.ReLU()\n        self.cached = vars(self)["operation"](x)',
        ('self.operation = nn.ReLU()\n        self.cached = self.__getattribute__("operation")(x)'),
        (
            "self.operation = nn.ReLU()\n"
            "        def apply(value):\n"
            "            return self.operation(value)\n"
            "        self.cached = apply(x)"
        ),
    ],
)
def test_calling_a_constructed_torch_module_inside_model_init_remains_blocked(body: str) -> None:
    code = f"""import torch.nn as nn
class ModelNew:
    def __init__(self, x):
        {body}

    def forward(self, x):
        return x
"""

    result = validate_kernel_static(code)

    assert result.valid is False
    assert any(error.startswith("framework_compute:") for error in result.errors)


def test_constructor_exemption_is_limited_to_the_selected_entry_point_class() -> None:
    helper = """import torch.nn as nn
class Helper:
    def __init__(self):
        self.operation = nn.ReLU()
class ModelNew:
    def forward(self, x):
        return x
"""
    custom_entry_point = """import torch.nn as nn
class CustomKernel:
    def __init__(self):
        self.operation = nn.ReLU()
    def forward(self, x):
        return x
"""

    assert validate_kernel_static(helper).valid is False
    assert validate_kernel_static(custom_entry_point, entry_point="CustomKernel").valid is True


def test_nested_class_does_not_inherit_entry_point_constructor_exemption() -> None:
    code = """import torch.nn as nn
class ModelNew:
    def __init__(self):
        class Helper:
            operation = nn.ReLU()
        self.helper_type = Helper

    def forward(self, x):
        return x
"""

    assert validate_kernel_static(code).valid is False


def test_storing_a_constructor_type_cannot_hide_constructor_stage_execution() -> None:
    code = """import torch.nn as nn
class ModelNew:
    def __init__(self, x):
        self.operation_type = nn.ReLU
        self.operation = self.operation_type()
        self.cached = self.operation(x)

    def forward(self, x):
        return x
"""

    assert validate_kernel_static(code).valid is False


def test_reassigned_init_module_attribute_is_not_treated_as_torch_compute() -> None:
    code = """import torch.nn as nn
class ModelNew:
    def __init__(self, x, custom):
        self.operation = nn.ReLU()
        self.operation = custom
        self.cached = self.operation(x)

    def forward(self, x):
        return x
"""

    assert validate_kernel_static(code).valid is True


@pytest.mark.parametrize(
    ("body", "category"),
    [
        ("self.cached = torch.softmax(x, dim=-1)", "framework_compute"),
        ("self.cached = x.half()", "precision_downgrade"),
        (
            "try:\n            self.cached = custom(x)\n        except Exception:\n            self.cached = x",
            "code_bypass",
        ),
        ("torch.cuda.synchronize = lambda: None", "timing_event_patch"),
    ],
)
def test_non_constructor_safety_checks_remain_active_in_model_init(body: str, category: str) -> None:
    code = f"""import torch
class ModelNew:
    def __init__(self, x):
        {body}

    def forward(self, x):
        return x
"""

    result = validate_kernel_static(code)

    assert result.valid is False
    assert any(error.startswith(f"{category}:") for error in result.errors)


def test_validated_extension_alias_is_allowed_but_rebinding_is_not() -> None:
    allowed = "import tvm_ffi_extension as ext\nclass ModelNew:\n    def forward(self, x): return ext.gelu(x)\n"
    rebound = "import tvm_ffi_extension as ext\nimport torch\next = torch\nclass ModelNew:\n    def forward(self, x): return ext.sum(x)\n"
    assert validate_kernel_static(allowed, allowed_extension_modules={"tvm_ffi_extension"}).valid is True
    assert (
        "framework_compute"
        in validate_kernel_static(rebound, allowed_extension_modules={"tvm_ffi_extension"}).errors[0]
    )


def test_precision_and_stream_checks_do_not_match_custom_or_documented_methods() -> None:
    harmless = "class ModelNew:\n    def forward(self, x):\n        note = 'x.half(); x.record_stream(stream)'\n        return extension.record_stream(x)\n"
    assert validate_kernel_static(harmless).valid is True
    assert validate_kernel_static(harmless).warnings == []
    downgrade = "import torch\nclass ModelNew:\n    def forward(self, x): return x.to(torch.float16)\n"
    stream = "import torch\nstream = torch.cuda.Stream()\nclass ModelNew: pass\n"
    assert "precision_downgrade" in validate_kernel_static(downgrade).errors[0]
    assert "stream_injection" in validate_kernel_static(stream).warnings[0]


def test_unknown_half_stays_fail_closed_but_validated_extension_half_is_allowed() -> None:
    # Runtime ATen legality does not prove that an intermediate dtype was not
    # downgraded, so the dedicated FP32 rule remains conservative for .half().
    unknown = "class ModelNew:\n    def forward(self, x): return x.half()\n"
    extension = "import tvm_ffi_extension as ext\ndef forward(x): return ext.half(x)\n"
    assert "precision_downgrade" in validate_kernel_static(unknown).errors[0]
    assert (
        validate_kernel_static(
            extension,
            allowed_extension_modules={"tvm_ffi_extension"},
        ).valid
        is True
    )


@pytest.mark.parametrize(
    "code",
    [
        "import torch\nclass ModelNew:\n    def forward(self, x): return x.to(torch.float16)\n",
        "import torch as t\nclass ModelNew:\n    def forward(self, x): return x.to(dtype=t.half)\n",
        "from torch import float16 as fp16\nclass ModelNew:\n    def forward(self, x): return x.to(fp16)\n",
    ],
)
def test_explicit_torch_fp16_dtype_is_blocked_for_positional_keyword_and_aliases(code: str) -> None:
    assert "precision_downgrade" in validate_kernel_static(code).errors[0]


def test_scope_shadowing_and_try_finally_do_not_spoof_torch_or_bypass() -> None:
    shadowed = """import torch
class ModelNew:
    def forward(self, torch):
        return torch.sum()
"""
    try_finally = """class ModelNew:
    def forward(self, x):
        try:
            return x
        finally:
            cleanup()
"""
    getattr_torch = """import torch
class ModelNew:
    def forward(self, x):
        return getattr(torch, 'sum')(x)
"""
    assert validate_kernel_static(shadowed).valid is True
    assert validate_kernel_static(try_finally).valid is True
    assert "framework_compute" in validate_kernel_static(getattr_torch).errors[0]


@pytest.mark.parametrize(
    ("code", "category"),
    [
        ("import torch\ndef f(x): return torch.ops.aten.sum.default(x)\n", "framework_compute"),
        ("import torch as t\nt.cuda.synchronize = lambda: None\n", "timing_event_patch"),
        ("import time as timer\ntimer.time = lambda: 0\n", "timing_event_patch"),
        ("import torch\nsetattr(torch.cuda, 'synchronize', lambda: None)\n", "timing_event_patch"),
        ("import torch as t\nclass Fake(t.Tensor):\n    pass\n", "lazy_eval"),
    ],
)
def test_provable_alias_and_dynamic_framework_paths_remain_blocked(code: str, category: str) -> None:
    assert any(error.startswith(f"{category}:") for error in validate_kernel_static(code).errors)


def test_other_python_checks_use_symbol_provenance_instead_of_bare_names() -> None:
    harmless = """from helpers import ThreadPoolExecutor
class Tensor:
    pass
class LocalTensor(Tensor):
    pass
class ModelNew:
    def forward(self, torch, time, setattr):
        torch.cuda.synchronize = lambda: None
        time.time = lambda: 0
        setattr(torch.cuda, 'synchronize', lambda: None)
"""
    assert validate_kernel_static(harmless).valid is True

    import_threading = "import threading\nclass ModelNew: pass\n"
    dotted_torch_import = "import torch.cuda\ntorch.cuda.synchronize = lambda: None\n"
    timing_annotation = "import torch\ntorch.cuda.synchronize: object\n"
    timing_delete = "import torch\ndel torch.cuda.synchronize\n"
    assert any(error.startswith("thread_injection:") for error in validate_kernel_static(import_threading).errors)
    assert any(error.startswith("timing_event_patch:") for error in validate_kernel_static(dotted_torch_import).errors)
    assert validate_kernel_static(timing_annotation).valid is True
    assert any(error.startswith("timing_event_patch:") for error in validate_kernel_static(timing_delete).errors)


@pytest.mark.parametrize(
    "body",
    [
        "return [torch.sum() for torch in configs]",
        "for torch in configs:\n            torch.sum()\n        return configs",
        "with manager() as torch:\n            torch.sum()\n        return configs",
    ],
)
def test_iteration_and_context_targets_shadow_framework_names(body: str) -> None:
    code = f"""import torch
class ModelNew:
    def forward(self, configs):
        {body}
"""
    assert validate_kernel_static(code).valid is True


def test_triton_fp16_cast_and_native_tensor_half_remain_blocked() -> None:
    triton_cast = """import triton.language as tl
class ModelNew:
    def forward(self, x): return tl.astype(x, tl.float16)
"""
    native_half = "torch::Tensor input; auto y = input.half();"
    assert "precision_downgrade" in validate_kernel_static(triton_cast).errors[0]
    assert (
        "precision_downgrade"
        in validate_kernel_static("class ModelNew: pass\n", source_map={"kernel.cu": native_half}).errors[0]
    )


def test_extension_call_result_is_not_mistaken_for_extension_namespace() -> None:
    code = "import tvm_ffi_extension as ext\ndef forward(x): return ext.identity(x).half()\n"
    referenced, calls = detect_extension_calls(code, "tvm_ffi_extension")
    result = validate_kernel_static(code, allowed_extension_modules={"tvm_ffi_extension"})
    assert referenced is True
    assert calls == ["identity"]
    assert any(error.startswith("precision_downgrade:") for error in result.errors)


def test_native_scanner_ignores_literals_comments_and_only_blocks_provable_aten() -> None:
    model = "class ModelNew:\n    pass\n"
    harmless_native = 'const char* example = "at::sum(x); tensor.sqrt()";\n// torch::relu(x); tensor.norm()\n/* at::gelu(x); */\nauto scale = std::sqrt(4.0);'
    unknown_receiver = "auto b = tensor.sqrt();"
    torch_tensor = "torch::Tensor input; auto y = input.sqrt();"
    bad_native = "auto a = at::sum(x);"
    spaced_bad_native = "auto a = torch :: nn :: functional :: gelu(x);"
    assert validate_kernel_static(model, source_map={"kernels/a.cu": harmless_native}).valid is True
    assert validate_kernel_static(model, source_map={"kernels/a.cu": unknown_receiver}).valid is True
    assert "framework_compute" in validate_kernel_static(model, source_map={"kernels/a.cu": torch_tensor}).errors[0]
    assert "framework_compute" in validate_kernel_static(model, source_map={"kernels/a.cu": bad_native}).errors[0]
    assert (
        "framework_compute"
        in validate_kernel_static(
            model,
            source_map={"kernels/a.cu": spaced_bad_native},
        ).errors[0]
    )


def test_cuda_agent_precheck_allows_math_extension_alias_and_init_module_constructor() -> None:
    model = """import math as m
import torch
import cuda_extension as ext
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.GELU()

    def forward(self, x): return ext.gelu(x) * m.sqrt(0.5)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include <torch/extension.h>
torch::Tensor gelu(torch::Tensor x) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gelu", &gelu); }""",
    }
    error, error_code, info = precheck_cuda_agent_submission(model, sources, entry_point="ModelNew")
    assert error == "" and error_code is None and info["passed"] is True
    assert info["detected_extension_calls"] == ["gelu"]


def test_cuda_agent_precheck_passes_custom_entry_point_to_static_checker() -> None:
    model = """import torch.nn as nn
import cuda_extension as ext
class CustomKernel:
    def __init__(self):
        self.activation = nn.GELU()

    def forward(self, x):
        return ext.gelu(x)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include <torch/extension.h>
torch::Tensor gelu(torch::Tensor x) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gelu", &gelu); }""",
    }

    error, error_code, info = precheck_cuda_agent_submission(model, sources, entry_point="CustomKernel")

    assert error == "" and error_code is None and info["passed"] is True


@pytest.mark.parametrize(
    "extension_reference",
    [
        "import torch\nresult = torch.ops.cuda_extension.gelu(x)",
        "from . import cuda_extension as ext\nresult = ext.gelu(x)",
    ],
)
def test_cuda_agent_precheck_accepts_supported_extension_reference_forms(extension_reference: str) -> None:
    model = f"""{extension_reference.splitlines()[0]}
class ModelNew:
    def forward(self, x):
        {extension_reference.splitlines()[1]}
        return result
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include <torch/extension.h>
torch::Tensor gelu(torch::Tensor x) { return x; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gelu", &gelu); }""",
    }
    error, error_code, info = precheck_cuda_agent_submission(model, sources, entry_point="ModelNew")
    assert error == "" and error_code is None and info["passed"] is True
    assert info["detected_extension_calls"] == ["gelu"]


def test_tvm_ffi_precheck_allows_exported_gelu_alias_math_and_init_module_constructor() -> None:
    model = """import math as m
import torch.nn as nn
from tvm_ffi_extension import gelu as fused_gelu
class ModelNew:
    def __init__(self):
        self.activation = nn.GELU()

    def forward(self, x): return fused_gelu(x) * m.sqrt(0.5)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include <tvm/ffi/tvm_ffi.h>
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gelu, [](void* x) { return x; });""",
    }
    error, error_code, info = precheck_tvm_ffi_submission(model, sources, entry_point="ModelNew")
    assert error == "" and error_code is None and info["passed"] is True
    assert info["detected_extension_calls"] == ["gelu"]


def test_extension_detection_respects_rebinding() -> None:
    model = """import tvm_ffi_extension as extension
import tvm_ffi_extension as actual_extension
import helper
extension = helper
class ModelNew:
    def forward(self, x):
        extension.gelu(x)
        return actual_extension.identity(x)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include <tvm/ffi/tvm_ffi.h>
TVM_FFI_DLL_EXPORT_TYPED_FUNC(identity, [](void* x) { return x; });""",
    }
    error, error_code, info = precheck_tvm_ffi_submission(model, sources, entry_point="ModelNew")
    assert error == "" and error_code is None and info["passed"] is True
    assert info["detected_extension_calls"] == ["identity"]


def test_cuda_structural_markers_ignore_comments_and_literals() -> None:
    model = """import cuda_extension
class ModelNew:
    def forward(self, x): return cuda_extension.identity(x)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include "../binding_registry.h"
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
const char* example = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)";
void bind_identity(pybind11::module& m) { m.def("identity", [](auto x) { return x; }); }
REGISTER_BINDING(identity, bind_identity);""",
    }
    error, error_code, info = precheck_cuda_agent_submission(model, sources, entry_point="ModelNew")
    assert error == "" and error_code is None and info["passed"] is True
    assert info["binding_mode"] == "register_binding"


def test_cuda_include_inside_raw_literal_does_not_satisfy_precheck() -> None:
    model = """import cuda_extension
class ModelNew:
    def forward(self, x): return cuda_extension.identity(x)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """const char* docs = R"DOC(
#include "../binding_registry.h"
)DOC";
void bind_identity(pybind11::module& m) {}
REGISTER_BINDING(identity, bind_identity);""",
    }
    error, error_code, info = precheck_cuda_agent_submission(model, sources, entry_point="ModelNew")
    assert "must include binding_registry.h" in error
    assert error_code is not None
    assert info["passed"] is False


def test_tvm_structural_markers_ignore_comments_and_literals() -> None:
    model = """import tvm_ffi_extension
class ModelNew:
    def forward(self, x): return tvm_ffi_extension.identity(x)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include <tvm/ffi/tvm_ffi.h>
// PYBIND11_MODULE and REGISTER_BINDING(fake, fake); and cudaStream_t are examples only.
const char* example = "binding_registry.h cudaStream_t PYBIND11_MODULE";
const char* docs = R"DOC(
#include <cuda_runtime.h>
)DOC";
TVM_FFI_DLL_EXPORT_TYPED_FUNC(identity, [](void* x) { return x; });""",
    }
    error, error_code, info = precheck_tvm_ffi_submission(model, sources, entry_point="ModelNew")
    assert error == "" and error_code is None and info["passed"] is True


def test_tvm_export_in_comment_does_not_satisfy_precheck() -> None:
    model = """import tvm_ffi_extension
class ModelNew:
    def forward(self, x): return tvm_ffi_extension.identity(x)
"""
    sources = {
        "kernels/generated.cu": "__global__ void kernel(float* x) {}",
        "kernels/generated_binding.cpp": """#include <tvm/ffi/tvm_ffi.h>
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(identity, identity);""",
    }
    error, error_code, info = precheck_tvm_ffi_submission(model, sources, entry_point="ModelNew")
    assert "must export functions" in error
    assert error_code is not None
    assert info["passed"] is False
