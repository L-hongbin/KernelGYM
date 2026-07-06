"""Unit tests for MusaCoder load_inline detection and code extraction.

CPU-only; guards the additive behavior in binding_detection.py: load_inline is
recognized without disturbing cuda_agent / tvm_ffi routing, and a clean ModelNew
module is recovered from a raw model response.
"""

from __future__ import annotations

from kernelgym.toolkit.kernelbench.binding_detection import (
    detect_kernel_backend,
    extract_model_code,
    resolve_kernel_backend,
)

LOAD_INLINE_RESPONSE = """<think>
Let me draft a kernel.
```python
# scratch draft, not the answer
tmp = 0
```
</think>

Final solution:
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

src = "..."
ext = load_inline(name="add", cpp_sources="", cuda_sources=src, functions=["f"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return ext.f(a, b)
```<|im_end|>"""

CUDA_AGENT_RESPONSE = (
    "### CUDA_KERNELS\n```cpp\n__global__ void k(){}\n```\n"
    "### APPLY_BINDINGS\n```cpp\n#include \"../binding_registry.h\"\nREGISTER_BINDING(k);\n```\n"
    "### MODEL_NEW\n```python\nclass ModelNew:\n    pass\n```"
)

TVM_RESPONSE = "```python\nimport tvm_ffi_extension\nclass ModelNew:\n    pass\n```"

CLEAN_MODULE = (
    "import torch\n"
    "from torch.utils.cpp_extension import load_inline\n"
    "class ModelNew(torch.nn.Module):\n"
    "    pass\n"
)


def test_detect_load_inline():
    assert detect_kernel_backend(LOAD_INLINE_RESPONSE) == "load_inline"
    assert resolve_kernel_backend(LOAD_INLINE_RESPONSE, "auto") == "load_inline"


def test_detect_cuda_agent_unchanged():
    assert detect_kernel_backend(CUDA_AGENT_RESPONSE) == "cuda_agent"


def test_detect_tvm_unchanged():
    assert detect_kernel_backend(TVM_RESPONSE) == "tvm_ffi"


def test_explicit_backend_overrides_auto_detection():
    # An explicit backend must not be overridden by auto-detection.
    assert resolve_kernel_backend(LOAD_INLINE_RESPONSE, "cuda_agent") == "cuda_agent"


def test_extract_recovers_final_module():
    code = extract_model_code(LOAD_INLINE_RESPONSE)
    assert code.startswith("import torch")
    assert "class ModelNew" in code
    assert "load_inline" in code
    # reasoning, the scratch draft, and the trailing chat token are gone
    assert "scratch draft" not in code
    assert "tmp = 0" not in code
    assert "<|im_end|>" not in code
    assert "<think>" not in code


def test_extract_idempotent_on_clean_module():
    assert extract_model_code(CLEAN_MODULE).strip() == CLEAN_MODULE.strip()


def test_extract_prefers_last_modelnew_block():
    text = (
        "```python\nclass Other:\n    pass\n```\n"
        "```python\nfrom torch.utils.cpp_extension import load_inline\nclass ModelNew:\n    pass\n```"
    )
    code = extract_model_code(text)
    assert "class ModelNew" in code
    assert "class Other" not in code
