"""Unit tests for static load_inline decoy detection (CPU-only)."""

from __future__ import annotations

from kernelgym.toolkit.kernelbench.load_inline_decoy import detect_load_inline_decoy

REAL_VAR = """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
ext = load_inline(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return ext.f(a, b)
"""

REAL_SELF_ATTR = """import torch.nn as nn
from torch.utils.cpp_extension import load_inline
ext = load_inline(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.add = ext
    def forward(self, a, b):
        return self.add.f(a, b)
"""

REAL_HELPER = """import torch.nn as nn
from torch.utils.cpp_extension import load_inline
ext = load_inline(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
def _run(a, b):
    return ext.f(a, b)
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return _run(a, b)
"""

DECOY_UNUSED_VAR = """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
ext = load_inline(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return torch.add(a, b)
"""

DECOY_STORED_UNUSED = """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
ext = load_inline(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.add = ext
    def forward(self, a, b):
        return torch.matmul(a, b)
"""

DECOY_NO_KERNEL = """import torch
import torch.nn as nn
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return a + b
"""


def test_real_var_use_not_decoy():
    v = detect_load_inline_decoy(REAL_VAR)
    assert v["decoy"] is False and v["used"] is True


def test_real_self_attr_not_decoy():
    assert detect_load_inline_decoy(REAL_SELF_ATTR)["decoy"] is False


def test_real_helper_not_decoy():
    # forward calls a module-level helper that uses the extension.
    assert detect_load_inline_decoy(REAL_HELPER)["decoy"] is False


def test_decoy_unused_var():
    v = detect_load_inline_decoy(DECOY_UNUSED_VAR)
    assert v["decoy"] is True and v["used"] is False


def test_decoy_stored_but_unused():
    assert detect_load_inline_decoy(DECOY_STORED_UNUSED)["decoy"] is True


def test_decoy_no_kernel_compiled():
    v = detect_load_inline_decoy(DECOY_NO_KERNEL)
    assert v["decoy"] is True


def test_syntax_error_not_flagged():
    # Deferred to the compile stage, never a spurious decoy.
    assert detect_load_inline_decoy("class ModelNew(:\n  pass")["decoy"] is False


# Direct `self.ext = load_inline(...)` (no intermediate variable), used in forward.
REAL_SELF_DIRECT = """import torch.nn as nn
from torch.utils.cpp_extension import load_inline
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = load_inline(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
    def forward(self, a, b):
        return self.ext.f(a, b)
"""

# Direct self-attr load_inline but forward falls back to torch -> decoy.
DECOY_SELF_DIRECT_UNUSED = """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = load_inline(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
    def forward(self, a, b):
        return torch.add(a, b)
"""

# Aliased import of load_inline, used in forward.
REAL_ALIAS = """import torch.nn as nn
from torch.utils.cpp_extension import load_inline as li
ext = li(name="add", cpp_sources="", cuda_sources="...", functions=["f"], verbose=False)
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return ext.f(a, b)
"""


def test_real_self_direct_not_decoy():
    # `self.ext = load_inline(...)` used via self.ext.f(...) must NOT be flagged.
    assert detect_load_inline_decoy(REAL_SELF_DIRECT)["decoy"] is False


def test_decoy_self_direct_unused():
    assert detect_load_inline_decoy(DECOY_SELF_DIRECT_UNUSED)["decoy"] is True


def test_real_alias_not_decoy():
    # `from ... import load_inline as li; ext = li(...)` used in forward.
    assert detect_load_inline_decoy(REAL_ALIAS)["decoy"] is False
