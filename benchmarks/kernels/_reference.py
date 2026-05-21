"""Shared reference (element-wise add of two 1-D tensors).

Both the cuda_agent and tvm_ffi fixtures evaluate against this same reference
so the only variable across backends is the binding/compile path.
"""

REFERENCE_CODE = '''
import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference: element-wise add of two 1-D tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    return [torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda")]


def get_init_inputs():
    return []
'''
