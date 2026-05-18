"""KernelBench backend implementations.

Backends are exposed lazily because CUDA/Torch imports are expensive and not
needed for configuration or schema-only tests.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "KernelBenchBackend",
    "KernelBenchBackendBase",
    "KernelBenchCudaAgentBackend",
    "KernelBenchCudaBackend",
    "KernelBenchTvmFfiBackend",
    "KernelBenchTritonBackend",
]


def __getattr__(name: str) -> Any:
    if name == "KernelBenchBackend":
        from .dispatcher import KernelBenchBackend

        return KernelBenchBackend
    if name == "KernelBenchBackendBase":
        from .base import KernelBenchBackendBase

        return KernelBenchBackendBase
    if name == "KernelBenchCudaAgentBackend":
        from .cuda_agent_backend import KernelBenchCudaAgentBackend

        return KernelBenchCudaAgentBackend
    if name == "KernelBenchCudaBackend":
        from .cuda_backend import KernelBenchCudaBackend

        return KernelBenchCudaBackend
    if name == "KernelBenchTvmFfiBackend":
        from .tvm_ffi_backend import KernelBenchTvmFfiBackend

        return KernelBenchTvmFfiBackend
    if name == "KernelBenchTritonBackend":
        from .triton_backend import KernelBenchTritonBackend

        return KernelBenchTritonBackend
    raise AttributeError(name)
