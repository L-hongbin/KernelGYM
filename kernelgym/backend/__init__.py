"""KernelGym backend interfaces.

Keep this module import-light so tests and schema imports do not require Torch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Backend",
    "KernelBenchBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]


def __getattr__(name: str) -> Any:
    if name == "Backend":
        from .base import Backend

        return Backend
    if name == "KernelBenchBackend":
        from .kernelbench.dispatcher import KernelBenchBackend

        return KernelBenchBackend
    if name in {"get_backend", "list_backends", "register_backend"}:
        from .registry import get_backend, list_backends, register_backend

        return {
            "get_backend": get_backend,
            "list_backends": list_backends,
            "register_backend": register_backend,
        }[name]
    raise AttributeError(name)
