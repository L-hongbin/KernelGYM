"""API module for KernelGym.

Import the FastAPI application lazily so schema-only imports do not require
server runtime dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from .server import app

        return app
    raise AttributeError(name)
