"""Detect KernelBench binding style from raw model submissions."""

from __future__ import annotations

import re
from typing import Any


THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_END_RE = re.compile(r"</think\s*>", re.IGNORECASE)
TVM_FFI_MARKER_RE = re.compile(
    r"(?:"
    r"\btvm_ffi_extension\b|"
    r"\bTVM_FFI_DLL_EXPORT_TYPED_FUNC\b|"
    r"#\s*include\s*<tvm/ffi/|"
    r"\bTVMFFIEnvGetStream\b|"
    r"\btvm::ffi::Tensor\b"
    r")",
    re.IGNORECASE,
)
AUTO_KERNEL_BACKENDS = {"auto", "mixed", "auto_cuda_tvm_ffi", "cuda_agent_or_tvm_ffi"}


def strip_think_blocks(text: str) -> str:
    """Return the final answer region after model reasoning."""
    text = text or ""
    think_end_matches = list(THINK_END_RE.finditer(text))
    if think_end_matches:
        return text[think_end_matches[-1].end() :]
    return THINK_BLOCK_RE.sub("", text)


def normalize_kernel_backend(kernel_backend: Any | None, *, default: str = "triton") -> str:
    if hasattr(kernel_backend, "value"):
        kernel_backend = kernel_backend.value
    return str(kernel_backend or default).strip().lower().replace("-", "_")


def is_auto_kernel_backend(kernel_backend: Any | None) -> bool:
    return normalize_kernel_backend(kernel_backend) in AUTO_KERNEL_BACKENDS


def detect_kernel_backend(text: str, *, default: str = "cuda_agent") -> str:
    """Detect the concrete KernelGym backend from strong final-answer markers."""
    stripped = strip_think_blocks(text)
    if TVM_FFI_MARKER_RE.search(stripped or ""):
        return "tvm_ffi"
    return normalize_kernel_backend(default, default="cuda_agent")


def resolve_kernel_backend(text: str, kernel_backend: Any | None = "triton") -> str:
    normalized = normalize_kernel_backend(kernel_backend)
    if normalized in AUTO_KERNEL_BACKENDS:
        return detect_kernel_backend(text)
    return normalized
