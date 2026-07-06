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
# CUDA-Agent three-section / pybind-registry markers. Their presence means the
# submission is the cuda_agent format, NOT the single-block load_inline format.
CUDA_AGENT_MARKER_RE = re.compile(
    r"(?:"
    r"###\s*CUDA_KERNELS|"
    r"###\s*APPLY_BINDINGS|"
    r"###\s*MODEL_NEW|"
    r"\bbinding_registry\.h\b|"
    r"\bREGISTER_BINDING\s*\(|"
    r"\bcuda_extension\b"
    r")",
    re.IGNORECASE,
)
# MusaCoder / stock-KernelBench single-block marker: a torch.utils.cpp_extension
# load_inline submission. The bare token is enough — cuda_agent (binding_registry
# / ninja) and tvm_ffi submissions never reference load_inline, and they are
# matched first, so this only fires on a genuine load_inline module.
LOAD_INLINE_MARKER_RE = re.compile(r"\bload_inline\b", re.IGNORECASE)
# Markdown python code fences inside a model response.
CODE_FENCE_RE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.DOTALL)
# Chat special tokens that may trail the closing code fence.
SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_end|im_start|endoftext|eot_id|eom_id)\|>")
AUTO_KERNEL_BACKENDS = {"auto", "mixed", "auto_cuda_tvm_ffi", "cuda_agent_or_tvm_ffi"}


def strip_think_blocks(text: str) -> str:
    """Return the final answer region after model reasoning."""
    text = text or ""
    think_end_matches = list(THINK_END_RE.finditer(text))
    if think_end_matches:
        return text[think_end_matches[-1].end() :]
    return THINK_BLOCK_RE.sub("", text)


def extract_model_code(text: str, *, entry_point: str = "ModelNew") -> str:
    """Extract the runnable ModelNew Python module from a raw model response.

    MusaCoder/load_inline submissions arrive as a free-form response: reasoning
    inside ``<think>...</think>`` followed by a single fenced ``python`` block
    holding the ``ModelNew`` solution, sometimes trailed by a chat special token
    (e.g. ``<|im_end|>``). This returns clean, exec-ready Python:

    1. drop the reasoning region (keep only text after the last ``</think>``);
    2. of the fenced ``python`` blocks, take the LAST one that defines
       ``class {entry_point}`` (fall back to the last block mentioning
       ``load_inline``, else the last block);
    3. if there are no fences, use the post-reasoning text as-is;
    4. strip trailing chat special tokens.

    Code that is already a clean module (no reasoning, no fences) passes through.
    """
    stripped = strip_think_blocks(text or "")
    blocks = [match.group(1) for match in CODE_FENCE_RE.finditer(stripped)]

    candidate: str | None = None
    for block in blocks:
        if f"class {entry_point}" in block:
            candidate = block
    if candidate is None:
        for block in blocks:
            if LOAD_INLINE_MARKER_RE.search(block):
                candidate = block
    if candidate is None:
        candidate = blocks[-1] if blocks else stripped

    candidate = SPECIAL_TOKEN_RE.sub("", candidate)
    return candidate.strip() + "\n"


def normalize_kernel_backend(kernel_backend: Any | None, *, default: str = "triton") -> str:
    if hasattr(kernel_backend, "value"):
        kernel_backend = kernel_backend.value
    return str(kernel_backend or default).strip().lower().replace("-", "_")


def is_auto_kernel_backend(kernel_backend: Any | None) -> bool:
    return normalize_kernel_backend(kernel_backend) in AUTO_KERNEL_BACKENDS


def detect_kernel_backend(text: str, *, default: str = "cuda_agent") -> str:
    """Detect the concrete KernelGym backend from strong final-answer markers.

    Order matters: TVM-FFI and cuda_agent (three-section / pybind-registry) use
    explicit markers and are checked first, so the load_inline branch only fires
    for a genuine single-block ``load_inline`` submission. Anything else keeps
    the historical ``cuda_agent`` default.
    """
    stripped = strip_think_blocks(text) or ""
    if TVM_FFI_MARKER_RE.search(stripped):
        return "tvm_ffi"
    if CUDA_AGENT_MARKER_RE.search(stripped):
        return "cuda_agent"
    if LOAD_INLINE_MARKER_RE.search(stripped):
        return "load_inline"
    return normalize_kernel_backend(default, default="cuda_agent")


def resolve_kernel_backend(text: str, kernel_backend: Any | None = "triton") -> str:
    normalized = normalize_kernel_backend(kernel_backend)
    if normalized in AUTO_KERNEL_BACKENDS:
        return detect_kernel_backend(text)
    return normalized
