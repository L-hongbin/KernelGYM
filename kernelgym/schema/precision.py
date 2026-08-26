"""Precision names shared by the API and KernelBench validation pipeline."""

from __future__ import annotations

from typing import Any


PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "torch.float32": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "torch.float16": "fp16",
    "torch.half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
}


def normalize_precision(value: Any, *, strict: bool = False) -> str:
    """Return a canonical precision name, defaulting unknown internal values to fp32."""

    if value is None:
        return "fp32"
    normalized = PRECISION_ALIASES.get(str(value).strip().lower())
    if normalized is not None:
        return normalized
    if strict:
        accepted = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"Unsupported precision {value!r}; expected one of: {accepted}")
    return "fp32"
