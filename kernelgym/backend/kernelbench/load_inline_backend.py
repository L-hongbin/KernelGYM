"""load_inline backend for MusaCoder / stock-KernelBench submissions.

MusaCoder emits a single self-contained Python module: a few imports, one or
more ``torch.utils.cpp_extension.load_inline(...)`` calls that JIT-compile the
custom CUDA, and a ``class ModelNew(nn.Module)`` whose ``forward`` calls the
compiled extension. This is exactly the stock-KernelBench custom-model format,
so once the clean module is recovered it can be evaluated by the existing CUDA
path (``load_custom_model`` execs the module — which triggers the load_inline
build into the per-task ``TORCH_EXTENSIONS_DIR`` — and extracts ``ModelNew``).

The only MusaCoder-specific work is recovering that module from the raw model
response (reasoning + a fenced ``python`` block + trailing chat tokens). That is
done by ``extract_model_code``; everything else is inherited from
``KernelBenchCudaBackend`` so correctness (seeded reference-vs-ModelNew compare),
timing, speedup, and the compile-artifact/build-dir handling are unchanged.
"""

from __future__ import annotations

from typing import Any, Dict

from kernelgym.toolkit.kernelbench.binding_detection import extract_model_code
from kernelgym.toolkit.kernelbench.load_inline_decoy import detect_load_inline_decoy

from .cuda_backend import KernelBenchCudaBackend


class KernelBenchLoadInlineBackend(KernelBenchCudaBackend):
    name = "kernelbench.load_inline"

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        entry_point = kwargs.get("entry_point", "ModelNew")
        cleaned = extract_model_code(code, entry_point=entry_point)
        kwargs.setdefault("backend", "load_inline")
        artifact = super().compile(cleaned, **kwargs)
        if isinstance(artifact, dict):
            artifact["backend"] = "load_inline"
            # Record whether extraction actually changed the input, so the raw
            # response vs the recovered module can be told apart in postmortems.
            artifact["load_inline_extracted"] = cleaned.strip() != (code or "").strip()
        return artifact

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        # Defensive: re-extract in case load() is reached with a raw code field
        # (e.g. a precompiled_artifact that defaulted to the raw submission).
        # extract_model_code is idempotent on already-clean modules.
        entry_point = "ModelNew"
        if isinstance(artifact, dict) and artifact.get("code"):
            entry_point = artifact.get("entry_point", kwargs.get("entry_point", "ModelNew"))
            artifact = {**artifact, "code": extract_model_code(artifact["code"], entry_point=entry_point)}
        handle = super().load(artifact, **kwargs)
        if isinstance(handle, dict):
            handle["backend"] = "load_inline"
            # Static decoy verdict travels via profiling_hints, which the pipeline
            # already surfaces as backend_profiling_hints after load().
            cleaned = artifact.get("code", "") if isinstance(artifact, dict) else ""
            verdict = detect_load_inline_decoy(cleaned, entry_point=entry_point)
            hints = dict(handle.get("profiling_hints") or {})
            hints["load_inline_decoy"] = verdict
            handle["profiling_hints"] = hints
        return handle
