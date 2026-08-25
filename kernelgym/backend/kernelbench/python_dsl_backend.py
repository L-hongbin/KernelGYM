"""Shared KernelBench backend for Python GPU DSLs."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from typing import Any, Dict

from kernelgym.toolkit.kernelbench.loading import load_custom_model_with_tempfile
from kernelgym.toolkit.validation import validate_code

from .base import KernelBenchBackendBase


class KernelBenchPythonDslBackend(KernelBenchBackendBase):
    """Source-artifact backend for GPU DSLs that JIT on first execution.

    The CPU compile stage validates and serializes source.  Loading and the
    first correctness forward happen in the isolated GPU subprocess, where
    Triton/TileLang can safely initialize CUDA and compile device code.
    """

    backend_name = "python_dsl"
    dependency = ""
    accepted_import_roots: frozenset[str] = frozenset()

    @classmethod
    def _import_roots(cls, code: str) -> set[str]:
        tree = ast.parse(code)
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        return roots

    @classmethod
    def _extract_kernel_names(cls, code: str) -> list[str]:
        return []

    @classmethod
    def _artifact_key(cls, code: str, entry_point: str, compiler_options: Dict[str, Any]) -> str:
        payload = {
            "backend": cls.backend_name,
            "code": code,
            "entry_point": entry_point,
            "compiler_options": compiler_options,
            "python": sys.version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        compiler_options = dict(kwargs.get("compiler_options") or {})
        cache_enabled = bool(kwargs.get("enable_compile_artifact_cache", False))

        valid, error = validate_code(code, entry_point)
        if not valid:
            return self._failure(error, device, entry_point)
        try:
            compile(code, "<string>", "exec")
            imported_roots = self._import_roots(code)
        except SyntaxError as exc:
            return self._failure(
                f"Syntax error in {self.backend_name} kernel code: {exc}",
                device,
                entry_point,
            )
        if not imported_roots.intersection(self.accepted_import_roots):
            expected = ", ".join(sorted(self.accepted_import_roots))
            return self._failure(
                f"{self.backend_name} submission must import its runtime ({expected})",
                device,
                entry_point,
            )
        if self.dependency and importlib.util.find_spec(self.dependency) is None:
            return self._failure(
                f"{self.backend_name} backend requires Python package '{self.dependency}'",
                device,
                entry_point,
            )

        kernel_names = self._extract_kernel_names(code)
        cache_key = self._artifact_key(code, entry_point, compiler_options)
        return {
            "compiled": True,
            "error": None,
            "backend": self.backend_name,
            "build_backend": f"{self.backend_name}_jit",
            "artifact_type": "python_jit_source",
            "jit_compile_on_execute": True,
            "code": code,
            "device": str(device),
            "entry_point": entry_point,
            "compiler_options": compiler_options,
            "compile_artifact_cache_enabled": cache_enabled,
            "compile_artifact_cache_hit": False,
            "compile_artifact_cache_key": cache_key,
            "profiling_hints": {
                "custom_kernel_names": kernel_names,
                "language": self.backend_name,
            },
        }

    def _failure(self, error: str, device: Any, entry_point: str) -> Dict[str, Any]:
        return {
            "compiled": False,
            "error": error,
            "backend": self.backend_name,
            "build_backend": f"{self.backend_name}_jit",
            "artifact_type": "python_jit_source",
            "device": str(device),
            "entry_point": entry_point,
        }

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        if not code:
            raise ValueError(f"{self.backend_name} artifact does not contain source code")

        device = self._normalize_device(kwargs.get("device") or artifact.get("device"))
        self._maybe_set_cuda_device(device)
        if self.backend_name == "triton":
            self._maybe_set_triton_env(device)

        try:
            model_cls, tempfile_handle = load_custom_model_with_tempfile(code, entry_point=entry_point)
        except AttributeError as exc:
            raise ValueError(
                f"Failed to load model class '{entry_point}' from {self.backend_name} submission"
            ) from exc

        return {
            "model_cls": model_cls,
            "tempfile_handle": tempfile_handle,
            "context": kwargs.get("context") or {},
            "backend": self.backend_name,
            "entry_point": entry_point,
            "device": device,
            "artifact_type": artifact.get("artifact_type"),
            "profiling_hints": artifact.get("profiling_hints") or {},
        }


class KernelBenchTritonBackend(KernelBenchPythonDslBackend):
    name = "kernelbench.triton"
    backend_name = "triton"
    dependency = "triton"
    accepted_import_roots = frozenset({"triton"})

    @classmethod
    def _extract_kernel_names(cls, code: str) -> list[str]:
        pattern = re.compile(
            r"@(?:triton\.jit|triton\.autotune\([^\n]*\))[^\n]*\n\s*def\s+([A-Za-z_]\w*)\s*\(",
            re.MULTILINE,
        )
        return sorted(set(pattern.findall(code)))


class KernelBenchTileLangBackend(KernelBenchPythonDslBackend):
    name = "kernelbench.tilelang"
    backend_name = "tilelang"
    dependency = "tilelang"
    accepted_import_roots = frozenset({"tilelang"})

    @classmethod
    def _extract_kernel_names(cls, code: str) -> list[str]:
        patterns = (
            re.compile(r"@(?:T|tilelang\.language)\.prim_func\s*\n\s*def\s+([A-Za-z_]\w*)\s*\("),
            re.compile(r"@tilelang\.jit(?:\([^)]*\))?\s*\n\s*def\s+([A-Za-z_]\w*)\s*\("),
        )
        return sorted({name for pattern in patterns for name in pattern.findall(code)})
