"""CUDA-Agent backend implementation for KernelBench.

This backend compiles and executes raw CUDA code using the CUDA-Agent workflow.
It supports both:
1. file-backed sources in a working directory,
2. in-memory sources (dict/str) compiled through ``load_inline``, and
3. TVM-FFI-backed shared libraries compiled through ``tvm_ffi.cpp.build``.
"""

from __future__ import annotations

import ast
import atexit
import ctypes
import hashlib
import inspect
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.utils.cpp_extension as cpp_ext

from kernelgym.backend.base import Backend
from kernelgym.toolkit.kernelbench.loading import graceful_eval_cleanup
from kernelgym.toolkit.validation import precheck_cuda_agent_submission


class CudaAgentBackendSession:
    """Session for a loaded CUDA-Agent backend."""

    def __init__(
        self,
        handle: Dict[str, Any],
        device: torch.device,
    ):
        self.handle = handle
        self.device = device
        self.model_cls = handle.get("model_cls")
        self.work_dir = handle.get("work_dir")
        self.so_path = handle.get("so_path")

    def create_model(self, init_inputs: Any, no_grad: bool = True, synchronize: bool = False):
        """Create a model instance."""
        if self.model_cls is None:
            raise ValueError("Model class not loaded")

        # Move inputs to device
        if isinstance(init_inputs, list):
            init_inputs = [
                x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]
        elif isinstance(init_inputs, dict):
            init_inputs = {
                k: v.cuda(device=self.device) if isinstance(v, torch.Tensor) else v
                for k, v in init_inputs.items()
            }

        if no_grad:
            with torch.no_grad():
                if isinstance(init_inputs, dict):
                    model = self.model_cls(**init_inputs)
                else:
                    model = self.model_cls(*init_inputs)
        else:
            if isinstance(init_inputs, dict):
                model = self.model_cls(**init_inputs)
            else:
                model = self.model_cls(*init_inputs)

        if hasattr(model, "to"):
            model = model.to(self.device)

        if synchronize and self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)

        return model

    def close(self):
        """Close the session and clean up."""
        if self.work_dir and Path(self.work_dir).exists():
            try:
                shutil.rmtree(self.work_dir)
            except Exception:
                pass


class CudaAgentBackend(Backend):
    """Backend for compiling and running raw CUDA code using CUDA-Agent workflow."""

    name = "kernelbench.cuda_agent"

    def __init__(self):
        self._work_dirs: list[Path] = []
        atexit.register(self._cleanup_registered_work_dirs)

    def _cleanup_registered_work_dirs(self) -> None:
        for work_dir in list(self._work_dirs):
            try:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
            except Exception:
                pass
        self._work_dirs.clear()

    @staticmethod
    def _resolve_source_mode(source_mode: Any, has_inline_sources: bool) -> str:
        mode = str(source_mode or "auto").strip().lower()
        if mode not in {"auto", "inline", "files", "tvm_ffi"}:
            raise ValueError(
                f"Unsupported source_mode '{source_mode}'. Expected one of: auto, inline, files, tvm_ffi."
            )
        if mode == "auto":
            return "inline" if has_inline_sources else "files"
        return mode

    def _normalize_device(self, device: Any | None) -> torch.device:
        if device is None:
            return torch.device("cuda:0")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    def _maybe_set_cuda_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass

    def _create_work_dir(self) -> Path:
        """Create a temporary working directory for CUDA compilation."""
        work_dir = Path(tempfile.mkdtemp(prefix="cuda_agent_"))
        self._work_dirs.append(work_dir)
        return work_dir

    def _cache_work_dir(self, cache_key: str) -> Path:
        root = Path(tempfile.gettempdir()) / "kernelgym_cuda_agent_cache"
        root.mkdir(parents=True, exist_ok=True)
        work_dir = root / cache_key
        work_dir.mkdir(parents=True, exist_ok=True)
        if work_dir not in self._work_dirs:
            self._work_dirs.append(work_dir)
        return work_dir

    @staticmethod
    def _artifact_cache_key(
        *,
        code: str,
        entry_point: str,
        resolved_source_mode: str,
        cuda_sources: dict[str, str],
    ) -> str:
        payload = {
            "code": code,
            "entry_point": entry_point,
            "source_mode": resolved_source_mode,
            "cuda_sources": cuda_sources,
        }
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def _setup_cuda_project(self, work_dir: Path, code: str, entry_point: str = "ModelNew"):
        """Set up the CUDA project structure in the working directory.

        Args:
            work_dir: The working directory to set up
            code: The Python model code that uses the CUDA extension
            entry_point: The name of the model class (default: ModelNew)
        """
        # Create kernels directory
        kernels_dir = work_dir / "kernels"
        kernels_dir.mkdir(exist_ok=True)

        # Write the Python model file
        model_file = work_dir / "model_new.py"
        model_file.write_text(code)

        # Create a default binding.cpp
        binding_cpp = work_dir / "binding.cpp"
        binding_cpp_content = '''#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
'''
        binding_cpp.write_text(binding_cpp_content)

        # Create binding_registry.h
        binding_registry_h = work_dir / "binding_registry.h"
        binding_registry_content = self._default_binding_registry_h()
        binding_registry_h.write_text(binding_registry_content)

        binding_registry_cpp = work_dir / "binding_registry.cpp"
        binding_registry_cpp.write_text(self._default_binding_registry_cpp())

    @staticmethod
    def _default_binding_cpp() -> str:
        return '''#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
'''

    @staticmethod
    def _default_binding_registry_h() -> str:
        return '''#pragma once

#include <vector>
#include <functional>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class BindingRegistry {
public:
    using BindingFunction = std::function<void(pybind11::module&)>;

    static BindingRegistry& getInstance();

    void registerBinding(const std::string& name, BindingFunction func);

    void applyBindings(pybind11::module& m);

private:
    std::vector<std::pair<std::string, BindingFunction>> bindings_;
    BindingRegistry() = default;
};

class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name, BindingRegistry::BindingFunction func) {
        BindingRegistry::getInstance().registerBinding(name, func);
    }
};

#define REGISTER_BINDING(name, func) \\
    static BindingRegistrar _registrar_##name(#name, [](pybind11::module& m) { func(m); })
'''

    @staticmethod
    def _default_binding_registry_cpp() -> str:
        return '''#include "binding_registry.h"

BindingRegistry& BindingRegistry::getInstance() {
    static BindingRegistry instance;
    return instance;
}

void BindingRegistry::registerBinding(const std::string& name, BindingFunction func) {
    bindings_.push_back({name, func});
}

void BindingRegistry::applyBindings(pybind11::module& m) {
    for (auto& [name, func] : bindings_) {
        func(m);
    }
}
'''

    @staticmethod
    def _is_header_file(filename: str) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix in {".h", ".hpp", ".hh", ".cuh"}

    @staticmethod
    def _is_cpp_file(filename: str) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix in {".cpp", ".cc", ".cxx"}

    @staticmethod
    def _is_cuda_file(filename: str) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix == ".cu"

    def _normalize_cuda_sources_input(self, cuda_sources: Any) -> dict[str, str]:
        """Normalize user-provided CUDA sources into a filename->content map."""
        if not cuda_sources:
            return {}

        if isinstance(cuda_sources, dict):
            normalized = {}
            for filename, content in cuda_sources.items():
                if not isinstance(filename, str):
                    raise TypeError("cuda_sources dict keys must be file names")
                normalized[filename] = str(content)
            return normalized

        if isinstance(cuda_sources, str):
            stripped = cuda_sources.strip()
            if not stripped:
                return {}
            if stripped.startswith("{"):
                parsed = ast.literal_eval(stripped)
                if not isinstance(parsed, dict):
                    raise TypeError("cuda_sources string dict literal must evaluate to a dict")
                return {str(k): str(v) for k, v in parsed.items()}
            return {"kernels/kernel.cu": cuda_sources}

        raise TypeError("cuda_sources must be a dict[str, str] or a source string")

    def _build_inline_sources(
        self,
        code: str,
        cuda_sources: dict[str, str],
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Build inline cpp/cu sources and header files for load_inline."""
        source_map: dict[str, str] = {
            "binding.cpp": self._default_binding_cpp(),
            "binding_registry.h": self._default_binding_registry_h(),
            "binding_registry.cpp": self._default_binding_registry_cpp(),
        }
        source_map.update(cuda_sources)

        cpp_sources: list[str] = []
        cuda_blobs: list[str] = []
        header_sources: dict[str, str] = {}

        basename_to_header: dict[str, str] = {}
        basename_conflicts: set[str] = set()

        for filename in source_map:
            if self._is_header_file(filename):
                basename = Path(filename).name
                if basename in basename_to_header and basename_to_header[basename] != filename:
                    basename_conflicts.add(basename)
                else:
                    basename_to_header[basename] = filename

        for basename in basename_conflicts:
            basename_to_header.pop(basename, None)

        def _rewrite_includes(content: str) -> str:
            def _replace(match: re.Match[str]) -> str:
                quote = match.group(1)
                include_target = match.group(2)
                normalized = include_target.replace("\\", "/")
                basename = Path(normalized).name

                if include_target in header_sources:
                    replacement = include_target
                elif include_target in source_map and self._is_header_file(include_target):
                    replacement = include_target
                else:
                    replacement = basename_to_header.get(basename)

                if not replacement:
                    return match.group(0)
                return f'#include {quote}{replacement}{quote}'

            return re.sub(r'#include\s+([<"])([^>"]+)[>"]', _replace, content)

        for filename, content in source_map.items():
            if self._is_header_file(filename):
                header_sources[filename] = content
            elif self._is_cuda_file(filename):
                cuda_blobs.append(_rewrite_includes(content))
            elif self._is_cpp_file(filename):
                cpp_sources.append(_rewrite_includes(content))

        if not cpp_sources:
            cpp_sources.append(self._default_binding_cpp())
        if not header_sources:
            header_sources["binding_registry.h"] = self._default_binding_registry_h()

        return cpp_sources, cuda_blobs, header_sources

    def _write_header_sources(self, base_dir: Path, headers: dict[str, str]) -> None:
        for filename, content in headers.items():
            file_path = base_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

    def _write_cuda_sources_to_work_dir(self, work_dir: Path, cuda_sources: dict[str, str]) -> None:
        """Materialize in-memory CUDA/C++ sources into the working directory."""
        for filename, content in cuda_sources.items():
            relative_path = Path(filename)
            if relative_path.parent == Path("."):
                file_path = work_dir / "kernels" / relative_path
            else:
                file_path = work_dir / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            print(f"write file {file_path}\n with content \n {content}")

    def _explicit_source_paths_from_cuda_sources(
        self,
        work_dir: Path,
        cuda_sources: dict[str, str],
    ) -> list[Path]:
        """Resolve only user-provided C++/CUDA source files under the work directory."""
        sources: list[Path] = []
        for filename in cuda_sources:
            relative_path = Path(filename)
            if relative_path.parent == Path("."):
                file_path = work_dir / "kernels" / relative_path
            else:
                file_path = work_dir / relative_path
            if self._is_cpp_file(filename) or self._is_cuda_file(filename):
                sources.append(file_path)
        return sorted(set(sources))

    def _extract_cuda_sources(self, work_dir: Path) -> list[str]:
        """Extract CUDA source files from the working directory.

        Args:
            work_dir: The working directory

        Returns:
            List of source file paths
        """
        kernels_dir = work_dir / "kernels"
        sources = []

        # Find all .cu and .cpp files in root and kernels directory
        root_sources = list(work_dir.glob("*.cu")) + list(work_dir.glob("*.cpp"))
        kernel_sources = []
        if kernels_dir.is_dir():
            kernel_sources = list(kernels_dir.glob("*.cu")) + list(kernels_dir.glob("*.cpp"))

        sources = sorted(set([str(s) for s in root_sources + kernel_sources]))

        return sources

    @staticmethod
    def _find_cuda_lib_path() -> Optional[str]:
        """Find the CUDA library directory for explicit linker flags."""
        nvcc = shutil.which("nvcc")
        if nvcc:
            cuda_home = Path(nvcc).resolve().parent.parent
            for candidate in (
                cuda_home / "targets" / "x86_64-linux" / "lib",
                cuda_home / "lib64",
                cuda_home / "lib",
            ):
                if (candidate / "libcudart.so").exists():
                    return str(candidate)

        for prefix in ("/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-13"):
            for subdir in ("targets/x86_64-linux/lib", "lib64", "lib"):
                candidate = Path(prefix) / subdir
                if (candidate / "libcudart.so").exists():
                    return str(candidate)
        return None

    @staticmethod
    def _classify_ninja_output(output: str) -> str:
        name = Path(output).name
        if name.endswith(".so"):
            return "link"
        if name.endswith(".cuda.o"):
            return "cuda_compile"
        if name.endswith(".o"):
            return "cpp_compile"
        return "other"

    @classmethod
    def _parse_ninja_compile_timing(cls, build_dir: Path) -> Dict[str, Any]:
        """Parse ninja's per-edge build timings from .ninja_log."""
        ninja_log = build_dir / ".ninja_log"
        timing: Dict[str, Any] = {
            "detailed_compile_timing_enabled": True,
            "ninja_log_path": str(ninja_log),
            "ninja_log_found": ninja_log.exists(),
            "ninja_wall_sec": None,
            "cuda_compile_sec": 0.0,
            "cpp_compile_sec": 0.0,
            "link_sec": 0.0,
            "other_sec": 0.0,
            "edge_count": 0,
            "edges": [],
        }
        if not ninja_log.exists():
            return timing

        starts: list[int] = []
        ends: list[int] = []
        try:
            with ninja_log.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue
                    try:
                        start_ms = int(parts[0])
                        end_ms = int(parts[1])
                    except ValueError:
                        continue
                    output = parts[3]
                    category = cls._classify_ninja_output(output)
                    duration_sec = round((end_ms - start_ms) / 1000.0, 6)
                    output_path = build_dir / output
                    edge = {
                        "output": output,
                        "category": category,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_sec": duration_sec,
                        "output_size_bytes": (
                            output_path.stat().st_size if output_path.exists() else None
                        ),
                    }
                    timing["edges"].append(edge)
                    timing[f"{category}_sec"] = round(
                        float(timing.get(f"{category}_sec", 0.0)) + duration_sec, 6
                    )
                    starts.append(start_ms)
                    ends.append(end_ms)
            timing["edge_count"] = len(timing["edges"])
            if starts and ends:
                timing["ninja_wall_sec"] = round((max(ends) - min(starts)) / 1000.0, 6)
        except Exception as exc:
            timing["parse_error"] = str(exc)
        return timing

    @classmethod
    def _new_compile_timing(cls, build_dir: Path) -> Dict[str, Any]:
        if cls._env_flag("DETAILED_COMPILE_TIMING", default=False):
            return cls._parse_ninja_compile_timing(build_dir)
        return {}

    @classmethod
    def _set_compile_timing_detail(
        cls,
        compile_timing: Dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        if cls._env_flag("DETAILED_COMPILE_TIMING", default=False):
            compile_timing[key] = value

    @staticmethod
    def _cuda_build_backend() -> str:
        backend = os.environ.get("CUDA_BUILD_BACKEND", "cpp_extension_load")
        backend = backend.strip().lower().replace("-", "_")
        aliases = {
            "cpp_extension": "cpp_extension_load",
            "load": "cpp_extension_load",
            "ninja": "manual_ninja",
        }
        backend = aliases.get(backend, backend)
        if backend not in {"cpp_extension_load", "manual_ninja"}:
            raise ValueError(
                "CUDA_BUILD_BACKEND must be one of "
                "{cpp_extension_load, manual_ninja}"
            )
        return backend

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _prepare_torch_extension_ldflags() -> list[str]:
        prepare_ldflags = cpp_ext._prepare_ldflags  # type: ignore[attr-defined]
        signature = inspect.signature(prepare_ldflags)
        kwargs = {
            "extra_ldflags": [],
            "with_cuda": True,
            "with_sycl": False,
            "verbose": False,
            "is_standalone": False,
        }
        supported_kwargs = {
            name: value for name, value in kwargs.items() if name in signature.parameters
        }
        return prepare_ldflags(**supported_kwargs)

    @staticmethod
    def _manual_ninja_object_cache_root() -> Path:
        root = os.environ.get("MANUAL_NINJA_OBJECT_CACHE_DIR")
        if root:
            return Path(root)
        return Path(tempfile.gettempdir()) / "kernelgym_manual_ninja_object_cache"

    @staticmethod
    def _manual_ninja_object_cache_index() -> str:
        index = os.environ.get("MANUAL_NINJA_OBJECT_CACHE_INDEX") or os.environ.get(
            "CACHE_INDEX", "fs"
        )
        index = index.strip().lower()
        if index in {"", "memory", "local", "fs", "file", "filesystem"}:
            return "fs"
        if index != "redis":
            raise ValueError(
                "MANUAL_NINJA_OBJECT_CACHE_INDEX must be one of {fs, redis}"
            )
        return index

    @staticmethod
    def _manual_ninja_redis_client() -> Any | None:
        try:
            import redis
        except Exception as exc:
            print(f"[CudaAgent] Redis cache index disabled: redis import failed: {exc}")
            return None
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        db = int(os.environ.get("REDIS_DB", "0"))
        password = os.environ.get("REDIS_PASSWORD") or None
        try:
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            client.ping()
            return client
        except Exception as exc:
            print(f"[CudaAgent] Redis cache index disabled: connection failed: {exc}")
            return None

    @staticmethod
    def _manual_ninja_redis_cache_key(cache_key: str) -> str:
        prefix = os.environ.get("REDIS_KEY_PREFIX", "kernelgym")
        node_id = os.environ.get("NODE_ID") or socket.gethostname()
        return f"{prefix}:manual_ninja_object_cache:{node_id}:{cache_key}"

    @staticmethod
    def _redis_json_get(client: Any | None, key: str) -> dict[str, Any] | None:
        if client is None:
            return None
        try:
            raw = client.get(key)
            if not raw:
                return None
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except Exception as exc:
            print(f"[CudaAgent] Redis cache index get failed for {key}: {exc}")
            return None

    @staticmethod
    def _redis_json_set(
        client: Any | None,
        key: str,
        value: dict[str, Any],
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        if client is None:
            return False
        try:
            return bool(client.set(key, json.dumps(value, sort_keys=True), nx=nx, ex=ex))
        except Exception as exc:
            print(f"[CudaAgent] Redis cache index set failed for {key}: {exc}")
            return False

    @staticmethod
    def _redis_increment_hit(client: Any | None, key: str) -> None:
        if client is None:
            return
        try:
            raw = client.get(key)
            if not raw:
                return
            value = json.loads(raw)
            if not isinstance(value, dict):
                return
            value["hit_count"] = int(value.get("hit_count") or 0) + 1
            value["last_used_at"] = time.time()
            client.set(key, json.dumps(value, sort_keys=True))
        except Exception as exc:
            print(f"[CudaAgent] Redis cache index hit update failed for {key}: {exc}")

    @staticmethod
    def _header_digest(work_dir: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(work_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".h", ".hpp", ".hh", ".cuh"}:
                continue
            try:
                digest.update(str(path.relative_to(work_dir)).encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
            except OSError:
                continue
        return digest.hexdigest()

    @staticmethod
    def _normalize_build_text_for_cache(text: str, ext_name: str) -> str:
        return text.replace(ext_name, "<TORCH_EXTENSION_NAME>")

    def _manual_ninja_cache_key(
        self,
        *,
        build_ninja_text: str,
        ext_name: str,
        object_name: str,
        rule: str,
        source_path: Path,
        header_digest: str,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(self._normalize_build_text_for_cache(build_ninja_text, ext_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(object_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(rule.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
        digest.update(header_digest.encode("utf-8"))
        digest.update(b"\0")
        digest.update(torch.__version__.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(torch.version.cuda).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sys.version.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _source_is_reusable_object(source_path: Path, object_name: str) -> tuple[bool, str | None]:
        if object_name == "binding.o":
            return False, "binding.cpp defines PYBIND11_MODULE/TORCH_EXTENSION_NAME"
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return False, f"source read failed: {exc}"
        if "TORCH_EXTENSION_NAME" in text or "PYBIND11_MODULE" in text:
            return False, "source references module name"
        return True, None

    @staticmethod
    def _ninja_header_and_object_edges(build_ninja_text: str) -> tuple[str, list[dict[str, str]]]:
        lines = build_ninja_text.splitlines()
        first_build_index = len(lines)
        edges: list[dict[str, str]] = []
        for index, line in enumerate(lines):
            if line.startswith("build "):
                first_build_index = min(first_build_index, index)
                match = re.match(r"^build\s+(\S+):\s+(compile|cuda_compile)\s+(.+)$", line)
                if match:
                    output, rule, source = match.groups()
                    edges.append({"output": output, "rule": rule, "source": source.strip()})
        header = "\n".join(lines[:first_build_index]).rstrip() + "\n"
        return header, edges

    def _prepare_manual_ninja_cached_objects(
        self,
        *,
        build_dir: Path,
        work_dir: Path,
        ext_name: str,
    ) -> dict[str, Any]:
        build_ninja_path = build_dir / "build.ninja"
        build_ninja_text = build_ninja_path.read_text(encoding="utf-8")
        header, edges = self._ninja_header_and_object_edges(build_ninja_text)
        header_digest = self._header_digest(work_dir)
        cache_root = self._manual_ninja_object_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_index = self._manual_ninja_object_cache_index()
        redis_client = self._manual_ninja_redis_client() if cache_index == "redis" else None
        if cache_index == "redis" and redis_client is None:
            cache_index = "fs"

        stats: dict[str, Any] = {
            "enabled": True,
            "index": cache_index,
            "root": str(cache_root),
            "hits": 0,
            "misses": 0,
            "objects": [],
            "skipped": [],
            "object_map": {},
            "lookup_wall_sec": 0.0,
            "store_wall_sec": 0.0,
        }
        cache_lookup_started = time.perf_counter()
        for edge in edges:
            output = edge["output"]
            source_path = Path(edge["source"])
            reusable, reason = self._source_is_reusable_object(source_path, output)
            if not reusable:
                stats["skipped"].append(
                    {
                        "object": output,
                        "source": str(source_path),
                        "reason": reason,
                    }
                )
                continue

            cache_key = self._manual_ninja_cache_key(
                build_ninja_text=build_ninja_text,
                ext_name=ext_name,
                object_name=output,
                rule=edge["rule"],
                source_path=source_path,
                header_digest=header_digest,
            )
            cache_dir = cache_root / cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_object = cache_dir / output
            lock_path = cache_dir / ".lock"
            redis_key = self._manual_ninja_redis_cache_key(cache_key)
            object_info = {
                "object": output,
                "source": str(source_path),
                "cache_key": cache_key,
                "cache_path": str(cache_object),
                "local_object": str(build_dir / output),
                "index": cache_index,
            }
            redis_entry = self._redis_json_get(redis_client, redis_key) if cache_index == "redis" else None
            if redis_entry and redis_entry.get("status") == "ready":
                indexed_path = Path(str(redis_entry.get("object_path") or ""))
                if indexed_path.exists():
                    stats["hits"] += 1
                    object_info["cache_status"] = "hit"
                    object_info["index_status"] = "redis_ready"
                    object_info["cache_path"] = str(indexed_path)
                    stats["object_map"][output] = str(indexed_path)
                    self._redis_increment_hit(redis_client, redis_key)
                else:
                    stats["misses"] += 1
                    object_info["cache_status"] = "miss_pending"
                    object_info["index_status"] = "redis_stale_missing_file"
                    object_info["lock_path"] = str(lock_path)
            elif cache_object.exists():
                stats["hits"] += 1
                object_info["cache_status"] = "hit"
                object_info["index_status"] = "fs_ready"
                stats["object_map"][output] = str(cache_object)
                if cache_index == "redis":
                    self._redis_json_set(
                        redis_client,
                        redis_key,
                        {
                            "status": "ready",
                            "object_path": str(cache_object),
                            "object_name": output,
                            "cache_key": cache_key,
                            "node_id": os.environ.get("NODE_ID") or socket.gethostname(),
                            "created_at": time.time(),
                            "last_used_at": time.time(),
                            "hit_count": 1,
                            "size_bytes": cache_object.stat().st_size,
                        },
                    )
            else:
                stats["misses"] += 1
                object_info["cache_status"] = "miss_pending"
                object_info["index_status"] = (
                    str(redis_entry.get("status")) if redis_entry else "missing"
                )
                object_info["lock_path"] = str(lock_path)
                if cache_index == "redis":
                    self._redis_json_set(
                        redis_client,
                        redis_key,
                        {
                            "status": "building",
                            "object_path": str(cache_object),
                            "object_name": output,
                            "cache_key": cache_key,
                            "node_id": os.environ.get("NODE_ID") or socket.gethostname(),
                            "created_at": time.time(),
                            "builder_pid": os.getpid(),
                        },
                        nx=True,
                        ex=600,
                    )
            stats["objects"].append(object_info)
        stats["lookup_wall_sec"] = round(time.perf_counter() - cache_lookup_started, 6)
        return stats

    @staticmethod
    def _store_manual_ninja_cached_objects(
        object_cache_stats: dict[str, Any] | None,
    ) -> None:
        if not object_cache_stats:
            return
        cache_index = str(object_cache_stats.get("index") or "fs")
        redis_client = CudaAgentBackend._manual_ninja_redis_client() if cache_index == "redis" else None
        store_started = time.perf_counter()
        for object_info in object_cache_stats.get("objects") or []:
            if object_info.get("cache_status") != "miss_pending":
                continue
            local_object = Path(str(object_info.get("local_object") or ""))
            cache_object = Path(str(object_info.get("cache_path") or ""))
            lock_path = Path(str(object_info.get("lock_path") or cache_object.parent / ".lock"))
            redis_key = CudaAgentBackend._manual_ninja_redis_cache_key(
                str(object_info.get("cache_key") or "")
            )
            if not local_object.exists():
                object_info["cache_status"] = "miss_not_built"
                if cache_index == "redis":
                    CudaAgentBackend._redis_json_set(
                        redis_client,
                        redis_key,
                        {
                            "status": "failed",
                            "object_path": str(cache_object),
                            "object_name": object_info.get("object"),
                            "cache_key": object_info.get("cache_key"),
                            "node_id": os.environ.get("NODE_ID") or socket.gethostname(),
                            "updated_at": time.time(),
                            "reason": "local_object_not_built",
                        },
                        ex=600,
                    )
                continue
            cache_object.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("w") as lock_handle:
                try:
                    import fcntl

                    fcntl.flock(lock_handle, fcntl.LOCK_EX)
                except Exception:
                    pass
                if cache_object.exists():
                    object_info["cache_status"] = "miss_lost_race"
                else:
                    temp_object = cache_object.with_suffix(cache_object.suffix + ".tmp")
                    shutil.copy2(local_object, temp_object)
                    temp_object.replace(cache_object)
                    object_info["cache_status"] = "miss_stored"
                if cache_index == "redis" and cache_object.exists():
                    CudaAgentBackend._redis_json_set(
                        redis_client,
                        redis_key,
                        {
                            "status": "ready",
                            "object_path": str(cache_object),
                            "object_name": object_info.get("object"),
                            "cache_key": object_info.get("cache_key"),
                            "node_id": os.environ.get("NODE_ID") or socket.gethostname(),
                            "created_at": time.time(),
                            "last_used_at": time.time(),
                            "hit_count": 0,
                            "size_bytes": cache_object.stat().st_size,
                        },
                    )
                try:
                    import fcntl

                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
                except Exception:
                    pass
        object_cache_stats["store_wall_sec"] = round(time.perf_counter() - store_started, 6)

    @staticmethod
    def _rewrite_manual_ninja_for_cached_objects(build_dir: Path, object_map: dict[str, str]) -> None:
        if not object_map:
            return
        build_ninja_path = build_dir / "build.ninja"
        rewritten_lines: list[str] = []
        for line in build_ninja_path.read_text(encoding="utf-8").splitlines():
            skip_line = False
            for object_name in object_map:
                if re.match(rf"^build\s+{re.escape(object_name)}:\s+", line):
                    skip_line = True
                    break
            if skip_line:
                continue
            if line.startswith("build ") and ": link " in line:
                for object_name, cache_path in object_map.items():
                    line = re.sub(
                        rf"(?<!\S){re.escape(object_name)}(?!\S)",
                        cache_path,
                        line,
                    )
            rewritten_lines.append(line)
        build_ninja_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

    def _compile_cuda(self, work_dir: Path, sources: list[str]) -> Dict[str, Any]:
        """Compile CUDA sources using torch.utils.cpp_extension.

        Args:
            work_dir: The working directory
            sources: List of source file paths

        Returns:
            Dictionary with compilation results
        """
        if not sources:
            return {
                "compiled": False,
                "error": "No CUDA source files found (*.cu, *.cpp)",
            }

        build_dir = work_dir / "build" / "forced_compile"
        output_so = work_dir / "cuda_extension.so"
        build_dir.mkdir(parents=True, exist_ok=True)

        # # 2. 【关键修复】清理全局 PyTorch 扩展缓存，防止 _v1, _v2 后缀产生
        # # torch 默认缓存位置: ~/.cache/torch_extensions/cuda_extension
        # global_cache_dir = Path.home() / ".cache" / "torch_extensions" / "cuda_extension"
        # if global_cache_dir.exists():
        #     try:
        #         shutil.rmtree(global_cache_dir)
        #         print(f"[CudaAgent] Cleaned global cache: {global_cache_dir}")
        #     except Exception as e:
        #         print(f"[CudaAgent] Warning: Could not clean global cache: {e}")

        build_backend = self._cuda_build_backend()
        if build_backend == "manual_ninja":
            return self._compile_cuda_manual_ninja(work_dir, sources)

        try:
            # Use torch.utils.cpp_extension to compile
            # Generate a unique extension name based on the work_dir to avoid torch cache conflicts
            # and file lock contentions during multiprocessing.
            ext_name = work_dir.name.replace("-", "_")

            print(f"Compiling CUDA sources in {str(build_dir)} with name {ext_name}")

            load_started = time.perf_counter()
            module = cpp_ext.load(
                name=ext_name,
                sources=sources,
                build_directory=str(build_dir),
                verbose=False,
                with_cuda=True,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
            load_wall_sec = time.perf_counter() - load_started
            compile_timing = self._new_compile_timing(build_dir)
            self._set_compile_timing_detail(
                compile_timing,
                "build_backend",
                "cpp_extension_load",
            )
            self._set_compile_timing_detail(
                compile_timing,
                "cpp_extension_load_wall_sec",
                round(load_wall_sec, 6),
            )
            # Get actual module name and file path
            module_name = module.__name__.split('.')[-1]
            if hasattr(module, '__file__'):
                built_so = Path(module.__file__)
            else:
                # Fallback to default behavior if __file__ is missing
                built_so = build_dir / "cuda_extension.so"
            
            print(f"Built so path: {built_so}")
            print(f"Output so path: {output_so}")
            # Copy the compiled .so file to the work directory
            # We use the original name "cuda_extension.so" for the output file
            # to maintain consistency, but we return the actual module name
            if built_so.exists():
                copy_started = time.perf_counter()
                shutil.copy2(built_so, output_so)
                copy_sec = time.perf_counter() - copy_started
                compile_timing["total_wall_sec"] = round(load_wall_sec + copy_sec, 6)
                self._set_compile_timing_detail(
                    compile_timing,
                    "copy_so_sec",
                    round(copy_sec, 6),
                )
                self._set_compile_timing_detail(
                    compile_timing,
                    "built_so_size_bytes",
                    built_so.stat().st_size,
                )
                self._set_compile_timing_detail(
                    compile_timing,
                    "output_so_size_bytes",
                    output_so.stat().st_size,
                )
                return {
                    "compiled": True,
                    "so_path": str(output_so),
                    "module_name": module_name,
                    "error": None,
                    "build_backend": "cpp_extension_load",
                    "compile_timing": compile_timing,
                }
            else:
                compile_timing["total_wall_sec"] = round(load_wall_sec, 6)
                self._set_compile_timing_detail(compile_timing, "copy_so_sec", None)
                return {
                    "compiled": False,
                    "error": "Compilation finished but .so file was not generated",
                    "compile_timing": compile_timing,
                }

        except Exception as exc:
            compile_timing = self._new_compile_timing(build_dir)
            self._set_compile_timing_detail(
                compile_timing,
                "build_backend",
                "cpp_extension_load",
            )
            if "load_started" in locals():
                load_wall_sec = time.perf_counter() - load_started
                compile_timing["total_wall_sec"] = round(load_wall_sec, 6)
                self._set_compile_timing_detail(
                    compile_timing,
                    "cpp_extension_load_wall_sec",
                    round(load_wall_sec, 6),
                )
            print(f"[Error] compiled fail:\n {str(exc)}")
            return {
                "compiled": False,
                "error": str(exc),
                "build_backend": "cpp_extension_load",
                "compile_timing": compile_timing,
            }

    def _compile_cuda_manual_ninja(self, work_dir: Path, sources: list[str]) -> Dict[str, Any]:
        """Compile CUDA sources by explicitly invoking PyTorch's ninja build helper."""
        build_dir = work_dir / "build" / "forced_compile"
        output_so = work_dir / "cuda_extension.so"
        build_dir.mkdir(parents=True, exist_ok=True)

        ext_name = work_dir.name.replace("-", "_")
        print(f"Compiling CUDA sources via manual ninja in {str(build_dir)} with name {ext_name}")

        try:
            build_started = time.perf_counter()
            extra_ldflags = self._prepare_torch_extension_ldflags()
            build_file_path = build_dir / "build.ninja"
            cpp_ext._write_ninja_file_to_build_library(  # type: ignore[attr-defined]
                path=str(build_file_path),
                name=ext_name,
                sources=sources,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                extra_sycl_cflags=[],
                extra_ldflags=extra_ldflags,
                extra_include_paths=[],
                with_cuda=True,
                with_sycl=False,
                is_standalone=False,
            )
            object_cache_stats: dict[str, Any] | None = None
            if self._env_flag("MANUAL_NINJA_OBJECT_CACHE", default=False):
                object_cache_stats = self._prepare_manual_ninja_cached_objects(
                    build_dir=build_dir,
                    work_dir=work_dir,
                    ext_name=ext_name,
                )
                self._rewrite_manual_ninja_for_cached_objects(
                    build_dir,
                    object_cache_stats.get("object_map", {}),
                )
            cpp_ext._run_ninja_build(  # type: ignore[attr-defined]
                str(build_dir),
                False,
                error_prefix=f"Error building extension '{ext_name}'",
            )
            self._store_manual_ninja_cached_objects(object_cache_stats)
            ninja_helper_wall_sec = time.perf_counter() - build_started

            lib_ext = getattr(cpp_ext, "LIB_EXT", ".so")
            built_so = build_dir / f"{ext_name}{lib_ext}"
            if not built_so.exists():
                candidates = sorted(build_dir.glob(f"{ext_name}*.so"))
                built_so = candidates[0] if candidates else built_so

            compile_timing = self._new_compile_timing(build_dir)
            self._set_compile_timing_detail(compile_timing, "build_backend", "manual_ninja")
            self._set_compile_timing_detail(
                compile_timing,
                "manual_ninja_build_wall_sec",
                round(ninja_helper_wall_sec, 6),
            )
            if object_cache_stats is not None:
                self._set_compile_timing_detail(
                    compile_timing,
                    "manual_ninja_object_cache",
                    {
                        key: value
                        for key, value in object_cache_stats.items()
                        if key != "object_map"
                    },
                )

            if not built_so.exists():
                compile_timing["total_wall_sec"] = round(ninja_helper_wall_sec, 6)
                self._set_compile_timing_detail(compile_timing, "copy_so_sec", None)
                return {
                    "compiled": False,
                    "error": "Manual ninja build finished but .so file was not generated",
                    "build_backend": "manual_ninja",
                    "compile_timing": compile_timing,
                }

            import_started = time.perf_counter()
            cpp_ext._import_module_from_library(  # type: ignore[attr-defined]
                ext_name,
                str(build_dir),
                is_python_module=True,
            )
            import_wall_sec = time.perf_counter() - import_started

            copy_started = time.perf_counter()
            shutil.copy2(built_so, output_so)
            copy_sec = time.perf_counter() - copy_started
            compile_timing["total_wall_sec"] = round(
                ninja_helper_wall_sec + import_wall_sec + copy_sec,
                6,
            )
            self._set_compile_timing_detail(
                compile_timing,
                "manual_ninja_import_wall_sec",
                round(import_wall_sec, 6),
            )
            self._set_compile_timing_detail(
                compile_timing,
                "copy_so_sec",
                round(copy_sec, 6),
            )
            self._set_compile_timing_detail(
                compile_timing,
                "built_so_size_bytes",
                built_so.stat().st_size,
            )
            self._set_compile_timing_detail(
                compile_timing,
                "output_so_size_bytes",
                output_so.stat().st_size,
            )

            return {
                "compiled": True,
                "so_path": str(output_so),
                "module_name": ext_name,
                "error": None,
                "build_backend": "manual_ninja",
                "compile_timing": compile_timing,
            }
        except Exception as exc:
            if "object_cache_stats" in locals() and object_cache_stats is not None:
                self._store_manual_ninja_cached_objects(object_cache_stats)
            compile_timing = self._new_compile_timing(build_dir)
            self._set_compile_timing_detail(compile_timing, "build_backend", "manual_ninja")
            if "object_cache_stats" in locals() and object_cache_stats is not None:
                self._set_compile_timing_detail(
                    compile_timing,
                    "manual_ninja_object_cache",
                    {
                        key: value
                        for key, value in object_cache_stats.items()
                        if key != "object_map"
                    },
                )
            if "build_started" in locals():
                build_wall_sec = time.perf_counter() - build_started
                compile_timing["total_wall_sec"] = round(build_wall_sec, 6)
                self._set_compile_timing_detail(
                    compile_timing,
                    "manual_ninja_build_wall_sec",
                    round(build_wall_sec, 6),
                )
            print(f"[Error] manual ninja compile fail:\n {str(exc)}")
            return {
                "compiled": False,
                "error": str(exc),
                "build_backend": "manual_ninja",
                "compile_timing": compile_timing,
            }

    def _compile_cuda_inline(
        self,
        work_dir: Path,
        cpp_sources: list[str],
        cuda_sources: list[str],
        headers: dict[str, str],
    ) -> Dict[str, Any]:
        """Compile CUDA sources from in-memory strings using load_inline."""
        if not cuda_sources:
            return {
                "compiled": False,
                "error": "No CUDA source blobs found for inline compilation",
            }

        build_dir = work_dir / "build" / "forced_compile"
        output_so = work_dir / "cuda_extension.so"

        build_dir.mkdir(parents=True, exist_ok=True)

        self._write_header_sources(build_dir, headers)

        try:
            ext_name = work_dir.name.replace("-", "_")
            print(f"Compiling inline CUDA sources in {str(build_dir)} with name {ext_name}")

            load_started = time.perf_counter()
            module = cpp_ext.load_inline(
                name=ext_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                build_directory=str(build_dir),
                extra_include_paths=[str(build_dir)],
                verbose=False,
                with_cuda=True,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
            load_wall_sec = time.perf_counter() - load_started
            compile_timing = self._new_compile_timing(build_dir)
            self._set_compile_timing_detail(
                compile_timing,
                "cpp_extension_load_wall_sec",
                round(load_wall_sec, 6),
            )

            module_name = module.__name__.split(".")[-1]
            if hasattr(module, "__file__"):
                built_so = Path(module.__file__)
            else:
                built_so = build_dir / "cuda_extension.so"

            print(f"Built so path: {built_so}")
            print(f"Output so path: {output_so}")
            if built_so.exists():
                copy_started = time.perf_counter()
                shutil.copy2(built_so, output_so)
                copy_sec = time.perf_counter() - copy_started
                compile_timing["total_wall_sec"] = round(load_wall_sec + copy_sec, 6)
                self._set_compile_timing_detail(
                    compile_timing,
                    "copy_so_sec",
                    round(copy_sec, 6),
                )
                self._set_compile_timing_detail(
                    compile_timing,
                    "built_so_size_bytes",
                    built_so.stat().st_size,
                )
                self._set_compile_timing_detail(
                    compile_timing,
                    "output_so_size_bytes",
                    output_so.stat().st_size,
                )
                return {
                    "compiled": True,
                    "so_path": str(output_so),
                    "module_name": module_name,
                    "error": None,
                    "compile_mode": "inline",
                    "compile_timing": compile_timing,
                }
            compile_timing["total_wall_sec"] = round(load_wall_sec, 6)
            self._set_compile_timing_detail(compile_timing, "copy_so_sec", None)
            return {
                "compiled": False,
                "error": "Inline compilation finished but .so file was not generated",
                "compile_timing": compile_timing,
            }
        except Exception as exc:
            compile_timing = self._new_compile_timing(build_dir)
            if "load_started" in locals():
                load_wall_sec = time.perf_counter() - load_started
                compile_timing["total_wall_sec"] = round(load_wall_sec, 6)
                self._set_compile_timing_detail(
                    compile_timing,
                    "cpp_extension_load_wall_sec",
                    round(load_wall_sec, 6),
                )
            return {
                "compiled": False,
                "error": str(exc),
                "compile_timing": compile_timing,
            }

    def _compile_cuda_tvm_ffi(self, work_dir: Path, source_paths: list[Path]) -> Dict[str, Any]:
        """Compile CUDA/C++ sources through tvm_ffi.cpp.build."""
        if not source_paths:
            return {
                "compiled": False,
                "error": "No C++/CUDA source files found for tvm_ffi compilation",
            }

        try:
            import tvm_ffi.cpp
        except ImportError as exc:
            return {
                "compiled": False,
                "error": f"tvm_ffi is not available: {exc}",
            }

        build_dir = work_dir / "build" / "tvm_ffi"
        build_dir.mkdir(parents=True, exist_ok=True)

        cpp_files: list[str] = []
        cuda_files: list[str] = []
        for source_path in source_paths:
            if self._is_cpp_file(source_path.name):
                cpp_files.append(str(source_path))
            elif self._is_cuda_file(source_path.name):
                cuda_files.append(str(source_path))

        if not cpp_files and not cuda_files:
            return {
                "compiled": False,
                "error": "tvm_ffi compilation requires at least one .cpp/.cu source file",
            }

        extra_include_paths = [str(work_dir), str(work_dir / "kernels")]
        extra_ldflags = ["-lcuda", "-lcublas"] if cuda_files else []
        cuda_lib_path = self._find_cuda_lib_path()
        if cuda_lib_path:
            extra_ldflags.insert(0, f"-L{cuda_lib_path}")

        ext_name = work_dir.name.replace("-", "_")

        try:
            output_lib_path = tvm_ffi.cpp.build(
                name=ext_name,
                cpp_files=cpp_files or None,
                cuda_files=cuda_files or None,
                extra_include_paths=extra_include_paths,
                extra_ldflags=extra_ldflags or None,
                build_directory=str(build_dir),
            )
            _probe = ctypes.CDLL(output_lib_path, mode=os.RTLD_NOW)
            del _probe
            return {
                "compiled": True,
                "so_path": str(output_lib_path),
                "module_name": ext_name,
                "error": None,
                "compile_mode": "tvm_ffi",
                "module_loader": "tvm_ffi",
            }
        except Exception as exc:
            return {
                "compiled": False,
                "error": str(exc),
            }

    def _parse_cuda_sources_from_code(self, code: str) -> tuple[dict[str, str], str]:
        """Parse CUDA sources embedded in the code string.

        Supported embedded format:
        ### CUDA_KERNELS
        ```cpp
        ...
        ```
        ### APPLY_BINDINGS
        ```cpp
        ...
        ```
        ### MODEL_NEW
        ```python
        ...
        ```
        
        Args:
            code: The code string that may contain embedded CUDA sources
            
        Returns:
            Tuple of (cuda_sources dict, python_code without cuda sources)
        """
        cuda_sources = {}
        python_code = code

        cuda_kernels_match = re.search(
            r"###\s*CUDA_KERNELS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        apply_bindings_match = re.search(
            r"###\s*APPLY_BINDINGS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        model_new_match = re.search(
            r"###\s*MODEL_NEW\s*```(?:python|py)?\s*\n(.*?)```",
            code,
            re.DOTALL | re.IGNORECASE,
        )

        if cuda_kernels_match or apply_bindings_match or model_new_match:
            if cuda_kernels_match:
                cuda_sources["kernels/generated.cu"] = cuda_kernels_match.group(1).strip()
            if apply_bindings_match:
                cuda_sources["kernels/generated_binding.cpp"] = apply_bindings_match.group(1).strip()
            if model_new_match:
                python_code = model_new_match.group(1).strip()
            return cuda_sources, python_code.strip()
        return cuda_sources, python_code.strip()

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        """Compile CUDA code.

        Args:
            code: The Python model code that imports and uses cuda_extension.
                  Can also contain embedded CUDA sources in the format:
                  ### CUDA_SOURCES ###
                  {'filename.cu': 'content', ...}
                  ### END_CUDA_SOURCES ###
            **kwargs: Additional arguments including:
                - device: The CUDA device to use
                - entry_point: The name of the model class (default: ModelNew)
                - work_dir: Optional working directory (created if not provided)
                - cuda_sources: Dict of {filename: content} for CUDA source files
                - source_mode: One of {"auto", "inline", "files", "tvm_ffi"}.
                  "inline" forces in-memory compilation from cuda_sources/embedded sources,
                  "files" forces filesystem-backed compilation from files under work_dir,
                  "tvm_ffi" forces framework-agnostic shared library compilation
                  from user-provided source files via tvm_ffi.cpp.build,
                  and "auto" chooses inline when in-memory sources are provided.

        Returns:
            Dictionary with compilation results
        """
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        work_dir = kwargs.get("work_dir")
        cuda_sources = kwargs.get("cuda_sources", {})
        source_mode = kwargs.get("source_mode", "files")
        enable_compile_artifact_cache = bool(
            kwargs.get("enable_compile_artifact_cache", False)
        )
        precheck_info = {}
        print(f"kwargs:\n{kwargs}")
        # print(f"code:\n{code}")
        # Parse CUDA sources from code if embedded
        embedded_sources, python_code = self._parse_cuda_sources_from_code(code)
        # print(f"embedded_sources:\n{embedded_sources}")
        # print(f"python_code:\n{python_code}")
        if embedded_sources:
            normalized_sources = self._normalize_cuda_sources_input(cuda_sources)
            normalized_sources.update(embedded_sources)
            cuda_sources = normalized_sources
            code = python_code
        else:
            cuda_sources = self._normalize_cuda_sources_input(cuda_sources)
        print(f"cuda_sources:\n{cuda_sources}")
        try:
            resolved_source_mode = self._resolve_source_mode(source_mode, bool(cuda_sources))
        except ValueError as exc:
            return {
                "compiled": False,
                "error": str(exc),
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
                "compile_artifact_cache_enabled": enable_compile_artifact_cache,
            }

        precheck_error, _precheck_error_code, precheck_info = precheck_cuda_agent_submission(
            code,
            cuda_sources,
            entry_point=entry_point,
            source_mode=resolved_source_mode,
        )
        if not precheck_info.get("passed", False):
            return {
                "compiled": False,
                "error": precheck_error,
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
                "source_mode": resolved_source_mode,
                "precheck": precheck_info,
                "compile_artifact_cache_enabled": enable_compile_artifact_cache,
            }

        # Create working directory
        cache_key = self._artifact_cache_key(
            code=code,
            entry_point=entry_point,
            resolved_source_mode=resolved_source_mode,
            cuda_sources=cuda_sources,
        )
        if work_dir is None and enable_compile_artifact_cache:
            work_dir = self._cache_work_dir(cache_key)
        elif work_dir is None:
            work_dir = self._create_work_dir()
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        # Set up the CUDA project structure
        self._setup_cuda_project(work_dir, code, entry_point)
        compile_mode = "filesystem"
        module_loader = "python_extension"
        # if resolved_source_mode == "inline":
        #     if not cuda_sources:
        #         return {
        #             "compiled": False,
        #             "error": "source_mode='inline' requires cuda_sources or embedded CUDA_SOURCES.",
        #             "device": str(device),
        #             "entry_point": entry_point,
        #             "backend": "cuda_agent",
        #             "work_dir": str(work_dir),
        #         }
        #     cpp_sources, cuda_blobs, header_sources = self._build_inline_sources(code, cuda_sources)
        #     result = self._compile_cuda_inline(work_dir, cpp_sources, cuda_blobs, header_sources)
        #     compile_mode = result.get("compile_mode", "inline")
        if resolved_source_mode == "tvm_ffi":
            if not cuda_sources:
                return {
                    "compiled": False,
                    "error": "source_mode='tvm_ffi' requires cuda_sources or embedded CUDA_SOURCES.",
                    "device": str(device),
                    "entry_point": entry_point,
                    "backend": "cuda_agent",
                    "work_dir": str(work_dir),
                    "compile_artifact_cache_enabled": enable_compile_artifact_cache,
                }
            self._write_cuda_sources_to_work_dir(work_dir, cuda_sources)
            source_paths = self._explicit_source_paths_from_cuda_sources(work_dir, cuda_sources)
            result = self._compile_cuda_tvm_ffi(work_dir, source_paths)
            compile_mode = result.get("compile_mode", "tvm_ffi")
            module_loader = result.get("module_loader", "tvm_ffi")
        else:
            if cuda_sources:
                self._write_cuda_sources_to_work_dir(work_dir, cuda_sources)
            sources = self._extract_cuda_sources(work_dir)
            if not sources:
                return {
                    "compiled": False,
                    "error": "No CUDA source files found. Please provide cuda_sources with .cu files.",
                    "device": str(device),
                    "entry_point": entry_point,
                    "backend": "cuda_agent",
                    "work_dir": str(work_dir),
                    "compile_artifact_cache_enabled": enable_compile_artifact_cache,
                }
            result = self._compile_cuda(work_dir, sources)

        artifact = {
            "compiled": result["compiled"],
            "error": result.get("error"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": "cuda_agent",
            "work_dir": str(work_dir),
            "so_path": result.get("so_path"),
            "module_name": result.get("module_name", "cuda_extension"),
            "code": code,
            "compile_mode": compile_mode,
            "module_loader": module_loader,
            "build_backend": result.get("build_backend", self._cuda_build_backend()),
            "source_mode": resolved_source_mode,
            "precheck": precheck_info,
            "compile_artifact_cache_enabled": enable_compile_artifact_cache,
        }
        if "compile_timing" in result:
            artifact["compile_timing"] = result["compile_timing"]
        if enable_compile_artifact_cache:
            artifact["cache_scope"] = "worker_process"
            artifact["cache_key"] = cache_key
        # print(f"[DEBUG] artifact:\n{artifact}")
        return artifact

    @staticmethod
    def _make_tvm_ffi_python_module(tvm_ffi_module: Any) -> types.ModuleType:
        """Wrap a TVM-FFI module so model code can keep importing cuda_extension."""
        shim = types.ModuleType("cuda_extension")
        shim._tvm_ffi_module = tvm_ffi_module  # type: ignore[attr-defined]

        def _module_getattr(name: str) -> Any:
            return getattr(tvm_ffi_module, name)

        def _module_dir() -> list[str]:
            names = set(shim.__dict__.keys())
            try:
                names.update(dir(tvm_ffi_module))
            except Exception:
                pass
            return sorted(names)

        shim.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        shim.__dir__ = _module_dir  # type: ignore[attr-defined]
        return shim

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        """Load the compiled CUDA extension and model.

        Args:
            artifact: The compilation artifact from compile()
            **kwargs: Additional arguments including:
                - device: The CUDA device to use
                - context: The execution context

        Returns:
            Handle containing the loaded model class and related info
        """
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        work_dir = artifact.get("work_dir")
        so_path = artifact.get("so_path")
        module_name = artifact.get("module_name", "cuda_extension")
        module_loader = artifact.get("module_loader", "python_extension")
        context = kwargs.get("context") or {}

        if not code:
            raise ValueError("CudaAgentBackend.load requires kernel code in artifact")

        if not so_path or not Path(so_path).exists():
            raise ValueError(f"Compiled shared library not found: {so_path}")

        device = self._normalize_device(kwargs.get("device") or artifact.get("device"))
        self._maybe_set_cuda_device(device)

        work_dir_path = Path(work_dir)
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        # Import the model module
        import importlib.util

        model_file = work_dir_path / "model_new.py"
        spec = importlib.util.spec_from_file_location("cuda_agent_model", str(model_file))
        model_module = importlib.util.module_from_spec(spec)

        # Clean up old cuda_extension from sys.modules to avoid conflicts
        if "cuda_extension" in sys.modules:
            del sys.modules["cuda_extension"]
        if module_loader == "tvm_ffi":
            import tvm_ffi

            tvm_ffi_module = tvm_ffi.load_module(str(so_path))
            cuda_ext_module = self._make_tvm_ffi_python_module(tvm_ffi_module)
        else:
            import importlib.machinery

            # We must load using the actual module_name (e.g. cuda_extension_v1) so PyInit_xxx matches
            loader = importlib.machinery.ExtensionFileLoader(module_name, so_path)
            cuda_ext_module = loader.load_module()
        # Inject into sys.modules under the name the model expects
        sys.modules["cuda_extension"] = cuda_ext_module

        # Now load the model
        spec.loader.exec_module(model_module)

        model_cls = getattr(model_module, entry_point)

        if model_cls is None:
            raise ValueError(f"Failed to load model class '{entry_point}' from code")

        return {
            "model_cls": model_cls,
            "work_dir": work_dir,
            "so_path": so_path,
            "context": context,
            "backend": "cuda_agent",
            "entry_point": entry_point,
            "device": device,
            "module_loader": module_loader,
            "tempfile_handle": None,
        }

    def create_model(self, handle: Any, init_inputs: Any, **kwargs: Any) -> Any:
        """Create an instance of the model.
        
        Args:
            handle: Handle returned by load()
            init_inputs: List of inputs for model initialization
            kwargs: Additional arguments
            
        Returns:
            Instantiated PyTorch model
        """
        if not isinstance(handle, dict) or "model_cls" not in handle:
            raise ValueError("CudaAgentBackend.create_model expects a handle from load()")
            
        device = self._normalize_device(kwargs.get("device") or handle.get("device"))
        session = CudaAgentBackendSession(handle, device)
        
        no_grad = kwargs.get("no_grad", True)
        synchronize = kwargs.get("synchronize", False)
        
        return session.create_model(init_inputs, no_grad=no_grad, synchronize=synchronize)

    def open_session(self, handle: Dict[str, Any], **kwargs: Any) -> CudaAgentBackendSession:
        """Open a session for the loaded backend.

        Args:
            handle: The handle from load()
            **kwargs: Additional arguments including:
                - device: The CUDA device to use

        Returns:
            CudaAgentBackendSession instance
        """
        device = self._normalize_device(kwargs.get("device") or handle.get("device"))
        return CudaAgentBackendSession(handle, device)

    def run(self, handle: Any, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Execute the model and return runtime metrics.

        Args:
            handle: The handle from load()
            inputs: Dictionary containing inputs for the model
            **kwargs: Additional arguments

        Returns:
            Dictionary with execution results
        """
        if not isinstance(handle, dict) or "model_cls" not in handle:
            raise ValueError("CudaAgentBackend.run expects a handle from load()")

        device = self._normalize_device(kwargs.get("device") or handle.get("device"))
        self._maybe_set_cuda_device(device)

        init_inputs = inputs.get("init_inputs", inputs.get("inputs", []))
        run_inputs = inputs.get("inputs", init_inputs)

        # Move inputs to device
        if isinstance(run_inputs, list):
            run_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in run_inputs
            ]
        elif isinstance(run_inputs, dict):
            run_inputs = {
                k: v.cuda(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in run_inputs.items()
            }

        # Create model
        model_cls = handle["model_cls"]
        if isinstance(init_inputs, dict):
            model = model_cls(**init_inputs)
        else:
            model = model_cls(*init_inputs)

        if hasattr(model, "to"):
            model = model.to(device)

        # Run inference
        with torch.no_grad():
            output = (
                model(**run_inputs)
                if isinstance(run_inputs, dict)
                else model(*run_inputs)
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

        return {"output": output}

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        """Clean up resources.

        Args:
            handle: The handle from load()
            **kwargs: Additional arguments
        """
        if not isinstance(handle, dict):
            return

        work_dir = handle.get("work_dir")
        if handle.get("cache_scope") == "worker_process":
            return
        if work_dir and Path(work_dir).exists():
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

    def close(self, handle: Any, **kwargs: Any) -> None:
        """Close the backend and clean up."""
        self.cleanup(handle, **kwargs)
