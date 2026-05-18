"""CUDA-Agent-specific KernelBench backend implementation."""

from __future__ import annotations

import ast
import hashlib
import inspect
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from kernelgym.toolkit.validation import precheck_cuda_agent_submission

from .base import KernelBenchBackendBase
from kernelgym.toolkit.kernelbench.binding_detection import strip_think_blocks


_CUDA_AGENT_DEFAULT_TMPDIR = "/dev/shm/kernelgym/work/cuda_agent"
_CUDA_AGENT_MIN_TMPDIR_FREE_BYTES = 512 * 1024 * 1024
_NVCC_THREADS_ENV = "KERNELGYM_NVCC_THREADS"
_CUDA_AGENT_DEFAULT_NVCC_THREADS = "4"
_CUDA_AGENT_FAST_RW_ROOT = Path("/dev/shm")
_CUDA_AGENT_OBJECT_CACHE_ENV = "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE"
_CUDA_AGENT_OBJECT_CACHE_INDEX_ENV = "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE_INDEX"
_CUDA_AGENT_OBJECT_CACHE_DIR_ENV = "KERNELGYM_MANUAL_NINJA_OBJECT_CACHE_DIR"
_CUDA_AGENT_DEFAULT_OBJECT_CACHE_DIR = "/dev/shm/kernelgym/compile_cache/manual_ninja_objects"
_CUDA_AGENT_COMPILE_ARTIFACT_CACHE_ENV = "KERNELGYM_COMPILE_ARTIFACT_CACHE"
_CUDA_AGENT_DEFAULT_ARTIFACT_CACHE_DIR = "/dev/shm/kernelgym/compile_cache/cuda_agent_artifacts"
_DETAILED_COMPILE_TIMING_ENV = "KERNELGYM_DETAILED_COMPILE_TIMING"
_CUDA_AGENT_COMPILE_SOURCE_EXTS = {".cu", ".cpp", ".cc", ".cxx"}


def _torch_modules() -> tuple[Any, Any]:
    import torch
    import torch.utils.cpp_extension as cpp_ext

    return torch, cpp_ext


class KernelBenchCudaAgentBackend(KernelBenchBackendBase):
    """Compile and load CUDA-Agent style submissions."""

    name = "kernelbench.cuda_agent"

    @staticmethod
    def _default_binding_cpp() -> str:
        return """#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
"""

    @staticmethod
    def _default_binding_registry_h() -> str:
        return """#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class BindingRegistry {
public:
    using BindingFunction = std::function<void(pybind11::module&)>;

    static BindingRegistry& getInstance() {
        static BindingRegistry instance;
        return instance;
    }

    void registerBinding(const std::string& name, BindingFunction func) {
        bindings_.push_back({name, std::move(func)});
    }

    void applyBindings(pybind11::module& m) {
        for (auto& binding : bindings_) {
            binding.second(m);
        }
    }

private:
    std::vector<std::pair<std::string, BindingFunction>> bindings_;
    BindingRegistry() = default;
};

class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name, BindingRegistry::BindingFunction func) {
        BindingRegistry::getInstance().registerBinding(name, std::move(func));
    }
};

#define REGISTER_BINDING(name, func) \\
    static BindingRegistrar _registrar_##name(#name, [](pybind11::module& m) { func(m); })
"""

    @staticmethod
    def _strip_think_blocks(code: str) -> str:
        return strip_think_blocks(code)

    @staticmethod
    def _section_pattern(section_name: str, language: str) -> re.Pattern[str]:
        if language == "python":
            language_pattern = r"(?:python|py)"
        else:
            language_pattern = r"(?:cpp|c\+\+|cxx|cuda|cu)?"
        return re.compile(
            rf"###\s*{section_name}\s*```{language_pattern}\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )

    @classmethod
    def _extract_cuda_sections(cls, code: str, *, require_complete: bool = False) -> dict[str, str]:
        code = cls._strip_think_blocks(code)
        section_order = (
            ("CUDA_KERNELS", "cpp"),
            ("APPLY_BINDINGS", "cpp"),
            ("MODEL_NEW", "python"),
        )
        matches = {name: list(cls._section_pattern(name, language).finditer(code)) for name, language in section_order}

        best_group: dict[str, str] = {}
        for cuda_match in matches["CUDA_KERNELS"]:
            binding_match = next(
                (match for match in matches["APPLY_BINDINGS"] if match.start() > cuda_match.end()),
                None,
            )
            if binding_match is None:
                continue
            model_match = next(
                (match for match in matches["MODEL_NEW"] if match.start() > binding_match.end()),
                None,
            )
            if model_match is None:
                continue
            best_group = {
                "CUDA_KERNELS": cuda_match.group(1).strip(),
                "APPLY_BINDINGS": binding_match.group(1).strip(),
                "MODEL_NEW": model_match.group(1).strip(),
            }
        if best_group:
            return best_group
        if require_complete:
            return {}

        sections: dict[str, str] = {}
        for name, language in section_order:
            section_matches = matches[name]
            if section_matches:
                sections[name] = section_matches[-1].group(1).strip()
        return sections

    @staticmethod
    def _parse_embedded_sources(code: str) -> tuple[dict[str, str], str]:
        legacy_match = re.search(
            r"###\s*CUDA_SOURCES\s*###\s*(.*?)###\s*END_CUDA_SOURCES\s*###",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if legacy_match is not None:
            source_blob = legacy_match.group(1).strip()
            parsed = ast.literal_eval(source_blob)
            if not isinstance(parsed, dict):
                raise TypeError("Embedded CUDA_SOURCES must evaluate to a dict[str, str]")
            normalized = {str(name): str(content) for name, content in parsed.items()}
            python_code = f"{code[: legacy_match.start()]}{code[legacy_match.end() :]}".strip()
            return normalized, python_code

        code_without_think = KernelBenchCudaAgentBackend._strip_think_blocks(code)
        sections = KernelBenchCudaAgentBackend._extract_cuda_sections(
            code_without_think,
            require_complete=True,
        )
        if not sections:
            sections = KernelBenchCudaAgentBackend._extract_cuda_sections(code_without_think)
        if not sections:
            return {}, code_without_think.strip()

        cuda_sources: dict[str, str] = {}
        if sections.get("CUDA_KERNELS"):
            cuda_sources["kernels/generated.cu"] = sections["CUDA_KERNELS"]
        if sections.get("APPLY_BINDINGS"):
            cuda_sources["kernels/generated_binding.cpp"] = sections["APPLY_BINDINGS"]
        python_code = sections.get("MODEL_NEW", "")
        return cuda_sources, python_code

    @staticmethod
    def _normalize_cuda_sources_input(cuda_sources: Any) -> dict[str, str]:
        if not cuda_sources:
            return {}
        if not isinstance(cuda_sources, dict):
            raise TypeError("cuda_sources must be a dict[str, str]")
        return {str(name): str(content) for name, content in cuda_sources.items()}

    @staticmethod
    def _decode_mountinfo_path(path: str) -> str:
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), path)

    @staticmethod
    def _mountinfo_path_has_noexec(path: Path, mountinfo_text: str) -> bool:
        try:
            resolved_path = path.resolve(strict=False)
        except OSError:
            resolved_path = path.absolute()

        best_mount_len = -1
        best_options: set[str] = set()
        for line in mountinfo_text.splitlines():
            fields = line.split()
            if len(fields) < 6:
                continue
            mount_point = Path(KernelBenchCudaAgentBackend._decode_mountinfo_path(fields[4]))
            try:
                resolved_mount = mount_point.resolve(strict=False)
            except OSError:
                resolved_mount = mount_point.absolute()
            if resolved_path != resolved_mount and resolved_mount not in resolved_path.parents:
                continue
            mount_len = len(str(resolved_mount))
            if mount_len > best_mount_len:
                best_mount_len = mount_len
                best_options = set(fields[5].split(","))
        return "noexec" in best_options

    @staticmethod
    def _path_has_noexec_mount(path: Path) -> bool:
        try:
            mountinfo_text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        except OSError:
            return False
        return KernelBenchCudaAgentBackend._mountinfo_path_has_noexec(path, mountinfo_text)

    @staticmethod
    def _path_is_under_fast_rw_root(path: Path) -> bool:
        try:
            resolved_path = path.resolve(strict=False)
            resolved_root = _CUDA_AGENT_FAST_RW_ROOT.resolve(strict=False)
        except OSError:
            resolved_path = path.absolute()
            resolved_root = _CUDA_AGENT_FAST_RW_ROOT.absolute()
        return resolved_path == resolved_root or resolved_root in resolved_path.parents

    @staticmethod
    def _require_fast_rw_path(path: Path, *, label: str) -> None:
        if not KernelBenchCudaAgentBackend._path_is_under_fast_rw_root(path):
            raise ValueError(f"{label} must be under /dev/shm for fast local I/O: {path}")

    @staticmethod
    def _select_work_dir_parent() -> str | None:
        candidate = _CUDA_AGENT_DEFAULT_TMPDIR
        path = Path(candidate)
        KernelBenchCudaAgentBackend._require_fast_rw_path(
            path,
            label="CUDA-Agent tmpdir",
        )
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
                raise RuntimeError(f"CUDA-Agent tmpdir is not writable/executable: {path}")
            if KernelBenchCudaAgentBackend._path_has_noexec_mount(path):
                raise RuntimeError(f"CUDA-Agent tmpdir is mounted noexec: {path}")
            if shutil.disk_usage(path).free < _CUDA_AGENT_MIN_TMPDIR_FREE_BYTES:
                raise RuntimeError(f"CUDA-Agent tmpdir has less than 512MiB free: {path}")
        except OSError as exc:
            raise RuntimeError(f"CUDA-Agent tmpdir is not usable: {path}") from exc
        return str(path)

    def _create_work_dir(self) -> Path:
        parent = self._select_work_dir_parent()
        return Path(tempfile.mkdtemp(prefix="kernelgym_cuda_agent_", dir=parent))

    def _write_runtime_scaffold(
        self,
        work_dir: Path,
        model_code: str,
        cuda_sources: dict[str, str] | None = None,
    ) -> None:
        (work_dir / "__init__.py").write_text("", encoding="utf-8")
        (work_dir / "model_new.py").write_text(model_code, encoding="utf-8")
        # If the model already provides a PYBIND11_MODULE in its APPLY_BINDINGS
        # source, do NOT write the framework's own binding.cpp/binding_registry.h —
        # two PYBIND11_MODULE blocks would produce a duplicate-symbol link error.
        # The model's APPLY_BINDINGS file gets compiled in the normal source sweep.
        has_pybind11_module = bool(cuda_sources) and any(
            re.search(r"\bPYBIND11_MODULE\s*\(", src) for src in cuda_sources.values()
        )
        if not has_pybind11_module:
            (work_dir / "binding.cpp").write_text(
                self._default_binding_cpp(),
                encoding="utf-8",
            )
            (work_dir / "binding_registry.h").write_text(
                self._default_binding_registry_h(),
                encoding="utf-8",
            )

    @staticmethod
    def _materialize_sources(work_dir: Path, cuda_sources: dict[str, str]) -> None:
        for filename, content in cuda_sources.items():
            relative_path = Path(filename)
            if relative_path.parent == Path("."):
                file_path = work_dir / "kernels" / relative_path
            else:
                file_path = work_dir / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

    @staticmethod
    def _collect_compile_sources(work_dir: Path) -> list[str]:
        sources: list[str] = []
        for path in work_dir.rglob("*"):
            if not path.is_file():
                continue
            if "build" in path.parts:
                continue
            if path.suffix.lower() in _CUDA_AGENT_COMPILE_SOURCE_EXTS:
                sources.append(str(path))
        return sorted(set(sources))

    @staticmethod
    @contextmanager
    def _file_lock(lock_path: Path):
        import fcntl

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _cuda_arch_fingerprint() -> str:
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if arch_list:
            return arch_list
        try:
            torch, _cpp_ext = _torch_modules()
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                return f"sm_{major}{minor}"
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _extract_custom_kernel_names(cuda_sources: dict[str, str]) -> list[str]:
        kernel_pattern = re.compile(
            r"__global__\s+"
            r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
            r"(?:(?:inline|static|constexpr|__forceinline__|__host__|__device__)\s+)*"
            r"(?:[\w:<>]+\s+)+"
            r"([A-Za-z_]\w*)\s*\(",
            re.MULTILINE,
        )
        kernel_names: set[str] = set()
        for filename, content in cuda_sources.items():
            if not filename.lower().endswith(".cu"):
                continue
            kernel_names.update(kernel_pattern.findall(content))
        return sorted(kernel_names)

    @staticmethod
    def _env_flag(name: str, *, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _node_id() -> str:
        return os.environ.get("NODE_ID") or socket.gethostname() or "local"

    @staticmethod
    def _manual_ninja_object_cache_root() -> Path:
        root = Path(os.environ.get(_CUDA_AGENT_OBJECT_CACHE_DIR_ENV, _CUDA_AGENT_DEFAULT_OBJECT_CACHE_DIR))
        KernelBenchCudaAgentBackend._require_fast_rw_path(root, label="CUDA-Agent manual ninja object cache")
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _manual_ninja_object_cache_index() -> str:
        index = os.environ.get(_CUDA_AGENT_OBJECT_CACHE_INDEX_ENV, "fs").strip().lower()
        if index in {"", "local", "file", "fs", "filesystem"}:
            return "fs"
        if index != "redis":
            raise ValueError(f"{_CUDA_AGENT_OBJECT_CACHE_INDEX_ENV} must be one of fs or redis")
        return index

    @staticmethod
    def _manual_ninja_redis_client() -> Any | None:
        try:
            import redis
        except Exception:
            return None
        try:
            client = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                db=int(os.environ.get("REDIS_DB", "0")),
                password=os.environ.get("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            client.ping()
            return client
        except Exception:
            return None

    @staticmethod
    def _manual_ninja_redis_cache_key(cache_key: str) -> str:
        prefix = os.environ.get("REDIS_KEY_PREFIX", "kernelgym")
        return f"{prefix}:manual_ninja_object_cache:{KernelBenchCudaAgentBackend._node_id()}:{cache_key}"

    @staticmethod
    def _redis_json_get(client: Any | None, key: str) -> dict[str, Any] | None:
        if client is None:
            return None
        try:
            raw = client.get(key)
            value = json.loads(raw) if raw else None
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    @staticmethod
    def _redis_json_set(
        client: Any | None, key: str, value: dict[str, Any], *, nx: bool = False, ex: int | None = None
    ) -> bool:
        if client is None:
            return False
        try:
            return bool(client.set(key, json.dumps(value, sort_keys=True), nx=nx, ex=ex))
        except Exception:
            return False

    @staticmethod
    def _prepare_torch_extension_ldflags(cpp_ext: Any) -> list[str]:
        prepare_ldflags = cpp_ext._prepare_ldflags  # type: ignore[attr-defined]
        signature = inspect.signature(prepare_ldflags)
        kwargs = {
            "extra_ldflags": [],
            "with_cuda": True,
            "with_sycl": False,
            "verbose": False,
            "is_standalone": False,
        }
        return prepare_ldflags(**{key: value for key, value in kwargs.items() if key in signature.parameters})

    @staticmethod
    def _write_ninja_file_to_build_library(
        cpp_ext: Any,
        *,
        path: Path,
        name: str,
        sources: list[str],
        extra_cflags: list[str],
        extra_cuda_cflags: list[str],
        extra_ldflags: list[str],
    ) -> None:
        writer = cpp_ext._write_ninja_file_to_build_library  # type: ignore[attr-defined]
        kwargs = {
            "path": str(path),
            "name": name,
            "sources": sources,
            "extra_cflags": extra_cflags,
            "extra_cuda_cflags": extra_cuda_cflags,
            "extra_sycl_cflags": [],
            "extra_ldflags": extra_ldflags,
            "extra_include_paths": [],
            "with_cuda": True,
            "with_sycl": False,
            "is_standalone": False,
        }
        signature = inspect.signature(writer)
        writer(**{key: value for key, value in kwargs.items() if key in signature.parameters})

    @staticmethod
    def _run_ninja_build(cpp_ext: Any, build_dir: Path, ext_name: str) -> None:
        runner = cpp_ext._run_ninja_build  # type: ignore[attr-defined]
        try:
            runner(str(build_dir), False, error_prefix=f"Error building extension '{ext_name}'")
        except TypeError:
            runner(str(build_dir), False)

    @staticmethod
    def _header_digest(work_dir: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(work_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".h", ".hh", ".hpp", ".hxx", ".cuh"}:
                continue
            if "build" in path.parts:
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
    def _normalize_build_text_for_cache(text: str, *, ext_name: str, work_dir: Path, build_dir: Path) -> str:
        normalized = text.replace(ext_name, "<TORCH_EXTENSION_NAME>")
        normalized = normalized.replace(str(work_dir), "<WORK_DIR>")
        normalized = normalized.replace(str(build_dir), "<BUILD_DIR>")
        return normalized

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

    @staticmethod
    def _resolve_ninja_source_path(source: str, build_dir: Path) -> Path:
        path = Path(source)
        if path.is_absolute():
            return path
        return build_dir / path

    @staticmethod
    def _source_is_reusable_object(source_path: Path) -> tuple[bool, str | None]:
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return False, f"source read failed: {exc}"
        if "TORCH_EXTENSION_NAME" in text or "PYBIND11_MODULE" in text:
            return False, "source references module name"
        return True, None

    @staticmethod
    def _manual_ninja_cache_key(
        *,
        build_ninja_text: str,
        ext_name: str,
        work_dir: Path,
        build_dir: Path,
        object_name: str,
        rule: str,
        source_path: Path,
        header_digest: str,
        extra_cflags: list[str],
        extra_cuda_cflags: list[str],
        torch: Any,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(
            KernelBenchCudaAgentBackend._normalize_build_text_for_cache(
                build_ninja_text,
                ext_name=ext_name,
                work_dir=work_dir,
                build_dir=build_dir,
            ).encode("utf-8")
        )
        for item in (
            object_name,
            rule,
            header_digest,
            torch.__version__,
            str(torch.version.cuda),
            sys.version,
            KernelBenchCudaAgentBackend._cuda_arch_fingerprint(),
            json.dumps(extra_cflags, sort_keys=True),
            json.dumps(extra_cuda_cflags, sort_keys=True),
        ):
            digest.update(b"\0")
            digest.update(item.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _prepare_manual_ninja_cached_objects(
        *,
        build_dir: Path,
        work_dir: Path,
        ext_name: str,
        extra_cflags: list[str],
        extra_cuda_cflags: list[str],
        torch: Any,
    ) -> dict[str, Any] | None:
        if not KernelBenchCudaAgentBackend._env_flag(_CUDA_AGENT_OBJECT_CACHE_ENV, default=True):
            return None
        build_ninja_path = build_dir / "build.ninja"
        build_ninja_text = build_ninja_path.read_text(encoding="utf-8")
        _header, edges = KernelBenchCudaAgentBackend._ninja_header_and_object_edges(build_ninja_text)
        header_digest = KernelBenchCudaAgentBackend._header_digest(work_dir)
        cache_root = KernelBenchCudaAgentBackend._manual_ninja_object_cache_root()
        cache_index = KernelBenchCudaAgentBackend._manual_ninja_object_cache_index()
        redis_client = KernelBenchCudaAgentBackend._manual_ninja_redis_client() if cache_index == "redis" else None
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
        lookup_started = time.perf_counter()
        for edge in edges:
            output = edge["output"]
            source_path = KernelBenchCudaAgentBackend._resolve_ninja_source_path(edge["source"], build_dir)
            reusable, reason = KernelBenchCudaAgentBackend._source_is_reusable_object(source_path)
            if not reusable:
                stats["skipped"].append({"object": output, "source": str(source_path), "reason": reason})
                continue

            cache_key = KernelBenchCudaAgentBackend._manual_ninja_cache_key(
                build_ninja_text=build_ninja_text,
                ext_name=ext_name,
                work_dir=work_dir,
                build_dir=build_dir,
                object_name=output,
                rule=edge["rule"],
                source_path=source_path,
                header_digest=header_digest,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                torch=torch,
            )
            cache_dir = cache_root / cache_key
            cache_object = cache_dir / output
            local_object = build_dir / output
            lock_path = cache_dir / ".lock"
            redis_key = KernelBenchCudaAgentBackend._manual_ninja_redis_cache_key(cache_key)
            object_info = {
                "object": output,
                "source": str(source_path),
                "cache_key": cache_key,
                "cache_path": str(cache_object),
                "local_object": str(local_object),
                "index": cache_index,
            }
            redis_entry = (
                KernelBenchCudaAgentBackend._redis_json_get(redis_client, redis_key)
                if cache_index == "redis"
                else None
            )
            indexed_path = (
                Path(str(redis_entry.get("object_path")))
                if redis_entry and redis_entry.get("status") == "ready"
                else None
            )
            if indexed_path and indexed_path.exists():
                object_info["cache_status"] = "hit"
                object_info["index_status"] = "redis_ready"
                object_info["cache_path"] = str(indexed_path)
                stats["hits"] += 1
                stats["object_map"][output] = str(indexed_path)
            elif cache_object.exists():
                object_info["cache_status"] = "hit"
                object_info["index_status"] = "fs_ready"
                stats["hits"] += 1
                stats["object_map"][output] = str(cache_object)
            else:
                object_info["cache_status"] = "miss_pending"
                object_info["index_status"] = str(redis_entry.get("status")) if redis_entry else "missing"
                object_info["lock_path"] = str(lock_path)
                stats["misses"] += 1
                if cache_index == "redis":
                    KernelBenchCudaAgentBackend._redis_json_set(
                        redis_client,
                        redis_key,
                        {
                            "status": "building",
                            "object_path": str(cache_object),
                            "object_name": output,
                            "cache_key": cache_key,
                            "node_id": KernelBenchCudaAgentBackend._node_id(),
                            "created_at": time.time(),
                            "builder_pid": os.getpid(),
                        },
                        nx=True,
                        ex=600,
                    )
            stats["objects"].append(object_info)
        stats["lookup_wall_sec"] = round(time.perf_counter() - lookup_started, 6)
        return stats

    @staticmethod
    def _rewrite_manual_ninja_for_cached_objects(build_dir: Path, object_map: dict[str, str]) -> None:
        if not object_map:
            return
        build_ninja_path = build_dir / "build.ninja"
        rewritten_lines: list[str] = []
        for line in build_ninja_path.read_text(encoding="utf-8").splitlines():
            if any(re.match(rf"^build\s+{re.escape(object_name)}:\s+", line) for object_name in object_map):
                continue
            if line.startswith("build ") and ": link " in line:
                for object_name, cache_path in object_map.items():
                    line = re.sub(rf"(?<!\S){re.escape(object_name)}(?!\S)", cache_path, line)
            rewritten_lines.append(line)
        build_ninja_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _store_manual_ninja_cached_objects(object_cache_stats: dict[str, Any] | None) -> None:
        if not object_cache_stats:
            return
        cache_index = str(object_cache_stats.get("index") or "fs")
        redis_client = KernelBenchCudaAgentBackend._manual_ninja_redis_client() if cache_index == "redis" else None
        store_started = time.perf_counter()
        for object_info in object_cache_stats.get("objects") or []:
            if object_info.get("cache_status") != "miss_pending":
                continue
            local_object = Path(str(object_info.get("local_object") or ""))
            cache_object = Path(str(object_info.get("cache_path") or ""))
            lock_path = Path(str(object_info.get("lock_path") or cache_object.parent / ".lock"))
            redis_key = KernelBenchCudaAgentBackend._manual_ninja_redis_cache_key(
                str(object_info.get("cache_key") or "")
            )
            if not local_object.exists():
                object_info["cache_status"] = "miss_not_built"
                continue
            cache_object.parent.mkdir(parents=True, exist_ok=True)
            with KernelBenchCudaAgentBackend._file_lock(lock_path):
                if cache_object.exists():
                    object_info["cache_status"] = "miss_lost_race"
                else:
                    temp_object = cache_object.with_suffix(cache_object.suffix + ".tmp")
                    shutil.copy2(local_object, temp_object)
                    temp_object.replace(cache_object)
                    object_info["cache_status"] = "miss_stored"
                if cache_index == "redis" and cache_object.exists():
                    KernelBenchCudaAgentBackend._redis_json_set(
                        redis_client,
                        redis_key,
                        {
                            "status": "ready",
                            "object_path": str(cache_object),
                            "object_name": object_info.get("object"),
                            "cache_key": object_info.get("cache_key"),
                            "node_id": KernelBenchCudaAgentBackend._node_id(),
                            "created_at": time.time(),
                            "last_used_at": time.time(),
                            "size_bytes": cache_object.stat().st_size,
                        },
                    )
        object_cache_stats["store_wall_sec"] = round(time.perf_counter() - store_started, 6)

    @staticmethod
    def _new_compile_timing(build_dir: Path, *, build_backend: str) -> dict[str, Any]:
        timing: dict[str, Any] = {
            "build_backend": build_backend,
            "build_dir": str(build_dir),
            "detailed_compile_timing_enabled": KernelBenchCudaAgentBackend._env_flag(
                _DETAILED_COMPILE_TIMING_ENV,
                default=False,
            ),
        }
        return timing

    @staticmethod
    def _compile_artifact_cache_root() -> Path:
        root = Path(os.environ.get("KERNELGYM_COMPILE_ARTIFACT_CACHE_DIR", _CUDA_AGENT_DEFAULT_ARTIFACT_CACHE_DIR))
        KernelBenchCudaAgentBackend._require_fast_rw_path(root, label="CUDA-Agent compile artifact cache")
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _compile_artifact_cache_enabled(kwargs: dict[str, Any]) -> bool:
        if kwargs.get("enable_compile_artifact_cache") is not None:
            return bool(kwargs.get("enable_compile_artifact_cache"))
        return KernelBenchCudaAgentBackend._env_flag(_CUDA_AGENT_COMPILE_ARTIFACT_CACHE_ENV, default=False)

    @staticmethod
    def _artifact_cache_key(
        *,
        model_code: str,
        cuda_sources: dict[str, str],
        entry_point: str,
    ) -> str:
        torch, _cpp_ext = _torch_modules()
        payload = {
            "backend": "cuda_agent",
            "entry_point": entry_point,
            "model_code": model_code,
            "cuda_sources": {key: cuda_sources[key] for key in sorted(cuda_sources)},
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
            "cuda_arch": KernelBenchCudaAgentBackend._cuda_arch_fingerprint(),
            "nvcc_threads": os.environ.get(_NVCC_THREADS_ENV, _CUDA_AGENT_DEFAULT_NVCC_THREADS),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _load_cached_artifact(ready_path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(ready_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        work_dir = payload.get("work_dir")
        so_path = payload.get("so_path")
        module_name = payload.get("module_name")
        code = payload.get("code")
        profiling_hints = payload.get("profiling_hints")
        if not work_dir or not so_path or not module_name or not code or not isinstance(profiling_hints, dict):
            return None
        if not Path(str(work_dir)).is_dir() or not Path(str(so_path)).is_file():
            return None
        payload["compiled"] = True
        payload["compile_artifact_cache_hit"] = True
        payload["persistent_work_dir"] = True
        return payload

    @staticmethod
    def _write_cached_artifact(ready_path: Path, artifact: dict[str, Any]) -> None:
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value
            for key, value in artifact.items()
            if key
            in {
                "compiled",
                "device",
                "entry_point",
                "backend",
                "work_dir",
                "so_path",
                "module_name",
                "code",
                "precheck",
                "profiling_hints",
                "build_backend",
                "compile_timing",
                "compile_artifact_cache_key",
            }
        }
        payload["persistent_work_dir"] = True
        tmp_path = ready_path.with_suffix(ready_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(ready_path)

    @staticmethod
    def _build_extension(work_dir: Path, sources: list[str]) -> Dict[str, Any]:
        if not sources:
            return {
                "compiled": False,
                "error": "No CUDA source files found for CUDA-Agent compilation",
            }
        KernelBenchCudaAgentBackend._require_fast_rw_path(
            work_dir,
            label="CUDA-Agent work_dir",
        )

        build_dir = work_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        nvcc_threads = os.environ.get(
            _NVCC_THREADS_ENV,
            _CUDA_AGENT_DEFAULT_NVCC_THREADS,
        )
        torch, cpp_ext = _torch_modules()
        extra_cflags = ["-O3", "-std=c++17"]
        extra_cuda_cflags = ["-O3", "--use_fast_math", "--threads", nvcc_threads]
        ext_name = work_dir.name.replace("-", "_")
        compile_timing = KernelBenchCudaAgentBackend._new_compile_timing(build_dir, build_backend="manual_ninja")

        try:
            build_started = time.perf_counter()
            extra_ldflags = KernelBenchCudaAgentBackend._prepare_torch_extension_ldflags(cpp_ext)
            KernelBenchCudaAgentBackend._write_ninja_file_to_build_library(
                cpp_ext,
                path=build_dir / "build.ninja",
                name=ext_name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
            )
            object_cache_stats = KernelBenchCudaAgentBackend._prepare_manual_ninja_cached_objects(
                build_dir=build_dir,
                work_dir=work_dir,
                ext_name=ext_name,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                torch=torch,
            )
            if object_cache_stats:
                KernelBenchCudaAgentBackend._rewrite_manual_ninja_for_cached_objects(
                    build_dir,
                    object_cache_stats.get("object_map", {}),
                )
            KernelBenchCudaAgentBackend._run_ninja_build(cpp_ext, build_dir, ext_name)
            KernelBenchCudaAgentBackend._store_manual_ninja_cached_objects(object_cache_stats)
            build_wall_sec = time.perf_counter() - build_started
            compile_timing["manual_ninja_build_wall_sec"] = round(build_wall_sec, 6)
            if object_cache_stats:
                compile_timing["manual_ninja_object_cache"] = {
                    key: value for key, value in object_cache_stats.items() if key != "object_map"
                }

            lib_ext = getattr(cpp_ext, "LIB_EXT", ".so")
            built_so = build_dir / f"{ext_name}{lib_ext}"
            if not built_so.exists():
                candidates = sorted(build_dir.glob(f"{ext_name}*.so"))
                built_so = candidates[0] if candidates else built_so
            if not built_so.exists():
                compile_timing["total_wall_sec"] = round(build_wall_sec, 6)
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
            compile_timing["manual_ninja_import_wall_sec"] = round(import_wall_sec, 6)
            compile_timing["total_wall_sec"] = round(build_wall_sec + import_wall_sec, 6)
            compile_timing["built_so_size_bytes"] = built_so.stat().st_size

            return {
                "compiled": True,
                "so_path": str(built_so),
                "module_name": ext_name,
                "build_backend": "manual_ninja",
                "compile_timing": compile_timing,
            }
        except Exception as exc:
            if "object_cache_stats" in locals():
                KernelBenchCudaAgentBackend._store_manual_ninja_cached_objects(object_cache_stats)
                if object_cache_stats:
                    compile_timing["manual_ninja_object_cache"] = {
                        key: value for key, value in object_cache_stats.items() if key != "object_map"
                    }
            if "build_started" in locals():
                compile_timing["manual_ninja_build_wall_sec"] = round(time.perf_counter() - build_started, 6)
                compile_timing["total_wall_sec"] = compile_timing["manual_ninja_build_wall_sec"]
            return {
                "compiled": False,
                "error": str(exc),
                "build_backend": "manual_ninja",
                "compile_timing": compile_timing,
            }

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        explicit_sources = self._normalize_cuda_sources_input(kwargs.get("cuda_sources"))
        enable_compile_artifact_cache = self._compile_artifact_cache_enabled(kwargs)

        try:
            embedded_sources, python_code = self._parse_embedded_sources(code)
        except Exception as exc:
            return {
                "compiled": False,
                "error": f"Failed to parse CUDA-Agent submission: {exc}",
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
            }

        cuda_sources = {**explicit_sources, **embedded_sources}
        model_code = python_code.strip() or code.strip()

        precheck_error, _precheck_code, precheck_info = precheck_cuda_agent_submission(
            model_code,
            cuda_sources,
            entry_point=entry_point,
        )
        if not precheck_info.get("passed", False):
            return {
                "compiled": False,
                "error": precheck_error,
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
                "precheck": precheck_info,
            }

        work_dir: Path
        cache_key = ""
        ready_path: Path | None = None
        if enable_compile_artifact_cache:
            cache_key = self._artifact_cache_key(
                model_code=model_code,
                cuda_sources=cuda_sources,
                entry_point=entry_point,
            )
            cache_entry = self._compile_artifact_cache_root() / cache_key
            ready_path = cache_entry / "ready.json"
            cached_artifact = self._load_cached_artifact(ready_path)
            if cached_artifact is not None:
                cached_artifact["device"] = str(device)
                cached_artifact["compile_artifact_cache_enabled"] = True
                cached_artifact["compile_artifact_cache_key"] = cache_key
                return cached_artifact
            work_dir = cache_entry / "work"
        else:
            work_dir = self._create_work_dir()

        def _compile_in_work_dir() -> dict[str, Any]:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            self._write_runtime_scaffold(work_dir, model_code, cuda_sources)
            self._materialize_sources(work_dir, cuda_sources)
            try:
                return self._build_extension(work_dir, self._collect_compile_sources(work_dir))
            except Exception as exc:
                return {"compiled": False, "error": str(exc)}

        if enable_compile_artifact_cache and ready_path is not None:
            with self._file_lock(ready_path.parent / "compile.lock"):
                cached_artifact = self._load_cached_artifact(ready_path)
                if cached_artifact is not None:
                    cached_artifact["device"] = str(device)
                    cached_artifact["compile_artifact_cache_enabled"] = True
                    cached_artifact["compile_artifact_cache_key"] = cache_key
                    return cached_artifact
                result = _compile_in_work_dir()
        else:
            result = _compile_in_work_dir()

        artifact = {
            "compiled": bool(result.get("compiled")),
            "error": result.get("error"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": "cuda_agent",
            "work_dir": str(work_dir),
            "so_path": result.get("so_path"),
            "module_name": result.get("module_name"),
            "build_backend": result.get("build_backend", "manual_ninja"),
            "compile_timing": result.get("compile_timing"),
            "compile_artifact_cache_enabled": enable_compile_artifact_cache,
            "compile_artifact_cache_hit": False,
            "compile_artifact_cache_key": cache_key or None,
            "persistent_work_dir": enable_compile_artifact_cache,
            "code": model_code,
            "precheck": precheck_info,
            "profiling_hints": {
                "backend": "cuda_agent",
                "custom_kernel_names": self._extract_custom_kernel_names(cuda_sources),
                "detected_extension_calls": list(precheck_info.get("detected_extension_calls", [])),
                "source_files": sorted(cuda_sources.keys()),
            },
        }
        if enable_compile_artifact_cache and ready_path is not None and artifact["compiled"]:
            self._write_cached_artifact(ready_path, artifact)
        return artifact

    @staticmethod
    def _load_extension_module(module_name: str, so_path: Path) -> types.ModuleType:
        loader = importlib.machinery.ExtensionFileLoader(module_name, str(so_path))
        spec = importlib.util.spec_from_file_location(module_name, str(so_path), loader=loader)
        if spec is None:
            raise ImportError(f"Failed to create import spec for {so_path}")
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        work_dir = artifact.get("work_dir")
        so_path = artifact.get("so_path")
        module_name = artifact.get("module_name")
        context = kwargs.get("context") or {}

        if not code:
            raise ValueError("KernelBenchCudaAgentBackend.load requires kernel code in artifact")
        if not work_dir:
            raise ValueError("KernelBenchCudaAgentBackend.load requires a work_dir in artifact")
        if not so_path or not Path(so_path).exists():
            raise ValueError(f"Compiled shared library not found: {so_path}")
        if not module_name:
            raise ValueError("KernelBenchCudaAgentBackend.load requires module_name in artifact")

        device = self._normalize_device(kwargs.get("device") or artifact.get("device"))
        self._maybe_set_cuda_device(device)
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        work_dir_path = Path(work_dir)
        so_path_path = Path(so_path)
        runtime_package_name = f"_kernelgym_cuda_agent_{work_dir_path.name.replace('-', '_')}"
        package_module = types.ModuleType(runtime_package_name)
        package_module.__path__ = [str(work_dir_path)]  # type: ignore[attr-defined]
        package_module.__package__ = runtime_package_name
        sys.modules[runtime_package_name] = package_module

        ext_module = sys.modules.get(module_name)
        if ext_module is None:
            ext_module = self._load_extension_module(module_name, so_path_path)

        module_aliases = [
            module_name,
            "cuda_extension",
            f"{runtime_package_name}.cuda_extension",
            runtime_package_name,
            f"{runtime_package_name}.model_new",
        ]
        sys.modules["cuda_extension"] = ext_module
        sys.modules[f"{runtime_package_name}.cuda_extension"] = ext_module

        model_name = f"{runtime_package_name}.model_new"
        model_file = work_dir_path / "model_new.py"
        spec = importlib.util.spec_from_file_location(model_name, str(model_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {model_file}")
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = model_module
        spec.loader.exec_module(model_module)

        model_cls = getattr(model_module, entry_point, None)
        if model_cls is None:
            raise ValueError(f"Failed to load model class '{entry_point}' from code")

        return {
            "model_cls": model_cls,
            "context": context,
            "backend": "cuda_agent",
            "entry_point": entry_point,
            "device": device,
            "work_dir": str(work_dir_path),
            "so_path": str(so_path_path),
            "module_aliases": module_aliases,
            "tempfile_handle": None,
            "profiling_hints": artifact.get("profiling_hints", {}),
            "persistent_work_dir": bool(artifact.get("persistent_work_dir")),
        }

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        super().cleanup(handle, **kwargs)
        if not isinstance(handle, dict):
            return

        for module_name in handle.get("module_aliases", []):
            sys.modules.pop(module_name, None)

        work_dir = handle.get("work_dir")
        if handle.get("persistent_work_dir"):
            return
        if work_dir and Path(work_dir).exists():
            shutil.rmtree(work_dir, ignore_errors=True)
