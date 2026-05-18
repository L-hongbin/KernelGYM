"""CUDA-Agent-specific KernelBench backend implementation."""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from kernelgym.toolkit.validation import precheck_cuda_agent_submission

from .base import KernelBenchBackendBase
from kernelgym.toolkit.kernelbench.binding_detection import strip_think_blocks


_CUDA_AGENT_TMPDIR_ENV = "KERNELGYM_CUDA_AGENT_TMPDIR"
_CUDA_AGENT_DEFAULT_TMPDIR = "/dev/shm/kernelgym/work/cuda_agent"
_CUDA_AGENT_MIN_TMPDIR_FREE_BYTES = 512 * 1024 * 1024
_CUDA_AGENT_NVCC_THREADS_ENV = "KERNELGYM_CUDA_AGENT_NVCC_THREADS"
_CUDA_AGENT_DEFAULT_NVCC_THREADS = "4"
_CUDA_AGENT_COMPILE_CACHE_DIR_ENV = "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DIR"
_CUDA_AGENT_COMPILE_CACHE_DISABLE_ENV = "KERNELGYM_CUDA_AGENT_COMPILE_CACHE_DISABLE"
_CUDA_AGENT_FAST_RW_ROOT = Path("/dev/shm")
_CUDA_AGENT_DEFAULT_COMPILE_CACHE_DIR = "/dev/shm/kernelgym/compile_cache/cuda_agent"
_CUDA_AGENT_COMPILE_SOURCE_EXTS = {".cu", ".cpp", ".cc", ".cxx"}
_CUDA_AGENT_CACHE_INPUT_EXTS = _CUDA_AGENT_COMPILE_SOURCE_EXTS | {
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}


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
        candidate = os.environ.get(_CUDA_AGENT_TMPDIR_ENV) or _CUDA_AGENT_DEFAULT_TMPDIR
        path = Path(candidate)
        KernelBenchCudaAgentBackend._require_fast_rw_path(
            path,
            label=_CUDA_AGENT_TMPDIR_ENV,
        )
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
                raise RuntimeError(f"{_CUDA_AGENT_TMPDIR_ENV} is not writable/executable: {path}")
            if KernelBenchCudaAgentBackend._path_has_noexec_mount(path):
                raise RuntimeError(f"{_CUDA_AGENT_TMPDIR_ENV} is mounted noexec: {path}")
            if shutil.disk_usage(path).free < _CUDA_AGENT_MIN_TMPDIR_FREE_BYTES:
                raise RuntimeError(f"{_CUDA_AGENT_TMPDIR_ENV} has less than 512MiB free: {path}")
        except OSError as exc:
            raise RuntimeError(f"{_CUDA_AGENT_TMPDIR_ENV} is not usable: {path}") from exc
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
    def _collect_cache_inputs(work_dir: Path, sources: list[str]) -> list[str]:
        return KernelBenchCudaAgentBackend._collect_cache_inputs_excluding(
            work_dir,
            sources,
            exclude_roots=[],
        )

    @staticmethod
    def _collect_cache_inputs_excluding(
        work_dir: Path,
        sources: list[str],
        *,
        exclude_roots: list[Path],
    ) -> list[str]:
        resolved_excludes = []
        for root in exclude_roots:
            try:
                resolved_excludes.append(root.resolve())
            except OSError:
                resolved_excludes.append(root)
        inputs = {str(Path(source)) for source in sources}
        for path in work_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                resolved_path = path.resolve()
            except OSError:
                resolved_path = path
            if any(resolved_path == root or resolved_path.is_relative_to(root) for root in resolved_excludes):
                continue
            if "build" in path.parts:
                continue
            if path.suffix.lower() in _CUDA_AGENT_CACHE_INPUT_EXTS:
                inputs.add(str(path))
        return sorted(inputs)

    @staticmethod
    def _disabled_by_env(env_name: str) -> bool:
        value = os.environ.get(env_name, "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _compile_cache_root(
        *,
        dir_env: str,
        disable_env: str,
        default_dir: str,
    ) -> Path | None:
        if KernelBenchCudaAgentBackend._disabled_by_env(disable_env):
            return None
        root_value = os.environ.get(dir_env, default_dir)
        if not root_value:
            return None
        root = Path(root_value)
        KernelBenchCudaAgentBackend._require_fast_rw_path(root, label=dir_env)
        try:
            root.mkdir(parents=True, exist_ok=True)
            if not os.access(root, os.W_OK | os.X_OK):
                raise RuntimeError(f"{dir_env} is not writable/executable: {root}")
            if KernelBenchCudaAgentBackend._path_has_noexec_mount(root):
                raise RuntimeError(f"{dir_env} is mounted noexec: {root}")
            if shutil.disk_usage(root).free < _CUDA_AGENT_MIN_TMPDIR_FREE_BYTES:
                raise RuntimeError(f"{dir_env} has less than 512MiB free: {root}")
        except OSError as exc:
            raise RuntimeError(f"{dir_env} is not usable: {root}") from exc
        return root

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
    def _source_digest(
        *,
        work_dir: Path,
        sources: list[str],
        metadata: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        for source in sorted(sources):
            path = Path(source)
            try:
                rel_path = path.relative_to(work_dir)
            except ValueError:
                rel_path = Path(path.name)
            digest.update(str(rel_path).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _copy_sources_to_cache(
        *,
        work_dir: Path,
        sources: list[str],
        src_dir: Path,
    ) -> list[str]:
        if src_dir.exists():
            shutil.rmtree(src_dir)
        src_dir.mkdir(parents=True, exist_ok=True)
        cached_sources: list[str] = []
        for source in sorted(sources):
            source_path = Path(source)
            try:
                rel_path = source_path.relative_to(work_dir)
            except ValueError:
                rel_path = Path(source_path.name)
            target_path = src_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            cached_sources.append(str(target_path))
        return cached_sources

    @staticmethod
    def _load_cached_compile_result(ready_path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(ready_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        so_path = payload.get("so_path")
        module_name = payload.get("module_name")
        if not so_path or not module_name:
            return None
        if not Path(str(so_path)).is_file():
            return None
        return {
            "compiled": True,
            "so_path": str(so_path),
            "module_name": str(module_name),
            "compile_cache_hit": True,
            "compile_cache_key": payload.get("compile_cache_key"),
            "compile_cache_dir": str(ready_path.parent),
        }

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
            _CUDA_AGENT_NVCC_THREADS_ENV,
            _CUDA_AGENT_DEFAULT_NVCC_THREADS,
        )
        torch, cpp_ext = _torch_modules()
        extra_cflags = ["-O3", "-std=c++17"]
        extra_cuda_cflags = ["-O3", "--use_fast_math", "--threads", nvcc_threads]
        cache_metadata = {
            "backend": "cuda_agent",
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_arch": KernelBenchCudaAgentBackend._cuda_arch_fingerprint(),
            "extra_cflags": extra_cflags,
            "extra_cuda_cflags": extra_cuda_cflags,
        }
        cache_root = KernelBenchCudaAgentBackend._compile_cache_root(
            dir_env=_CUDA_AGENT_COMPILE_CACHE_DIR_ENV,
            disable_env=_CUDA_AGENT_COMPILE_CACHE_DISABLE_ENV,
            default_dir=_CUDA_AGENT_DEFAULT_COMPILE_CACHE_DIR,
        )
        cache_key = ""
        compile_cache_dir = ""
        compile_cache_hit = False
        if cache_root is not None:
            cache_inputs = KernelBenchCudaAgentBackend._collect_cache_inputs_excluding(
                work_dir,
                sources,
                exclude_roots=[cache_root],
            )
            cache_key = KernelBenchCudaAgentBackend._source_digest(
                work_dir=work_dir,
                sources=cache_inputs,
                metadata=cache_metadata,
            )
            cache_entry = cache_root / cache_key
            ready_path = cache_entry / "ready.json"
            cached = KernelBenchCudaAgentBackend._load_cached_compile_result(ready_path)
            if cached is not None:
                return cached

            with KernelBenchCudaAgentBackend._file_lock(cache_entry / "compile.lock"):
                cached = KernelBenchCudaAgentBackend._load_cached_compile_result(ready_path)
                if cached is not None:
                    return cached
                cached_sources = KernelBenchCudaAgentBackend._copy_sources_to_cache(
                    work_dir=work_dir,
                    sources=cache_inputs,
                    src_dir=cache_entry / "src",
                )
                sources = [
                    source
                    for source in cached_sources
                    if Path(source).suffix.lower() in _CUDA_AGENT_COMPILE_SOURCE_EXTS
                ]
                build_dir = cache_entry / "build"
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                build_dir.mkdir(parents=True, exist_ok=True)
                ext_name = f"kernelgym_cuda_agent_{cache_key[:16]}"
                module = cpp_ext.load(
                    name=ext_name,
                    sources=sources,
                    build_directory=str(build_dir),
                    verbose=False,
                    with_cuda=True,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                )
                so_path = getattr(module, "__file__", "")
                ready_payload = {
                    "compiled": True,
                    "so_path": so_path,
                    "module_name": ext_name,
                    "compile_cache_key": cache_key,
                }
                ready_path.write_text(
                    json.dumps(ready_payload, sort_keys=True),
                    encoding="utf-8",
                )
                return {
                    "compiled": True,
                    "so_path": so_path,
                    "module_name": ext_name,
                    "compile_cache_hit": compile_cache_hit,
                    "compile_cache_key": cache_key,
                    "compile_cache_dir": str(cache_entry),
                }

        ext_name = work_dir.name.replace("-", "_")
        module = cpp_ext.load(
            name=ext_name,
            sources=sources,
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
        )
        return {
            "compiled": True,
            "so_path": getattr(module, "__file__", ""),
            "module_name": ext_name,
            "compile_cache_hit": compile_cache_hit,
            "compile_cache_key": cache_key,
            "compile_cache_dir": compile_cache_dir,
        }

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        explicit_sources = self._normalize_cuda_sources_input(kwargs.get("cuda_sources"))

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

        work_dir = self._create_work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        self._write_runtime_scaffold(work_dir, model_code, cuda_sources)
        self._materialize_sources(work_dir, cuda_sources)

        try:
            result = self._build_extension(work_dir, self._collect_compile_sources(work_dir))
        except Exception as exc:
            result = {"compiled": False, "error": str(exc)}

        return {
            "compiled": bool(result.get("compiled")),
            "error": result.get("error"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": "cuda_agent",
            "work_dir": str(work_dir),
            "so_path": result.get("so_path"),
            "module_name": result.get("module_name"),
            "compile_cache_hit": result.get("compile_cache_hit"),
            "compile_cache_key": result.get("compile_cache_key"),
            "compile_cache_dir": result.get("compile_cache_dir"),
            "code": model_code,
            "precheck": precheck_info,
            "profiling_hints": {
                "backend": "cuda_agent",
                "custom_kernel_names": self._extract_custom_kernel_names(cuda_sources),
                "detected_extension_calls": list(precheck_info.get("detected_extension_calls", [])),
                "source_files": sorted(cuda_sources.keys()),
            },
        }

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
        }

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        super().cleanup(handle, **kwargs)
        if not isinstance(handle, dict):
            return

        for module_name in handle.get("module_aliases", []):
            sys.modules.pop(module_name, None)

        work_dir = handle.get("work_dir")
        if work_dir and Path(work_dir).exists():
            shutil.rmtree(work_dir, ignore_errors=True)
