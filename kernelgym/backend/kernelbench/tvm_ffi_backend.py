"""TVM-FFI-specific KernelBench backend implementation."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

from kernelgym.toolkit.validation import precheck_tvm_ffi_submission

from .base import KernelBenchBackendBase
from .cuda_agent_backend import KernelBenchCudaAgentBackend


_TVM_FFI_DEFAULT_TMPDIR = "/dev/shm/kernelgym/work/tvm_ffi"
_TVM_FFI_MIN_TMPDIR_FREE_BYTES = 512 * 1024 * 1024
_NVCC_THREADS_ENV = "KERNELGYM_NVCC_THREADS"
_TVM_FFI_DEFAULT_NVCC_THREADS = "4"
_TVM_FFI_COMPILE_CACHE_DISABLE_ENV = "KERNELGYM_TVM_FFI_COMPILE_CACHE_DISABLE"
_TVM_FFI_DEFAULT_COMPILE_CACHE_DIR = "/dev/shm/kernelgym/compile_cache/tvm_ffi"


class _TvmFfiExtensionModule(types.ModuleType):
    def __init__(self, name: str, tvm_module: Any, tvm_ffi_api: Any) -> None:
        super().__init__(name)
        self._tvm_ffi_module = tvm_module
        self._tvm_ffi_api = tvm_ffi_api

    def _wrap_callable(self, func: Any) -> Any:
        if not callable(func):
            return func

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            stream_context = getattr(self._tvm_ffi_api, "use_torch_stream", None)
            context = stream_context() if callable(stream_context) else nullcontext()
            with context:
                return func(*args, **kwargs)

        _wrapped.__name__ = getattr(func, "__name__", "tvm_ffi_function")
        _wrapped.__doc__ = getattr(func, "__doc__", None)
        return _wrapped

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._tvm_ffi_module, name)
        wrapped = self._wrap_callable(value)
        setattr(self, name, wrapped)
        return wrapped


class KernelBenchTvmFfiBackend(KernelBenchBackendBase):
    """Compile and load TVM-FFI style CUDA submissions."""

    name = "kernelbench.tvm_ffi"

    _parse_embedded_sources = staticmethod(KernelBenchCudaAgentBackend._parse_embedded_sources)
    _materialize_sources = staticmethod(KernelBenchCudaAgentBackend._materialize_sources)
    _extract_custom_kernel_names = staticmethod(KernelBenchCudaAgentBackend._extract_custom_kernel_names)

    @staticmethod
    def _normalize_cuda_sources_input(cuda_sources: Any) -> dict[str, str]:
        if not cuda_sources:
            return {}
        if not isinstance(cuda_sources, dict):
            raise TypeError("cuda_sources must be a dict[str, str]")
        return {str(name): str(content) for name, content in cuda_sources.items()}

    @staticmethod
    def _select_work_dir_parent() -> str | None:
        candidate = _TVM_FFI_DEFAULT_TMPDIR
        path = Path(candidate)
        KernelBenchCudaAgentBackend._require_fast_rw_path(
            path,
            label="TVM-FFI tmpdir",
        )
        try:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
                raise RuntimeError(f"TVM-FFI tmpdir is not writable/executable: {path}")
            if KernelBenchCudaAgentBackend._path_has_noexec_mount(path):
                raise RuntimeError(f"TVM-FFI tmpdir is mounted noexec: {path}")
            if shutil.disk_usage(path).free < _TVM_FFI_MIN_TMPDIR_FREE_BYTES:
                raise RuntimeError(f"TVM-FFI tmpdir has less than 512MiB free: {path}")
        except OSError as exc:
            raise RuntimeError(f"TVM-FFI tmpdir is not usable: {path}") from exc
        return str(path)

    def _create_work_dir(self) -> Path:
        parent = self._select_work_dir_parent()
        return Path(tempfile.mkdtemp(prefix="kernelgym_tvm_ffi_", dir=parent))

    @staticmethod
    def _write_runtime_scaffold(work_dir: Path, model_code: str) -> None:
        (work_dir / "__init__.py").write_text("", encoding="utf-8")
        (work_dir / "model_new.py").write_text(model_code, encoding="utf-8")

    @staticmethod
    def _collect_compile_sources(work_dir: Path) -> tuple[list[str], list[str]]:
        cpp_files: list[str] = []
        cuda_files: list[str] = []
        for path in work_dir.rglob("*"):
            if not path.is_file():
                continue
            if "build" in path.parts:
                continue
            suffix = path.suffix.lower()
            if suffix in {".cpp", ".cc", ".cxx"}:
                cpp_files.append(str(path))
            elif suffix == ".cu":
                cuda_files.append(str(path))
        return sorted(set(cpp_files)), sorted(set(cuda_files))

    @staticmethod
    def _extract_tvm_ffi_exports(cuda_sources: dict[str, str]) -> list[str]:
        export_pattern = re.compile(
            r"\bTVM_FFI_DLL_EXPORT_TYPED_FUNC\s*\(\s*([A-Za-z_]\w*)\s*,",
            re.MULTILINE,
        )
        exported: set[str] = set()
        for filename, content in cuda_sources.items():
            if filename.lower().endswith((".cpp", ".cc", ".cxx")):
                exported.update(export_pattern.findall(content))
        return sorted(exported)

    @staticmethod
    def _import_tvm_ffi() -> tuple[Any, Any]:
        try:
            tvm_ffi_api = importlib.import_module("tvm_ffi")
            tvm_ffi_cpp = importlib.import_module("tvm_ffi.cpp")
        except ImportError as exc:
            raise ImportError(
                "tvm_ffi is required for backend=tvm_ffi; install the tvm-ffi package "
                "in the reward/runtime environment"
            ) from exc
        return tvm_ffi_api, tvm_ffi_cpp

    @staticmethod
    def _build_extension(work_dir: Path, cpp_files: list[str], cuda_files: list[str]) -> Dict[str, Any]:
        if not cpp_files:
            return {
                "compiled": False,
                "error": "No TVM-FFI binding C++ source files found",
            }
        if not cuda_files:
            return {
                "compiled": False,
                "error": "No CUDA source files found for TVM-FFI compilation",
            }

        build_dir = work_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        _tvm_ffi_api, tvm_ffi_cpp = KernelBenchTvmFfiBackend._import_tvm_ffi()
        ext_name = work_dir.name.replace("-", "_")
        nvcc_threads = os.environ.get(_NVCC_THREADS_ENV, _TVM_FFI_DEFAULT_NVCC_THREADS)
        extra_cflags = ["-O3"]
        extra_cuda_cflags = ["-O3", "--use_fast_math", "--threads", nvcc_threads]
        all_sources = sorted(set(cpp_files + cuda_files))
        cache_root = KernelBenchCudaAgentBackend._compile_cache_root(
            label="TVM-FFI compile cache",
            disable_env=_TVM_FFI_COMPILE_CACHE_DISABLE_ENV,
            default_dir=_TVM_FFI_DEFAULT_COMPILE_CACHE_DIR,
        )
        cache_key = ""
        if cache_root is not None:
            cache_inputs = KernelBenchCudaAgentBackend._collect_cache_inputs_excluding(
                work_dir,
                all_sources,
                exclude_roots=[cache_root],
            )
            cache_metadata = {
                "backend": "tvm_ffi",
                "cuda_arch": KernelBenchCudaAgentBackend._cuda_arch_fingerprint(),
                "extra_cflags": extra_cflags,
                "extra_cuda_cflags": extra_cuda_cflags,
            }
            cache_key = KernelBenchCudaAgentBackend._source_digest(
                work_dir=work_dir,
                sources=cache_inputs,
                metadata=cache_metadata,
            )
            cache_entry = cache_root / cache_key
            ready_path = cache_entry / "ready.json"
            cached = KernelBenchCudaAgentBackend._load_cached_compile_result(ready_path)
            if cached is not None:
                return {
                    **cached,
                    "backend": "tvm_ffi",
                }
            with KernelBenchCudaAgentBackend._file_lock(cache_entry / "compile.lock"):
                cached = KernelBenchCudaAgentBackend._load_cached_compile_result(ready_path)
                if cached is not None:
                    return {
                        **cached,
                        "backend": "tvm_ffi",
                    }
                cached_sources = KernelBenchCudaAgentBackend._copy_sources_to_cache(
                    work_dir=work_dir,
                    sources=cache_inputs,
                    src_dir=cache_entry / "src",
                )
                cached_cpp_files = [
                    source for source in cached_sources if Path(source).suffix.lower() in {".cpp", ".cc", ".cxx"}
                ]
                cached_cuda_files = [source for source in cached_sources if Path(source).suffix.lower() == ".cu"]
                build_dir = cache_entry / "build"
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                build_dir.mkdir(parents=True, exist_ok=True)
                ext_name = f"kernelgym_tvm_ffi_{cache_key[:16]}"
                so_path = tvm_ffi_cpp.build(
                    name=ext_name,
                    cpp_files=cached_cpp_files,
                    cuda_files=cached_cuda_files,
                    build_directory=str(build_dir),
                    backend="cuda",
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                )
                ready_payload = {
                    "compiled": True,
                    "so_path": str(so_path),
                    "module_name": ext_name,
                    "compile_cache_key": cache_key,
                }
                ready_path.write_text(
                    json.dumps(ready_payload, sort_keys=True),
                    encoding="utf-8",
                )
                return {
                    "compiled": True,
                    "so_path": str(so_path),
                    "module_name": ext_name,
                    "compile_cache_hit": False,
                    "compile_cache_key": cache_key,
                    "compile_cache_dir": str(cache_entry),
                }

        so_path = tvm_ffi_cpp.build(
            name=ext_name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            build_directory=str(build_dir),
            backend="cuda",
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
        )
        return {
            "compiled": True,
            "so_path": str(so_path),
            "module_name": ext_name,
            "compile_cache_hit": False,
            "compile_cache_key": cache_key,
            "compile_cache_dir": "",
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
                "error": f"Failed to parse TVM-FFI submission: {exc}",
                "device": str(device),
                "entry_point": entry_point,
                "backend": "tvm_ffi",
            }

        cuda_sources = {**explicit_sources, **embedded_sources}
        model_code = python_code.strip() or code.strip()

        precheck_error, _precheck_code, precheck_info = precheck_tvm_ffi_submission(
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
                "backend": "tvm_ffi",
                "precheck": precheck_info,
            }

        work_dir = self._create_work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        self._write_runtime_scaffold(work_dir, model_code)
        self._materialize_sources(work_dir, cuda_sources)
        cpp_files, cuda_files = self._collect_compile_sources(work_dir)

        try:
            result = self._build_extension(work_dir, cpp_files, cuda_files)
        except Exception as exc:
            result = {"compiled": False, "error": str(exc)}

        return {
            "compiled": bool(result.get("compiled")),
            "error": result.get("error"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": "tvm_ffi",
            "work_dir": str(work_dir),
            "so_path": result.get("so_path"),
            "module_name": result.get("module_name"),
            "compile_cache_hit": result.get("compile_cache_hit"),
            "compile_cache_key": result.get("compile_cache_key"),
            "compile_cache_dir": result.get("compile_cache_dir"),
            "code": model_code,
            "precheck": precheck_info,
            "profiling_hints": {
                "backend": "tvm_ffi",
                "custom_kernel_names": self._extract_custom_kernel_names(cuda_sources),
                "detected_extension_calls": list(precheck_info.get("detected_extension_calls", [])),
                "exported_functions": self._extract_tvm_ffi_exports(cuda_sources),
                "source_files": sorted(cuda_sources.keys()),
            },
        }

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        work_dir = artifact.get("work_dir")
        so_path = artifact.get("so_path")
        module_name = artifact.get("module_name")
        context = kwargs.get("context") or {}

        if not code:
            raise ValueError("KernelBenchTvmFfiBackend.load requires kernel code in artifact")
        if not work_dir:
            raise ValueError("KernelBenchTvmFfiBackend.load requires a work_dir in artifact")
        if not so_path or not Path(so_path).exists():
            raise ValueError(f"Compiled shared library not found: {so_path}")
        if not module_name:
            raise ValueError("KernelBenchTvmFfiBackend.load requires module_name in artifact")

        device = self._normalize_device(kwargs.get("device") or artifact.get("device"))
        self._maybe_set_cuda_device(device)
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        tvm_ffi_api, _tvm_ffi_cpp = self._import_tvm_ffi()
        tvm_module = tvm_ffi_api.load_module(str(so_path), keep_module_alive=True)
        ext_module = _TvmFfiExtensionModule("tvm_ffi_extension", tvm_module, tvm_ffi_api)
        for func_name in artifact.get("profiling_hints", {}).get("exported_functions", []):
            try:
                getattr(ext_module, func_name)
            except AttributeError:
                pass

        work_dir_path = Path(work_dir)
        runtime_package_name = f"_kernelgym_tvm_ffi_{work_dir_path.name.replace('-', '_')}"
        package_module = types.ModuleType(runtime_package_name)
        package_module.__path__ = [str(work_dir_path)]  # type: ignore[attr-defined]
        package_module.__package__ = runtime_package_name
        sys.modules[runtime_package_name] = package_module

        module_aliases = [
            module_name,
            "tvm_ffi_extension",
            f"{runtime_package_name}.tvm_ffi_extension",
            runtime_package_name,
            f"{runtime_package_name}.model_new",
        ]
        sys.modules[module_name] = ext_module
        sys.modules["tvm_ffi_extension"] = ext_module
        sys.modules[f"{runtime_package_name}.tvm_ffi_extension"] = ext_module

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
            "backend": "tvm_ffi",
            "entry_point": entry_point,
            "device": device,
            "work_dir": str(work_dir_path),
            "so_path": str(so_path),
            "module_aliases": module_aliases,
            "tempfile_handle": None,
            "tvm_module": tvm_module,
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
