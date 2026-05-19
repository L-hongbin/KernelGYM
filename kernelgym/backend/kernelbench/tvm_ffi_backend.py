"""TVM-FFI-specific KernelBench backend implementation."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import hashlib
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
_COMPILE_ARTIFACT_CACHE_ENV = "KERNELGYM_COMPILE_ARTIFACT_CACHE"
_TVM_FFI_COMPILE_ARTIFACT_CACHE_DIR_ENV = "KERNELGYM_TVM_FFI_COMPILE_ARTIFACT_CACHE_DIR"
_TVM_FFI_DEFAULT_ARTIFACT_CACHE_DIR = "/dev/shm/kernelgym/compile_cache/tvm_ffi_artifacts"


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
    def _tvm_ffi_version() -> str:
        for package_name in ("apache-tvm-ffi", "tvm-ffi"):
            try:
                return importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return "unknown"

    @staticmethod
    def _compile_artifact_cache_enabled(kwargs: dict[str, Any]) -> bool:
        if kwargs.get("enable_compile_artifact_cache") is not None:
            return bool(kwargs.get("enable_compile_artifact_cache"))
        return KernelBenchCudaAgentBackend._env_flag(_COMPILE_ARTIFACT_CACHE_ENV, default=False)

    @staticmethod
    def _compile_artifact_cache_root() -> Path:
        root = Path(os.environ.get(_TVM_FFI_COMPILE_ARTIFACT_CACHE_DIR_ENV, _TVM_FFI_DEFAULT_ARTIFACT_CACHE_DIR))
        KernelBenchCudaAgentBackend._require_fast_rw_path(root, label="TVM-FFI compile artifact cache")
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _artifact_cache_key(
        *,
        model_code: str,
        cuda_sources: dict[str, str],
        entry_point: str,
    ) -> str:
        payload = {
            "backend": "tvm_ffi",
            "entry_point": entry_point,
            "model_code": model_code,
            "cuda_sources": {key: cuda_sources[key] for key in sorted(cuda_sources)},
            "tvm_ffi_version": KernelBenchTvmFfiBackend._tvm_ffi_version(),
            "python": sys.version,
            "cuda_arch": KernelBenchCudaAgentBackend._cuda_arch_fingerprint(),
            "nvcc_threads": os.environ.get(_NVCC_THREADS_ENV, _TVM_FFI_DEFAULT_NVCC_THREADS),
            "extra_cflags": ["-O3"],
            "extra_cuda_cflags": ["-O3", "--use_fast_math"],
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
        payload["compile_cache_hit"] = True
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
                "compile_artifact_cache_key",
                "compile_cache_key",
                "compile_artifact_cache_dir",
                "compile_cache_dir",
            }
        }
        payload["persistent_work_dir"] = True
        tmp_path = ready_path.with_suffix(ready_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(ready_path)

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
            "build_backend": "tvm_ffi.cpp.build",
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

        profiling_hints = {
            "backend": "tvm_ffi",
            "custom_kernel_names": self._extract_custom_kernel_names(cuda_sources),
            "detected_extension_calls": list(precheck_info.get("detected_extension_calls", [])),
            "exported_functions": self._extract_tvm_ffi_exports(cuda_sources),
            "source_files": sorted(cuda_sources.keys()),
        }

        work_dir: Path
        cache_key = ""
        cache_dir = ""
        ready_path: Path | None = None
        if enable_compile_artifact_cache:
            cache_key = self._artifact_cache_key(
                model_code=model_code,
                cuda_sources=cuda_sources,
                entry_point=entry_point,
            )
            cache_entry = self._compile_artifact_cache_root() / cache_key
            cache_dir = str(cache_entry)
            ready_path = cache_entry / "ready.json"
            cached_artifact = self._load_cached_artifact(ready_path)
            if cached_artifact is not None:
                cached_artifact["device"] = str(device)
                cached_artifact["compile_artifact_cache_enabled"] = True
                cached_artifact["compile_artifact_cache_key"] = cache_key
                cached_artifact["compile_artifact_cache_dir"] = cache_dir
                cached_artifact["compile_cache_key"] = cache_key
                cached_artifact["compile_cache_dir"] = cache_dir
                return cached_artifact
            work_dir = cache_entry / f"kernelgym_tvm_ffi_{cache_key[:16]}"
        else:
            work_dir = self._create_work_dir()

        def _compile_in_work_dir() -> dict[str, Any]:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            self._write_runtime_scaffold(work_dir, model_code)
            self._materialize_sources(work_dir, cuda_sources)
            cpp_files, cuda_files = self._collect_compile_sources(work_dir)
            try:
                return self._build_extension(work_dir, cpp_files, cuda_files)
            except Exception as exc:
                return {"compiled": False, "error": str(exc)}

        if enable_compile_artifact_cache and ready_path is not None:
            with KernelBenchCudaAgentBackend._file_lock(ready_path.parent / "compile.lock"):
                cached_artifact = self._load_cached_artifact(ready_path)
                if cached_artifact is not None:
                    cached_artifact["device"] = str(device)
                    cached_artifact["compile_artifact_cache_enabled"] = True
                    cached_artifact["compile_artifact_cache_key"] = cache_key
                    cached_artifact["compile_artifact_cache_dir"] = cache_dir
                    cached_artifact["compile_cache_key"] = cache_key
                    cached_artifact["compile_cache_dir"] = cache_dir
                    return cached_artifact
                result = _compile_in_work_dir()
        else:
            result = _compile_in_work_dir()

        artifact = {
            "compiled": bool(result.get("compiled")),
            "error": result.get("error"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": "tvm_ffi",
            "work_dir": str(work_dir),
            "so_path": result.get("so_path"),
            "module_name": result.get("module_name"),
            "build_backend": result.get("build_backend", "tvm_ffi.cpp.build"),
            "compile_artifact_cache_enabled": enable_compile_artifact_cache,
            "compile_artifact_cache_hit": False,
            "compile_artifact_cache_key": cache_key or None,
            "compile_artifact_cache_dir": cache_dir or None,
            "compile_cache_hit": False,
            "compile_cache_key": cache_key or None,
            "compile_cache_dir": cache_dir or None,
            "persistent_work_dir": enable_compile_artifact_cache,
            "code": model_code,
            "precheck": precheck_info,
            "profiling_hints": profiling_hints,
        }
        if enable_compile_artifact_cache and ready_path is not None and artifact["compiled"]:
            self._write_cached_artifact(ready_path, artifact)
        return artifact

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
