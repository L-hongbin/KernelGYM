"""KernelBench backend dispatcher."""

from __future__ import annotations

from typing import Any, Dict

from kernelgym.backend.base import Backend

from .cuda_agent_backend import KernelBenchCudaAgentBackend
from .cuda_backend import KernelBenchCudaBackend
from .tvm_ffi_backend import KernelBenchTvmFfiBackend
from .triton_backend import KernelBenchTritonBackend
from .tilelang_backend import KernelBenchTileLangBackend


class KernelBenchBackend(Backend):
    name = "kernelbench"

    def __init__(self) -> None:
        self._triton = KernelBenchTritonBackend()
        self._tilelang = KernelBenchTileLangBackend()
        self._cuda = KernelBenchCudaBackend()
        self._cuda_agent = KernelBenchCudaAgentBackend()
        self._tvm_ffi = KernelBenchTvmFfiBackend()

    @staticmethod
    def _resolve_backend_name(name: Any | None) -> str:
        key = (name or "triton").strip().lower()
        if key == "triton":
            return "triton"
        if key == "tilelang":
            return "tilelang"
        if key in ("cuda", "torch", "torch_compile", "torch-compile"):
            return "cuda"
        if key == "cuda_agent":
            return "cuda_agent"
        if key in ("tvm_ffi", "tvm-ffi", "tvmffi"):
            return "tvm_ffi"
        raise ValueError(f"Unsupported KernelBench backend '{name}'")

    def _select(self, name: Any | None) -> Backend:
        backend = self._resolve_backend_name(name)
        if backend == "triton":
            return self._triton
        if backend == "tilelang":
            return self._tilelang
        if backend == "cuda_agent":
            return self._cuda_agent
        if backend == "tvm_ffi":
            return self._tvm_ffi
        if backend == "cuda":
            return self._cuda
        raise ValueError(f"Unsupported KernelBench backend '{name}'")

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        backend_name = kwargs.get("backend", "triton")
        backend = self._select(backend_name)
        artifact = backend.compile(code, **kwargs)
        if isinstance(artifact, dict):
            artifact.setdefault("backend", self._resolve_backend_name(backend_name))
        return artifact

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        backend_name = artifact.get("backend") if isinstance(artifact, dict) else None
        if backend_name is None:
            backend_name = kwargs.get("backend", "triton")
        backend = self._select(backend_name)
        handle = backend.load(artifact, **kwargs)
        if isinstance(handle, dict):
            handle.setdefault("backend", self._resolve_backend_name(backend_name))
        return handle

    def create_model(self, handle: Any, init_inputs: Any, **kwargs: Any) -> Any:
        backend_name = handle.get("backend") if isinstance(handle, dict) else None
        backend = self._select(backend_name or kwargs.get("backend"))
        return backend.create_model(handle, init_inputs, **kwargs)

    def run(self, handle: Any, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        backend_name = handle.get("backend") if isinstance(handle, dict) else None
        backend = self._select(backend_name or kwargs.get("backend"))
        return backend.run(handle, inputs, **kwargs)

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        backend_name = handle.get("backend") if isinstance(handle, dict) else None
        backend = self._select(backend_name or kwargs.get("backend"))
        backend.cleanup(handle, **kwargs)

    def close(self, handle: Any, **kwargs: Any) -> None:
        self.cleanup(handle, **kwargs)
