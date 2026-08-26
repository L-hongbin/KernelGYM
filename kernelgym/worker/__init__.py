"""Worker module for KernelGym."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cpu_worker import CPUCompileWorker
    from .gpu_worker import GPUWorker, WorkerManager

__all__ = ["CPUCompileWorker", "GPUWorker", "WorkerManager"]


def __getattr__(name: str) -> Any:
    """Preserve public imports without eagerly loading Torch during spawn."""

    if name == "CPUCompileWorker":
        from .cpu_worker import CPUCompileWorker

        return CPUCompileWorker
    if name in {"GPUWorker", "WorkerManager"}:
        from .gpu_worker import GPUWorker, WorkerManager

        return {"GPUWorker": GPUWorker, "WorkerManager": WorkerManager}[name]
    raise AttributeError(name)
