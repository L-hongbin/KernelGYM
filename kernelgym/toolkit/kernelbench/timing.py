"""KernelBench timing helpers (toolkit layer)."""

from __future__ import annotations

from functools import partial
from importlib.metadata import PackageNotFoundError, version as importlib_metadata_version
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from kernelgym.utils.device_info import get_cuda_device_info
from kernelgym.toolkit.kernelbench.profiling import (
    extract_profiling_metrics,
    profiling_context,
)
from kernelgym.toolkit.kernelbench.flashinfer_testing_utils_local import (
    bench_gpu_time_with_cupti as _flashinfer_bench_gpu_time_with_cupti,
)

_device_locks: dict[str, LockType] = {}
_registry_lock = Lock()


def _device_lock(device_key: str) -> LockType:
    with _registry_lock:
        lock = _device_locks.get(device_key)
        if lock is None:
            lock = Lock()
            _device_locks[device_key] = lock
        return lock


def _normalize_cuda_device(device: torch.device | int | str | None, verbose: bool) -> torch.device:
    if device is None:
        current_device = torch.cuda.current_device()
        if verbose:
            print(f"Using current device: {current_device}")
        return torch.device(f"cuda:{current_device}")

    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError(f"Expected a CUDA device, got: {device}")
        if device.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return device

    if isinstance(device, int):
        return torch.device(f"cuda:{device}")

    normalized = torch.device(device)
    if normalized.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got: {device}")
    if normalized.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return normalized


def _run_additional_profiling(
    kernel_fn: callable,
    *args,
    num_trials: int,
    device: torch.device,
) -> Dict[str, Any]:
    profiling_metrics: Dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device=device)

        num_profiling_trials = min(10, num_trials)
        print(
            f"[Profiling] Running {num_profiling_trials} additional iterations for profiling..."
        )

        with profiling_context(True) as prof:
            for _ in range(num_profiling_trials):
                kernel_fn(*args)
            torch.cuda.synchronize(device=device)

        profiling_metrics = extract_profiling_metrics(prof)
        if profiling_metrics:
            print(
                f"[Profiling] Captured {profiling_metrics.get('kernel_count', 0)} CUDA kernels"
            )
            print(
                f"[Profiling] Total CUDA time: {profiling_metrics.get('total_cuda_time_us', 0):.2f} us"
            )
    except Exception as e:
        print(f"[Profiling] Warning: Profiling failed: {e}")
        profiling_metrics = {"profiling_error": str(e)}

    return profiling_metrics


def _load_cupti_module():
    try:
        from cupti import cupti
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CUPTI timing requires cupti-python. Install it first, for example: pip install -U cupti-python"
        ) from exc

    try:
        cupti_version = importlib_metadata_version("cupti-python")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "CUPTI timing requires the cupti-python package metadata, but it was not found."
        ) from exc

    return cupti, cupti_version


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    enable_profiling: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    device = _normalize_cuda_device(device, verbose=verbose)

    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        torch.cuda.synchronize(device=device)

        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    profiling_metrics: Dict[str, Any] = {}
    if enable_profiling:
        profiling_metrics = _run_additional_profiling(
            kernel_fn,
            *args,
            num_trials=num_trials,
            device=device,
        )

    return elapsed_times, profiling_metrics


def time_execution_with_cupti(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    enable_profiling: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    device = _normalize_cuda_device(device, verbose=verbose)
    cupti, cupti_version = _load_cupti_module()

    print(
        f"[CUPTI] Using device: {device} {torch.cuda.get_device_name(device)}, cupti-python {cupti_version}, warm up {num_warmup}, trials {num_trials}"
    )

    with _device_lock(str(device)):
        with torch.cuda.device(device):
            torch.cuda.synchronize(device=device)

            for _ in range(num_warmup):
                kernel_fn(*args)
                torch.cuda.synchronize(device=device)

            kernels: list[tuple[str, float, float, int]] = []
            iter_timestamps: list[tuple[float, float]] = []

            def func_buffer_requested():
                buffer_size = 8 * 1024 * 1024
                max_num_records = 0
                return buffer_size, max_num_records

            def func_buffer_completed(
                kernel_records: list[tuple[str, float, float, int]],
                activities: list,
            ):
                for activity in activities:
                    if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                        kernel_records.append(
                            (
                                activity.name,
                                activity.start,
                                activity.end,
                                activity.correlation_id,
                            )
                        )

            cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
            cupti.activity_register_callbacks(
                func_buffer_requested,
                partial(func_buffer_completed, kernels),
            )

            try:
                for _ in range(num_trials):
                    start_cpu = cupti.get_timestamp()
                    kernel_fn(*args)
                    torch.cuda.synchronize(device=device)
                    end_cpu = cupti.get_timestamp()
                    iter_timestamps.append((start_cpu, end_cpu))
            finally:
                cupti.activity_flush_all(0)
                cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
                cupti.finalize()

            elapsed_times: list[float] = []
            for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
                iter_kernels = [
                    kernel
                    for kernel in kernels
                    if not (kernel[2] < start_cpu or kernel[1] > end_cpu)
                ]
                if not iter_kernels:
                    raise RuntimeError(f"No kernel activities recorded for iteration {idx}")

                min_start = min(kernel[1] for kernel in iter_kernels)
                max_end = max(kernel[2] for kernel in iter_kernels)
                elapsed_times.append(float((max_end - min_start) / 1e6))

            torch.cuda.synchronize(device=device)

    if verbose:
        for trial, elapsed_time_ms in enumerate(elapsed_times, start=1):
            print(f"Trial {trial}: {elapsed_time_ms:.3g} ms")

    profiling_metrics: Dict[str, Any] = {}
    if enable_profiling:
        profiling_metrics = _run_additional_profiling(
            kernel_fn,
            *args,
            num_trials=num_trials,
            device=device,
        )

    return elapsed_times, profiling_metrics


def time_execution_with_cupti_flashinfer_bench(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    enable_profiling: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    device = _normalize_cuda_device(device, verbose=verbose)

    print(
        f"[CUPTI:flashinfer-bench] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )

    # with _device_lock(str(device)):
        # with torch.cuda.device(device):
    torch.cuda.synchronize(device=device)
    elapsed_times = _flashinfer_bench_gpu_time_with_cupti(
        fn=kernel_fn,
        dry_run_iters=num_warmup,
        repeat_iters=num_trials,
        input_args=tuple(args),
        l2_flush=True,
        use_cuda_graph=False,
    )
    torch.cuda.synchronize(device=device)

    elapsed_times = [float(t) for t in elapsed_times]
    if verbose:
        for trial, elapsed_time_ms in enumerate(elapsed_times, start=1):
            print(f"Trial {trial}: {elapsed_time_ms:.3g} ms")

    profiling_metrics: Dict[str, Any] = {}
    if enable_profiling:
        profiling_metrics = _run_additional_profiling(
            kernel_fn,
            *args,
            num_trials=num_trials,
            device=device,
        )

    return elapsed_times, profiling_metrics


def run_profiling_only(
    kernel_fn: callable,
    *args,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> Dict[str, Any]:
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    profiling_metrics: Dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device=device)
        print(f"[Profiling] Running {num_trials} iterations (profiling-only)...")
        with profiling_context(True) as prof:
            for _ in range(num_trials):
                kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        profiling_metrics = extract_profiling_metrics(prof)
        if profiling_metrics:
            print(
                f"[Profiling] Captured {profiling_metrics.get('kernel_count', 0)} CUDA kernels"
            )
    except Exception as e:
        print(f"[Profiling] Warning: Profiling-only failed: {e}")
        profiling_metrics = {"profiling_error": str(e)}

    return profiling_metrics


def get_timing_stats(elapsed_times: List[float], device: torch.device = None) -> dict:
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["device_info"] = get_cuda_device_info(device)
        stats["device"] = str(device)

    return stats
