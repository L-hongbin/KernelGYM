"""KernelBench timing helpers (toolkit layer)."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from kernelgym.config import settings
from kernelgym.toolkit.kernelbench.profiling import (
    extract_profiling_metrics,
    profiling_context,
)
from kernelgym.utils import cupti_tsc_shim

logger = logging.getLogger(__name__)

# NVIDIA-confirmed CUPTI defect: with cuptiActivityRegisterTimestampCallback
# (Kineto's TSC fast path), kernel start timestamps can be written as 0, so
# Kineto drops the record as out-of-window and the reward profiler sees zero
# CUDA kernels. Introduced in CUDA 12.6 Update 2, fixed in CUDA 13.1. The
# CUDA version string cannot distinguish 12.6 GA/U1/U2, so the gate
# conservatively covers all of 12.6.
_CUPTI_TSC_BUG_MIN_CUDA = (12, 6)
_CUPTI_TSC_BUG_FIXED_CUDA = (13, 1)
_LEGACY_PROFILING_TRIALS_CAP = 10


def _parse_cuda_version(version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not version:
        return None
    parts = str(version).split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    return (major, minor)


def cupti_tsc_timestamp_bug_suspected(
    cuda_version: Optional[str] = None,
    kineto_tsc_fixed: Optional[bool] = None,
) -> bool:
    """Whether the loaded CUDA/CUPTI can emit start=0 kernel timestamps under Kineto's TSC callback.

    Fail-safe: an unknown or unparseable CUDA version counts as affected, so the
    legacy multi-forward workaround stays active rather than risking empty captures.
    """
    if kineto_tsc_fixed is None:
        kineto_tsc_fixed = settings.kineto_tsc_fixed
    if kineto_tsc_fixed:
        return False
    if cuda_version is None:
        cuda_version = getattr(torch.version, "cuda", None)
    parsed = _parse_cuda_version(cuda_version)
    if parsed is None:
        return True
    return _CUPTI_TSC_BUG_MIN_CUDA <= parsed < _CUPTI_TSC_BUG_FIXED_CUDA


def resolve_num_profiling_trials(
    num_trials: int,
    configured: Optional[int] = None,
    cuda_version: Optional[str] = None,
    kineto_tsc_fixed: Optional[bool] = None,
) -> int:
    """Resolve how many extra candidate forwards to run inside one profiler context.

    An explicit configuration (>= 1) wins. Otherwise auto mode: a single forward is
    enough for coverage/decoy semantics, but while the CUPTI TSC timestamp bug is
    suspected we keep the legacy min(10, num_trials) workaround because a single
    forward captures no CUDA kernels for slow kernels most of the time.
    """
    if configured is None:
        configured = settings.num_profiling_trials
    if configured >= 1:
        return configured
    if cupti_tsc_timestamp_bug_suspected(cuda_version=cuda_version, kineto_tsc_fixed=kineto_tsc_fixed):
        return min(_LEGACY_PROFILING_TRIALS_CAP, max(1, num_trials))
    return 1


def kineto_tsc_fix_verified() -> Optional[bool]:
    """Whether the declared Kineto TSC fix is actually active in this process.

    Meaningful only after a profiler context has run (the LD_PRELOAD shim's
    state is decided when Kineto first registers its timestamp callback).
    None when no fix is declared; False when a shim was expected but did not
    engage — retries must then fall back to the legacy multi-forward count.
    """
    return cupti_tsc_shim.kineto_tsc_fix_verified(settings.kineto_tsc_fixed)


def _annotate_shim_state(profiling_metrics: Dict[str, Any]) -> None:
    if cupti_tsc_shim.expected_shim_path() is None:
        return
    state = cupti_tsc_shim.shim_state()
    profiling_metrics["cupti_tsc_shim_state"] = state
    if not cupti_tsc_shim.shim_state_healthy(state):
        logger.error(
            "[Profiling] CUPTI TSC shim expected but not engaged (state=%s); "
            "single-forward profiling cannot be trusted in this process",
            state,
        )


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    enable_profiling: bool = False,
<<<<<<< HEAD
    num_profiling_trials: Optional[int] = None,
=======
    adaptive: bool = False,
    min_trials: int = 20,
    cv_threshold: float = 0.05,
>>>>>>> 7342b02 (feat(kernelbench): adaptive kernel perf trials + separate reference trial count)
) -> Tuple[List[float], Dict[str, Any], Dict[str, Any]]:
    if device is None:
        if verbose:
            logger.info("Using current device: %s", torch.cuda.current_device())
        device = torch.cuda.current_device()

    overall_start = perf_counter()

    if num_profiling_trials is None or num_profiling_trials < 1:
        num_profiling_trials = resolve_num_profiling_trials(num_trials)

    profiling_metrics: Dict[str, Any] = {}
    profiling_wall_s = 0.0

    # Disable autograd for the whole measurement window. The forward passes here
    # are pure inference; without this, models that hold nn.Parameter (which
    # default to requires_grad=True) would build and immediately discard an
    # autograd graph on every trial, wasting memory and adding timing noise.
    # Use no_grad (not the stricter inference_mode) to keep the same autograd
    # semantics as the correctness checks, so a kernel that passes correctness
    # is timed under identical conditions.
    with torch.no_grad():
        warmup_start = perf_counter()
        for _ in range(num_warmup):
            kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        warmup_wall_s = perf_counter() - warmup_start

        logger.debug(
            "[Profiling] Using device: %s %s, warm up %s, trials %s",
            device,
            torch.cuda.get_device_name(device),
            num_warmup,
            num_trials,
        )
        elapsed_times = []

        # Adaptive trial count: always run at least `min_trials`, then keep going
        # only while the coefficient of variation (std/mean) stays above
        # `cv_threshold` (i.e. the timing is not yet stable), up to `num_trials`
        # as the hard maximum. This caps the cost of timing slow-but-correct
        # kernels, whose per-iter time dominates the eval budget.
        max_trials = max(1, num_trials)
        effective_min = max(1, min(min_trials, max_trials)) if adaptive else max_trials
        stopped_early = False
        final_cv: float | None = None

        measure_start = perf_counter()
        for trial in range(max_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            kernel_fn(*args)
            end_event.record()

            torch.cuda.synchronize(device=device)

            elapsed_time_ms = start_event.elapsed_time(end_event)
            if verbose:
                logger.info("Trial %s: %.3g ms", trial + 1, elapsed_time_ms)
            elapsed_times.append(elapsed_time_ms)

            if adaptive and (trial + 1) >= effective_min:
                mean_ms = float(np.mean(elapsed_times))
                std_ms = float(np.std(elapsed_times))
                final_cv = (std_ms / mean_ms) if mean_ms > 0 else 0.0
                if final_cv <= cv_threshold:
                    stopped_early = True
                    break
        measure_wall_s = perf_counter() - measure_start

        if enable_profiling:
            try:
                torch.cuda.synchronize(device=device)

                logger.info("[Profiling] Running %s additional iterations for profiling...", num_profiling_trials)

                profiling_start = perf_counter()
                with profiling_context(True) as prof:
                    for _ in range(num_profiling_trials):
                        kernel_fn(*args)
                    torch.cuda.synchronize(device=device)
                profiling_wall_s = perf_counter() - profiling_start

                profiling_metrics = extract_profiling_metrics(prof)
                if profiling_metrics:
                    _annotate_shim_state(profiling_metrics)
                    logger.info("[Profiling] Captured %s CUDA kernels", profiling_metrics.get("kernel_count", 0))
                    logger.info("[Profiling] Total CUDA time: %.2f us", profiling_metrics.get("total_cuda_time_us", 0))

            except Exception as e:
                logger.warning("[Profiling] Profiling failed: %s", e)
                profiling_metrics = {"profiling_error": str(e)}

    timing_info = {
        "warmup_wall_s": warmup_wall_s,
        "measure_wall_s": measure_wall_s,
        "profiling_wall_s": profiling_wall_s,
        "timed_trials_cuda_event_s": sum(elapsed_times) / 1000.0,
        "num_warmup": num_warmup,
<<<<<<< HEAD
        "num_trials": num_trials,
        "num_profiling_trials": num_profiling_trials if enable_profiling else 0,
=======
        # `num_trials` is the number actually run (may be < requested when adaptive
        # early-stops on a stable CV); `num_trials_requested` is the configured max.
        "num_trials": len(elapsed_times),
        "num_trials_requested": num_trials,
        "adaptive_perf_trials": adaptive,
        "adaptive_min_trials": effective_min if adaptive else None,
        "adaptive_cv_threshold": cv_threshold if adaptive else None,
        "adaptive_stopped_early": stopped_early,
        "adaptive_final_cv": final_cv,
        "num_profiling_trials": min(10, num_trials) if enable_profiling else 0,
>>>>>>> 7342b02 (feat(kernelbench): adaptive kernel perf trials + separate reference trial count)
        "total_wall_s": perf_counter() - overall_start,
    }

    return elapsed_times, profiling_metrics, timing_info


def run_profiling_only(
    kernel_fn: callable,
    *args,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> Dict[str, Any]:
    if device is None:
        if verbose:
            logger.info("Using current device: %s", torch.cuda.current_device())
        device = torch.cuda.current_device()

    profiling_metrics: Dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device=device)
        logger.info("[Profiling] Running %s iterations (profiling-only)...", num_trials)
        with profiling_context(True) as prof:
            for _ in range(num_trials):
                kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        profiling_metrics = extract_profiling_metrics(prof)
        if profiling_metrics:
            _annotate_shim_state(profiling_metrics)
            logger.info("[Profiling] Captured %s CUDA kernels", profiling_metrics.get("kernel_count", 0))
    except Exception as e:
        logger.warning("[Profiling] Profiling-only failed: %s", e)
        profiling_metrics = {"profiling_error": str(e)}

    return profiling_metrics


def get_timing_stats(
    elapsed_times: List[float],
    device: torch.device = None,
    trim_count: int = 0,
) -> dict:
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if trim_count > 0 and len(elapsed_times) > 2 * trim_count:
        sorted_times = sorted(elapsed_times)
        trimmed = sorted_times[trim_count:-trim_count]
        stats["trimmed_mean"] = float(f"{np.mean(trimmed):.3g}")
        stats["trimmed_std"] = float(f"{np.std(trimmed):.3g}")
        stats["trim_count"] = trim_count
        # Use trimmed mean as the primary mean
        stats["raw_mean"] = stats["mean"]
        stats["mean"] = stats["trimmed_mean"]

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)

    return stats
