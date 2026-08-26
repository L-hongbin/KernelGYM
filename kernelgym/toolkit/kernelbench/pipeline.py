"""KernelBench evaluation pipeline (task-level, toolkit layer)."""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from time import monotonic_ns, perf_counter, time
from typing import Any, Dict, Optional, Union

import torch

from kernelgym.config import settings
from kernelgym.toolkit.kernelbench import triton_detect as detect
from kernelgym.toolkit.kernelbench.compute_sanitizer import (
    FULL_SANITIZER_TOOLS,
    SANITIZER_MODE_FULL,
    classify_compute_sanitizer_error,
    run_compute_sanitizer,
    skipped_compute_sanitizer_result,
)
from kernelgym.toolkit.kernelbench.correctness import run_and_check_correctness
from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)
from kernelgym.toolkit.kernelbench.loading import (
    OriginalModelLoadError,
    graceful_eval_cleanup,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
)
from kernelgym.toolkit.kernelbench.memory import (
    capture_cuda_memory_environment_floor,
    detect_direct_cuda_allocations,
    measure_cuda_memory_trial,
)
from kernelgym.toolkit.kernelbench.ncu_profiler import (
    run_ncu_profile,
    select_kernel_names,
    skipped_ncu_result,
)
from kernelgym.toolkit.kernelbench.profiling import (
    compute_named_kernel_coverage,
    compute_triton_kernel_coverage,
)
from kernelgym.toolkit.kernelbench.timing import (
    get_timing_stats,
    kineto_tsc_fix_verified as timing_kineto_tsc_fix_verified,
    resolve_num_profiling_trials,
    run_profiling_only,
    time_execution_with_cuda_event,
)
from kernelgym.utils.error_classifier import classify_compile_error_detail

logger = logging.getLogger(__name__)
_STAGE_METADATA_PATH_ENV = "KERNELGYM_STAGE_METADATA_PATH"
_FAST_RW_ROOT = Path("/dev/shm")
_HARD_DECOY_COVERAGE_THRESHOLD = 0.001
_SUSPECTED_DECOY_COVERAGE_THRESHOLD = 0.30


def _path_is_under_fast_rw_root(path: Path) -> bool:
    try:
        resolved_path = path.resolve(strict=False)
        resolved_root = _FAST_RW_ROOT.resolve(strict=False)
    except OSError:
        resolved_path = path.absolute()
        resolved_root = _FAST_RW_ROOT.absolute()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _write_stage_metadata(metadata: Dict[str, Any]) -> None:
    path_value = os.environ.get(_STAGE_METADATA_PATH_ENV)
    if not path_value:
        return
    path = Path(path_value)
    if not _path_is_under_fast_rw_root(path):
        raise ValueError(f"{_STAGE_METADATA_PATH_ENV} must be under /dev/shm for fast local I/O: {path}")
    try:
        now_unix = time()
        now_mono = monotonic_ns()
        current_start = metadata.get("kg_stage_current_started_monotonic_ns")
        total_start = metadata.get("kg_stage_total_started_monotonic_ns")
        current_elapsed = metadata.get("kg_stage_current_elapsed_s")
        if metadata.get("kg_stage_is_active") and isinstance(current_start, int):
            current_elapsed = max(0.0, (now_mono - current_start) / 1e9)
        total_elapsed = metadata.get("kg_stage_total_elapsed_s")
        if isinstance(total_start, int):
            total_elapsed = max(0.0, (now_mono - total_start) / 1e9)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kg_stage_current": metadata.get("kg_stage_current"),
            "kg_stage_current_prefix": metadata.get("kg_stage_current_prefix"),
            "kg_stage_is_active": metadata.get("kg_stage_is_active"),
            "kg_stage_current_elapsed_s": current_elapsed,
            "kg_stage_total_elapsed_s": total_elapsed,
            "kg_stage_current_started_at_unix_s": metadata.get("kg_stage_current_started_at_unix_s"),
            "kg_stage_current_started_monotonic_ns": metadata.get("kg_stage_current_started_monotonic_ns"),
            "kg_stage_last_update_at_unix_s": now_unix,
            "kg_stage_last_update_monotonic_ns": now_mono,
            "kg_stage_total_started_at_unix_s": metadata.get("kg_stage_total_started_at_unix_s"),
            "kg_stage_total_started_monotonic_ns": metadata.get("kg_stage_total_started_monotonic_ns"),
            "kg_stage_completed_s": metadata.get("kg_stage_completed_s", {}),
            "kg_stage_last_completed": metadata.get("kg_stage_last_completed"),
            "kg_stage_metadata_path": str(path),
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        return


def _begin_stage(
    metadata: Dict[str, Any],
    *,
    prefix: str,
    stage: str,
    overall_start: float,
) -> float:
    now_perf = perf_counter()
    now_mono = monotonic_ns()
    metadata.setdefault("kg_stage_completed_s", {})
    metadata.setdefault(
        "kg_stage_total_started_at_unix_s",
        time() - (now_perf - overall_start),
    )
    metadata.setdefault(
        "kg_stage_total_started_monotonic_ns",
        now_mono - int((now_perf - overall_start) * 1e9),
    )
    metadata["kg_stage_current_prefix"] = prefix
    metadata["kg_stage_current"] = stage
    metadata["kg_stage_is_active"] = True
    metadata["kg_stage_current_started_at_unix_s"] = time()
    metadata["kg_stage_current_started_monotonic_ns"] = now_mono
    metadata["kg_stage_current_elapsed_s"] = 0.0
    _write_stage_metadata(metadata)
    return now_perf


def _record_phase_timing(metadata: Dict[str, Any], key: str, start_time: float) -> float:
    elapsed = perf_counter() - start_time
    metadata[key] = elapsed
    return elapsed


def _finish_stage(
    metadata: Dict[str, Any],
    *,
    stage: str,
    timing_key: str,
    start_time: float,
) -> float:
    elapsed = _record_phase_timing(metadata, timing_key, start_time)
    completed = metadata.setdefault("kg_stage_completed_s", {})
    if isinstance(completed, dict):
        completed[stage] = elapsed
    metadata["kg_stage_last_completed"] = stage
    metadata["kg_stage_is_active"] = False
    metadata["kg_stage_current_elapsed_s"] = elapsed
    _write_stage_metadata(metadata)
    return elapsed


def _sync_exec_result_metadata(result: Optional[KernelExecResult], metadata: Dict[str, Any]) -> None:
    if result is not None and isinstance(result.metadata, dict):
        result.metadata.update(metadata)


def _record_model_load_error(metadata: Dict[str, Any], exc: Exception) -> KernelExecResult:
    metadata["model_load_error"] = str(exc)
    metadata["model_load_error_name"] = get_error_name(exc)
    if exc.__cause__ is not None:
        metadata["model_load_error_cause"] = str(exc.__cause__)
        metadata["model_load_error_cause_name"] = get_error_name(exc.__cause__)
    return KernelExecResult(compiled=False, correctness=False, metadata=metadata)


def _sanitize_compile_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    hidden = {
        "work_dir",
        "so_path",
        "code",
        "compile_artifact_cache_key",
        "compile_cache_key",
        "compile_cache_dir",
        "persistent_work_dir",
    }
    sanitized = {key: value for key, value in artifact.items() if key not in hidden}
    timing = sanitized.get("compile_timing")
    if isinstance(timing, dict):
        timing = dict(timing)
        object_cache = timing.get("manual_ninja_object_cache")
        if isinstance(object_cache, dict):
            object_cache = dict(object_cache)
            for item in object_cache.get("objects") or []:
                if isinstance(item, dict):
                    item.pop("cache_path", None)
                    item.pop("local_object", None)
                    item.pop("lock_path", None)
                    item.pop("cache_key", None)
            object_cache.pop("root", None)
            timing["manual_ninja_object_cache"] = object_cache
        sanitized["compile_timing"] = timing
    return sanitized


def _copy_compile_artifact_metadata(metadata: Dict[str, Any], artifact: Dict[str, Any]) -> None:
    for artifact_key in (
        "build_backend",
        "compile_timing",
        "compile_artifact_cache_enabled",
        "compile_artifact_cache_hit",
    ):
        if artifact_key in artifact:
            metadata[artifact_key] = artifact.get(artifact_key)
    metadata["compile_artifact"] = _sanitize_compile_artifact(artifact)


def _apply_coverage_metadata(
    *,
    metadata: Dict[str, Any],
    kernel_exec_result: KernelExecResult,
    coverage_result_dict: Dict[str, Any],
    coverage_backend: str,
    detect_decoy_kernel: bool,
) -> None:
    num_custom_kernels = coverage_result_dict["num_custom_kernels"]
    num_total_kernels = coverage_result_dict["num_total_kernels"]
    custom_kernels_not_in_profiling = coverage_result_dict.get("custom_kernels_not_in_profiling", [])
    custom_kernels_in_profiling = coverage_result_dict.get("custom_kernels_in_profiling", [])
    total_kernel_run_time_in_profiling_us = coverage_result_dict["total_kernel_run_time_in_profiling_us"]
    total_kernel_cuda_time_in_profiling_us = coverage_result_dict.get(
        "total_kernel_cuda_time_in_profiling_us",
        total_kernel_run_time_in_profiling_us,
    )
    total_kernel_run_time_in_profiling_us_cpu_cuda = coverage_result_dict.get(
        "total_kernel_run_time_in_profiling_us_cpu_cuda",
        total_kernel_run_time_in_profiling_us,
    )
    custom_kernel_cuda_time_in_profiling_us = coverage_result_dict["custom_kernel_cuda_time_in_profiling_us"]

    metadata["coverage_backend"] = coverage_backend
    metadata["num_custom_kernels"] = num_custom_kernels
    metadata["num_total_kernels"] = num_total_kernels
    ratio = num_custom_kernels / num_total_kernels if num_total_kernels > 0 else 0
    coverage_text = (
        f"Run {num_custom_kernels} custom kernels / Total {num_total_kernels} kernels, Coverage: {ratio:.2%}"
    )
    metadata["custom_kernel_coverage"] = coverage_text
    metadata["custom_kernel_not_in_profiling"] = custom_kernels_not_in_profiling
    metadata["custom_kernel_in_profiling"] = custom_kernels_in_profiling
    metadata["total_kernel_run_time_in_profiling_us"] = total_kernel_run_time_in_profiling_us
    metadata["total_kernel_cuda_time_in_profiling_us"] = total_kernel_cuda_time_in_profiling_us
    metadata["total_kernel_run_time_in_profiling_us_cpu_cuda"] = total_kernel_run_time_in_profiling_us_cpu_cuda
    metadata["custom_kernel_cuda_time_in_profiling_us"] = custom_kernel_cuda_time_in_profiling_us
    ratio_time = (
        custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us
        if total_kernel_run_time_in_profiling_us > 0
        else 0
    )
    coverage_measurement_valid = total_kernel_run_time_in_profiling_us > 0
    metadata["coverage_measurement_valid"] = coverage_measurement_valid
    metadata["raw_custom_kernel_time_coverage"] = ratio_time if coverage_measurement_valid else None
    metadata["custom_kernel_cuda_time_coverage"] = (
        f"Custom kernel CUDA time: {custom_kernel_cuda_time_in_profiling_us:.2f}us / "
        f"Total CUDA time: {total_kernel_run_time_in_profiling_us:.2f}us, "
        f"Coverage: {ratio_time:.2%}"
    )
    if coverage_backend == "triton":
        metadata["triton_kernel_coverage"] = coverage_text
        metadata["triton_kernel_not_in_profiling"] = custom_kernels_not_in_profiling
        metadata["triton_kernel_in_profiling"] = custom_kernels_in_profiling

    if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
        kernel_exec_result.metadata["coverage_backend"] = coverage_backend
        kernel_exec_result.metadata["num_custom_kernels"] = num_custom_kernels
        kernel_exec_result.metadata["num_total_kernels"] = num_total_kernels
        kernel_exec_result.metadata["custom_kernel_coverage"] = coverage_text
        kernel_exec_result.metadata["custom_kernel_not_in_profiling"] = custom_kernels_not_in_profiling
        kernel_exec_result.metadata["custom_kernel_in_profiling"] = custom_kernels_in_profiling
        kernel_exec_result.metadata["custom_kernel_cuda_time_in_profiling_us"] = (
            custom_kernel_cuda_time_in_profiling_us
        )
        kernel_exec_result.metadata["total_kernel_run_time_in_profiling_us"] = total_kernel_run_time_in_profiling_us
        kernel_exec_result.metadata["total_kernel_cuda_time_in_profiling_us"] = total_kernel_cuda_time_in_profiling_us
        kernel_exec_result.metadata["total_kernel_run_time_in_profiling_us_cpu_cuda"] = (
            total_kernel_run_time_in_profiling_us_cpu_cuda
        )
        kernel_exec_result.metadata["custom_kernel_cuda_time_coverage"] = metadata["custom_kernel_cuda_time_coverage"]
        kernel_exec_result.metadata["coverage_measurement_valid"] = coverage_measurement_valid
        kernel_exec_result.metadata["raw_custom_kernel_time_coverage"] = (
            ratio_time if coverage_measurement_valid else None
        )
        if coverage_backend == "triton":
            kernel_exec_result.metadata["triton_kernel_coverage"] = coverage_text

    if not detect_decoy_kernel:
        return

    if not coverage_measurement_valid:
        logger.warning("Profiler captured 0 total kernels - likely profiler bug, NOT marking as decoy")
        metadata["coverage_unavailable"] = True
        if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
            kernel_exec_result.metadata["coverage_unavailable"] = True
        return

    if ratio_time < _SUSPECTED_DECOY_COVERAGE_THRESHOLD:
        reasons = metadata.setdefault("suspected_decoy_reasons", [])
        reason = "LOW_CUSTOM_KERNEL_TIME_COVERAGE"
        if reason not in reasons:
            reasons.append(reason)
        metadata["suspected_decoy"] = True
        metadata["suspected_decoy_reason"] = reason
        metadata["suspected_decoy_threshold"] = _SUSPECTED_DECOY_COVERAGE_THRESHOLD
        metadata["suspected_decoy_enforced"] = False
        metadata["suspected_decoy_effect"] = "DIAGNOSTIC_ONLY"
        logger.warning(
            "Custom-kernel CUDA time coverage %.6f is below %.2f for backend=%s; marking as suspected decoy",
            ratio_time,
            _SUSPECTED_DECOY_COVERAGE_THRESHOLD,
            coverage_backend,
        )

    if ratio_time < _HARD_DECOY_COVERAGE_THRESHOLD:
        # Do not hard-reject from named-kernel coverage alone. The unmatched
        # CUDA time may be legal cuBLAS/cuDNN work called directly by the
        # candidate extension. A hard coverage verdict requires runtime
        # provenance that can exclude those allowed library calls.
        metadata["hard_decoy_coverage_candidate"] = True
        metadata["hard_decoy_coverage_threshold"] = _HARD_DECOY_COVERAGE_THRESHOLD
        metadata["hard_decoy_coverage_gate_applied"] = False
        metadata["hard_decoy_coverage_gate_skip_reason"] = "ALLOWED_LIBRARY_PROVENANCE_UNAVAILABLE"

    if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
        for key in (
            "suspected_decoy",
            "suspected_decoy_reason",
            "suspected_decoy_reasons",
            "suspected_decoy_threshold",
            "suspected_decoy_enforced",
            "suspected_decoy_effect",
            "hard_decoy_coverage_candidate",
            "hard_decoy_coverage_threshold",
            "hard_decoy_coverage_gate_applied",
            "hard_decoy_coverage_gate_skip_reason",
        ):
            if key in metadata:
                kernel_exec_result.metadata[key] = metadata[key]


def _run_correctness_step(
    original_model,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    num_correct_trials: int,
    verbose: bool,
    seed_num: int,
    device: Union[torch.device, int],
    overall_start: float | None = None,
    detect_decoy_kernel: bool = False,
) -> KernelExecResult:
    if verbose:
        logger.info("[Eval] Checking Correctness")
    stage_update_fn = None
    if overall_start is not None:

        def stage_update_fn(stage: str) -> None:
            _begin_stage(
                metadata,
                prefix="kg_kernel",
                stage=stage,
                overall_start=overall_start,
            )

    try:
        return run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            stage_update_fn=stage_update_fn,
            detect_aten_fallback=detect_decoy_kernel,
        )
    except Exception as e:
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def _is_candidate_correctness_runtime_failure(metadata: Dict[str, Any]) -> bool:
    return bool(metadata.get("runtime_error")) and (
        metadata.get("correctness_runtime_error_stage") == "custom_forward"
    )


def _select_compute_sanitizer_execution_mode(
    runtime_error: Exception | str, requested_mode: Optional[str]
) -> tuple[str, Optional[str]]:
    selection_mode = str(requested_mode or "error_based").strip().lower()
    if selection_mode not in {"error_based", SANITIZER_MODE_FULL}:
        raise ValueError(
            "Unsupported Compute Sanitizer selection mode " f"{requested_mode!r}; expected 'error_based' or 'full'"
        )
    preferred_tool = classify_compute_sanitizer_error(runtime_error)
    execution_mode = (
        SANITIZER_MODE_FULL if selection_mode == SANITIZER_MODE_FULL else preferred_tool or SANITIZER_MODE_FULL
    )
    return execution_mode, preferred_tool


def _run_triton_detection_step(
    *,
    enable_triton_detection: bool,
    is_triton: bool,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    seed_num: int,
    device: Union[torch.device, int],
    verbose: bool,
    backend: str,
    detect_decoy_kernel: bool,
):
    if not enable_triton_detection:
        return False
    try:
        logger.info("Begin Triton usage detection")
        if kernel_exec_result and kernel_exec_result.correctness:
            torch.cuda.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()
            inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
            model_new = custom_model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            used, matches = detect.detect_triton_usage_for_module(
                model_new,
                *inputs,
                warmup=1,
                steps=1,
                use_cuda=True,
                return_matches=True,
            )
            metadata["triton_profiler_used"] = used
            metadata["triton_profiler_matches"] = matches
            logger.debug("Triton usage detection result: %s", used)
            logger.debug("Triton usage detection matches: %s", matches)
            if not used and is_triton and detect_decoy_kernel:
                logger.warning("[Eval] Backend is 'triton' but no Triton usage detected, marking as decoy")
                kernel_exec_result.decoy_kernel = True
                kernel_exec_result.runtime = -1.0
                return True
                if not used:
                    logger.info(
                        "[Eval] No Triton usage detected, but backend is '%s', continuing to performance measurement",
                        backend,
                    )
    except Exception as e:
        if verbose:
            logger.warning("[Eval] Error in Triton usage detection: %s", e)
        metadata["error_in_triton_detection"] = e
    return False


def _run_performance_step(
    *,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    num_perf_trials: int,
    num_warmup: int = 3,
    perf_trim_count: int = 0,
    verbose: bool,
    seed_num: int,
    device: Union[torch.device, int],
    enable_profiling: bool,
    enable_triton_detection: bool,
    detect_decoy_kernel: bool,
    backend: str,
    backend_profiling_hints: Optional[Dict[str, Any]],
    adaptive_perf_trials: bool = False,
    perf_min_trials: int = 20,
    perf_cv_threshold: float = 0.05,
):
    def _profiling_empty(metrics: Dict[str, Any]) -> bool:
        if not metrics:
            return True
        if "kernels" not in metrics:
            return True
        if len(metrics.get("kernels", [])) == 0:
            return True
        return False

    try:
        if kernel_exec_result and kernel_exec_result.correctness:
            if verbose:
                logger.info("[Eval] Measuring Performance as Sample is Correct")

            torch.cuda.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()
            inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
            model_new = custom_model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            elapsed_times, profiling_metrics, timing_info = time_execution_with_cuda_event(
                model_new,
                *inputs,
                num_warmup=num_warmup,
                num_trials=num_perf_trials,
                verbose=verbose,
                device=device,
                enable_profiling=enable_profiling,
                adaptive=adaptive_perf_trials,
                min_trials=perf_min_trials,
                cv_threshold=perf_cv_threshold,
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device, trim_count=perf_trim_count)
            metadata["kg_kernel_perf_warmup_s"] = timing_info["warmup_wall_s"]
            metadata["kg_kernel_perf_measure_wall_s"] = timing_info["measure_wall_s"]
            metadata["kg_kernel_perf_measure_cuda_event_s"] = timing_info["timed_trials_cuda_event_s"]
            metadata["kg_kernel_perf_profile_s"] = timing_info["profiling_wall_s"]
            metadata["kg_kernel_perf_total_s"] = timing_info["total_wall_s"]
            metadata["kg_kernel_perf_num_trials"] = timing_info["num_trials"]
            metadata["kg_kernel_perf_num_trials_max"] = timing_info["num_trials_requested"]
            metadata["kg_kernel_perf_adaptive"] = timing_info["adaptive_perf_trials"]
            metadata["kg_kernel_perf_adaptive_stopped_early"] = timing_info["adaptive_stopped_early"]
            metadata["kg_kernel_perf_adaptive_final_cv"] = timing_info["adaptive_final_cv"]
            metadata["kg_kernel_perf_num_warmup"] = timing_info["num_warmup"]
            metadata["kg_kernel_perf_mean_ms"] = runtime_stats["mean"]
            metadata["kg_kernel_perf_std_ms"] = runtime_stats["std"]
            metadata["kg_kernel_perf_min_ms"] = runtime_stats["min"]
            metadata["kg_kernel_perf_max_ms"] = runtime_stats["max"]
            metadata["kg_kernel_perf_num_profile_trials"] = timing_info["num_profiling_trials"]

            profiling_empty_initial = enable_profiling and _profiling_empty(profiling_metrics)
            profiling_retries_used = 0
            if profiling_empty_initial:
                retry_count = max(0, int(getattr(settings, "profiling_retry_count", 0)))
                # If a CUPTI TSC shim was expected but did not engage in this
                # process, single-forward profiling cannot be trusted: retry
                # with the legacy multi-forward workaround instead.
                retry_trials = resolve_num_profiling_trials(
                    num_perf_trials,
                    kineto_tsc_fixed=timing_kineto_tsc_fix_verified(),
                )
                for attempt in range(retry_count):
                    logger.warning(
                        "Profiler returned empty results. Retrying (%s/%s)...",
                        attempt + 1,
                        retry_count,
                    )
                    profiling_retries_used = attempt + 1
                    retry_metrics = run_profiling_only(
                        model_new,
                        *inputs,
                        num_trials=retry_trials,
                        verbose=verbose,
                        device=device,
                    )
                    if not _profiling_empty(retry_metrics):
                        profiling_metrics = retry_metrics
                        break
                    profiling_metrics = retry_metrics

            if enable_profiling:
                # Per-task empty-capture bookkeeping so the fleet-wide empty rate can
                # be computed from result metadata (see docs/design-doc/PROFILER_EMPTY_CAPTURE.md).
                profiling_empty_final = _profiling_empty(profiling_metrics)
                metadata["kg_kernel_profiling_empty_initial"] = profiling_empty_initial
                metadata["kg_kernel_profiling_retries_used"] = profiling_retries_used
                metadata["kg_kernel_profiling_empty_final"] = profiling_empty_final
                if profiling_empty_final:
                    logger.warning(
                        "[Profiling] empty-capture: no CUDA kernels after %s retries "
                        "(initial_empty=%s, num_profile_trials=%s)",
                        profiling_retries_used,
                        profiling_empty_initial,
                        timing_info["num_profiling_trials"],
                    )

            if enable_profiling:
                logger.debug(
                    "profiling_metrics type: %s, empty: %s",
                    type(profiling_metrics),
                    not profiling_metrics,
                )
                if profiling_metrics.get("profiling_warning"):
                    logger.warning("Profiling warning: %s", profiling_metrics["profiling_warning"])

                if _profiling_empty(profiling_metrics):
                    logger.warning("Profiler returned empty results!")
                    logger.warning("This may be a profiler bug, not a decoy kernel issue.")
                    logger.warning(
                        "Triton hook detected: %s",
                        metadata.get("triton_profiler_used", False),
                    )
                    logger.warning(
                        "Triton matches: %s",
                        len(metadata.get("triton_profiler_matches", [])),
                    )
                    if metadata.get("triton_profiler_used", False):
                        logger.info("Skipping decoy detection due to profiler failure (Triton hook passed)")

            if profiling_metrics and len(profiling_metrics) > 0:
                metadata["profiling"] = profiling_metrics
                if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
                    kernel_exec_result.metadata["profiling"] = profiling_metrics

                logger.debug("profiling_metrics keys: %s", profiling_metrics.keys())
                logger.debug("kernel_count: %s", profiling_metrics.get("kernel_count", "N/A"))
                if enable_triton_detection:
                    triton_profiler_matches = metadata.get("triton_profiler_matches", [])
                    logger.debug("triton_profiler_matches: %s", triton_profiler_matches)
                    try:
                        coverage_result_dict = compute_triton_kernel_coverage(
                            triton_profiler_matches,
                            profiling_metrics,
                        )
                    except Exception as coverage_error:
                        logger.exception("compute_triton_kernel_coverage failed: %s", coverage_error)
                        coverage_result_dict = {
                            "num_custom_kernels": 0,
                            "num_total_kernels": 0,
                            "custom_kernels_not_in_profiling": triton_profiler_matches,
                            "custom_kernels_in_profiling": [],
                            "total_kernel_run_time_in_profiling_us": 0,
                            "total_kernel_cuda_time_in_profiling_us": 0,
                            "total_kernel_run_time_in_profiling_us_cpu_cuda": 0,
                            "custom_kernel_cuda_time_in_profiling_us": 0,
                        }
                    _apply_coverage_metadata(
                        metadata=metadata,
                        kernel_exec_result=kernel_exec_result,
                        coverage_result_dict=coverage_result_dict,
                        coverage_backend="triton",
                        detect_decoy_kernel=detect_decoy_kernel,
                    )
                elif backend in {"cuda_agent", "tvm_ffi"}:
                    custom_kernel_names = []
                    if backend_profiling_hints:
                        custom_kernel_names = list(backend_profiling_hints.get("custom_kernel_names", []))
                    metadata["custom_kernel_names"] = custom_kernel_names
                    logger.debug("%s custom_kernel_names: %s", backend, custom_kernel_names)
                    if custom_kernel_names:
                        coverage_result_dict = compute_named_kernel_coverage(
                            custom_kernel_names,
                            profiling_metrics,
                        )
                    else:
                        coverage_result_dict = {
                            "num_custom_kernels": 0,
                            "num_total_kernels": profiling_metrics.get("kernel_count", 0),
                            "custom_kernels_not_in_profiling": [],
                            "custom_kernels_in_profiling": [],
                            "total_kernel_run_time_in_profiling_us": profiling_metrics.get("total_cuda_time_us", 0.0),
                            "total_kernel_cuda_time_in_profiling_us": profiling_metrics.get("total_cuda_time_us", 0.0),
                            "total_kernel_run_time_in_profiling_us_cpu_cuda": profiling_metrics.get(
                                "total_cpu_time_us", 0.0
                            )
                            + profiling_metrics.get("total_cuda_time_us", 0.0),
                            "custom_kernel_cuda_time_in_profiling_us": 0.0,
                        }
                    _apply_coverage_metadata(
                        metadata=metadata,
                        kernel_exec_result=kernel_exec_result,
                        coverage_result_dict=coverage_result_dict,
                        coverage_backend=backend,
                        detect_decoy_kernel=detect_decoy_kernel and bool(custom_kernel_names),
                    )
            if verbose:
                logger.info("[Eval] Performance Stats: %s", runtime_stats)
            kernel_exec_result.runtime = runtime_stats["mean"]
            kernel_exec_result.runtime_stats = runtime_stats
    except Exception as e:
        if verbose:
            logger.warning("[Eval] Error in Measuring Performance: %s", e)
        kernel_exec_result.metadata["error_during_performance"] = e


def _run_memory_step(
    *,
    kernel_exec_result: KernelExecResult,
    model,
    get_inputs,
    source: str,
    metadata: Dict[str, Any],
    allocator_check_metadata_key: str,
    seed_num: int,
    environment_floor: Dict[str, int],
    device: Union[torch.device, int],
    verbose: bool,
) -> None:
    """Run one forward in a measurement scope separate from correctness/timing."""

    if not kernel_exec_result or not kernel_exec_result.correctness:
        return
    allocation_check = detect_direct_cuda_allocations(source)
    metadata[allocator_check_metadata_key] = allocation_check

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device=device)
    try:
        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
        model = model.cuda(device=device)
        torch.cuda.synchronize(device=device)
        kernel_exec_result.memory = measure_cuda_memory_trial(
            model,
            *inputs,
            device=device,
            source=source,
            allocation_check=allocation_check,
            environment_floor_allocated_bytes=environment_floor.get("allocated_bytes"),
            environment_floor_reserved_bytes=environment_floor.get("reserved_bytes"),
        )
        if verbose:
            logger.info(
                "[Eval] Memory peak increment: %s bytes, total task peak: %s bytes (complete=%s)",
                kernel_exec_result.memory.get("forward_incremental_peak_allocated_bytes"),
                kernel_exec_result.memory.get("total_task_peak_allocated_bytes"),
                kernel_exec_result.memory.get("measurement_complete"),
            )
    except Exception as exc:
        if verbose:
            logger.warning("[Eval] Error in Measuring CUDA Memory: %s", exc)
        kernel_exec_result.memory = {
            "schema_version": 2,
            "method": "torch_cuda_peak_allocated_delta",
            "allocator_scope": "pytorch_cuda_caching_allocator",
            "environment_floor_available": bool(environment_floor),
            "environment_floor_allocated_bytes": environment_floor.get("allocated_bytes"),
            "environment_floor_reserved_bytes": environment_floor.get("reserved_bytes"),
            "measurement_valid": False,
            "measurement_complete": False,
            "measurement_is_lower_bound": allocation_check["direct_cuda_allocation_detected"],
            "direct_cuda_allocation_detected": allocation_check["direct_cuda_allocation_detected"],
            "direct_cuda_allocation_apis": allocation_check["direct_cuda_allocation_apis"],
            "direct_cuda_allocation_matches": allocation_check["direct_cuda_allocation_matches"],
            "warnings": [warning for warning in (allocation_check["warning"], str(exc)) if warning],
            "error": str(exc),
        }
        metadata["memory_measurement_error"] = str(exc)

    finally:
        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device=device)


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    num_warmup: int = 3,
    perf_trim_count: int = 0,
    verbose: bool = True,
    measure_performance: bool = True,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (torch.cuda.current_device() if torch.cuda.is_available() else None),
    backend: str = "cuda",
    entry_point: str = "Model",
    enable_profiling: bool = True,
    enable_ncu: bool = True,
    enable_compute_sanitizer: bool = True,
    compute_sanitizer_mode: Optional[str] = None,
    enable_triton_detection: bool = True,
    detect_decoy_kernel: bool = True,
    backend_adapter: Optional[Any] = None,
    precompiled_artifact: Optional[Dict[str, Any]] = None,
    enable_compile_artifact_cache: bool = False,
    compile_only: bool = False,
    return_internal_compile_artifact: bool = False,
    adaptive_perf_trials: bool = False,
    perf_min_trials: int = 20,
    perf_cv_threshold: float = 0.05,
) -> KernelExecResult:
    if not compile_only:
        assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    memory_environment_floor: Dict[str, int] = {}
    if not compile_only:
        torch.cuda.set_device(device)
        memory_environment_floor = capture_cuda_memory_environment_floor(device)
    is_triton = backend == "triton"
    metadata: Dict[str, Any] = {}
    metadata["memory_environment_floor"] = dict(memory_environment_floor)
    metadata["hardware"] = "compile-only" if compile_only else torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    overall_start = perf_counter()

    if is_triton and not compile_only:
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert device.type == "cuda", "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(f"device must be an int or torch.device, got {type(device)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if compile_only:
        metadata["task_stage"] = "compile"
        metadata["required_resource"] = "cpu"
        if backend_adapter is None:
            metadata["compilation_error_name"] = "compile_error"
            metadata["compilation_error"] = "compile_only requires a backend adapter"
            return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
        try:
            compile_start = _begin_stage(
                metadata,
                prefix="kg_kernel",
                stage="kernel.compile_only",
                overall_start=overall_start,
            )
            artifact = backend_adapter.compile(
                custom_model_src,
                device=device,
                backend=backend,
                entry_point=f"{entry_point}New",
                build_dir=build_dir,
                enable_compile_artifact_cache=enable_compile_artifact_cache,
            )
            _finish_stage(
                metadata,
                stage="kernel.compile_only",
                timing_key="kg_kernel_backend_compile_s",
                start_time=compile_start,
            )
            _copy_compile_artifact_metadata(metadata, artifact)
            if return_internal_compile_artifact:
                metadata["_internal_compile_artifact"] = artifact
            if not artifact.get("compiled"):
                error = artifact.get("error", "Unknown compile error")
                metadata["compilation_error_name"] = "compile_error"
                metadata["compilation_error"] = error
                metadata["compilation_error_detail"] = classify_compile_error_detail(str(error), backend=backend)
                return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
            return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
        except Exception as exc:
            metadata["compilation_error_name"] = get_error_name(exc)
            metadata["compilation_error"] = exc
            metadata["compilation_error_detail"] = classify_compile_error_detail(str(exc), backend=backend)
            return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    if verbose:
        logger.info("[Eval] Start Evaluation! on device: %s", device)
        logger.info("[Eval] Loading Original Model")

    load_original_start = _begin_stage(
        metadata,
        prefix="kg_kernel",
        stage="kernel.load_original_src",
        overall_start=overall_start,
    )
    try:
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(original_model_src, context, entry_point)
    except OriginalModelLoadError as exc:
        _finish_stage(
            metadata,
            stage="kernel.load_original_src",
            timing_key="kg_kernel_load_original_src_s",
            start_time=load_original_start,
        )
        metadata["kg_kernel_total_s"] = perf_counter() - overall_start
        return _record_model_load_error(metadata, exc)
    else:
        _finish_stage(
            metadata,
            stage="kernel.load_original_src",
            timing_key="kg_kernel_load_original_src_s",
            start_time=load_original_start,
        )

    init_inputs_start = _begin_stage(
        metadata,
        prefix="kg_kernel",
        stage="kernel.prepare_init_inputs",
        overall_start=overall_start,
    )
    set_seed(seed_num)
    init_inputs = get_init_inputs()
    init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]
    _finish_stage(
        metadata,
        stage="kernel.prepare_init_inputs",
        timing_key="kg_kernel_prepare_init_inputs_s",
        start_time=init_inputs_start,
    )

    if (
        len(init_inputs) > 1
        and hasattr(init_inputs[0], "__len__")
        and not isinstance(init_inputs[0], (str, torch.Tensor))
        and len(init_inputs[0]) == 0
    ):
        init_inputs = init_inputs[1]

    with torch.no_grad():
        original_model_start = _begin_stage(
            metadata,
            prefix="kg_kernel",
            stage="kernel.build_reference_model",
            overall_start=overall_start,
        )
        set_seed(seed_num)

        if type(init_inputs) == list:
            original_model = Model(*init_inputs)
        else:
            original_model = Model(**init_inputs)

        assert hasattr(original_model, "forward")
        if verbose:
            logger.info("[Eval] Original Model Loaded")
    _finish_stage(
        metadata,
        stage="kernel.build_reference_model",
        timing_key="kg_kernel_build_reference_model_s",
        start_time=original_model_start,
    )
    if verbose:
        logger.info("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    tempfile_handle = None
    backend_handle = None
    backend_session = None
    backend_profiling_hints: Optional[Dict[str, Any]] = None
    artifact: Optional[Dict[str, Any]] = precompiled_artifact

    def _cleanup():
        if backend_session is not None:
            backend_session.close()
            return
        if backend_adapter is not None and backend_handle is not None:
            backend_adapter.cleanup(backend_handle)
            return
        graceful_eval_cleanup(context, device, tempfile_handle)

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        compile_start = _begin_stage(
            metadata,
            prefix="kg_kernel",
            stage="kernel.compile_and_load",
            overall_start=overall_start,
        )
        if backend_adapter is not None:
            if precompiled_artifact is not None:
                artifact = dict(precompiled_artifact)
                artifact.setdefault("compiled", True)
                artifact.setdefault("code", custom_model_src)
                artifact.setdefault("entry_point", f"{entry_point}New")
                artifact.setdefault("backend", backend)
                artifact.setdefault("device", str(device))
                metadata["precompiled_artifact_used"] = True
                metadata["kg_kernel_backend_compile_s"] = 0.0
            else:
                backend_compile_start = perf_counter()
                artifact = backend_adapter.compile(
                    custom_model_src,
                    device=device,
                    backend=backend,
                    entry_point=f"{entry_point}New",
                    build_dir=build_dir,
                    enable_compile_artifact_cache=enable_compile_artifact_cache,
                )
                _record_phase_timing(
                    metadata,
                    "kg_kernel_backend_compile_s",
                    backend_compile_start,
                )
            _copy_compile_artifact_metadata(metadata, artifact)
            if not artifact.get("compiled"):
                error = artifact.get("error", "Unknown compile error")
                if "lock" in str(error) or "No such file or directory" in str(error):
                    logger.warning(
                        "[Eval] Lock file error during compilation, please retry. Error: %s",
                        error,
                    )
                    metadata["compilation_error_name"] = "compile_error"
                    metadata["compilation_error"] = error
                    metadata["compilation_error_detail"] = classify_compile_error_detail(
                        str(error),
                        backend=backend,
                    )
                    _finish_stage(
                        metadata,
                        stage="kernel.compile_and_load",
                        timing_key="kg_kernel_compile_and_load_s",
                        start_time=compile_start,
                    )
                    _cleanup()
                    return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
                metadata["compilation_error_name"] = "compile_error"
                metadata["compilation_error"] = error
                metadata["compilation_error_detail"] = classify_compile_error_detail(
                    str(error),
                    backend=backend,
                )
                _finish_stage(
                    metadata,
                    stage="kernel.compile_and_load",
                    timing_key="kg_kernel_compile_and_load_s",
                    start_time=compile_start,
                )
                _cleanup()
                return KernelExecResult(compiled=False, metadata=metadata)

            backend_load_start = perf_counter()
            backend_handle = backend_adapter.load(
                artifact,
                device=device,
                context=context,
                build_dir=build_dir,
            )
            _record_phase_timing(
                metadata,
                "kg_kernel_backend_load_s",
                backend_load_start,
            )
            backend_session_start = perf_counter()
            backend_session = backend_adapter.open_session(backend_handle, device=device)
            _record_phase_timing(
                metadata,
                "kg_kernel_backend_session_open_s",
                backend_session_start,
            )
            if isinstance(backend_handle, dict):
                backend_profiling_hints = backend_handle.get("profiling_hints")
            tempfile_handle = backend_handle.get("tempfile_handle")
        else:
            if is_triton:
                ModelNew, tempfile_handle = load_custom_model_with_tempfile(
                    custom_model_src, entry_point=f"{entry_point}New"
                )
                if verbose:
                    logger.info("[Eval] Model with Triton Loaded")
            else:
                ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)
        _finish_stage(
            metadata,
            stage="kernel.compile_and_load",
            timing_key="kg_kernel_compile_and_load_s",
            start_time=compile_start,
        )
    except Exception as e:
        logger.warning(
            "Failed to compile custom CUDA kernel; recording compilation failure. Error: %s",
            e,
        )
        _finish_stage(
            metadata,
            stage="kernel.compile_and_load",
            timing_key="kg_kernel_compile_and_load_s",
            start_time=compile_start,
        )

        if "lock" in str(e) or "No such file or directory" in str(e):
            logger.warning("[Eval] Lock file error during compilation, please retry. Error: %s", e)
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = e
            metadata["compilation_error_detail"] = classify_compile_error_detail(
                str(e),
                backend=backend,
            )
            _cleanup()
            return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = e
        metadata["compilation_error_detail"] = classify_compile_error_detail(
            str(e),
            backend=backend,
        )
        _cleanup()
        return KernelExecResult(compiled=False, metadata=metadata)

    runtime_sanitizer = skipped_compute_sanitizer_result("not_triggered")

    try:

        def _create_custom_model():
            if backend_session is not None:
                return backend_session.create_model(
                    init_inputs,
                    no_grad=True,
                    synchronize=False,
                )
            if type(init_inputs) == list:
                return ModelNew(*init_inputs)
            return ModelNew(**init_inputs)

        with torch.no_grad():
            custom_model_start = _begin_stage(
                metadata,
                prefix="kg_kernel",
                stage="kernel.build_custom_model",
                overall_start=overall_start,
            )
            set_seed(seed_num)
            custom_model = _create_custom_model()

            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        del _create_custom_model
        init_inputs = None
        gc.collect()
        torch.cuda.synchronize(device=device)
        _finish_stage(
            metadata,
            stage="kernel.build_custom_model",
            timing_key="kg_kernel_build_custom_model_s",
            start_time=custom_model_start,
        )
        if verbose:
            logger.info("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        logger.warning(
            "Failed to load custom CUDA kernel; compiled but not able to run, counting as runtime error. Error: %s",
            e,
        )
        _cleanup()
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        if "custom_model_start" in locals():
            _finish_stage(
                metadata,
                stage="kernel.build_custom_model",
                timing_key="kg_kernel_build_custom_model_s",
                start_time=custom_model_start,
            )
        return KernelExecResult(
            compiled=True,
            correctness=False,
            metadata=metadata,
            runtime_sanitizer=runtime_sanitizer,
        )

    kernel_exec_result = None

    correctness_start = _begin_stage(
        metadata,
        prefix="kg_kernel",
        stage="kernel.correctness",
        overall_start=overall_start,
    )
    kernel_exec_result = _run_correctness_step(
        original_model,
        custom_model,
        get_inputs,
        metadata,
        num_correct_trials,
        verbose,
        seed_num,
        device,
        overall_start,
        detect_decoy_kernel,
    )
    _finish_stage(
        metadata,
        stage="kernel.correctness",
        timing_key="kg_kernel_correctness_s",
        start_time=correctness_start,
    )

    selection_mode = str(compute_sanitizer_mode or "error_based").strip().lower()
    correctness_runtime_failure = _is_candidate_correctness_runtime_failure(metadata)
    if not enable_compute_sanitizer:
        runtime_sanitizer = skipped_compute_sanitizer_result("disabled")
    elif correctness_runtime_failure:
        sanitizer_start = _begin_stage(
            metadata,
            prefix="kg_kernel",
            stage="kernel.runtime_sanitizer",
            overall_start=overall_start,
        )
        runtime_error = metadata.get("runtime_error", "")
        execution_mode, preferred_tool = _select_compute_sanitizer_execution_mode(runtime_error, selection_mode)
        sanitizer_tools = list(FULL_SANITIZER_TOOLS) if execution_mode == SANITIZER_MODE_FULL else [execution_mode]
        run_all_checks = execution_mode == SANITIZER_MODE_FULL
        metadata["runtime_sanitizer_trigger"] = "correctness_runtime_error"
        metadata["runtime_sanitizer_mode"] = selection_mode
        metadata["runtime_sanitizer_execution_mode"] = execution_mode
        metadata["runtime_sanitizer_tool_order"] = sanitizer_tools
        metadata["runtime_sanitizer_error_classification"] = preferred_tool or "ambiguous"
        metadata["runtime_sanitizer_run_all_checks"] = run_all_checks
        sanitizer_kernel_names = select_kernel_names(
            metadata,
            settings.compute_sanitizer_max_kernels,
        )
        sanitizer_kwargs = dict(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            artifact=artifact,
            backend=backend,
            entry_point=entry_point,
            device=device,
            kernel_names=sanitizer_kernel_names,
            sanitizer_path=settings.compute_sanitizer_path,
            timeout_s=settings.compute_sanitizer_timeout_s,
            max_kernels=settings.compute_sanitizer_max_kernels,
            max_issues=settings.compute_sanitizer_max_issues,
            input_seed=metadata.get("correctness_failed_trial_seed"),
            model_seed=seed_num,
            generate_inputs_on_gpu=bool(metadata.get("correctness_inputs_generated_on_gpu", True)),
        )
        runtime_sanitizer = run_compute_sanitizer(
            **sanitizer_kwargs,
            mode=execution_mode,
            primary_tool=preferred_tool,
        )
        final_error_classification = preferred_tool or "ambiguous"
        runtime_sanitizer["selection_mode"] = selection_mode
        runtime_sanitizer["error_classification"] = final_error_classification
        metadata["runtime_sanitizer_error_classification"] = final_error_classification
        metadata["runtime_sanitizer_run_all_checks"] = run_all_checks
        _finish_stage(
            metadata,
            stage="kernel.runtime_sanitizer",
            timing_key="kg_kernel_runtime_sanitizer_s",
            start_time=sanitizer_start,
        )
    else:
        skip_reason = (
            "correctness_passed" if kernel_exec_result.correctness else "correctness_failed_without_runtime_error"
        )
        runtime_sanitizer = skipped_compute_sanitizer_result(skip_reason)

    runtime_sanitizer.setdefault("selection_mode", selection_mode)
    runtime_sanitizer.setdefault("mode", None)
    metadata["runtime_sanitizer_status"] = runtime_sanitizer.get("status")
    metadata["runtime_sanitizer_issue_count"] = runtime_sanitizer.get("detected_issue_count", 0)
    kernel_exec_result.runtime_sanitizer = runtime_sanitizer

    if correctness_runtime_failure:
        # A CUDA launch failure may poison this worker's context. Sanitizer ran
        # in a fresh process, so do not synchronize or start another trial here.
        metadata["kg_kernel_total_s"] = perf_counter() - overall_start
        _sync_exec_result_metadata(kernel_exec_result, metadata)
        try:
            _cleanup()
        except Exception as cleanup_exc:
            metadata["cleanup_after_runtime_error"] = str(cleanup_exc)
        return kernel_exec_result

    del original_model
    gc.collect()
    torch.cuda.synchronize(device=device)

    if kernel_exec_result.correctness and kernel_exec_result.decoy_kernel:
        logger.warning(
            "[Eval] Correct candidate used forbidden ATen compute; skipping performance (reason=%s)",
            metadata.get("decoy_reason"),
        )
        metadata["kg_kernel_total_s"] = perf_counter() - overall_start
        _sync_exec_result_metadata(kernel_exec_result, metadata)
        _cleanup()
        return kernel_exec_result

    triton_detect_start = _begin_stage(
        metadata,
        prefix="kg_kernel",
        stage="kernel.triton_detect",
        overall_start=overall_start,
    )
    decoy_detected = _run_triton_detection_step(
        enable_triton_detection=enable_triton_detection,
        is_triton=is_triton,
        kernel_exec_result=kernel_exec_result,
        custom_model=custom_model,
        get_inputs=get_inputs,
        metadata=metadata,
        seed_num=seed_num,
        device=device,
        verbose=verbose,
        backend=backend,
        detect_decoy_kernel=detect_decoy_kernel,
    )
    _finish_stage(
        metadata,
        stage="kernel.triton_detect",
        timing_key="kg_kernel_triton_detect_s",
        start_time=triton_detect_start,
    )
    if decoy_detected:
        metadata["ncu"] = skipped_ncu_result(
            "skipped_decoy" if enable_ncu else "disabled",
            settings.ncu_profile_version,
        )
        metadata["kg_kernel_total_s"] = perf_counter() - overall_start
        _sync_exec_result_metadata(kernel_exec_result, metadata)
        _cleanup()
        return kernel_exec_result

    if measure_performance:
        performance_start = _begin_stage(
            metadata,
            prefix="kg_kernel",
            stage="kernel.performance",
            overall_start=overall_start,
        )
        _run_performance_step(
            kernel_exec_result=kernel_exec_result,
            custom_model=custom_model,
            get_inputs=get_inputs,
            metadata=metadata,
            num_perf_trials=num_perf_trials,
            num_warmup=num_warmup,
            perf_trim_count=perf_trim_count,
            verbose=verbose,
            seed_num=seed_num,
            device=device,
            enable_profiling=enable_profiling,
            enable_triton_detection=enable_triton_detection,
            detect_decoy_kernel=detect_decoy_kernel,
            backend=backend,
            backend_profiling_hints=backend_profiling_hints,
            adaptive_perf_trials=adaptive_perf_trials,
            perf_min_trials=perf_min_trials,
            perf_cv_threshold=perf_cv_threshold,
        )
        _finish_stage(
            metadata,
            stage="kernel.performance",
            timing_key="kg_kernel_performance_step_s",
            start_time=performance_start,
        )

    memory_start = _begin_stage(
        metadata,
        prefix="kg_kernel",
        stage="kernel.memory",
        overall_start=overall_start,
    )
    _run_memory_step(
        kernel_exec_result=kernel_exec_result,
        model=custom_model,
        get_inputs=get_inputs,
        source=custom_model_src,
        metadata=metadata,
        allocator_check_metadata_key="kernel_memory_allocator_check",
        seed_num=seed_num,
        environment_floor=memory_environment_floor,
        device=device,
        verbose=verbose,
    )
    _finish_stage(
        metadata,
        stage="kernel.memory",
        timing_key="kg_kernel_memory_step_s",
        start_time=memory_start,
    )

    ncu_start = _begin_stage(
        metadata,
        prefix="kg_kernel",
        stage="kernel.ncu_profile",
        overall_start=overall_start,
    )
    if not enable_ncu:
        metadata["ncu"] = skipped_ncu_result("disabled", settings.ncu_profile_version)
    elif not kernel_exec_result or not kernel_exec_result.correctness:
        metadata["ncu"] = skipped_ncu_result("skipped_incorrect", settings.ncu_profile_version)
    elif not measure_performance:
        metadata["ncu"] = skipped_ncu_result("skipped_performance_disabled", settings.ncu_profile_version)
    else:
        ncu_kernel_names = select_kernel_names(metadata, settings.ncu_max_kernels)
        metadata["ncu"] = run_ncu_profile(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            artifact=artifact,
            backend=backend,
            entry_point=entry_point,
            device=device,
            kernel_names=ncu_kernel_names,
            ncu_path=settings.ncu_path,
            metrics=settings.ncu_metrics,
            timeout_s=settings.ncu_timeout_s,
            max_kernels=settings.ncu_max_kernels,
            warmup=settings.ncu_warmup,
            profile_version=settings.ncu_profile_version,
        )
    _finish_stage(
        metadata,
        stage="kernel.ncu_profile",
        timing_key="kg_kernel_ncu_profile_s",
        start_time=ncu_start,
    )

    metadata["kg_kernel_total_s"] = perf_counter() - overall_start
    _sync_exec_result_metadata(kernel_exec_result, metadata)
    _cleanup()
    return kernel_exec_result


def eval_reference_only(
    original_model_src: str,
    seed_num: int = 42,
    num_perf_trials: int = 10,
    num_warmup: int = 3,
    perf_trim_count: int = 0,
    verbose: bool = False,
    device: Union[torch.device, int] = (torch.cuda.current_device() if torch.cuda.is_available() else None),
    entry_point: str = "Model",
    reference_backend: Optional[str] = None,
    backend_adapter: Optional[Any] = None,
) -> KernelExecResult:
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    torch.cuda.set_device(device)
    memory_environment_floor = capture_cuda_memory_environment_floor(device)
    metadata: Dict[str, Any] = {}
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    metadata["memory_environment_floor"] = dict(memory_environment_floor)
    overall_start = perf_counter()

    context: Dict[str, Any] = {}

    if verbose:
        logger.info("[Eval] Start Evaluation! on device: %s", device)
        logger.info("[Eval] Loading Original Model")

    try:
        load_original_start = _begin_stage(
            metadata,
            prefix="kg_reference",
            stage="reference.load_original_src",
            overall_start=overall_start,
        )
        try:
            Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                original_model_src,
                context,
                entry_point,
            )
        except OriginalModelLoadError:
            _finish_stage(
                metadata,
                stage="reference.load_original_src",
                timing_key="kg_reference_load_original_src_s",
                start_time=load_original_start,
            )
            raise
        else:
            _finish_stage(
                metadata,
                stage="reference.load_original_src",
                timing_key="kg_reference_load_original_src_s",
                start_time=load_original_start,
            )

        init_inputs_start = _begin_stage(
            metadata,
            prefix="kg_reference",
            stage="reference.prepare_init_inputs",
            overall_start=overall_start,
        )
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]
        _finish_stage(
            metadata,
            stage="reference.prepare_init_inputs",
            timing_key="kg_reference_prepare_init_inputs_s",
            start_time=init_inputs_start,
        )

        with torch.no_grad():
            original_model_start = _begin_stage(
                metadata,
                prefix="kg_reference",
                stage="reference.build_model",
                overall_start=overall_start,
            )
            set_seed(seed_num)
            if type(init_inputs) == list:
                original_model = Model(*init_inputs)
            else:
                original_model = Model(**init_inputs)
            assert hasattr(original_model, "forward")
        _finish_stage(
            metadata,
            stage="reference.build_model",
            timing_key="kg_reference_build_model_s",
            start_time=original_model_start,
        )
        del init_inputs
        gc.collect()
        torch.cuda.synchronize(device=device)
        if verbose:
            logger.info("[Eval] Original Model Loaded")

    except Exception as e:
        logger.warning("Failed to load original model: %s", e)
        return _record_model_load_error(metadata, e)

    kernel_exec_result = KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    inputs = None
    model = original_model

    try:
        if verbose:
            logger.info("[Eval] Measuring Performance of Original Model")

        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
        model = original_model.cuda(device=device)
        metadata["kg_reference_backend_compile_s"] = 0.0
        if reference_backend:
            backend_name = reference_backend.lower()
            metadata["reference_backend"] = backend_name
            logger.info("[Eval] reference_backend=%s", backend_name)
            if backend_name in ("torch_compile", "torch-compile", "compile"):
                try:
                    if not hasattr(torch, "compile"):
                        raise RuntimeError("torch.compile is not available")
                    compile_start = _begin_stage(
                        metadata,
                        prefix="kg_reference",
                        stage="reference.backend_compile",
                        overall_start=overall_start,
                    )
                    model = torch.compile(model)
                    _finish_stage(
                        metadata,
                        stage="reference.backend_compile",
                        timing_key="kg_reference_backend_compile_s",
                        start_time=compile_start,
                    )
                    metadata["reference_backend_compiled"] = True
                    logger.info("[Eval] torch.compile succeeded")
                except Exception as e:
                    if "compile_start" in locals():
                        _finish_stage(
                            metadata,
                            stage="reference.backend_compile",
                            timing_key="kg_reference_backend_compile_s",
                            start_time=compile_start,
                        )
                    metadata["reference_backend_error"] = str(e)
                    logger.warning("[Eval] torch.compile failed: %s", e)
                    return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
        torch.cuda.synchronize(device=device)

        perf_start = _begin_stage(
            metadata,
            prefix="kg_reference",
            stage="reference.performance",
            overall_start=overall_start,
        )
        elapsed_times, _, timing_info = time_execution_with_cuda_event(
            model,
            *inputs,
            num_warmup=num_warmup,
            num_trials=num_perf_trials,
            verbose=verbose,
            device=device,
            enable_profiling=False,
        )
        runtime_stats = get_timing_stats(elapsed_times, device=device, trim_count=perf_trim_count)
        metadata["kg_reference_perf_warmup_s"] = timing_info["warmup_wall_s"]
        metadata["kg_reference_perf_measure_wall_s"] = timing_info["measure_wall_s"]
        metadata["kg_reference_perf_measure_cuda_event_s"] = timing_info["timed_trials_cuda_event_s"]
        metadata["kg_reference_perf_total_s"] = timing_info["total_wall_s"]
        metadata["kg_reference_perf_num_trials"] = timing_info["num_trials"]
        metadata["kg_reference_perf_num_warmup"] = timing_info["num_warmup"]
        metadata["kg_reference_perf_mean_ms"] = runtime_stats["mean"]
        metadata["kg_reference_perf_std_ms"] = runtime_stats["std"]
        metadata["kg_reference_perf_min_ms"] = runtime_stats["min"]
        metadata["kg_reference_perf_max_ms"] = runtime_stats["max"]
        _finish_stage(
            metadata,
            stage="reference.performance",
            timing_key="kg_reference_performance_step_s",
            start_time=perf_start,
        )

        if verbose:
            logger.info("[Eval] Performance Stats: %s", runtime_stats)
        kernel_exec_result.runtime = runtime_stats["mean"]
        kernel_exec_result.runtime_stats = runtime_stats
    except Exception as e:
        if verbose:
            logger.warning("[Eval] Error in Measuring Performance: %s", e)
        kernel_exec_result.metadata["error_during_performance"] = e
    finally:
        inputs = None
        gc.collect()
        torch.cuda.synchronize(device=device)

    memory_start = _begin_stage(
        metadata,
        prefix="kg_reference",
        stage="reference.memory",
        overall_start=overall_start,
    )
    _run_memory_step(
        kernel_exec_result=kernel_exec_result,
        model=model,
        get_inputs=get_inputs,
        source=original_model_src,
        metadata=metadata,
        allocator_check_metadata_key="reference_memory_allocator_check",
        seed_num=seed_num,
        environment_floor=memory_environment_floor,
        device=device,
        verbose=verbose,
    )
    _finish_stage(
        metadata,
        stage="reference.memory",
        timing_key="kg_reference_memory_step_s",
        start_time=memory_start,
    )

    metadata["kg_reference_total_s"] = perf_counter() - overall_start
    _sync_exec_result_metadata(kernel_exec_result, metadata)
    graceful_eval_cleanup(context, device, None)
    return kernel_exec_result
