"""KernelBench evaluation pipeline (task-level, toolkit layer)."""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from time import monotonic_ns, perf_counter, time
from typing import Any, Dict, Optional, Union

import torch

from kernelgym.config import settings
from kernelgym.toolkit.kernelbench import triton_detect as detect
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult, get_error_name, set_seed
from kernelgym.toolkit.kernelbench.loading import (
    graceful_eval_cleanup,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
)
from kernelgym.toolkit.kernelbench.correctness import run_and_check_correctness
from kernelgym.toolkit.kernelbench.profiling import (
    compute_named_kernel_coverage,
    compute_triton_kernel_coverage,
)
from kernelgym.toolkit.kernelbench.timing import (
    get_timing_stats,
    run_profiling_only,
    time_execution_with_cuda_event,
)
from kernelgym.utils.error_classifier import classify_compile_error_detail


logger = logging.getLogger("kernelgym.toolkit.kernelbench.pipeline")
_STAGE_METADATA_PATH_ENV = "KERNELGYM_STAGE_METADATA_PATH"
_FAST_RW_ROOT = Path("/dev/shm")


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
        if coverage_backend == "triton":
            kernel_exec_result.metadata["triton_kernel_coverage"] = coverage_text

    if not detect_decoy_kernel:
        return

    if num_custom_kernels == 0 and num_total_kernels > 0:
        logger.warning(
            "Profiler captured %s kernels but 0 custom kernels for backend=%s - marking as decoy",
            num_total_kernels,
            coverage_backend,
        )
        kernel_exec_result.decoy_kernel = True
    elif num_custom_kernels == 0 and num_total_kernels == 0:
        logger.warning("Profiler captured 0 total kernels - likely profiler bug, NOT marking as decoy")


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
        )
    except Exception as e:
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


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
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device, trim_count=perf_trim_count)
            metadata["kg_kernel_perf_warmup_s"] = timing_info["warmup_wall_s"]
            metadata["kg_kernel_perf_measure_wall_s"] = timing_info["measure_wall_s"]
            metadata["kg_kernel_perf_measure_cuda_event_s"] = timing_info["timed_trials_cuda_event_s"]
            metadata["kg_kernel_perf_profile_s"] = timing_info["profiling_wall_s"]
            metadata["kg_kernel_perf_total_s"] = timing_info["total_wall_s"]
            metadata["kg_kernel_perf_num_trials"] = timing_info["num_trials"]
            metadata["kg_kernel_perf_num_warmup"] = timing_info["num_warmup"]
            metadata["kg_kernel_perf_mean_ms"] = runtime_stats["mean"]
            metadata["kg_kernel_perf_std_ms"] = runtime_stats["std"]
            metadata["kg_kernel_perf_min_ms"] = runtime_stats["min"]
            metadata["kg_kernel_perf_max_ms"] = runtime_stats["max"]
            metadata["kg_kernel_perf_num_profile_trials"] = timing_info["num_profiling_trials"]

            if enable_profiling and _profiling_empty(profiling_metrics):
                retry_count = max(0, int(getattr(settings, "profiling_retry_count", 0)))
                for attempt in range(retry_count):
                    logger.warning(
                        "Profiler returned empty results. Retrying (%s/%s)...",
                        attempt + 1,
                        retry_count,
                    )
                    retry_metrics = run_profiling_only(
                        model_new,
                        *inputs,
                        num_trials=max(1, min(num_perf_trials, 10)),
                        verbose=verbose,
                        device=device,
                    )
                    if not _profiling_empty(retry_metrics):
                        profiling_metrics = retry_metrics
                        break
                    profiling_metrics = retry_metrics

            if enable_profiling:
                logger.debug("profiling_metrics type: %s, empty: %s", type(profiling_metrics), not profiling_metrics)
                if profiling_metrics.get("profiling_warning"):
                    logger.warning("Profiling warning: %s", profiling_metrics["profiling_warning"])

                if _profiling_empty(profiling_metrics):
                    logger.warning("Profiler returned empty results!")
                    logger.warning("This may be a profiler bug, not a decoy kernel issue.")
                    logger.warning("Triton hook detected: %s", metadata.get("triton_profiler_used", False))
                    logger.warning("Triton matches: %s", len(metadata.get("triton_profiler_matches", [])))
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
    enable_triton_detection: bool = True,
    detect_decoy_kernel: bool = True,
    backend_adapter: Optional[Any] = None,
    precompiled_artifact: Optional[Dict[str, Any]] = None,
    enable_compile_artifact_cache: bool = False,
    compile_only: bool = False,
    return_internal_compile_artifact: bool = False,
) -> KernelExecResult:
    if not compile_only:
        assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    if not compile_only:
        torch.cuda.set_device(device)
    is_triton = backend == "triton"
    metadata: Dict[str, Any] = {}
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
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(original_model_src, context, entry_point)
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
                    logger.warning("[Eval] Lock file error during compilation, please retry. Error: %s", error)
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
        logger.warning("Failed to compile custom CUDA kernel; recording compilation failure. Error: %s", e)
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
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

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
    )
    _finish_stage(
        metadata,
        stage="kernel.correctness",
        timing_key="kg_kernel_correctness_s",
        start_time=correctness_start,
    )

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
        )
        _finish_stage(
            metadata,
            stage="kernel.performance",
            timing_key="kg_kernel_performance_step_s",
            start_time=performance_start,
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
    metadata: Dict[str, Any] = {}
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
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
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(original_model_src, context, entry_point)
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
        if verbose:
            logger.info("[Eval] Original Model Loaded")

    except Exception as e:
        logger.warning("Failed to load original model: %s", e)
        metadata["model_load_error"] = e
        metadata["model_load_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    kernel_exec_result = KernelExecResult(compiled=True, correctness=True, metadata=metadata)

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

    metadata["kg_reference_total_s"] = perf_counter() - overall_start
    _sync_exec_result_metadata(kernel_exec_result, metadata)
    graceful_eval_cleanup(context, device, None)
    return kernel_exec_result
