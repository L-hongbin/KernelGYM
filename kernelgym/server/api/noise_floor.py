"""Speedup noise-floor calibration scheduling and statistics."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Sequence

from .noise_floor_cases import NoiseFloorCase


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile_value) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_interleaved_schedule(
    cases: Sequence[NoiseFloorCase], blocks_per_kernel: int, seed: int
) -> list[tuple[NoiseFloorCase, int]]:
    """Shuffle case order independently in every block round."""
    rng = random.Random(seed)
    schedule: list[tuple[NoiseFloorCase, int]] = []
    for block_index in range(1, blocks_per_kernel + 1):
        round_cases = list(cases)
        rng.shuffle(round_cases)
        schedule.extend((case, block_index) for case in round_cases)
    return schedule


def build_calibration_payload(
    case: NoiseFloorCase,
    *,
    task_id: str,
    warmup: int,
    trials: int,
    reference_trials: int,
    trim_count: int,
    timeout: int,
    target_node_id: str | None,
    target_hostname: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "reference_code": case.reference_code,
        "kernel_code": case.kernel_code,
        "toolkit": "kernelbench",
        "backend_adapter": "kernelbench",
        "backend": "tvm_ffi",
        "precision": "fp32",
        "num_correct_trials": 1,
        "num_perf_trials": trials,
        "refer_num_perf_trials": reference_trials,
        "num_warmup": warmup,
        "perf_trim_count": trim_count,
        "adaptive_perf_trials": False,
        "timeout": timeout,
        "priority": "low",
        "entry_point": "Model",
        # Re-run correctness and timings for every block while allowing the
        # identical fixed CUDA sources to reuse their compiled shared object.
        "force_refresh": True,
        "use_reference_cache": False,
        "is_valid": False,
        "enable_compile_artifact_cache": True,
        "enable_profiling": False,
        "enable_ncu": False,
        "enable_compute_sanitizer": False,
        "enable_correctness_input_perturbations": False,
        "enable_triton_detection": False,
        "detect_decoy_kernel": False,
        "run_correctness": True,
        "run_performance": True,
        "target_node_id": target_node_id,
        "target_hostname": target_hostname,
        "workflow": "kernelbench",
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def extract_block_result(
    *,
    case: NoiseFloorCase,
    block_index: int,
    schedule_index: int,
    task_id: str,
    status: str,
    result: dict[str, Any],
    started_at: str,
    completed_at: str,
    end_to_end_s: float,
) -> dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    kernel_mean = (
        _number(metadata.get("kg_kernel_perf_mean_ms"))
        or _number(result.get("kernel_runtime"))
        or _number(result.get("runtime"))
    )
    kernel_std = _number(metadata.get("kg_kernel_perf_std_ms"))
    kernel_trials = _positive_int(metadata.get("kg_kernel_perf_num_trials"))
    reference_mean = _number(metadata.get("kg_reference_perf_mean_ms")) or _number(
        result.get("reference_runtime")
    )
    reference_std = _number(metadata.get("kg_reference_perf_std_ms"))
    reference_trials = _positive_int(metadata.get("kg_reference_perf_num_trials"))
    speedup = _number(result.get("speedup"))
    cached = metadata.get("cached") is True
    reference_stats_available = (
        reference_mean is not None
        and reference_mean > 0
        and reference_std is not None
        and reference_trials is not None
    )
    passed = (
        status == "completed"
        and result.get("compiled") is True
        and result.get("correctness") is True
        and speedup is not None
        and speedup > 0
        and kernel_mean is not None
        and kernel_mean > 0
        and kernel_std is not None
        and kernel_trials is not None
        and (cached or reference_stats_available)
    )

    kernel_cv = kernel_std / kernel_mean if passed and kernel_std is not None and kernel_mean else None
    reference_cv = (
        reference_std / reference_mean
        if reference_std is not None and reference_mean is not None and reference_mean > 0
        else None
    )
    within_variance = None
    if passed and kernel_cv is not None and kernel_trials is not None:
        within_variance = kernel_cv * kernel_cv / kernel_trials
        if not cached and reference_cv is not None and reference_trials is not None:
            within_variance += reference_cv * reference_cv / reference_trials

    return {
        "kernel_id": case.case_id,
        "block_index": block_index,
        "schedule_index": schedule_index,
        "task_id": task_id,
        "status": status,
        "passed": passed,
        "speedup": speedup,
        "log_speedup": math.log(speedup) if passed and speedup is not None else None,
        "kg_kernel_perf_mean_ms": kernel_mean,
        "kg_kernel_perf_std_ms": kernel_std,
        "kg_kernel_perf_num_trials": kernel_trials,
        "kernel_cv": kernel_cv,
        "kg_reference_perf_mean_ms": reference_mean,
        "kg_reference_perf_std_ms": reference_std,
        "kg_reference_perf_num_trials": reference_trials,
        "reference_cv": reference_cv,
        "within_log_speedup_variance": within_variance,
        "cached": cached,
        "device": metadata.get("kernel_execution_device") or metadata.get("device") or "unknown",
        "device_id": metadata.get("kernel_execution_device_id"),
        "device_info": metadata.get("device_info"),
        "process": {
            "kernel_pid": metadata.get("kernel_execution_pid"),
            "kernel_hostname": metadata.get("kernel_execution_hostname"),
            "kernel_execution_epoch_ns": metadata.get("kernel_execution_epoch_ns"),
            "reference_pid": metadata.get("reference_execution_pid"),
            "reference_hostname": metadata.get("reference_execution_hostname"),
            "reference_execution_epoch_ns": metadata.get("reference_execution_epoch_ns"),
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "end_to_end_s": end_to_end_s,
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
    }


def _runtime_bucket(runtime_ms: float) -> str:
    if runtime_ms < 0.1:
        return "lt_0_1_ms"
    if runtime_ms <= 1.0:
        return "0_1_to_1_ms"
    return "gt_1_ms"


def analyze_calibration(
    cases: Sequence[NoiseFloorCase],
    blocks: Iterable[dict[str, Any]],
    *,
    heldout_blocks_per_kernel: int,
    global_percentile: float,
    z_value: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        grouped[str(block["kernel_id"])].append(block)

    per_kernel: list[dict[str, Any]] = []
    calibration_rows_by_kernel: dict[str, list[dict[str, Any]]] = {}
    heldout_rows_by_kernel: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        case_rows = grouped.get(case.case_id, [])
        passed = sorted(
            (row for row in case_rows if row.get("passed") is True),
            key=lambda row: int(row["block_index"]),
        )
        # Keep the held-out rounds fixed even when a request fails. Otherwise a
        # failed early block would silently move a later held-out block into the
        # calibration set and leak validation data into the floor estimate.
        max_block_index = max((int(row["block_index"]) for row in case_rows), default=0)
        heldout_start = max_block_index - heldout_blocks_per_kernel + 1
        calibration_rows = [row for row in passed if int(row["block_index"]) < heldout_start]
        heldout_rows = [row for row in passed if int(row["block_index"]) >= heldout_start]
        calibration_rows_by_kernel[case.case_id] = calibration_rows
        heldout_rows_by_kernel[case.case_id] = heldout_rows

        if len(calibration_rows) < 2:
            per_kernel.append(
                {
                    **case.public_metadata(),
                    "status": "insufficient_blocks",
                    "passed_blocks": len(passed),
                    "calibration_blocks": len(calibration_rows),
                    "heldout_blocks": len(heldout_rows),
                }
            )
            continue

        log_speedups = [float(row["log_speedup"]) for row in calibration_rows]
        within_variances = [float(row["within_log_speedup_variance"]) for row in calibration_rows]
        between_variance = statistics.variance(log_speedups)
        explained_variance = statistics.fmean(within_variances)
        floor_variance = max(between_variance - explained_variance, 0.0)
        noise_floor = math.sqrt(floor_variance)
        kernel_runtimes = [float(row["kg_kernel_perf_mean_ms"]) for row in calibration_rows]
        mean_log_speedup = statistics.fmean(log_speedups)
        per_kernel.append(
            {
                **case.public_metadata(),
                "status": "ok",
                "passed_blocks": len(passed),
                "calibration_blocks": len(calibration_rows),
                "heldout_blocks": len(heldout_rows),
                "mean_log_speedup": mean_log_speedup,
                "geometric_mean_speedup": math.exp(mean_log_speedup),
                "between_log_speedup_variance": between_variance,
                "mean_within_log_speedup_variance": explained_variance,
                "noise_floor": noise_floor,
                "noise_floor_multiplicative_percent": 100.0 * math.expm1(noise_floor),
                "median_kernel_runtime_ms": statistics.median(kernel_runtimes),
                "runtime_bucket": _runtime_bucket(statistics.median(kernel_runtimes)),
            }
        )

    valid_kernels = [item for item in per_kernel if item.get("status") == "ok"]
    floors = [float(item["noise_floor"]) for item in valid_kernels]
    global_floor = percentile(floors, global_percentile) if floors else None

    bucket_floors: dict[str, dict[str, Any]] = {}
    for bucket in ("lt_0_1_ms", "0_1_to_1_ms", "gt_1_ms"):
        values = [float(item["noise_floor"]) for item in valid_kernels if item["runtime_bucket"] == bucket]
        bucket_floors[bucket] = {
            "kernel_count": len(values),
            "p75_noise_floor": percentile(values, 75.0) if values else None,
        }

    coverage_rows: list[dict[str, Any]] = []
    selected = 0
    false_positives = 0
    if global_floor is not None:
        for item in valid_kernels:
            case_id = str(item["case_id"])
            center = float(item["mean_log_speedup"])
            for row in heldout_rows_by_kernel[case_id]:
                variance = float(row["within_log_speedup_variance"]) + global_floor * global_floor
                standardized = (float(row["log_speedup"]) - center) / math.sqrt(variance) if variance > 0 else 0.0
                coverage_rows.append(
                    {
                        "kernel_id": case_id,
                        "block_index": row["block_index"],
                        "standardized_residual": standardized,
                        "above_one_sided_lower_bound": standardized >= -z_value,
                    }
                )

            calibration_rows = calibration_rows_by_kernel[case_id]
            heldout_rows = heldout_rows_by_kernel[case_id]
            if calibration_rows and heldout_rows:
                decision_row = calibration_rows[-1]
                decision_variance = float(decision_row["within_log_speedup_variance"]) + global_floor * global_floor
                lcb_log = float(decision_row["log_speedup"]) - z_value * math.sqrt(decision_variance)
                if lcb_log > 0:
                    selected += 1
                    actual_log_speedup = statistics.fmean(float(row["log_speedup"]) for row in heldout_rows)
                    if actual_log_speedup <= 0:
                        false_positives += 1

    covered = sum(1 for row in coverage_rows if row["above_one_sided_lower_bound"])
    blocks_list = [row for rows in grouped.values() for row in rows]
    device_targets = sorted(
        {
            (
                str(row.get("process", {}).get("kernel_hostname") or "unknown"),
                str(row.get("device_id") if row.get("device_id") is not None else row.get("device")),
            )
            for row in blocks_list
        }
    )
    kernel_processes = {
        (
            str(row.get("process", {}).get("kernel_hostname") or "unknown"),
            int(row["process"]["kernel_pid"]),
        )
        for row in blocks_list
        if isinstance(row.get("process"), dict)
        and isinstance(row["process"].get("kernel_pid"), int)
    }
    return {
        "per_kernel": per_kernel,
        "global": {
            "percentile": global_percentile,
            "noise_floor": global_floor,
            "noise_floor_multiplicative_percent": (
                100.0 * math.expm1(global_floor) if global_floor is not None else None
            ),
            "valid_kernel_count": len(valid_kernels),
            "runtime_buckets": bucket_floors,
        },
        "validation": {
            "heldout_rows": len(coverage_rows),
            "one_sided_z": z_value,
            "lower_bound_coverage": covered / len(coverage_rows) if coverage_rows else None,
            "target_coverage": 0.95,
            "lcb_selected_kernel_count": selected,
            "lcb_false_positive_count": false_positives,
            "lcb_false_positive_rate": false_positives / selected if selected else None,
            "coverage_rows": coverage_rows,
            "lcb_decision_rule": "last calibration block LCB; held-out geometric mean is ground truth",
        },
        "execution_environment": {
            "device_targets": [
                {"hostname": hostname, "device": device} for hostname, device in device_targets
            ],
            "single_target_device": len(device_targets) == 1,
            "observed_block_count": len(blocks_list),
            "observed_kernel_process_count": len(kernel_processes),
            "fresh_kernel_process_per_block": (
                len(kernel_processes) == len(blocks_list) if blocks_list else None
            ),
        },
    }
