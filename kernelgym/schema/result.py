"""Shared result models for KernelBench workflows."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from kernelgym.common import ErrorCode
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult
from kernelgym.utils.device_info import with_device_info

from .serialization import coerce_error_code, make_json_safe, serialize_error_code


def _filter_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in valid_fields}


_MEMORY_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")
_INTERNAL_METADATA_FIELDS = ("correctness_failed_trial_seed",)


def _prepare_public_metadata(value: Any) -> Dict[str, Any]:
    metadata = make_json_safe(with_device_info(value))
    for key in _INTERNAL_METADATA_FIELDS:
        metadata.pop(key, None)
    return metadata


def _format_memory_bytes(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value

    unit_index = 0
    divisor = 1.0
    magnitude = abs(float(value))
    while magnitude >= 1024.0 and unit_index < len(_MEMORY_BYTE_UNITS) - 1:
        magnitude /= 1024.0
        divisor *= 1024.0
        unit_index += 1
    return f"{float(value) / divisor:.2f} {_MEMORY_BYTE_UNITS[unit_index]}"


def _public_memory_field_name(name: str) -> str:
    return name.removesuffix("_bytes")


def _serialize_memory_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _public_memory_field_name(key): (
                _format_memory_bytes(item)
                if key.endswith("_bytes")
                else _serialize_memory_fields(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_serialize_memory_fields(item) for item in value]
    return value


_PUBLIC_MEMORY_MEASUREMENT_FIELDS = (
    ("absolute_peak_allocated_bytes", "absolute_peak_allocated_bytes"),
    ("task_peak_allocated_delta_bytes", "total_task_peak_allocated_bytes"),
    ("forward_peak_allocated_delta_bytes", "forward_incremental_peak_allocated_bytes"),
)


def _pop_measurement_status(value: Dict[str, Any]) -> Optional[str]:
    valid = value.pop("measurement_valid", None)
    complete = value.pop("measurement_complete", None)
    value.pop("measurement_is_lower_bound", None)
    if valid is None and complete is None:
        return None
    if valid is not True:
        return "invalid"
    return "complete" if complete is True else "partial"


def _prepare_public_memory_measurement(
    value: Dict[str, Any],
) -> Dict[str, Any]:
    return _serialize_memory_fields(
        {
            public_key: value[internal_key]
            for public_key, internal_key in _PUBLIC_MEMORY_MEASUREMENT_FIELDS
            if internal_key in value
        }
    )


def _prepare_public_memory_comparison(value: Dict[str, Any]) -> Dict[str, Any]:
    comparison = dict(value)
    measurement_status = _pop_measurement_status(comparison)
    public_comparison = {
        "measurement_status": measurement_status,
        "kernel_minus_reference_bytes": comparison.get(
            "primary_kernel_minus_reference_bytes"
        ),
        "kernel_to_reference_ratio": comparison.get(
            "primary_kernel_to_reference_ratio"
        ),
    }
    warnings = comparison.get("warnings")
    if isinstance(warnings, list) and warnings:
        public_comparison["warnings"] = warnings
    ratio_warning = comparison.get("ratio_warning")
    if isinstance(ratio_warning, str) and ratio_warning:
        public_comparison["warning"] = ratio_warning
    return _serialize_memory_fields(public_comparison)


def _prepare_public_allocator_check(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    detected = value.get("direct_cuda_allocation_detected") is True
    severity = value.get("severity")
    if not detected and severity in (None, "none"):
        return None
    return {
        "severity": severity,
        "measurement_impact": value.get("measurement_impact"),
        "detected_apis": value.get("direct_cuda_allocation_apis", []),
        "matches": value.get("direct_cuda_allocation_matches", []),
    }


def _build_memory_comparison(
    reference_memory: Optional[Dict[str, Any]],
    kernel_memory: Optional[Dict[str, Any]],
    memory_ratio_threshold: Optional[float] = 1.8,
) -> Dict[str, Any]:
    reference_memory = reference_memory or {}
    kernel_memory = kernel_memory or {}
    reference_forward_peak = reference_memory.get(
        "forward_incremental_peak_allocated_bytes"
    )
    kernel_forward_peak = kernel_memory.get("forward_incremental_peak_allocated_bytes")
    reference_total_peak = reference_memory.get("total_task_peak_allocated_bytes")
    kernel_total_peak = kernel_memory.get("total_task_peak_allocated_bytes")
    reference_persistent = reference_memory.get("persistent_allocated_bytes")
    kernel_persistent = kernel_memory.get("persistent_allocated_bytes")

    trial_valid = (
        isinstance(reference_forward_peak, int)
        and isinstance(kernel_forward_peak, int)
        and reference_memory.get("measurement_valid") is True
        and kernel_memory.get("measurement_valid") is True
    )
    valid = (
        trial_valid
        and isinstance(reference_total_peak, int)
        and isinstance(kernel_total_peak, int)
    )
    complete = (
        valid
        and reference_memory.get("measurement_complete") is True
        and kernel_memory.get("measurement_complete") is True
    )
    comparison_metric = "total_task_peak_allocated_bytes"
    primary_reference = reference_total_peak
    primary_kernel = kernel_total_peak
    kernel_to_reference_ratio = (
        primary_kernel / primary_reference
        if valid and primary_reference > 0
        else None
    )

    ratio_warning = None
    if memory_ratio_threshold is not None:
        threshold = float(memory_ratio_threshold)
        if (
            threshold > 1.0
            and kernel_to_reference_ratio is not None
            and kernel_to_reference_ratio >= threshold
        ):
            ratio_warning = (
                "Kernel total-task peak allocated memory is "
                f"{kernel_to_reference_ratio:.3f}x the reference, meeting or exceeding the configured "
                f"{threshold:.2f}x warning threshold. This Kernel may be impractical because "
                "it uses excessive GPU memory."
            )

    warnings = []
    if valid and not complete:
        warnings.append(
            "Memory comparison is a lower bound because at least one implementation may allocate outside "
            "PyTorch's CUDA caching allocator."
        )
    elif not trial_valid:
        warnings.append(
            "Memory comparison is unavailable because one or both memory trials are missing or invalid."
        )
    elif not valid:
        warnings.append(
            "Memory comparison is unavailable because total-task peak memory is missing or invalid."
        )

    return {
        "measurement_valid": valid,
        "measurement_complete": complete,
        "total_task_measurement_valid": valid,
        "comparison_metric": comparison_metric,
        "primary_reference_bytes": primary_reference if valid else None,
        "primary_kernel_bytes": primary_kernel if valid else None,
        "primary_kernel_minus_reference_bytes": (
            primary_kernel - primary_reference if valid else None
        ),
        "primary_memory_savings_bytes": (
            primary_reference - primary_kernel if valid else None
        ),
        "primary_kernel_to_reference_ratio": kernel_to_reference_ratio,
        "ratio_warning": ratio_warning,
        "reference_forward_incremental_peak_allocated_bytes": reference_forward_peak,
        "kernel_forward_incremental_peak_allocated_bytes": kernel_forward_peak,
        "reference_persistent_allocated_bytes": reference_persistent,
        "kernel_persistent_allocated_bytes": kernel_persistent,
        "reference_total_task_peak_allocated_bytes": reference_total_peak,
        "kernel_total_task_peak_allocated_bytes": kernel_total_peak,
        "total_task_kernel_minus_reference_bytes": (
            kernel_total_peak - reference_total_peak if valid else None
        ),
        "total_task_memory_savings_bytes": (
            reference_total_peak - kernel_total_peak if valid else None
        ),
        "total_task_kernel_to_reference_ratio": (
            kernel_total_peak / reference_total_peak
            if valid and reference_total_peak > 0
            else None
        ),
        "warnings": warnings,
    }


@dataclass
class ReferenceTimingResult:
    task_id: str
    base_task_id: str
    reference_runtime: float
    metadata: Dict[str, Any]
    reference_memory: Optional[Dict[str, Any]] = None
    status: str = "completed"
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode | str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["metadata"] = _prepare_public_metadata(result.get("metadata"))
        result["error_code"] = serialize_error_code(result.get("error_code"))
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTimingResult":
        filtered_data = _filter_fields(cls, data)
        if "error_code" in filtered_data:
            filtered_data["error_code"] = coerce_error_code(filtered_data["error_code"])
        return cls(**filtered_data)


@dataclass
class KernelEvaluationResult:
    task_id: str
    base_task_id: str
    compiled: bool
    correctness: Optional[bool]
    decoy_kernel: bool
    kernel_runtime: float
    metadata: Dict[str, Any]
    kernel_memory: Optional[Dict[str, Any]] = None
    runtime_sanitizer: Optional[Dict[str, Any]] = None
    status: str = "completed"
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode | str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["metadata"] = _prepare_public_metadata(result.get("metadata"))
        result["error_code"] = serialize_error_code(result.get("error_code"))
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelEvaluationResult":
        filtered_data = _filter_fields(cls, data)
        if "error_code" in filtered_data:
            filtered_data["error_code"] = coerce_error_code(filtered_data["error_code"])
        return cls(**filtered_data)

    @classmethod
    def from_kernel_exec_result(
        cls,
        task_id: str,
        base_task_id: str,
        result: KernelExecResult,
        verbose_errors: bool = True,
    ) -> "KernelEvaluationResult":
        metadata: Dict[str, Any] = dict(result.metadata or {})

        for key in (
            "compilation_error",
            "runtime_error",
            "error",
            "correctness_issue",
            "triton_kernel_coverage",
            "num_custom_kernels",
            "num_total_kernels",
            "triton_profiler_matches",
            "custom_kernel_cuda_time_in_profiling_us",
            "total_kernel_cuda_time_in_profiling_us",
            "total_kernel_run_time_in_profiling_us",
            "total_kernel_run_time_in_profiling_us_cpu_cuda",
            "custom_kernel_cuda_time_coverage",
        ):
            if (
                key in metadata
                and metadata[key] is not None
                and not isinstance(metadata[key], (str, int, float, bool))
            ):
                if isinstance(metadata[key], BaseException):
                    if verbose_errors:
                        if metadata[key].__traceback__:
                            metadata[key] = "".join(
                                traceback.format_exception(
                                    type(metadata[key]),
                                    metadata[key],
                                    metadata[key].__traceback__,
                                )
                            )
                        else:
                            metadata[key] = (
                                f"{type(metadata[key]).__name__}: {str(metadata[key])}"
                            )
                    else:
                        metadata[key] = str(metadata[key])
                else:
                    metadata[key] = str(metadata[key])

        error_message: Optional[str] = None
        error_code: Optional[ErrorCode] = None
        runtime_sanitizer = dict(result.runtime_sanitizer or {}) or None
        sanitizer_issues_found = bool(
            runtime_sanitizer and runtime_sanitizer.get("status") == "issues_found"
        )
        sanitizer_detail = None
        if sanitizer_issues_found:
            for check_result in runtime_sanitizer.get("check_results", []):
                issues = (
                    check_result.get("issues")
                    if isinstance(check_result, dict)
                    else None
                )
                if isinstance(issues, list) and issues:
                    sanitizer_detail = issues[0].get("message")
                    break

        if not result.compiled:
            detail = (
                metadata.get("compilation_error")
                or metadata.get("error")
                or metadata.get("validation_error")
            )
            if detail:
                error_message = f"Kernel compilation failed: {detail}"
            else:
                error_message = "Kernel compilation failed"
            error_code = ErrorCode.COMPILATION_ERROR
        elif sanitizer_issues_found:
            error_message = "Runtime Sanitizer detected an unsafe CUDA kernel"
            if sanitizer_detail:
                error_message = f"{error_message}: {sanitizer_detail}"
            error_code = ErrorCode.RUNTIME_ERROR
        elif not result.correctness:
            detail = metadata.get("runtime_error") or metadata.get("error") or metadata.get("correctness_issue")
            if detail:
                if metadata.get("runtime_error") or metadata.get("error"):
                    error_message = f"Kernel execution failed: {detail}"
                    error_code = ErrorCode.RUNTIME_ERROR
                else:
                    error_message = f"Kernel produced incorrect results: {detail}"
                    error_code = ErrorCode.CORRECTNESS_ERROR
            else:
                error_message = "Kernel produced incorrect results"
                error_code = ErrorCode.CORRECTNESS_ERROR

        return cls(
            task_id=task_id,
            base_task_id=base_task_id,
            compiled=result.compiled,
            decoy_kernel=result.decoy_kernel,
            correctness=result.correctness,
            kernel_runtime=result.runtime,
            metadata=metadata,
            kernel_memory=dict(result.memory or {}) or None,
            runtime_sanitizer=runtime_sanitizer,
            status=(
                "failed"
                if sanitizer_issues_found or not result.compiled
                else "completed"
            ),
            error_message=error_message,
            error_code=error_code,
        )


@dataclass
class EvaluationResult:
    task_id: str
    compiled: bool
    correctness: bool
    decoy_kernel: bool
    reference_runtime: float
    kernel_runtime: float
    speedup: float
    metadata: Dict[str, Any]
    reference_memory: Optional[Dict[str, Any]] = None
    kernel_memory: Optional[Dict[str, Any]] = None
    memory_comparison: Optional[Dict[str, Any]] = None
    runtime_sanitizer: Optional[Dict[str, Any]] = None
    status: str = "completed"
    error_message: Optional[str] = None
    error_code: Optional[ErrorCode | str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        metadata = _prepare_public_metadata(result.get("metadata"))
        for metadata_key in (
            "memory_environment_floor",
            "kg_reference_memory_step_s",
            "kg_kernel_memory_step_s",
            "reference_memory_allocator_check",
        ):
            metadata.pop(metadata_key, None)
        memory_measurement_error = metadata.pop("memory_measurement_error", None)
        kernel_allocator_check = metadata.pop("kernel_memory_allocator_check", None)
        result["metadata"] = metadata

        measurement_sections = {}
        for result_key, memory_key in (
            ("reference_memory", "reference"),
            ("kernel_memory", "kernel"),
        ):
            value = result.pop(result_key, None)
            if not isinstance(value, dict):
                continue
            measurement_sections[memory_key] = _prepare_public_memory_measurement(value)

        comparison = result.pop("memory_comparison", None)
        memory = dict(measurement_sections)
        if isinstance(comparison, dict):
            memory["comparison"] = _prepare_public_memory_comparison(comparison)

        allocator_check = _prepare_public_allocator_check(kernel_allocator_check)
        if allocator_check is not None:
            memory["allocator_check"] = allocator_check
        if memory_measurement_error is not None:
            memory["measurement_error"] = memory_measurement_error
        result["memory"] = memory or None

        result["error_code"] = serialize_error_code(result.get("error_code"))
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        filtered_data = _filter_fields(cls, data)
        if "error_code" in filtered_data:
            filtered_data["error_code"] = coerce_error_code(filtered_data["error_code"])
        return cls(**filtered_data)

    @classmethod
    def from_kernel_exec_result(
        cls, task_id: str, result: KernelExecResult, reference_runtime: float
    ) -> "EvaluationResult":
        speedup = 0.0
        if result.correctness and result.runtime > 0 and reference_runtime > 0:
            speedup = reference_runtime / result.runtime

        kernel_result = KernelEvaluationResult.from_kernel_exec_result(
            task_id, task_id, result
        )
        return cls(
            task_id=task_id,
            compiled=result.compiled,
            correctness=result.correctness,
            decoy_kernel=result.decoy_kernel,
            reference_runtime=reference_runtime,
            kernel_runtime=result.runtime,
            speedup=speedup,
            metadata=kernel_result.metadata,
            kernel_memory=dict(result.memory or {}) or None,
            runtime_sanitizer=dict(result.runtime_sanitizer or {}) or None,
            status=kernel_result.status,
            error_message=kernel_result.error_message,
            error_code=kernel_result.error_code,
        )

    @classmethod
    def from_paired_results(
        cls,
        base_task_id: str,
        reference_result: ReferenceTimingResult,
        kernel_result: KernelEvaluationResult,
        memory_ratio_threshold: Optional[float] = 1.8,
    ) -> "EvaluationResult":
        speedup = 0.0
        if (
            kernel_result.correctness
            and kernel_result.kernel_runtime > 0
            and reference_result.reference_runtime > 0
        ):
            speedup = reference_result.reference_runtime / kernel_result.kernel_runtime

        combined_metadata: Dict[str, Any] = {}
        combined_metadata.update(reference_result.metadata or {})
        combined_metadata.update(kernel_result.metadata or {})
        combined_metadata["reference_task_id"] = reference_result.task_id
        combined_metadata["kernel_task_id"] = kernel_result.task_id

        memory_comparison = _build_memory_comparison(
            reference_result.reference_memory,
            kernel_result.kernel_memory,
            memory_ratio_threshold,
        )
        status = "completed"
        error_message = None
        error_code = None

        if reference_result.status != "completed":
            status = "failed"
            error_message = f"Reference timing failed: {reference_result.error_message}"
            error_code = reference_result.error_code
        elif kernel_result.status != "completed":
            status = "failed"
            error_message = f"Kernel evaluation failed: {kernel_result.error_message}"
            error_code = kernel_result.error_code

        return cls(
            task_id=base_task_id,
            compiled=kernel_result.compiled,
            correctness=kernel_result.correctness,
            decoy_kernel=kernel_result.decoy_kernel,
            reference_runtime=reference_result.reference_runtime,
            kernel_runtime=kernel_result.kernel_runtime,
            speedup=speedup,
            metadata=combined_metadata,
            reference_memory=reference_result.reference_memory,
            kernel_memory=kernel_result.kernel_memory,
            memory_comparison=memory_comparison,
            runtime_sanitizer=kernel_result.runtime_sanitizer,
            status=status,
            error_message=error_message,
            error_code=error_code,
        )
