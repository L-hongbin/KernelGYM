"""KernelBench toolkit wrapper."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict

import torch

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.schema import (
    EvaluationTask,
    EvaluationResult,
    KernelEvaluationResult,
    KernelEvaluationTask,
    ReferenceTimingResult,
    ReferenceTimingTask,
)
from kernelgym.toolkit.validation import validate_code
from kernelgym.toolkit.kernelbench.binding_detection import resolve_kernel_backend
from kernelgym.toolkit.kernelbench.exec_types import set_seed
from kernelgym.toolkit.kernelbench import pipeline as kernelbench_pipeline

from ..base import Toolkit

logger = logging.getLogger(__name__)


class KernelBenchToolkit(Toolkit):
    """Toolkit adapter around KernelBench evaluation."""

    name = "kernelbench"

    def __init__(self) -> None:
        pass

    @staticmethod
    def _prepare_contained_fault_metadata(
        result: Any,
        *,
        device: torch.device,
        backend: str,
        num_correct_trials: int,
        num_perf_trials: int,
    ) -> bool:
        metadata = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
        candidate_runtime_fault = bool(metadata.get("runtime_error")) and (
            metadata.get("correctness_runtime_error_stage") == "custom_forward"
            or metadata.get("runtime_sanitizer_trigger") == "correctness_runtime_error"
        )
        if not candidate_runtime_fault:
            return False
        hardware = str(metadata.get("hardware") or "unknown")
        metadata.update(
            {
                "device": str(device),
                "gpu_name": hardware,
                "backend": backend,
                "num_correct_trials": num_correct_trials,
                "num_perf_trials": num_perf_trials,
                "candidate_runtime_fault_contained": True,
            }
        )
        # Result serialization normally probes missing device metadata with
        # Torch. A sticky CUDA context must not make another CUDA API call, so
        # provide a complete safe fallback from already-recorded information.
        metadata.setdefault(
            "device_info",
            {
                "gpu_name": hardware,
                "compute_capability": "unknown",
                "cuda_version": str(torch.version.cuda or "unknown"),
                "driver_version": "unknown",
                "nvcc_version": "unknown",
            },
        )
        result.metadata = metadata
        return True

    def _resolve_eval_flags(self, task: Any) -> tuple[bool, bool, bool, bool]:
        run_correctness = task.run_correctness
        if run_correctness is None:
            run_correctness = True

        run_triton_detection = task.run_triton_detection
        if run_triton_detection is None:
            run_triton_detection = task.enable_triton_detection
        if run_triton_detection is None:
            run_triton_detection = task.backend == "triton"

        run_performance = task.run_performance
        if run_performance is None:
            run_performance = task.measure_performance
        if run_performance is None:
            run_performance = True

        detect_decoy_kernel = task.detect_decoy_kernel
        if detect_decoy_kernel is None:
            detect_decoy_kernel = True

        return run_correctness, run_triton_detection, run_performance, detect_decoy_kernel

    def evaluate(self, task: Dict[str, Any], backend=None, **kwargs: Any) -> Dict[str, Any]:
        task_type = task.get("task_type", "evaluation")
        if task_type == "evaluation":
            result = self.evaluate_kernel(EvaluationTask.from_dict(task), backend_adapter=backend)
        elif task_type == "reference_timing":
            result = self.evaluate_reference_timing(
                ReferenceTimingTask.from_dict(task),
                backend_adapter=backend,
            )
        elif task_type == "kernel_evaluation":
            result = self.evaluate_kernel_only(
                KernelEvaluationTask.from_dict(task),
                verbose_errors=task.get("verbose_errors", True),
                enable_profiling=task.get("enable_profiling", settings.enable_profiling),
                backend_adapter=backend,
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return result.to_dict()

    def evaluate_kernel(self, task: EvaluationTask, backend_adapter=None) -> EvaluationResult:
        task = replace(task, backend=resolve_kernel_backend(task.kernel_code, task.backend))
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"validation_error": kernel_error},
                status="failed",
                error_message=f"Kernel code validation failed: {kernel_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            (
                run_correctness,
                enable_triton_detection,
                measure_performance,
                detect_decoy_kernel,
            ) = self._resolve_eval_flags(task)
            num_correct_trials = task.num_correct_trials if run_correctness else 0

            enable_profiling = task.enable_profiling
            if enable_profiling is None:
                enable_profiling = settings.enable_profiling
            enable_compute_sanitizer = task.enable_compute_sanitizer
            if enable_compute_sanitizer is None:
                enable_compute_sanitizer = settings.enable_compute_sanitizer

            num_warmup = getattr(task, "num_warmup", 3)
            perf_trim_count = getattr(task, "perf_trim_count", 0)

            result = kernelbench_pipeline.eval_kernel_against_ref(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
                num_warmup=num_warmup,
                perf_trim_count=perf_trim_count,
                measure_performance=measure_performance,
                verbose=False,
                device=device,
                backend=task.backend,
                precision=task.precision,
                entry_point=task.entry_point,
                enable_profiling=bool(enable_profiling),
                enable_compute_sanitizer=bool(enable_compute_sanitizer),
                compute_sanitizer_mode=task.compute_sanitizer_mode,
                enable_triton_detection=enable_triton_detection,
                detect_decoy_kernel=detect_decoy_kernel,
                backend_adapter=backend_adapter,
                precompiled_artifact=task.compile_artifact,
                enable_compile_artifact_cache=task.enable_compile_artifact_cache,
                compile_only=str(task.task_stage or "").lower() == "compile" or bool(task.pure_compile_task),
                return_internal_compile_artifact=task.return_internal_compile_artifact,
            )
            if result is None:
                return EvaluationResult(
                    task_id=task.task_id,
                    compiled=False,
                    correctness=False,
                    decoy_kernel=False,
                    reference_runtime=0.0,
                    kernel_runtime=0.0,
                    speedup=0.0,
                    metadata={"error": "eval_kernel_against_ref returned None"},
                    status="failed",
                    error_message="Kernel evaluation failed: empty evaluation result",
                    error_code=ErrorCode.RUNTIME_ERROR,
                )

            if self._prepare_contained_fault_metadata(
                result,
                device=device,
                backend=task.backend,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
            ):
                return EvaluationResult.from_kernel_exec_result(task.task_id, result, 0.0)

            if not run_correctness:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["correctness_skipped"] = True

            reference_runtime = kernelbench_pipeline.eval_reference_only(
                original_model_src=task.reference_code,
                num_perf_trials=task.num_perf_trials,
                num_warmup=num_warmup,
                perf_trim_count=perf_trim_count,
                verbose=False,
                device=device,
                entry_point=task.entry_point,
                backend_adapter=backend_adapter,
            ).runtime

            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device),
                    "backend": task.backend,
                    "num_correct_trials": num_correct_trials,
                    "num_perf_trials": task.num_perf_trials,
                }
            )

            return EvaluationResult.from_kernel_exec_result(task.task_id, result, reference_runtime)

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Evaluation failed: {str(e)}",
                error_code=error_code,
            )

    def evaluate_reference_timing(self, task: ReferenceTimingTask, backend_adapter=None) -> ReferenceTimingResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            if task.reference_backend:
                logger.info("[RefTiming] task=%s reference_backend=%s", task.task_id, task.reference_backend)

            num_warmup = getattr(task, "num_warmup", 3)
            perf_trim_count = getattr(task, "perf_trim_count", 0)

            ref_exec_result = kernelbench_pipeline.eval_reference_only(
                original_model_src=task.reference_code,
                num_perf_trials=task.num_perf_trials,
                num_warmup=num_warmup,
                perf_trim_count=perf_trim_count,
                verbose=False,
                device=device,
                entry_point=task.entry_point,
                reference_backend=task.reference_backend,
                backend_adapter=backend_adapter,
            )
            reference_runtime = ref_exec_result.runtime

            metadata = {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "backend": task.backend,
                "num_perf_trials": task.num_perf_trials,
            }
            if ref_exec_result.metadata:
                metadata.update(ref_exec_result.metadata)

            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=reference_runtime,
                metadata=metadata,
                status="completed",
            )

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=0.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Reference timing failed: {str(e)}",
                error_code=error_code,
            )

    def evaluate_kernel_only(
        self,
        task: KernelEvaluationTask,
        verbose_errors: bool = True,
        enable_profiling: bool = False,
        backend_adapter=None,
    ) -> KernelEvaluationResult:
        task = replace(task, backend=resolve_kernel_backend(task.kernel_code, task.backend))
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"validation_error": kernel_error},
                status="failed",
                error_message=f"Kernel code validation failed: {kernel_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            (
                run_correctness,
                enable_triton_detection,
                measure_performance,
                detect_decoy_kernel,
            ) = self._resolve_eval_flags(task)
            num_correct_trials = task.num_correct_trials if run_correctness else 0
            num_warmup = getattr(task, "num_warmup", 3)
            perf_trim_count = getattr(task, "perf_trim_count", 0)
            enable_compute_sanitizer = task.enable_compute_sanitizer
            if enable_compute_sanitizer is None:
                enable_compute_sanitizer = settings.enable_compute_sanitizer

            result = kernelbench_pipeline.eval_kernel_against_ref(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
                num_warmup=num_warmup,
                perf_trim_count=perf_trim_count,
                measure_performance=measure_performance,
                verbose=False,
                device=device,
                backend=task.backend,
                precision=task.precision,
                entry_point=task.entry_point,
                enable_profiling=enable_profiling,
                enable_compute_sanitizer=bool(enable_compute_sanitizer),
                compute_sanitizer_mode=task.compute_sanitizer_mode,
                enable_triton_detection=enable_triton_detection,
                detect_decoy_kernel=detect_decoy_kernel,
                backend_adapter=backend_adapter,
                precompiled_artifact=task.compile_artifact,
                enable_compile_artifact_cache=task.enable_compile_artifact_cache,
                compile_only=str(task.task_stage or "").lower() == "compile" or bool(task.pure_compile_task),
                return_internal_compile_artifact=task.return_internal_compile_artifact,
            )
            if result is None:
                return KernelEvaluationResult(
                    task_id=task.task_id,
                    base_task_id=task.base_task_id,
                    compiled=False,
                    correctness=False,
                    decoy_kernel=False,
                    kernel_runtime=0.0,
                    metadata={"error": "eval_kernel_against_ref returned None"},
                    status="failed",
                    error_message="Kernel evaluation failed: empty evaluation result",
                    error_code=ErrorCode.RUNTIME_ERROR,
                )

            if self._prepare_contained_fault_metadata(
                result,
                device=device,
                backend=task.backend,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
            ):
                return KernelEvaluationResult.from_kernel_exec_result(
                    task.task_id,
                    task.base_task_id,
                    result,
                    verbose_errors=verbose_errors,
                )

            compile_only = str(task.task_stage or "").lower() == "compile" or bool(task.pure_compile_task)
            if compile_only:
                metadata = dict(result.metadata or {})
                metadata["compile_only"] = True
                return KernelEvaluationResult(
                    task_id=task.task_id,
                    base_task_id=task.base_task_id,
                    compiled=result.compiled,
                    correctness=False,
                    decoy_kernel=False,
                    kernel_runtime=-1.0,
                    metadata=metadata,
                    status="completed" if result.compiled else "failed",
                    error_message=None if result.compiled else "Kernel compilation failed",
                    error_code=None if result.compiled else ErrorCode.COMPILATION_ERROR,
                )

            if not run_correctness:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["correctness_skipped"] = True

            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device),
                    "backend": task.backend,
                    "num_correct_trials": num_correct_trials,
                    "num_perf_trials": task.num_perf_trials,
                }
            )

            if enable_profiling and "profiling" in result.metadata:
                profiling_metrics = result.metadata["profiling"]
                if profiling_metrics:
                    logger.debug("Profiling captured %s kernels", profiling_metrics.get("kernel_count", 0))

            return KernelEvaluationResult.from_kernel_exec_result(
                task.task_id,
                task.base_task_id,
                result,
                verbose_errors=verbose_errors,
            )

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Kernel evaluation failed: {str(e)}",
                error_code=error_code,
            )
