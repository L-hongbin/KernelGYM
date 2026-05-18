"""KernelBench workflow controller (server-side orchestration)."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import json
from datetime import datetime, timezone

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.toolkit.kernelbench.binding_detection import resolve_kernel_backend
from .kernelbench_types import (
    EvaluationTask,
    ReferenceTimingTask,
    ReferenceTimingResult,
    KernelEvaluationResult,
    EvaluationResult,
)
from .kernelbench_helpers import (
    _combine_results,
    _create_paired_tasks,
    _get_cached_reference_runtime,
    _put_cached_reference_runtime,
    _validate_code,
)

from ..core.types import TaskSpec
from ..core.workflow import WorkflowController, WorkflowState
from ..core.scheduler import SchedulerAPI
from kernelgym.utils.task_status import task_status_from_result_payload


class KernelBenchWorkflowController(WorkflowController):
    """Main controller for KernelBench evaluation workflow."""

    async def validate_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        eval_task = EvaluationTask.from_dict(input_data)
        self._resolve_auto_backend(eval_task)
        validation = self._validate_inputs(eval_task)
        validation["task_id"] = eval_task.task_id
        validation["workflow"] = "kernelbench"
        return validation

    async def handle_request(self, input_data: Dict[str, Any], scheduler: SchedulerAPI) -> Dict[str, Any]:
        eval_task = EvaluationTask.from_dict(input_data)
        self._resolve_auto_backend(eval_task)
        state = WorkflowState({"base_task_id": eval_task.task_id})

        if eval_task.reference_backend:
            print(f"[Workflow] task={eval_task.task_id} reference_backend={eval_task.reference_backend}")

        validation = self._validate_inputs(eval_task)
        if not validation["valid"]:
            message = validation["errors"][0] if validation["errors"] else "Validation failed"
            result = self._validation_failed_result(eval_task.task_id, message)
            self._persist_result(eval_task, result)
            return result

        compile_only = self._is_compile_only(eval_task)
        ref_task, kernel_task = _create_paired_tasks(eval_task)

        if eval_task.split_compile_and_execute and not compile_only:
            kernel_result_dict = await self._run_split_kernel_task(eval_task, kernel_task, scheduler)
        else:
            kernel_payload = kernel_task.to_dict()
            if compile_only:
                kernel_payload["task_id"] = f"{eval_task.task_id}_compile"
            kernel_payload["task_type"] = "kernel_evaluation"
            kernel_payload["toolkit"] = kernel_payload.get("toolkit", "kernelbench")
            kernel_payload["backend_adapter"] = kernel_payload.get("backend_adapter", "kernelbench")
            kernel_payload.update(self._kernel_execution_options(eval_task, compile_only=compile_only))
            kernel_task_spec = TaskSpec(
                kind="kernelbench.kernel",
                payload=kernel_payload,
                resources=eval_task.resources,
                metadata={"base_task_id": eval_task.task_id},
            )
            kernel_task_id = await scheduler.submit(kernel_task_spec)
            kernel_result_dict = await scheduler.wait(kernel_task_id)

        if not kernel_result_dict:
            result = self._failed_result(eval_task.task_id, "kernel result missing")
            self._persist_result(eval_task, result)
            return result
        if "error_message" in kernel_result_dict and "compiled" not in kernel_result_dict:
            result = self._failed_result(
                eval_task.task_id,
                kernel_result_dict.get("error_message", "kernel task failed"),
                kernel_result_dict.get("error_code"),
            )
            self._persist_result(eval_task, result)
            return result

        required_kernel_fields = {
            "task_id",
            "base_task_id",
            "compiled",
            "correctness",
            "decoy_kernel",
            "kernel_runtime",
            "metadata",
        }
        if not required_kernel_fields.issubset(kernel_result_dict.keys()):
            missing = sorted(required_kernel_fields - set(kernel_result_dict.keys()))
            result = self._failed_result(
                eval_task.task_id,
                f"kernel result missing required fields: {missing}",
            )
            self._persist_result(eval_task, result)
            return result

        kernel_result = KernelEvaluationResult.from_dict(kernel_result_dict)
        state.data["kernel_result"] = kernel_result.to_dict()

        if compile_only:
            result = self._kernel_only_result(eval_task, kernel_result)
            self._persist_result(eval_task, result)
            return result

        if not (kernel_result.compiled and kernel_result.correctness):
            result = self._kernel_only_result(eval_task, kernel_result)
            self._persist_result(eval_task, result)
            return result

        ref_result: Optional[ReferenceTimingResult] = None
        if ref_task is None:
            cached_runtime = _get_cached_reference_runtime(
                eval_task.uuid, eval_task.reference_code, eval_task.is_valid
            )
            if cached_runtime is not None:
                ref_result = self._cached_reference_result(eval_task, cached_runtime)
            else:
                ref_task = ReferenceTimingTask(
                    task_id=f"{eval_task.task_id}_ref",
                    base_task_id=eval_task.task_id,
                    reference_code=eval_task.reference_code,
                    toolkit=eval_task.toolkit,
                    backend_adapter=eval_task.backend_adapter,
                    backend=eval_task.backend,
                    num_perf_trials=eval_task.num_perf_trials,
                    num_warmup=eval_task.num_warmup,
                    perf_trim_count=eval_task.perf_trim_count,
                    timeout=eval_task.timeout,
                    device=eval_task.device,
                    priority=eval_task.priority,
                    entry_point=eval_task.entry_point,
                    reference_backend=eval_task.reference_backend,
                    device_preference=eval_task.device_preference,
                    resources=eval_task.resources,
                )

        if ref_result is None and ref_task is not None:
            ref_payload = ref_task.to_dict()
            ref_payload["task_type"] = "reference_timing"
            ref_payload["toolkit"] = ref_payload.get("toolkit", "kernelbench")
            ref_payload["backend_adapter"] = ref_payload.get("backend_adapter", "kernelbench")
            ref_task_spec = TaskSpec(
                kind="kernelbench.ref",
                payload=ref_payload,
                resources=eval_task.resources,
                metadata={"base_task_id": eval_task.task_id},
            )
            ref_task_id = await scheduler.submit(ref_task_spec)
            ref_result_dict = await scheduler.wait(ref_task_id)
            if ref_result_dict:
                if "error_message" in ref_result_dict and "reference_runtime" not in ref_result_dict:
                    result = self._kernel_only_result(eval_task, kernel_result)
                    self._persist_result(eval_task, result)
                    return result
                ref_result = ReferenceTimingResult.from_dict(ref_result_dict)

        if ref_result is None:
            result = self._kernel_only_result(eval_task, kernel_result)
            self._persist_result(eval_task, result)
            return result

        if eval_task.use_reference_cache and eval_task.uuid:
            _put_cached_reference_runtime(
                eval_task.uuid,
                eval_task.reference_code,
                eval_task.is_valid,
                ref_result.reference_runtime,
            )

        combined = _combine_results(ref_result, kernel_result)
        result = combined.to_dict()
        self._persist_result(eval_task, result)
        return result

    @staticmethod
    def _resolve_auto_backend(eval_task: EvaluationTask) -> None:
        eval_task.backend = resolve_kernel_backend(eval_task.kernel_code, eval_task.backend)

    @staticmethod
    def _first_not_none(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _is_compile_only(self, eval_task: EvaluationTask) -> bool:
        return str(eval_task.task_stage or "").lower() == "compile" or bool(eval_task.pure_compile_task)

    def _kernel_execution_options(self, eval_task: EvaluationTask, *, compile_only: bool) -> Dict[str, Any]:
        run_correctness = self._first_not_none(eval_task.run_correctness, True)
        run_triton_detection = self._first_not_none(
            eval_task.run_triton_detection,
            eval_task.enable_triton_detection,
            eval_task.backend == "triton",
        )
        run_performance = self._first_not_none(eval_task.run_performance, eval_task.measure_performance, True)
        enable_profiling = self._first_not_none(eval_task.enable_profiling, settings.enable_profiling)
        if compile_only:
            run_correctness = False
            run_performance = False
            enable_profiling = False
        return {
            "run_correctness": run_correctness,
            "run_triton_detection": run_triton_detection,
            "run_performance": run_performance,
            "enable_triton_detection": run_triton_detection,
            "measure_performance": run_performance,
            "enable_profiling": enable_profiling,
        }

    async def _run_split_kernel_task(
        self,
        eval_task: EvaluationTask,
        kernel_task: Any,
        scheduler: SchedulerAPI,
    ) -> Dict[str, Any]:
        execute_options = self._kernel_execution_options(eval_task, compile_only=False)
        compile_payload = kernel_task.to_dict()
        compile_payload.update(
            {
                "task_id": f"{eval_task.task_id}_compile",
                "task_type": "kernel_evaluation",
                "toolkit": compile_payload.get("toolkit", "kernelbench"),
                "backend_adapter": compile_payload.get("backend_adapter", "kernelbench"),
                "task_stage": "compile",
                "required_resource": "cpu",
                "pure_compile_task": True,
                "return_internal_compile_artifact": True,
                "enable_compile_artifact_cache": eval_task.enable_compile_artifact_cache,
            }
        )
        compile_payload.update(self._kernel_execution_options(eval_task, compile_only=True))
        compile_task_spec = TaskSpec(
            kind="kernelbench.kernel.compile",
            payload=compile_payload,
            resources=eval_task.resources,
            metadata={"base_task_id": eval_task.task_id, "stage": "compile"},
        )
        compile_task_id = await scheduler.submit(compile_task_spec)
        compile_result = await scheduler.wait(compile_task_id)
        if not compile_result:
            return self._failed_result(eval_task.task_id, "compile result missing")

        compile_metadata = compile_result.get("metadata") or {}
        compile_artifact = compile_metadata.get("_internal_compile_artifact")
        if compile_result.get("compiled") is not True:
            result = dict(compile_result)
            metadata = dict(compile_metadata)
            metadata["split_compile_and_execute"] = True
            metadata.pop("_internal_compile_artifact", None)
            result["metadata"] = metadata
            return result
        if not isinstance(compile_artifact, dict):
            return self._failed_result(eval_task.task_id, "compile stage did not return a reusable compile_artifact")

        execute_payload = kernel_task.to_dict()
        execute_payload.update(
            {
                "task_id": f"{eval_task.task_id}_kernel",
                "task_type": "kernel_evaluation",
                "toolkit": execute_payload.get("toolkit", "kernelbench"),
                "backend_adapter": execute_payload.get("backend_adapter", "kernelbench"),
                "task_stage": "execute",
                "required_resource": "gpu",
                "compile_artifact": compile_artifact,
                "pure_compile_task": False,
                "enable_compile_artifact_cache": eval_task.enable_compile_artifact_cache,
            }
        )
        execute_payload.update(execute_options)
        execute_task_spec = TaskSpec(
            kind="kernelbench.kernel.execute",
            payload=execute_payload,
            resources=eval_task.resources,
            metadata={"base_task_id": eval_task.task_id, "stage": "execute", "compile_task_id": compile_task_id},
        )
        execute_task_id = await scheduler.submit(execute_task_spec)
        execute_result = await scheduler.wait(execute_task_id)
        if execute_result:
            execute_result = dict(execute_result)
            metadata = dict(execute_result.get("metadata") or {})
            metadata["split_compile_and_execute"] = True
            metadata.pop("_internal_compile_artifact", None)
            execute_result["metadata"] = metadata
        return execute_result

    def _cached_reference_result(self, eval_task: EvaluationTask, runtime: float) -> ReferenceTimingResult:
        return ReferenceTimingResult(
            task_id=f"{eval_task.task_id}_ref",
            base_task_id=eval_task.task_id,
            reference_runtime=runtime,
            metadata={
                "cached": True,
                "uuid": eval_task.uuid,
                "device": "cached",
                "backend": eval_task.backend,
                "cache_type": "validation" if eval_task.is_valid else "regular",
            },
            status="completed",
        )

    def _kernel_only_result(self, eval_task: EvaluationTask, kernel_result: KernelEvaluationResult) -> Dict[str, Any]:
        metadata = dict(kernel_result.metadata or {})
        metadata["kernel_task_id"] = kernel_result.task_id
        result = EvaluationResult(
            task_id=eval_task.task_id,
            compiled=kernel_result.compiled,
            correctness=kernel_result.correctness,
            decoy_kernel=kernel_result.decoy_kernel,
            reference_runtime=-1.0,
            kernel_runtime=kernel_result.kernel_runtime,
            speedup=0.0,
            metadata=metadata,
            status=kernel_result.status,
            error_message=kernel_result.error_message,
            error_code=kernel_result.error_code,
        )
        return result.to_dict()

    def _failed_result(
        self, task_id: str, message: str, error_code: Optional[ErrorCode | str] = None
    ) -> Dict[str, Any]:
        status = task_status_from_result_payload({"status": "failed", "error_code": error_code}).value
        result = EvaluationResult(
            task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=-1.0,
            kernel_runtime=-1.0,
            speedup=0.0,
            metadata={"error": message},
            status=status,
            error_message=message,
            error_code=error_code,
        )
        return result.to_dict()

    def _validation_failed_result(self, task_id: str, message: str) -> Dict[str, Any]:
        result = EvaluationResult(
            task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=-1.0,
            kernel_runtime=-1.0,
            speedup=0.0,
            metadata={"error": message},
            status="failed",
            error_message=message,
            error_code=ErrorCode.VALIDATION_ERROR.value,
        )
        return result.to_dict()

    def _validate_inputs(self, eval_task: EvaluationTask) -> Dict[str, Any]:
        errors = []

        resources = eval_task.resources or {}
        if resources:
            gpus = resources.get("gpus")
            if gpus is not None:
                try:
                    gpus_int = int(gpus)
                    if gpus_int < 1:
                        errors.append("resources.gpus must be >= 1")
                except (TypeError, ValueError):
                    errors.append("resources.gpus must be an integer")

        if eval_task.use_reference_cache and not eval_task.uuid:
            errors.append("UUID is required when use_reference_cache is True")

        ref_valid, ref_error = _validate_code(eval_task.reference_code, eval_task.entry_point)
        if not ref_valid:
            errors.append(f"Reference code validation failed: {ref_error}")

        kernel_entry_point = f"{eval_task.entry_point}New"
        kernel_valid, kernel_error = _validate_code(eval_task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            errors.append(f"Kernel code validation failed: {kernel_error}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "reference": {"valid": ref_valid, "error": ref_error, "entry_point": eval_task.entry_point},
            "kernel": {"valid": kernel_valid, "error": kernel_error, "entry_point": kernel_entry_point},
            "cache": {"use_reference_cache": eval_task.use_reference_cache, "uuid": eval_task.uuid},
            "resources": resources,
        }

    def _persist_result(self, eval_task: EvaluationTask, result: Dict[str, Any]) -> None:
        if not settings.save_eval_results:
            return
        try:
            path = Path(settings.eval_results_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": eval_task.task_id,
                "base_task_id": eval_task.task_id,
                "toolkit": eval_task.toolkit,
                "backend": eval_task.backend,
                "result": result,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:
            return
