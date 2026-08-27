"""Shared task models for KernelBench workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class EvaluationTask:
    task_id: str
    reference_code: str
    kernel_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    precision: str = "fp32"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    num_warmup: int = 3
    perf_trim_count: int = 0
    # Adaptive kernel-perf trial control (None -> fall back to server settings).
    adaptive_perf_trials: Optional[bool] = None
    perf_min_trials: Optional[int] = None
    perf_cv_threshold: Optional[float] = None
    # Correctness-stage timeout control (None -> fall back to server settings).
    correctness_timeout: Optional[float] = None
    correctness_timeout_enabled: Optional[bool] = None
    # Reference perf trial count; None -> reuse num_perf_trials.
    refer_num_perf_trials: Optional[int] = None
    # Warn when Kernel total-task peak memory meets or exceeds this multiple of reference.
    # None disables the warning.
    memory_ratio_warning_threshold: Optional[float] = 1.8
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    required_resource: Optional[str] = None
    task_stage: Optional[str] = None
    assigned_worker: Optional[str] = None
    reference_backend: Optional[str] = None
    device_preference: Optional[str] = None
    target_node_id: Optional[str] = None
    target_hostname: Optional[str] = None
    force_refresh: bool = False
    uuid: Optional[str] = None
    use_reference_cache: bool = False
    is_valid: bool = False
    enable_profiling: Optional[bool] = None
    enable_ncu: Optional[bool] = None
    enable_compute_sanitizer: Optional[bool] = None
    compute_sanitizer_mode: Optional[str] = None
    enable_correctness_input_perturbations: Optional[bool] = None
    enable_triton_detection: Optional[bool] = None
    detect_decoy_kernel: Optional[bool] = None
    measure_performance: Optional[bool] = None
    run_correctness: Optional[bool] = None
    run_triton_detection: Optional[bool] = None
    run_performance: Optional[bool] = None
    compile_artifact: Optional[Dict[str, Any]] = None
    split_compile_and_execute: bool = False
    pure_compile_task: bool = False
    enable_compile_artifact_cache: bool = False
    return_internal_compile_artifact: bool = False
    resources: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class ReferenceTimingTask:
    task_id: str
    base_task_id: str
    reference_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    num_perf_trials: int = 100
    num_warmup: int = 3
    perf_trim_count: int = 0
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    required_resource: Optional[str] = None
    task_stage: Optional[str] = None
    assigned_worker: Optional[str] = None
    reference_backend: Optional[str] = None
    device_preference: Optional[str] = None
    resources: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTimingTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class KernelEvaluationTask:
    task_id: str
    base_task_id: str
    reference_code: str
    kernel_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    precision: str = "fp32"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    num_warmup: int = 3
    perf_trim_count: int = 0
    # Adaptive kernel-perf trial control (None -> fall back to server settings).
    adaptive_perf_trials: Optional[bool] = None
    perf_min_trials: Optional[int] = None
    perf_cv_threshold: Optional[float] = None
    # Correctness-stage timeout control (None -> fall back to server settings).
    correctness_timeout: Optional[float] = None
    correctness_timeout_enabled: Optional[bool] = None
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    required_resource: Optional[str] = None
    task_stage: Optional[str] = None
    assigned_worker: Optional[str] = None
    device_preference: Optional[str] = None
    force_refresh: bool = False
    enable_profiling: Optional[bool] = None
    enable_ncu: Optional[bool] = None
    enable_compute_sanitizer: Optional[bool] = None
    compute_sanitizer_mode: Optional[str] = None
    enable_correctness_input_perturbations: Optional[bool] = None
    enable_triton_detection: Optional[bool] = None
    detect_decoy_kernel: Optional[bool] = None
    measure_performance: Optional[bool] = None
    run_correctness: Optional[bool] = None
    run_triton_detection: Optional[bool] = None
    run_performance: Optional[bool] = None
    compile_artifact: Optional[Dict[str, Any]] = None
    split_compile_and_execute: bool = False
    pure_compile_task: bool = False
    enable_compile_artifact_cache: bool = False
    return_internal_compile_artifact: bool = False
    resources: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelEvaluationTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
