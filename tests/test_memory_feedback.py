from kernelgym.schema.result import (
    EvaluationResult,
    KernelEvaluationResult,
    ReferenceTimingResult,
)
from kernelgym.server.api.models import EvaluationResponse
from kernelgym.toolkit.kernelbench import pipeline as kernelbench_pipeline
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult
from kernelgym.toolkit.kernelbench.memory import detect_direct_cuda_allocations
from kernelgym.workflow.reference_cache import ReferenceRuntimeCache


def _memory(
    peak: int,
    *,
    complete: bool = True,
    persistent: int = 0,
    floor_allocated: int | None = None,
    floor_reserved: int | None = None,
) -> dict:
    memory = {
        "method": "torch_cuda_peak_allocated_delta",
        "allocator_scope": "pytorch_cuda_caching_allocator",
        "forward_incremental_peak_allocated_bytes": peak,
        "persistent_allocated_bytes": persistent,
        "total_task_peak_allocated_bytes": persistent + peak,
        "absolute_peak_allocated_bytes": (floor_allocated or 0) + persistent + peak,
        "measurement_valid": True,
        "measurement_complete": complete,
        "measurement_is_lower_bound": not complete,
        "direct_cuda_allocation_detected": not complete,
        "direct_cuda_allocation_apis": ["cudaMalloc"] if not complete else [],
        "direct_cuda_allocation_matches": [],
        "recommended_comparison_metric": "total_task_peak_allocated_bytes",
    }
    if floor_allocated is not None or floor_reserved is not None:
        memory.update(
            {
                "environment_floor_available": floor_allocated is not None,
                "environment_floor_allocated_bytes": floor_allocated,
                "environment_floor_reserved_bytes": floor_reserved,
            }
        )
    return memory


def _legacy_memory(peak: int) -> dict:
    return {
        "method": "torch_cuda_peak_allocated_delta",
        "allocator_scope": "pytorch_cuda_caching_allocator",
        "peak_allocated_bytes": peak,
        "measurement_valid": True,
        "measurement_complete": True,
        "measurement_is_lower_bound": False,
    }


def _assert_no_bytes_suffix(value) -> None:
    if isinstance(value, dict):
        assert all(not key.endswith("_bytes") for key in value)
        for item in value.values():
            _assert_no_bytes_suffix(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_bytes_suffix(item)


def test_direct_cuda_allocation_detection_reports_calls_but_ignores_comments() -> None:
    source = """
// cudaMalloc(&commented, 4);
/* cuMemAlloc(&also_commented, 8); */
cudaError_t status = cudaMallocAsync(&ptr, bytes, stream);
CUresult driver_status = cuMemAlloc(&driver_ptr, bytes);
"""

    result = detect_direct_cuda_allocations(source)

    assert result["direct_cuda_allocation_detected"] is True
    assert result["direct_cuda_allocation_apis"] == ["cuMemAlloc", "cudaMallocAsync"]
    assert [match["line"] for match in result["direct_cuda_allocation_matches"]] == [
        4,
        5,
    ]
    assert "underestimate" in result["warning"]


def test_memory_fields_are_serialized_at_feedback_top_level() -> None:
    reference_memory = _memory(
        1_000,
        persistent=2_000,
        floor_allocated=256,
        floor_reserved=1024**2,
    )
    kernel_memory = _memory(
        700,
        persistent=1_500,
        floor_allocated=512,
        floor_reserved=2 * 1024**2,
    )
    reference = ReferenceTimingResult(
        task_id="task_ref",
        base_task_id="task",
        reference_runtime=2.0,
        metadata={
            "reference_memory_allocator_check": {"severity": "none"},
            "kg_reference_memory_step_s": 0.12,
        },
        reference_memory=reference_memory,
    )
    kernel = KernelEvaluationResult(
        task_id="task_kernel",
        base_task_id="task",
        compiled=True,
        correctness=True,
        decoy_kernel=False,
        kernel_runtime=1.0,
        metadata={
            "memory_environment_floor": {
                "allocated_bytes": 512,
                "reserved_bytes": 2 * 1024**2,
            },
            "kernel_memory_allocator_check": {"severity": "none"},
            "kg_kernel_memory_step_s": 0.34,
            "memory_measurement_error": "example measurement error",
        },
        kernel_memory=kernel_memory,
    )

    feedback = EvaluationResult.from_paired_results("task", reference, kernel).to_dict()
    memory = feedback["memory"]
    comparison = memory["comparison"]

    assert {
        "reference_memory",
        "kernel_memory",
        "memory_comparison",
    }.isdisjoint(feedback)
    assert memory == {
        "reference": {
            "absolute_peak_allocated": "3.18 KB",
            "task_peak_allocated_delta": "2.93 KB",
            "forward_peak_allocated_delta": "1000.00 B",
        },
        "kernel": {
            "absolute_peak_allocated": "2.65 KB",
            "task_peak_allocated_delta": "2.15 KB",
            "forward_peak_allocated_delta": "700.00 B",
        },
        "comparison": {
            "measurement_status": "complete",
            "kernel_minus_reference": "-800.00 B",
            "kernel_to_reference_ratio": 0.7333333333333333,
        },
        "measurement_error": "example measurement error",
    }
    assert {
        "memory_environment_floor",
        "reference_memory_allocator_check",
        "kernel_memory_allocator_check",
        "kg_reference_memory_step_s",
        "kg_kernel_memory_step_s",
        "memory_measurement_error",
    }.isdisjoint(feedback["metadata"])
    assert reference_memory["total_task_peak_allocated_bytes"] == 3_000
    assert kernel_memory["total_task_peak_allocated_bytes"] == 2_200
    assert comparison["measurement_status"] == "complete"
    _assert_no_bytes_suffix(memory)

    response_payload = EvaluationResponse(**feedback).model_dump()
    assert response_payload["memory"] == memory
    assert {
        "reference_memory",
        "kernel_memory",
        "memory_comparison",
    }.isdisjoint(response_payload)


def test_legacy_memory_is_rejected_without_fallback() -> None:
    reference = ReferenceTimingResult(
        task_id="task_ref",
        base_task_id="task",
        reference_runtime=2.0,
        metadata={},
        reference_memory=_legacy_memory(1_000),
    )
    kernel = KernelEvaluationResult(
        task_id="task_kernel",
        base_task_id="task",
        compiled=True,
        correctness=True,
        decoy_kernel=False,
        kernel_runtime=1.0,
        metadata={},
        kernel_memory=_legacy_memory(700),
    )

    memory = EvaluationResult.from_paired_results("task", reference, kernel).to_dict()[
        "memory"
    ]
    comparison = memory["comparison"]

    assert comparison["measurement_status"] == "invalid"
    assert comparison["kernel_minus_reference"] is None
    assert "unavailable" in comparison["warnings"][0]


def test_direct_allocation_marks_feedback_comparison_as_lower_bound() -> None:
    reference = ReferenceTimingResult(
        task_id="task_ref",
        base_task_id="task",
        reference_runtime=2.0,
        metadata={},
        reference_memory=_memory(1_000),
    )
    kernel = KernelEvaluationResult.from_kernel_exec_result(
        "task_kernel",
        "task",
        KernelExecResult(
            compiled=True,
            correctness=True,
            runtime=1.0,
            memory=_memory(700, complete=False),
            metadata={
                "kernel_memory_allocator_check": {
                    "severity": "warning",
                    "measurement_impact": "may_underestimate",
                    "direct_cuda_allocation_detected": True,
                    "direct_cuda_allocation_apis": ["cudaMalloc"],
                    "direct_cuda_allocation_matches": [
                        {"api": "cudaMalloc", "line": 3, "snippet": "cudaMalloc(...)"}
                    ],
                }
            },
        ),
    )

    feedback = EvaluationResult.from_paired_results("task", reference, kernel).to_dict()

    comparison = feedback["memory"]["comparison"]
    assert comparison["measurement_status"] == "partial"
    assert comparison["kernel_minus_reference"] == "-300.00 B"
    assert "lower bound" in comparison["warnings"][0]
    assert feedback["memory"]["allocator_check"] == {
        "severity": "warning",
        "measurement_impact": "may_underestimate",
        "detected_apis": ["cudaMalloc"],
        "matches": [{"api": "cudaMalloc", "line": 3, "snippet": "cudaMalloc(...)"}],
    }


def test_reference_cache_round_trips_current_memory_and_rejects_legacy_entries() -> None:
    cache = ReferenceRuntimeCache()
    reference_code = "class Model: pass"
    reference_memory = _memory(4_096)

    cache.put("new", reference_code, False, 1.25, reference_memory)
    cache.put("legacy", reference_code, False, 1.5, _legacy_memory(4_096))

    assert cache.get("new", reference_code, False) == 1.25
    assert cache.get_memory("new", reference_code, False) == reference_memory
    assert cache.get("legacy", reference_code, False) == 1.5
    assert cache.get_memory("legacy", reference_code, False) is None


def test_memory_step_runs_allocator_check_once_and_reuses_result(monkeypatch) -> None:
    allocation_check = {
        "direct_cuda_allocation_detected": True,
        "direct_cuda_allocation_apis": ["cudaMalloc"],
        "direct_cuda_allocation_matches": [{"api": "cudaMalloc", "line": 1}],
        "warning": "direct allocation warning",
    }
    detected_sources = []
    measured_kwargs = {}

    def fake_detect(source):
        detected_sources.append(source)
        return allocation_check

    def fake_measure(model, *inputs, **kwargs):
        measured_kwargs.update(kwargs)
        return {"measurement_valid": True, "measurement_complete": False}

    monkeypatch.setattr(
        kernelbench_pipeline, "detect_direct_cuda_allocations", fake_detect
    )
    monkeypatch.setattr(kernelbench_pipeline, "measure_cuda_memory_trial", fake_measure)
    monkeypatch.setattr(kernelbench_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        kernelbench_pipeline.torch.random, "get_rng_state", lambda: "cpu-rng"
    )
    monkeypatch.setattr(
        kernelbench_pipeline.torch.random, "set_rng_state", lambda state: None
    )
    monkeypatch.setattr(
        kernelbench_pipeline.torch.cuda,
        "get_rng_state",
        lambda device: "cuda-rng",
    )
    monkeypatch.setattr(
        kernelbench_pipeline.torch.cuda,
        "set_rng_state",
        lambda state, device: None,
    )
    monkeypatch.setattr(
        kernelbench_pipeline.torch.cuda, "synchronize", lambda device: None
    )

    class FakeModel:
        def cuda(self, device):
            return self

    result = KernelExecResult(correctness=True)
    metadata = {}
    kernelbench_pipeline._run_memory_step(
        kernel_exec_result=result,
        model=FakeModel(),
        get_inputs=lambda: [],
        source="cudaMalloc(&ptr, bytes);",
        metadata=metadata,
        allocator_check_metadata_key="kernel_memory_allocator_check",
        seed_num=42,
        environment_floor={"allocated_bytes": 0, "reserved_bytes": 0},
        device=0,
        verbose=False,
    )

    assert detected_sources == ["cudaMalloc(&ptr, bytes);"]
    assert metadata["kernel_memory_allocator_check"] is allocation_check
    assert measured_kwargs["allocation_check"] is allocation_check
    assert result.memory == {"measurement_valid": True, "measurement_complete": False}


def test_memory_step_skips_allocator_check_when_correctness_fails(monkeypatch) -> None:
    def fail_if_detected(source):
        raise AssertionError("allocator check should not run")

    monkeypatch.setattr(
        kernelbench_pipeline, "detect_direct_cuda_allocations", fail_if_detected
    )
    metadata = {}

    kernelbench_pipeline._run_memory_step(
        kernel_exec_result=KernelExecResult(correctness=False),
        model=None,
        get_inputs=lambda: [],
        source="cudaMalloc(&ptr, bytes);",
        metadata=metadata,
        allocator_check_metadata_key="kernel_memory_allocator_check",
        seed_num=42,
        environment_floor={},
        device=0,
        verbose=False,
    )

    assert "kernel_memory_allocator_check" not in metadata


def test_memory_environment_floor_is_grouped_without_trial_results() -> None:
    result = EvaluationResult(
        task_id="task",
        compiled=False,
        correctness=False,
        decoy_kernel=False,
        reference_runtime=0.0,
        kernel_runtime=0.0,
        speedup=0.0,
        metadata={
            "memory_environment_floor": {
                "allocated_bytes": 512,
                "reserved_bytes": 2 * 1024**2,
            }
        },
    )

    payload = result.to_dict()

    assert payload["memory"] is None
    assert "memory_environment_floor" not in payload["metadata"]


def test_memory_response_formats_bytes_with_adaptive_units() -> None:
    result = EvaluationResult(
        task_id="task",
        compiled=True,
        correctness=True,
        decoy_kernel=False,
        reference_runtime=2.0,
        kernel_runtime=1.0,
        speedup=2.0,
        metadata={
            "memory_environment_floor": {
                "allocated_bytes": 512,
                "reserved_bytes": 2 * 1024**3,
            }
        },
        kernel_memory={
            "small_bytes": 512,
            "kilobyte_bytes": 1536,
            "megabyte_bytes": 5 * 1024**2 // 2,
            "gigabyte_bytes": 3 * 1024**3,
            "forward_incremental_peak_allocated_bytes": 1536,
            "total_task_peak_allocated_bytes": 3 * 1024**3,
            "absolute_peak_allocated_bytes": 4 * 1024**3,
            "measurement_valid": True,
            "measurement_complete": True,
            "measurement_is_lower_bound": False,
        },
        memory_comparison={
            "measurement_valid": True,
            "measurement_complete": True,
            "primary_kernel_minus_reference_bytes": -(1024**2),
            "primary_kernel_to_reference_ratio": 0.5,
        },
    )

    payload = result.to_dict()

    memory = payload["memory"]
    assert memory == {
        "kernel": {
            "absolute_peak_allocated": "4.00 GB",
            "task_peak_allocated_delta": "3.00 GB",
            "forward_peak_allocated_delta": "1.50 KB",
        },
        "comparison": {
            "measurement_status": "complete",
            "kernel_minus_reference": "-1.00 MB",
            "kernel_to_reference_ratio": 0.5,
        },
    }
    _assert_no_bytes_suffix(memory)
    assert "memory_environment_floor" not in payload["metadata"]
