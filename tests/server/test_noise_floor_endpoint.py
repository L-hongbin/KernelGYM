"""Tests for speedup noise-floor calibration."""

from __future__ import annotations

import asyncio
import math

import pytest

from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend
from kernelgym.common import TaskStatus
from kernelgym.server.api import server
from kernelgym.server.api.models import SpeedupNoiseFloorCalibrationRequest
from kernelgym.server.api.noise_floor import (
    analyze_calibration,
    build_calibration_payload,
    build_interleaved_schedule,
    extract_block_result,
)
from kernelgym.server.api.noise_floor_cases import get_noise_floor_cases
from kernelgym.toolkit.validation import precheck_tvm_ffi_submission


def test_fixed_suite_has_ten_real_tvm_ffi_cases_and_seven_dataset_selections():
    cases = get_noise_floor_cases()

    assert len(cases) == 10
    assert {case.dataset_problem_id for case in cases if case.dataset_problem_id is not None} == {
        3,
        12,
        19,
        40,
        44,
        47,
        88,
    }
    assert {case.case_id for case in cases[:3]} == {"gemm_rmsnorm", "batch_norm", "conv2d"}
    for case in cases:
        sources, model_code = KernelBenchTvmFfiBackend._parse_embedded_sources(case.kernel_code)
        error_message, error_code, precheck = precheck_tvm_ffi_submission(model_code, sources)
        assert error_message == ""
        assert error_code is None
        assert precheck["passed"] is True
        assert precheck["exported_functions"] == precheck["detected_extension_calls"]


def test_schedule_is_seeded_and_payload_only_caches_compilation():
    cases = get_noise_floor_cases()
    schedule = build_interleaved_schedule(cases, 10, 7)

    assert len(schedule) == 100
    assert schedule == build_interleaved_schedule(cases, 10, 7)
    for block_index in range(1, 11):
        round_cases = [case.case_id for case, block in schedule if block == block_index]
        assert len(round_cases) == len(set(round_cases)) == 10

    payload = build_calibration_payload(
        cases[0],
        task_id="calibration-task",
        warmup=3,
        trials=50,
        reference_trials=50,
        trim_count=0,
        timeout=300,
        target_node_id=None,
        target_hostname=None,
    )
    assert payload["num_perf_trials"] == 50
    assert payload["refer_num_perf_trials"] == 50
    assert payload["num_warmup"] == 3
    assert payload["perf_trim_count"] == 0
    assert payload["force_refresh"] is True
    assert payload["use_reference_cache"] is False
    assert payload["enable_compile_artifact_cache"] is True
    assert payload["kernel_code"] == cases[0].kernel_code


def _synthetic_block(case_id: str, block_index: int, speedup: float, within_variance: float) -> dict:
    return {
        "kernel_id": case_id,
        "block_index": block_index,
        "passed": True,
        "speedup": speedup,
        "log_speedup": math.log(speedup),
        "within_log_speedup_variance": within_variance,
        "kg_kernel_perf_mean_ms": 0.2,
    }


def test_analysis_subtracts_within_variance_and_uses_p75_floor():
    cases = get_noise_floor_cases()[:2]
    blocks = []
    for case_index, case in enumerate(cases):
        for block_index in range(1, 11):
            log_speedup = 0.2 + (block_index - 5.5) * (0.01 + case_index * 0.002)
            blocks.append(_synthetic_block(case.case_id, block_index, math.exp(log_speedup), 0.00001))

    result = analyze_calibration(
        cases,
        blocks,
        heldout_blocks_per_kernel=2,
        global_percentile=75.0,
        z_value=1.645,
    )

    assert result["global"]["valid_kernel_count"] == 2
    floors = [item["noise_floor"] for item in result["per_kernel"]]
    assert floors[0] > 0
    assert result["global"]["noise_floor"] == pytest.approx(floors[0] * 0.25 + floors[1] * 0.75)
    assert result["validation"]["heldout_rows"] == 4
    assert result["execution_environment"]["single_target_device"] is True


def test_calibration_request_requires_two_estimation_blocks():
    with pytest.raises(ValueError):
        SpeedupNoiseFloorCalibrationRequest(blocks_per_kernel=5, heldout_blocks_per_kernel=4)


def test_cached_reference_uses_result_mean_and_omits_reference_variance():
    block = extract_block_result(
        case=get_noise_floor_cases()[0],
        block_index=1,
        schedule_index=1,
        task_id="cached",
        status="completed",
        result={
            "compiled": True,
            "correctness": True,
            "speedup": 2.0,
            "kernel_runtime": 0.1,
            "reference_runtime": 0.2,
            "metadata": {
                "cached": True,
                "kg_kernel_perf_std_ms": 0.01,
                "kg_kernel_perf_num_trials": 50,
            },
        },
        started_at="start",
        completed_at="end",
        end_to_end_s=1.0,
    )

    assert block["passed"] is True
    assert block["kg_reference_perf_mean_ms"] == 0.2
    assert block["within_log_speedup_variance"] == pytest.approx(0.1**2 / 50)


def test_uncached_block_without_reference_std_is_excluded():
    block = extract_block_result(
        case=get_noise_floor_cases()[0],
        block_index=1,
        schedule_index=1,
        task_id="missing-ref-std",
        status="completed",
        result={
            "compiled": True,
            "correctness": True,
            "speedup": 2.0,
            "kernel_runtime": 0.1,
            "reference_runtime": 0.2,
            "metadata": {
                "kg_kernel_perf_std_ms": 0.01,
                "kg_kernel_perf_num_trials": 50,
            },
        },
        started_at="start",
        completed_at="end",
        end_to_end_s=1.0,
    )

    assert block["passed"] is False
    assert block["within_log_speedup_variance"] is None


def test_endpoint_runs_one_independent_request_per_case_block(monkeypatch):
    calls = []

    class FakeTaskManager:
        def __init__(self):
            self.discarded = []

        async def discard_task_records(self, task_ids):
            self.discarded.append(tuple(task_ids))
            return len(tuple(task_ids))

    async def fake_execute_workflow(**kwargs):
        calls.append(kwargs)
        index = len(calls)
        speedup = 1.25 + (index % 5) * 0.002
        return (
            kwargs["task_id"],
            {
                "task_id": kwargs["task_id"],
                "compiled": True,
                "correctness": True,
                "speedup": speedup,
                "metadata": {
                    "kg_kernel_perf_mean_ms": 0.2,
                    "kg_kernel_perf_std_ms": 0.02,
                    "kg_kernel_perf_num_trials": 50,
                    "kg_reference_perf_mean_ms": 0.25,
                    "kg_reference_perf_std_ms": 0.025,
                    "kg_reference_perf_num_trials": 50,
                    "kernel_execution_pid": 1000 + index,
                    "reference_execution_pid": 2000 + index,
                    "kernel_execution_device": "cuda:0",
                },
            },
            TaskStatus.COMPLETED,
        )

    monkeypatch.setattr(server, "_execute_workflow", fake_execute_workflow)
    manager = FakeTaskManager()
    request = SpeedupNoiseFloorCalibrationRequest(persist_artifact=False)

    response = asyncio.run(server.benchmark_speedup_noise_floor(request=request, task_mgr=manager))

    assert response.calibration_status == "passed"
    assert len(calls) == 100
    assert len({call["task_id"] for call in calls}) == 100
    assert len(response.blocks) == 100
    assert len(manager.discarded) == 100
    assert response.analysis["global"]["valid_kernel_count"] == 10
    assert all(block["process"]["kernel_pid"] is not None for block in response.blocks)
    assert response.analysis["execution_environment"]["fresh_kernel_process_per_block"] is True


def test_noise_floor_route_is_registered():
    paths = {route.path for route in server.app.routes if hasattr(route, "path")}
    assert "/benchmark/speedup-noise-floor" in paths
