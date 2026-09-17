"""Tests for the fixed GEMM + RMSNorm speed-test endpoint."""

import asyncio

from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend
from kernelgym.common import TaskStatus
from kernelgym.server.api import server
from kernelgym.server.api.speed_test import KERNEL_CODE, REPEAT_COUNT, build_payload
from kernelgym.server.task_manager import TaskManager
from kernelgym.toolkit.validation import precheck_tvm_ffi_submission


def test_fixed_case_passes_tvm_ffi_precheck():
    sources, model_code = KernelBenchTvmFfiBackend._parse_embedded_sources(KERNEL_CODE)

    error_message, error_code, precheck = precheck_tvm_ffi_submission(model_code, sources)

    assert error_message == ""
    assert error_code is None
    assert precheck["passed"] is True
    assert precheck["exported_functions"] == ["gemm_rmsnorm_forward"]
    assert precheck["detected_extension_calls"] == ["gemm_rmsnorm_forward"]
    assert KERNEL_CODE.count("__global__ void") == 1
    assert "wmma::mma_sync" in KERNEL_CODE
    assert "gemm_rmsnorm_kernel" in KERNEL_CODE


def test_speed_test_payload_forces_full_uncached_flow():
    first = build_payload("speed-one", "one")
    second = build_payload("speed-two", "two")

    assert first["backend"] == "tvm_ffi"
    assert first["num_correct_trials"] == 5
    assert first["num_perf_trials"] == 300
    assert first["num_warmup"] == 3
    assert first["force_refresh"] is True
    assert first["use_reference_cache"] is False
    assert first["enable_compile_artifact_cache"] is False
    assert first["enable_ncu"] is True
    assert first["run_correctness"] is True
    assert first["run_performance"] is True
    assert first["kernel_code"] != second["kernel_code"]


def test_speed_test_runs_three_times_and_averages_passed_results(monkeypatch):
    calls = []

    class FakeTaskManager:
        def __init__(self):
            self.discarded = []

        async def get_task_result(self, task_id):
            assert task_id.endswith("_compile")
            return {
                "metadata": {
                    "kg_kernel_backend_compile_s": 3.0,
                    "cpu_worker_run_s": 3.25,
                }
            }

        async def discard_task_records(self, task_ids):
            self.discarded.append(tuple(task_ids))
            return len(tuple(task_ids))

    async def fake_execute_workflow(**kwargs):
        calls.append(kwargs)
        run_index = len(calls)
        result = {
            "task_id": kwargs["task_id"],
            "compiled": True,
            "correctness": True,
            "reference_runtime": float(run_index),
            "kernel_runtime": float(run_index) / 2,
            "speedup": 2.0,
            "metadata": {
                "split_compile_and_execute": True,
                "kg_reference_total_s": float(run_index),
                "kg_kernel_backend_compile_s": float(run_index) * 2,
                "kg_kernel_correctness_s": 0.25,
                "kg_kernel_ncu_profile_s": float(run_index) / 2,
                "ncu": {
                    "status": "ok",
                    "profiled_kernel_count": 1,
                    "kernels": [{"kernel_name": "gemm_rmsnorm_kernel"}],
                },
            },
        }
        return kwargs["task_id"], result, TaskStatus.COMPLETED

    monkeypatch.setattr(server, "_execute_workflow", fake_execute_workflow)

    manager = FakeTaskManager()
    response = asyncio.run(server.benchmark_speed_test(task_mgr=manager))

    assert len(calls) == REPEAT_COUNT == 3
    assert len({call["task_id"] for call in calls}) == 3
    assert all(call["force_refresh"] is True for call in calls)
    assert len(manager.discarded) == 3
    assert all(len(task_ids) == 4 for task_ids in manager.discarded)
    assert all(task_ids[0] == calls[index]["task_id"] for index, task_ids in enumerate(manager.discarded))
    assert response.all_passed is True
    assert response.passed_runs == 3
    assert response.average.reference_runtime_ms == 2.0
    assert response.average.kernel_runtime_ms == 1.0
    assert response.average.speedup == 2.0
    assert response.average.stage_timings["reference_total_s"] == 2.0
    assert response.average.stage_timings["kernel_compile_s"] == 3.0
    assert response.average.stage_timings["compile_worker_total_s"] == 3.25
    assert response.average.stage_timings["kernel_correctness_s"] == 0.25
    assert response.average.stage_timings["ncu_profile_s"] == 1.0
    assert response.runs[0].ncu["status"] == "ok"
    assert response.runs[0].ncu["profiled_kernel_count"] == 1
    assert response.average.end_to_end_s >= 0
    assert response.total_end_to_end_s >= response.average.end_to_end_s


def test_speed_test_route_is_registered():
    paths = {route.path for route in server.app.routes if hasattr(route, "path")}
    assert "/benchmark/speed-test" in paths
    assert "/benchmark/gemm-rmsnorm" not in paths


def test_discard_task_records_removes_current_and_legacy_cache_entries():
    class FakeRedis:
        def __init__(self):
            self.deleted_keys = ()

        async def delete(self, *keys):
            self.deleted_keys = keys
            return len(keys)

        async def hgetall(self, key):
            return {}

    manager = TaskManager.__new__(TaskManager)
    manager.redis = FakeRedis()
    manager.key_prefix = "kernelgym:v1"
    manager.legacy_prefix = "kernelgym"
    manager.active_tasks = {"probe": object(), "probe_ref": object()}
    manager._task_claims = {"probe": object(), "probe_ref": object()}

    deleted = asyncio.run(manager.discard_task_records(["probe", "probe_ref", "probe"]))

    assert deleted == 20
    assert len(manager.redis.deleted_keys) == 20
    assert "kernelgym:v1:result:probe" in manager.redis.deleted_keys
    assert "kernelgym:result:probe_ref" in manager.redis.deleted_keys
    assert manager.active_tasks == {}
    assert manager._task_claims == {}
