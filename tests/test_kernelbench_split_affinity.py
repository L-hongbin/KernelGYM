import asyncio
from typing import Any

from kernelgym.workflow.kernelbench import KernelBenchWorkflowController
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks
from kernelgym.schema.task import EvaluationTask


class FakeScheduler:
    def __init__(self) -> None:
        self.submissions: list[dict[str, Any]] = []
        self.selection_kwargs: dict[str, Any] = {}

    async def select_idle_worker(self, resource: str, **kwargs) -> dict[str, str]:
        assert resource == "gpu"
        self.selection_kwargs = kwargs
        return {
            "worker_id": "node-b_gpu_3",
            "device": "cuda:3",
            "node_id": "node-b",
            "hostname": "host-b",
        }

    async def submit(self, task) -> str:
        self.submissions.append(dict(task.payload))
        return task.payload["task_id"]

    async def wait(self, task_id: str, timeout=None) -> dict[str, Any]:
        if task_id.endswith("_compile"):
            return {
                "task_id": task_id,
                "base_task_id": "parent",
                "compiled": True,
                "correctness": False,
                "decoy_kernel": False,
                "kernel_runtime": -1.0,
                "metadata": {
                    "compile_node_id": "node-b",
                    "compile_hostname": "host-b",
                    "_internal_compile_artifact": {
                        "compiled": True,
                        "so_path": "/dev/shm/kernelgym/example.so",
                    },
                },
            }
        return {
            "task_id": task_id,
            "base_task_id": "parent",
            "compiled": True,
            "correctness": False,
            "decoy_kernel": False,
            "kernel_runtime": -1.0,
            "metadata": {},
        }

    async def wait_unless_cancelled(self, task_id: str, _base_id: str, timeout=None) -> dict[str, Any]:
        return await self.wait(task_id, timeout)


def test_split_compile_execute_targets_idle_gpu_node() -> None:
    async def scenario() -> None:
        task = EvaluationTask(
            task_id="parent",
            reference_code="class Model: pass",
            kernel_code="class ModelNew: pass",
            backend="cuda_agent",
            split_compile_and_execute=True,
            enable_compile_artifact_cache=True,
        )
        _ref_task, kernel_task = _create_paired_tasks(task)
        scheduler = FakeScheduler()

        result = await KernelBenchWorkflowController()._run_split_kernel_task(task, kernel_task, scheduler)  # noqa: SLF001

        assert result["task_id"] == "parent_kernel"
        compile_payload, execute_payload = scheduler.submissions
        assert compile_payload["required_resource"] == "cpu"
        assert compile_payload["node_affinity"] == "required"
        assert compile_payload["target_node_id"] == "node-b"
        assert compile_payload["target_gpu_worker_id"] == "node-b_gpu_3"
        assert compile_payload["target_gpu_selection_strategy"] == "idle"
        assert execute_payload["required_resource"] == "gpu"
        assert execute_payload["node_affinity"] == "required"
        assert execute_payload["target_node_id"] == "node-b"
        assert execute_payload["artifact_node_id"] == "node-b"
        assert execute_payload["compile_artifact"]["artifact_node_id"] == "node-b"
        assert execute_payload["compile_artifact"]["target_gpu_worker_id"] == "node-b_gpu_3"
        assert execute_payload["compile_artifact"]["target_gpu_selection_strategy"] == "idle"

    asyncio.run(scenario())


def test_split_compile_execute_honors_requested_startup_node() -> None:
    async def scenario() -> None:
        task = EvaluationTask(
            task_id="startup",
            reference_code="class Model: pass",
            kernel_code="class ModelNew: pass",
            backend="cuda_agent",
            split_compile_and_execute=True,
            target_node_id="node-b",
            target_hostname="host-b",
        )
        _ref_task, kernel_task = _create_paired_tasks(task)
        scheduler = FakeScheduler()

        await KernelBenchWorkflowController()._run_split_kernel_task(task, kernel_task, scheduler)  # noqa: SLF001

        assert scheduler.selection_kwargs == {
            "timeout": 0,
            "poll_interval": 0.5,
            "target_node_id": "node-b",
            "target_hostname": "host-b",
        }
        assert scheduler.submissions[0]["target_node_id"] == "node-b"
        assert scheduler.submissions[0]["target_hostname"] == "host-b"

    asyncio.run(scenario())


def test_requested_affinity_fails_closed_without_scheduler_selector() -> None:
    async def scenario() -> None:
        result = await KernelBenchWorkflowController()._select_split_target_gpu(  # noqa: SLF001
            object(),
            "startup",
            target_hostname="host-b",
        )
        assert result is None

    asyncio.run(scenario())


class BusyFallbackScheduler(FakeScheduler):
    async def select_idle_worker(self, resource: str, **_kwargs) -> None:
        assert resource == "gpu"
        return None

    async def select_worker_by_task_id(self, resource: str, task_id: str, **_kwargs) -> dict[str, str]:
        assert resource == "gpu"
        assert task_id == "parent"
        return {
            "worker_id": "node-c_gpu_1",
            "device": "cuda:1",
            "node_id": "node-c",
            "hostname": "host-c",
        }

    async def wait(self, task_id: str, timeout=None) -> dict[str, Any]:
        result = await super().wait(task_id, timeout)
        if task_id.endswith("_compile"):
            result = dict(result)
            metadata = dict(result["metadata"])
            metadata["compile_node_id"] = "node-c"
            metadata["compile_hostname"] = "host-c"
            result["metadata"] = metadata
        return result


def test_split_compile_execute_falls_back_to_task_id_mod_when_all_gpus_busy() -> None:
    async def scenario() -> None:
        task = EvaluationTask(
            task_id="parent",
            reference_code="class Model: pass",
            kernel_code="class ModelNew: pass",
            backend="cuda_agent",
            split_compile_and_execute=True,
            enable_compile_artifact_cache=True,
        )
        _ref_task, kernel_task = _create_paired_tasks(task)
        scheduler = BusyFallbackScheduler()

        result = await KernelBenchWorkflowController()._run_split_kernel_task(task, kernel_task, scheduler)  # noqa: SLF001

        assert result["task_id"] == "parent_kernel"
        compile_payload, execute_payload = scheduler.submissions
        assert compile_payload["target_node_id"] == "node-c"
        assert compile_payload["target_gpu_worker_id"] == "node-c_gpu_1"
        assert compile_payload["target_gpu_selection_strategy"] == "task_id_mod"
        assert execute_payload["target_node_id"] == "node-c"
        assert execute_payload["compile_artifact"]["target_gpu_selection_strategy"] == "task_id_mod"

    asyncio.run(scenario())
