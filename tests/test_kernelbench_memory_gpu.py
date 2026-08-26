import pytest


def _require_cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is not available")
    return torch


@pytest.mark.gpu
def test_memory_trial_measures_matmul_forward_and_total_task_peak() -> None:
    torch = _require_cuda_runtime()
    from kernelgym.toolkit.kernelbench.memory import (
        capture_cuda_memory_environment_floor,
        measure_cuda_memory_trial,
    )

    device = torch.device("cuda:0")
    environment_floor = capture_cuda_memory_environment_floor(device)
    lhs = torch.randn(1024, 1024, device=device)
    rhs = torch.randn(1024, 1024, device=device)
    warmup_output = torch.matmul(lhs, rhs)
    torch.cuda.synchronize(device=device)
    del warmup_output

    expected_input_bytes = 2 * 1024 * 1024 * lhs.element_size()
    expected_output_bytes = 1024 * 1024 * lhs.element_size()

    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device=device).clone()
    stats = measure_cuda_memory_trial(
        torch.matmul,
        lhs,
        rhs,
        device=device,
        environment_floor_allocated_bytes=environment_floor["allocated_bytes"],
        environment_floor_reserved_bytes=environment_floor["reserved_bytes"],
    )

    assert stats["schema_version"] == 2
    assert stats["measurement_valid"] is True
    assert stats["measurement_complete"] is True
    assert stats["environment_floor_available"] is True
    assert "peak_allocated_bytes" not in stats
    assert stats["forward_incremental_peak_allocated_bytes"] >= expected_output_bytes
    assert stats["persistent_allocated_bytes"] >= expected_input_bytes
    assert (
        stats["total_task_peak_allocated_bytes"]
        == stats["persistent_allocated_bytes"] + stats["forward_incremental_peak_allocated_bytes"]
    )
    assert stats["recommended_comparison_metric"] == "total_task_peak_allocated_bytes"
    assert stats["output_tensor_bytes"] == expected_output_bytes
    assert stats["post_cleanup_allocated_bytes"] <= stats["baseline_allocated_bytes"]
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(device=device), cuda_rng_before)


@pytest.mark.gpu
def test_candidate_memory_trial_runs_when_performance_is_disabled() -> None:
    torch = _require_cuda_runtime()
    from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref

    reference_source = """
import torch

class Model(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(512, 512), torch.randn(512, 512)]
"""
    candidate_source = """
import torch

class ModelNew(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)
"""

    result = eval_kernel_against_ref(
        reference_source,
        candidate_source,
        device=torch.device("cuda:0"),
        measure_performance=False,
        enable_profiling=False,
        enable_ncu=False,
        enable_triton_detection=False,
        detect_decoy_kernel=False,
        verbose=False,
    )

    assert result.compiled is True
    assert result.correctness is True
    assert result.runtime == -1.0
    assert result.memory["measurement_valid"] is True
    assert (
        result.memory["total_task_peak_allocated_bytes"]
        >= result.memory["forward_incremental_peak_allocated_bytes"]
    )
    assert (
        result.metadata["kernel_memory_allocator_check"][
            "direct_cuda_allocation_detected"
        ]
        is False
    )
    assert result.metadata["kg_kernel_memory_step_s"] > 0
