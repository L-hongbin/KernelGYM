"""CUDA memory measurement and allocator-scope diagnostics."""

from __future__ import annotations

import gc
import re
from time import perf_counter
from typing import Any, Dict, Iterable, Union

import torch

MEMORY_SCHEMA_VERSION = 2
_DIRECT_CUDA_ALLOCATION_APIS = (
    "cudaMalloc3D",
    "cudaMalloc3DArray",
    "cudaMallocArray",
    "cudaMalloc",
    "cudaMallocAsync",
    "cudaMallocManaged",
    "cudaMallocPitch",
    "cuMemAlloc",
    "cuMemAllocAsync",
    "cuMemAllocFromPoolAsync",
    "cuArray3DCreate",
    "cuArrayCreate",
    "cuMipmappedArrayCreate",
    "cuMemAllocManaged",
    "cuMemAllocPitch",
    "cuMemCreate",
)
_DIRECT_CUDA_ALLOCATION_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _DIRECT_CUDA_ALLOCATION_APIS), key=len, reverse=True)) + r")\s*\("
)


def _strip_cpp_comments(source: str) -> str:
    """Blank C/C++ comments while preserving line numbers and code strings.

    Candidate CUDA is often embedded in Python strings or Markdown fences, so
    stripping all string literals would also hide the code that is compiled.
    """

    def replace_block(match: re.Match[str]) -> str:
        value = match.group(0)
        return "".join("\n" if char == "\n" else " " for char in value)

    without_blocks = re.sub(r"/\*.*?\*/", replace_block, source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", lambda match: " " * len(match.group(0)), without_blocks)


def detect_direct_cuda_allocations(source: str) -> Dict[str, Any]:
    """Find APIs whose device allocations bypass PyTorch allocator statistics."""

    searchable = _strip_cpp_comments(source or "")
    matches = []
    for match in _DIRECT_CUDA_ALLOCATION_PATTERN.finditer(searchable):
        line_number = searchable.count("\n", 0, match.start()) + 1
        source_line = (source or "").splitlines()[line_number - 1].strip()
        matches.append(
            {
                "api": match.group(1),
                "line": line_number,
                "snippet": source_line[:240],
            }
        )

    apis = sorted({item["api"] for item in matches})
    warning = None
    if matches:
        warning = (
            "Direct CUDA allocation APIs were found. torch.cuda peak-memory counters only track allocations "
            "managed by PyTorch's CUDA caching allocator, so the reported memory can be an underestimate."
        )
    return {
        "rule_id": "DIRECT_CUDA_ALLOCATION_BYPASSES_TORCH_ALLOCATOR",
        "severity": "warning" if matches else "none",
        "measurement_impact": "may_underestimate" if matches else "none_detected",
        "direct_cuda_allocation_detected": bool(matches),
        "direct_cuda_allocation_apis": apis,
        "direct_cuda_allocation_matches": matches,
        "warning": warning,
    }


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def capture_cuda_memory_environment_floor(
    device: Union[torch.device, int],
) -> Dict[str, int]:
    """Capture allocator state before task-owned models and inputs are created."""

    gc.collect()
    torch.cuda.synchronize(device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device=device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device=device),
        "reserved_bytes": torch.cuda.memory_reserved(device=device),
    }


def measure_cuda_memory_trial(
    kernel_fn: callable,
    *args: Any,
    device: Union[torch.device, int],
    source: str = "",
    allocation_check: Dict[str, Any] | None = None,
    environment_floor_allocated_bytes: int | None = None,
    environment_floor_reserved_bytes: int | None = None,
) -> Dict[str, Any]:
    """Measure one post-warmup forward using PyTorch allocator peak counters.

    Inputs and persistent model state are established before the baseline. The
    forward metric is the peak allocated-byte increment caused by the forward,
    including returned outputs and PyTorch-managed temporary workspaces. When
    an evaluation-start allocator floor is supplied, the result also reports
    persistent task memory and the total task peak. This prevents model setup
    or post-warmup cached workspaces from disappearing from memory feedback.
    """

    if allocation_check is None:
        allocation_check = detect_direct_cuda_allocations(source)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device=device)
    output = None
    started = perf_counter()

    gc.collect()
    torch.cuda.synchronize(device=device)
    baseline_allocated = torch.cuda.memory_allocated(device=device)
    baseline_reserved = torch.cuda.memory_reserved(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)

    try:
        with torch.no_grad():
            output = kernel_fn(*args)
        torch.cuda.synchronize(device=device)
        current_allocated = torch.cuda.memory_allocated(device=device)
        peak_allocated = torch.cuda.max_memory_allocated(device=device)
        peak_reserved = torch.cuda.max_memory_reserved(device=device)
        output_bytes = sum(tensor.numel() * tensor.element_size() for tensor in _iter_tensors(output))
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device=device)

    environment_floor_available = isinstance(environment_floor_allocated_bytes, int)
    environment_floor_valid = (
        environment_floor_available and environment_floor_allocated_bytes <= baseline_allocated
    )
    forward_incremental_peak = max(0, peak_allocated - baseline_allocated)
    persistent_allocated = (
        baseline_allocated - environment_floor_allocated_bytes if environment_floor_valid else None
    )
    total_task_peak_allocated = (
        peak_allocated - environment_floor_allocated_bytes if environment_floor_valid else None
    )

    environment_reserved_available = isinstance(environment_floor_reserved_bytes, int)
    environment_reserved_valid = (
        environment_reserved_available and environment_floor_reserved_bytes <= baseline_reserved
    )
    persistent_reserved = (
        baseline_reserved - environment_floor_reserved_bytes if environment_reserved_valid else None
    )
    total_task_peak_reserved = (
        peak_reserved - environment_floor_reserved_bytes if environment_reserved_valid else None
    )

    warnings = [allocation_check["warning"]] if allocation_check["warning"] else []
    if environment_floor_available and not environment_floor_valid:
        warnings.append(
            "The evaluation-start allocated-memory floor exceeded the forward baseline; "
            "persistent and total-task allocated metrics are unavailable."
        )
    if environment_reserved_available and not environment_reserved_valid:
        warnings.append(
            "The evaluation-start reserved-memory floor exceeded the forward baseline; "
            "persistent and total-task reserved metrics are unavailable."
        )

    result = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "method": "torch_cuda_peak_allocated_delta",
        "allocator_scope": "pytorch_cuda_caching_allocator",
        "forward_incremental_peak_allocated_bytes": forward_incremental_peak,
        "environment_floor_available": environment_floor_available,
        "environment_floor_allocated_bytes": environment_floor_allocated_bytes,
        "persistent_allocated_bytes": persistent_allocated,
        "total_task_peak_allocated_bytes": total_task_peak_allocated,
        "baseline_allocated_bytes": baseline_allocated,
        "current_allocated_bytes": current_allocated,
        "absolute_peak_allocated_bytes": peak_allocated,
        "environment_floor_reserved_bytes": environment_floor_reserved_bytes,
        "persistent_reserved_bytes": persistent_reserved,
        "total_task_peak_reserved_bytes": total_task_peak_reserved,
        "baseline_reserved_bytes": baseline_reserved,
        "absolute_peak_reserved_bytes": peak_reserved,
        "output_tensor_bytes": output_bytes,
        "measurement_valid": True,
        "measurement_complete": not allocation_check["direct_cuda_allocation_detected"],
        "measurement_is_lower_bound": allocation_check["direct_cuda_allocation_detected"],
        "direct_cuda_allocation_detected": allocation_check["direct_cuda_allocation_detected"],
        "direct_cuda_allocation_apis": allocation_check["direct_cuda_allocation_apis"],
        "direct_cuda_allocation_matches": allocation_check["direct_cuda_allocation_matches"],
        "recommended_comparison_metric": "total_task_peak_allocated_bytes",
        "warnings": warnings,
        "trial_wall_s": perf_counter() - started,
    }

    del output
    gc.collect()
    torch.cuda.synchronize(device=device)
    result["post_cleanup_allocated_bytes"] = torch.cuda.memory_allocated(device=device)
    return result
