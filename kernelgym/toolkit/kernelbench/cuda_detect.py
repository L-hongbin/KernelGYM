from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from kernelgym.toolkit.kernelbench.profiling import (
    extract_profiling_metrics,
    is_framework_cuda_event_name,
    profiling_context,
)


def _call_inference(run: Callable, *args, **kwargs):
    with torch.inference_mode():
        return run(*args, **kwargs)


def _kernel_name_matches(expected_name: str, observed_name: str) -> bool:
    observed_clean = observed_name.split("(")[0].strip()
    observed_clean = observed_clean.split("<")[0].split("::")[-1]
    expected = expected_name.strip()
    return expected == observed_clean or expected in observed_clean


def _extract_matches(
    profiled_kernel_names: List[str],
    expected_kernel_names: Optional[List[str]],
) -> List[str]:
    expected = expected_kernel_names or []
    if expected:
        return [
            name
            for name in expected
            if any(_kernel_name_matches(name, captured) for captured in profiled_kernel_names)
        ]
    return [
        name
        for name in profiled_kernel_names
        if name and not is_framework_cuda_event_name(name)
    ]


def detect_cuda_usage_for_module(
    run: Callable,
    *args: Any,
    expected_kernel_names: Optional[List[str]] = None,
    warmup: int = 1,
    steps: int = 1,
    use_cuda: bool = True,
    return_matches: bool = False,
) -> Tuple[bool, List[str]] | bool:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(max(0, warmup)):
        _call_inference(run, *args)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    with profiling_context(enabled=True) as prof:
        for _ in range(max(1, steps)):
            _call_inference(run, *args)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

    profiling_metrics: Dict[str, Any] = extract_profiling_metrics(prof)
    profiled_kernel_names = [
        kernel.get("name", "")
        for kernel in profiling_metrics.get("kernels", [])
        if isinstance(kernel, dict)
    ]
    matches = _extract_matches(profiled_kernel_names, expected_kernel_names)
    cuda_launch_api_calls = int(profiling_metrics.get("cuda_launch_api_calls", 0) or 0)
    used = bool(matches) or (cuda_launch_api_calls > 0 and bool(expected_kernel_names))

    if return_matches:
        return used, matches
    return used
