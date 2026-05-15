"""KernelBench evaluation pipeline (task-level, toolkit layer)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union

import torch

from kernelgym.config import settings
from kernelgym.worker.compile_stage import compile_kernel_artifact
from kernelgym.toolkit.kernelbench import cuda_detect
from kernelgym.toolkit.kernelbench import triton_detect as detect
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult, get_error_name, set_seed
from kernelgym.toolkit.kernelbench.loading import (
    graceful_eval_cleanup,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
)
from kernelgym.toolkit.kernelbench.correctness import run_and_check_correctness
from kernelgym.toolkit.kernelbench.profiling import (
    compute_cuda_kernel_coverage,
    compute_triton_kernel_coverage,
)
from kernelgym.toolkit.kernelbench.timing import (
    get_timing_stats,
    run_profiling_only,
    time_execution_with_cuda_event,
)
from kernelgym.utils.device_info import get_cuda_device_info


def _extract_expected_cuda_kernel_names(custom_model_src: str) -> List[str]:
    if not custom_model_src:
        return []

    patterns = [
        r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(",
        r"__global__\s+[\w:<>]+\s+([A-Za-z_]\w*)\s*\(",
    ]

    names = set()
    for pattern in patterns:
        for match in re.findall(pattern, custom_model_src):
            if match:
                names.add(match)
    return sorted(names)


def _kernel_name_matches(expected_name: str, observed_name: str) -> bool:
    """
    智能匹配：忽略 C++ 参数签名
    例如：'my_kernel' 匹配 'my_kernel(float const*, float*, int)'
    """
    # 1. 清洗捕获到的名字：去掉括号及参数
    observed_clean = observed_name.split('(')[0].strip()
    
    # 2. 处理可能存在的模板符号 <...> 和命名空间 ::
    observed_clean = observed_clean.split('<')[0].split('::')[-1]
    
    expected = expected_name.strip()
    
    # 3. 进行匹配
    return expected == observed_clean or expected in observed_clean


def _is_retryable_lock_file_error(error_text: object) -> bool:
    text = str(error_text or "").lower()
    lock_markers = (
        "filebaton",
        "baton",
        "lock file",
        "lockfile",
        "file lock",
        "cache lock",
        "waiting for lock",
        "timeout waiting for lock",
        "timed out waiting for lock",
        "failed to acquire lock",
        "could not acquire lock",
    )
    if any(marker in text for marker in lock_markers):
        return True

    # Ordinary missing headers/includes are compile failures, not retryable lock errors.
    return "no such file or directory" in text and ".lock" in text


def _run_correctness_step(
    original_model,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    num_correct_trials: int,
    verbose: bool,
    seed_num: int,
    device: Union[torch.device, int],
) -> KernelExecResult:
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        return run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def _run_triton_detection_step(
    *,
    enable_triton_detection: bool,
    is_triton: bool,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    seed_num: int,
    device: Union[torch.device, int],
    verbose: bool,
    backend: str,
):
    if not enable_triton_detection:
        return False
    try:
        print("Begin Triton usage detection")
        if kernel_exec_result and kernel_exec_result.correctness:
            torch.cuda.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            model_new = custom_model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            used, matches = detect.detect_triton_usage_for_module(
                model_new,
                *inputs,
                warmup=1,
                steps=1,
                use_cuda=True,
                return_matches=True,
            )
            metadata["is_kernel_used"] = used
            metadata["profiler_matches"] = matches
            print(f"Triton usage detection result: {used}")
            print(f"Triton usage detection matches: {matches}")
            if not used and is_triton:
                print(
                    "[Eval] Backend is 'triton' but no Triton usage detected, marking as decoy"
                )
                kernel_exec_result.decoy_kernel = True
                kernel_exec_result.runtime = -1.0
                return True
                if not used:
                    print(
                        f"[Eval] No Triton usage detected, but backend is '{backend}', continuing to performance measurement"
                    )
    except Exception as e:
        if verbose:
            print(f"[Eval] Error in Triton usage detection: {e}")
        metadata["error_in_detection"] = e
    return False


def _run_cuda_detection_step(
    *,
    enable_cuda_detection: bool,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    seed_num: int,
    device: Union[torch.device, int],
    verbose: bool,
    expected_cuda_kernel_names: Optional[List[str]] = None,
):
    if not enable_cuda_detection:
        return False
    try:
        print("Begin CUDA usage detection")
        if kernel_exec_result and kernel_exec_result.correctness:
            torch.cuda.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            model_new = custom_model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            used, matches = cuda_detect.detect_cuda_usage_for_module(
                model_new,
                *inputs,
                expected_kernel_names=expected_cuda_kernel_names or [],
                warmup=1,
                steps=1,
                use_cuda=True,
                return_matches=True,
            )
            metadata["is_kernel_used"] = used
            metadata["profiler_matches"] = matches
            print(f"CUDA usage detection result: {used}")
            print(f"CUDA usage detection matches: {matches}")
            if not used:
                print(
                    "[Eval] CUDA detect did not observe a custom kernel in the pre-check; continuing to profiling-based validation"
                )
    except Exception as e:
        if verbose:
            print(f"[Eval] Error in CUDA usage detection: {e}")
        metadata["error_in_detection"] = e
    return False


def _run_performance_step(
    *,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    num_perf_trials: int,
    verbose: bool,
    seed_num: int,
    device: Union[torch.device, int],
    enable_profiling: bool,
    backend: str = "cuda",
    expected_cuda_kernel_names: Optional[List[str]] = None,
):
    def _profiling_empty(metrics: Dict[str, Any]) -> bool:
        if not metrics:
            return True
        if "kernels" not in metrics:
            return True
        if len(metrics.get("kernels", [])) == 0:
            return True
        return False

    try:
        if kernel_exec_result and kernel_exec_result.correctness:
            if verbose:
                print("[Eval] Measuring Performance as Sample is Correct")

            torch.cuda.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            model_new = custom_model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            elapsed_times, profiling_metrics = time_execution_with_cuda_event(
                model_new,
                *inputs,
                num_trials=num_perf_trials,
                verbose=verbose,
                device=device,
                enable_profiling=enable_profiling,
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if enable_profiling and _profiling_empty(profiling_metrics):
                retry_count = max(0, int(getattr(settings, "profiling_retry_count", 0)))
                for attempt in range(retry_count):
                    print(
                        f"[WARNING] Profiler returned empty results. Retrying ({attempt + 1}/{retry_count})..."
                    )
                    retry_metrics = run_profiling_only(
                        model_new,
                        *inputs,
                        num_trials=max(1, min(num_perf_trials, 10)),
                        verbose=verbose,
                        device=device,
                    )
                    if not _profiling_empty(retry_metrics):
                        profiling_metrics = retry_metrics
                        break
                    profiling_metrics = retry_metrics

            if enable_profiling:
                print(
                    f"[DEBUG] profiling_metrics type: {type(profiling_metrics)}, empty: {not profiling_metrics}"
                )
                if profiling_metrics.get("profiling_warning"):
                    print(
                        f"[WARNING] Profiling warning: {profiling_metrics['profiling_warning']}"
                    )

                if _profiling_empty(profiling_metrics):
                    print("[WARNING] Profiler returned empty results!")
                    print(
                        "[WARNING] This may be a profiler bug, not a decoy kernel issue."
                    )
                    print(
                        f"[WARNING] Kernel detected: {metadata.get('is_kernel_used', False)}"
                    )
                    print(
                        f"[WARNING] Profiler matches: {len(metadata.get('profiler_matches', []))}"
                    )
                    if metadata.get("is_kernel_used", False):
                        print(
                            "[INFO] Skipping decoy detection due to profiler failure (Triton hook passed)"
                        )

            if profiling_metrics and len(profiling_metrics) > 0:
                metadata["profiling"] = profiling_metrics
                if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
                    kernel_exec_result.metadata["profiling"] = profiling_metrics

                print(
                    f"[DEBUG Profiling] profiling_metrics keys: {profiling_metrics.keys()}"
                )
                print(
                    f"[DEBUG Profiling] kernel_count: {profiling_metrics.get('kernel_count', 'N/A')}"
                )
                print(
                    f"[DEBUG Profiling] profiler_matches: {metadata.get('profiler_matches', [])}"
                )
                is_cuda = (backend == "cuda") or (backend == "cuda_agent")
                try: 
                    if is_cuda:
                        coverage_result_dict = compute_cuda_kernel_coverage(profiling_metrics)
                    else:
                        triton_matches = metadata.get("profiler_matches", [])
                        coverage_result_dict = compute_triton_kernel_coverage(
                            triton_matches, profiling_metrics
                        )
                except Exception as coverage_error:
                    print(
                        f"[ERROR] compute_triton_kernel_coverage failed: {coverage_error}"
                    )
                    import traceback

                    traceback.print_exc()
                    coverage_result_dict = {
                        "num_custom_kernels": 0,
                        "num_total_kernels": 0,
                        "triton_kernels_not_in_profiling": metadata.get(
                            "profiler_matches", []
                        ),
                        "triton_kernels_in_profiling": [],
                        "total_kernel_run_time_in_profiling_us": 0,
                        "custom_kernel_cuda_time_in_profiling_us": 0,
                    }
                print(
                    f"[DEBUG Coverage] num_custom_kernels: {coverage_result_dict['num_custom_kernels']}"
                )
                print(
                    f"[DEBUG Coverage] num_total_kernels: {coverage_result_dict['num_total_kernels']}"
                )
                num_custom_kernels = coverage_result_dict["num_custom_kernels"]
                num_total_kernels = coverage_result_dict["num_total_kernels"]
                triton_kernels_not_in_profiling = coverage_result_dict[
                    "triton_kernels_not_in_profiling"
                ]
                triton_kernels_in_profiling = coverage_result_dict[
                    "triton_kernels_in_profiling"
                ]
                total_kernel_run_time_in_profiling_us = coverage_result_dict[
                    "total_kernel_run_time_in_profiling_us"
                ]
                custom_kernel_time_in_profiling_us = coverage_result_dict[
                    "custom_kernel_cuda_time_in_profiling_us"
                ]

                metadata["num_custom_kernels"] = num_custom_kernels
                metadata["num_total_kernels"] = num_total_kernels
                ratio = num_custom_kernels / num_total_kernels if num_total_kernels > 0 else 0
                coverage_info = f"Run {num_custom_kernels} custom kernels / Total {num_total_kernels} kernels, Coverage: {ratio:.2%}"
                metadata["kernel_coverage"] = coverage_info
                metadata["kernel_not_in_profiling"] = triton_kernels_not_in_profiling
                metadata["kernel_in_profiling"] = triton_kernels_in_profiling

                metadata[
                    "total_kernel_run_time_in_profiling_us"
                ] = total_kernel_run_time_in_profiling_us
                metadata[
                    "custom_kernel_time_in_profiling_us"
                ] = custom_kernel_time_in_profiling_us
                ratio_time = (
                    custom_kernel_time_in_profiling_us
                    / total_kernel_run_time_in_profiling_us
                    if total_kernel_run_time_in_profiling_us > 0
                    else 0
                )
                metadata[
                    "custom_kernel_time_coverage"
                ] = (
                    f"Custom kernel time: {custom_kernel_time_in_profiling_us:.2f}us / Total time: {total_kernel_run_time_in_profiling_us:.2f}us, Coverage: {ratio_time:.2%}"
                )

                precheck_kernel_used = bool(metadata.get("is_kernel_used"))
                expected_kernel_names = expected_cuda_kernel_names or []
                profiled_kernel_names = [
                    k.get("name", "")
                    for k in profiling_metrics.get("kernels", [])
                    if isinstance(k, dict)
                ]
                matched_expected_kernels = [
                    name
                    for name in expected_kernel_names
                    if any(_kernel_name_matches(name, captured) for captured in profiled_kernel_names)
                ]
                cuda_launch_api_calls = int(profiling_metrics.get("cuda_launch_api_calls", 0) or 0)
                launch_only_inference = (
                    num_total_kernels == 0
                    and cuda_launch_api_calls > 0
                    and len(expected_kernel_names) > 0
                    and bool(kernel_exec_result.correctness)
                )
                profile_kernel_used = (
                    len(matched_expected_kernels) > 0
                    or num_custom_kernels > 0
                    or launch_only_inference
                )
                is_kernel_used = precheck_kernel_used or profile_kernel_used

                metadata["expected_kernel_names"] = expected_kernel_names
                metadata["profiled_kernel_names"] = profiled_kernel_names
                metadata["matched_expected_kernel_names"] = matched_expected_kernels
                metadata["launch_api_calls"] = cuda_launch_api_calls
                metadata["is_kernel_used"] = is_kernel_used
                metadata["custom_kernel_usage_inferred_from_launch"] = launch_only_inference
                if profile_kernel_used and not precheck_kernel_used:
                    # The final kernel-used decision came from performance profiling,
                    # so do not expose stale pre-check matches.
                    metadata.pop("profiler_matches", None)

                if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
                    kernel_exec_result.metadata.update(metadata)

                # For CUDA backend (non-triton), we don't use triton-based decoy detection
                # Instead, we check if any CUDA kernels were captured in profiling
                is_cuda_agent = backend == "cuda_agent"

                if is_cuda_agent:
                    is_pytorch_decoy = coverage_result_dict.get("is_pytorch_decoy", False)
                    # For CUDA agent backend, if we captured any custom kernels, accept as valid.
                    # (Profiling may still include unavoidable ATen launches like allocations.)
                    if num_custom_kernels > 0:
                        print(
                            f"[INFO] CUDA backend: Profiler captured {num_custom_kernels} custom kernels - accepting as valid"
                        )
                        kernel_exec_result.decoy_kernel = False
                    elif is_kernel_used:
                        print(
                            f"[INFO] CUDA backend: Matched expected custom kernels in profiler: {matched_expected_kernels}"
                        )
                        kernel_exec_result.decoy_kernel = False
                    elif launch_only_inference:
                        print(
                            "[INFO] CUDA backend: kernel launch observed but profiler has 0 CUDA kernels; treating as profiler failure fallback"
                        )
                        kernel_exec_result.decoy_kernel = False
                    elif is_pytorch_decoy:
                        print(
                            "[WARNING] CUDA backend: High PyTorch native op usage detected and no custom kernels captured - marking as decoy"
                        )
                        kernel_exec_result.decoy_kernel = True
                    else:
                        print(
                            f"[WARNING] CUDA backend: Profiler captured 0 custom kernels (out of {num_total_kernels} total) - marking as decoy"
                        )
                        kernel_exec_result.decoy_kernel = True
                elif num_custom_kernels == 0 and num_total_kernels > 0:
                    print(
                        f"[WARNING] Profiler captured {num_total_kernels} kernels but 0 custom kernels - marking as decoy"
                    )
                    kernel_exec_result.decoy_kernel = True
                elif num_custom_kernels == 0 and num_total_kernels == 0:
                    print(
                        "[WARNING] Profiler captured 0 total kernels - likely profiler bug, NOT marking as decoy"
                    )
                    print(
                        f"[INFO] Relying on kernel detection instead (detected: {metadata.get('is_kernel_used', False)})"
                    )
            if verbose:
                print(f"[Eval] Performance Stats: {runtime_stats}")
            kernel_exec_result.runtime = runtime_stats["mean"]
            kernel_exec_result.runtime_stats = runtime_stats
    except Exception as e:
        if verbose:
            print(f"[Eval] Error in Measuring Performance: {e}")
        kernel_exec_result.metadata["error_during_performance"] = e

def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = True,
    measure_performance: bool = True,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),
    backend: str = "cuda",
    entry_point: str = "Model",
    enable_profiling: bool = True,
    enable_detection: bool = True,
    enable_triton_detection: Optional[bool] = None,
    backend_adapter: Optional[Any] = None,
    precompiled_artifact: Optional[Dict[str, Any]] = None,
    enable_compile_artifact_cache: bool = False,
) -> KernelExecResult:
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    torch.cuda.set_device(device)
    is_triton = backend == "triton"
    metadata: Dict[str, Any] = {}
    metadata["device_info"] = get_cuda_device_info(device)
    metadata["device"] = str(device)
    expected_cuda_kernel_names = _extract_expected_cuda_kernel_names(custom_model_src)
    metadata["expected_kernel_names"] = expected_cuda_kernel_names
    metadata["is_kernel_used"] = False
    metadata["kernel_coverage"] = ""

    if is_triton:
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert device.type == "cuda", "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(f"device must be an int or torch.device, got {type(device)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context, entry_point
    )
    set_seed(seed_num)
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    # print(f"[DEBUG] init inputs: {init_inputs}")

    if (
        len(init_inputs) > 1
        and hasattr(init_inputs[0], "__len__")
        and not isinstance(init_inputs[0], (str, torch.Tensor))
        and len(init_inputs[0]) == 0
    ):
        init_inputs = init_inputs[1]

    with torch.no_grad():
        set_seed(seed_num)

        if type(init_inputs) == list:
            original_model = Model(*init_inputs)
        else:
            original_model = Model(**init_inputs)

        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    tempfile_handle = None
    backend_handle = None
    backend_session = None

    def _cleanup():
        if backend_session is not None:
            backend_session.close()
            return
        if backend_adapter is not None and backend_handle is not None:
            backend_adapter.cleanup(backend_handle)
            return
        graceful_eval_cleanup(context, device, tempfile_handle)

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        if backend_adapter is not None:
            if precompiled_artifact is not None:
                artifact = dict(precompiled_artifact)
                artifact.setdefault("compiled", True)
                artifact.setdefault("code", custom_model_src)
                artifact.setdefault("entry_point", f"{entry_point}New")
                artifact.setdefault("backend", backend)
                artifact.setdefault("device", str(device))
                metadata["precompiled_artifact_used"] = True
            else:
                artifact, cache_hit = compile_kernel_artifact(
                    {
                        "backend_adapter": backend_adapter.name,
                        "backend": backend,
                        "entry_point": entry_point,
                        "kernel_code": custom_model_src,
                        "device": str(device),
                        "build_dir": build_dir,
                        "enable_compile_artifact_cache": enable_compile_artifact_cache,
                    },
                    worker_id=f"pipeline_{os.getpid()}",
                )
                metadata["compile_artifact_cache_enabled"] = (
                    enable_compile_artifact_cache
                )
                metadata["compile_artifact_cache_hit"] = cache_hit
                if not artifact.get("compiled"):
                    error = artifact.get("error", "Unknown compile error")
                    if _is_retryable_lock_file_error(error):
                        print(
                            f"[Eval] Lock file error during compilation, Please retry. Error: {error}"
                        )
                        _cleanup()
                        return None
                    if "precheck" in artifact and artifact["precheck"].get("passed", False):
                        metadata["compilation_error_name"] = "precheck_error"
                    else:
                        metadata["compilation_error_name"] = "compile_error"
                    metadata["compilation_error"] = error
                    print(f"[Info] metadata\n{metadata}")
                    _cleanup()
                    return KernelExecResult(compiled=False, metadata=metadata)

            backend_handle = backend_adapter.load(
                artifact,
                device=device,
                context=context,
                build_dir=build_dir,
            )
            backend_session = backend_adapter.open_session(backend_handle, device=device)
            tempfile_handle = backend_handle.get("tempfile_handle")
        else:
            if is_triton:
                ModelNew, tempfile_handle = load_custom_model_with_tempfile(
                    custom_model_src, entry_point=f"{entry_point}New"
                )
                if verbose:
                    print("[Eval] Model with Triton Loaded")
            else:
                ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )

        if _is_retryable_lock_file_error(e):
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            _cleanup()
            return None
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = e
        _cleanup()
        return KernelExecResult(compiled=False, metadata=metadata)

    try:
        def _create_custom_model():
            if backend_session is not None:
                return backend_session.create_model(
                    init_inputs,
                    no_grad=True,
                    synchronize=False,
                )
            if type(init_inputs) == list:
                return ModelNew(*init_inputs)
            return ModelNew(**init_inputs)

        with torch.no_grad():
            set_seed(seed_num)
            custom_model = _create_custom_model()

            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            "Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \n"
            f"Error: {e}"
        )
        _cleanup()
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    kernel_exec_result = None
    decoy_detected = False

    kernel_exec_result = _run_correctness_step(
        original_model,
        custom_model,
        get_inputs,
        metadata,
        num_correct_trials,
        verbose,
        seed_num,
        device,
    )
    if is_triton:
        if enable_triton_detection is not None:
            enable_detection = enable_triton_detection
        decoy_detected = _run_triton_detection_step(
            enable_triton_detection=enable_detection,
            is_triton=is_triton,
            kernel_exec_result=kernel_exec_result,
            custom_model=custom_model,
            get_inputs=get_inputs,
            metadata=metadata,
            seed_num=seed_num,
            device=device,
            verbose=verbose,
            backend=backend,
        )
    elif backend == "cuda_agent":
        decoy_detected = _run_cuda_detection_step(
            enable_cuda_detection=enable_detection,
            kernel_exec_result=kernel_exec_result,
            custom_model=custom_model,
            get_inputs=get_inputs,
            metadata=metadata,
            seed_num=seed_num,
            device=device,
            verbose=verbose,
            expected_cuda_kernel_names=expected_cuda_kernel_names,
        )
    
    if decoy_detected:
        _cleanup()
        return kernel_exec_result

    if measure_performance:
        _run_performance_step(
            kernel_exec_result=kernel_exec_result,
            custom_model=custom_model,
            get_inputs=get_inputs,
            metadata=metadata,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            seed_num=seed_num,
            device=device,
            enable_profiling=enable_profiling,
            backend=backend,
            expected_cuda_kernel_names=expected_cuda_kernel_names,
        )

    if not metadata.get("profiler_matches"):
        metadata.pop("profiler_matches", None)

    _cleanup()
    return kernel_exec_result




def eval_reference_only(
    original_model_src: str,
    seed_num: int = 42,
    num_perf_trials: int = 10,
    verbose: bool = False,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),
    entry_point: str = "Model",
    reference_backend: Optional[str] = None,
    backend_adapter: Optional[Any] = None,
) -> KernelExecResult:
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    torch.cuda.set_device(device)
    metadata: Dict[str, Any] = {}
    metadata["device_info"] = get_cuda_device_info(device)
    metadata["device"] = str(device)

    context: Dict[str, Any] = {}

    if verbose:
        print(f"[Eval] Start Evaluation! on device: {device}")
        print("[Eval] Loading Original Model")

    try:
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
            original_model_src, context, entry_point
        )
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]

        with torch.no_grad():
            set_seed(seed_num)
            if type(init_inputs) == list:
                original_model = Model(*init_inputs)
            else:
                original_model = Model(**init_inputs)
            assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")

    except Exception as e:
        print(f"Failed to load original model: {e}")
        metadata["model_load_error"] = e
        metadata["model_load_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    kernel_exec_result = KernelExecResult(compiled=True, correctness=True, metadata=metadata)

    try:
        if verbose:
            print("[Eval] Measuring Performance of Original Model")

        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        model = original_model.cuda(device=device)
        if reference_backend:
            backend_name = reference_backend.lower()
            metadata["reference_backend"] = backend_name
            print(f"[Eval] reference_backend={backend_name}")
            if backend_name in ("torch_compile", "torch-compile", "compile"):
                try:
                    if not hasattr(torch, "compile"):
                        raise RuntimeError("torch.compile is not available")
                    model = torch.compile(model)
                    metadata["reference_backend_compiled"] = True
                    print("[Eval] torch.compile succeeded")
                except Exception as e:
                    metadata["reference_backend_error"] = str(e)
                    print(f"[Eval] torch.compile failed: {e}")
                    return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
        torch.cuda.synchronize(device=device)

        elapsed_times, _ = time_execution_with_cuda_event(
            model,
            *inputs,
            num_trials=num_perf_trials,
            verbose=verbose,
            device=device,
            enable_profiling=False,
        )
        runtime_stats = get_timing_stats(elapsed_times, device=device)

        if verbose:
            print(f"[Eval] Performance Stats: {runtime_stats}")
        kernel_exec_result.runtime = runtime_stats["mean"]
        kernel_exec_result.runtime_stats = runtime_stats
    except Exception as e:
        if verbose:
            print(f"[Eval] Error in Measuring Performance: {e}")
        kernel_exec_result.metadata["error_during_performance"] = e

    graceful_eval_cleanup(context, device, None)
    return kernel_exec_result
