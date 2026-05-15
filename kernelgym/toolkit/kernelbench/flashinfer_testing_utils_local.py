"""
Extracted timing helpers from flashinfer.testing.utils.

Original source is licensed under the Apache License, Version 2.0.
This local copy keeps the timing-related utilities needed by KernelGym
for side-by-side benchmarking and debugging.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch


def sleep_after_kernel_run(execution_time):
    """
    Sleep after kernel run. Dynamically adjust sleep time up to 1 sec based on execution time.

    Args:
        execution_time (float): Kernel execution time in milliseconds.

    Returns:
        None
    """
    if not math.isinf(execution_time):
        sleep_time = np.min([execution_time / 200, 1.0])
    else:
        sleep_time = 0.01
    time.sleep(sleep_time)
    return


def bench_gpu_time_with_cupti(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    use_cuda_graph: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
):
    """
    Benchmark GPU time using CUPTI activity tracing for precise kernel timing.

    CUPTI (CUDA Profiling Tools Interface) provides hardware-level profiling that
    measures actual GPU kernel execution time, excluding CPU-side launch overhead.
    This gives the most accurate kernel performance measurements.

    Cold L2 cache is achieved via L2 flush between iterations. CUPTI measures
    per-iteration, so L2 flush works correctly regardless of ``use_cuda_graph``.

    Behavior:
    - Uses CUPTI (requires version >= 13, i.e., CUDA 13+) to trace kernel activities
      and compute per-iteration GPU time from recorded start/end timestamps.
    - Optionally captures operations in a CUDA graph (use_cuda_graph=True) for
      reduced launch overhead during measurement.
    - If CUPTI is unavailable, falls back to:
      - ``bench_gpu_time_with_cudagraph`` if use_cuda_graph=True (uses rotating buffers
        for cold L2)
      - ``bench_gpu_time_with_cuda_event`` otherwise (uses L2 flush for cold L2)

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations.
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        sleep_after_run (bool): If True, sleep briefly after each iteration (default: False).
        use_cuda_graph (bool): If True, capture and replay a CUDA graph (default: False).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, flush L2 cache before each iteration to
            ensure cold-cache performance measurements (default: True).

    Returns:
        List[float]: Per-iteration GPU kernel execution times in milliseconds.

    Example:
        Basic CUPTI benchmarking (requires cupti-python >= 13):

        >>> def my_kernel(a, b):
        ...     return torch.matmul(a, b.T)
        >>> q = torch.randn(1024, 128, device="cuda")
        >>> k = torch.randn(1024, 128, device="cuda")
        >>> times = bench_gpu_time_with_cupti(
        ...     fn=my_kernel,
        ...     input_args=(q, k),
        ... )
        >>> print(f"Median GPU time: {np.median(times):.3f} ms")

    Note:
        Requires ``cupti-python`` package version >= 13.0.0:
        ``pip install -U cupti-python``

        If CUPTI is not available, a warning is issued and the function
        automatically falls back to CUDA event or CUDA graph timing.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        _do_l2_flush = l2_flush if l2_flush is not None else True
        _l2_flush_size_mb = l2_flush_size_mb if l2_flush_size_mb is not None else 256
        _l2_flush_device = l2_flush_device if l2_flush_device is not None else "cuda"
    else:
        _do_l2_flush = cold_l2_cache
        # Dynamically determine L2 flush size and device
        _l2_flush_device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
        l2_size = get_l2_cache_size(_l2_flush_device)
        # Use 2x L2 size to ensure complete flush
        _l2_flush_size_mb = (l2_size * 2) // (1024 * 1024)

    # check if CUPTI is installed and its version is >= 13.0.0
    from cupti import cupti
    from importlib.metadata import version as importlib_metadata_version

    cupti_version = importlib_metadata_version("cupti-python")
    if int(cupti_version.split(".")[0]) < 13:
        raise Exception(
            "CUPTI needs to be >= 13.0.0. Try 'pip install -U cupti-python'."
        )
    from functools import partial

    # CUPTI buffer callbacks
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024
        max_num_records = 0
        return buffer_size, max_num_records

    def set_kernel_name(activity):
        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            return activity.name
        elif activity.kind == cupti.ActivityKind.MEMCPY:
            return "MEMCPY"
        elif activity.kind == cupti.ActivityKind.MEMSET:
            return "MEMSET"

    def get_bytes(activity):
        if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
            return activity.bytes
        else:
            return 0

    def get_copy_kind(activity):
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return activity.copy_kind
        else:
            return 0

    def get_value(activity):
        if activity.kind == cupti.ActivityKind.MEMSET:
            return activity.value
        else:
            return 0

    def collect_kernel_info(activity):
        return (
            set_kernel_name(activity),
            activity.start,
            activity.end,
            activity.correlation_id,
            get_copy_kind(activity),
            get_bytes(activity),
            get_value(activity),
            activity.kind,
        )

    def func_buffer_completed(
        launches: list[tuple[float, float, int, int, int]],
        kernels: list[tuple[str, float, float, int, int, int, int, int]],
        activities: list,
    ):
        for activity in activities:
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                # Kernel activity
                kernels.append(collect_kernel_info(activity))
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                # Runtime or Driver activity
                launches.append(
                    (
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                        activity.cbid,
                        activity.kind,
                    )
                )

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    buffer = None
    if _do_l2_flush:
        l2_flush_size = int(_l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=_l2_flush_device, dtype=torch.int8)

    # Prepare runner (either direct fn or CUDA graph replay)
    runner = call_fn
    g = None
    if use_cuda_graph:
        # Warmup run to avoid capturing one-time inits
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call_fn()
        torch.cuda.current_stream().wait_stream(s)

        # Capture kernel in graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_fn()
        runner = g.replay

    ## Estimate kernel execution time by running the runner 5 times
    measurement_iters = 5
    torch.cuda.synchronize()
    call_fn()  # Call once to exclude initial overhead
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(measurement_iters):
        if _do_l2_flush:
            buffer.zero_()
        runner()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = (
        start_event.elapsed_time(end_event) / measurement_iters
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if _do_l2_flush:
            buffer.zero_()
        runner()
    torch.cuda.synchronize()

    # CUPTI measurement
    launches: list[tuple[float, float, int, int, int]] = []
    kernels: list[tuple[str, float, float, int, int, int, int, int]] = []
    iter_timestamps = []
    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.MEMSET)
    cupti.activity_register_callbacks(
        func_buffer_requested, partial(func_buffer_completed, launches, kernels)
    )
    for _ in range(repeat_iters):
        if _do_l2_flush:
            buffer.zero_()
        start_cpu = cupti.get_timestamp()
        runner()
        end_cpu = cupti.get_timestamp()
        torch.cuda.synchronize()
        iter_timestamps.append((start_cpu, end_cpu))
        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)
    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)
    cupti.activity_disable(cupti.ActivityKind.MEMSET)
    cupti.finalize()

    def generate_kernel_string(kernel):
        # No start, end, correlation_id is considered in the kernel string
        return f"{kernel[0]}_{kernel[4]}_{kernel[5]}_{kernel[6]}_{kernel[7]}"

    # Process activities - OPTIMIZED O(N + M log M) algorithm
    import bisect

    # Step 1: Sort launches by start timestamp - O(M log M)
    sorted_launches = sorted(launches, key=lambda l: l[0])
    launch_starts = [l[0] for l in sorted_launches]

    # Step 2: Build correlation_id -> kernels mapping - O(K)
    corr_id_to_kernels: dict[
        int, list[tuple[str, float, float, int, int, int, int, int]]
    ] = {}
    for k in kernels:
        corr_id = k[3]
        if corr_id not in corr_id_to_kernels:
            corr_id_to_kernels[corr_id] = []
        corr_id_to_kernels[corr_id].append(k)

    measured_times = []
    kernel_names = None
    for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
        # Use binary search to find launches within time range - O(log M)
        left_idx = bisect.bisect_left(launch_starts, start_cpu)
        right_idx = bisect.bisect_right(launch_starts, end_cpu)

        # Get correlation IDs for launches in range - O(range size)
        corr_ids = set(sorted_launches[i][2] for i in range(left_idx, right_idx))

        # Find all GPU kernels using the mapping - O(range size)
        iter_kernels = []
        for corr_id in corr_ids:
            if corr_id in corr_id_to_kernels:
                iter_kernels.extend(corr_id_to_kernels[corr_id])

        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for iteration {idx}")
        current_kernel_names = set(generate_kernel_string(k) for k in iter_kernels)
        # check if the kernel names are consistent
        if kernel_names is None:
            kernel_names = current_kernel_names
        else:
            if kernel_names != current_kernel_names:
                raise ValueError(
                    f"Inconsistent kernel names: {kernel_names} != {current_kernel_names}"
                )
        min_start = min(k[1] for k in iter_kernels)
        max_end = max(k[2] for k in iter_kernels)
        span_ms = (max_end - min_start) / 1e6  # ns to ms
        measured_times.append(span_ms)
    return measured_times


def bench_gpu_time_with_cudagraph(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    num_iters_within_graph: int = 10,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
):
    """
    Benchmark GPU time using CUDA graphs with amortized kernel launch overhead.

    CUDA graphs capture a sequence of GPU operations and replay them with minimal
    CPU overhead. By running multiple iterations within a single graph, kernel
    launch latency is amortized, yielding measurements closer to pure GPU time.

    **Cold-L2 Benchmarking**:

    When ``cold_l2_cache=True``, the function uses **rotating buffers** to ensure
    cold L2 cache for each kernel invocation within the graph. Multiple copies of
    the GPU tensors in ``input_args``/``input_kwargs`` are created and rotated
    through during graph capture, ensuring each kernel invocation operates on
    different memory regions. The number of buffer copies is automatically
    calculated based on the device's L2 cache size.

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations (graph replays).
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        num_iters_within_graph (int): Number of kernel calls captured in the graph
            (default: 10). Higher values better amortize launch overhead but use
            more memory when rotating buffers.
        sleep_after_run (bool): If True, sleep briefly after each iteration (default: False).
        input_args (tuple): Positional arguments to pass to fn. GPU tensors in
            this structure will be cloned when ``cold_l2_cache=True``.
        input_kwargs (dict, optional): Keyword arguments to pass to fn. GPU tensors
            in this structure will be cloned when ``cold_l2_cache=True``.
        cold_l2_cache (bool): If True, use rotating buffers to ensure cold L2 cache
            for each kernel invocation within the graph (default: True).

    Returns:
        List[float]: Per-iteration execution times in milliseconds. Each time is
        the graph replay duration divided by ``num_iters_within_graph``.

    Example:
        Cold-L2 benchmarking (default, for memory-bound kernels):

        >>> def run_attention(q, k, v, o):
        ...     flashinfer.single_prefill_with_kv_cache(q, k, v, o)
        ...
        >>> q = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> k = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> v = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> o = torch.empty_like(q)
        >>> times = bench_gpu_time_with_cudagraph(
        ...     fn=run_attention,
        ...     input_args=(q, k, v, o),
        ... )
        >>> print(f"Cold-L2 median time: {np.median(times):.3f} ms")

    Example:
        Hot L2 benchmarking (for compute-bound kernels):

        >>> times = bench_gpu_time_with_cudagraph(
        ...     fn=lambda: torch.matmul(q, k.T),
        ...     cold_l2_cache=False,
        ... )

    Note:
        - When using ``input_args``/``input_kwargs``, the function must accept the
          tensors as arguments (not capture them from closure).
        - GPU tensors are automatically detected and cloned. Non-tensor arguments
          (scalars, booleans, etc.) are preserved across all copies.
        - Memory usage scales with the number of rotations needed to exceed L2 cache.

    See Also:
        - ``calculate_rotation_count``: Computes required buffer copies for cold-L2.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead. For CUDA graphs, cold_l2_cache uses "
            "rotating buffers (not L2 flush) to ensure cold cache.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # For CUDA graphs, l2_flush had limited effectiveness, so we translate
        # l2_flush=True to cold_l2_cache=True (rotating buffers)
        _do_rotate = l2_flush if l2_flush is not None else True
    else:
        _do_rotate = cold_l2_cache

    # Dynamically determine device from input tensors
    _device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Determine rotation count if rotating buffers
    num_rotations = 1
    rotated_copies = None
    if _do_rotate:
        # Extract all GPU tensors from args and kwargs
        gpu_tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(
            input_kwargs
        )
        if len(gpu_tensors) == 0:
            warnings.warn(
                "cold_l2_cache=True but no GPU tensors found in input_args/input_kwargs. "
                "Cold L2 benchmarking disabled.",
                category=UserWarning,
                stacklevel=2,
            )
            _do_rotate = False
        else:
            num_rotations = calculate_rotation_count(gpu_tensors, _device)
            if num_rotations > 1:
                rotated_copies = _create_rotated_buffer_copies(
                    input_args, input_kwargs, num_rotations
                )
            else:
                # No rotation needed (tensors exceed L2)
                _do_rotate = False

    # Define how to call fn
    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    def call_fn_with_rotation(buf_idx: int):
        args, kwargs = rotated_copies[buf_idx]
        fn(*args, **kwargs)

    # Warmup run
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call_fn()
    torch.cuda.current_stream().wait_stream(s)

    # Capture kernel in graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if _do_rotate and num_rotations > 1:
            # Capture with rotating buffers: use buffer[iter % num_rotations]
            for iter_idx in range(num_iters_within_graph):
                buf_idx = iter_idx % num_rotations
                call_fn_with_rotation(buf_idx)
        else:
            # Non-rotating capture (uses original args if provided)
            for _ in range(num_iters_within_graph):
                call_fn()
    torch.cuda.synchronize()

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    start_event.record()
    for _ in range(measurement_iters):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = (
        start_event.elapsed_time(end_event) / measurement_iters
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry run
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        g.replay()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        start_events[iter_idx].record()
        g.replay()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.cuda.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(
            start_events[iter_idx].elapsed_time(end_events[iter_idx])
            / num_iters_within_graph
        )
    return measured_times


def bench_gpu_time(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    enable_cupti: bool = False,
    use_cuda_graph: bool = False,
    num_iters_within_graph: int = 10,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
):
    """
    Unified GPU benchmarking interface with configurable timing backends.

    This is the recommended entry point for GPU kernel benchmarking. It provides
    a single interface that dispatches to the appropriate timing implementation
    based on the configuration flags.

    **Timing Backends** (in order of precedence):

    1. **CUPTI** (``enable_cupti=True``): Most accurate, measures pure GPU kernel
       time via hardware profiling. Requires cupti-python >= 13.
    2. **CUDA Graphs** (``use_cuda_graph=True``): Amortizes launch overhead by
       capturing and replaying multiple kernel calls. Good balance of accuracy
       and availability.
    3. **CUDA Events** (default): Simplest method, measures launch + execution.
       Available everywhere but includes CPU overhead.

    **Cold-L2 Strategy** (automatically selected based on timing backend):

    .. list-table::
       :header-rows: 1

       * - Timing Backend
         - Cold-L2 Strategy
         - How it Works
       * - CUPTI
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events (no CUDA Graphs)
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events + CUDA Graphs
         - Rotating Buffers
         - Clone GPU tensors in input_args/input_kwargs and rotate through them
        use_cuda_graph (bool): If True, use CUDA graph timing (default: False).
        num_iters_within_graph (int): Kernel calls per graph (CUDA graph mode only,
            default: 10).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, ensure cold L2 cache for each iteration
            (default: True). The strategy is automatically selected based on timing
            backend.

    Returns:
        List[float]: Per-iteration execution times in milliseconds.

    Example:
        Simple benchmarking with CUDA events (default):

        >>> times = bench_gpu_time(fn=lambda: my_kernel())
        >>> print(f"Median: {np.median(times):.3f} ms")

    Example:
        CUDA graph benchmarking for reduced launch overhead:

        >>> def run_kernel(x, y, out):
        ...     my_memory_bound_kernel(x, y, out)
        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     use_cuda_graph=True,
        ... )

    Example:
        CUPTI benchmarking for most accurate GPU kernel time:

        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     enable_cupti=True,
        ... )

    See Also:
        - ``bench_gpu_time_with_cuda_event``: Direct CUDA event timing.
        - ``bench_gpu_time_with_cudagraph``: Direct CUDA graph timing.
        - ``bench_gpu_time_with_cupti``: Direct CUPTI timing.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device``
        parameters are deprecated. Use ``cold_l2_cache`` instead.
    """
    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # If l2_flush was explicitly set, use it as the cold_l2_cache value
        _cold_l2_cache = l2_flush if l2_flush is not None else cold_l2_cache
    else:
        _cold_l2_cache = cold_l2_cache

    if enable_cupti:
        return bench_gpu_time_with_cupti(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            sleep_after_run=sleep_after_run,
            use_cuda_graph=use_cuda_graph,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=_cold_l2_cache,
        )
    if use_cuda_graph:
        return bench_gpu_time_with_cudagraph(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            num_iters_within_graph=num_iters_within_graph,
            sleep_after_run=sleep_after_run,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=_cold_l2_cache,
        )
    return bench_gpu_time_with_cuda_event(
        fn=fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        sleep_after_run=sleep_after_run,
        input_args=input_args,
        input_kwargs=input_kwargs,
        cold_l2_cache=_cold_l2_cache,
    )


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# copied from DeepGEMM
def bench_kineto(
    fn,
    kernel_names,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str = None,
    flush_l2: bool = True,
    with_multiple_kernels: bool = False,
):
    # Conflict with Nsight Systems
    using_nsys = int(os.environ.get("DG_NSYS_PROFILING", 0))

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = (
        suppress_stdout_stderr
        if suppress_kineto_output and not using_nsys
        else empty_suppress
    )
    with suppress():
        schedule = (
            torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            if not using_nsys
            else None
        )
        profiler: Any = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
            )
            if not using_nsys
            else empty_suppress()
        )
        with profiler:
            for _i in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(
                            flush_l2_size, dtype=torch.int, device="cuda"
                        ).zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) == 1, (
                f"Errors of the kernel {name} in the profiling table"
            )

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0.0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += (
                            float(time_str.replace(unit, "")) / scale * int(num_str)
                        )
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total
