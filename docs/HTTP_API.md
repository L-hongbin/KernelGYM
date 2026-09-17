# HTTP API Reference

Reference for the reward service's HTTP surface.

- Base URL: `http://<api-host>:<api-port>` (default `127.0.0.1:20111`, see [DEPLOYMENT.md](DEPLOYMENT.md))
- Auth: none (intended to run on a trusted internal network)
- Content type: `application/json` for all bodies
- Concrete schemas live in `kernelgym/server/api/models.py`; this doc is the human-readable view

For a quick end-to-end probe, run `bash test_reward.sh` (single CUDA-Agent add) or [`benchmarks/run_compile_benchmark.py`](../benchmarks/README.md) (parametrized over backend and scenario).

## Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service identity |
| GET | `/device-info` | Static capabilities detected from the ENV node's local CUDA device |
| GET | `/health` | Aggregated GPU + queue + memory health |
| GET | `/metrics` | Performance / resource / queue / error counters |
| POST | `/evaluate` | **Submit one kernel evaluation (primary endpoint)** |
| POST | `/benchmark/speed-test` | Run the fixed correct end-to-end speed test three times |
| POST | `/benchmark/speedup-noise-floor` | Calibrate block-level log-speedup noise with ten fixed TVM-FFI cases |
| POST | `/evaluate/batch` | Submit a batch of evaluations |
| POST | `/workflow/submit` | Submit any workflow with an arbitrary payload |
| POST | `/debug/validate` | Dry-run request validation (does not run) |
| GET | `/status/{task_id}` | Task lifecycle status |
| GET | `/results/{task_id}` | Final evaluation result for a task |
| GET | `/workflow/results/{task_id}` | Same, in `WorkflowResponse` shape |
| DELETE | `/tasks/{task_id}` | Cancel a task: drop it from the queue if pending, or interrupt the running CUDA subprocess if in-flight |
| GET | `/queue/status` | Queue depth per priority/resource |
| GET | `/workers/status` | Registered workers + load-balancer state |
| POST | `/worker/register` | Worker→server registration *(internal)* |
| POST | `/worker/unregister` | Worker→server deregistration *(internal)* |
| POST | `/worker/heartbeat` | Worker→server liveness *(internal)* |
| POST | `/worker/evict_from_lb` | Drop a worker from the LB without deleting Redis state *(internal)* |
| POST | `/node/allocate` | Allocate / look up a stable `node_id` for a hostname *(internal)* |
| GET | `/monitoring/problematic-codes` | Codes hitting error-rate threshold |
| GET | `/monitoring/retry-queue` | Pending retries with ETA |
| GET | `/monitoring/worker-health` | Workers + CUDA-error shutdown flags |
| POST | `/monitoring/clear-error-history/{code_hash}` | Reset error counters for one code hash |

Endpoints marked *(internal)* are used by the in-process worker subprocesses and the deploy scripts; RL clients don't need to call them.

## `GET /device-info` — local static GPU capabilities

Returns the same static `device_info` object attached to evaluation metadata, without submitting a Kernel task. The
ENV process detects these values locally at deployment time through PyTorch's CUDA device properties, with
the CUDA Runtime API used for the runtime version and `nvidia-smi` used for the driver version. Counts remain JSON integers, while capacities and bandwidth
use explicit human-readable units for direct model input. Unsupported properties are returned as `null`;
`peak_compute_tflops` and dynamic profiling counters are intentionally not included.

```bash
curl -sS http://127.0.0.1:20111/device-info
```

The response groups thread and launch-dimension limits, register limits, shared-memory limits, and the locally detected `software.cuda_version`,
`software.driver_version`, and `software.nvcc_version` values. `cuda_version` is queried with
`cudaRuntimeGetVersion()` rather than read from the driver compatibility ceiling or PyTorch build, and framework
build versions are not included. Device memory uses GiB,
shared memory uses KiB, L2 uses MiB, and theoretical DDR bandwidth uses TB/s or GB/s. Clock and memory-bus source
values are used locally to derive the bandwidth but are not exposed because they are redundant for model input.

## `POST /evaluate` — the main endpoint

Submits a single kernel evaluation and waits for the result. The server runs the request through the configured workflow (`kernelbench` by default), compiles, runs correctness + performance, and returns a single `EvaluationResponse`.

### Request body (`EvaluationRequest`)

Required:

| Field | Type | Notes |
|---|---|---|
| `task_id` | string (1–100 chars) | Unique per submission. Used for de-dup, status lookup, and the result cache key. |
| `kernel_code` | string (10 B – 100 KB) | The submission. For `cuda_agent` / `tvm_ffi` backends, this is the three-section text (`### CUDA_KERNELS` / `### APPLY_BINDINGS` / `### MODEL_NEW`). |
| `reference_code` | string | Required for the default `kernelbench` workflow. Plain PyTorch `Model` that defines the reference behavior + `get_inputs()` + `get_init_inputs()`. |

Backend / workflow selection:

| Field | Default | Notes |
|---|---|---|
| `backend` | `auto` | One of `cuda`, `triton`, `cuda_agent`, `tvm_ffi`, `auto`. `auto` lets the backend-adapter pick. |
| `backend_adapter` | `"kernelbench"` | Adapter that interprets the request and dispatches to a backend. |
| `toolkit` | `"kernelbench"` | Toolkit driver for correctness + perf measurement. |
| `workflow` | `"kernelbench"` | Controller name. Other workflows accept this via `/workflow/submit`. |
| `entry_point` | `"Model"` | Reference class name in `reference_code`. The kernel side uses `ModelNew` by convention. |

Trial budget:

| Field | Range | Default | Notes |
|---|---|---|---|
| `num_correct_trials` | 1–20 | 5 | Correctness trials with fresh random inputs each. First-failure aborts the run (see [correctness](#correctness-semantics)). |
| `num_perf_trials` | 1–1000 | 100 | Perf trials. With profiler on (default), a subset of these is also profiled. |
| `num_warmup` | 0–100 | 3 | Warmup iters before timed perf trials. |
| `perf_trim_count` | 0–50 | 0 | Trim N highest + N lowest perf samples before mean. |
| `memory_ratio_threshold` | >1 or null | 1.8 | Add `memory.comparison.warning` when Kernel total-task peak allocated memory is greater than or equal to this multiple of the reference. `null` disables the warning. The warning can be used by downstream reward shaping without changing correctness or task status. |
| `timeout` | 10–3600 s | 300 (model default) / 180 (v1 deployment) | Per-task wall budget. Hard kill once exceeded. |
| `workflow_timeout` | Positive finite seconds | `WORKFLOW_TIMEOUT`, default 1800 | Independent end-to-end parent budget, including CPU/GPU queue wait and all stages. Does not change `timeout` or `DEFAULT_TIMEOUT`. |

Caching / dedup:

| Field | Default | Notes |
|---|---|---|
| `force_refresh` | `false` | Bypass the per-task **result** cache (does NOT bypass compile-layer caches). |
| `enable_compile_artifact_cache` | `false` | Opt into the whole-`.so` cache keyed by content hash. Independent of the object cache (always on for `cuda_agent`). |
| `simplify_error` | `true` | Simplify compilation, runtime, and Sanitizer raw errors by removing local `.venv`, kernel build-directory, and dynamically detected KernelGYM source-root prefixes while keeping relative filenames and line/column numbers. Set `false` to retain full output paths for debugging. |
| `use_reference_cache` | `false` | Reuse cached reference timing (paired with `uuid`). |
| `uuid` | null | Reference timing cache key for `use_reference_cache=true`. |
| `is_valid` | `false` | If true, route to the `val_data_cache` namespace instead of the default cache. |

Step toggles (override service defaults):

| Field | Notes |
|---|---|
| `run_correctness`, `run_performance`, `run_triton_detection` | Per-call overrides for each evaluation step. |
| `enable_profiling` | `null` = use server `ENABLE_PROFILING` env, else explicit `true`/`false`. |
| `enable_ncu` | `false` by default; `null` uses server `ENABLE_NCU` (also default `false`). Set `true` to collect NCU metrics after correctness and performance gates pass. |
| `return_detail_correctness` | `false` by default. When false, correctness uses and returns the legacy `allclose` plus max/average-difference path. When true, mismatch diagnostics are computed and Compute Sanitizer is allowed to run. |
| `enable_compute_sanitizer` | `false` by default; `null` uses server `ENABLE_COMPUTE_SANITIZER`. Sanitizer runs only when both this field and `return_detail_correctness` are true; then a fresh child process is launched after a candidate correctness runtime error or output mismatch. |
| `compute_sanitizer_mode` | Sanitizer strategy: `error_based` (default) selects one check for a classified runtime error or uses a trigger-specific staged order with first-issue early stopping; `full` always runs all four checks without early stopping. Individual check names are internal execution modes and are not accepted in the payload. |
| `enable_correctness_input_perturbations` | `null` = use server `ENABLE_CORRECTNESS_INPUT_PERTURBATIONS` (default `false`). When enabled, correctness cycles through original, scale-up, scale-down, and sign-challenge inputs. Direct `torch.rand` and `torch.randn` inputs use different sign-challenge transforms. |
| `enable_triton_detection`, `detect_decoy_kernel` | Decoy-kernel checks; see [REWARD_HACKING_DEFENSES](design-doc/REWARD_HACKING_DEFENSES.md). |
| `measure_performance` | Legacy alias for `run_performance`. |
| `verbose_errors` | `null` = server default (`VERBOSE_ERROR_TRACEBACK`). |
| `priority` | `low` / `normal` / `high` (scheduler hint). |
| `device_preference` | E.g. `cuda:3`. Hint only; load balancer decides. |

Split compile/execute (advanced, see [COMPILE_ACCELERATION](design-doc/COMPILE_ACCELERATION.md) §"Split Compile/Execute"):

| Field | Notes |
|---|---|
| `split_compile_and_execute` | Run compile on a CPU worker, hand artifact to a GPU worker. |
| `pure_compile_task` | Compile only; do not run the kernel. |
| `task_stage`, `required_resource`, `assigned_worker`, `compile_artifact` | Used internally by the split flow; clients normally leave these unset. |
| `resources` | E.g. `{"gpus": 2}`. Resource-aware scheduling. |

`workflow == "kernel_simple"` additionally accepts:

| Field | Notes |
|---|---|
| `cases_code` | Python defining `get_cases()` / `get_inputs()`. |
| `cases` | Inline list of cases. |

### Response (`EvaluationResponse`)

```json
{
  "task_id": "rl_batch_001",
  "status": "completed",
  "compiled": true,
  "correctness": true,
  "decoy_kernel": false,
  "reference_runtime": 0.0264,
  "kernel_runtime": 0.0233,
  "speedup": 1.13,
  "memory": {
    "reference": {
      "absolute_peak_allocated": "35.50 MB",
      "task_peak_allocated_delta": "35.00 MB",
      "forward_peak_allocated_delta": "1.00 MB"
    },
    "kernel": {
      "absolute_peak_allocated": "34.50 MB",
      "task_peak_allocated_delta": "34.00 MB",
      "forward_peak_allocated_delta": "512.00 KB"
    },
    "comparison": {
      "measurement_status": "complete",
      "kernel_minus_reference": "-1.00 MB",
      "kernel_to_reference_ratio": 0.9714
    }
  },
  "metadata": { /* see below */ },
  "error_message": null,
  "error_code": null,
  "submitted_at": "2026-05-21T10:30:00Z",
  "completed_at": "2026-05-21T10:30:15Z",
  "processing_time": 15.2
}
```

`status` values: `pending`, `processing`, `completed`, `failed`, `timeout`.

Memory feedback is returned for correct kernels:

| Field | Meaning |
|---|---|
| `memory` | Contains the reference/kernel absolute allocator peak, task peak delta, and forward peak delta, plus the task-peak comparison. Public values use adaptive B/KB/MB/GB strings; internal arithmetic still uses integer bytes. |
| `memory.reference.absolute_peak_allocated`, `memory.kernel.absolute_peak_allocated` | Absolute `torch.cuda.max_memory_allocated()` observed during the measured forward. This includes the evaluation environment floor and is not a delta. |
| `memory.reference.task_peak_allocated_delta`, `memory.kernel.task_peak_allocated_delta` | Peak allocated memory above the environment floor captured before task-owned models and inputs are created. |
| `memory.reference.forward_peak_allocated_delta`, `memory.kernel.forward_peak_allocated_delta` | Peak allocated-memory increase above the baseline taken after models and inputs are prepared. |
| `memory.comparison.measurement_status` | `complete` means usable and complete, `partial` means usable but potentially a lower bound, and `invalid` means the measurement cannot be compared. |
| `memory.comparison.kernel_minus_reference` | Signed difference computed as Kernel minus reference for `task_peak_allocated_delta`. Negative means the Kernel uses less memory; positive means it uses more. |
| `memory.comparison.kernel_to_reference_ratio` | Kernel divided by reference for `task_peak_allocated_delta`, rounded to 4 decimal places; below 1 means the Kernel uses less memory. |
| `memory.comparison.warning` | A single warning string returned when `kernel_to_reference_ratio` is greater than or equal to `memory_ratio_threshold`. The message includes the actual ratio and configured threshold. |
| `memory.allocator_check` | Returned only when the Kernel source contains a direct CUDA allocation or another allocator warning. |

The default `1.8` memory-ratio warning threshold is an engineering policy, not a threshold prescribed by
[KernelBench-Verified](https://arxiv.org/abs/2607.16241). The paper treats any Kernel memory increase as reduced
memory efficiency, shows a `1.81x` duplicate-weight implementation as counter-productive, and notes that a `3x`
memory footprint can make an otherwise faster Kernel impractical. Because this signal may drive a binary reward
penalty, the configurable `1.8x` line catches regressions comparable to the paper's `1.81x` counter-productive
example while avoiding penalties for smaller workspace increases.

Runtime Sanitizer execution is gated by `return_detail_correctness=true` and `enable_compute_sanitizer=true`. With either field false, no sanitizer child is launched and no sanitizer metadata is added. When both are true, compiled CUDA candidates trigger it if the candidate `custom_forward` raises during correctness or completes with an output value/shape mismatch. Under `error_based`, the checks are ordered by trigger and stop at the first detected issue: ambiguous runtime errors use `memcheck`, `synccheck`, `racecheck`, `initcheck`, while output mismatches use `racecheck`, `initcheck`, `memcheck`, `synccheck`. A specifically classified runtime error runs only the matching check. Mismatch-triggered feedback is returned only when at least one sanitizer issue is detected; clean, unavailable, timeout, and tool-error results add no sanitizer fields. Runtime-error-triggered diagnostics retain their existing status feedback.

| Field | Meaning |
|---|---|
| `runtime_sanitizer.status` | `clean`, `issues_found`, `partial`, `error`, `unavailable`, or `skipped`. `clean` means the selected checks explicitly reported zero sanitizer issues; the replayed target may still reproduce the known correctness failure, recorded by `target_application_failed`. Tool timeout/unavailability is fail-open metadata. |
| `runtime_sanitizer.requested_checks` | Ordered checks planned for this run. |
| `runtime_sanitizer.check_results[].check` | Check represented by this result: `memcheck`, `synccheck`, `racecheck`, or `initcheck`. |
| `runtime_sanitizer.check_results[].issues[]` | Unique issue groups containing hazard, `kernel_info`, access type, `occurrence_count`, compact thread/block axis values such as `"x": [start, end]`, address `ranges: [start, end]`, two representative occurrences, and a bounded raw excerpt. Equivalent diagnostics that differ only by Kernel name and source line are secondarily merged into `kernel_info`; the check name is stored only on the parent check result. |
| `runtime_sanitizer.check_results[].unique_issue_count` | Number of final issue groups after repeated occurrences are aggregated and equivalent Kernel/source locations are secondarily merged. |
| `runtime_sanitizer.check_results[].parsed_issue_count` | Number of individual diagnostic occurrences parsed before aggregation, capped at 5000. |
| `runtime_sanitizer.check_results[].aggregation_complete` | Whether all detected occurrences were available within the parsing cap. |
| `runtime_sanitizer.replayed_input_seed` | Failed correctness trial seed regenerated in the child; `initcheck` may switch GPU-generated inputs to CPU + H2D as described below. |
| `runtime_sanitizer.executed_checks` | Checks actually executed by the selected mode. |
| `runtime_sanitizer.skipped_checks` | Planned checks not executed because a staged path found an issue or exhausted its total budget. |
| `runtime_sanitizer.stop_reason` | `first_issue`, `total_timeout`, or `null`. |
| `runtime_sanitizer.execution_policy` | `classified_single_check`, `runtime_error_likely_first`, `mismatch_likely_first`, or `full`. |
| `runtime_sanitizer.mode` | Actual `run_compute_sanitizer` execution mode: one check or `full`. |
| `runtime_sanitizer.selection_mode` | Payload strategy: `error_based` or `full`. |
| `runtime_sanitizer.error_classification` | Check selected from the error, or `ambiguous`. |
| `runtime_sanitizer.run_all_checks` | `true` only for explicit `full`, which has no first-issue early stop. |
| `runtime_sanitizer.measurement_complete` | Whether every planned check completed with a usable result. |
| `runtime_sanitizer.diagnostic_policy_complete` | Also becomes true when a staged path intentionally stops after finding its first issue. |
| `runtime_sanitizer.total_timeout_s` | Whole-diagnostic wall-time budget; the default is 60 seconds. |
| `runtime_sanitizer.check_results[].timeout_s` | Effective timeout for this check after applying its per-check limit and remaining total budget. |
| `runtime_sanitizer.primary_check` | Error-classified check used for the top-level issue count; for an ambiguous full run, the first check that reports an issue. |
| `runtime_sanitizer.detected_issue_count` | Issue count from `primary_check`; counts from heterogeneous tools are not added together. |
| `runtime_sanitizer.issue_count_by_check` | Per-check issue counts for full diagnostics. |
| `runtime_sanitizer.issues_truncated` | `true` when at least one check exceeds the 5000-occurrence parsing cap or the hard limit of four unique groups. Repeated occurrences merged into one group do not count as truncation. |
| `runtime_sanitizer.check_results[].input_generation` | `gpu` normally; `initcheck` uses `cpu_then_h2d` so filtered-out PyTorch RNG kernels do not cause false uninitialized-read reports. |
| `runtime_sanitizer.check_results[].input_values_exactly_replayed` | `false` only when an originally GPU-generated input is regenerated on CPU for `initcheck`; shape, dtype, and seed are replayed but RNG values may differ. |
| `runtime_sanitizer.check_results[].target_application_failed` | Whether Compute Sanitizer reported that the target application itself failed. |

A specific correctness runtime error is classified outside `run_compute_sanitizer`: memory errors select `memcheck`, synchronization errors select `synccheck`, race errors select `racecheck`, and uninitialized-read errors select `initcheck`. In payload strategy `error_based`, an ambiguous runtime error or output mismatch uses the scenario-specific staged order described above. Payload strategy `full` always runs `memcheck`, `synccheck`, `racecheck`, and `initcheck` without stopping after the first issue. The failing input is regenerated from the recorded trial seed. Defaults limit candidate launches to 8 and sanitizer reports to 1000; per-check timeouts are 20 seconds for memcheck, 15 for synccheck, 30 for racecheck, and 20 for initcheck, all capped by the 60-second total budget. Racecheck uses analysis reporting rather than the more verbose all-hazard report.

When `enable_correctness_input_perturbations=true`, at least four correctness trials are run. Direct floating-point outputs of `torch.rand` use `original`, `x3`, `x0.01`, and negation; direct floating-point outputs of `torch.randn` use `original`, `x3`, `x0.01`, and absolute value. Integer, boolean, scalar, and unrecognized inputs are unchanged. If the reference raises or produces NaN/Inf for a non-original perturbation, that perturbation is recorded in `metadata.correctness_reference_skipped_perturbations` and excluded from the correctness denominator. Numerical kernel mismatches always retain the legacy `max_difference`, `avg_difference`, `correctness_atol`, and `correctness_rtol` metadata fields. The remaining diagnostics in this paragraph require `return_detail_correctness=true`: `element_correctness` reports the element-count-weighted percentage satisfying `abs(candidate - reference) <= atol + rtol * abs(reference)`, formatted with two decimal places such as `99.80%`; `element_correctness_curve` reports the same percentage at `1x`, `2x`, `4x`, `8x`, and `16x` the base tolerance. Once any curve point reaches `100.00%`, later points are neither compared nor included in the returned curve or issue text. `nan_count` and `inf_count` count non-finite candidate-output elements. `mismatch_coordinate` contains the first and last mismatch in output traversal order plus `top_3`, the three most severe coordinates ordered by normalized error; each coordinate carries an `output_path` for nested outputs. These diagnostics are lists aligned with numerical mismatch trials. Both correctness percentages are included in the `correctness_issue` text, which also identifies the failed perturbation. Performance and memory measurements continue to use the original input distribution.

With `return_detail_correctness=true`, rank-2-or-higher tensor outputs also return `batch_correctness`, `row_correctness`, and `tile_correctness` as verifier fractions from 0 to 1. A rank-3 `B x M x N` output is split into whole `B` slices, `(B,M)` rows over `N`, and `32 x 32` tiles over the last two axes; partial tail tiles use their actual size. `output_space_localization` contains per-tensor verifier counts, at most 32 failed units per category, and inclusive mismatch bounds such as `"N": [128, 143]`. Tile ranges use half-open `[start, end]` coordinates.

`metadata` is a large dict of server-side timing + caching diagnostics. Notable keys:

| Key | Meaning |
|---|---|
| `device`, `gpu_name` | GPU the run landed on |
| `device_info` | Device metadata detected at service startup from torch, `nvidia-smi`, and `nvcc` |
| `kg_kernel_total_s` | Total task time inside the GPU worker |
| `kg_kernel_backend_compile_s` | Compile + import (or `dlopen` on cache hit) |
| `kg_kernel_backend_load_s` | Pure `dlopen` portion |
| `kg_kernel_performance_step_s` | Perf phase incl. profiler |
| `kg_kernel_correctness_s` | Correctness trials |
| `kg_reference_total_s` | Reference timing |
| `wg_pool_total_s`, `wg_pool_idle_wait_s`, `wg_pool_restart_s` | Subprocess pool dispatch metrics |
| `build_backend` | `manual_ninja` / `tvm_ffi.cpp.build` / cached |
| `compilation_error_detail` | Object mapping each stable compile-error category to its source-ordered, deduplicated compiler excerpts. Each excerpt contains the `error:` line and, when emitted immediately afterward, its source line; caret-only locator lines are omitted. Categories include `tvm_ffi_api_dtype`, `undefined_identifier`, `invalid_type_conversion`, `syntax_error`, `incomplete_type`, and `other`. Locations are simplified before the response is returned. |
| `compile_artifact_cache_enabled`, `compile_artifact_cache_hit`, `compile_artifact_cache_key` | Artifact-cache state |
| `compile_timing.manual_ninja_build_wall_sec`, `compile_timing.manual_ninja_import_wall_sec` | Cuda_agent ninja path internals |
| `compile_timing.manual_ninja_object_cache.{hits,misses,skipped,objects}` | Per-object cache outcome |
| `correctness_early_stop_enabled`, `correctness_trials_run`, `correctness_current_trial` | Correctness loop state |
| `element_correctness` | Detailed mode only: per-mismatching-trial element-count-weighted correctness percentages, formatted with two decimal places. |
| `element_correctness_curve` | Detailed mode only: per-mismatching-trial correctness curves at `1x`, `2x`, `4x`, `8x`, and `16x` the base tolerance. |
| `nan_count`, `inf_count` | Detailed mode only: per-mismatching-trial counts of non-finite candidate-output elements. |
| `mismatch_coordinate` | Detailed mode only: per-mismatching-trial first, last, and normalized-error-ranked `top_3` mismatch coordinates, including nested-output paths. |
| `batch_correctness`, `row_correctness`, `tile_correctness` | Detailed mode only: per-mismatching-trial output-space verifier fractions from 0 to 1. |
| `output_space_localization` | Detailed mode only: per-tensor batch, row, and `32 x 32` tile verifier counts, bounded failed-unit lists, and mismatch axis bounds. |
| `correctness_diagnosis` | Detailed mode only: high-confidence pattern diagnosis with `category`, `confidence`, `evidence`, and human-readable `text`; `text` is also appended to `error_message`. Omitted when Sanitizer runs or the mismatch pattern is not sufficiently typical. |
| `kg_kernel_perf_mean_ms`, `kg_kernel_perf_std_ms`, `kg_kernel_perf_min_ms`, `kg_kernel_perf_max_ms` | Per-trial perf stats |
| `custom_kernel_cuda_time_in_profiling_us`, `*_coverage` | Profiler attribution |
| `ncu.status`, `ncu.kernels`, `kg_kernel_ncu_profile_s` | Nsight Compute status, compact per-kernel metrics, and collection wall time. The default set includes L1/L2 throughput utilization and sector hit rates: `l1tex__throughput.avg.pct_of_peak_sustained_active`, `l1tex__t_sector_hit_rate.pct`, `lts__throughput.avg.pct_of_peak_sustained_elapsed`, and `lts__t_sector_hit_rate.pct`. It does not include request/sector counts or read/write byte totals. |

### Example

```bash
curl -s --noproxy '*' -X POST http://127.0.0.1:20111/evaluate \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "task_id": "demo_001",
  "reference_code": "import torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self, a, b):\n        return a + b\ndef get_inputs():\n    return [torch.randn(4096, device='cuda'), torch.randn(4096, device='cuda')]\ndef get_init_inputs():\n    return []",
  "kernel_code": "### CUDA_KERNELS\n```cpp\n...\n```\n### APPLY_BINDINGS\n```cpp\n...\n```\n### MODEL_NEW\n```python\n...\n```",
  "backend": "cuda_agent",
  "num_correct_trials": 3,
  "num_perf_trials": 20,
  "timeout": 180,
  "entry_point": "Model"
}
JSON
```

### Result cache + `force_refresh`

The server computes a request hash (see `kernelgym/server/request_hash.py`) over the request body, excluding fields like `force_refresh` and `task_id`. Identical content gets returned from the result cache without re-running.

- `force_refresh: true` → skip the cache for this submission (still writes the new result back).
- `force_refresh: false` (default) → cache hit returns a previous identical result in milliseconds.

The compile-layer caches (per-object cache, compile artifact cache) live below this and operate even when `force_refresh: true`.

### Correctness semantics

`stop_on_first_failure` is on by default: as soon as one correctness trial fails, the run aborts with `correctness=false`. The v1 deployment also disables the wall-clock time-budget early-pass mechanism — every configured trial runs unless `stop_on_first_failure` fires. Env-var overrides exist in `kernelgym/toolkit/kernelbench/correctness.py`.

## `POST /benchmark/speed-test` — fixed end-to-end speed test

Runs a built-in correct FP32 TVM-FFI case with a `4096×16` by `16×512` GEMM followed by RMSNorm three times in sequence. The candidate uses TF32 Tensor Cores and fuses GEMM, row reduction, and normalization into one CUDA kernel so the test exercises a real fusion speedup over the eager PyTorch expression. Each run uses a unique task id and CUDA source marker, forces a result-cache miss, disables the compile-artifact and reference caches, and executes five correctness trials plus 300 performance trials after three warmups. Each successful correctness/performance run also invokes NCU and returns its status and compact per-kernel metrics under `runs[].ncu`; NCU remains fail-open, so an unavailable profiler or performance-counter permission failure does not change the correctness-based `passed` value. After timing information is collected, the endpoint removes the generated parent, compile, kernel, and reference task/result records from Redis, so the probe leaves no reusable result-cache entries. Detailed correctness, Compute Sanitizer, adaptive performance trials, and correctness input perturbations remain disabled.

The endpoint takes no request body:

```bash
curl -sS -X POST http://127.0.0.1:20111/benchmark/speed-test
```

The response contains every run and the three-run aggregate. `end_to_end_s`, `total_end_to_end_s`, and all `stage_timings` values are seconds; `reference_runtime_ms` and `kernel_runtime_ms` are CUDA-event milliseconds. Runtime and stage averages use successful runs, while average end-to-end time covers all three attempts. `all_passed` is true only when all three runs compile and pass correctness.

When split compile/execute is enabled, `stage_timings.kernel_compile_s` and `compile_worker_total_s` come from the corresponding CPU compile sub-task. `stage_timings.ncu_profile_s` is the NCU collection wall time, including target launch, profiling, report export, and parsing. The end-to-end time also includes scheduler, Redis, queue, reference, compile, execute, NCU, and response assembly overhead.

```json
{
  "benchmark_status": "passed",
  "case_name": "gemm_rmsnorm_fp32",
  "backend": "tvm_ffi",
  "precision": "fp32",
  "input_shapes": {"lhs": [4096, 16], "rhs": [16, 512]},
  "repeat_count": 3,
  "passed_runs": 3,
  "all_passed": true,
  "total_end_to_end_s": 18.9,
  "average": {
    "end_to_end_s": 6.3,
    "reference_runtime_ms": 0.08,
    "kernel_runtime_ms": 0.068,
    "speedup": 1.19,
    "stage_timings": {
      "kernel_compile_s": 2.4,
      "kernel_correctness_s": 1.4,
      "kernel_performance_s": 0.06,
      "ncu_profile_s": 1.8
    }
  },
  "runs": [
    {
      "run_index": 1,
      "task_id": "speed-test-012345abcdef-1",
      "status": "completed",
      "passed": true,
      "compiled": true,
      "correctness": true,
      "end_to_end_s": 6.2,
      "reference_runtime_ms": 0.08,
      "kernel_runtime_ms": 0.068,
      "speedup": 1.19,
      "stage_timings": {"kernel_compile_s": 2.4, "ncu_profile_s": 1.8},
      "ncu": {"status": "ok", "profiled_kernel_count": 1, "kernels": []},
      "error_code": null,
      "error_message": null
    }
  ]
}
```

The example abbreviates `runs`; a real response always contains three objects.

## `POST /benchmark/speedup-noise-floor` — offline speedup-noise calibration

Runs ten fixed correct FP32 CUDA TVM-FFI candidates in a round-wise shuffled order. The suite contains GEMM + RMSNorm, BatchNorm, Conv2D, and seven operations selected from the configured KernelBench level-1 parquet: ReLU, diagonal matrix multiplication, MinGPT GELU, sum reduction, average pooling 1D, LayerNorm, and batched matrix multiplication. The reference, CUDA, binding, and fixed input definitions are embedded in `noise_floor_cases.py`; runtime calibration never reads the parquet. Its path and problem ids are provenance metadata only. Fixed calibration shapes span short, medium, and long runtimes; the source problem id and any shape override are returned in `cases`.

The minimum default protocol makes 100 independent evaluator requests: ten kernels by ten blocks. Every request has a unique task id, performs one correctness trial, then three warmups and 50 candidate/reference timing trials, forces fresh result/timing measurement, never uses the reference cache, and deletes its Redis task records afterward. The ten fixed CUDA sources are identical across blocks, so their compiled shared objects may be reused through the compile-artifact cache; compile latency is not part of the speedup statistic. Two rounds per kernel are held out by default, leaving eight for floor estimation. The schedule shuffles all ten kernels separately in each round so adjacent requests do not repeatedly measure one case. With the deployment default `MAX_TASKS_PER_WORKER=1`, each evaluator request also runs in a fresh CUDA subprocess; the returned PID and timestamp fields allow this to be audited.

```bash
curl -sS -X POST http://127.0.0.1:20111/benchmark/speedup-noise-floor \
  -H 'Content-Type: application/json' \
  -d '{
    "blocks_per_kernel": 10,
    "heldout_blocks_per_kernel": 2,
    "num_warmup": 3,
    "num_perf_trials": 50,
    "global_percentile": 75
  }'
```

Important request fields:

| Field | Default | Meaning |
|---|---:|---|
| `blocks_per_kernel` | 10 | Total independently submitted blocks for each of the ten fixed cases. |
| `heldout_blocks_per_kernel` | 2 | Final numbered rounds excluded from fitting and used only for validation. |
| `num_warmup` | 3 | Warmups inside every evaluator request. Set this to the training protocol value. |
| `num_perf_trials` | 50 | Candidate CUDA-event samples in each block. |
| `refer_num_perf_trials` | null | Reference samples; null means use `num_perf_trials`. |
| `perf_trim_count` | 0 | Samples trimmed from each tail before mean/std reporting. |
| `random_seed` | 20260909 | Reproducible round-wise interleaving order. |
| `global_percentile` | 75 | Percentile of valid per-kernel floors used as the global floor. |
| `one_sided_z` | 1.645 | One-sided LCB and held-out coverage threshold. |
| `target_node_id`, `target_hostname` | null | Route every block to the intended calibration node, such as the A800 node. |
| `persist_artifact` | true | Atomically save the full response under `<LOG_DIR>/noise_floor/`. |

For block `b` of kernel `q`, the response computes `log_speedup = log(speedup)` and

```text
within_log_speedup_variance = kernel_cv^2 / kernel_num_trials
                              + reference_cv^2 / reference_num_trials
```

The calibration endpoint always measures the reference afresh, so both terms are present. For each kernel, fitting uses the sample variance of calibration-block log speedups:

```text
noise_floor = sqrt(max(
    variance(log_speedup) - mean(within_log_speedup_variance),
    0
))
```

`analysis.global.noise_floor` is the requested percentile (p75 by default) across valid kernel floors. `noise_floor_multiplicative_percent = 100 * (exp(noise_floor) - 1)` gives a more intuitive multiplicative scale. The response also reports p75 by measured runtime bucket (`<0.1 ms`, `0.1–1 ms`, `>1 ms`), held-out one-sided lower-bound coverage, and the false-positive rate among kernels whose last calibration-block LCB claims `speedup > 1`.

Every `blocks[]` record contains `kernel_id`, block/schedule indexes, `speedup`, candidate and reference mean/std/trial counts, CVs, within-block variance, cache status, UTC times, end-to-end latency, device metadata/index, and candidate/reference execution PID, hostname, and epoch timestamp. `analysis.execution_environment` reports whether all blocks actually used one host/device and whether each observed candidate block had a unique process. Failed compile/correctness/performance requests remain in the artifact with their error information and are excluded from fitting. `calibration_status=passed` requires all blocks, all ten kernel estimates, and Redis cleanup to succeed; partial data returns `partial` and is still analyzable.

## `POST /evaluate/batch`

Submits up to 100 `EvaluationRequest`s in one body. Returns a `BatchEvaluationResponse` with one `EvaluationResponse` per task plus aggregate counts. Server processes tasks sequentially within the batch.

```json
{
  "batch_id": "rl_batch_001",
  "tasks": [ /* EvaluationRequest, ... */ ]
}
```

## `POST /workflow/submit` + `GET /workflow/results/{task_id}`

Generic workflow submission for non-`kernelbench` workflows.

```json
{
  "workflow": "kernel_simple",
  "task_id": "wf_demo_001",
  "force_refresh": false,
  "payload": { /* workflow-specific */ }
}
```

`/workflow/results/{task_id}` returns the same task result in `WorkflowResponse` shape (wraps `result` instead of flattening evaluation fields).

## `POST /debug/validate`

Validates the request shape and runs the workflow's `validate_request` step without submitting. Useful for dry-running malformed `kernel_code` parsing.

## Task lifecycle

| Endpoint | Returns |
|---|---|
| `GET /status/{task_id}` | `TaskStatusResponse`: status and timing fields; workflow parents additionally expose `workflow_generation`, Unix `workflow_deadline`, and planned `children` IDs. Available from acceptance, including queue wait. 404 if unknown/expired. |
| `GET /results/{task_id}` | Full `EvaluationResponse` (404 if not yet stored). |
| `DELETE /tasks/{task_id}` | Idempotently records cancellation, including unknown/terminal IDs. Returns 200 only after the fence is persisted; storage failure returns 500. Does not certify GPU process exit. |

### `DELETE /tasks/{task_id}` — cancellation semantics

The synchronous POST contract is unchanged, but acceptance atomically creates a parent `pending` record before execution. One Redis-owned generation runs the controller, renewing a separate active lease (`WORKFLOW_LEASE_SECONDS`, default 60 seconds). The parent becomes `processing` while the controller runs, including waits for children; it has no result-retention TTL until terminal commit. Redis time establishes the absolute deadline, so queue wait consumes the same end-to-end budget as execution. Deadline/lease checks are enforced again atomically when publishing a child, claiming/dispatching GPU work, and dispatching CPU compile work. An API watchdog and status/wait queries terminalize expired deadlines or abandoned owners without launching a replacement workflow.

Concurrent same-ID, same-content POSTs join the existing generation, even when `force_refresh=true`; only one controller submits children. Conflicting content while active returns HTTP 409. A terminal task that was not explicitly cancelled can be rerun with `force_refresh=true`, which creates new generation-scoped child IDs. Ordinary cached results retain request-hash validation. Disconnecting one HTTP waiter does not cancel a workflow shared with other callers; use `DELETE /tasks/{id}` for explicit cancellation. Owner shutdown records failure, and a crashed owner is fenced by lease expiry.

Cancellation is a real interrupt, not just a status flag. `/evaluate` decomposes the parent into generation-scoped sub-tasks such as `{id}_compile_{generation}`, `{id}_kernel_{generation}`, and `{id}_ref_{generation}`, each carrying `base_task_id == {id}`. Read `/status/{id}.children` instead of constructing suffixes. These IDs are reserved at acceptance; a cached/skipped stage may never create its child record. Cancelling the parent propagates to in-flight children and fences future stages, including late CPU completions from an older generation.

- **Pending / queued** — the parent terminal commit atomically publishes cancellation, removes known child queue entries, and finalizes children that have not crossed the execution fence. A worker that already popped a child rechecks the parent generation/deadline at dispatch and cannot start it after cancellation. The synchronous POST returns the parent's terminal result without waiting for a worker to free up.
- **Running on GPU** — the GPU worker running the task polls the cancellation marker (~1 s) and, on seeing it, kills the CUDA subprocess executing the task (the pool spawns a clean replacement) instead of letting it run to its `timeout`.
- **Running CPU compile** — the compile stage is not preemptively killed (it finishes on its own, usually quickly), but the workflow returns the cancelled result immediately without waiting for the GPU stages.

Explicit cancellation normally produces `error_message: "Task cancelled"` (`error_code: SYSTEM_ERROR`); an already-expired workflow records `timeout` / `TIMEOUT_ERROR`. A direct task also records `cancelled_at`. The first generation-fenced parent terminal commit wins: a later controller result cannot overwrite cancellation/timeout. Running or frozen GPU claims are not released by parent or direct-child cancellation; the worker's existing safe-reap/containment path owns that cleanup. Worker polling may observe cancellation after the current GPU operation finishes, but the parent terminal result remains immutable. Result and task retention use the existing `TERMINAL_RESULT_TTL_SEC` / `TERMINAL_TASK_TTL_SEC` settings, starting at terminal commit.

DELETE returns `{"message": "Cancellation recorded for task <id>", "cancellation_recorded": true}` even if POST has not arrived. The separate `{prefix}:cancelled:{id}` tombstone has no TTL and is excluded from result/ephemeral cleanup. Subsequent same-ID POSTs return HTTP 409, including cached results and `force_refresh=true`; retries must use a new ID. Parent and planned child IDs, legacy stage aliases, and child submissions carrying `base_task_id` are fenced. Both enqueue and CPU/GPU execution gates check cancellation atomically. A status/result 404 merely means no retained record exists; it does not indicate that future execution is impossible. Cancelling an already-terminal ID preserves its existing GET result while prohibiting future execution under that ID.

Tombstones are Redis coordination state, not a filesystem journal: preserving them across Redis loss requires Redis persistence and no destructive namespace reset. They intentionally do not auto-expire because an unknown delayed submission has no bounded arrival time. Normal result TTL cleanup is not authorization to remove cancellation tombstones; each cancelled ID requires a new ID for future requests.

If a registered child is frozen or its bound worker is quarantined, parent reconciliation returns `failed` / `SYSTEM_ERROR` with an explicit `Infrastructure failure` message and stops later stages. This does not publish a normal terminal result for a frozen child, clear its claim/inflight entry, or lift quarantine. Quarantine observation is read-only (including durable latch lookup after worker heartbeat expiry) and does not wait for the physical recovery lock. Business completion and safe GPU containment remain separate lifecycles.

For `/workflow/submit`, put a per-request `workflow_timeout` inside `payload`. Read-only status waits and Redis connection/command waits are bounded; increasing `DEFAULT_TIMEOUT` is not required to keep an active workflow visible.

See [TASK_CANCELLATION](design-doc/TASK_CANCELLATION.md) for the full design (markers, the worker watcher, and workflow propagation).

## Health and observability

### `GET /health`

Aggregated health snapshot.

```json
{
  "status": "healthy",
  "timestamp": "2026-05-21T10:30:00Z",
  "gpu_status": {
    "cuda:0": {
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total": "23.5GB",
      "memory_allocated": "0.0GB",
      "memory_reserved": "0.0GB",
      "memory_used_percent": "0.0%",
      "available": true
    }
  },
  "queue_status": {"pending": 0, "processing": 0, "completed": 1250},
  "memory_usage": {"cpu_percent": 45.2, "memory_percent": 67.8},
  "active_tasks": 0,
  "total_processed": 1245,
  "uptime": 86400.5
}
```

`status` is `healthy` when at least one GPU is `available: true`.

### `GET /metrics`

Counters and rolling stats. Shape:

```json
{
  "timestamp": "...",
  "performance_metrics": { /* avg compile/perf, throughput */ },
  "resource_metrics": { /* CPU, RAM, GPU util */ },
  "queue_metrics": { /* depth per priority */ },
  "error_metrics": { /* by ErrorCode */ }
}
```

A separate Prometheus exporter listens on `METRICS_PORT` (set by deployment profile).

### `GET /queue/status` and `GET /workers/status`

Untyped JSON; reflect raw Redis state. `workers/status` includes per-worker load-balancer fields, last-heartbeat timestamps, and assigned devices.

## Internal endpoints

These exist for the worker subprocesses and the multi-node deploy script. RL clients should not call them.

### `POST /worker/register`, `/worker/unregister`, `/worker/heartbeat`, `/worker/evict_from_lb`

Worker lifecycle. `register` takes `worker_id`, `device`, optional `node_id`, optional `hostname` as query parameters. `heartbeat` auto-registers an unknown `worker_id` if a `device` is provided AND not in conflict with another worker on the same node. `evict_from_lb` removes a worker from the in-memory load balancer without touching its Redis state — used to quarantine a flaky worker.

### `POST /node/allocate`

Allocates or returns a stable `node_id` keyed by `hostname`. Two modes:

| Call | Behavior |
|---|---|
| `POST /node/allocate?hostname=h1&node_name=node2` | Use `node2` as the id; conflict (409) if `node_name` is already bound to a different hostname. |
| `POST /node/allocate?hostname=h1` | Auto-allocate a sequential `node-<N>`. Idempotent per hostname. |

## Monitoring endpoints (`/monitoring/*`)

| Endpoint | Notes |
|---|---|
| `GET /monitoring/problematic-codes?min_errors=3` | Code hashes that have failed `min_errors` or more times. |
| `GET /monitoring/retry-queue` | Pending retries (next 10) with `scheduled_for` timestamps. |
| `GET /monitoring/worker-health` | Per-worker state plus CUDA-error-shutdown flags. |
| `POST /monitoring/clear-error-history/{code_hash}` | Reset the error counter for one specific submission hash. |

## Error model

Standard FastAPI error envelope for 4xx / 5xx:

```json
{ "detail": "Failed to submit task: ..." }
```

5xx responses also include an `X-Error-Code` header drawn from the `ErrorCode` enum:
`VALIDATION_ERROR`, `COMPILATION_ERROR`, `RUNTIME_ERROR`, `CORRECTNESS_ERROR`, `TIMEOUT_ERROR`, `SYSTEM_ERROR`, `RESOURCE_ERROR`, `UNKNOWN_ERROR`, `SYNTAX_ERROR`, `IMPORT_ERROR`, `INSTANTIATION_ERROR`.

When the evaluation itself fails inside the workflow (compile error, correctness mismatch, timeout), the HTTP status is still 200 — the `EvaluationResponse` carries `status: "failed" | "timeout"` plus `error_message` and `error_code`.

## OpenAPI

FastAPI publishes the live spec at `/openapi.json`; for an interactive browser hit `/docs` (Swagger UI) or `/redoc`. The hand-written tables above are the operator-friendly view; the OpenAPI spec is authoritative for exact field types.
