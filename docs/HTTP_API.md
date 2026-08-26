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
| GET | `/health` | Aggregated GPU + queue + memory health |
| GET | `/metrics` | Performance / resource / queue / error counters |
| POST | `/evaluate` | **Submit one kernel evaluation (primary endpoint)** |
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
| `timeout` | 10–3600 s | 300 (model default) / 180 (v1 deployment) | Per-task wall budget. Hard kill once exceeded. |

Caching / dedup:

| Field | Default | Notes |
|---|---|---|
| `force_refresh` | `false` | Bypass the per-task **result** cache (does NOT bypass compile-layer caches). |
| `enable_compile_artifact_cache` | `false` | Opt into the whole-`.so` cache keyed by content hash. Independent of the object cache (always on for `cuda_agent`). |
| `use_reference_cache` | `false` | Reuse cached reference timing (paired with `uuid`). |
| `uuid` | null | Reference timing cache key for `use_reference_cache=true`. |
| `is_valid` | `false` | If true, route to the `val_data_cache` namespace instead of the default cache. |

Step toggles (override service defaults):

| Field | Notes |
|---|---|
| `run_correctness`, `run_performance`, `run_triton_detection` | Per-call overrides for each evaluation step. |
| `enable_profiling` | `null` = use server `ENABLE_PROFILING` env, else explicit `true`/`false`. |
| `enable_ncu` | `null` = use server `ENABLE_NCU` env (default `true`), else explicit `true`/`false`. NCU runs only after correctness and performance gates pass. |
| `enable_compute_sanitizer` | `null` = use server `ENABLE_COMPUTE_SANITIZER` env (default `true`). A fresh child process is launched only when the candidate forward fails during correctness. |
| `compute_sanitizer_mode` | Sanitizer strategy: `error_based` (default) selects an internal check from the correctness error and falls back to all checks when ambiguous; `full` always runs all four checks. Individual check names are internal execution modes and are not accepted in the payload. |
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
  "runtime_sanitizer": { "status": "skipped", "reason": "correctness_passed", "check_results": [] },
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
| `memory.comparison.kernel_to_reference_ratio` | Kernel divided by reference for `task_peak_allocated_delta`; below 1 means the Kernel uses less memory. |
| `memory.allocator_check` | Returned only when the Kernel source contains a direct CUDA allocation or another allocator warning. |

Runtime Sanitizer feedback is returned for compiled CUDA candidates. It is normally `skipped`; execution is
triggered only when the candidate `custom_forward` raises during correctness. Output value/shape mismatch does not
trigger it.

| Field | Meaning |
|---|---|
| `runtime_sanitizer.status` | `clean`, `issues_found`, `partial`, `error`, `unavailable`, or `skipped`. `clean` means the selected checks explicitly reported zero sanitizer issues; the replayed target may still reproduce the known correctness failure, recorded by `target_application_failed`. Tool timeout/unavailability is fail-open metadata. |
| `runtime_sanitizer.requested_checks` | Checks selected for this run. |
| `runtime_sanitizer.check_results[].check` | Check represented by this result: `memcheck`, `synccheck`, `racecheck`, or `initcheck`. |
| `runtime_sanitizer.check_results[].issues[]` | Unique issue groups containing hazard, `kernel_info`, access type, `occurrence_count`, compact thread/block axis values such as `"x": [start, end]`, address `ranges: [start, end]`, two representative occurrences, and a bounded raw excerpt. Equivalent diagnostics that differ only by Kernel name and source line are secondarily merged into `kernel_info`; the check name is stored only on the parent check result. |
| `runtime_sanitizer.check_results[].unique_issue_count` | Number of final issue groups after repeated occurrences are aggregated and equivalent Kernel/source locations are secondarily merged. |
| `runtime_sanitizer.check_results[].parsed_issue_count` | Number of individual diagnostic occurrences parsed before aggregation, capped at 5000. |
| `runtime_sanitizer.check_results[].aggregation_complete` | Whether all detected occurrences were available within the parsing cap. |
| `runtime_sanitizer.replayed_input_seed` | Failed correctness trial seed regenerated in the child; `initcheck` may switch GPU-generated inputs to CPU + H2D as described below. |
| `runtime_sanitizer.executed_checks` | Checks actually executed by the selected mode. |
| `runtime_sanitizer.mode` | Actual `run_compute_sanitizer` execution mode: one check or `full`. |
| `runtime_sanitizer.selection_mode` | Payload strategy: `error_based` or `full`. |
| `runtime_sanitizer.error_classification` | Check selected from the error, or `ambiguous`. |
| `runtime_sanitizer.run_all_checks` | `true` only when the actual execution mode is `full`. |
| `runtime_sanitizer.primary_check` | Error-classified check used for the top-level issue count; for an ambiguous full run, the first check that reports an issue. |
| `runtime_sanitizer.detected_issue_count` | Issue count from `primary_check`; counts from heterogeneous tools are not added together. |
| `runtime_sanitizer.issue_count_by_check` | Per-check issue counts for full diagnostics. |
| `runtime_sanitizer.issues_truncated` | `true` when at least one check exceeds the 5000-occurrence parsing cap or the hard limit of four unique groups. Repeated occurrences merged into one group do not count as truncation. |
| `runtime_sanitizer.check_results[].input_generation` | `gpu` normally; `initcheck` uses `cpu_then_h2d` so filtered-out PyTorch RNG kernels do not cause false uninitialized-read reports. |
| `runtime_sanitizer.check_results[].input_values_exactly_replayed` | `false` only when an originally GPU-generated input is regenerated on CPU for `initcheck`; shape, dtype, and seed are replayed but RNG values may differ. |
| `runtime_sanitizer.check_results[].target_application_failed` | Whether Compute Sanitizer reported that the target application itself failed. |

A specific correctness error is classified outside `run_compute_sanitizer`: memory errors select `memcheck`,
synchronization errors select `synccheck`, race errors select `racecheck`, and uninitialized-read errors select
`initcheck`. In payload strategy `error_based`, an ambiguous error selects `full`; strategy `full` always selects
`full`. The concrete internal execution mode is then passed to the single `run_compute_sanitizer` entry point;
`full` runs `memcheck`, `synccheck`, `racecheck`, and `initcheck` without stopping after the first issue. The failing
input is regenerated from the recorded trial seed.

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
| `compile_artifact_cache_enabled`, `compile_artifact_cache_hit`, `compile_artifact_cache_key` | Artifact-cache state |
| `compile_timing.manual_ninja_build_wall_sec`, `compile_timing.manual_ninja_import_wall_sec` | Cuda_agent ninja path internals |
| `compile_timing.manual_ninja_object_cache.{hits,misses,skipped,objects}` | Per-object cache outcome |
| `correctness_early_stop_enabled`, `correctness_trials_run`, `correctness_current_trial` | Correctness loop state |
| `kg_kernel_perf_mean_ms`, `kg_kernel_perf_std_ms`, `kg_kernel_perf_min_ms`, `kg_kernel_perf_max_ms` | Per-trial perf stats |
| `custom_kernel_cuda_time_in_profiling_us`, `*_coverage` | Profiler attribution |
| `ncu.status`, `ncu.kernels`, `kg_kernel_ncu_profile_s` | Nsight Compute status, compact per-kernel metrics, and collection wall time |

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
| `GET /status/{task_id}` | `TaskStatusResponse`: `status`, `progress`, `queue_position`, `assigned_device`, `estimated_completion`. 404 if unknown. |
| `GET /results/{task_id}` | Full `EvaluationResponse` (404 if not yet stored). |
| `DELETE /tasks/{task_id}` | Cancels if pending/in-flight. 404 if unknown or already terminal. |

### `DELETE /tasks/{task_id}` — cancellation semantics

Cancellation is a real interrupt, not just a status flag. `/evaluate` runs as a workflow that decomposes the parent `task_id` into sub-tasks (`{id}_compile` on CPU, then `{id}_kernel` + `{id}_ref` on GPU, each carrying `base_task_id == {id}`). Cancelling the parent id propagates to whichever sub-task is in flight.

- **Pending / queued** — the id is removed from its resource/worker queue, and (for a workflow parent) a cancellation marker is published. A worker that later dequeues a task whose own id **or** `base_task_id` is cancelled drops it instead of running it, and the in-flight `/evaluate` request returns promptly (its wait is cancellation-aware) rather than blocking until a worker frees up.
- **Running on GPU** — the GPU worker running the task polls the cancellation marker (~1 s) and, on seeing it, kills the CUDA subprocess executing the task (the pool spawns a clean replacement) instead of letting it run to its `timeout`.
- **Running CPU compile** — the compile stage is not preemptively killed (it finishes on its own, usually quickly), but the workflow returns the cancelled result immediately without waiting for the GPU stages.

Either way the task ends terminal with `error_message: "Task cancelled"` (`error_code: SYSTEM_ERROR`); a direct task also records a `cancelled_at` timestamp. Cancelling a task that is already `completed`/`failed`/`timeout`, or an unknown id, returns 404. There is an inherent race where a task that finishes within the poll window may still record its real result.

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
