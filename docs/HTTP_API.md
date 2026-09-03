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
| `enable_compute_sanitizer` | `null` = use server `ENABLE_COMPUTE_SANITIZER` env (default `false`). A fresh diagnostic process is launched only after a candidate `custom_forward` runtime failure. |
| `compute_sanitizer_mode` | `error_based` (default) selects the relevant internal check and falls back to all checks when ambiguous; `full` always runs all four checks. Individual check names are not accepted in the public payload. |
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
  "metadata": { /* see below */ },
  "error_message": null,
  "error_code": null,
  "submitted_at": "2026-05-21T10:30:00Z",
  "completed_at": "2026-05-21T10:30:15Z",
  "processing_time": 15.2
}
```

`status` values: `pending`, `processing`, `completed`, `failed`, `timeout`.

When enabled and triggered, `runtime_sanitizer` contains the isolated replay result. The field is omitted for ordinary evaluations where the diagnostic is disabled or not triggered.

| Field | Meaning |
|---|---|
| `runtime_sanitizer.status` | `clean`, `issues_found`, `partial`, `error`, or `unavailable`. A tool timeout or a safety-driven replay skip is reported as diagnostic `error`, not as a second execution attempt in the original CUDA context. |
| `runtime_sanitizer.requested_checks`, `executed_checks` | Selected and completed checks among `memcheck`, `synccheck`, `racecheck`, and `initcheck`. |
| `runtime_sanitizer.check_results[].issues[]` | Bounded, aggregated hazards with kernel/source information, access type, occurrence counts, thread/block ranges, and a short raw excerpt. |
| `runtime_sanitizer.replayed_input_seed` | Seed from the failed correctness trial. `initcheck` may regenerate inputs on CPU then copy them to the GPU, so exact RNG values can differ while shape, dtype, and seed remain the same. |
| `runtime_sanitizer.detected_issue_count` | Count for the classified primary check; heterogeneous check counts are not summed. |
| `runtime_sanitizer.selection_mode`, `mode` | Public selection strategy and actual internal execution mode. |
| `runtime_sanitizer.kernel_filter_empty` | `true` when no backend kernel name was available. In that case no launch-count cap is applied, so framework launches cannot consume the cap before the candidate kernel. |

After a correctness-time CUDA failure, the original evaluation subprocess publishes a deferred diagnostic descriptor without another CUDA synchronization or cleanup call. Its parent freezes and reaps the process group, gives each requested diagnostic check its independent `COMPUTE_SANITIZER_TIMEOUT_S` budget, and only then starts fresh-context validation before admitting more work on that GPU. The diagnostic timeout does not inherit the evaluation task's remaining budget; `full` mode may therefore use the per-check timeout up to four times. This fault containment remains active when the sanitizer feature is disabled.

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
