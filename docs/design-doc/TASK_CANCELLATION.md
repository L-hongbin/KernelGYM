# Task Cancellation & Preemptive Interrupt

Status: current design
Date: 2026-06-09

This document describes how `DELETE /tasks/{task_id}` cancels work in the reward service. Cancellation is a real interrupt: a pending task is pulled from the queue before it runs, and a task already executing on a GPU has its CUDA subprocess killed instead of being allowed to run to completion or `timeout`. Because the primary `/evaluate` path is a workflow that decomposes one request into several sub-tasks, cancellation also has to propagate from the parent id the caller holds to whichever sub-task is in flight.

## Goals

- Cancelling a **pending** task removes it from its queue so it is never dispatched.
- Cancelling an **in-flight GPU** task stops the running CUDA work promptly (~1 s), rather than waiting for the per-task `timeout`.
- Cancelling an in-flight `/evaluate` **parent id** propagates to its decomposed sub-tasks even though the parent has no task hash while it runs.
- The in-flight HTTP request returns promptly on cancel, and never hangs waiting for a sub-task that has been cancelled.
- No queued sub-task can run "orphaned" after the workflow has already been cancelled.
- Cancellation is recorded as a terminal result that status/result queries can observe.

## Background: why a naive cancel was not enough

Two problems motivated this design.

1. **Pending cancel left the queue dirty.** The previous `cancel_task` only marked the task failed in Redis. It did not remove the id from `…:queue:resource:{cpu,gpu}` or the per-worker queue, and `get_next_task` did not re-check status on dequeue, so a task cancelled while still queued could later be popped and executed, overwriting the cancelled result.

2. **No preemptive interrupt.** Once a task reached a GPU subprocess, the worker blocked on a single `result_queue.get(timeout=...)` and only the per-task / subprocess `timeout` could stop it. Cancellation had no effect on running CUDA work.

A third complication is structural: `/evaluate` runs the KernelBench workflow **synchronously inside the request handler** (`_execute_workflow` → `KernelBenchWorkflowController.handle_request`) and only writes the parent `task_id`'s result at the very end. While it runs, the parent id has no `…:task:{id}` hash — `GET /status/{parent}` is 404 and `cancel_task(parent)` would find nothing. The actual work is decomposed (see [COMPILE_ACCELERATION](COMPILE_ACCELERATION.md) §Split Compile/Execute) into sequential sub-tasks `{parent}_compile` (CPU), `{parent}_kernel` (GPU execute), and `{parent}_ref` (GPU reference timing), each carrying `base_task_id == parent`.

## Design overview

Cancellation is coordinated through two short-lived Redis markers plus a worker-side watcher:

| Key | Set by | Read by | TTL |
| --- | --- | --- | --- |
| `{prefix}:cancel:{id}` | `cancel_task` (`_mark_task_cancelled`) | GPU worker watcher, `get_next_task`, `wait_unless_cancelled` | `max(60, default_timeout*2)` |
| `{prefix}:workflow:{base}` | controller (`register_workflow`) | `cancel_task` (`_is_workflow_active`) | `max(60, default_timeout*2)` |

The marker is the durable signal; everything else polls it. `id` here may be a concrete sub-task id **or** an `/evaluate` parent id — a running sub-task watches both its own id and its `base_task_id`, so marking the parent is sufficient to reach the child.

The work is split across four layers:

- **`TaskManager`** (`kernelgym/server/task_manager.py`) — marker bookkeeping, queue removal, dequeue-time drop.
- **GPU worker** (`kernelgym/worker/gpu_worker.py`) — per-task watcher that trips a `cancel_event`.
- **Subprocess pool** (`kernelgym/worker/subprocess_pool.py`) — turns `cancel_event` into a real subprocess kill.
- **Workflow controller + scheduler** (`kernelgym/workflow/kernelbench.py`, `kernelgym/server/scheduler.py`) — makes the parent id cancellable and the per-stage wait cancellation-aware.

## `cancel_task`

`TaskManager.cancel_task(task_id)` handles two cases:

1. **Direct task** (a `…:task:{id}` hash exists): if it is already terminal, return `False` (404). Otherwise publish `{prefix}:cancel:{id}`, remove the id from both resource queues and the assigned worker queue (`_remove_task_from_queues`), write the terminal `"Task cancelled"` result (`fail_task`, `error_code=SYSTEM_ERROR`), and stamp `cancelled_at`.

2. **Workflow parent** (no task hash, but `{prefix}:workflow:{id}` is active): publish `{prefix}:cancel:{id}` only and return `True`. The running sub-task's watcher sees the marker via its `base_task_id`; the controller writes the parent's terminal result when it aborts.

An id that is neither a live task nor a live workflow returns `False` (404), preserving the documented "unknown / already terminal → 404" contract.

## Dequeue-time drop

`get_next_task` is the single consumer of the queues, so it is also the last line of defense. After popping an id and loading its data it drops the task instead of dispatching when either:

- `_dropped_before_dispatch(task_hash)` — the task hash is already terminal (`completed`/`failed`/`timeout`), or
- `_dequeued_task_cancelled(prefix, task_id, base_task_id)` — a prefix-local single `EXISTS` over the task's own cancel marker **and** its `base_task_id` marker.

A dropped task is removed (not re-deferred), so a sub-task whose parent `/evaluate` was cancelled while it sat queued never starts. This adds one `EXISTS` to the dispatch hot path, scoped to the prefix being scanned.

## In-flight GPU interrupt

While a GPU task runs, `_run_toolkit_task` starts a `_cancellation_watcher` coroutine alongside `worker_pool.execute_task`. The watcher polls `is_task_cancelled` for both the sub-task id and its `base_task_id` every `cancel_poll_interval_sec` (default 1 s). On a hit it sets a `threading.Event` `cancel_event`.

`cancel_event` is threaded into the pool:

- `PersistentWorker.execute_task` no longer does one long blocking `result_queue.get(timeout)`. It polls the result queue in `poll_interval` slices, checking `cancel_event` and the deadline each slice. If `cancel_event` is set it marks itself not-alive and raises `TaskCancelledError`.
- `SubprocessWorkerPool.execute_task` catches `TaskCancelledError`, calls `_restart_worker` (which kills and recycles the subprocess — see [TWO_WORKER_WARM_POOL](TWO_WORKER_WARM_POOL.md) §Recycling), does **not** retry, and re-raises. `_get_idle_worker` also honors `cancel_event` so a cancel that lands while waiting for a free subprocess is not blocked.

Back in `gpu_worker._process_task`, `TaskCancelledError` is caught and the worker **writes a terminal cancelled result** for its sub-task via `complete_task`. This is essential: the workflow controller is blocked in `scheduler.wait*` on that sub-task id, and without a result it would hang until the request timeout.

Killing the subprocess (rather than just ignoring the result) is what actually frees the GPU; the warm-pool replacement path then restores a clean subprocess in the background.

## Workflow propagation

The controller makes the parent id cancellable and keeps each stage interruptible.

- **Registration.** `handle_request` is a thin wrapper that calls `scheduler.begin_workflow(base_id)` before running and `scheduler.end_workflow(base_id)` in a `finally`. These map to `register_workflow` / `unregister_workflow` (the `{prefix}:workflow:{base}` key), so `cancel_task` can recognize a live parent even though it has no task hash.

- **Cancellation-aware wait.** Every per-stage `scheduler.wait(child_id)` is replaced by `scheduler.wait_unless_cancelled(child_id, base_id)`. It returns the result if one is written, but returns `None` promptly once `is_task_cancelled(base_id)` is true — and, critically, it then calls `cancel_task(child_id)` to pull that specific queued/running child out of its queue so it cannot run orphaned after the parent marker eventually expires. The controller turns a `None` into `_cancelled_result(base_id)`.

- **Between-stage checks.** `_run_workflow` also calls `_is_cancelled(scheduler, base)` at each stage boundary (before kernel, before/after ref, before the split execute submit) and returns a cancelled result early, so a cancel that lands between sub-tasks does not start the next one.

The net effect: cancelling a parent during the CPU compile, during a queued GPU stage, or during a running GPU stage all return a terminal `"Task cancelled"` result to the caller within roughly the watcher/poll interval.

## Interaction with the warm pool

Interrupting a running task reuses the existing subprocess warm-pool recycle path ([TWO_WORKER_WARM_POOL](TWO_WORKER_WARM_POOL.md)) rather than adding a new one, so it does not destabilize the spare mechanism.

- **The killed subprocess is the active one, not the spare.** A running task occupies the subprocess that was moved from `idle_workers` to `busy_workers`; the warm spare sits untouched in `idle_workers`. Cancellation kills only the active subprocess, and the spare immediately serves the next task while a replacement spawns in the background — exactly as after a normal task with `MAX_TASKS_PER_WORKER=1`.
- **Same recycle machinery.** The `except TaskCancelledError` branch of `SubprocessWorkerPool.execute_task` calls the same `_restart_worker` (then `_return_worker` in `finally`) used by normal completion, CUDA errors, and timeouts. `_restart_worker` removes the dead worker and bumps `pending_replacements` only when `len(workers) + pending_replacements < pool_size`, so the capacity invariant `len(workers) + pending_replacements <= pool_size` is preserved by the same bookkeeping. `_return_worker` is a no-op for the already-removed, not-alive worker.
- **Kill path matches the timeout path.** A subprocess stuck in a CUDA kernel will not answer graceful shutdown, so the replenishment thread escalates graceful → terminate → SIGKILL → `waitpid`, waits ~2 s for the driver to reclaim VRAM, and the zombie reaper backstops. This is the same escalation timeouts already use; there is no new GPU-memory leak path.
- **No extra churn at the default.** With `MAX_TASKS_PER_WORKER=1` the subprocess would have been recycled after the task anyway, so cancellation only recycles it *earlier*; it adds no extra recycle. With `MAX_TASKS_PER_WORKER>1` it does force one recycle that would not otherwise happen, which is correct because a killed subprocess cannot be reused. Because the outer `GPUWorker` loop is serial, cancellations cannot pile up on one GPU worker — each is one bounded recycle.
- **Idle-wait cancel is state-free.** The `cancel_event` check in `_get_idle_worker` returns before taking the pool lock, so a cancel that lands while waiting for a free subprocess touches no pool state (no subprocess was acquired yet).

The only cost is the usual replacement latency, which the warm spare is designed to hide; the warm-pool "both subprocesses unavailable before a replacement is ready" gap is unchanged by cancellation.

## Status reporting

`cancel_task` (direct) writes the legacy `error`/`failed_at` result fields; the worker's cancelled result is a full `EvaluationResult` JSON. `get_task_result` (`/results`) already returns either shape. `get_task_status` (`/status`) was extended to pull `error_message` out of the JSON result payload for non-completed tasks, and `TaskStatusResponse` gained an `error_message` field (it was previously absent, so the value was silently dropped by the response model). After cancellation, `/status/{id}` returns `status: "failed"` with `error_message: "Task cancelled"`.

## Races and edge cases

- **Task finishes as cancel arrives.** `wait_unless_cancelled` checks for a written result before checking cancellation, and `cancel_task` refuses an already-terminal task. If a task completes within the poll window it may record its real result; this is an accepted, benign race.
- **Double result write.** For a directly cancelled running task, both `cancel_task` (legacy error shape) and the worker (`EvaluationResult` shape) may write a result. `get_task_result` prefers the JSON `result`, and both encode cancellation, so last-writer-wins is harmless.
- **Marker expiry.** Because `wait_unless_cancelled` removes the specific in-flight child from its queue, prompt cancellation no longer depends on the marker surviving until dispatch. The marker only needs to outlive the ~1 s watcher poll for a running task, which `max(60, default_timeout*2)` comfortably does.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `cancel_poll_interval_sec` | `1.0` | GPU worker watcher poll interval (optional `getattr` override). |
| marker TTL | `max(60, default_timeout*2)` | Lifetime of the `cancel:` and `workflow:` markers. |

## Observability

Log markers, by layer:

- Worker: `detected cancellation (marker=<id>) … aborting in-flight execution`, `task <id> cancelled; recording cancelled result`.
- Pool: `Cancellation requested for task <id>; recycling worker to abort in-flight CUDA work`, `Task <id> cancelled mid-flight; recycling worker`.
- TaskManager: `Cancelled task <id> …`, `Cancelled in-flight workflow <id> …`, `Removed cancelled task <id> from N queue position(s)`, `Dropping cancelled/terminal task <id> from … queue`.

## Verification

Validated on the 8×4090 `.39` node (`ai-16-39`, `v1` profile).

- **Offline** (`scripts/test_cancel_logic.py`): drives the real `TaskManager` (and `TaskManagerScheduler.wait_unless_cancelled`) against an in-memory async-redis fake — 24/24 checks covering pending-cancel queue removal, dequeue-skip of terminal and base-cancelled tasks, worker-queue removal, the workflow-parent branch, and the orphan-close.
- **Live in-flight** (`scripts/test_cancel.py --mode inflight`): submits a deliberately slow evaluation, waits until `{id}_kernel` is `processing`, cancels the parent, and observes the parent end `failed` / `"Task cancelled"`. Cancel→return latency was ~0.2–0.5 s and total submit→return ~5.7–11.1 s, against a 300 s task `timeout`. Worker logs show the full chain: `processing … _kernel` → marker detection → subprocess recycle → cancelled-result write, with no normal completion line for that id.
- **Cancel during CPU compile**: cancelling while `{id}_compile` is `processing` returned the parent `"Task cancelled"` in ~0.3 s.
- **Status**: `/status/{id}` returned `failed` with `error_message: "Task cancelled"`; a normal `/evaluate` after cancellation still returned `completed`/`correct` (no leak or wedged worker).

Honest framing of the savings: the ~124 s "natural runtime" of the test payload is end-to-end and dominated by the CPU compile stage; the GPU stages were ~18.9 s (kernel) and ~15 s (ref). The interrupt saves the remaining GPU stage time plus the rest of the pipeline, not "124 s of GPU execution."

## Known gaps

- **CPU compile is not preemptively killed.** `cpu_worker` runs `toolkit.evaluate()` in-process with no cancellation watcher, so a cancel during compile returns to the caller promptly (via `wait_unless_cancelled`) but the compile itself finishes on its own. Compile is short and CPU-bound, so this is wasted CPU rather than a correctness issue.
- **Watcher latency.** A running GPU task is interrupted within roughly `cancel_poll_interval_sec` (~1 s), not instantly.
- **Single-store assumption.** Markers live in the shared Redis the API and workers already use; the design assumes the cancelling API instance and the executing worker see the same Redis (true for the current single-primary deployment, including the legacy-prefix read path).
- **Best-effort marker writes.** Marker set/remove failures are logged and swallowed; a lost marker degrades to "cancel had no effect," never to running the wrong task.
