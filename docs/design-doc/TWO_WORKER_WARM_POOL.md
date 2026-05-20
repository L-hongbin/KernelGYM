# Two-Worker Warm Pool

Status: current design
Date: 2026-05-20

This document describes the current GPU subprocess warm-pool design used by the reward worker. The default configuration is a two-worker pool per GPU: one subprocess can run the current task while one already-initialized subprocess remains available as a warm spare. This is not two-task GPU concurrency; the current `GPUWorker` consumes tasks serially, and the spare exists to hide worker recycle latency.

`WORKER_POOL_SIZE=2` is now the deployed `v1` profile setting and matches the documented default. The race that previously grew the pool past its configured size during concurrent recycle and emergency paths is closed by the `pending_replacements` bookkeeping described under [Recycling](#recycling).

## Goals

- Keep CUDA and PyTorch faults isolated inside subprocesses rather than the long-lived GPU worker process.
- Avoid paying torch import, CUDA initialization, and first allocation startup cost on every task.
- Recycle subprocesses aggressively enough to reduce CUDA context, allocator, and extension-state leakage between submissions.
- Keep the next task from waiting for the previous subprocess shutdown and replacement path.

## Configuration

The controlling settings are in `kernelgym/config/settings.py`:

| Setting | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `worker_pool_size` | `WORKER_POOL_SIZE` | `2` | Number of persistent subprocess workers created per GPU worker. With the current serial GPU worker loop, this means one active subprocess plus one warm spare. |
| `max_tasks_per_worker` | `MAX_TASKS_PER_WORKER` | `1` | Number of tasks a subprocess may return before it is marked unavailable and recycled. |

The default `2 x 1` mode is the important behavior: keep two warm CUDA subprocesses around, but retire each subprocess after one task.

## Startup

Each `GPUWorker` creates one `SubprocessWorkerPool` after API registration and GPU discovery. The pool starts `worker_pool_size` `PersistentWorker` processes immediately. Each subprocess uses `spawn`, imports torch/toolkit/backend code, initializes CUDA on the assigned device, performs a tiny CUDA allocation, synchronizes, and then reports `READY` to the parent.

The parent only adds a subprocess to `idle_workers` after the ready message arrives. Startup therefore pays the CUDA initialization cost before live traffic reaches the subprocess.

## Task Flow

1. The GPU worker receives one task from Redis.
2. `SubprocessWorkerPool.execute_task()` takes the first idle `PersistentWorker`.
3. The worker is moved from `idle_workers` to `busy_workers`.
4. The task payload is sent over the subprocess queue.
5. The subprocess executes toolkit evaluation and returns the result.
6. The parent records pool timing metadata such as `pool_idle_wait_s`, `pool_execute_s`, `pool_restart_s`, `pool_return_s`, and `pool_total_s`.
7. If the subprocess is still eligible for reuse, it is returned to `idle_workers`; with the default `MAX_TASKS_PER_WORKER=1`, it is instead marked unavailable and recycled.

Because the outer `GPUWorker` processes tasks serially, this design is a latency-hiding spare mechanism rather than a throughput mechanism for parallel GPU execution.

## Recycling

Subprocess recycling is deliberately split into a fast path and a slow path.

The fast path runs under the pool lock. It removes the old subprocess from `workers`, `idle_workers`, and `busy_workers` immediately, then bumps `pending_replacements` only if `len(workers) + pending_replacements < pool_size`. After this removal, any already-idle spare can be handed to the next task without waiting for the old process to exit.

The slow path runs in a daemon background thread. It sends graceful shutdown commands, escalates through terminate and kill if needed, reaps the old process, waits briefly for the GPU driver to reclaim resources, creates a replacement `PersistentWorker`, and registers the replacement back into `idle_workers` through the asyncio event loop. `_register` decrements `pending_replacements` and refuses to append the new worker if the pool has meanwhile been filled — the extra worker is shut down instead.

This is the reason for the two-worker default. With only one subprocess, every recycle would put the GPU worker on the cold replacement path. With two subprocesses, the pool can promote the warm spare while replacement happens in the background.

### Capacity invariant

The single invariant that every recycle, register, and emergency path preserves is:

```
len(workers) + pending_replacements <= pool_size
```

Concretely:

- `_restart_worker` increments `pending_replacements` only when the inequality would still hold afterwards; otherwise it shuts down the recycled worker synchronously and schedules no background thread.
- The background `_register` decrements `pending_replacements`; if the pool reached `pool_size` while the replacement was in flight, the newly created worker is shut down rather than appended.
- Emergency recovery in `_get_idle_worker` starts only when `len(workers) == 0` and `pending_replacements == 0`; it reuses the same accounting and discards itself if capacity is no longer needed.

Tests in `tests/test_subprocess_pool.py` pin this invariant under all the recycle interleavings that previously grew the pool.

## Failure Handling

CUDA and profiler errors are treated as subprocess-fatal. The subprocess returns an error payload with `worker_exiting=True`, the parent marks the `PersistentWorker` unavailable, and the pool starts the same recycle path.

Timeouts are not retried. On timeout, the pool restarts the stuck subprocess and returns a timeout failure so the GPU queue is not blocked by repeated attempts.

Non-timeout runtime failures may be retried by `SubprocessWorkerPool.execute_task()` up to its retry limit, after restarting the failed subprocess.

If the pool loses all workers, `_get_idle_worker()` has an emergency recovery path that tries to create a new worker after a short delay. This is a last resort; normal operation should keep at least one warm subprocess available.

## Observability

Pool behavior is visible through result metadata and pool stats:

- `wg_pool_idle_wait_s`
- `wg_pool_execute_s`
- `wg_pool_restart_s`
- `wg_pool_return_s`
- `wg_pool_total_s`
- `wg_pool_retry_count`
- `workers_alive`
- `idle_workers`
- `busy_workers`
- `total_workers_restarted`

Useful log markers include `PoolTiming`, `Recycling worker`, `Background spare ready`, and `Worker pool initialized`.

## Tradeoffs

The two-worker warm pool uses more host RAM and GPU context memory than a one-worker pool. That is intentional: it trades extra resident resources for lower task-to-task latency and better isolation after each submission.

Increasing `WORKER_POOL_SIZE` above `2` only helps if the outer scheduling path can issue overlapping work to the same GPU or if replacement churn is high enough that one spare is insufficient. It also increases resident CUDA context memory. For the current serial GPU worker loop, `2` is the practical default.

Increasing `MAX_TASKS_PER_WORKER` reduces recycle overhead but weakens per-task isolation and can allow allocator, extension, profiler, or CUDA state to survive across submissions. The current reward-only default keeps this at `1`.

## Verification

The deployed `v1` setup has been verified on the 8x4090 `.40` node:

- Idle steady state shows exactly two Python CUDA contexts per GPU and no `Pool has no workers!`, `Emergency worker`, or `workers > 2` entries across multi-hour replays.
- A 706-row Qwen3.6 27B turn-2 replay and a 704-row turn-3 replay both completed end-to-end with the pool absorbing roughly 1500 total worker recycles, all bounded at `pool_size=2`.
- On the turn-3 replay with `--run-performance`, the per-row `replay_speedup / original_score` ratio has geometric mean `0.997` and median `1.000` across the rows where the original rollout also measured a positive speedup, so the warm-pool deployment reproduces the original acceleration ratios faithfully.

## Known Gaps

- The design does not provide intra-GPU task parallelism in the current worker loop.
- Replacement creation still has a cold-start cost; it is just moved off the critical path when a spare is available.
- If both subprocesses become unavailable before a replacement is ready, the next task can still wait for emergency or background recovery.
- GPU memory pressure depends on the resident cost of multiple initialized CUDA contexts. On the 24 GiB 4090s in the `v1` node this manifests as a small number of additional kernel timeouts (~12 of 706 turn-2 rows) versus the prior single-worker deployment.
