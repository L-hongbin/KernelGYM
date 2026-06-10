# Runtime Coordination Storage

Status: proposed design
Date: 2026-06-10

This document describes the storage split that should replace the current single Redis keyspace for worker coordination, task scheduling, and long-lived result history. The immediate incident behind this proposal was a healthy `.39` primary and `.40` worker node being reported as `gpu_workers_online: 0/0` because `/workers/status` and `check_node.sh` depended on Redis keyspace scans after historical task/result keys had grown to roughly 180k keys.

## Current Problem

The current Redis layout stores runtime coordination data and historical/cache data under the same Redis prefix:

| Key family | Meaning | Lifecycle today |
| --- | --- | --- |
| `kernelgym:worker:<worker_id>` | Worker heartbeat/status/current task | Refreshed by workers; worker hashes have short TTL in some worker-side paths but server-side registration/heartbeat does not consistently set one. |
| `kernelgym:queue:resource:{gpu,cpu}` | Pending resource queues | Drained by workers. |
| `kernelgym:queue:worker:<worker_id>` | Worker-affined queues | Drained by matching workers. |
| `kernelgym:task:<task_id>` | Submitted task payload and mutable task status | Created on submit, updated on dispatch/completion/failure, not deleted on normal completion. |
| `kernelgym:result:<task_id>` | Completed result/error payload and optional request hash | Created on completion/failure, not deleted on normal completion. |
| `kernelgym:cancel:<task_id>` and `kernelgym:workflow:<base_id>` | Runtime cancellation/workflow markers | Short TTL. |
| `kernelgym:retry_*` and error-pattern keys | Retry/error tracking | Short TTL. |

This is not a semantic correctness bug by itself; Redis can store all of these keys. The bug is that runtime query paths used keyspace scans to rediscover small active sets such as workers and processing tasks. Redis `SCAN MATCH kernelgym:worker:*` still iterates the database keyspace and filters matches, so historical task/result accumulation makes worker status slower even though worker keys have a different prefix.

## Why Completed Tasks Accumulated

Completed tasks were retained because the task manager treats Redis task/result hashes as both status records and a result cache.

On submit, `TaskManager.submit_task()` writes `kernelgym:task:<task_id>` with the serialized task payload, status, priority, and timestamps, then enqueues the task id. On normal completion, `TaskManager.complete_task()` updates the task hash with terminal status and `completed_at`, then writes the full result payload to `kernelgym:result:<task_id>`. It does not delete either key and does not set an expiry. `TaskManager.get_task_result()` later reads `kernelgym:result:<task_id>` and validates `request_hash` when present, so these records also act as a server-side cache for repeated requests.

There are targeted cleanup paths, but not a general retention policy. `force_refresh` deletes the specific task/result pair before resubmitting. Cancellation and workflow markers use TTLs. Retry/error-pattern records use TTLs. Normal completed task/result records do not. A benchmark run can create multiple subtask records per row, such as compile, kernel, and reference timing tasks, so repeated benchmark/evaluation runs naturally produce tens or hundreds of thousands of task/result keys.

The design mistake is therefore not “we accidentally stored task status”; task/result retention was useful for result lookup and cache reuse. The mistake is allowing runtime coordination queries to depend on scans over the same keyspace that contains unbounded history.

## Goals

- Keep worker health, node membership, queues, leases, cancellation, and active task scheduling fast regardless of historical result volume.
- Make all monitoring and scheduler request paths O(active workers + active tasks), not O(all Redis keys).
- Preserve result caching and post-run inspection, but move it behind explicit retention and storage boundaries.
- Allow safe migration from the current Redis layout without dropping in-flight work or invalidating useful historical results.
- Keep a low-frequency reconciliation path for repairing indexes, but keep it out of normal API/check-node request paths.

## Non-Goals

- This design does not change CUDA execution semantics, worker pool behavior, or benchmark scoring.
- This design does not require deleting all historical results immediately.
- This design does not require replacing Redis for queues; Redis remains a reasonable coordination store.

## Target Architecture

Split storage into two planes.

### Coordination Store

The coordination store is the live cluster control plane. It contains only data needed to schedule current work and report current health. It may be Redis DB 0, a separate Redis instance, or a distinct prefix with strict TTL and indexing rules.

Recommended key families:

| Key | Type | Purpose |
| --- | --- | --- |
| `coord:workers` | set | Authoritative worker id index. |
| `coord:worker:<worker_id>` | hash | Worker device, node, hostname, status, current task, stats, last heartbeat. |
| `coord:nodes` / `coord:nodes_by_host` | hash | Node id allocation and hostname mapping. |
| `coord:queue:resource:{gpu,cpu}` | list/stream | Pending resource queues. |
| `coord:queue:worker:<worker_id>` | list/stream | Worker-affined task queue. |
| `coord:task:active:<task_id>` | hash | Active or recently active task scheduling metadata only. |
| `coord:tasks:processing` | set | All processing task ids. |
| `coord:tasks:processing:{gpu,cpu}` | set | Processing task ids by resource. |
| `coord:cancel:<task_id>` | string | Cancellation marker with TTL. |
| `coord:workflow:<base_id>` | string | In-flight workflow marker with TTL. |

Worker state should be lease-based: heartbeats refresh the worker hash TTL and index membership. A worker is online only if its lease is fresh. Offline status may be written for observability, but scheduling must not rely on stale permanent worker hashes.

Task scheduling state should be active-state-based: submit adds `coord:task:active:<task_id>` and queue membership; dispatch moves the task id into `coord:tasks:processing` and a resource-specific processing set; completion/failure removes it from active/processing indexes after writing result history to the result store. If short-term status introspection is needed, keep completed active task records for a small TTL, not forever.

### Result Store

The result store contains historical data and reusable cache records. It may be Redis DB 1, SQLite/Postgres, object storage, or Redis with explicit retention. It is not used for worker discovery, queue status, scheduling, or liveness.

Recommended key families:

| Key | Type | Purpose |
| --- | --- | --- |
| `result:by_task:<task_id>` | hash/blob | Full result/error payload keyed by task id. |
| `result:by_request:<request_hash>` | hash/blob | Deduplicated result cache keyed by canonical request hash. |
| `result:task_to_request:<task_id>` | string | Optional mapping for task-id lookup. |
| `result:metadata:<run_id>` | hash/list | Optional benchmark/evaluation run metadata. |

Long-lived cache keys need explicit policy: TTL, max count, dataset/run-scoped export, or durable archival outside Redis. The default should not be unbounded Redis retention unless Redis is intentionally sized and monitored as a cache/database for that purpose.

## Request Path Rules

Runtime API endpoints must not use `KEYS` or unbounded `SCAN` in the request path. This includes `/workers/status`, `/queue/status`, `/health`, and `check_node.sh`.

Expected implementations:

- `/workers/status`: `SMEMBERS coord:workers`, then bounded `HGETALL coord:worker:<id>` for those ids.
- `/queue/status`: `LLEN` resource queues plus `SCARD coord:tasks:processing:{gpu,cpu}`.
- Worker selection: read indexed worker hashes and leases, not keyspace scans.
- Cancellation: direct key lookup and active indexes, not task key scans.
- Reconciliation: a background job may scan periodically to repair indexes and remove dangling ids, but it must not block request handlers.

## Migration Plan

1. Add indexes in the current Redis prefix while preserving existing keys: `kernelgym:workers`, `kernelgym:tasks:processing`, and resource-specific processing sets.
2. Update writer paths first: worker register/heartbeat/unregister maintain `kernelgym:workers`; task submit/dispatch/complete/fail maintain active and processing indexes.
3. Update read paths: `/workers/status`, `/queue/status`, scheduler helpers, and `check_node.sh` read indexes first and only use scans as a one-time fallback when an index is missing.
4. Backfill indexes from current Redis once per deployment using a bounded maintenance command, not inside a hot request.
5. Add retention for completed task hashes. Keep only short-lived status records in the coordination store after completion; write full results to the result store.
6. Split Redis DBs/prefixes or instances. The coordination store should remain small enough that `DBSIZE` is proportional to live cluster state, not historical benchmark volume.
7. Add observability: key counts by family, index cardinality, stale worker ids, active task count, result cache count, and oldest unexpired coordination task.

## Invariants

- Every worker hash with a fresh lease is present in `coord:workers`.
- `coord:workers` may contain stale ids, but readers must remove ids whose worker hash is missing or expired.
- Every processing task appears in `coord:tasks:processing` and exactly one resource-specific processing set.
- Completed/failed/timeout tasks are removed from processing sets before or atomically with result persistence.
- Result cache lookup never participates in live scheduling decisions.
- Runtime health must remain bounded by active state size.

## Operational Policy

Use Redis as a queue/lease/active-state store first. Treat it as a result cache only with an explicit retention policy. For benchmark evidence and durable audit trails, prefer append-only artifacts, SQLite/Postgres, or object storage over unbounded Redis hashes.

For current deployments, the minimum safe policy is:

- Maintain `kernelgym:workers`.
- Maintain processing-task indexes.
- Stop using keyspace scans in `/workers/status`, `/queue/status`, and `check_node.sh`.
- Set TTL on terminal `kernelgym:task:*` and `kernelgym:result:*` records. The current implementation exposes this through `TERMINAL_TASK_TTL_SEC` and `TERMINAL_RESULT_TTL_SEC`, defaulting to 24 hours; values `<=0` keep records indefinitely.
- Add a periodic export job if terminal results need to outlive the Redis TTL for audit or benchmark evidence.

## Open Questions

- Should result cache identity be `task_id`, `request_hash`, or both? Current code validates `request_hash` inside `kernelgym:result:<task_id>`, but task ids are often generated per workflow/subtask.
- How long should terminal task status remain queryable through `/tasks/{task_id}` after completion?
- Should historical results remain in Redis with TTL, or move to a durable local artifact/SQLite/Postgres store?
- Should worker leases rely exclusively on Redis key expiry, or keep explicit `last_heartbeat` for human-readable diagnostics as well?
