# Server Result Cache Guard

## Problem

KernelGym stores completed evaluation results in Redis under `<redis_key_prefix>:result:<task_id>`. This is useful for idempotent retries and `/results/{task_id}` lookups, but it is unsafe when a client reuses a `task_id` for a different evaluation payload. The old behavior returned the cached result solely by `task_id`, so replay jobs could silently consume stale results from another run.

## Design

The server now treats `task_id` as a lookup handle, not as sufficient proof that two requests are identical. Before `/evaluate` or `/workflow/submit` reuses a cached result, the server computes a canonical request hash from the workflow name and semantic payload fields. The cached Redis result must contain the same `request_hash`; otherwise the server ignores the cache and reruns the workflow.

The request hash intentionally excludes non-semantic identity and provenance fields: `task_id`, `force_refresh`, status/progress fields, and provenance fields such as `metadata`, `run_id`, `turn_id`, `model_id`, `line_index`, and `output_index`. These fields are useful for tracing but must not define cache identity. Semantic inputs such as `reference_code`, `kernel_code`, toolkit/backend settings, correctness/performance trial counts, timeout, entry point, and workflow name remain part of the hash.

## Legacy Entries

Redis entries written before this change do not have `request_hash`. When `/evaluate` or `/workflow/submit` sees such an entry while checking a concrete request, it treats the entry as unsafe for reuse and reruns the workflow. Read-only endpoints such as `/results/{task_id}` still return legacy entries because they are explicitly asking for historical task data.

## Replay Policy

Replay clients should still generate content-addressed `task_id` values and default to `force_refresh=True` when validating the current reward service. The server guard is a defense-in-depth layer: it prevents stale result reuse if a future client accidentally reuses a `task_id`, but it does not replace good replay task ID construction.

## Current Implementation

- `kernelgym/server/request_hash.py` computes the canonical request hash.
- `_execute_workflow()` passes that hash to `TaskManager.get_task_result()` before cache reuse and to `TaskManager.complete_task()` when storing new results.
- `kernelgym/server/task_manager.py` stores `request_hash` as a separate Redis hash field next to `result` and `completed_at`.
- `TaskManager.get_task_result(task_id)` remains backward compatible for status/result lookup paths; hash validation is only enabled when `expected_request_hash` is provided.
