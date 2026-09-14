# P02: parent workflow lifecycle verification

## Implemented behavior

Parent acceptance now atomically creates a durable pending record, a workflow generation, a Redis-time end-to-end deadline, planned compile/kernel/reference child IDs, and a renewable owner lease. Same-content duplicate POSTs join one owner across TaskManager instances; conflicting content returns HTTP 409. The synchronous POST response contract remains intact. A disconnected HTTP waiter does not cancel shared execution; DELETE provides explicit cancellation.

The controller runs under the remaining parent deadline. A watchdog plus status/wait reconciliation converts expired deadlines or lost owner leases to terminal results. Parent terminal publication is generation-fenced and atomic with cancellation, active-index removal, queued/unstarted-child cancellation, and retention TTLs. It cannot release running or frozen GPU claims. Generation-scoped child IDs and shared Lua admission checks prevent stale submissions, CPU/GPU dispatch, and old parent completions from affecting the next workflow generation.

Settings: `WORKFLOW_TIMEOUT=1800` seconds, overridable by `workflow_timeout` on `/evaluate` or inside `/workflow/submit.payload`; `WORKFLOW_LEASE_SECONDS=60` seconds. Existing `DEFAULT_TIMEOUT` and ordinary per-stage timeout behavior are unchanged. Active parent task records do not receive terminal retention TTLs until their terminal transaction.

## Verification commands and results

```bash
.venv/bin/python -m pytest tests/server tests/kernelbench/workflow tests/workers/test_gpu_quarantine_gate.py tests/workers/test_worker_monitor.py tests/deployment/test_service_cli.py --disable-warnings --maxfail=1 --junitxml=/tmp/kernelgym-p02-pytest.xml
.venv/bin/python -m pytest tests/server/test_workflow_lifecycle.py::test_queue_time_counts_towards_parent_deadline -s --disable-warnings
.venv/bin/python scripts/test_cancel_logic.py --disable-warnings --maxfail=1
.venv/bin/ruff check kernelgym/server/workflow_lifecycle.py kernelgym/server/task_manager.py kernelgym/server/scheduler.py kernelgym/server/api/server.py kernelgym/server/api/models.py kernelgym/workflow/kernelbench.py kernelgym/config/settings.py tests/server/test_workflow_lifecycle.py tests/kernelbench/workflow/test_split_affinity.py tests/server/test_speed_test_endpoint.py
git diff --check
```

Final regression: **323 passed, 4 skipped, 112 warnings in 18.28 seconds**. Ruff and whitespace checks passed. The new lifecycle file contains 23 passing cases: 21 use an actual temporary Redis server, not a Lua-emulating test double; two check live-probe child-ID discovery using mocked HTTP. The server accepts only a temporary Unix socket (`--port 0`), disables persistence, and is terminated by its owning fixture. No production Redis connection is used. The maintained offline script entrypoint also passed all 81 queue/lifecycle cases in 5.10 seconds; it now delegates to pytest rather than retaining a second marker-only implementation.

Coverage includes early pending/running status, lease renewal, multi-manager deduplication, request-content conflict, queued deadline expiry, lease loss, parent cancellation, real Lua CPU/GPU admission gates, unchanged frozen claims, cancellation after marker expiry, late parent completion, force-refresh generation isolation, missing terminal result handling, old-version active-marker protection, watchdog recovery, controller errors/shutdown, HTTP waiter disconnect, retention TTLs, and generation-CAS ephemeral cleanup. An ASGI HTTP test checks GET status and DELETE while the synchronous POST remains in flight. A real KernelBench controller test consumes synthetic worker feedback through the actual TaskManager queue/result path and preserves the parent ID and nullable result fields; it does not execute generated code or CUDA.

## Representative observed trace

Reviewed `benchmarks/review_evidence/deployed_return_detail_correctness_gate_h100_20260909.json`, which records real H100 `/evaluate` responses with correctness mismatches. That historical deployment was not repeated or modified.

The new isolated queue-deadline test submits a compile child with no worker consuming it and sets `workflow_timeout=0.1`. Its actual parent status output was:

```json
{
  "task_id": "parent",
  "status": "timeout",
  "submitted_at": "2026-09-14T12:20:08.733243",
  "started_at": "2026-09-14T12:20:08.735994",
  "completed_at": "2026-09-14T12:20:08.835080",
  "error_message": "Workflow end-to-end deadline exceeded",
  "workflow_generation": "cf471d5dfaa644318ff37b2f174b2f90",
  "workflow_deadline": 1789388408.834,
  "children": {
    "compile": "parent_compile_cf471d5dfaa644318ff37b2f174b2f90",
    "kernel": "parent_kernel_cf471d5dfaa644318ff37b2f174b2f90",
    "ref": "parent_ref_cf471d5dfaa644318ff37b2f174b2f90"
  }
}
```

The parent result contained `error_code=TIMEOUT_ERROR`, `compiled=false`, and `correctness=false`; the test also asserted that the CPU resource queue was empty. The standalone trace test passed in 2.77 seconds, including process/import setup.

## Deployment and remaining boundaries

No deployed API/worker service was restarted; no live Redis state, quarantine latch, GPU context, or frozen claim was cleared. Deployment requires the updated API and worker-side TaskManager code so both admission gates and cancellation polling honor workflow generations. Old-version markers are rejected rather than overwritten; this is not a guarantee of safe arbitrary mixed-version writes.

Already-running CPU compilation is not force-killed. The parent returns terminal, prohibits later GPU stages, and ignores the old CPU result. Running GPU cancellation remains the existing worker/subprocess safe-reap path; parent cancellation never proves CUDA containment itself. Owner loss is a controlled failure, not automatic re-execution of potentially active kernels. GPU hardware, production load, and multi-host deployment were not exercised by these local checks.
