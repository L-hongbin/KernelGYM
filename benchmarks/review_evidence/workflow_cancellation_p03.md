# P03: cancellation admission fences and frozen-child business completion

## Source changes

Explicit DELETE now atomically persists a separate `{prefix}:cancelled:{id}` tombstone, including when the task does not exist yet. Cancellation is idempotent; the HTTP acknowledgement means the fence is recorded, not that GPU execution has exited. Same-ID POST is rejected with 409 even with `force_refresh=true` or a cached result. Normal uncancelled POST/cache/force-refresh behavior remains compatible. Parent IDs, generation-scoped compile/kernel/ref IDs, stage aliases, and submissions linked by `base_task_id` are checked. Tombstones have no TTL and are not deleted by result/ephemeral cleanup: an unknown delayed submission has no bounded arrival time. Redis loss/flush is outside this guarantee; persistence must be configured separately.

The same Lua admission guard is used at enqueue, GPU claim, CPU/GPU execution start, and requeue. Cancellation and publishing a pending terminal result are atomic. Running/execution-fenced/frozen children are not finalized by the control plane; their worker claim and inflight entry remain owned by the safe-reap path. Cancellation persistence failures propagate instead of being acknowledged as success.

Parent reconciliation now inspects registered/planned children for cancellation, frozen claims, and quarantine of a bound worker. It publishes an explicit parent `failed` / `SYSTEM_ERROR` infrastructure result and stops subsequent workflow stages. It does not write a normal result for the frozen child, clear the claim, release inflight ownership, or lift quarantine. The existing quarantine reader gained an opt-in read-only mode, used by business reconciliation so it neither repairs latch replicas nor acquires the physical GPU recovery lock. Durable worker-alias latches remain readable after heartbeat expiry.

## Verification

```bash
.venv/bin/python -m pytest tests/server tests/kernelbench/workflow tests/workers/test_gpu_quarantine_gate.py tests/workers/test_worker_monitor.py tests/deployment/test_service_cli.py --disable-warnings --maxfail=1 --junitxml=/tmp/kernelgym-p03-pytest.xml
.venv/bin/python scripts/test_cancel_logic.py --disable-warnings --maxfail=1
.venv/bin/python -m pytest tests/server/test_workflow_lifecycle.py::test_quarantined_child_ends_http_wait_without_releasing_containment -s --disable-warnings
.venv/bin/ruff check kernelgym/server/workflow_lifecycle.py kernelgym/server/task_manager.py kernelgym/server/scheduler.py kernelgym/server/api/server.py kernelgym/utils/gpu_quarantine.py tests/server/test_workflow_lifecycle.py tests/server/test_task_manager_resource_queues.py
git diff --check
```

Regression: **351 passed, 4 skipped, 112 warnings in 21.12 seconds**. Maintained offline cancellation entrypoint: **109 passed, 110 warnings in 7.79 seconds**. The lifecycle suite now contains **51 cases**, 49 using a disposable real Redis server and two using mocked HTTP for the live probe. Lua fences are exercised against actual Redis transactions, not only a Lua-emulating fake. Ruff and whitespace checks passed.

Coverage includes:

- Parent lease renewal across four accelerated old-TTL windows (0.3 seconds per window, 1.2 seconds total), followed by successful status query and cancellation; this is an accelerated lifetime test, not a literal multi-minute endurance test.
- Cancel-before-POST and delayed compile/kernel/ref enqueue rejection, both ordinary and force-refresh; no queued GPU work or accepted parent is created.
- Parent and direct-child cancellation across all three stages in both pending and processing states (12 combinations); running claim fields remain intact and the parent terminates within a one-second test bound.
- Persistent tombstones after ephemeral cleanup, old child aliases, submit/cancel races, cancellation storage errors, and rejection of cancelled cache hits.
- Quarantined and frozen children, each with and without a worker heartbeat record; parent POST returns within a two-second test bound, no ref stage starts, and frozen/active execution claims, inflight entries, and quarantine remain intact.
- Read-only quarantine lookup with durable latch present and Redis replicas missing; no physical recovery lock is acquired and no replica is rewritten.
- Existing duplicate-POST single-owner behavior, normal cached responses and force-refresh generation isolation, queue-inclusive deadlines, parent terminal CAS, and normal KernelBench response shapes.

## Representative observed output

Reviewed the historical real H100 response in `benchmarks/review_evidence/deployed_return_detail_correctness_gate_h100_20260909.json`: default mismatch evaluation returned HTTP 200, `status=completed`, `compiled=true`, `correctness=false`, and `error_code=CORRECTNESS_ERROR`. It was not rerun on a live GPU. The normal-response compatibility tests continue to preserve that distinction between execution completion and correctness.

The isolated frozen-child test on 2026-09-14 produced this excerpt (unmodified field values, nonessential result fields omitted):

```json
{
  "result": {
    "task_id": "parent",
    "status": "failed",
    "error_code": "SYSTEM_ERROR",
    "error_message": "Infrastructure failure: child parent_kernel_a50f141df9eb42f8883d6e2ad1f4322c frozen: containment uncertain",
    "completed_at": "2026-09-14T13:24:00.385874"
  },
  "fence": "frozen",
  "child_result": null,
  "inflight_preserved": true
}
```

All four trace variants passed in 3.48 seconds, including test startup. The tests assert exact preservation of the child hash and inflight list after parent completion, plus a still-present quarantine latch.

## Deployment boundaries

No deployed service was restarted; no production Redis, GPU context, worker process, or quarantine state was modified. Tests start and stop only their own Unix-socket-only Redis (`--port 0`, no persistence) and isolate durable latches in temporary directories. This patch has not been deployed or committed. API and worker-side TaskManager code must be upgraded together for the admission guarantees; old binaries that do not check tombstones are not made safe by updating the API alone. Already-running CPU compilation is not force-killed. Hardware containment and multi-host/load behavior were not exercised here.
