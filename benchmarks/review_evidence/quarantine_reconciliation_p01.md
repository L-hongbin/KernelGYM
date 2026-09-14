# P01: quarantined worker generation reconciliation

## Scope and safety

Implemented in `kernelgym/worker/worker_monitor.py` and `scripts/manage_gpu_quarantine.py`. Reconciliation removes only a proven-drained, unchanged local process generation. It does not clear quarantine, release frozen claims, send termination signals, or start a worker. Manual `clear` retains its existing confirmation gates and points operators to the shared `reconcile` command when maps remain.

The monitor scans process maps before health/restart processing, including workers without heartbeat keys and workers with queued restarts. Local ownership, PID start ticks, PGID, SID, surviving session members, and generation CAS are required. Incomplete legacy identity, foreign/conflicting ownership, PID reuse, live descendants, inspection errors, and CAS races retain the map. A zombie-only session does not excuse a still-live known PGID outside that snapshot.

## Verification

Run from the KernelGYM repository root using its existing `.venv`; the system Python lacks `redis` and was not used for the completed regression run.

```bash
.venv/bin/python -m pytest tests/workers/test_worker_monitor.py tests/workers/test_gpu_quarantine_gate.py tests/deployment/test_service_cli.py --disable-warnings --junitxml=/tmp/kernelgym-p01-pytest.xml
.venv/bin/ruff check kernelgym/worker/worker_monitor.py scripts/manage_gpu_quarantine.py tests/workers/test_worker_monitor.py
.venv/bin/python scripts/manage_gpu_quarantine.py reconcile --help
```

Regression result: **196 passed, 95 warnings, 15.08 seconds**. Ruff checks passed. Coverage includes incomplete identity fields, local-host verification, reused/moved leaders, separate child PGIDs, permission/inspection failures, changes to each of the four CAS identity fields, Redis failure before CAS, zombie handling, persistent/non-persistent monitor paths, queued restarts, CLI retention/success, host/device rejection, and preservation of signal handlers in CLI mode.

## Real process evidence

Reviewed existing deployment evidence in `benchmarks/review_evidence/kernelgym_restart_gpu_ownership_h100_20260908.json`, which records an outer worker and a separate warm-pool CUDA child. This illustrates why leader absence alone is insufficient; no deployment operations from that historical artifact were executed.

Separately ran a disposable CPU-only process in a new Linux session, authenticated its actual `/proc` identity, and invoked the production reconciliation method before and after its normal stdin-triggered exit. Redis was an isolated in-memory test double, not the running service. The captured output was:

```json
{
  "hostname": "ai-11-229",
  "recorded_identity": {
    "pid": 2362384,
    "start_ticks": "118744167",
    "state": "R",
    "process_group": 2362384,
    "session_id": 2362384
  },
  "reconciled_while_live": false,
  "process_exit_code": 0,
  "reconciled_after_exit": true,
  "process_map_absent": true,
  "quarantine_unchanged": true,
  "claim_unchanged": true,
  "restart_queue_empty": true,
  "redis": "isolated in-memory FakeRedis, not a service"
}
```

The live-generation attempt emitted `Retaining worker p01_isolated_cpu_evidence process map during reconciliation: RuntimeError: recorded generation is still live (state=R)`. The owned process then exited with code 0 and was reaped; no live worker was signalled.

## Operator command and boundaries

```bash
.venv/bin/python scripts/manage_gpu_quarantine.py reconcile --worker-id WORKER_ID --device cuda:0 --hostname RECORDED_HOST
```

Run on the recorded host in the worker's PID namespace with complete `/proc` visibility. The command only reconciles the specified worker map; repeat for other retained aliases if necessary. Missing identity/ownership requires investigation, not an absent-PID override. The existing documented limitation for deliberate descendants escaping their session remains; this change does not establish a GPU-wide containment proof or replace the unsafe-orphan confirmation gate.

No service was restarted, no real Redis records or safety latches were cleared, and no GPU recovery was attempted. Live deployment and real-Redis race testing were not performed; CAS races above use the existing test-double implementation of the generation comparison.
