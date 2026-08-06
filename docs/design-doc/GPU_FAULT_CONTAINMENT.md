# GPU Fault Containment for Untrusted CUDA Kernels

## Scope and limits

KernelGYM runs generated CUDA code in Docker on a host driver shared by every container. A subprocess boundary can contain Python state and most CUDA-context failures, but it cannot make a device-global or driver-global GPU failure impossible. Without a VM or MIG, the safe response to an unresolved physical-device fault is to stop dispatching to that GPU, reap every known CUDA context, perform one fresh-context probe, and require manual recovery if the probe fails.

This design does not change the container startup command and never invokes automatic `nvidia-smi --gpu-reset`. Production keeps `MAX_TASKS_PER_WORKER=1`: each evaluation task owns one CUDA subprocess, while a second subprocess is initialized ahead of time as the warm spare and replacement construction happens in the background. The isolation boundary is one evaluation task, not one individual CUDA launch inside that evaluation.

The persistent subprocess is a fault-containment boundary for generated kernels, not a complete hostile-Python security sandbox. Candidate code shares an interpreter with toolkit code and can use frame inspection, `ctypes`, background threads, or a fork followed by `setsid()` to evade ordinary Python-level restrictions. The controls below fail closed for detected lifecycle uncertainty, but preventing deliberate same-UID escape requires a stronger broker, seccomp, or VM boundary.

## Normal task commit barrier

A CUDA launch is asynchronous. Returning a Python result only proves that launches were enqueued; an illegal access may surface at the next unrelated CUDA call. Before a subprocess publishes success, it performs a strict task-boundary synchronization. A configuration that deliberately allows reuse first performs strict allocator/context cleanup and then a final strict synchronization. The low-level `_cuda_setDevice` and `_cuda_synchronize` callable objects and the complete commit/error operations are captured before candidate dispatch, so ordinary module monkeypatching cannot replace the barrier. No CUDA cleanup call for that task follows successful publication.

“Forced synchronize” therefore means waiting at the result boundary until that task's queued GPU work has really completed. With the production per-task setting, the used subprocess is then reaped, but the already-initialized spare remains available and its replacement is constructed in the background. The latency cost includes the completion wait and the proof that the retired process group is gone; it does not put fresh CUDA-context construction on the next task's critical path while a spare remains available.

Child results no longer cross a `multiprocessing.Queue` pickle boundary. The child accepts only exact JSON primitive containers, encodes a fixed typed envelope with a 64 MiB, 200,000-node, depth-32 bound, and writes raw bytes to a one-way connection. The parent independently validates message kind and schema and strengthens CUDA-fault classification. A candidate-controlled `__reduce__` object is rejected in the child rather than deserialized in the parent.

## Recovery state machine

| Event | Admission | Context action | Probe action |
| --- | --- | --- | --- |
| Healthy task | Open | Reuse/recycle under normal pool policy | None |
| First context-local CUDA fault | `DEGRADED_CHECK` | Synchronously prove the culprit PGID reaped before returning the task; retain at most one pre-fault spare | Construct exactly one fresh validation context |
| Second fault during validation, timeout, cancellation, severe device marker, or topology uncertainty | `SUSPECT` | Force terminate and reap every tracked context; wait for all constructor/reaper tickets | Construct exactly one fresh probe after old contexts are proven gone |
| Fresh probe succeeds | `HEALTHY` | Rebuild capacity from the new generation | Admission reopens |
| Reap cannot be proven or fresh probe fails | `QUARANTINED` | No retry loop and no automatic reset | Manual clear and worker restart required |

The pool uses a hard-recovery epoch and lifecycle tickets so a stale background constructor cannot publish a worker into a newer generation. A Python-caught CUDA fault is published over the raw result channel and then the child waits without making another CUDA call; the parent owns STOP/KILL/reap. Every retiring child, including normal max-task/OOM/profiler recycling, is synchronously proven reaped before its task may return; only construction of the replacement remains asynchronous, so another already-warm spare remains dispatchable. Shutdown and recovery are cancellation-shielded until containment is complete and concurrent recovery waits on the shared shutdown proof. Busy workers are force-reaped during shutdown; idle workers may exit gracefully. An unproven culprit, validation context, retirement, or constructor ticket raises a dedicated unsafe-containment error rather than an ordinary task error.

## Scheduler and persistence gates

GPU dequeue requires an explicit CUDA device, a known hostname, an online and fresh heartbeat, `accepting_tasks=true`, a known `healthy` or `degraded_check` state, and no persistent quarantine. Instead of a destructive `RPOP`, the task manager atomically moves each GPU task into a per-worker Redis inflight list. Every attempt carries a random claim token plus a per-process worker-instance id. Processing, return, recovery, acknowledgement, and terminal publication all compare that exact token, so a late result from an old process cannot complete, fail, remove, or requeue a replacement process's attempt.

The same Lua transaction that changes a GPU attempt to `processing` first marks its recovery state `execution_fenced`, before the task is returned to GPUWorker or any child. A process crash can therefore never make an in-flight untrusted payload ordinarily retryable. Normal same-token terminal commit clears this fence. A path that proves the task never reached a child may release only `execution_fenced`; it cannot release a concurrently upgraded containment freeze.

The terminal commit writes the task state, result record, TTLs, and exact inflight acknowledgement in one Lua transaction. If Redis rejects or loses that commit, the worker closes local admission and retains the recoverable claim instead of publishing an ordinary retry failure. During an unsafe shutdown or task-local containment failure, the current claim is atomically upgraded to `frozen`; routine terminal writers, acknowledgement, recovery, requeue, and `force_refresh` cannot expose or replace it. The worker closes admission and retains the local claim without publishing a terminal result. Only a shutdown caller that has affirmatively proven every child reaped may finalize that same token, or a replacement that sees no quarantine latch and successfully initializes a fresh healthy CUDA pool may release it for retry. If such a safely released frozen claim has a cancellation marker, the same terminal Lua transaction records its failure result and exact-token acknowledgement rather than leaving it indefinitely in `processing`. A concurrent owner change or failure to add the decorative `cancelled_at` field does not abort recovery of later inflight entries.

A replacement worker recovers claims from an earlier worker instance before it dequeues new work. It checks admission before each claim and again afterward; a task that loses the race is conditionally restored to its original direct queue or resource queue without overwriting cancellation or a terminal result. Missing or unreadable health state fails closed. Ordinary submission is an atomic create-if-absent operation, and `force_refresh` is an atomic terminal-only replacement. Pending, processing, tokenized, or frozen work is never deleted by refresh.

The queue-wait monitor uses a non-destructive bounded `LRANGE` snapshot and a Lua compare-and-set for the actual worker queue key. If a stale copy remains in worker A's queue after the task was assigned to worker B, it removes only A's stale list entry and never steals or unassigns B's task.

Quarantine is stored in both non-expiring Redis hashes and atomic JSON safety latches under `logs/safety_latches/`. The durable copy survives this deployment's Redis `NOSAVE` restart behavior and is authoritative for physical-device recovery and notification success. A physical latch applies to replacement worker aliases on the same hostname/device pair. Scope-less, incomplete, or Redis-only legacy latches are normalized fail-closed and materialized under the same device lock before notification; if durable materialization fails, the positive latch still blocks admission and takes an explicitly unlatched best-effort notification path. Clearing a physical latch removes all matching durable and Redis-only aliases without allowing a concurrent normalization read to recreate them.

## Process and shutdown containment

Each inner CUDA worker leads a dedicated PGID while remaining inside the outer GPU worker's session. The supervisor and service CLI authenticate the outer generation with PID start ticks, PGID, and SID. On escalation they repeatedly SIGSTOP every PGID found in the SID until two stable frozen snapshots, SIGKILL only authenticated groups, and require both an empty SID and `ESRCH` for every observed PGID before deleting the Redis process map or starting a replacement.

A missing leader PID alone is never a drain proof. Any scan, D-state/freeze, generation, or drain uncertainty causes physical quarantine, page-user notification, retained process-map ownership, and replacement refusal. The existing startup command and flags are unchanged; monitor restarts use `start_new_session=True` and the default stop grace is the configured worker drain window plus 30 seconds.

Shutdown closes admission locally and in Redis before draining. A still-running task is only marked failed and made retryable after every child CUDA context is proven terminated and reaped. If containment cannot be proven, or the terminal Redis transaction fails, the worker quarantines the GPU and retains the inflight claim for replacement-worker recovery. This prevents work from running concurrently on an unsafe old context and a retry.

## Page-user notification

Every persistent CUDA quarantine/exclusion event pages the user: a proven or unresolved physical-device quarantine uses the physical-GPU message, while restart-limit or other worker-only exclusions explicitly say that a physical fault is not proven. Transient admission gates such as a stale heartbeat do not create a page unless they escalate to a durable exclusion. A shared per-device lock serializes competing worker aliases. Successful delivery is recorded in the durable latch and prevents repeat pages across worker restarts. Failed delivery uses a 60-second backoff and at most two attempts per worker or monitor process; the monitor also retries a physical latch initially created by the service CLI. Notification failure never reopens GPU admission.

Each latch has an immutable event generation. Durable JSON mutation, its Redis mirror, notification claim/finish, and manual clear serialize on the same device lock; late notification outcomes use generation compare-and-set and cannot recreate a cleared latch or mark a newer event sent. Physical escalation supersedes a worker-only page that has not begun, while a completed worker-only page and a later physical escalation remain two distinct alerts. A writer failure does not strip durable claim provenance from an existing same-device latch, so another process already delivering that generation cannot be bypassed into a duplicate page. If restart-limit persistence itself fails, the restart budget remains exhausted and the monitor still issues a bounded unlatched worker-exclusion page rather than silently omitting the alert.

The gitignored credential is read from `.secrets/page_user_mcp.json` by default, or from `KERNELGYM_PAGE_USER_MCP_CONFIG`. Only an HTTPS MCP endpoint is accepted. The `.secrets` directory and config file must not grant group/world permissions; owner-only config modes such as `0600` or `0400` are accepted. The config file must be a single-link regular file, and secure descriptor-relative opens reject symlink substitution. HTTPS POST requests do not follow redirects, and response bodies are limited to 256 KiB. Authorization, endpoint components, and oversized response content are not exposed in returned errors or logs. Advisory-lock acquisition and durable notification completion run to completion even if the surrounding coroutine is cancelled, so cancellation cannot leak a lock or release it before the durable write. Unit tests replace the HTTP client and never send real notifications.

Because the user explicitly selected a repository-local credential on a shared filesystem, these file permissions protect against other Unix users but not against untrusted code already executing under the same uid. Strong same-uid credential isolation would require moving page delivery into a separately credentialed service or broker outside the kernel-execution security boundary.

## Probe cost

The real probe creates a fresh subprocess CUDA context, allocates and executes a small CUDA operation, and synchronizes it. It runs at startup and after a fault, not on each heartbeat or normal task. Historical node21 child initialization logs measured approximately p50 `0.74 s`, p95 `1.82 s`, and p99 `2.30 s`; full parent-observed replacement readiness included process/pool orchestration and measured roughly p50 `16.3 s` and p95 `31.1 s`. A non-destructive node21 A800 probe completed in `18.942 s` while the existing deployed worker count remained unchanged.

## Manual recovery

First diagnose and repair the host/device. Stop the affected worker before clearing; clearing a live process is refused because it would leave the old CUDA context running. Inspect and clear with:

```bash
python scripts/manage_gpu_quarantine.py inspect \
  --worker-id node21_gpu_0 --device cuda:0 --hostname ai-16-21

python scripts/manage_gpu_quarantine.py clear \
  --worker-id node21_gpu_0 --device cuda:0 --hostname ai-16-21 \
  --confirm ai-16-21/cuda:0
```

For a latch whose fault class records an unproven process-group or CUDA-context reap, the clear additionally requires `--confirm-unsafe-orphan ai-16-21/cuda:0/NO_GPU_PROCESSES` after an operator has verified that no process remains on that GPU. Clear is refused while any matching supervisor generation map remains, even if its leader PID is already gone; the monitor must first prove the whole SID drained and CAS-delete that map.

After clearing, restart the worker through the normal deployment workflow so all CUDA contexts are fresh. Service restart still requires explicit operator approval.

## Remaining boundary

The physical key currently uses a stable deployment hostname plus the configured `cuda:N` index. Device ordering must therefore remain consistent across restarts; moving to GPU UUID or PCI identity is the next hardening step if device remapping becomes possible. Worker processes can currently see every container-visible GPU, so deliberately selecting another device is outside the assigned-device barrier. The atomic queue scripts assume the current standalone Redis deployment; their multi-key operations would need explicit hash-tag and key-layout work before Redis Cluster. Launching a duplicate worker id outside the supported service/monitor lifecycle also bypasses the process-map owner gate.

On node21 the available cgroup v1/hybrid controllers are mounted read-only, and the container has neither a usable per-worker device cgroup nor a seccomp filter. A normal descendant that stays in the outer SID is swept, but a deliberately forked child can create a new session and escape that boundary. The repository-local page credential is unreadable to other Unix users but not to candidate code already running under the same uid. A remote page accepted just before the sender crashes may be delivered twice unless the page service adds its own idempotency key.

Most importantly, Docker alone cannot isolate a kernel that wedges the physical GPU or host driver, and all container-visible GPUs share that host driver. The implemented controls minimize blast radius and fail closed, but a hardware reset, container restart, or host intervention may still be required. No automatic reset is attempted.
