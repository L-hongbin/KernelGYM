# KernelGYM Reward-Only Index

This file indexes stable repository docs and evidence locations.

## Core Docs

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Collaboration and maintenance rules. |
| `RUNTIME.md` | Reward-node runtime facts, ports, and deployment details. |
| `docs/DEPLOYMENT.md` | Reward service setup and operation. |
| `docs/DEVELOPMENT.md` | Local development and test conventions. |
| `docs/SOURCE_LINEAGE.md` | Source repositories and imported/excluded behavior. |
| `docs/IMPLEMENTATION_DIFFERENCES.md` | Current implementation differences from source repositories. |
| `docs/design-doc/COMPILE_ACCELERATION.md` | CUDA-Agent compile acceleration design. |
| `docs/design-doc/EVAL_NO_GRAD_EXECUTION.md` | Fixed KernelBench eval-mode plus no-grad correctness/timing policy and cache fences. |
| `docs/design-doc/GPU_FAULT_CONTAINMENT.md` | Docker-only CUDA fault containment, fresh-context probe, durable quarantine, page-user alert, and manual recovery design. |
| `docs/design-doc/PROFILER_EMPTY_CAPTURE.md` | CUPTI TSC timestamp bug root cause and version-gated profiling-trial policy. |
| `docs/design-doc/REWARD_HACKING_DEFENSES.md` | Current reward-hacking defense design notes. |
| `docs/design-doc/RUNTIME_COORDINATION_STORAGE.md` | Proposed split between live runtime coordination and long-lived result/cache storage. |
| `docs/design-doc/SYSTEM_WORKFLOW.md` | Chinese overview of the architecture, staged evaluation, cache semantics, result merging, and fault paths. |
| `docs/design-doc/TRUE_FP32_CORRECTNESS.md` | Scoped TF32 execution policy for correctness/timing and FP32 tolerance rationale. |
| `docs/design-doc/TWO_WORKER_WARM_POOL.md` | Two-worker GPU subprocess warm-pool design, capacity invariant, and `v1` verification. |
| `docs/server-result-cache-guard.md` | Server result cache hash guard design for safe `/evaluate` reuse. |

## Important Code Areas

| Path | Purpose |
| --- | --- |
| `deploy_node.sh` | Container-only single/multi-node startup with automatic visible-GPU discovery, per-node correctness/profiling warmup, GPU/CPU worker overrides, `--clear-cache` cold start, and `--block-terminal` foreground lifecycle. |
| `ensure_venv.sh`, `set_env.sh`, `scripts/runtime_paths.sh`, `scripts/ensure_redis.sh` | Node-local Python bootstrap plus pinned Redis installation from fixed absolute offline bundles; the shared repo-local venv is deprecated. |
| `requirements-offline.txt` | Exact CPython 3.12/CUDA 12.9 environment lock whose wheels are staged in the absolute shared wheelhouse. |
| `wheels/redis/ubuntu-24.04-amd64/` | Shared gitignored Redis `.deb` bundle with exact package/platform manifests and SHA-256 checksums. |
| `scripts/start_container.sh` | Physical-host Docker container startup; defaults to Docker `--init` for subprocess reaping. |
| `scripts/debug_line451_rmsnorm_nondeterminism.py` | Standalone reproduction for line 451 RMSNorm CUDA-Agent nondeterministic correctness. |
| `scripts/benchmark_worker_spawn.py` | Isolated staged-import, real worker-constructor, and subprocess-pool replenishment benchmark with JSON evidence output. |
| `kernelgym/backend/kernelbench/cuda_agent_backend.py` | CUDA-Agent parsing, validation scaffold, compile/load backend. |
| `kernelgym/backend/kernelbench/tvm_ffi_backend.py` | TVM-FFI compile/load backend and compile artifact cache. |
| `kernelgym/schema/precision.py` | Canonical FP32/FP16/BF16 aliases and fail-closed internal normalization. |
| `kernelgym/toolkit/kernelbench/pipeline.py` | KernelBench compile/load/correctness/performance pipeline. |
| `kernelgym/toolkit/kernelbench/profiling.py` | CUDA profiling, exact MusaCoder Appendix J plus explicit PyTorch compatibility ATen classification, and named-kernel coverage extraction. |
| `kernelgym/native/cupti_tsc_shim.cpp` | Version-gated LD_PRELOAD shim suppressing Kineto's CUPTI TSC timestamp callback on affected CUDA versions. |
| `kernelgym/utils/cupti_tsc_shim.py` | Shim build, state query, and Kineto-TSC-fix verification gates. |
| `kernelgym/utils/device_info.py` | Startup/runtime device metadata detection and serialized-result injection. |
| `kernelgym/utils/core_dumps.py` | Core dump directory resolution, migration, and retention helpers. |
| `kernelgym/utils/gpu_quarantine.py` | Redis plus shared-filesystem GPU/worker quarantine latch and manual-clear primitives. |
| `kernelgym/utils/page_user_notifier.py` | Mode-restricted page-user MCP client for physical-GPU quarantine and worker-process exclusion alerts. |
| `kernelgym/cli/service.py` | Service lifecycle with admission-first shutdown, process-generation fencing, whole-group drain proof, and fail-closed replacement startup. |
| `kernelgym/workflow/kernelbench.py` | Server-side KernelBench workflow orchestration. |
| `kernelgym/server/task_manager.py` | Redis task queue and worker coordination. |
| `kernelgym/worker/gpu_worker.py` | Worker-side task execution and failure handling. |
| `kernelgym/worker/subprocess_pool.py` | Persistent GPU subprocess pool, crash containment proof, fresh-context recovery, recycle, timeout, and pool-size enforcement. |
| `kernelgym/worker/worker_monitor.py` | Generation-fenced worker supervision, bounded restart, and unsafe process-group quarantine. |
| `tests/deployment/` | Deployment scripts, service CLI, runtime validation, static profiles, and reward-smoke tests. |
| `tests/server/` | API, request defaults, task-manager queues, Redis integration, and heartbeat-route tests. |
| `tests/workers/` | CPU/GPU worker, subprocess-pool, monitor, shutdown-drain, and quarantine tests. |
| `tests/utils/` | Core-dump and page-user notification utility tests. |
| `tests/kernelbench/backends/` | CUDA-Agent and TVM-FFI backend/schema tests. |
| `tests/kernelbench/correctness/` | Correctness, cache-poison, and true-FP32 policy tests. |
| `tests/kernelbench/execution_modes/` | Active eval plus no-grad correctness, timing, Triton-detection, and cache-fence regressions. |
| `tests/kernelbench/profiling/` | CUPTI, profiler capture/trial, and ATen decoy-detection tests. |
| `tests/kernelbench/timing/` | CUDA timing-window tests. |
| `tests/kernelbench/workflow/` | Precision propagation and split-stage affinity tests. |
| `scripts/manage_core_dumps.py` | Move root-level core dumps into the configured directory and keep only the newest retained files. |
| `scripts/manage_gpu_quarantine.py` | Inspect or explicitly clear a stopped GPU worker's durable safety latch. |
| `docs/testing/KERNELBENCH_EXECUTION_MODES.md` | Execution-mode regression scope and category-level invocation. |

## External Source References

| Path | Purpose |
| --- | --- |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent` | Current reward implementation source lineage. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb` | Logic reference for ninja-driven fine-grained compilation, object cache, split compile/execute. |

## Evidence Locations

Tracked repository evidence artifacts only. Local-only `docs/evidence/`, run logs, and debug artifacts are gitignored and indexed in `RUNTIME.md`.

| Path | Purpose |
| --- | --- |
| `benchmarks/review_evidence/official_27b_review_evidence.json` | Adversarial review evidence for official 27B 3-binding c3/c8 runs: pairing, sample IDs, coverage, statuses, queue deltas, residuals, and c3/c8 consistency. |
| `benchmarks/review_evidence/official_27b_perf_step_correctness_summary.json` | Perf-step breakdown split by completed, correct-only, and incorrect-completed rows for official 27B c3/c8 runs. |
