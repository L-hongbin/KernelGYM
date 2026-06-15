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
| `docs/design-doc/REWARD_HACKING_DEFENSES.md` | Current reward-hacking defense design notes. |
| `docs/design-doc/RUNTIME_COORDINATION_STORAGE.md` | Proposed split between live runtime coordination and long-lived result/cache storage. |
| `docs/design-doc/TWO_WORKER_WARM_POOL.md` | Two-worker GPU subprocess warm-pool design, capacity invariant, and `v1` verification. |
| `docs/server-result-cache-guard.md` | Server result cache hash guard design for safe `/evaluate` reuse. |

## Important Code Areas

| Path | Purpose |
| --- | --- |
| `deploy_node.sh` | Container-only single/multi-node startup with `--nnodes`, `--node-rank`, and `--master-addr`. |
| `scripts/start_container.sh` | Physical-host Docker container startup; defaults to Docker `--init` for subprocess reaping. |
| `scripts/debug_line451_rmsnorm_nondeterminism.py` | Standalone reproduction for line 451 RMSNorm CUDA-Agent nondeterministic correctness. |
| `kernelgym/backend/kernelbench/cuda_agent_backend.py` | CUDA-Agent parsing, validation scaffold, compile/load backend. |
| `kernelgym/backend/kernelbench/tvm_ffi_backend.py` | TVM-FFI compile/load backend and compile artifact cache. |
| `kernelgym/toolkit/kernelbench/pipeline.py` | KernelBench compile/load/correctness/performance pipeline. |
| `kernelgym/utils/device_info.py` | Startup/runtime device metadata detection and serialized-result injection. |
| `kernelgym/utils/core_dumps.py` | Core dump directory resolution, migration, and retention helpers. |
| `kernelgym/workflow/kernelbench.py` | Server-side KernelBench workflow orchestration. |
| `kernelgym/server/task_manager.py` | Redis task queue and worker coordination. |
| `kernelgym/worker/gpu_worker.py` | Worker-side task execution and failure handling. |
| `kernelgym/worker/subprocess_pool.py` | Persistent GPU subprocess pool, recycle, timeout, and pool-size enforcement. |
| `scripts/manage_core_dumps.py` | Move root-level core dumps into the configured directory and keep only the newest retained files. |

## External Source References

| Path | Purpose |
| --- | --- |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent` | Current reward implementation source lineage. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb` | Logic reference for ninja-driven fine-grained compilation, object cache, split compile/execute. |

## Evidence Locations

Tracked review-evidence files only. Untracked run logs and debug artifacts
(gitignored `logs/` and `artifacts/`) are indexed in `RUNTIME.md`.

| Path | Purpose |
| --- | --- |
| `benchmarks/review_evidence/official_27b_review_evidence.json` | Adversarial review evidence for official 27B 3-binding c3/c8 runs: pairing, sample IDs, coverage, statuses, queue deltas, residuals, and c3/c8 consistency. |
| `benchmarks/review_evidence/official_27b_perf_step_correctness_summary.json` | Perf-step breakdown split by completed, correct-only, and incorrect-completed rows for official 27B c3/c8 runs. |
