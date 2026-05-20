# KernelGYM Reward-Only Index

This file indexes stable repository docs and evidence locations.

## Core Docs

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Collaboration and maintenance rules. |
| `SPEC.md` | Reward-node runtime facts, ports, and deployment details. |
| `docs/DEPLOYMENT.md` | Reward service setup and operation. |
| `docs/DEVELOPMENT.md` | Local development and test conventions. |
| `docs/SOURCE_LINEAGE.md` | Source repositories and imported/excluded behavior. |
| `docs/IMPLEMENTATION_DIFFERENCES.md` | Current implementation differences from source repositories. |
| `docs/design-doc/COMPILE_ACCELERATION.md` | CUDA-Agent compile acceleration design. |
| `docs/design-doc/REWARD_HACKING_DEFENSES.md` | Current reward-hacking defense design notes. |
| `docs/design-doc/TWO_WORKER_WARM_POOL.md` | Two-worker GPU subprocess warm-pool design, capacity invariant, and `v1` verification. |
| `docs/server-result-cache-guard.md` | Server result cache hash guard design for safe `/evaluate` reuse. |

## Important Code Areas

| Path | Purpose |
| --- | --- |
| `scripts/deploy_node.sh` | Container-only single/multi-node startup with `--nnodes`, `--node-rank`, and `--master-addr`. |
| `scripts/debug_line451_rmsnorm_nondeterminism.py` | Standalone reproduction for line 451 RMSNorm CUDA-Agent nondeterministic correctness. |
| `kernelgym/backend/kernelbench/cuda_agent_backend.py` | CUDA-Agent parsing, validation scaffold, compile/load backend. |
| `kernelgym/backend/kernelbench/tvm_ffi_backend.py` | TVM-FFI compile/load backend and compile artifact cache. |
| `kernelgym/toolkit/kernelbench/pipeline.py` | KernelBench compile/load/correctness/performance pipeline. |
| `kernelgym/workflow/kernelbench.py` | Server-side KernelBench workflow orchestration. |
| `kernelgym/server/task_manager.py` | Redis task queue and worker coordination. |
| `kernelgym/worker/gpu_worker.py` | Worker-side task execution and failure handling. |
| `kernelgym/worker/subprocess_pool.py` | Persistent GPU subprocess pool, recycle, timeout, and pool-size enforcement. |

## External Source References

| Path | Purpose |
| --- | --- |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent` | Current reward implementation source lineage. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb` | Logic reference for ninja-driven fine-grained compilation, object cache, split compile/execute. |

## Evidence Locations

| Path | Purpose |
| --- | --- |
| `logs/compile_acceleration/` | Planned benchmark results for compile acceleration work. |
| `artifacts/line451_rmsnorm_debug/repeat50_trial0_result.jsonl` | Standalone `.40` evidence: line 451 fixed-input RMSNorm custom kernel failed 6/50 repeats with exactly 32 mismatched elements per failure. |
