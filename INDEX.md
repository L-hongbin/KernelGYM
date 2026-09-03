# KernelGYM Reward-Only Index

This is a compact navigation map for stable, high-value entrypoints. Individual implementation files and unit tests are intentionally omitted; use repository search and the canonical docs below for detail.

## Canonical Docs

| Path | Purpose |
| --- | --- |
| `AGENTS.md`, `RUNTIME.md` | Collaboration policy and current runtime facts. |
| `docs/DEPLOYMENT.md`, `docs/DEVELOPMENT.md` | Service operation and development conventions. |
| `docs/SOURCE_LINEAGE.md`, `docs/IMPLEMENTATION_DIFFERENCES.md` | Upstream lineage and intentional local differences. |
| `docs/NODE_LOCAL_RUNTIME_EFFICIENCY.md` | Shared-NFS versus node-local Python performance evidence and reproduction. |
| `docs/design-doc/SYSTEM_WORKFLOW.md` | Architecture, staged evaluation, caches, result merging, and failure paths. |
| `docs/design-doc/REWARD_HACKING_DEFENSES.md` | Static and runtime reward-hacking defenses. |
| `docs/design-doc/GPU_FAULT_CONTAINMENT.md` | CUDA fault containment, quarantine, and recovery. |
| `docs/design-doc/COMPILE_ACCELERATION.md` | CUDA-Agent compilation acceleration. |
| `docs/design-doc/EVAL_NO_GRAD_EXECUTION.md`, `docs/design-doc/TRUE_FP32_CORRECTNESS.md` | Correctness execution and precision policy. |
| `docs/design-doc/PROFILER_EMPTY_CAPTURE.md` | CUPTI empty-capture diagnosis and retry policy. |
| `docs/design-doc/TWO_WORKER_WARM_POOL.md`, `docs/design-doc/RUNTIME_COORDINATION_STORAGE.md` | Worker capacity and runtime-state storage designs. |
| `docs/design-doc/NPU_PROFILER_REQUIREMENTS_SURVEY.md` | P0/P1/P2 requirements for an internal NPU profiler. |
| `docs/server-result-cache-guard.md` | Safe `/evaluate` result-cache reuse. |
| `docs/testing/` | Test-suite scope and execution guidance. |

## Operational Entry Points

| Path | Purpose |
| --- | --- |
| `deploy_node.sh`, `scripts/start_container.sh` | Node deployment and container startup. |
| `ensure_venv.sh`, `set_env.sh`, `scripts/runtime_paths.sh`, `scripts/ensure_redis.sh` | Node-local Python and Redis bootstrap. |
| `requirements-offline.txt`, `wheels/` | Pinned offline runtime dependencies. |
| `kernelgym/backend/kernelbench/` | Triton, CUDA-Agent, and TVM-FFI compilation/loading backends. |
| `kernelgym/toolkit/kernelbench/`, `kernelgym/toolkit/validation.py` | KernelBench evaluation, correctness, profiling, and validation policy. |
| `kernelgym/server/`, `kernelgym/worker/`, `kernelgym/cli/service.py` | Queueing, worker lifecycle, subprocess containment, and service orchestration. |
| `scripts/benchmark_worker_spawn.py`, `scripts/reproduce_runtime_import_latency.py` | Worker-startup and import-latency benchmarks. |
| `scripts/manage_gpu_quarantine.py`, `scripts/manage_core_dumps.py` | Operator recovery and artifact-retention utilities. |

## External Source References

| Path | Purpose |
| --- | --- |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent` | Current reward implementation source lineage. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb` | Fine-grained compilation and split-execution reference. |

## Tracked Evidence

Local-only `docs/evidence/`, logs, and debug artifacts are indexed in `RUNTIME.md`, not here.

| Path | Purpose |
| --- | --- |
| `benchmarks/review_evidence/official_27b_review_evidence.json` | Official 27B adversarial review evidence. |
| `benchmarks/review_evidence/official_27b_perf_step_correctness_summary.json` | Official 27B correctness/performance-step summary. |
| `benchmarks/review_evidence/static_checker_systematic_fix.md` | Static-checker B+ replay, validation, and accepted boundaries. |
