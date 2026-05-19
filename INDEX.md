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
| `docs/design-doc/TWO_WORKER_WARM_POOL.md` | Current two-worker GPU subprocess warm-pool design. |

## Important Code Areas

| Path | Purpose |
| --- | --- |
| `scripts/deploy_node.sh` | Container-only single/multi-node startup with `--nnodes`, `--node-rank`, and `--master-addr`. |
| `kernelgym/backend/kernelbench/cuda_agent_backend.py` | CUDA-Agent parsing, validation scaffold, compile/load backend. |
| `kernelgym/backend/kernelbench/tvm_ffi_backend.py` | TVM-FFI compile/load backend and compile artifact cache. |
| `kernelgym/toolkit/kernelbench/pipeline.py` | KernelBench compile/load/correctness/performance pipeline. |
| `kernelgym/workflow/kernelbench.py` | Server-side KernelBench workflow orchestration. |
| `kernelgym/server/task_manager.py` | Redis task queue and worker coordination. |
| `kernelgym/worker/gpu_worker.py` | Worker-side task execution and failure handling. |

## External Source References

| Path | Purpose |
| --- | --- |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent` | Current reward implementation source lineage. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb` | Logic reference for ninja-driven fine-grained compilation, object cache, split compile/execute. |

## Evidence Locations

| Path | Purpose |
| --- | --- |
| `logs/compile_acceleration/` | Planned benchmark results for compile acceleration work. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_results.jsonl` | `.40` reward-only full replay results after the `assigned_worker=None` queue fix. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_compare_*.json` | Response-hash-aligned comparison of `.40` replay vs original reward results. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_turn1_results.jsonl` | Corrected raw-response `.40` replay for turn 1 only; request backend omitted and API defaulted to `auto`. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_turn1_compare_summary.json` | Turn-1 comparison summary: original `26/800`, replay `19/800`, 793 correctness matches, 7 original-true/replay-false mismatches, 20 validation 400s. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_turn1_results.tvmfix.jsonl` | Seven TVM-FFI mismatch lines rerun successfully on `.40` after installing `apache-tvm-ffi` and fixing TVM-FFI compile. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_turn1_compare_summary.final_overlay.json` | Turn-1 final overlay summary: 800/800 compile matches and 800/800 correctness matches after TVM-FFI, compile, and reference-extraction reruns. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_turn2_compare_summary.json` | Turn-2 comparison summary: 783 comparable rows, 780 correctness matches, 3 stable mismatches after retry. |
| `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen35-9b-l1fullset-mixedauto-orphanclose-t60-n8-tp1-seqs16-32k-24-r1.run.20260512-012142/eval_results/step_0/reward40_replay_turn3_compare_summary.json` | Turn-3 comparison summary: 781 comparable HTTP-200 rows, 0 compile mismatches, 3 stable original-true/replay-false correctness mismatches after retry. |
