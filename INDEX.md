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
| `docs/design-doc/PROFILER_EMPTY_CAPTURE.md` | CUPTI TSC timestamp bug root cause and version-gated profiling-trial policy. |
| `docs/design-doc/REWARD_HACKING_DEFENSES.md` | Current reward-hacking defense design notes. |
| `docs/design-doc/RUNTIME_COORDINATION_STORAGE.md` | Proposed split between live runtime coordination and long-lived result/cache storage. |
| `docs/design-doc/TRUE_FP32_CORRECTNESS.md` | Correctness-time TF32 disable policy and true-fp32 oracle design. |
| `docs/design-doc/TWO_WORKER_WARM_POOL.md` | Two-worker GPU subprocess warm-pool design, capacity invariant, and `v1` verification. |
| `docs/server-result-cache-guard.md` | Server result cache hash guard design for safe `/evaluate` reuse. |

## Important Code Areas

| Path | Purpose |
| --- | --- |
| `deploy_node.sh` | Container-only single/multi-node startup with cluster/join, CPU worker override, and `--clear-cache` cold-start option. |
| `scripts/start_container.sh` | Physical-host Docker container startup; defaults to Docker `--init` for subprocess reaping. |
| `scripts/debug_line451_rmsnorm_nondeterminism.py` | Standalone reproduction for line 451 RMSNorm CUDA-Agent nondeterministic correctness. |
| `kernelgym/backend/kernelbench/cuda_agent_backend.py` | CUDA-Agent parsing, validation scaffold, compile/load backend. |
| `kernelgym/backend/kernelbench/tvm_ffi_backend.py` | TVM-FFI compile/load backend and compile artifact cache. |
| `kernelgym/toolkit/kernelbench/pipeline.py` | KernelBench compile/load/correctness/performance pipeline. |
| `kernelgym/toolkit/kernelbench/input_perturbation.py` | Distribution-aware `torch.rand`/`torch.randn` correctness input capture and transformations. |
| `kernelgym/toolkit/kernelbench/profiling.py` | CUDA profiling, exact MusaCoder Appendix J plus explicit PyTorch compatibility ATen classification, and named-kernel coverage extraction. |
| `kernelgym/toolkit/kernelbench/compute_sanitizer.py` | Isolated memcheck/racecheck/synccheck/initcheck execution and structured report parsing. |
| `kernelgym/toolkit/kernelbench/compute_sanitizer_runner.py` | Fresh-process candidate launcher used as the Compute Sanitizer target. |
| `kernelgym/native/cupti_tsc_shim.cpp` | Version-gated LD_PRELOAD shim suppressing Kineto's CUPTI TSC timestamp callback on affected CUDA versions. |
| `kernelgym/utils/cupti_tsc_shim.py` | Shim build, state query, and Kineto-TSC-fix verification gates. |
| `kernelgym/toolkit/kernelbench/ncu_profiler.py` | Fail-open Nsight Compute collection, report export, and compact per-kernel metric parsing. |
| `kernelgym/utils/device_info.py` | Startup/runtime device metadata detection and serialized-result injection. |
| `kernelgym/utils/core_dumps.py` | Core dump directory resolution, migration, and retention helpers. |
| `kernelgym/workflow/kernelbench.py` | Server-side KernelBench workflow orchestration. |
| `kernelgym/server/task_manager.py` | Redis task queue and worker coordination. |
| `kernelgym/worker/gpu_worker.py` | Worker-side task execution and failure handling. |
| `kernelgym/worker/subprocess_pool.py` | Persistent GPU subprocess pool, recycle, timeout, and pool-size enforcement. |
| `scripts/manage_core_dumps.py` | Move root-level core dumps into the configured directory and keep only the newest retained files. |
| `tests/test_aten_decoy_detection.py` | Source/compat ATen allowlists, device-only CUDA timing, and conservative low-coverage decoy regression tests. |
| `tests/test_cuda_agent_gpu.py` | Real CUDA-Agent compile/run coverage, including the `.float()` ATen compatibility path around a custom CUDA kernel. |

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
| `benchmarks/review_evidence/gemm_large_memory_delta_kernel_schema_h100_20260826.json` | Final redeployed 1024x1024 H100 GEMM evidence using reference/kernel role names and a deterministic 64 MB Kernel memory delta. |
| `benchmarks/review_evidence/official_27b_review_evidence.json` | Adversarial review evidence for official 27B 3-binding c3/c8 runs: pairing, sample IDs, coverage, statuses, queue deltas, residuals, and c3/c8 consistency. |
| `benchmarks/review_evidence/official_27b_perf_step_correctness_summary.json` | Perf-step breakdown split by completed, correct-only, and incorrect-completed rows for official 27B c3/c8 runs. |
| `benchmarks/review_evidence/runtime_sanitizer_tvm_ffi_h100_20260826_pass.json` | Current-schema H100 TVM-FFI validation for clean, OOB, race, invalid synchronization, and uninitialized-read fixtures. |

External end-to-end feedback evidence: `/data/lihongbin/code/Code-Agent/slime/examples/kernel_agent/test/log/pseudo_relu_tvm_ffi_input_perturbations_20260826_v2.json` contains the deployed TVM-FFI pseudo-ReLU A/B request, raw KernelGYM responses, and normalized slime environment feedback using the final difference-field schema.
