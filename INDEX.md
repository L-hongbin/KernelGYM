# KernelGYM Reward-Only Index

This file indexes stable repository docs and evidence locations. Compiler feedback evidence: `benchmarks/review_evidence/compile_error_detail_limit_20260918.json` (synthetic classifier fixture), `benchmarks/review_evidence/compile_feedback_deployed_20260918.json` (real HTTP requests/responses), reproduced by `benchmarks/validate_compile_feedback.py`.

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
| `ensure_venv.sh`, `set_env.sh`, `scripts/runtime_paths.sh`, `scripts/ensure_redis.sh` | Project-local Python bootstrap, with an absolute-path override, plus Redis installation from offline paths defaulted from the selected `WHELL_PATH`. |
| `requirements-offline.txt` | Exact CPython 3.12/CUDA 12.9 environment lock whose wheels are staged in the absolute shared wheelhouse. |
| `wheels/redis/ubuntu-24.04-amd64/` | Shared gitignored Redis `.deb` bundle with exact package/platform manifests and SHA-256 checksums. |
| `scripts/start_container.sh` | Physical-host Docker container startup; defaults to Docker `--init` for subprocess reaping. |
| `scripts/debug_line451_rmsnorm_nondeterminism.py` | Standalone reproduction for line 451 RMSNorm CUDA-Agent nondeterministic correctness. |
| `scripts/benchmark_worker_spawn.py` | Isolated staged-import, real worker-constructor, and subprocess-pool replenishment benchmark with JSON evidence output. |
| `scripts/reproduce_runtime_import_latency.py` | Fresh-process serial and concurrent module-import comparison between shared and node-local Python environments. |
| `scripts/profile_speed_test_ncu.py`, `scripts/benchmark_speed_test_block_cv.py`, `scripts/benchmark_speed_test_block_cv_fresh_process.py`, `scripts/validate_noise_floor_cases.py` | Reproducible NCU, same/fresh-process block CV comparisons, and fresh-process compile/correctness validation of the ten-case TVM-FFI noise-floor suite. |
| `kernelgym/backend/kernelbench/cuda_agent_backend.py` | CUDA-Agent parsing, validation scaffold, compile/load backend. |
| `kernelgym/backend/kernelbench/tvm_ffi_backend.py` | TVM-FFI compile/load backend and compile artifact cache. |
| `kernelgym/schema/precision.py` | Canonical FP32/FP16/BF16 aliases and fail-closed internal normalization. |
| `kernelgym/toolkit/kernelbench/pipeline.py` | KernelBench compile/load/correctness/performance pipeline, including request-gated detailed correctness and Sanitizer execution. |
| `kernelgym/toolkit/kernelbench/input_perturbation.py` | Distribution-aware `torch.rand`/`torch.randn` correctness input capture and transformations. |
| `kernelgym/toolkit/kernelbench/profiling.py` | CUDA profiling, exact MusaCoder Appendix J plus explicit PyTorch compatibility ATen classification, and named-kernel coverage extraction. |
| `kernelgym/toolkit/kernelbench/compute_sanitizer.py` | Isolated scenario-ordered memcheck/racecheck/synccheck/initcheck execution, bounded budgets, and structured report parsing. |
| `kernelgym/toolkit/kernelbench/compute_sanitizer_runner.py` | Fresh-process candidate launcher used as the Compute Sanitizer target. |
| `kernelgym/native/cupti_tsc_shim.cpp` | Version-gated LD_PRELOAD shim suppressing Kineto's CUPTI TSC timestamp callback on affected CUDA versions. |
| `kernelgym/utils/cupti_tsc_shim.py` | Shim build, state query, and Kineto-TSC-fix verification gates. |
| `kernelgym/toolkit/kernelbench/ncu_profiler.py` | Fail-open Nsight Compute collection, report export, and compact per-kernel metric parsing. |
| `kernelgym/utils/device_info.py` | Startup/runtime device metadata detection and serialized-result injection. |
| `kernelgym/utils/core_dumps.py` | Core dump directory resolution, migration, and retention helpers. |
| `kernelgym/utils/gpu_quarantine.py` | Redis plus shared-filesystem GPU/worker quarantine latch and manual-clear primitives. |
| `kernelgym/utils/page_user_notifier.py` | Mode-restricted page-user MCP client for physical-GPU quarantine and worker-process exclusion alerts. |
| `kernelgym/cli/service.py` | Service lifecycle with admission-first shutdown, process-generation fencing, whole-group drain proof, and fail-closed replacement startup. |
| `kernelgym/workflow/kernelbench.py` | Server-side KernelBench workflow orchestration. |
| `kernelgym/server/api/speed_test.py`, `kernelgym/server/api/noise_floor.py`, `kernelgym/server/api/noise_floor_cases.py` | Fixed TVM-FFI speed probe plus the ten-case interleaved block-level noise-floor suite, request construction, result extraction, p75/bucket estimation, and held-out LCB validation. |
| `kernelgym/server/task_manager.py`, `kernelgym/server/workflow_lifecycle.py` | Redis queues, parent lifecycle/deadline/lease, cancellation tombstones and frozen-child business completion; evidence: `benchmarks/review_evidence/workflow_lifecycle_p02.md`, `benchmarks/review_evidence/workflow_cancellation_p03.md`. |
| `kernelgym/worker/gpu_worker.py` | Worker-side task execution and failure handling. |
| `kernelgym/worker/subprocess_pool.py` | Persistent GPU subprocess pool, crash containment proof, fresh-context recovery, recycle, timeout, and pool-size enforcement. |
| `kernelgym/worker/worker_monitor.py` | Generation-fenced supervision, bounded restart, quarantine, and safe exited-generation reconciliation; P01 evidence: `benchmarks/review_evidence/quarantine_reconciliation_p01.md`. |
| `tests/deployment/` | Deployment scripts, service CLI, runtime validation, static profiles, and reward-smoke tests. |
| `tests/server/` | API, request defaults, task-manager queues, Redis integration, and heartbeat-route tests. |
| `tests/workers/` | CPU/GPU worker, subprocess-pool, monitor, shutdown-drain, and quarantine tests. |
| `tests/utils/` | Core-dump and page-user notification utility tests. |
| `tests/kernelbench/backends/` | CUDA-Agent and TVM-FFI backend/schema tests. |
| `tests/kernelbench/correctness/` | Legacy/default and detailed correctness, cache-poison, and true-FP32 policy tests. |
| `tests/kernelbench/execution_modes/` | Active eval plus no-grad correctness, timing, Triton-detection, and cache-fence regressions. |
| `tests/kernelbench/profiling/` | CUPTI, profiler capture/trial, and ATen decoy-detection tests. |
| `tests/kernelbench/timing/` | CUDA timing-window tests. |
| `tests/kernelbench/workflow/` | Precision propagation and split-stage affinity tests. |
| `scripts/manage_core_dumps.py` | Move root-level core dumps into the configured directory and keep only the newest retained files. |
| `scripts/manage_gpu_quarantine.py` | Inspect latches, reconcile exited maps using the monitor's proof/CAS, or explicitly clear a stopped GPU worker's latch. |
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
| `benchmarks/review_evidence/*memory*.json`, `benchmarks/review_evidence/torch_cuda_memory_trial_h100.json` | H100 memory accounting evidence spanning PyTorch and TVM-FFI trials, absolute peaks, deltas, response units, and the final reference/kernel schema. |
| `benchmarks/review_evidence/gemm_rmsnorm_speed_test_local_20260902.json`, `benchmarks/review_evidence/gemm_rmsnorm_fused_speed_test_h100_20260902.json`, `benchmarks/review_evidence/gemm_rmsnorm_speed_test_ncu_h100_20260902.json` | Live-worker validation of the fixed and fused TVM-FFI GEMM + RMSNorm speed-test handler, including three-run timing, Redis cleanup, and NCU output. |
| `benchmarks/review_evidence/gemm_rmsnorm_reference_vs_candidate_ncu_h100_20260903.json` | Full same-case NCU output and aggregate GPU-kernel comparison for eager PyTorch versus the fused TVM-FFI candidate. |
| `benchmarks/review_evidence/official_27b_review_evidence.json` | Adversarial review evidence for official 27B 3-binding c3/c8 runs: pairing, sample IDs, coverage, statuses, queue deltas, residuals, and c3/c8 consistency. |
| `benchmarks/review_evidence/official_27b_perf_step_correctness_summary.json` | Perf-step breakdown split by completed, correct-only, and incorrect-completed rows for official 27B c3/c8 runs. |
| `benchmarks/review_evidence/runtime_sanitizer_*.json`, `benchmarks/review_evidence/sanitizer_independent_detail_h100_20260918.json`, `benchmarks/review_evidence/kernelgym_post_sanitizer_recycle_pass_h100_20260825.json`, `benchmarks/review_evidence/cuda_stage_fault_propagation_h100_20260915.md`, `benchmarks/review_evidence/cuda_stage_fault_post_restart_h100_20260915.json` | Local/service/redeployment H100 sanitizer experiments plus current-schema clean, OOB, race, invalid synchronization, uninitialized-read fixtures, process recycle, early stage fault propagation, post-restart verification, and independent Sanitizer/Detail gates. |
| `benchmarks/review_evidence/current_correctness_nonfinite_case_h100_20260908.json`, `benchmarks/review_evidence/return_detail_correctness_gate_h100_20260909.json`, `benchmarks/review_evidence/deployed_return_detail_correctness_gate_h100_20260909.json` | Detailed-mode mismatch output plus local and deployed H100 default-vs-detailed gate comparisons, including effective Sanitizer gating. |
| `benchmarks/review_evidence/current_correctness_finite_case_h100_20260908.json`, `benchmarks/review_evidence/correctness_diagnosis_v1_h100_20260917.json`, `benchmarks/review_evidence/correctness_diagnosis_deployed_h100_20260917.json` | Detailed-mode correctness response plus local and deployed metadata-only high-confidence diagnosis, terminal-tile localization, and Sanitizer-suppression A/B evidence. |
| `benchmarks/review_evidence/mismatch_triggered_sanitizer_h100_20260908.json` | H100 integration evidence that output mismatches trigger full sanitizer replay, return detected race issues, and omit clean sanitizer feedback. |
| `benchmarks/review_evidence/gemm_rmsnorm_block_cv_h100_20260908.json`, `benchmarks/review_evidence/gemm_rmsnorm_block_cv_internal_range_h100_20260908.json`, `benchmarks/review_evidence/gemm_rmsnorm_block_cv_post_restart_h100_20260908.json`, `benchmarks/review_evidence/gemm_rmsnorm_block_cv_post_restart_repeat2_h100_20260908.json`, `benchmarks/review_evidence/gemm_rmsnorm_block_cv_warmup10_h100_20260908.json`, `benchmarks/review_evidence/gemm_rmsnorm_block_cv_fresh_process_warmup10_h100_20260908.json`, `benchmarks/review_evidence/kernelgym_restart_gpu_ownership_h100_20260908.json`, `benchmarks/review_evidence/noise_floor_case_smoke_h100_20260909.json` | Six ten-block uncached H100 GEMM + RMSNorm studies plus 10/10 fresh-process compile/correctness evidence for the short/medium/long TVM-FFI calibration suite. |
| `benchmarks/benchmark_correctness_fusion.py`, `benchmarks/review_evidence/correctness_fusion*_h100_20260918.json`, `benchmarks/review_evidence/detail_correctness_overhead_h100_20260918.json` | Reproducible fused-vs-PyTorch diagnostics: fresh-process and steady comparator timings, actual TVM-FFI evaluator timings, paired HTTP responses, accuracy/sanitizer/SM80 validation, and historical default-vs-detail evidence. Explicit decoy/profiling controls separate comparator cost from profiler startup; no worker prewarming. |
External end-to-end feedback evidence: `/data/lihongbin/code/Code-Agent/slime/examples/kernel_agent/test/log/pseudo_relu_tvm_ffi_input_perturbations_20260826_v2.json` contains the deployed TVM-FFI pseudo-ReLU A/B request, raw KernelGYM responses, and normalized slime environment feedback using the final difference-field schema.
