# Profiler Empty Capture: CUPTI TSC Timestamp Bug and Profiling-Trial Policy

## Problem

During the performance stage KernelGym runs the candidate forward inside `torch.profiler` (Kineto → CUPTI) to collect executed CUDA kernel names and per-kernel GPU durations. Those feed `time_coverage` and decoy detection. On some runs the profiler exports zero CUDA kernel activities even though the candidate ran correctly (`Profiler captured no CUDA kernels.`), the CUDA-Event runtime is normal, and `cudaLaunchKernel` is visible in the runtime trace. The failure concentrates on slow kernels: for an L2 P90 TVM-FFI sample, a single-forward profiler context captured kernels only 1/10 of the time. An empty capture never marks a decoy, but `time_coverage` becomes 0 and the coverage reward can lose up to 0.5.

## Root cause

NVIDIA-confirmed CUPTI defect: when a client registers `cuptiActivityRegisterTimestampCallback` (Kineto's TSC fast path for low-overhead host timestamps), CUPTI can write kernel activity records with `start=0`. Kineto then discards the record as outside the profiling window, so `events()`, `key_averages()`, and the Chrome trace all lack the kernel. The defect was introduced in CUDA 12.6 Update 2 and fixed in CUDA 13.1 (see the [CUPTI release notes](https://docs.nvidia.com/cupti/release-notes/release-notes.html#updates-in-cuda-13-1)). The reward nodes currently run torch 2.11.0+cu129 with CUPTI API version 28 (CUDA 12.9), which is inside the affected range, and this Kineto build has no runtime knob to disable the TSC callback.

The historical mitigation ran `min(10, num_trials)` candidate forwards inside one profiler context (10 in production, where `num_trials=100`). Longer traces with consecutive launches avoid the trigger in practice (verified 45/45 non-empty), but for slow kernels the cost is large: an L2 P90 sample spends ~5.7 s per profiler context versus ~0.6 s with a single forward; L3 P90 ~11.4 s versus ~1.15 s.

Full root-cause analysis, experiments, and validation live outside this repo:

- Handoff: `/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/handoffs/rollout_speedup/handoff_reward_profiler_iterations.md`
- Experiments: `/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/experiments/tvmffi_kineto_rootcause_20260710/summary.md`

## What KernelGym does now

`kernelgym/toolkit/kernelbench/timing.py` resolves the number of extra profiler forwards per context (`resolve_num_profiling_trials`) instead of hardcoding `min(10, num_trials)`:

| Condition | Profiler forwards per context |
| --- | --- |
| `NUM_PROFILING_TRIALS >= 1` (explicit) | that value |
| Auto, CUPTI TSC bug suspected (CUDA 12.6–13.0, or unknown/unparseable CUDA version, and no patched Kineto declared) | legacy `min(10, max(1, num_trials))` workaround |
| Auto, bug absent (CUDA >= 13.1, CUDA <= 12.5, or `KINETO_TSC_FIXED=true`) | 1 |

The version gate is deliberately conservative: CUDA 12.6 GA/U1 cannot be distinguished from U2 by version string, and an unknown version fails safe into the workaround. The empty-capture retry in `pipeline._run_performance_step` also uses the resolved count (previously a hardcoded `min(10, num_perf_trials)`).

Settings (env vars, `kernelgym/config/settings.py`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `NUM_PROFILING_TRIALS` | `-1` (auto) | Explicit profiler forwards per context; values < 1 select auto resolution. |
| `KINETO_TSC_FIXED` | `false` | Declare the deployed Kineto build already version-gates the TSC callback, so auto mode may use 1 forward on affected CUPTI. Only set after deploying such a build and restarting the service (the callback is process-level state). |
| `PROFILING_RETRY_COUNT` | `1` | Existing retry when a capture comes back empty; intended as a canary-period guard, not a long-term 10-forward fallback. |

Empty-capture rate is observable per task from result metadata: `kg_kernel_profiling_empty_initial`, `kg_kernel_profiling_retries_used`, `kg_kernel_profiling_empty_final`, alongside the existing `kg_kernel_perf_num_profile_trials`. A final empty capture also logs a `[Profiling] empty-capture:` warning line for log-based rate scraping. The long-standing protection that an empty capture (0 total kernels) is treated as a profiler failure and never marks `decoy_kernel` is unchanged and now pinned by tests.

## Why not 1 forward today

The 10→1 reduction is only safe after the timestamp source is fixed. That requires one of:

1. A Kineto build that skips `cuptiActivityRegisterTimestampCallback` and sets `use_cupti_tsc=false` on affected CUPTI versions (version-gated via `cuptiGetVersion()`; concept patch in the handoff). This patch has not been built or deployed; the experimental `LD_PRELOAD` shim that proved causality is diagnostic-only and must not be deployed. Once such a build is live, set `KINETO_TSC_FIXED=true` and restart.
2. Upgrading the whole stack to a matched CUDA/CUPTI 13.1+ combination (driver must move to >= 580; `.21` currently runs 575.51.03). Auto resolution then drops to 1 forward with no config change. Do not swap `libcupti.so` in isolation.

Until then, auto mode keeps the legacy workaround on CUDA 12.6–13.0, trading profiler wall time for reliable captures. Expected savings once single-forward mode activates: 81.8%–90.8% of profiler context time on the validated L1/L2/L3 samples, with coverage differences within 1.38e-4 (reward impact <= 6.89e-5).

One boundary to respect when single-forward mode activates: one forward only observes the kernel path that this forward takes. If future candidates have data-dependent branches, random paths, or first-iteration kernel selection that differs from steady state, choose the sampling count from reward semantics explicitly rather than re-enabling a multi-forward count to mask profiler issues.

## Tests

- `tests/test_profiling_trials.py` — version gate, fail-safe on unknown versions, explicit/env overrides, retry uses the resolved count, empty-capture metadata bookkeeping, and empty-capture-never-decoy.
- `tests/test_profiling_empty_capture_gpu.py` — on a real GPU, consecutive profiler contexts around a slow (~120 ms) kernel must each capture kernel names with positive CUDA durations using the production-resolved trial count; explicit trial counts control the exact number of profiler forwards.
