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

Settings (env vars, `kernelgym/config/settings.py` and profile env):

| Setting | Default | Meaning |
| --- | --- | --- |
| `NUM_PROFILING_TRIALS` | `-1` (auto) | Explicit profiler forwards per context; values < 1 select auto resolution. |
| `KERNELGYM_CUPTI_TSC_SHIM` | `true` in profile `v1` | Build and inject the LD_PRELOAD shim (next section); on success the service also sets `KINETO_TSC_FIXED=true`. |
| `KINETO_TSC_FIXED` | `false` | Declare the TSC timestamp source fixed (shim injected, or a patched Kineto build), so auto mode uses 1 forward on affected CUPTI. The callback is process-level state: only effective from process start. |
| `PROFILING_RETRY_COUNT` | `1` | Existing retry when a capture comes back empty; retries fall back to the legacy multi-forward count when an expected shim did not engage. |

Empty-capture rate is observable per task from result metadata: `kg_kernel_profiling_empty_initial`, `kg_kernel_profiling_retries_used`, `kg_kernel_profiling_empty_final`, alongside the existing `kg_kernel_perf_num_profile_trials`. A final empty capture also logs a `[Profiling] empty-capture:` warning line for log-based rate scraping. The long-standing protection that an empty capture (0 total kernels) is treated as a profiler failure and never marks `decoy_kernel` is unchanged and now pinned by tests.

## How the fix is deployed: production LD_PRELOAD shim

The deployed fix is a version-gated `LD_PRELOAD` shim (`kernelgym/native/cupti_tsc_shim.cpp`, built by `kernelgym/utils/cupti_tsc_shim.py`) that interposes `cuptiActivityRegisterTimestampCallback`. On affected CUPTI versions (queried live via `cuptiGetVersion()`, resolved through libcupti's own handle because it may sit outside the global symbol scope) it suppresses the registration and flips Kineto's exported `libkineto::use_cupti_tsc()` flag to false so Kineto interprets CUPTI's native nanosecond timestamps — the exact configuration the handoff validated 45/45 on real TVM-FFI kernels. On CUPTI >= 13.1 (or <= 12.5) it passes the call through to the real CUPTI function, so upgrading the stack automatically returns to stock behavior. It has no static constructors and touches nothing else, so inheriting it into child processes (nvcc, ninja, redis) is inert.

The shim exposes its decision via `kernelgym_cupti_tsc_shim_state()`: `0` not yet invoked, `1` engaged (native timestamps), `2` passthrough on fixed CUPTI, `3` passthrough because Kineto's flag symbol was missing (stock behavior kept — suppressing registration without flipping the flag would corrupt timestamps), `4` failed. Gates around it:

- **Deploy preflight** (`scripts/validate_runtime.py`): builds the shim and probes a profiler context under `LD_PRELOAD` in a subprocess, printing `shim_state=... shim_cupti_version=... probe_kernels=...`; failures are loud warnings, never deploy blockers.
- **Injection fail-open** (`kernelgym/cli/service.py`): `KERNELGYM_CUPTI_TSC_SHIM=true` (profile `v1` default) injects `LD_PRELOAD`, `KINETO_TSC_FIXED=true`, and `KERNELGYM_CUPTI_TSC_SHIM_EXPECTED=<path>` into service processes; if the build fails, nothing is injected and the legacy multi-forward workaround stays active.
- **Runtime verification** (`timing.kineto_tsc_fix_verified`): after any profiler context, results record `cupti_tsc_shim_state` in profiling metrics, and the empty-capture retry falls back to the legacy `min(10, num_trials)` count whenever a shim was expected but did not engage in that process.

Two alternatives make the shim unnecessary; both keep working unchanged:

1. A Kineto build with the version gate compiled in (rebuilds `libtorch_cpu.so`): set `KINETO_TSC_FIXED=true` without `KERNELGYM_CUPTI_TSC_SHIM`; the declaration is then trusted as-is.
2. Upgrading to a matched CUDA/CUPTI 13.1+ stack (driver >= 580; `.21` currently runs 575.51.03; do not swap `libcupti.so` in isolation). The shim passes through and auto resolution independently detects the fixed CUDA version.

Measured effect (handoff, L1/L2/L3 canary): 81.8%–90.8% of profiler context wall time saved with single-forward profiling; coverage differences within 1.38e-4 (reward impact <= 6.89e-5). A/B on `.21` (2026-07-11, L2 P90 sample `fused_linear_sum_kernel`, ~570 ms): shim arm captured 9/9 single-forward contexts with profiler durations 545–605 ms consistent with CUDA events. Caveat recorded honestly: the baseline arm on `.21` also captured 10/10 that day — the CUPTI bug's trigger conditions are opaque and were only reliably reproducible on `.22` (1/10), so the shim on `.21` is a defense against a version-latent bug rather than a fix for an actively reproducing one; the runtime gates above cover the case where it starts triggering.

One boundary to respect when single-forward mode activates: one forward only observes the kernel path that this forward takes. If future candidates have data-dependent branches, random paths, or first-iteration kernel selection that differs from steady state, choose the sampling count from reward semantics explicitly rather than re-enabling a multi-forward count to mask profiler issues.

## Tests

- `tests/kernelbench/profiling/test_profiling_trials.py` — version gate, fail-safe on unknown versions, explicit/env overrides, retry uses the resolved count, empty-capture metadata bookkeeping, and empty-capture-never-decoy.
- `tests/kernelbench/profiling/test_empty_capture_gpu.py` — on a real GPU, consecutive profiler contexts around a slow (~120 ms) kernel must each capture kernel names with positive CUDA durations using the production-resolved trial count; explicit trial counts control the exact number of profiler forwards.
- `tests/kernelbench/profiling/test_cupti_tsc_shim.py` — shim builder produces a loadable artifact with state symbols, service-env injection and fail-open, `kineto_tsc_fix_verified` semantics per shim state, and the retry fallback to legacy trials when the shim did not engage.
- `tests/kernelbench/profiling/test_cupti_tsc_shim_gpu.py` — end-to-end in a subprocess configured like a deployed worker (shim preloaded, fix declared): auto resolution uses 1 forward, three consecutive contexts each capture the slow kernel, profiler durations match CUDA events within 30%, and the shim reports a healthy state.
