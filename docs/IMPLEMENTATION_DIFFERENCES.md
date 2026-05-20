# Implementation Differences

This document records how the new reward-only repository differs from both source repositories.

## Difference From KernelGYM-vllm018-cuda-agent

Same core reward behavior:

- Keeps the `kernelgym/` implementation as the active reward service.
- Keeps CUDA-Agent, Triton, CUDA, and TVM-FFI backend wiring.
- Keeps current CUDA-Agent parser, validation, static checker, timing metadata, and KernelBench reward behavior.

Removed scope:

- Removes `drkernel/` entirely.
- Removes training launchers, rollout scripts, checkpoint merge/eval utilities, and model-serving runbooks.
- Removes run-specific project files such as progress logs, handoffs, and experiment specs.
- Removes reward startup code that is coupled to `drkernel/kernel/scripts/rl/start_reward.sh`.

Deployment/runtime changes:

- Uses Python deployment profiles for the known reward hosts.
- Keeps service orchestration in `kernelgym.cli.service`.
- Keeps shell scripts only for shell-native host operations such as profile detection, GPU clock locking, container startup, and virtualenv creation.

Env-var name mapping from `start_reward.sh`:

| CUDA-Agent env var | Reward-only env var | Setting | `v1` default |
| --- | --- | --- | --- |
| `REWARD_TASK_TIMEOUT` | `DEFAULT_TIMEOUT` | `settings.default_timeout` | `90` (pinned in `deployment_profiles.py`) |
| `REWARD_TASK_TIMEOUT_CLIENT` | n/a | client-side; sent in the `/evaluate` request body | — |

`REWARD_TASK_TIMEOUT` is the per-evaluation task budget enforced by the GPU worker; `start_reward.sh` previously defaulted it to `60`, so any deployment launched through that script would shadow the reward-only `90` default. The `v1` profile now sets `DEFAULT_TIMEOUT=90` explicitly so the timeout matches across both launch paths.

## Difference From KernelGYM-lhb

This repo adopts the useful compile acceleration mechanics without adopting the LHB repository's code structure or reward-policy differences.

Compile backend:

- This repo removes CUDA-Agent `torch.utils.cpp_extension.load(...)` from the active compile path.
- CUDA-Agent compilation writes an explicit PyTorch-compatible ninja build graph, invokes ninja, and imports the built extension.
- The backend identity is reported as `build_backend="manual_ninja"` for compatibility with existing metadata conventions.

Cache model:

- This repo adds object-level cache for reusable `.o` files and rewrites `build.ninja` link inputs for cache hits.
- Redis is used only as metadata/index coordination for object cache entries; compiled objects and artifacts stay on local fast storage.
- Compile artifact cache is used for exact repeated payloads and split compile/execute handoff.

Workflow and scheduling:

- This repo keeps the current reference-timing plus kernel-evaluation workflow.
- CUDA-Agent requests can split kernel compilation and execution into CPU/GPU stages.
- TaskManager routes work through CPU and GPU resource queues plus direct worker queues.
- GPU workers consume GPU work; CPU compile workers consume compile-stage CPU work.

API/schema:

- This repo keeps current fields such as `num_warmup` and `perf_trim_count`.
- It adds `split_compile_and_execute`, `pure_compile_task`, `enable_compile_artifact_cache`, `task_stage`, `required_resource`, `assigned_worker`, and `compile_artifact`.

Validation and parsing:

- This repo keeps newer CUDA-Agent parsing:
  - strips think blocks;
  - supports legacy `CUDA_SOURCES`;
  - selects the last complete CUDA section group;
  - handles `PYBIND11_MODULE` without writing duplicate framework bindings.
- LHB's CUDA section parser is simpler and primarily takes first regex matches.

Profiling and reward semantics:

- This repo keeps named-kernel coverage for CUDA-Agent/TVM-FFI based on backend profiling hints.
- It avoids automatically marking CUDA-Agent submissions as decoys when no custom kernel name can be extracted.
- LHB has an additional CUDA detection precheck and a stricter CUDA-Agent decoy policy.

## Compile Acceleration Notes

The active CUDA-Agent acceleration path has these components:

- ninja-driven fine-grained compilation;
- reusable object cache keyed by build graph, source/header content, toolchain, CUDA architecture, and flags;
- complete compile artifact cache for exact repeats and compile/execute handoff;
- CPU/GPU resource queues with dedicated CPU compile workers;
- sanitized public compile artifact metadata and internal full-artifact handoff.

See `docs/design-doc/COMPILE_ACCELERATION.md` for the target design.
