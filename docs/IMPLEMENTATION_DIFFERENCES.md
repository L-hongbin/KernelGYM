# Implementation Differences

This document records how the new reward-only repository differs from both source repositories.

## Difference From KernelGYM-vllm018-cuda-agent

Same core reward behavior:

- Keeps the `kernelgym/` implementation as the active reward service.
- Keeps CUDA-Agent, Triton, CUDA, and TVM-FFI backend wiring.
- Keeps current CUDA-Agent parser, validation, whole-extension compile cache, static checker, timing metadata,
  and KernelBench workflow behavior.

Removed scope:

- Removes `drkernel/` entirely.
- Removes training launchers, rollout scripts, checkpoint merge/eval utilities, and model-serving runbooks.
- Removes run-specific project files such as progress logs, handoffs, and experiment specs.
- Removes reward startup code that is coupled to `drkernel/kernel/scripts/rl/start_reward.sh`.

Repository-level changes:

- Adds packaging metadata in `pyproject.toml`.
- Adds focused extraction tests under `tests/`.
- Adds Ruff-only formatting and linting through pre-commit.
- Uses `ruff format`; Black is intentionally absent.
- Refactors complex shell startup/stop logic into `kernelgym.cli.service`; shell scripts are compatibility
  wrappers only.

## Difference From KernelGYM-lhb

This repo does not adopt LHB's compile-optimization architecture as active code.

Compile backend:

- This repo uses the current CUDA-Agent `torch.utils.cpp_extension.load(...)` path.
- It preserves whole-extension content-addressed compile cache metadata.
- It preserves `KERNELGYM_CUDA_AGENT_NVCC_THREADS` support.
- LHB supports `CUDA_BUILD_BACKEND=manual_ninja|cpp_extension_load`, defaults operationally to
  `manual_ninja`, and manually invokes PyTorch private ninja helpers.

Cache model:

- This repo caches complete compiled extensions.
- LHB adds object-level cache for selected `.o` files and rewrites `build.ninja` to link cached objects.

Workflow and scheduling:

- This repo submits one kernel-evaluation task per request.
- LHB can split a request into CPU compile and GPU execute stages.
- This repo's TaskManager uses priority queues.
- LHB adds CPU/GPU resource queues and optional GPU fallback polling of CPU compile tasks.

API/schema:

- This repo keeps current fields such as `num_warmup` and `perf_trim_count`.
- LHB adds `split_compile_and_execute`, `pure_compile_task`, `enable_compile_artifact_cache`,
  `task_stage`, `required_resource`, and `compile_artifact`.

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

## Migration Notes

The safest future path for LHB compile optimizations is to add them behind explicit feature flags and benchmark:

1. Current `cpp_extension.load` plus whole-extension cache.
2. LHB-style `manual_ninja` without object cache.
3. LHB-style `manual_ninja` with object cache.
4. Optional split compile/execute with CPU compile workers.

Until those benchmarks exist, this repository preserves current reward correctness and observability over
compile-path churn.
