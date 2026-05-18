# Source Lineage

This reward-only repository is intentionally small in scope: it is a standalone KernelGYM reward service, not a training or rollout repository.

## Source Repository A: KernelGYM-vllm018-cuda-agent

Path:

```text
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent
```

Role in this extraction:

- Primary implementation source.
- `kernelgym/` was copied from this repository.
- Root KernelGYM service scripts were copied from this repository.
- Reward semantics are preserved:
  - CUDA-Agent and TVM-FFI backend dispatcher.
  - CUDA-Agent ninja-driven fine-grained compilation.
  - CUDA-Agent object cache and compile artifact cache.
  - `/dev/shm`-oriented CUDA-Agent and TVM-FFI temp/cache behavior.
  - CUDA-Agent section parsing with think-block stripping and last complete CUDA section group selection.
  - static checker and stricter binding validation.
  - stage timing metadata and performance warmup/trim controls.
  - split compile/execute workflow with CPU compile workers and GPU execution workers.

Excluded from this source:

- `drkernel/`.
- RL training scripts.
- rollout, model, checkpoint, and offline-eval orchestration.
- run-specific progress/spec/handoff files.

## Source Repository B: KernelGYM-lhb

Path:

```text
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb
```

Role in this extraction:

- Comparative CUDA reward-env optimization source.
- The LHB repository was used as a logic reference for:
  - ninja-driven fine-grained compilation.
  - object-level cache and `build.ninja` link-input rewriting.
  - split compile/execute request fields.
  - CPU compile workers and CPU/GPU resource queues.
  - compile artifact cache.

Not adopted from this source:

- repository-specific implementation style and service wiring.
- LHB CUDA decoy policy.

The reward-only implementation keeps this repo's newer CUDA-Agent parsing, TVM-FFI support, static validation, and profiling metadata while adopting the compile acceleration mechanics described in `docs/COMPILE_ACCELERATION_DESIGN.md`.
