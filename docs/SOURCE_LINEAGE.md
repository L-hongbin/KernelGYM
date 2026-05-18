# Source Lineage

This reward-only repository is intentionally small in scope: it is a standalone KernelGYM reward service,
not a training or rollout repository.

## Source Repository A: KernelGYM-vllm018-cuda-agent

Path:

```text
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent
```

Role in this extraction:

- Primary implementation source.
- `kernelgym/` was copied from this repository.
- Root KernelGYM service scripts were copied from this repository.
- Current reward semantics are preserved:
  - CUDA-Agent and TVM-FFI backend dispatcher.
  - Whole-extension compile cache for CUDA-Agent.
  - `/dev/shm`-oriented CUDA-Agent and TVM-FFI temp/cache behavior.
  - CUDA-Agent section parsing with think-block stripping and last complete CUDA section group selection.
  - static checker and stricter binding validation.
  - stage timing metadata and performance warmup/trim controls.

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

- Comparative CUDA reward-env design source.
- The LHB repository was used to document an alternate reward implementation with:
  - `CUDA_BUILD_BACKEND=manual_ninja`.
  - manual ninja object cache.
  - split compile/execute request fields.
  - CPU compile workers and CPU/GPU resource queues.
  - compile artifact cache.

Not adopted in this active implementation:

- LHB `manual_ninja` backend implementation.
- LHB object-cache build.ninja rewriting.
- LHB split compile/execute workflow.
- LHB CPU compile workers and resource-aware task queues.
- LHB CUDA decoy policy.

Those choices are deliberate. The current reward semantics from `KernelGYM-vllm018-cuda-agent` are newer for
CUDA-Agent parsing, TVM-FFI support, static validation, and profiling metadata. LHB's compile optimizations are
kept as documented future work rather than mixed into this extraction without benchmarks.
