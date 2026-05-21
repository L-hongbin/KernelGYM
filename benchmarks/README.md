# Compile-speed benchmarks

End-to-end measurements of the reward service's `/evaluate` path, used to
investigate compile-time regressions and validate compile-acceleration
changes (object cache, stable ext_name, artifact cache, future PCH).

Created during the `KernelGYM-vllm018-cuda-agent` → `KernelGYM-reward-only`
investigation (May 2026); see commits around the same date in
`docs/design-doc/COMPILE_ACCELERATION.md`.

## Layout

```
benchmarks/
├── README.md
├── run_compile_benchmark.py     # the runner (stdlib only, can run from any venv)
├── kernels/
│   ├── _reference.py            # shared reference Model (element-wise add)
│   ├── cuda_agent_vector_add.py # torch/extension.h + pybind11 binding
│   └── tvm_ffi_vector_add.py    # tvm/ffi/tvm_ffi.h + TVM_FFI_DLL_EXPORT_TYPED_FUNC
└── results/                     # append-only JSONL output (gitignored if you prefer)
```

The two fixtures evaluate against the **same** reference Model, so any timing
difference is attributable to the backend's compile path, not the workload.

## Running

```bash
# Default: cold -> warm -> 2 novels -> warm-again, against .40
python benchmarks/run_compile_benchmark.py --backend cuda_agent
python benchmarks/run_compile_benchmark.py --backend tvm_ffi

# Other scenarios
python benchmarks/run_compile_benchmark.py --backend tvm_ffi --scenario warm-only
python benchmarks/run_compile_benchmark.py --backend tvm_ffi --scenario novel

# Persist for later comparison
python benchmarks/run_compile_benchmark.py --backend cuda_agent \
    --out benchmarks/results/$(date +%Y%m%d_%H%M%S)_cuda_agent_sequence.jsonl

# Point at a different host/port
KERNELGYM_REWARD_HOST=127.0.0.1 KERNELGYM_REWARD_PORT=20111 \
    python benchmarks/run_compile_benchmark.py --backend cuda_agent
```

Each run prints one JSON record per scenario step on stdout (machine-readable)
and a one-line human summary on stderr. Server-side per-task timeout is
`--timeout` seconds (default 240); HTTP-side timeout is `timeout + 60`.

## Scenarios

| Scenario | Steps | Purpose |
|---|---|---|
| `sequence` (default) | identical-1, identical-2, mutant-1, mutant-2, identical-3 | One pass exercises cold compile, warm-repeat cache hit, two cold-novel cache misses, then a warm hit again — covers every regime in five requests. |
| `warm-only` | warm-1, warm-2, warm-3 | Steady-state floor (best-case cache hit). |
| `novel` | novel-1, novel-2, novel-3 | Cold floor: each step inserts a fresh `// bench-mutant-<tag>` comment so the source-content hash differs, forcing a full recompile. |

## Captured timing fields

Per-run JSON includes (omitted when not exposed by the backend):

| Field | Meaning |
|---|---|
| `elapsed_s` | Client wall time |
| `kg_kernel_total_s` | Server-side task total |
| `kg_kernel_backend_compile_s` | Compile + import (or `dlopen` on cache hit) |
| `kg_kernel_backend_load_s` | Pure `dlopen` portion of load |
| `kg_kernel_performance_step_s` | Perf phase incl. profiler |
| `kg_kernel_correctness_s` | Correctness trials |
| `kg_reference_total_s` | Reference baseline run |
| `wg_pool_total_s` | Subprocess pool dispatch/execute total |
| `compile_artifact_cache_enabled` / `compile_artifact_cache_hit` | Whole-`.so` cache state |
| `object_cache_hits` / `_misses` / `_skipped` | Per-object cache state (cuda_agent only) |
| `manual_ninja_build_wall_sec` | Time inside ninja (build + link) |
| `manual_ninja_import_wall_sec` | Time to `dlopen` after build |
| `build_backend` | `manual_ninja` / `tvm_ffi.cpp.build` / cached |

## Force-refresh behaviour

The runner always sends `force_refresh: True` so the per-task **result** cache
(the cache that stores compile/correctness/speedup keyed by request hash) is
skipped on every request. What you measure is *real* work modulo whatever
compile-layer caches (object cache, compile artifact cache) the service has
enabled. To benchmark the result-cache hit path, run the regular
`scripts/test_reward.py` smoke without `force_refresh`.

## Findings log

When you add a meaningful run, drop a short note here with link to the JSONL
file under `results/`. Don't paste full output — keep this file scannable.

### 2026-05-21 — backend comparison, vector-add, .40 (`results/*_sequence.jsonl`)

Direct head-to-head on identical reference and identical sequence steps.
Service: stable-ext_name + relaxed binding skip in effect for cuda_agent;
artifact cache off for both backends.

| Scenario step | cuda_agent (elapsed / compile) | tvm_ffi (elapsed / compile) |
|---|---:|---:|
| identical-1 (cold or near-cold) | 6.0 s / 1.1 s¹ | 10.5 s / 2.5 s |
| identical-2 (warm) | 6.5 s / 0.7 s | 5.5 s / 2.1 s |
| mutant-1 (genuinely cold novel) | **99.7 s / 90.9 s** | **6.0 s / 2.0 s** |
| mutant-2 (genuinely cold novel) | 101.2 s / 92.8 s | 6.0 s / 2.0 s |
| identical-3 (warm) | 6.0 s / 0.8 s | 6.0 s / 2.1 s |

¹ cuda_agent identical-1 hit the per-object cache from earlier identical runs.

**Key observations:**

- **tvm_ffi cold-novel compile is ~2 s** vs cuda_agent's ~91 s (~45× faster).
  Mostly explained by header surface: torch include tree is ~40 MB,
  tvm_ffi include tree is ~844 KB (~47×).
- **tvm_ffi's mutant cost ≈ tvm_ffi's warm cost.** Either `tvm_ffi.cpp.build()`
  has its own content-hashed cache that survives across our random work_dirs,
  or the absolute compile is genuinely cheap enough that caching wouldn't move
  the needle.
- cuda_agent's per-object cache (after stable-ext_name change) is doing what
  it should: 2 hits on identical reruns, 2 misses on novel content.
- The artifact-cache layer is not engaged in either run (`compile_artifact_cache_enabled=false`).

**Implication for further work:**

- For cuda_agent, the next lever is **PCH for `<torch/extension.h>`** — it's
  the only thing that can cut the ~90 s cold-novel cost since neither cache
  can hit when content is genuinely new.
- For tvm_ffi, no further compile-time work is worth doing at this kernel
  size. Any optimization budget goes elsewhere.
