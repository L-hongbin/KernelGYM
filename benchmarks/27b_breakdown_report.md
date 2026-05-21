# 27B 3-Binding Reward-Time Breakdown

End-to-end comparison of `/evaluate` reward latency across the three
submission-shape bindings the reward service supports, using real 27B
rollouts as the workload.

  * **cuda_agent** — `APPLY_BINDINGS` carries an explicit
    `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` block (one host C++
    binding TU compiled).
  * **pybind11** — `APPLY_BINDINGS` uses `REGISTER_BINDING(...)` + the
    framework's own `binding.cpp` + `binding_registry.h` scaffold
    (two host C++ binding TUs compiled).
  * **tvm_ffi** — `TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)` routed through
    the separate tvm_ffi backend (which calls `tvm_ffi.cpp.build`,
    not the manual-ninja path).

The 27B model produces all three shapes natively (mixedauto backend
on the upstream offline-eval run). No mechanical cross-binding
rewriting — each binding's samples are the **as-emitted LLM output**.

## TL;DR

`conc=8`, 74 paired KernelBench Level 1 problems, server timeout 180s:

| Metric | cuda_agent | pybind11 | tvm_ffi |
|---|---:|---:|---:|
| Compile time p50 (compiled) | 33.21 s | 33.68 s | **2.39 s** |
| Compile time p99 (compiled) | 45.40 s | 46.09 s | **3.23 s** |
| Correctness time p50 (compiled) | 0.26 s | 0.30 s | 0.54 s |
| Profile time p50 (correct-only) | 14.72 s | 17.35 s | 19.25 s |
| Completion rate | 65 % | 57 % | 53 % |
| Mean speedup (correct-only) | 1.19 | 1.20 | 1.02 |
| Elapsed mean (all-attempts, censored 180s) | 33.8 s | 37.6 s | **24.0 s** |

**Wall time for the whole 222-request pass**: conc=3 = 2399 s ≈
40 min, conc=8 = 938 s ≈ 16 min (2.56× speedup).

**Key honest framings** (per the adversarial codex audit — see
[review evidence](#codex-review-evidence) below):

  * **This benchmarks "reward cost of a 27B-emitted binding-shape on
    the same KernelBench problem," not "adapter overhead for identical
    kernels."** Each (problem, binding) row is a different LLM
    rollout, not the same code under different bindings.
  * The pybind11-vs-cuda_agent **completed-only** p50s look like
    pybind11 is 2.5× slower at perf-step (12.8 s vs 5.2 s) but that
    is a **correctness-mix artifact**: cuda_agent has more
    compiled-but-incorrect rows (perf step ≈ 0.0002 s when
    correctness fails). The 3 tables below filter on the right
    completion signal per metric. The genuine pybind11 perf overhead
    is **~17 %** (17.35 s vs 14.72 s on correctness-passed rows).
  * `tvm_ffi` compile is genuinely **~14× faster** than the manual
    ninja path (`tvm_ffi.cpp.build` vs `cpp_extension`'s ninja).
    Header set is ~47× smaller and tvm_ffi avoids pybind11's template
    instantiation cost.
  * tvm_ffi's lower completion rate (53 % vs 65 %) is **tvm-specific
    failure**, not just hard problems: the 5 timeouts (`pid 1, 2, 9,
    13, 61` — matmul / batched-matmul / 3D-matmul / triu / Conv2d) all
    hang in tvm_ffi but cuda_agent gets `1, 2, 9` correct.

## Reproduction

The benchmark is self-contained from this repo. It needs:

  * The reward service healthy at `192.168.16.40:20111` (v1 profile,
    8 GPU workers, redis backend). `bash check_node.sh` verifies.
  * `ssh chenshuailin@192.168.16.40` access (used by the runner
    scripts to clear `/dev/shm` compile caches between passes).
  * Python 3.10+ for the runner / aggregator (stdlib only — no
    extra deps needed).
  * The upstream 27B offline-eval run pinned at
    `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen36-27b-l1fullset-mixedauto-t180opt-n8-tp2-seqs16-32k-24-r1.run.20260509-180712`.
    The committed `benchmarks/data_27b/samples_*.jsonl.xz` are the
    exact 74 × 3 rollouts the report was built from; you do NOT have
    to re-extract from upstream to reproduce.

### Step 1 — extract paired samples (optional, deterministic)

The committed `samples_*.jsonl.xz` + `manifest.json` already contain
the 74 paired rollouts. To re-extract from the upstream eval run:

```bash
python benchmarks/data_27b/extract.py
```

This walks the upstream `eval_outputs/problem_X_sample_Y/` tree,
classifies every turn-1 response by binding shape, picks the 74
problems covered by all three shapes, and writes:

  * `benchmarks/data_27b/samples_cuda_agent.jsonl.xz` (~608 KB, 74 rows)
  * `benchmarks/data_27b/samples_pybind11.jsonl.xz`   (~705 KB, 74 rows)
  * `benchmarks/data_27b/samples_tvm_ffi.jsonl.xz`    (~530 KB, 74 rows)
  * `benchmarks/data_27b/manifest.json`               (SHA-256 of
    upstream `graded_results_conversations.jsonl` + per-file SHA-256
    of the three xz outputs)

Per-row schema:

```json
{
  "uid":            "test_example_...",
  "problem_id":     58,
  "sample_id":      7,
  "binding":        "cuda_agent" | "pybind11" | "tvm_ffi",
  "backend":        "cuda_agent" | "tvm_ffi",
  "score":          0.63,
  "reference_code": "<contents of upstream reference.py>",
  "kernel_code":    "<turn[0].response — 3-section LLM output>"
}
```

### Step 2 — calibrate concurrency (optional)

```bash
bash benchmarks/calibrate_concurrency.sh
```

Runs the first 10 problems × 3 bindings = 30 requests at
`--concurrency 3, 8, 16` in sequence, clearing the manual-ninja
object cache + artifact caches on `.40` between passes. Writes
`benchmarks/results/calib_c{3,8,16}_*.jsonl`.

Result on `.40` (v1 profile, 8 GPU workers):

| Concurrency | Wall | per-task compile p50 | observation |
|---|---:|---:|---|
| 3 | 679 s | 33.5 s | underutilized: 5 GPUs sit idle |
| **8** | **340 s** | **28.0 s** | matches 8-GPU pool, fastest |
| 16 | 342 s | 28.8 s | queue saturation, fast tasks wait behind slow ones |

`conc=8` is locked for the official run.

### Step 3 — run the official 222-sample paired benchmark

```bash
bash benchmarks/run_official_27b.sh
```

Sequentially:

  1. Clear `/dev/shm/kernelgym/compile_cache/{manual_ninja_objects,
     cuda_agent_artifacts, tvm_ffi_artifacts}/*` on `.40`.
  2. Run all 222 requests at `--concurrency 3 --timeout 180
     --seed 2026`.
  3. Clear the same caches again.
  4. Run all 222 requests at `--concurrency 8 --timeout 180
     --seed 2026` (same seed → same problem visitation order).

Per-pass total wall: `c3 ≈ 40 min`, `c8 ≈ 16 min`.
Outputs:

  * `benchmarks/results/official_c3_27b_breakdown_{cuda_agent,pybind11,tvm_ffi}.jsonl`
  * `benchmarks/results/official_c8_27b_breakdown_{cuda_agent,pybind11,tvm_ffi}.jsonl`

Each row has the full `compile_timing` + `kg_*` field set plus
trial counts (`num_correct_trials`, `kg_kernel_perf_num_trials`,
`kg_kernel_perf_num_profile_trials`) so a re-audit can confirm each
binding ran the same correctness + perf machinery.

### Step 4 — aggregate

```bash
python benchmarks/aggregate_27b.py --tag official_c8 --censor-at 180 --markdown
python benchmarks/aggregate_27b.py --tag official_c3 --censor-at 180 --markdown
```

Renders status distribution, completed-only + all-attempts
(censored) percentile tables per metric, kernel-phase residual
table, and a backend-specific diagnostics section (manual-ninja-only
fields are gated so they cannot be mis-compared against tvm_ffi).

## Runner methodology

All requests use the same evaluation parameters:

```python
num_correct_trials = 3
num_perf_trials    = 20
num_warmup         = 3
perf_trim_count    = 0
timeout            = 180   # server-side per-task ceiling
priority           = "normal"
entry_point        = "Model"
force_refresh      = True  # bypasses per-task result cache
run_performance    = True
```

`force_refresh=True` means the per-request result cache is bypassed,
so every (sample, binding) actually exercises the compile → load →
correctness → perf pipeline. The lower-level object cache + compile
artifact cache are off (`compile_artifact_cache_enabled=false` on
every row; the manual-ninja per-object cache misses on first compile,
hits would only matter if the same uid were sent twice, which the
runner's resume-by-uid logic prevents).

The runner uses a `ThreadPoolExecutor` of size `--concurrency` to
dispatch HTTP POSTs. The 8-GPU pool on `.40` handles up to 8 tasks in
flight at once via redis queues; concurrencies above 8 just queue
behind the workers.

`seed=2026` shuffles the 74 problem indices once, then visits them in
that fixed order across all three bindings — so for any problem the
three binding submissions arrive within seconds of each other. This
keeps page-cache / FS-cache state comparable across bindings.

Cache clearing between passes is handled by the official script and
the calibration script. It targets the local `/dev/shm` compile-layer
caches; the redis result cache is bypassed via `force_refresh=True`.

## Results

All three tables below filter each row by the appropriate completion
signal so the percentile is not contaminated by rows where that
phase didn't run.

### 1. Compile time — `kg_kernel_backend_compile_s`

Rows where `compiled=True` (kernel + binding compiled, whether or
not correctness later passed). For cuda_agent / pybind11 the
backend is the manual-ninja `cpp_extension` path; for tvm_ffi it is
`tvm_ffi.cpp.build`.

| Binding | n | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| `cuda_agent` | 48 | 33.21 s | 41.73 s | 45.40 s |
| `pybind11`   | 42 | 33.68 s | 41.63 s | 46.09 s |
| `tvm_ffi`    | 39 | **2.39 s** | **2.82 s** | **3.23 s** |

`cuda_agent` and `pybind11` are within ~1 % at every percentile —
the framework scaffold's extra binding TU compile is invisible in
aggregate (the wall is dominated by `<torch/extension.h>` parsing,
which both shapes pay). `tvm_ffi` is ~14× faster — `tvm/ffi/*`
include tree is ~844 KB vs `torch/include` ~40 MB, and the
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` registration avoids pybind11
template instantiation.

### 2. Correctness test time — `kg_kernel_correctness_s`

Rows where `compiled=True` (correctness step ran). The reward
service runs 3 trial pairs of (reference, kernel) outputs and
compares.

| Binding | n | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| `cuda_agent` | 48 | 0.26 s | 0.62 s | 0.79 s |
| `pybind11`   | 42 | 0.30 s | 0.58 s | 0.69 s |
| `tvm_ffi`    | 39 | 0.54 s | 0.71 s | **7.22 s** |

Sub-second across the board for all three bindings. `tvm_ffi` p99 =
7.22 s is a single outlier kernel whose execution wall during
correctness happens to be long (still 3 trials, same machinery,
just an expensive kernel call).

### 3. Profile (perf-trial) time — `kg_kernel_performance_step_s`

Rows where `correctness=True` (profile only runs if correctness
passed). Includes 3 warmup + 20 perf-trial + 10 profiler-trial
runs.

| Binding | n | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| `cuda_agent` | 26 | 14.72 s | 22.73 s | 25.56 s |
| `pybind11`   | 30 | 17.35 s | 26.89 s | 32.35 s |
| `tvm_ffi`    | 27 | 19.25 s | 27.68 s | **104.47 s** |

`pybind11` is ~17 % slower than `cuda_agent` at p50 and ~18 % at
p90 — the only metric where pybind11's extra dispatch path (through
the framework's `BindingRegistry::applyBindings` lookup) shows up
in aggregate. Note this is **NOT** the 2.5× difference a naive
"completed-only" filter would imply — that artifact comes from
cuda_agent's larger pool of compiled-but-incorrect rows whose
profile step never ran.

`tvm_ffi` p99 = 104 s is 1-2 outlier kernels that are dramatically
slower than the torch reference. Censoring at the 180 s timeout
caps this at 180; the raw uncensored value is preserved here for
diagnostic visibility.

## Outcome distribution

| Binding | Total | Compiled | Correct | Failed | Timeout | Mean speedup (correct-only) |
|---|---:|---:|---:|---:|---:|---:|
| `cuda_agent` | 74 | 48 (65 %) | 26 | 22 | 0 | **1.19** |
| `pybind11`   | 74 | 42 (57 %) | 30 | 26 | 1 | **1.20** |
| `tvm_ffi`    | 74 | 39 (53 %) | 27 | 26 | 5 | **1.02** |

Plus ~4-5 rows per binding with `http_status=400` from the FastAPI
schema validator (kernel_code field too long, or other body
validation). These are counted as "failed" by the upstream pipeline
but lack a server-side status string; the aggregator counts them
under `status=null`.

**`cuda_agent` and `pybind11` produce essentially the same mean
speedup on correct kernels (1.19 vs 1.20)** — binding shape does
not bias the speedup metric. `tvm_ffi`'s lower 1.02 reflects worse
27B kernel quality at the same KernelBench problems (the model is
less good at TVM-FFI submissions, not that the binding makes the
kernel slow).

## Caveats (what this benchmark is NOT)

These are real and apply to anyone reading the headline numbers:

1. **Not "same kernel, different binding"** — each (problem,
   binding) row is a different LLM rollout. The model emits
   different C++ for cuda_agent vs pybind11 vs tvm_ffi formats. The
   comparison is "for a given problem, what's the reward cost of the
   shape the 27B picked?"
2. **74 / 100 sub-sampled** — only problems where the 27B emitted
   all three binding shapes (in at least one of its 8 rollouts) are
   included. 26 problems excluded; among those, every classified
   sample includes pybind11 — so the included set is mildly biased
   toward problems where the model emits a clean cuda_agent or
   tvm_ffi output.
3. **Lowest sample_id picked per (problem, binding)** —
   deterministic but not random. Mean sample_id for the picked
   rollouts: pybind11 1.23, cuda_agent 2.08, tvm_ffi 2.62. If
   rollout index correlates with prompt warmth or any sampling
   effect, that's a sample-selection bias.
4. **Completed-only filters can mislead** — see the aggregator's
   `aggregate_official_c8.md` for the alternative "all-attempts
   censored at 180 s" table. The 3 tables above already filter on
   the right completion signal per metric.
5. **Per-row compile_s noise is ~50 %** — same kernel on a different
   pass can compile in 23 s vs 41 s (max delta in our two passes).
   Aggregate p50/p90/p99 are stable to within ~5 % across passes,
   though.

## Concurrency stability — `conc=3` vs `conc=8`

Per-metric aggregate comparison across the two passes (all metrics
in seconds):

| Metric | binding | p50 c3 | p50 c8 | Δ p50 |
|---|---|---:|---:|---:|
| compile | cuda_agent | 31.57 | 33.21 | +1.6 |
| compile | pybind11   | 31.92 | 33.68 | +1.8 |
| compile | tvm_ffi    | 2.39  | 2.39  | 0.0 |
| correct | cuda_agent | 0.26  | 0.26  | 0.0 |
| correct | pybind11   | 0.32  | 0.30  | −0.02 |
| correct | tvm_ffi    | 0.49  | 0.54  | +0.05 |
| profile | cuda_agent | 15.04 | 14.72 | −0.32 |
| profile | pybind11   | 16.34 | 17.35 | +1.01 |
| profile | tvm_ffi    | 18.03 | 19.25 | +1.22 |

**Aggregates are stable to ~5 %** between the two concurrency
settings. Per-row noise is larger (up to 25 s for individual
cuda_agent compiles), but the distribution stays put: same number
of fast outliers, same number of slow outliers, randomly assigned
across uids.

**Status counts are byte-identical across the two passes for all
three bindings**, so the comparison itself is deterministic — only
timing varies with concurrency.

## Codex review evidence

This benchmark went through two independent adversarial codex
reviews (process and results). Both flagged the same correctness-
mix artifact in completed-only p50; both confirmed there is no
silent shortcut for tvm_ffi anywhere in the reward service code
path. Evidence files:

  * `artifacts/reviews/concurrency_tvm_ffi_integrity_evidence.json`
    — calibration-phase audit, including direct redis-side recovery
    of trial counts for completed tvm_ffi rows.
  * `benchmarks/review_evidence/official_27b_review_evidence.json`
    — pairing / sampling / queue-delta / residual / c3-c8
    consistency evidence for the official runs.
  * `benchmarks/review_evidence/official_27b_perf_step_correctness_summary.json`
    — perf-step split by completed vs correct-only vs
    incorrect-completed, the data that surfaced the
    correctness-mix artifact.
  * `artifacts/reviews/official_27b_3binding_adversarial_review/`
    — full results-review bundle (paired comparison CSVs, example
    kernel-code snippets where pybind11 is correct and cuda_agent
    is not, tvm_ffi timeout per-pid breakdown, c3/c8 row-level
    diff CSV).

## Files reference

```
benchmarks/
├── 27b_breakdown_report.md         # this file
├── data_27b/
│   ├── extract.py                  # paired-by-problem extractor
│   ├── manifest.json               # upstream SHA-256, per-sample SHA-256, classification rule
│   ├── samples_cuda_agent.jsonl.xz # 74 rows (xz)
│   ├── samples_pybind11.jsonl.xz   # 74 rows (xz)
│   ├── samples_tvm_ffi.jsonl.xz    # 74 rows (xz)
│   ├── aggregate_official_c3.md    # full c3 aggregate (rendered)
│   └── aggregate_official_c8.md    # full c8 aggregate (rendered)
├── run_27b_breakdown.py            # threaded runner (resume-by-uid, interleaved order)
├── aggregate_27b.py                # percentile aggregator
├── calibrate_concurrency.sh        # conc=3/8/16 calibration
├── run_official_27b.sh             # official 222 × {c3, c8} run
├── results/                        # JSONL outputs (gitignored)
└── review_evidence/                # codex review evidence files
```
