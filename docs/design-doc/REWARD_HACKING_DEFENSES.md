# Reward Hacking Defenses

This document records the current harness-level defenses against reward hacking in KernelBench evaluation.

## Threat Model

Generated kernels are untrusted code. They may try to pass correctness without doing the intended computation, hide runtime from timing, or manipulate the evaluator's CUDA/PyTorch state. The reward harness should catch common accidental and adversarial false positives while keeping the default reward path practical enough for large GPU tasks.

## Correctness Defenses

### Per-forward RNG parity

For each correctness trial, the harness seeds input generation and then resets the same trial seed immediately before each of the reference and candidate forwards. This makes stochastic reference and candidate implementations consume the same PyTorch RNG sequence even though they run sequentially. The reset is an independent correctness-policy change; it is not part of ATen fallback detection.

### Reference before custom

Correctness trials deliberately run the reference model before the custom model. The reference result is computed while only trusted benchmark code has run for the current trial. This avoids a class of hacks where custom code mutates process-wide PyTorch or CUDA state before the reference executes, for example:

- changing `torch.backends.cuda.matmul.allow_tf32`;
- changing cuDNN benchmark or deterministic settings;
- changing default dtype or default device;
- mutating inputs before the reference sees them.

The tradeoff is that reference forward can create and release temporary CUDA allocations before custom code runs. Those released blocks may remain in the PyTorch CUDA caching allocator, so a custom implementation that returns an unwritten `empty` tensor can sometimes reuse stale reference data. The reference cache poison below is the current pragmatic mitigation for that tradeoff.

### Reference output alias protection

`kernelgym/toolkit/kernelbench/correctness.py` checks whether the reference output aliases any input storage. If it does, the output is cloned before custom code runs. This prevents a custom kernel that mutates inputs in place from changing the stored reference answer before comparison.

Metadata:

- `correctness_reference_alias_clone_trials`
- `correctness_reference_alias_clone_trial_s`

### Reference cache poison

After the reference forward pass and any alias-protection clone, the harness allocates zero-filled scratch tensors with the same structure and shapes as the reference output:

```python
poison_scratch = _zero_poison_like(output)
torch.cuda.synchronize(device=device)
del poison_scratch
```

This is a pragmatic defense against CUDA caching allocator reuse. The concrete failure mode was a custom kernel returning `torch::empty` without writing data. In a reference such as `triu(matmul(A, B))`, the `matmul` intermediate can be released into the allocator cache before custom code runs. If the custom output reuses that cached block, stale reference data can make an unwritten output pass correctness.

The poison scratch is not a separate timed substage. It is part of correctness hardening and only records:

- `correctness_reference_cache_poison_enabled`

Limitations:

- This is not a formal guarantee. It targets output-shaped cached blocks and may not cover every released intermediate size.
- It preserves the existing `reference -> custom` order to avoid custom code influencing reference computation.
- Stronger guarantees require process isolation or an isolated allocator/pool, with higher runtime or memory cost.

## Timing Defenses

### ATen compute fallback legality gate

When decoy detection is enabled, every real `ModelNew` correctness forward is wrapped in a CPU-only `torch.profiler` context. Reference forwards run outside that context. The captured `aten::*` calls are checked against the explicit [MooreEval/MusaCoder Appendix J](https://arxiv.org/abs/2606.04847) allowlist in `kernelgym/toolkit/kernelbench/profiling.py`.

The published Appendix J list is preserved as a separate constant. KernelGYM adds a narrowly scoped PyTorch 2.11 compatibility set for modern dispatcher spellings of the same tensor-plumbing behavior: `aten::to`, `aten::_to_copy`, `aten::detach`, `aten::fill_`, `aten::zero_`, `aten::new_empty`, and `aten::empty_strided`. For example, a CUDA wrapper that calls `.float()` emits `aten::to`, `aten::_to_copy`, `aten::empty_strided`, and `aten::copy_`; treating those names as compute fallback would reject a legal wrapper even though Appendix J permits older `_cast_*`, allocation, and copy spellings.

Any other `aten::*` call is treated as a high-level compute fallback. A candidate that is numerically correct but invokes `aten::mm`, `aten::convolution`, `aten::sum`, or another unlisted ATen operator is returned with:

- `decoy_kernel=true`
- `policy_violation=true`
- `policy_violation_reason=DISALLOWED_ATEN_COMPUTE`
- `decoy_reason=DISALLOWED_ATEN_COMPUTE`
- `forbidden_aten_op_names` and per-trial operator evidence

ATen legality is categorical and does not use a CUDA-time percentage threshold. Profiler initialization/extraction failure sets `aten_detection_valid=false` and records the error; it does not create a decoy verdict.

### CUDA synchronization

Correctness and timing paths call `torch.cuda.synchronize(device=device)` after reference, custom, warmup, timing, and profiling loops. This prevents default-stream timing from returning before queued work is finished.

### CUDA event timing

Performance timing records CUDA events around the custom forward call and synchronizes the device before reading elapsed time. This is the primary kernel runtime measurement.

### Profiler coverage metadata

When profiling is enabled, the harness records captured CUDA kernel metadata and backend-specific coverage:

- `num_total_kernels`
- `num_custom_kernels`
- `custom_kernel_in_profiling`
- `custom_kernel_not_in_profiling`
- `total_kernel_cuda_time_in_profiling_us`
- `custom_kernel_cuda_time_in_profiling_us`
- `raw_custom_kernel_time_coverage`
- `coverage_measurement_valid`

For CUDA-Agent, TVM-FFI, and TileLang, expected custom kernel names come from backend profiling hints when available. Triton keeps its runtime launch hook. Coverage uses only actual CUDA device events; CPU-side ATen aggregates that repeat child CUDA time are excluded from the denominator.

Named custom-kernel CUDA-time coverage below 30% records `suspected_decoy=true`. Coverage below 0.1% additionally records `hard_decoy_coverage_candidate=true`, but does not hard-reject while direct extension-internal cuBLAS/cuDNN provenance remains unavailable: unmatched CUDA time may be a legal vendor-library implementation. The hard legality decision is therefore driven by forbidden ATen compute; named-kernel coverage remains a conservative diagnostic/bypass signal.

`suspected_decoy` does not change correctness or `decoy_kernel` by itself; the response makes this explicit with `suspected_decoy_enforced=false` and `suspected_decoy_effect=DIAGNOSTIC_ONLY`. The current downstream reward client does not consume this field, and its default configurations have `coverage_reward.enable=false`, so low coverage currently has no training penalty. This is intentional until the profiler can attribute unmatched vendor-library kernels to the candidate extension call chain; restoring the old `num_custom_kernels == 0` hard gate would reject legal direct cuBLAS/cuDNN bindings.

### Profiler dropout retry

If profiling is enabled and returns no kernels, the performance step can retry profiling according to `settings.profiling_retry_count`. Empty profiler data is treated as a profiler reliability issue, not automatically as a decoy kernel.

## Static And Compile-Time Defenses

### Submission validation

`kernelgym/toolkit/validation.py` and backend-specific prechecks reject malformed model code and CUDA-Agent binding mistakes before runtime. CUDA-Agent validation checks embedded sections, binding registration syntax, and supported binding forms.

### Static checker

`kernelgym/toolkit/kernelbench/static_checker.py` blocks known direct evaluator tampering patterns such as assigning to `torch.cuda.synchronize`.

## Regression Tests

`tests/kernelbench/correctness/test_correctness_gpu.py` contains GPU-only tests for the cache-poison defense:

- With poison enabled, a custom model returning `torch.empty_like` after a reference with a same-shaped intermediate fails correctness.
- With poison monkeypatched off, the same custom model can reproduce the hacking behavior and incorrectly pass by reusing stale reference intermediate memory.
- A normal matching CUDA model still passes with poison enabled.
- Forbidden `aten::mm` is detected on the candidate correctness forward while allowlisted `aten::view` passes.
- A compiled CUDA extension whose wrapper calls `.float()` is correct and is not rejected; the test verifies the modern `aten::to`/`aten::_to_copy` dispatcher path on GPU.
- All candidate correctness trials are profiled, so a stateful implementation cannot defer an ATen fallback until a later trial.

`tests/kernelbench/profiling/test_aten_decoy_detection.py` covers the explicit allowlist, CUDA-device-only coverage extraction, low-coverage suspicion, and profiler-unavailable behavior.

These tests skip on hosts without PyTorch CUDA.

## Known Gaps

- Zero-like poison is heuristic and output-shape based. It does not prove all reference intermediates were overwritten.
- The default path does not isolate reference and custom code in separate processes.
- The default path does not snapshot or restore every possible PyTorch/CUDA global state mutation by custom code.
- cuBLAS/cuDNN status returns inside generated extensions cannot be checked by the Python harness unless the extension reports errors.
- Named CUDA-kernel coverage cannot yet prove whether unmatched GPU time came from extension-internal cuBLAS/cuDNN, so the 0.1% hard coverage gate remains intentionally disabled.

## Stronger Future Options

1. Run reference and custom correctness in separate worker processes, with reference outputs materialized outside the reference GPU context.
2. Use a private allocator or CUDA memory pool for reference temporaries, while accounting for the extra peak memory.
3. Add optional debug mode with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` to reduce allocator reuse during investigations.
4. Add excessive-speedup flags as a reward-side warning, similar to upstream KernelBench, without replacing correctness checks.
