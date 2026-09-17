# CUDA stage-fault propagation — H100 — 2026-09-15

## Historical real case

Source: `benchmarks/review_evidence/runtime_sanitizer_service_ab_20260825_deployment_mismatch.json`, case `illegal_address_enabled`.

The historical response completed `kernel.correctness` and `kernel.incorrect_backend_usage_probe`, then failed at the worker commit barrier with `CudaFinalSyncError: CUDA error: misaligned address`. No `runtime_error` or Sanitizer result survived in the response.

## Current source result

The same archived TVM-FFI reference and candidate were executed locally in a disposable Python process on `NVIDIA H100 PCIe`, with one correctness trial, no performance measurement, detailed correctness enabled, and Compute Sanitizer enabled.

Observed summary:

```json
{
  "compiled": true,
  "correctness": false,
  "runtime_error_stage": "custom_forward",
  "sanitizer_trigger": "correctness_runtime_error",
  "sanitizer_status": "issues_found",
  "sanitizer_checks": ["memcheck"],
  "issue_count": 1
}
```

This run caught the asynchronous fault at an existing correctness synchronization point. The source change also prevents actionable CUDA memory/synchronization faults from being swallowed by profiling, performance, memory, or the incorrect-output usage probe.

The pipeline now crosses the worker-captured low-level CUDA barrier once after all evaluator timing windows and before returning. A successful barrier is recorded in worker-local state: single-use workers publish directly, while reusable workers skip the old pre-cleanup barrier and retain only the required post-cleanup barrier. Therefore the successful-task synchronization count is unchanged, and neither candidate nor reference timing includes the moved barrier.

An H100 run injected `CUDA error: misaligned address` from this final trusted barrier after an otherwise-correct safe TVM-FFI case. The callback was invoked exactly once; the response was converted to `correctness=false`, `runtime_error_stage=finalize`, and `runtime_sanitizer_trigger=finalize_runtime_error`. The focused `memcheck` replay completed cleanly, as expected because this test injected the barrier error rather than an actual faulty kernel.

A second H100 A/B run used the same safe kernel with `return_detail_correctness=false`. Baseline and trusted-barrier runs were both correct, both returned an empty `runtime_sanitizer`, and their non-timing metadata key sets were identical; the trusted barrier was invoked exactly once.

## Regression checks

- `pytest -q tests/workers/test_subprocess_pool.py tests/test_compute_sanitizer.py tests/kernelbench/test_cuda_stage_fault_propagation.py`: 132 passed.
- `RUN_COMPUTE_SANITIZER_INTEGRATION=1 pytest -q tests/test_compute_sanitizer_gpu.py::test_tvm_ffi_sanitizer_runs_after_correctness_runtime_failure`: passed.
- `RUN_COMPUTE_SANITIZER_INTEGRATION=1 pytest -q tests/test_compute_sanitizer_gpu.py::test_tvm_ffi_sanitizer_reports_issue_after_output_mismatch`: passed.
- Ruff checks passed for all touched Python files.

The running service was not restarted, so this evidence validates the source tree rather than the deployed worker generation.
