# Scoped TF32 Execution Policy

Date: 2026-08-21

## Outcome

KernelBench correctness, candidate/reference timing, inline profiling, and profiling-only retry now run under the same scoped TF32 policy plus `model.eval()` and `torch.no_grad()`. FP32 comparisons use `atol=rtol=1e-3`; FP16 and BF16 remain at `1e-2`. The context restores the PyTorch backend state that it changes on both normal and exceptional exit.

`EXECUTION_POLICY_VERSION` changed to `eval_no_grad_tf32_v2`, fencing request-result and reference-timing caches created under the prior arithmetic policy. `kernel_simple` explicitly opts out of the new TF32 timing default.

## Validation

All source edits and review ran on the current development machine. GPU tests ran on debug host `.17`; no source file was edited there.

- Relevant correctness, execution-mode, timing, profiling, cache, and integration tests before review closure: `19 passed, 0 skipped`.
- After adding the PyTorch 2.11 restoration regression, the complete TF32 policy test file passed: `5 passed`.
- Full suite with the Redis fixture and TVM FFI integration enabled: `568 passed, 0 failed`.
- Current-machine pre-commit suite, including Ruff lint and format checks: passed.

Some combined invocations encountered a transient CUDA driver-initialization warning and skipped four cases. A fresh process previously passed the four GPU timing cases, the relevant run recorded above had no skips, and the full suite completed without failures.

## PyTorch 2.11 State Check

Kimi K3 review requested a direct check of the relationship between the PyTorch 2.9+ `fp32_precision` API and the legacy global matmul-precision view. On `.17` with PyTorch `2.11.0+cu129`, the observed state transition was:

```text
before: cuda.matmul.fp32_precision=none, get_float32_matmul_precision()=highest
forced: cuda.matmul.fp32_precision=tf32
after:  cuda.matmul.fp32_precision=none, get_float32_matmul_precision()=highest
```

This confirms that saving and restoring the new per-backend attribute also restores the prior legacy view; calling the legacy getter while the new setting is active is intentionally avoided because PyTorch rejects mixed old/new API inspection. A regression test now asserts the before/after legacy view on installations that expose the new API.

## Review

A read-only current-machine `kimip` review used Kimi K3 with high thinking. It verified the required execution paths, tolerance changes, cache fence, metadata, documentation, and reviewer switch. Its two actionable checks were resolved as follows:

- The PyTorch 2.9+ restoration concern was closed by the PyTorch 2.11 state check and regression test above; no production-code change was required.
- The new evidence documents are ignored by the worktree's broad evidence rule, so they are force-added explicitly rather than leaving `INDEX.md` with dangling references.

The review also noted a pre-existing `kernel_simple` timing return-arity issue that is unchanged at `HEAD` and is outside this TF32 change.

## Deployment

No service was restarted and no deployment was performed.
