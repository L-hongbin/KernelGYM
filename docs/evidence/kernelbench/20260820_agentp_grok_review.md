# Agentp Grok 4.6 XHigh Review — 2026-08-20

## Reviewer

The review ran read-only through `agentp --mode ask --model cursor-grok-4.6-xhigh` against the complete working tree, including untracked files. It covered the KernelBench eval plus no-grad execution policy, reference-cache and request-hash fencing, categorized test moves, path-sensitive tests, and documentation references.

## Outcome

The reviewer reported **no production findings**. It confirmed that correctness, candidate timing/profiling, standalone reference timing, and Triton detection consistently use `model.eval()` plus `torch.no_grad()`; no runtime `torch.inference_mode()` path remains. It also confirmed that cache fencing is fail-closed and that the reorganized tests collect with corrected repository-root calculations and no stale path references.

## Residual Test Gaps Reported

1. Existing GPU tests did not directly assert `training=False` and inference mode disabled.
2. `eval_reference_only` lacked a direct regression for model preparation in eval mode.
3. Reference-cache preload had a legacy-policy rejection test but no current-policy positive test.
4. Correctness and timing CPU policy tests asserted gradients disabled but did not directly assert inference mode disabled.

## Follow-up

All four gaps were addressed after the review: the CPU policy observations now include `torch.is_inference_mode_enabled()`, standalone reference timing and current nested-policy preload have direct tests, and the GPU timing suite now exercises the production candidate performance entry while asserting `(training, grad_enabled, inference_enabled) == (False, False, False)` on every CUDA forward.

The reviewer also noted two non-scoring side paths (`KernelBenchBackendBase.run` and generic `detect_cuda_usage`) that do not independently apply eval mode. They are outside the KernelBench scoring pipeline and were not changed in this scope.
