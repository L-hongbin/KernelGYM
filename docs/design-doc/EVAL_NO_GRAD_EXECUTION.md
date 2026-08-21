# Eval plus No-Grad Execution

KernelGym uses one fixed forward-execution policy for KernelBench correctness and timing: every reference and candidate model runs in `model.eval()` mode, every forward runs under `torch.no_grad()`, and FP32 math runs with TF32 enabled.

## Policy

| Stage | Module mode | Gradient mode | FP32 math |
| --- | --- | --- | --- |
| correctness reference | `eval()` | `no_grad()` | TF32 |
| correctness candidate | `eval()` | `no_grad()` | TF32 |
| candidate warmup/timing/profile | `eval()` | `no_grad()` | TF32 |
| reference warmup/timing | `eval()` | `no_grad()` | TF32 |
| profiling-only retry | caller-provided eval model | `no_grad()` | TF32 |

`torch.inference_mode()` is deliberately not part of the runtime policy. This keeps ordinary tensor view/version behavior and avoids inference tensors escaping through stateful models while still preventing backward-graph construction.

Calling `eval()` is an intentional benchmark semantic change. Dropout is disabled, BatchNorm uses running statistics, and custom modules observe `self.training == False`. Both sides receive the same mode.

## Implementation

`kernelgym/toolkit/kernelbench/execution_policy.py` defines policy version `eval_no_grad_tf32_v2`, applies `eval()`, scopes TF32 state, and records:

- `execution_policy=eval_no_grad_tf32_v2`
- `model_mode=eval`
- `grad_mode=no_grad`
- `fp32_math_mode=tf32`

Correctness prepares both models after moving them to the selected device. Candidate performance, Triton detection, and reference-only timing also prepare their model explicitly so correctness-skipped and standalone timing paths remain consistent. The CUDA-event timing helper and profiling-only retry scope all forwards with `no_grad()` and the shared TF32 context. Each context restores the prior process-wide backend settings on exit.

## Cache Fences

Reference timings generated before this policy are not comparable because module, gradient, or FP32 math semantics may differ. Reference-runtime keys include the execution-policy version, entries store it, and preload ignores legacy rows without a matching version. Server request hashes also include the policy version so completed results from the previous execution semantics are not reused.

## Validation

Categorized regression tests live under `tests/kernelbench/execution_modes/`. They verify recursive eval mode, no-grad correctness forwards, candidate and standalone-reference timing preparation, no-grad CUDA-event and profiling-only windows, Triton detection, and cache fencing. Test-scope documentation is kept in `docs/testing/KERNELBENCH_EXECUTION_MODES.md`, and the superseded inference-mode characterization is retained only as recorded evidence in `docs/evidence/kernelbench/`.
