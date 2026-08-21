# Scoped TF32 Execution

KernelGym explicitly enables TF32 for KernelBench FP32 correctness, timing, and profiling windows. Each window restores the PyTorch process state on exit so one evaluation cannot leak backend settings into another task.

## Policy

Correctness and timing use the same execution policy:

| Stage | Model mode | Autograd | FP32 math |
| --- | --- | --- | --- |
| correctness | `eval()` | `no_grad()` | TF32 enabled |
| candidate/reference timing | `eval()` | `no_grad()` | TF32 enabled |
| profiling and profiling retry | `eval()` | `no_grad()` | TF32 enabled |

KernelBench FP32 correctness uses `atol=1e-3` and `rtol=1e-3`. FP16 and BF16 remain at `1e-2`. The separate `kernel_simple` toolkit keeps its existing per-case tolerance and timing behavior.

## Implementation

`kernelgym/toolkit/kernelbench/execution_policy.py` owns the shared `tf32_execution_context`. On PyTorch 2.9+ it uses the per-operator APIs:

- `torch.backends.cudnn.conv.fp32_precision = "tf32"`
- `torch.backends.cuda.matmul.fp32_precision = "tf32"`

On older PyTorch it uses the legacy equivalents:

- `torch.backends.cudnn.allow_tf32 = True`
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.set_float32_matmul_precision("high")`, when available

The context records the prior values, applies TF32, and restores the prior values in `finally`, including when execution raises. Correctness metadata uses the `correctness_tf32_*` fields; timing and profiling use `timing_tf32_*` and `profiling_tf32_*` respectively.

The execution-policy version includes this FP32 math change. Request hashes and reference-runtime cache keys therefore reject results produced under the previous true-FP32 correctness policy.

## Non-Goals

- TF32 is not enabled permanently for the worker process.
- Tensor dtype is unchanged; inputs and outputs remain `torch.float32`.
- The policy does not change FP16/BF16 tolerance.
- The policy does not change `kernel_simple` defaults.
