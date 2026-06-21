# True-FP32 Correctness

KernelGym correctness checks now force CUDA float32 operators to run as true fp32 by default. Performance/profile stages restore the original PyTorch backend state and therefore continue to measure the deployment-like default path, including TF32 conv when PyTorch enables it.

## Problem

KernelBench compares custom kernels against a PyTorch reference model. On Ampere and newer GPUs, PyTorch/cuDNN convolution can execute float32 tensors with TF32 tensor cores by default. Matmul defaults are different in modern PyTorch: `cuda.matmul.allow_tf32` is normally off, so matmul references are usually true fp32.

This creates a correctness asymmetry:

- reference `nn.Conv*` can be TF32 even though tensors are `torch.float32`;
- ordinary custom CUDA kernels using `float` arithmetic remain true fp32 unless they explicitly call cuDNN/cuBLAS or TF32 tensor-core instructions;
- KernelBench fp32 tolerance is `1e-4`, while TF32 conv error can exceed that scale.

The result is a false negative: a semantically correct fp32 custom kernel can fail because it does not match the lower-precision TF32 reference closely enough.

## Policy

Correctness and performance use different precision policies:

| Stage | Precision policy | Rationale |
| --- | --- | --- |
| correctness | Force true fp32 for PyTorch CUDA fp32 ops | Use a stable oracle for semantic comparison. |
| performance/profile | Restore original PyTorch backend state | Measure the realistic runtime baseline, including default TF32 conv. |

The behavior is enabled by default. Set `KERNELGYM_CORRECTNESS_DISABLE_TF32=0` to opt out for compatibility investigations.

## Implementation

`kernelgym/toolkit/kernelbench/correctness.py` wraps model construction and correctness trials in `_true_fp32_correctness_context(metadata)`.

The context records and restores the PyTorch backend state. On PyTorch 2.9+ it prefers the new per-op precision APIs:

- `torch.backends.cudnn.conv.fp32_precision = "ieee"`
- `torch.backends.cuda.matmul.fp32_precision = "ieee"`

On older PyTorch it falls back to legacy flags:

- `torch.backends.cudnn.allow_tf32 = False`
- `torch.backends.cuda.matmul.allow_tf32 = False`
- `torch.set_float32_matmul_precision("highest")`, when available

The implementation deliberately avoids mixing old and new APIs for the same operator family because newer PyTorch can reject inconsistent policy combinations.

Metadata fields:

- `correctness_tf32_disabled`: whether the policy was active.
- `correctness_tf32_state_before`: observed backend state before forcing true fp32.
- `correctness_tf32_state_forced`: state changes applied for the correctness window.

## Non-Goals

- Do not change tensor dtype. Inputs and outputs remain `torch.float32`.
- Do not disable TF32 globally for the worker process after correctness returns.
- Do not force profile/reference timing to true fp32; that would change speedup semantics.
- Do not loosen fp32 tolerance to hide the issue. The stable policy is true-fp32 correctness at `1e-4`.

## Validation

Regression test:

```bash
PYTHONPATH=$PWD python3 -m pytest -q tests/test_kernelbench_tf32_correctness_policy.py
```

The test verifies that the context forces true-fp32 backend settings, restores the prior state, and can be disabled through `KERNELGYM_CORRECTNESS_DISABLE_TF32=0`.
