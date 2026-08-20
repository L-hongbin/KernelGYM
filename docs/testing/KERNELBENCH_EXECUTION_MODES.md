# KernelBench Execution-Mode Testing

The execution-policy tests live in `tests/kernelbench/execution_modes/`. The adopted runtime policy is `model.eval()` plus `torch.no_grad()`.

`test_eval_no_grad_policy.py` covers the active policy, pipeline preparation, profiling retry, and cache fences. `test_inference_mode_semantics.py` and `test_kernelbench_reference_modes.py` retain the characterization evidence that led to rejecting inference mode; they are not descriptions of the runtime policy.

The reference tests resolve data from `KERNELBENCH_DATA_ROOT`, then fall back to the sibling `KernelBench-oldsize/KernelBench` or `kernel_bench_verified/KernelBench` checkout. They skip when no reference checkout is available.

Recorded run evidence lives under `docs/evidence/kernelbench/`, outside the unit-test tree.

Run only this category with:

```bash
pytest -q tests/kernelbench/execution_modes
```
