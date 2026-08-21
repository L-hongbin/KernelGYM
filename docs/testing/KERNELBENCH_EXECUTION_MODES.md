# KernelBench Execution-Mode Testing

The execution-policy tests live in `tests/kernelbench/execution_modes/`. The adopted runtime policy is `model.eval()` plus `torch.no_grad()` with scoped TF32 enabled for FP32 math.

`test_eval_no_grad_policy.py` covers the active policy, pipeline preparation, profiling retry, and cache fences. The superseded inference-mode characterization remains available as historical evidence rather than executable regression code.

Recorded run evidence lives under `docs/evidence/kernelbench/`, outside the unit-test tree.

Run only this category with:

```bash
pytest -q tests/kernelbench/execution_modes
```
