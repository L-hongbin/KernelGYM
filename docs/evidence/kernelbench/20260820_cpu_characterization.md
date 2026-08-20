# Inference-Mode CPU Characterization — 2026-08-20

Status: historical characterization evidence only. KernelGym subsequently adopted `model.eval()` plus `torch.no_grad()` and does not use inference mode in the runtime policy.

## Environment

- PyTorch: `2.11.0+cu129`
- CUDA devices visible to this shell: `0`
- KernelBench data: sibling `KernelBench-oldsize/KernelBench` checkout
- Model policy in reference comparisons: fresh models with `model.eval()`

## Command

```bash
PYTHONPATH=$PWD:$PWD/.venv/lib/python3.12/site-packages /home/csl001898@intellif.com/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -m pytest -q tests/kernelbench/execution_modes
```

## Result

All 16 tests passed. Pytest emitted one unrelated warning because the existing `.pytest_cache` directory is not writable from this shell.

The generic characterization tests confirmed that inference tensors have no readable version counter, reject mutation and `requires_grad_(True)` after leaving inference mode, and cannot be escaped with a nested `torch.enable_grad()` in the way `no_grad` can.

Ten reduced-shape real KernelBench references from Level 1, Level 2, and Level 3 produced matching outputs under `eval() + no_grad()` and `eval() + inference_mode()` when each mode received a fresh identically seeded model and inputs. The cases cover pointwise, normalization, cumulative reduction, scaled-dot-product attention, convolution plus normalization, dropout, MLP, stateful VanillaRNN, and LSTM paths.

The real `level3/33_VanillaRNN.py` reference exposed a concrete fallback hazard. Its forward assigns a newly computed tensor to `self.hidden`. An inference-mode forward therefore leaves `self.hidden` as an inference tensor. Reusing that model under `no_grad` raises `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed` at the next `self.hidden.copy_`. Reconstructing the model before the `no_grad` fallback succeeds.

## Remaining Gap

This shell exposes no CUDA device, so these tests do not yet cover CUDA-only operator dispatch, cuDNN/cuBLAS behavior, Triton detection, generated candidate kernels, or the full-size KernelBench corpus. A GPU sweep remains necessary before changing the runtime default.
