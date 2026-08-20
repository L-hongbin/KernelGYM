# Eval plus No-Grad Implementation Evidence — 2026-08-20

## Scope

The implementation applies policy `eval_no_grad_v1` to KernelBench reference and candidate correctness, candidate performance and profiling, Triton detection, and standalone reference timing. It also fences old reference-runtime and server result caches.

No service was restarted. The currently serving workers retain their previously loaded code until a controlled restart.

## Validation

```bash
PYTHONPATH=$PWD:$PWD/.venv/lib/python3.12/site-packages /home/csl001898@intellif.com/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -m pytest -q -p no:cacheprovider tests/kernelbench/execution_modes tests/kernelbench/profiling/test_profiling_trials.py tests/kernelbench/profiling/test_cupti_tsc_shim.py tests/kernelbench/workflow/test_precision_passthrough.py tests/server/test_task_manager_resource_queues.py::test_request_hash_ignores_identity_and_provenance_fields
```

Result: 68 passed, 1 skipped. The output contained existing Pydantic deprecation warnings but no test failures.

After the Triton detector was aligned from inference/grad dual execution to the single eval plus no-grad runtime path, the nine active-policy tests were rerun and all passed. The added regression test verifies that one warmup plus two measured Triton-detection steps invoke the model exactly three times, always with `training=False`, gradients disabled, and inference mode disabled.

Static validation:

```bash
.venv/bin/ruff check --no-cache kernelgym/toolkit/kernelbench/execution_policy.py kernelgym/toolkit/kernelbench/correctness.py kernelgym/toolkit/kernelbench/pipeline.py kernelgym/toolkit/kernelbench/timing.py kernelgym/toolkit/kernelbench/triton_detect.py kernelgym/workflow/reference_cache.py kernelgym/server/request_hash.py tests/kernelbench/execution_modes tests/kernelbench/profiling/test_profiling_trials.py tests/kernelbench/profiling/test_cupti_tsc_shim.py
git diff --check
```

Both checks passed.

The complete non-GPU, non-integration test selection also ran to completion. It reported three failures outside the changed execution-policy paths: the existing CUDA-Agent reusable-object source classifier, a worker CLI test that requires a visible CUDA device in this environment, and a subprocess process-group drain-proof test. Focused execution-policy and affected-path tests remained green.

## GPU Validation

The full suite was subsequently run on `ai-16-17` with one A800 exposed through `CUDA_VISIBLE_DEVICES=0`, PyTorch `2.11.0+cu129`, and the repository venv:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/bin/python -m pytest -q -p no:cacheprovider
```

The first GPU run produced 556 passed, 4 skipped, and the pre-existing CUDA-Agent reusable-source classifier failure. The classifier was subsequently fixed to reject module-name-bound source objects, with parameterized coverage for Pybind11, Boost.Python, direct `PyInit_` entry points, and a reusable ordinary CUDA source.

The final full run used the same A800 plus a disposable localhost Redis instance so all four opt-in Redis integration tests executed. Result: 567 passed, 1 skipped, and 0 failed. The remaining skip was the separately gated real TVM-FFI strict-link integration; rerunning that test with `KERNELGYM_RUN_TVM_FFI_LINK_INTEGRATION=1` passed. Across the full run and the explicit opt-in follow-up, all 568 collected tests executed successfully. The temporary Redis instance was shut down and its temporary directory removed after validation.

The focused policy suite was expanded following the Grok review to directly assert inference mode disabled, standalone reference timing eval preparation, current-policy cache preload, and the production candidate GPU timing context. The final focused run combined 11 active-policy tests with 4 real-CUDA timing tests and passed all 15. A full 250-problem KernelBench corpus sweep remains separate from this unit-test validation. Deployment and service restart require separate user approval.
