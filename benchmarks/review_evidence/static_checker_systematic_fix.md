# Static checker systematic-fix evidence

## Scope

Implementation worktree used during review and removed after integration: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-static-checker-systematic`

Branch/base: `fix/static-checker-systematic` at `681171573cdf731cca41f0553b4196a0c555dcce`

Integration target: `dev_csl`

This evidence accompanies the language-aware static-checker change. Python model code is parsed with `ast`, which naturally excludes comments, literals, and docstrings from call analysis. Every native source is scanned separately after C/C++ comments and literals are masked. Python and native sources are never concatenated into an undifferentiated regex input.

## Isolated replay (2026-09-03)

| Input | Result |
| --- | --- |
| MinGPT/NetVLAD-style `math.sqrt(...)`, including `math.sqrt(feature_size)` | pass |
| imported `tvm_ffi_extension as e; e.gelu(x)` with validated module | pass |
| `torch.ops.cuda_extension.gelu(x)` and relative CUDA-extension import | pass after backend precheck validates the extension form |
| `x.sqrt()` | static pass; runtime ATen gate is required |
| `config.max()` or a comprehension target named `torch` | static pass |
| `import torch as t; t.sum(x)` | `framework_compute` |
| `torch.ops.aten.sum.default(x)` | `framework_compute` |
| `e = torch; e.sum(x)` after extension import | `framework_compute` |
| empty method with `pass` | pass |
| `try`/`except` fallback | `code_bypass` |
| `try`/`finally` cleanup | pass |
| docstring/comment containing `x.sqrt()` | pass |
| `x.to(torch.float16)` | `precision_downgrade` |
| `tl.astype(x, tl.float16)` | `precision_downgrade` |
| native string/comment containing `at::sum` | pass |
| native `at::sum(x)` or declared `torch::Tensor x; x.sqrt()` | `framework_compute` |
| structural include/export markers inside comments, strings, or C++ raw strings | ignored |

`python -m compileall`, `ruff format --check`/`ruff check`, and `git diff --check` passed for all changed files. On node17 (Python 3.12.3, Torch 2.11.0+cu129, pytest 9.0.3, CUDA available), the focused regression command passed: **109 passed**:

```text
pytest -q tests/kernelbench/workflow/test_static_checker_language_aware.py tests/kernelbench/workflow/test_precision_passthrough.py tests/kernelbench/backends/test_reward_schema_and_cuda_agent.py tests/kernelbench/profiling/test_aten_decoy_detection.py tests/kernelbench/correctness/test_correctness_gpu.py
```

The whole repository suite passed after deselecting one proven pre-existing worker test: **628 passed, 5 skipped, 1 deselected** out of 634 collected tests. `tests/workers/test_subprocess_pool.py::test_shutdown_returns_false_when_process_group_proof_fails` fails identically on a clean detached worktree at the base commit; neither that test nor its implementation is modified by this branch.

The preferred cross-model reviewer was unavailable, the fallback reviewer repeatedly lost its service connection, and the final Kimi CLI was not installed. Per the review skill: `Cross-model review timed out; self-review substituted.` The adversarial self-review found and fixed profiler-stop failure propagation and raw-string include detection before the final test runs.

## Security boundary

The user-selected B+ policy removes generic unknown-receiver framework-compute matching. Correctness now marks an unavailable or invalid candidate-forward ATen capture—including initialization, extraction, or profiler-stop failure—as `ATEN_DETECTION_UNAVAILABLE`, a policy violation/decoy, so it cannot receive valid reward. Explicit PyTorch/ATen provenance remains statically blocked. FP32 precision checks intentionally keep unknown `.half()`/`.float16()` calls fail-closed because the runtime compatibility allowlist permits tensor casts and cannot prove precision preservation.

The ATen gate covers candidate `forward` calls only. Constructor-stage unknown-receiver compute remains an explicitly accepted gap; this change intentionally does not add `ModelNew.__init__` profiling. Explicitly disabling correctness or decoy detection also disables the runtime half of B+; reward-producing defaults enable both, while the switches remain diagnostic opt-outs.
