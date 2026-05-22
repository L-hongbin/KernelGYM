# 27B 3-binding breakdown — tag `official_c3`

Per binding: paired by problem_id (same KernelBench Level 1 problems across all three). Censored timing percentiles clamp at 180s (the server-side per-task timeout).

## Outcome distribution

| Binding | Total | Completed | Failed | Timeout | Runner-exc | Compiled | Correct | Decoy | Speedup>1 | Mean speedup (compiled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pybind11_inline` | 74 | 48 | 22 | 0 | 0 | 48 | 26 | 0 | 26 | 0.645 |
| `pybind11_registry` | 74 | 42 | 26 | 1 | 0 | 42 | 30 | 0 | 30 | 0.855 |
| `tvm_ffi` | 74 | 39 | 26 | 5 | 0 | 39 | 27 | 0 | 27 | 0.708 |

## All-attempts (censored at server timeout)

### `elapsed_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 74 | 33.81s | 32.79s | 52.49s | 76.43s |
| `pybind11_registry` | 74 | 34.81s | 37.87s | 67.87s | 130.87s |
| `tvm_ffi` | 74 | 3.52s | 25.86s | 45.62s | 180.00s |

### `kg_kernel_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 39.17s | 39.97s | 50.85s | 60.18s |
| `pybind11_registry` | 42 | 43.92s | 44.36s | 57.96s | 76.52s |
| `tvm_ffi` | 39 | 14.48s | 17.56s | 27.39s | 103.13s |

### `kg_kernel_backend_compile_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 58 | 30.16s | 26.41s | 38.03s | 40.96s |
| `pybind11_registry` | 50 | 29.97s | 26.77s | 38.05s | 41.76s |
| `tvm_ffi` | 44 | 2.37s | 2.19s | 2.79s | 2.98s |

### `kg_kernel_backend_load_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.00s | 0.00s | 0.00s | 0.00s |
| `pybind11_registry` | 42 | 0.00s | 0.00s | 0.00s | 0.00s |
| `tvm_ffi` | 39 | 0.00s | 0.00s | 0.00s | 0.01s |

### `kg_kernel_performance_step_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 7.67s | 7.70s | 17.90s | 23.96s |
| `pybind11_registry` | 42 | 13.12s | 12.12s | 24.34s | 37.50s |
| `tvm_ffi` | 39 | 11.54s | 14.35s | 24.58s | 93.84s |

### `kg_kernel_correctness_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.26s | 0.34s | 0.64s | 0.73s |
| `pybind11_registry` | 42 | 0.32s | 0.36s | 0.57s | 0.69s |
| `tvm_ffi` | 39 | 0.49s | 0.72s | 0.72s | 7.15s |

### `kg_reference_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 26 | 8.71s | 7.16s | 14.30s | 18.88s |
| `pybind11_registry` | 30 | 10.51s | 9.19s | 15.16s | 25.94s |
| `tvm_ffi` | 27 | 11.25s | 9.26s | 15.10s | 16.85s |

### `wg_pool_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 58 | 36.84s | 33.99s | 50.90s | 60.19s |
| `pybind11_registry` | 51 | 39.09s | 37.69s | 58.08s | 75.14s |
| `tvm_ffi` | 44 | 13.30s | 16.05s | 27.97s | 97.86s |

## Completed-only

### `elapsed_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 40.58s | 42.73s | 54.80s | 76.68s |
| `pybind11_registry` | 42 | 50.59s | 51.59s | 75.37s | 100.99s |
| `tvm_ffi` | 39 | 16.54s | 24.82s | 43.78s | 119.41s |

### `kg_kernel_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 39.17s | 39.97s | 50.85s | 60.18s |
| `pybind11_registry` | 42 | 43.92s | 44.36s | 57.96s | 76.52s |
| `tvm_ffi` | 39 | 14.48s | 17.56s | 27.39s | 103.13s |

### `kg_kernel_backend_compile_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 31.57s | 31.91s | 39.09s | 41.00s |
| `pybind11_registry` | 42 | 31.92s | 31.86s | 38.13s | 41.90s |
| `tvm_ffi` | 39 | 2.39s | 2.47s | 2.81s | 2.99s |

### `kg_kernel_backend_load_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.00s | 0.00s | 0.00s | 0.00s |
| `pybind11_registry` | 42 | 0.00s | 0.00s | 0.00s | 0.00s |
| `tvm_ffi` | 39 | 0.00s | 0.00s | 0.00s | 0.01s |

### `kg_kernel_performance_step_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 7.67s | 7.70s | 17.90s | 23.96s |
| `pybind11_registry` | 42 | 13.12s | 12.12s | 24.34s | 37.50s |
| `tvm_ffi` | 39 | 11.54s | 14.35s | 24.58s | 93.84s |

### `kg_kernel_correctness_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.26s | 0.34s | 0.64s | 0.73s |
| `pybind11_registry` | 42 | 0.32s | 0.36s | 0.57s | 0.69s |
| `tvm_ffi` | 39 | 0.49s | 0.72s | 0.72s | 7.15s |

### `kg_reference_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 26 | 8.71s | 7.16s | 14.30s | 18.88s |
| `pybind11_registry` | 30 | 10.51s | 9.19s | 15.16s | 25.94s |
| `tvm_ffi` | 27 | 11.25s | 9.26s | 15.10s | 16.85s |

### `wg_pool_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 39.60s | 40.64s | 51.78s | 60.25s |
| `pybind11_registry` | 42 | 45.11s | 45.06s | 59.64s | 76.63s |
| `tvm_ffi` | 39 | 14.55s | 18.10s | 28.87s | 103.25s |

## Kernel-phase residual

`kg_kernel_total_s − (backend_compile + backend_load + correctness + performance)`. A non-zero residual means kg_kernel_total includes work not surfaced by the per-phase fields.

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.01s | 0.02s | 0.04s | 0.05s |
| `pybind11_registry` | 42 | 0.01s | 0.02s | 0.04s | 0.05s |
| `tvm_ffi` | 39 | 0.01s | 0.02s | 0.04s | 0.06s |

## Backend-specific diagnostics (manual ninja path only)

These fields exist for the cuda_agent + pybind11 manual-ninja compile path; tvm_ffi compiles via `tvm_ffi.cpp.build` and exposes neither. Do NOT use these to compare against tvm_ffi.

### `manual_ninja_build_wall_sec`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 30.83s | 31.26s | 38.33s | 40.22s |
| `pybind11_registry` | 42 | 31.53s | 31.30s | 37.42s | 40.83s |
| `tvm_ffi` | 0 | — | — | — | — |

### `manual_ninja_import_wall_sec`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.00s | 0.00s | 0.00s | 0.00s |
| `pybind11_registry` | 42 | 0.00s | 0.00s | 0.00s | 0.00s |
| `tvm_ffi` | 0 | — | — | — | — |
