# 27B 3-binding breakdown — tag `official_c8`

Per binding: paired by problem_id (same KernelBench Level 1 problems across all three). Censored timing percentiles clamp at 180s (the server-side per-task timeout).

## Outcome distribution

| Binding | Total | Completed | Failed | Timeout | Runner-exc | Compiled | Correct | Decoy | Speedup>1 | Mean speedup (compiled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pybind11_inline` | 74 | 48 | 22 | 0 | 0 | 48 | 26 | 0 | 26 | 0.645 |
| `pybind11_registry` | 74 | 42 | 26 | 1 | 0 | 42 | 30 | 0 | 30 | 0.856 |
| `tvm_ffi` | 74 | 39 | 26 | 5 | 0 | 39 | 27 | 0 | 27 | 0.708 |

## All-attempts (censored at server timeout)

### `elapsed_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 74 | 35.56s | 33.79s | 59.15s | 76.98s |
| `pybind11_registry` | 74 | 37.57s | 37.56s | 61.38s | 108.93s |
| `tvm_ffi` | 74 | 8.27s | 24.03s | 32.13s | 180.00s |

### `kg_kernel_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 39.59s | 41.15s | 58.69s | 68.53s |
| `pybind11_registry` | 42 | 42.90s | 45.47s | 63.70s | 70.63s |
| `tvm_ffi` | 39 | 14.09s | 18.28s | 28.27s | 101.49s |

### `kg_kernel_backend_compile_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 58 | 29.77s | 27.39s | 41.39s | 45.22s |
| `pybind11_registry` | 50 | 30.85s | 27.74s | 40.84s | 45.60s |
| `tvm_ffi` | 44 | 2.38s | 2.20s | 2.81s | 3.22s |

### `kg_kernel_backend_load_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.00s | 0.00s | 0.00s | 0.00s |
| `pybind11_registry` | 42 | 0.00s | 0.00s | 0.00s | 0.00s |
| `tvm_ffi` | 39 | 0.00s | 0.00s | 0.00s | 0.00s |

### `kg_kernel_performance_step_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 5.16s | 7.68s | 20.70s | 25.15s |
| `pybind11_registry` | 42 | 12.80s | 12.07s | 23.44s | 31.74s |
| `tvm_ffi` | 39 | 11.38s | 15.05s | 25.18s | 92.19s |

### `kg_kernel_correctness_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.26s | 0.34s | 0.62s | 0.79s |
| `pybind11_registry` | 42 | 0.30s | 0.36s | 0.58s | 0.69s |
| `tvm_ffi` | 39 | 0.54s | 0.73s | 0.71s | 7.22s |

### `kg_reference_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 26 | 8.71s | 7.16s | 14.30s | 18.88s |
| `pybind11_registry` | 30 | 10.51s | 9.19s | 15.16s | 25.94s |
| `tvm_ffi` | 27 | 11.25s | 9.26s | 15.10s | 16.85s |

### `wg_pool_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 58 | 38.68s | 36.38s | 60.59s | 77.58s |
| `pybind11_registry` | 51 | 45.50s | 40.92s | 64.27s | 75.99s |
| `tvm_ffi` | 44 | 13.83s | 17.96s | 29.71s | 100.19s |

## Completed-only

### `elapsed_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 41.32s | 43.98s | 62.17s | 78.28s |
| `pybind11_registry` | 42 | 48.83s | 48.41s | 65.04s | 77.31s |
| `tvm_ffi` | 39 | 20.04s | 20.39s | 31.07s | 106.43s |

### `kg_kernel_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 39.59s | 41.15s | 58.69s | 68.53s |
| `pybind11_registry` | 42 | 42.90s | 45.47s | 63.70s | 70.63s |
| `tvm_ffi` | 39 | 14.09s | 18.28s | 28.27s | 101.49s |

### `kg_kernel_backend_compile_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 33.21s | 33.10s | 41.73s | 45.40s |
| `pybind11_registry` | 42 | 33.68s | 33.02s | 41.63s | 46.09s |
| `tvm_ffi` | 39 | 2.39s | 2.49s | 2.82s | 3.23s |

### `kg_kernel_backend_load_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.00s | 0.00s | 0.00s | 0.00s |
| `pybind11_registry` | 42 | 0.00s | 0.00s | 0.00s | 0.00s |
| `tvm_ffi` | 39 | 0.00s | 0.00s | 0.00s | 0.00s |

### `kg_kernel_performance_step_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 5.16s | 7.68s | 20.70s | 25.15s |
| `pybind11_registry` | 42 | 12.80s | 12.07s | 23.44s | 31.74s |
| `tvm_ffi` | 39 | 11.38s | 15.05s | 25.18s | 92.19s |

### `kg_kernel_correctness_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.26s | 0.34s | 0.62s | 0.79s |
| `pybind11_registry` | 42 | 0.30s | 0.36s | 0.58s | 0.69s |
| `tvm_ffi` | 39 | 0.54s | 0.73s | 0.71s | 7.22s |

### `kg_reference_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 26 | 8.71s | 7.16s | 14.30s | 18.88s |
| `pybind11_registry` | 30 | 10.51s | 9.19s | 15.16s | 25.94s |
| `tvm_ffi` | 27 | 11.25s | 9.26s | 15.10s | 16.85s |

### `wg_pool_total_s`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 40.89s | 43.70s | 61.90s | 78.09s |
| `pybind11_registry` | 42 | 48.34s | 48.08s | 64.53s | 77.16s |
| `tvm_ffi` | 39 | 19.75s | 20.05s | 30.57s | 106.02s |

## Kernel-phase residual

`kg_kernel_total_s − (backend_compile + backend_load + correctness + performance)`. A non-zero residual means kg_kernel_total includes work not surfaced by the per-phase fields.

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.01s | 0.02s | 0.02s | 0.05s |
| `pybind11_registry` | 42 | 0.01s | 0.01s | 0.02s | 0.04s |
| `tvm_ffi` | 39 | 0.01s | 0.01s | 0.02s | 0.04s |

## Backend-specific diagnostics (manual ninja path only)

These fields exist for the cuda_agent + pybind11 manual-ninja compile path; tvm_ffi compiles via `tvm_ffi.cpp.build` and exposes neither. Do NOT use these to compare against tvm_ffi.

### `manual_ninja_build_wall_sec`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 32.71s | 32.43s | 40.92s | 44.59s |
| `pybind11_registry` | 42 | 32.97s | 32.43s | 40.92s | 45.26s |
| `tvm_ffi` | 0 | — | — | — | — |

### `manual_ninja_import_wall_sec`

| Binding | n | p50 | mean | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| `pybind11_inline` | 48 | 0.00s | 0.00s | 0.00s | 0.00s |
| `pybind11_registry` | 42 | 0.00s | 0.00s | 0.00s | 0.00s |
| `tvm_ffi` | 0 | — | — | — | — |
