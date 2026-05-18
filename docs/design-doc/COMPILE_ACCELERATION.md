# CUDA-Agent Compile Acceleration Design

Status: proposed target design
Date: 2026-05-18

This document defines the target CUDA-Agent compile acceleration design for the reward-only service.

## Goals

- Remove `torch.utils.cpp_extension.load(...)` from CUDA-Agent compilation.
- Compile through an explicit ninja build graph.
- Restrict CUDA code generation to the visible GPU architectures.
- Reuse compiled object files when source/header/build inputs are unchanged.
- Split CPU-heavy compile work from GPU execution work.
- Preserve CUDA-Agent parsing, validation, binding handling, profiling metadata, and reward semantics.
- Keep large artifacts on local fast storage; use Redis only for coordination and indexes.

## Target Architecture

The compile acceleration has four required parts:

1. **Manual ninja backend**
   CUDA-Agent writes `build.ninja`, runs ninja, and imports the built extension. This replaces `cpp_extension.load`.

2. **Object-level cache**
   Reusable object files are looked up before ninja runs. Cache hits are linked directly; only misses are compiled.

3. **CUDA architecture selection**
   Service startup honors an existing `TORCH_CUDA_ARCH_LIST`; otherwise it detects visible CUDA device capabilities and generates a deduplicated semicolon-separated value such as `8.9`. This prevents PyTorch extension builds from emitting unnecessary architectures.

4. **Compile/execute resource separation**
   CPU workers produce compile artifacts. GPU workers consume those artifacts and run correctness/performance.

The first three reduce compile work. The fourth reduces GPU worker occupancy caused by compilation.

## Compile Flow

CUDA-Agent compilation should run as:

1. Parse and validate the CUDA-Agent submission.
2. Materialize `model_new.py`, CUDA sources, and binding scaffold under fast local storage.
3. Collect `.cu`, `.cpp`, `.cc`, and `.cxx` sources.
4. Generate `build.ninja` with PyTorch's ninja helper.
5. Prepare object-cache hits and rewrite `build.ninja`.
6. Run ninja.
7. Store newly built reusable objects in the object cache.
8. Import the extension module.
9. Return a compile artifact with `work_dir`, `so_path`, `module_name`, `build_backend="manual_ninja"`, compile timing, object-cache stats, and profiling hints.

The complete compile artifact is internal by default. Public API responses should expose only sanitized metadata unless debug output is explicitly requested.

## Object Cache

The object cache operates on ninja object edges:

```text
build <object>: compile <source>
build <object>: cuda_compile <source>
```

Reusable objects are source objects that are not tied to a specific Python extension module name. Objects are not reusable when the source references `PYBIND11_MODULE`, `TORCH_EXTENSION_NAME`, or other module-name-specific symbols.

Each object cache key must include:

- normalized `build.ninja` text with the extension name replaced by a stable placeholder;
- object output name;
- ninja rule name;
- source bytes;
- relevant header digest;
- Python version;
- PyTorch version;
- CUDA version;
- CUDA architecture fingerprint;
- C++ and CUDA compiler flags.

For each reusable object:

1. Compute the cache key.
2. Check local cache storage.
3. If Redis index is enabled, check Redis metadata and verify the local file exists.
4. For hits, remove that object's build edge from `build.ninja`.
5. Replace the link input with the cached object path.
6. After ninja succeeds, copy missed reusable objects into the cache under a lock.

Default object cache root:

```text
/dev/shm/kernelgym/compile_cache/manual_ninja_objects
```

Redis index keys should include node identity:

```text
{REDIS_KEY_PREFIX}:manual_ninja_object_cache:{NODE_ID}:{cache_key}
```

Redis values store metadata only: local path, node id, status, timestamps, size, and hit count.

## Compile Artifact Cache

Compile artifact cache is separate from object cache. It is useful for exact repeated payloads and for compile/execute handoff, but it is not the main RL workload speedup.

A cached compile artifact is reusable only if:

- `work_dir` exists;
- `so_path` exists;
- `module_name` is present;
- `code` is present;
- `profiling_hints` are present;
- the artifact belongs to the same node when paths are local.

The artifact cache key should include parsed model code, CUDA source filenames and bytes, entry point, backend identity, compile flags, CUDA architecture, Python version, PyTorch version, and CUDA version.

## Split Compile/Execute

The request schema and internal task schema should support:

- `split_compile_and_execute: bool`
- `pure_compile_task: bool`
- `enable_compile_artifact_cache: bool`
- `task_stage: Optional[str]`
- `required_resource: Optional[str]`
- `assigned_worker: Optional[str]`
- `compile_artifact: Optional[Dict[str, Any]]`

When split mode is enabled:

1. Workflow submits a compile task with `task_stage="compile"` and `required_resource="cpu"`.
2. The CPU worker compiles and returns an internal compile artifact.
3. Workflow submits an execute task with `task_stage="execute"`, `required_resource="gpu"`, and the compile artifact.
4. The GPU worker loads the artifact and runs correctness, profiling, timing, and reference timing.

## Resource Queues

Use resource-aware queues:

```text
queue:resource:cpu
queue:resource:gpu
queue:worker:{worker_id}
```

Resource resolution:

- explicit `required_resource` wins;
- `task_stage="compile"` and `pure_compile_task=true` imply CPU;
- `task_stage="execute"`, reference timing, and normal kernel execution imply GPU;
- assigned-worker queues are used for direct handoff or recovery.

## Metadata

Results should include enough metadata to diagnose acceleration behavior:

- `build_backend="manual_ninja"`
- `compile_mode="filesystem"`
- `task_stage`
- `required_resource`
- `split_compile_and_execute`
- `compile_artifact_cache_enabled`
- `compile_artifact_cache_hit`
- `kg_kernel_backend_compile_s`
- `kg_kernel_backend_load_s`
- `compile_timing.total_wall_sec`
- `compile_timing.manual_ninja_build_wall_sec`
- `compile_timing.manual_ninja_import_wall_sec`
- `compile_timing.manual_ninja_object_cache.hits`
- `compile_timing.manual_ninja_object_cache.misses`
- `compile_timing.manual_ninja_object_cache.skipped`

Detailed `.ninja_log` parsing is optional diagnostic mode and should stay disabled by default.

## Correctness and Validation

- Object cache hits must link to behavior equivalent to a clean ninja build.
- Cache keys must change when source, headers, flags, architecture, Python, PyTorch, or CUDA version changes.
- Module-name-bound objects must never be reused across extension names.
- Execute tasks must not silently recompile unless explicit recovery behavior is added.
- Compile-only tasks must not instantiate or run user kernels.
- Existing CUDA-Agent parser, validation, binding, profiling, and decoy semantics must not regress.
- Unit test object edge parsing, cache key invalidation, non-reusable object detection, and `build.ninja` rewrite.
- Unit test schema/task propagation for split compile/execute fields.
- GPU smoke test manual ninja compile/load/run.
- Integration test compile-only artifact production and execute-with-artifact consumption.
- Benchmark clean manual ninja, cold object cache, warm object cache, and split compile/execute throughput.
- Save benchmark JSONL evidence under `logs/compile_acceleration/`.
