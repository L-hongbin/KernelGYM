# KernelGYM Reward Only

## Quick Start — Single-Node Deployment

The shortest path from a fresh host to a running reward service on `127.0.0.1:20111`.

### 0. (Optional) Start the runtime container

Skip this step if you are already inside the runtime container. Use it only when starting from a physical host (e.g. `192.168.16.39` / `192.168.16.40`): it locks GPU clocks and launches the Docker container that everything else runs inside. All commands assume the repo root as cwd.

```bash
bash scripts/lock_gpu_clocks.sh --sudo --gpu-clock 2700 --power-limit 400
bash scripts/start_container.sh
# then docker exec / docker attach into the container printed by start_container.sh
```

### 1. Bootstrap the environment

`set_env.sh` idempotently repairs the interpreter link used by an existing shared `.venv`. `ensure_venv.sh` then validates CUDA 12.9, installs `redis-server` (via apt when missing), creates the repo-local `.venv` with Python 3.12 when absent, and installs torch / torchvision / apache-tvm-ffi (preferring local `./wheels/*.whl` over the configured index).

```bash
bash set_env.sh
bash ensure_venv.sh
```

### 2. Launch the reward service

```bash
bash deploy_node.sh --nnodes 1
```

`deploy_node.sh` activates `.venv`, scrubs `LD_LIBRARY_PATH` / `PYTHONPATH` of host-Python torch trees, runs `scripts/validate_runtime.py`, then starts the API server (`:20111`), worker monitor, one GPU worker per container-visible CUDA device, and the profile's CPU compile workers. Once this node's workers are ready, deployment performs one force-refreshed correctness and CUDA-profiling warmup affined to this hostname; cold `/nfs` compilation can keep the startup command busy for several minutes. Override the CPU count with `--cpu-compile-workers N` / `--cpu-workers N`, select a logical GPU subset with `--gpu-devices 0,1`, or use `--no-startup-warmup` only for diagnostics.

Add `--block-terminal` when the deploy command should stay in the foreground (for example, as the container's foreground command). After startup succeeds, Ctrl-C, SIGTERM, or a terminal hangup stops this node's KernelGym services before the command exits:

```bash
bash deploy_node.sh --cluster --block-terminal
```

For a cold restart that removes local Redis persistence plus KernelGym compile/work caches before launching:

```bash
bash deploy_node.sh --clear-cache --nnodes 1
```

### 3. Verify

```bash
bash check_node.sh                  # GPU + worker health summary (ASCII tables with -v)
bash test_reward.sh                 # round-trip a hand-written CUDA add kernel
```

For a broader real-GPU regression across all supported KernelBench languages, run the shared language matrix. It
submits CUDA (through the TVM-FFI backend), Triton, and TileLang fixtures through the HTTP API, uses two concurrent requests by default,
and requires correctness, profiler kernel capture, and decoy detection to pass:

```bash
python benchmarks/run_language_matrix.py \
    --api http://127.0.0.1:20111 \
    --languages cuda,triton,tilelang \
    --concurrency 2

# DSL-only fast regression
python benchmarks/run_language_matrix.py \
    --languages triton,tilelang \
    --concurrency 2 \
    --out /tmp/kernelgym_language_matrix.json

# Exercise CPU compile -> GPU execute artifact handoff
python benchmarks/run_language_matrix.py \
    --languages triton,tilelang \
    --split
```

The versioned fixtures live under `benchmarks/kernels/{cuda,triton,tilelang}/`. The CUDA fixture uses the
`CUDA_KERNELS + APPLY_BINDINGS + MODEL_NEW` submission format with `backend=tvm_ffi`; it does not use the legacy
CUDA-Agent adapter. Triton and TileLang cover
inference-style operations such as RMSNorm, RoPE, KV gather, attention-state merge, matmul, mixed dtypes, odd
shapes, and non-contiguous strides. Add `--include-negative` to also verify that torch-only candidates are rejected.

### 4. Stop

```bash
bash stop_node.sh
```

Stops the API server, worker monitor, GPU/CPU workers, and shuts down local Redis without saving.

---

## About this Repository

This is a standalone reward-service extraction of KernelGYM. It keeps the API server, task manager, workers, workflow layer, KernelBench toolkits, and CUDA / Triton / TileLang / TVM-FFI reward backends, while intentionally excluding `drkernel`, training launchers, rollout code, model-serving launchers, and offline-eval runbooks.

## Source Lineage

This repo is derived from two local source repositories:

- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent`
  - Primary code source for the current reward implementation.
  - Provides the copied `kernelgym/` package, CUDA-Agent parser, TVM-FFI backend, timing metadata, static checker, and reward API behavior.
- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb`
  - Comparative source for CUDA reward-env optimization logic.
  - Provides reference logic for ninja-driven fine-grained compilation, object cache, split compile/execute, CPU compile workers, and resource-aware scheduling. The reward-only implementation keeps the useful mechanics while preserving this repo's parser, validation, metadata, and workflow contracts.

See [docs/SOURCE_LINEAGE.md](docs/SOURCE_LINEAGE.md), [docs/IMPLEMENTATION_DIFFERENCES.md](docs/IMPLEMENTATION_DIFFERENCES.md), and [docs/design-doc/COMPILE_ACCELERATION.md](docs/design-doc/COMPILE_ACCELERATION.md). Full deployment details are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## What Is Included

- `kernelgym/`: reward API, scheduler, CPU/GPU workers, workflow, schema, backends, validation, and KernelBench toolkit.
- `ensure_venv.sh`: idempotent CUDA 12.9 uv environment and `redis-server` bootstrap.
- `deploy_node.sh` / `stop_node.sh`: thin wrappers around `kernelgym.cli.service` for single-node start/stop.
- `check_node.sh` / `test_reward.sh`: operator probes for `/health`, `/workers/status`, and end-to-end `/evaluate`.
- `scripts/lock_gpu_clocks.sh`: host-level GPU persistence, clock, and power-limit setup.
- `scripts/start_container.sh`: physical-host Docker container startup for external nodes.
- `scripts/ensure_redis.sh`, `scripts/scrub_venv_env.sh`, `scripts/validate_runtime.py`, `scripts/deploy_node.py`: bootstrap and runtime helpers used by the top-level wrappers.
- `kernelgym/deployment_profiles.py`: Python reward runtime profile definitions.
- `tests/`: unit tests that verify extraction boundaries, source-lineage docs, pre-commit policy, schema behavior, CUDA-Agent parsing, validation behavior, resource queue routing, and a GPU-gated CUDA-Agent compile/load/run smoke test.
- Ruff-only formatting and linting via `.pre-commit-config.yaml`.

## What Is Excluded

- No `drkernel/` package or training scripts.
- No model training, rollout, merge, checkpoint, or offline-eval orchestration.
- No Docker reward-cluster launcher from `drkernel/kernel/scripts/rl/start_reward.sh`.

## Multi-Node Deployment

Multi-node deployment uses the same `deploy_node.sh` from inside each container. The node matching `--master-addr` must use `--node-rank 0` and becomes the primary API/Redis node; other ranks become worker-only nodes.

```bash
bash deploy_node.sh --nnodes 2 --node-rank 0 --master-addr 192.168.16.40   # on master
bash deploy_node.sh --nnodes 2 --node-rank 1 --master-addr 192.168.16.40   # on worker
```

See [docs/DEPLOYMENT.md#multi-node-tutorial](docs/DEPLOYMENT.md#multi-node-tutorial) for the full runbook, including host/container preparation, startup order, verification, and common mistakes.

## Development Setup

Install the pre-commit hooks and run the test suite from inside the activated `.venv`:

```bash
source .venv/bin/activate
pre-commit install
pytest
ruff format .
ruff check .
```

Formatting is done by `ruff format`, not Black. Linting is done by `ruff check`. The pre-commit hook includes `ruff-check --fix` and `ruff-format`, plus basic file hygiene hooks.

## Documentation

### Operations

- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Full single-node and multi-node deployment details (extends the Quick Start above).
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Local development workflow, test layout, and contribution conventions.
- [docs/HTTP_API.md](docs/HTTP_API.md) — HTTP endpoints, request/response schemas, metadata fields, error model, and `curl` example.

### Design

- [docs/design-doc/COMPILE_ACCELERATION.md](docs/design-doc/COMPILE_ACCELERATION.md) — Manual-ninja compile path, per-object cache, compile artifact cache, split compile/execute.
- [docs/design-doc/TWO_WORKER_WARM_POOL.md](docs/design-doc/TWO_WORKER_WARM_POOL.md) — GPU subprocess pool architecture (`WORKER_POOL_SIZE=2`, `MAX_TASKS_PER_WORKER=1`) and recycle invariants.
- [docs/design-doc/REWARD_HACKING_DEFENSES.md](docs/design-doc/REWARD_HACKING_DEFENSES.md) — Decoy-kernel detection and reward-hacking defenses.

### Repository context

- [docs/SOURCE_LINEAGE.md](docs/SOURCE_LINEAGE.md) — Which files were copied from which upstream repo.
- [docs/IMPLEMENTATION_DIFFERENCES.md](docs/IMPLEMENTATION_DIFFERENCES.md) — Behavioural deltas vs the source repos.
- [docs/server-result-cache-guard.md](docs/server-result-cache-guard.md) — Rationale for the per-request-hash result-cache guard.

### Benchmarks

- [benchmarks/README.md](benchmarks/README.md) — Cross-language correctness/profiling matrix plus compile-speed benchmarks and kernel fixtures.
