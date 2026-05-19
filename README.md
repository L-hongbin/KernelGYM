# KernelGYM Reward Only

This repository is a standalone reward-service extraction of KernelGYM. It keeps the API server, task manager, workers, workflow layer, KernelBench toolkits, and CUDA/Triton/TVM-FFI reward backends, while intentionally excluding `drkernel`, training launchers, rollout code, model-serving launchers, and offline-eval runbooks.

## Source Lineage

This repo is derived from two local source repositories:

- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent`
  - Primary code source for the current reward implementation.
  - Provides the copied `kernelgym/` package, CUDA-Agent parser, TVM-FFI backend, timing metadata, static checker, and reward API behavior.
- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb`
  - Comparative source for CUDA reward-env optimization logic.
  - Provides reference logic for ninja-driven fine-grained compilation, object cache, split compile/execute, CPU compile workers, and resource-aware scheduling. The reward-only implementation keeps the useful mechanics while preserving this repo's parser, validation, metadata, and workflow contracts.

See [docs/SOURCE_LINEAGE.md](docs/SOURCE_LINEAGE.md), [docs/IMPLEMENTATION_DIFFERENCES.md](docs/IMPLEMENTATION_DIFFERENCES.md), and [docs/design-doc/COMPILE_ACCELERATION.md](docs/design-doc/COMPILE_ACCELERATION.md).
Deployment is documented in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## What Is Included

- `kernelgym/`: reward API, scheduler, CPU/GPU workers, workflow, schema, backends, validation, and KernelBench toolkit.
- `ensure_venv.sh`: idempotent CUDA 12.9 uv environment and Redis server bootstrap.
- `scripts/lock_gpu_clocks.sh`: host-level GPU persistence, clock, and power-limit setup.
- `scripts/start_container.sh`: physical-host Docker container startup for external nodes.
- `kernelgym/deployment_profiles.py`: Python reward runtime profile definitions.
- `tests/`: unit tests that verify extraction boundaries, source-lineage docs, pre-commit policy, schema behavior, CUDA-Agent parsing, validation behavior, resource queue routing, and a GPU-gated CUDA-Agent compile/load/run smoke test.
- Ruff-only formatting and linting via `.pre-commit-config.yaml`.

## What Is Excluded

- No `drkernel/` package or training scripts.
- No model training, rollout, merge, checkpoint, or offline-eval orchestration.
- No Docker reward-cluster launcher from `drkernel/kernel/scripts/rl/start_reward.sh`.

## Quick Start

```bash
bash ensure_venv.sh --recreate
source .venv/bin/activate
pre-commit install
pytest
ruff format .
ruff check .
```

`ensure_venv.sh` installs/checks `redis-server`, creates the repo-local uv `.venv` with Python 3.12 when missing, installs Python dependencies when needed, and validates CUDA 12.9 through `/usr/local/cuda-12.9/bin/nvcc`.

Local reward service:

```bash
python -m kernelgym.cli.service start-local --profile v1
```

Stop local service:

```bash
python -m kernelgym.cli.service stop
```

The default deployment profile is `v1` in `kernelgym/deployment_profiles.py`; `auto` is an alias for it. Physical-host deployment, such as external `192.168.16.39` / `192.168.16.40` reward nodes, uses `scripts/lock_gpu_clocks.sh` for host GPU clocks and `scripts/start_container.sh` to start the Docker container first.
Deployments that are already inside a container skip Docker and run `ensure_venv.sh` plus `kernelgym.cli.service` directly.

Physical-host setup before container-only deployment:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash scripts/lock_gpu_clocks.sh --sudo --gpu-clock 2700 --power-limit 400
bash scripts/start_container.sh
```

Then enter the container printed by `scripts/start_container.sh`, create `.venv` if needed, and run the container-only deployment script:

```bash
bash ensure_venv.sh --recreate
bash scripts/deploy_node.sh --nnodes 1
bash scripts/deploy_node.sh --nnodes 2 --node-rank 0 --master-addr 192.168.16.40
bash scripts/deploy_node.sh --nnodes 2 --node-rank 1 --master-addr 192.168.16.40
```

For multi-node deployment, run the command manually on every container node with that node's `--node-rank`. The node matching `--master-addr` must use rank `0` and becomes the primary API/Redis node; other ranks become worker-only nodes.

## Development Policy

Formatting is done by `ruff format`, not Black. Linting is done by `ruff check`. The pre-commit hook includes `ruff-check --fix` and `ruff-format`, plus basic file hygiene hooks.
