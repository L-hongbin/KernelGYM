# KernelGYM Reward Only

This repository is a standalone reward-service extraction of KernelGYM. It keeps the API server, task
manager, workers, workflow layer, KernelBench toolkits, and CUDA/Triton/TVM-FFI reward backends, while
intentionally excluding `drkernel`, training launchers, rollout code, model-serving launchers, and
offline-eval runbooks.

## Source Lineage

This repo is derived from two local source repositories:

- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent`
  - Primary code source for the current reward implementation.
  - Provides the copied `kernelgym/` package, current CUDA-Agent parser, TVM-FFI backend, timing metadata,
    static checker, and reward API behavior.
- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-lhb`
  - Comparative source for CUDA reward-env optimization design.
  - Provides the documented reference design for `manual_ninja`, object cache, split compile/execute, CPU
    compile workers, and resource-aware scheduling. Those mechanisms are documented here but are not the
    default implementation in this extraction.

See [docs/SOURCE_LINEAGE.md](docs/SOURCE_LINEAGE.md) and
[docs/IMPLEMENTATION_DIFFERENCES.md](docs/IMPLEMENTATION_DIFFERENCES.md).

## What Is Included

- `kernelgym/`: reward API, scheduler, worker, workflow, schema, backends, validation, and KernelBench toolkit.
- `scripts/auto_configure.sh`: local `.env` generation for reward service ports and GPU devices.
- `start_all_with_monitor.sh`, `start_worker_node.sh`, `start_worker_multinode.sh`, `stop_all.sh`: local and
  worker-node service entrypoints. These are thin wrappers; the real logic lives in
  `kernelgym.cli.service`.
- `tests/`: unit tests that verify extraction boundaries, source-lineage docs, pre-commit policy, schema
  behavior, CUDA-Agent parsing, validation behavior, and a GPU-gated CUDA-Agent compile/load/run smoke test.
- Ruff-only formatting and linting via `.pre-commit-config.yaml`.

## What Is Excluded

- No `drkernel/` package or training scripts.
- No model training, rollout, merge, checkpoint, or offline-eval orchestration.
- No Docker reward-cluster launcher from `drkernel/kernel/scripts/rl/start_reward.sh`.
- No LHB split compile/execute or `manual_ninja` object-cache code in the active implementation.

## Quick Start

```bash
python -m pip install -r requirements.txt
pre-commit install
pytest
ruff format .
ruff check .
```

Local reward service:

```bash
bash scripts/auto_configure.sh --force
bash start_all_with_monitor.sh
```

Equivalent Python CLI:

```bash
python -m kernelgym.cli.service auto-configure --force
python -m kernelgym.cli.service start-local
```

Stop local service:

```bash
bash stop_all.sh
```

## Development Policy

Formatting is done by `ruff format`, not Black. Linting is done by `ruff check`. The pre-commit hook includes
`ruff-check --fix` and `ruff-format`, plus basic file hygiene hooks.
