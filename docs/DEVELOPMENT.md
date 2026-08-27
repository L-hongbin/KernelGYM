# Development

## Environment

```bash
bash ensure_venv.sh --recreate
source /root/kernelgym-reward-only/.venv/bin/activate
pre-commit install
```

## Test

```bash
pytest
```

The default test suite includes a CUDA-Agent GPU smoke test. It is marked with `@pytest.mark.gpu` and skips
when torch, CUDA, nvcc, a C++ compiler, or an executable `/dev/shm` workspace is unavailable. Redis or API
worker integration checks should be marked with `@pytest.mark.integration`.

## Format And Lint

```bash
ruff format .
ruff check .
```

Pre-commit runs:

- file hygiene hooks from `pre-commit-hooks`;
- `ruff-check --fix`;
- `ruff-format`.

Black is intentionally not configured.

## Service CLI

Operational service logic belongs in Python, not long shell scripts. Use the service CLI directly:

```bash
python -m kernelgym.cli.service --help
```

Add new startup, shutdown, or worker-node behavior in `kernelgym/cli/service.py`. Do not add bash wrappers that
only forward arguments without adding a real operator-facing behavior.

CUDA 12.9 uv environment and Redis bootstrap belongs in `ensure_venv.sh` because it is environment assembly, not service orchestration. `runtime_paths.sh` resolves the shared wheelhouse once as `WHELL_PATH`; `KERNELGYM_OFFLINE_WHEEL_DIR` defaults to that root and `KERNELGYM_OFFLINE_REDIS_DIR` defaults to its Redis bundle. If the bundle is unavailable, the bootstrap reuses an existing system Redis or installs `redis-server` and `redis-tools` online from the configured apt repositories. Python dependencies remain strictly offline: the script bootstraps missing `uv` from the selected wheelhouse, creates and activates `/root/kernelgym-reward-only/.venv` on local disk with Python 3.12, and checks `/usr/local/cuda-12.9/bin/nvcc`. Deployment profiles are Python classes in
`kernelgym/deployment_profiles.py`; do not add a CLI that generates env files. Direct host operations belong in bash:
`scripts/lock_gpu_clocks.sh` and `scripts/start_container.sh`.
