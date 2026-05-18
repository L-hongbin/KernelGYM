# Development

## Environment

```bash
python -m kernelgym.cli.deploy create-venv --recreate --cuda-home /usr/local/cuda-12.9
source .venv/bin/activate
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

Operational logic belongs in Python, not long shell scripts. The root shell entrypoints are compatibility
wrappers around:

```bash
python -m kernelgym.cli.service --help
```

Add new startup, shutdown, or worker-node behavior in `kernelgym/cli/service.py`, then keep the shell wrapper
small enough to only resolve the repository root and delegate to Python.

Deployment and environment-preparation logic belongs in `kernelgym/cli/deploy.py`. This includes CUDA 12.9
uv environment creation, physical-host GPU clock locking, and Docker container preparation for hosts where SSH
does not land inside the container. Internal/external profile detection must only inspect `/ms`: a real path is
internal, while missing `/ms` or a symlink is external.
