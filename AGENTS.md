# Repository Guidelines

## Project Structure & Module Organization

KernelGYM is a Python service for distributed GPU kernel evaluation. Core runtime code lives in `kernelgym/`: `backend/` handles kernel compilation and execution, `toolkit/` contains validation helpers, `workflow/` defines evaluation flows, `server/` manages the API and scheduling, and `worker/` runs CPU/GPU workers. Dr.Kernel training and reward code lives in `drkernel/`, with tests and smoke scripts in `drkernel/test_script/`. Shared images and diagrams are in `assets/`; generated logs, Redis dumps, TensorBoard files, and dated outputs should stay out of normal code changes.

## Build, Test, and Development Commands

- `bash setup.sh`: install Python dependencies, Redis utilities, and local setup requirements.
- `bash scripts/auto_configure.sh`: generate a machine-specific `.env.<hostname>` with ports, IP, and GPU devices.
- `./start_all_with_monitor.sh`: start Redis if needed, the FastAPI server, monitor, and local workers.
- `./start_all_with_monitor_debug.sh`: start a debug deployment writing logs under `logs/debug/`.
- `./stop_all.sh`: stop locally launched KernelGYM processes.
- `curl http://localhost:10907/health`: verify the service after startup.
- `pytest drkernel/test_script/test_kernelgym_cpu_worker.py`: run a targeted test; use `pytest drkernel/test_script` for the local Dr.Kernel suite.

## Coding Style & Naming Conventions

Use Python 3.10+ and follow the existing package style: four-space indentation, type hints on public interfaces, dataclasses where they simplify structured state, and async APIs for server/workflow paths. Keep module and file names lowercase with underscores, test files named `test_*.py`, and task or workflow identifiers explicit, for example `kernel_simple`. Dependencies include `black`, `isort`, and `flake8`; run them on touched Python files before submitting substantial changes.

## Testing Guidelines

Tests use `pytest` and `pytest-asyncio`. Prefer focused tests near the affected component, especially for scheduler, worker, reward, and workflow behavior. GPU-dependent tests may require CUDA, Redis, and configured `GPU_DEVICES`; document any skipped hardware coverage in the PR. Avoid committing generated artifacts from `logs/`, `drkernel/outputs/`, or TensorBoard directories.

## Commit & Pull Request Guidelines

Recent history mixes imperative summaries with scoped tags, such as `Fix import warnings`, `ci: adding errors to Github summary`, and `[model-gateway] Add streaming metrics...`. Use a concise imperative subject, add a scope when useful, and reference issues or experiments in the body. PRs should describe the behavior changed, commands run, environment assumptions such as CUDA/Redis availability, and include screenshots only for documentation or asset changes.

## Security & Configuration Tips

Do not commit `.env.<hostname>` files, credentials, Redis dumps, or machine-specific logs. Keep ports, hostnames, and GPU lists configurable through env files or startup script flags rather than hard-coding local cluster details.
