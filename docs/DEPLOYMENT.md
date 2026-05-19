# Deployment

KernelGYM reward-only supports two deployment modes. Runtime env values come from the `v1` Python profile in
`kernelgym/deployment_profiles.py`. GPU clock locking, container startup, and CUDA 12.9 virtualenv bootstrap are bash scripts because they are shell-native operations around
`nvidia-smi`, Docker, Python, uv, pip, and proxy environment variables.

## Shared Runtime Policy

- The default reward runtime profile is `v1`; `auto` is an alias for it.
- Service ports are fixed: API `20111`, Redis `20110`, metrics `20112`.
- API workers/reload and Redis db/password/key-prefix are fixed.
- Use a repo-local uv virtual environment: `.venv`.
- If `redis-server` is missing, `ensure_venv.sh` installs it with apt.
- If `uv` is missing, `ensure_venv.sh` installs it with `pip install uv`.
- Use CUDA 12.9 explicitly:
  - `requirements-cuda129.txt` only pins package versions; pip/uv index or mirror selection must come from the
    container image, pip config, uv config, or environment, not from the requirements file.
  - `/usr/local/cuda-12.9/bin/nvcc --version` must report CUDA 12.9.
- If CUDA wheel dependencies cannot be fetched directly, `ensure_venv.sh` retries with
  `http://192.168.28.186:7897` on external nodes. Override with `KERNELGYM_PROXY` or
  `KERNELGYM_FALLBACK_PROXY` only when needed.
- Do not reuse older KernelGYM or drkernel virtual environments.

Create the environment in the runtime where reward will execute:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash ensure_venv.sh --recreate
source .venv/bin/activate
```

The script validates `redis-server`, `torch.version.cuda == "12.9"`, and `nvcc` from CUDA 12.9. Common overrides are
not needed: it creates and activates `.venv` with Python 3.12 when missing, then checks `/usr/local/cuda-12.9/bin/nvcc`
directly.

Use `--profile v1`:

```bash
python -m kernelgym.cli.service start-local --profile v1
```

The deployment convenience script is container-only. It runs `ensure_venv.sh`, sources `.venv/bin/activate`, and always stops existing KernelGym worker processes before starting worker-only nodes.

## Mode 1: Physical Host, Then Docker

Use this mode for external reward nodes such as `192.168.16.39` and `192.168.16.40`, where the operator starts
from the physical host. Host-level duties happen before starting the container:

1. Stop old reward services if needed.
2. Lock GPU clocks on the host.
3. Start or replace the Docker container.
4. Enter the container and ensure `.venv` plus Redis there with CUDA 12.9.
5. Start the reward API/workers from inside the container.

Host preparation example:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash scripts/lock_gpu_clocks.sh --sudo --gpu-clock 2700 --power-limit 400
bash scripts/start_container.sh
```

The generated container command uses:

- `--gpus all`;
- `--network host`;
- executable `/dev/shm` through `--tmpfs /dev/shm:rw,nosuid,nodev,exec,size=256g`;
- `--privileged`;
- `-v /nfs:/nfs`;
- a read-only mount of `/usr/local/cuda-12.9`.

The default container image is `192.168.14.129:80/library/slime:nightly-dev-20260430b`.
If the image already has CUDA 12.9, the explicit CUDA mount is harmless. The environment bootstrap still
validates `/usr/local/cuda-12.9/bin/nvcc` inside the container before installing the CUDA 12.9 wheel set.

Inside the container:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash ensure_venv.sh --recreate
source .venv/bin/activate
python -m kernelgym.cli.service start-local --profile v1
```

The same startup can be run with:

```bash
bash scripts/deploy_node.sh --nnodes 1
```

Worker-only multi-node deployment uses `scripts/deploy_node.sh` from inside each container after `.venv` exists.

## Mode 2: Already Inside A Container

Use this mode when the operator is already in the runtime container. Do not start Docker from inside this
container. Create `.venv` and start services directly:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash ensure_venv.sh --recreate
source .venv/bin/activate
python -m kernelgym.cli.service start-local --profile v1
```

After `.venv` exists, the single-node convenience entrypoint is:

```bash
bash scripts/deploy_node.sh --nnodes 1
```

Worker-only containers follow the same rule: use `scripts/deploy_node.sh --nnodes N` from inside each container.

## Convenience Scripts

Single node:

```bash
bash scripts/deploy_node.sh --nnodes 1
```

Multiple nodes:

```bash
bash scripts/deploy_node.sh --nnodes 2 --node-rank 0 --master-addr 192.168.16.40
bash scripts/deploy_node.sh --nnodes 2 --node-rank 1 --master-addr 192.168.16.40
```

The script is intended to run from inside containers. For multi-node deployment, run it manually on every node with that node's `--node-rank`. The node matching `--master-addr` must use rank `0` and becomes primary; other ranks become worker-only.

## Verification

Run lint and tests from the CUDA 12.9 `.venv`:

```bash
source .venv/bin/activate
ruff format .
ruff check .
pytest
```

On a GPU runtime with CUDA 12.9, `tests/test_cuda_agent_gpu.py` compiles, loads, and runs a minimal
CUDA-Agent extension. Without GPU, torch, nvcc, or executable `/dev/shm`, that test skips.
