# Deployment

KernelGYM reward-only supports two deployment modes. Runtime env values come from Python profiles in
`kernelgym/deployment_profiles.py`; there is no env-file generation CLI. GPU clock locking, container startup,
and CUDA 12.9 virtualenv bootstrap are bash scripts because they are shell-native operations around
`nvidia-smi`, Docker, Python, uv, pip, and proxy environment variables.

## Shared Runtime Policy

- Profile detection from `/ms` is available as an operator check only:
  - `/ms` exists and is a real path: `internal`;
  - `/ms` missing: `external`;
  - `/ms` is a symlink: `external`.
- External profiles are Python classes: `reward-39` and `reward-40`.
- Service ports are fixed: API `20111`, Redis `20110`, metrics `20112`.
- API workers/reload and Redis db/password/key-prefix are fixed.
- Use a repo-local uv virtual environment: `.venv`.
- If `uv` is missing, `create_venv.sh` installs it with `pip install uv`.
- Use CUDA 12.9 explicitly:
  - PyTorch wheels are installed from `https://download.pytorch.org/whl/cu129`.
  - `/usr/local/cuda-12.9/bin/nvcc --version` must report CUDA 12.9.
- If CUDA wheel dependencies cannot be fetched directly, `create_venv.sh` retries with
  `http://192.168.28.186:7897` on external nodes. Override with `KERNELGYM_PROXY` or
  `KERNELGYM_FALLBACK_PROXY` only when needed.
- Do not reuse older KernelGYM or drkernel virtual environments.

Create the environment in the runtime where reward will execute:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash create_venv.sh --recreate
source .venv/bin/activate
```

The script validates both `torch.version.cuda == "12.9"` and `nvcc` from CUDA 12.9. Common overrides are
not needed: it creates and activates `.venv` with Python 3.12, then checks `/usr/local/cuda-12.9/bin/nvcc`
directly.

Detect the current deployment profile:

```bash
bash scripts/detect_profile.sh
```

Select the profile explicitly, or use `auto` on the matching host:

```bash
python -m kernelgym.cli.service start-local --profile reward-40
```

## Mode 1: Physical Host SSH, Then Docker

Use this mode for external reward nodes such as `192.168.16.39` and `192.168.16.40`, where SSH lands on the
physical host. Host-level duties happen before starting the container:

1. Stop old reward services if needed.
2. Lock GPU clocks on the host.
3. Start or replace the Docker container.
4. Enter the container and create `.venv` there with CUDA 12.9.
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
bash create_venv.sh --recreate
source .venv/bin/activate
python -m kernelgym.cli.service start-local --profile reward-39
```

Worker-only multi-node deployment should use an explicit Python profile for that topology. Do not generate it
from a CLI; add the profile class in `kernelgym/deployment_profiles.py` when that topology is needed.

## Mode 2: SSH Already Lands Inside A Container

Use this mode for internal nodes where SSH goes directly into the runtime container. Do not start Docker from
inside this container. Create `.venv` and start services directly:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash create_venv.sh --recreate
source .venv/bin/activate
python -m kernelgym.cli.service start-local --profile reward-40
```

Worker-only containers follow the same rule: use a checked-in Python profile for the concrete topology.

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
