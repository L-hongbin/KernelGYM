# Deployment

KernelGYM reward-only supports two deployment modes. Keep operational logic in Python; shell files in this
repo should stay thin wrappers.

## Shared Runtime Policy

- Deployment profile is detected from `/ms` only:
  - `/ms` exists and is a real path: `internal`;
  - `/ms` missing: `external`;
  - `/ms` is a symlink: `external`.
- The `/ms` check is unrelated to `CUDA_HOME`.
- Generate env files through `kernelgym.cli.deploy write-env`; it auto-selects internal/external defaults.
- Use a repo-local uv virtual environment: `.venv`.
- Use CUDA 12.9 explicitly:
  - PyTorch wheels are installed from `https://download.pytorch.org/whl/cu129`.
  - `CUDA_HOME` should point at `/usr/local/cuda-12.9`.
  - `nvcc --version` must report CUDA 12.9 before running CUDA-Agent compile tests.
- If CUDA wheel dependencies cannot be fetched directly, `create-venv` retries with
  `http://192.168.28.186:7897`. Override this with `--proxy`, `--fallback-proxy`, `KERNELGYM_PROXY`, or
  `KERNELGYM_FALLBACK_PROXY`.
- Do not reuse older KernelGYM or drkernel virtual environments.

Create the environment in the runtime where reward will execute:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
python -m kernelgym.cli.deploy create-venv --recreate --cuda-home /usr/local/cuda-12.9
source .venv/bin/activate
```

The command validates both `torch.version.cuda == "12.9"` and `nvcc` from CUDA 12.9 unless
`--skip-validate` is passed.

Detect the current deployment profile:

```bash
python -m kernelgym.cli.deploy detect-profile
```

Write an API-node env using the detected profile:

```bash
python -m kernelgym.cli.deploy write-env --role api --env-file .env --force
```

Write a worker-node env using the detected profile and a known API/Redis host:

```bash
python -m kernelgym.cli.deploy write-env --role worker --server-host 192.168.16.39 --env-file .env --force
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
python -m kernelgym.cli.deploy host-container \
  --name kernelgym-reward-39 \
  --replace \
  --sudo \
  --lock-gpu-clocks \
  --gpu-clock 2700 \
  --power-limit 400 \
  --cuda-home /usr/local/cuda-12.9 \
  --image 192.168.14.129:80/fm/llmc:v1.1
```

The generated container command uses:

- `--gpus all`;
- `--network host`;
- executable `/dev/shm` through `--tmpfs /dev/shm:rw,nosuid,nodev,exec,size=256g`;
- `--privileged`;
- `-v /nfs:/nfs`;
- a read-only mount of `/usr/local/cuda-12.9`;
- `CUDA_HOME=/usr/local/cuda-12.9`.

If the image already has CUDA 12.9, the explicit CUDA mount is harmless. If the image has an older toolkit,
the mount plus `CUDA_HOME` ensures CUDA-Agent compilation uses host CUDA 12.9.

Inside the container:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
python -m kernelgym.cli.deploy create-venv --recreate --cuda-home /usr/local/cuda-12.9
source .venv/bin/activate
python -m kernelgym.cli.deploy write-env --role api --env-file .env --force
python -m kernelgym.cli.service start-local
```

For a second physical host that only runs workers, copy or mount the server env from the API node and run:

```bash
source .venv/bin/activate
python -m kernelgym.cli.deploy write-env --role worker --server-host 192.168.16.39 --env-file .env --force
python -m kernelgym.cli.service start-worker-node /path/to/server.env
```

## Mode 2: SSH Already Lands Inside A Container

Use this mode for internal nodes where SSH goes directly into the runtime container. Do not start Docker from
inside this container. Create `.venv` and start services directly:

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
python -m kernelgym.cli.deploy create-venv --recreate --cuda-home /usr/local/cuda-12.9
source .venv/bin/activate
python -m kernelgym.cli.deploy write-env --role api --env-file .env --force
python -m kernelgym.cli.service start-local
```

Worker-only containers use the same `.venv` creation step, then:

```bash
source .venv/bin/activate
python -m kernelgym.cli.deploy write-env --role worker --server-host <api-host> --env-file .env --force
python -m kernelgym.cli.service start-worker-node /path/to/server.env
```

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
