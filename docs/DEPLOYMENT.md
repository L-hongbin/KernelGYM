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
- `set_env.sh` repairs the Python interpreter path used by the shared repo-local `.venv`: newer images provide
  `/usr/bin/python3.12`, while older `.venv/bin/python` links through `/usr/local/bin/python3`. The deploy wrapper
  runs `set_env.sh` before activating `.venv` so the existing environment remains usable after replacing containers.
- Do not reuse older KernelGYM or drkernel virtual environments.

Create the environment in the runtime where reward will execute (run from the repo root):

```bash
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

Override CPU compile-worker count at startup when the profile default is not appropriate. `--cpu-workers` is accepted as a short alias for `--cpu-compile-workers`.

```bash
python -m kernelgym.cli.service start-local --profile v1 --cpu-compile-workers 8
bash deploy_node.sh --nnodes 1 --cpu-compile-workers 8
```

To stop the running service (kills the API, monitor, GPU/CPU workers, and clears Redis state with the `kernelgym:` prefix):

```bash
python -m kernelgym.cli.service stop --profile v1
# or with the installed entrypoint:
kernelgym-service stop --profile v1
# or the convenience wrapper (mirrors deploy_node.sh):
bash stop_node.sh
```

A typical restart cycle inside the container is `bash stop_node.sh && bash deploy_node.sh --nnodes 1`.

The deployment convenience script is container-only. It runs `ensure_venv.sh`, sources `.venv/bin/activate`, and always stops existing KernelGym worker processes before starting worker-only nodes.

## Mode 1: Physical Host, Then Docker

Use this mode for external reward nodes such as `192.168.16.39` and `192.168.16.40`, where the operator starts
from the physical host. Host-level duties happen before starting the container:

1. Stop old reward services if needed.
2. Lock GPU clocks on the host.
3. Start or replace the Docker container.
4. Enter the container and ensure `.venv` plus Redis there with CUDA 12.9.
5. Start the reward API/workers from inside the container.

Host preparation example (run from the repo root):

```bash
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

The default container image is `192.168.14.129:80/library/slime:nightly-dev-20260526a`.
If the image already has CUDA 12.9, the explicit CUDA mount is harmless. The environment bootstrap still
validates `/usr/local/cuda-12.9/bin/nvcc` inside the container before installing the CUDA 12.9 wheel set.

Inside the container (run from the repo root):

```bash
bash set_env.sh
bash ensure_venv.sh --recreate
source .venv/bin/activate
python -m kernelgym.cli.service start-local --profile v1
```

The same startup can be run with:

```bash
bash deploy_node.sh --nnodes 1
```

Worker-only multi-node deployment uses `deploy_node.sh` from inside each container after `.venv` exists.

## Mode 2: Already Inside A Container

Use this mode when the operator is already in the runtime container. Do not start Docker from inside this
container. From the repo root, create `.venv` and start services directly:

```bash
bash set_env.sh
bash ensure_venv.sh --recreate
source .venv/bin/activate
python -m kernelgym.cli.service start-local --profile v1
```

After `.venv` exists, the single-node convenience entrypoint is:

```bash
bash deploy_node.sh --nnodes 1
```

Worker-only containers follow the same rule: use `scripts/deploy_node.sh --nnodes N` from inside each container.

## Convenience Scripts

Single node:

```bash
bash deploy_node.sh --nnodes 1
```

Multiple nodes, short form:

```bash
bash deploy_node.sh --nnodes 2 --node-rank 0 --master-addr 192.168.16.40
bash deploy_node.sh --nnodes 2 --node-rank 1 --master-addr 192.168.16.40
```

The script is intended to run from inside containers. For multi-node deployment, run it manually on every node with that node's `--node-rank`. The node matching `--master-addr` must use rank `0` and becomes primary; other ranks become worker-only. `--cpu-compile-workers N` / `--cpu-workers N` is forwarded to both primary and worker-only node startup.

## Multi-Node Tutorial

This tutorial assumes two physical reward nodes, `192.168.16.40` as the primary and `192.168.16.39` as the worker-only node. Replace the addresses and ranks for larger clusters. The primary hosts Redis and the HTTP API on fixed ports; every worker-only node connects back to the primary and registers its local GPU/CPU workers.

### 1. Prepare each physical host

Run host-level setup on every machine before entering the container. This locks GPU clocks and starts or replaces the runtime container with host networking and `/nfs` mounted.

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
bash scripts/lock_gpu_clocks.sh --sudo --gpu-clock 2700 --power-limit 400
bash scripts/start_container.sh
```

Enter the printed container on each host:

```bash
docker exec -it kernelgym-reward bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only
```

### 2. Prepare the repo-local environment in each container

Run this on the primary container and every worker-only container:

```bash
bash set_env.sh
bash ensure_venv.sh --recreate
```

`deploy_node.sh` activates `.venv` itself, so you do not need to keep the shell activated after this step.

### 3. Start the primary node first

On the node whose IP is `--master-addr`, use rank `0`:

```bash
bash deploy_node.sh --nnodes 2 --node-rank 0 --master-addr 192.168.16.40
```

For multi-node primary startup, `deploy_node.sh` automatically enables Redis remote access on port `20110` by forwarding `--redis-remote-access` to the service CLI. This disables Redis protected mode inside the container so worker-only nodes can connect to the primary Redis instance. Single-node startup does not enable this flag.

Wait until the primary reports API readiness:

```text
API ready: http://127.0.0.1:20111/health
```

### 4. Start each worker-only node

On every non-primary node, use a unique nonzero rank and the same `--master-addr`:

```bash
bash deploy_node.sh --nnodes 2 --node-rank 1 --master-addr 192.168.16.40
```

For three or more nodes, increment ranks:

```bash
bash deploy_node.sh --nnodes 3 --node-rank 0 --master-addr 192.168.16.40   # primary
bash deploy_node.sh --nnodes 3 --node-rank 1 --master-addr 192.168.16.40   # worker A
bash deploy_node.sh --nnodes 3 --node-rank 2 --master-addr 192.168.16.40   # worker B
```

If a worker-only node should use fewer CPU compile workers than the profile default, pass the same override on that node:

```bash
bash deploy_node.sh --nnodes 2 --node-rank 1 --master-addr 192.168.16.40 --cpu-workers 8
```

### 5. Verify from the primary

Run the checks inside the primary container. `check_node.sh -v` should show local and remote GPU workers with fresh heartbeats, plus CPU workers from each node.

```bash
bash check_node.sh -v
bash test_reward.sh
```

Expected high-level signals:

```text
api_status:          healthy
gpu_workers_fresh:   <all>/<all>
stale_gpu_workers:   0
task_status: completed
compiled: True
correctness: True
```

For direct API probing from a worker-only node or an RL client:

```bash
curl http://192.168.16.40:20111/health
curl http://192.168.16.40:20111/workers/status
```

### 6. Restart or stop

Restart the whole cluster by stopping worker-only nodes first, then the primary, then starting the primary before workers again:

```bash
bash stop_node.sh
```

For worker-only nodes, `deploy_node.sh` already runs the service stop path before starting workers, so re-running the worker command is safe after the primary is healthy.

### Common mistakes

- Do not run `--node-rank 0` on a node that is not `--master-addr`; the script rejects this because rank `0` owns API/Redis.
- Do not start worker-only nodes before the primary API is reachable; worker startup waits for `/health` and fails after the timeout.
- Do not use different `--master-addr` values across nodes in the same cluster.
- If worker startup reaches `/health` but fails with `Cannot connect to Redis` or a Redis protected-mode error, restart the primary with the multi-node command above so Redis accepts remote worker connections.
- Do not run the deployment script on the physical host shell; use it inside the runtime container after `/nfs` and CUDA are mounted.
- If `check_node.sh` briefly shows one stale worker after a long smoke test, wait one heartbeat interval and rerun it before treating the node as unhealthy.

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
