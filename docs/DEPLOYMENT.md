# Deployment

KernelGYM reward-only supports two deployment modes. Runtime env values come from the `v1` Python profile in
`kernelgym/deployment_profiles.py`. GPU clock locking, container startup, and CUDA 12.9 virtualenv bootstrap are bash scripts because they are shell-native operations around
`nvidia-smi`, Docker, Python, uv, pip, and proxy environment variables.

## Runtime Storage Policy

- The default reward runtime profile is `v1`; `auto` is an alias for it.
- Service ports are fixed: API `20111`, Redis `20110`, metrics `20112`.
- API workers/reload and Redis db/password/key-prefix are fixed.
- Use the node-local uv virtual environment `/root/kernelgym-reward-only/.venv`. The repo-local shared `.venv` is deprecated and ignored.
- Install the versions pinned in `requirements-offline.txt` directly from the absolute wheelhouse `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only/wheels`; `ensure_venv.sh` passes `--offline --no-index` and does not fall back to a package index.
- Keep source, `logs/`, `py_logs/`, core dumps, compile artifacts, and other long-lived evidence under the shared checkout.
- Redis `5:7.0.15-1ubuntu0.24.04.4` and its Redis-specific libraries are pinned under `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only/wheels/redis/ubuntu-24.04-amd64`; `ensure_venv.sh` installs those local `.deb` files with apt downloads disabled and never runs `apt update`.
- If `uv` is missing, `ensure_venv.sh` bootstraps it from the offline wheelhouse.
- Use CUDA 12.9 explicitly:
  - `requirements-cuda129.txt` pins the CUDA-sensitive package versions; all candidates must exist in the offline wheelhouse.
  - `/usr/local/cuda-12.9/bin/nvcc --version` must report CUDA 12.9.
- Nsight Compute collection is enabled by default at `/usr/local/cuda-12.9/bin/ncu`. The runtime validator checks the executable and version; deployed workers must have permission to access NVIDIA GPU performance counters. The default compact metric set includes L1 throughput (`l1tex__throughput.avg.pct_of_peak_sustained_active`), L1 sector hit rate (`l1tex__t_sector_hit_rate.pct`), L2 throughput (`lts__throughput.avg.pct_of_peak_sustained_elapsed`), and L2 sector hit rate (`lts__t_sector_hit_rate.pct`). It does not currently collect request/sector counts or read/write byte totals.
- Set `ENABLE_NCU=false` to disable collection globally, or use request field `enable_ncu=false` for one evaluation.
- Runtime Sanitizer is disabled by default. When enabled, it uses `/usr/local/cuda-12.9/bin/compute-sanitizer`; deployment validation then fails fast if the executable is missing.
- It runs only after a candidate correctness forward raises and regenerates that trial's input in an isolated process. `error_based` runs the check selected from the failure and falls back to all four checks when classification is ambiguous; `full` always runs all four checks.
- Set `ENABLE_COMPUTE_SANITIZER=true` globally or request field `enable_compute_sanitizer=true` per evaluation to enable its isolated trials. An explicit request value overrides the server default for that evaluation.
- Hidden correctness input perturbations are disabled by default. Set `ENABLE_CORRECTNESS_INPUT_PERTURBATIONS=true` globally or request field `enable_correctness_input_perturbations=true` per evaluation to enable distribution-aware correctness trials.
- `set_env.sh` validates and reports the node-local venv and absolute wheelhouse paths. It only reports the deprecated shared `.venv`; it never reads, repairs, deletes, or activates it.
- Do not reuse older KernelGYM or drkernel virtual environments.

Create the environment in the runtime where reward will execute (run from the repo root):

```bash
bash set_env.sh
bash ensure_venv.sh --recreate
source /root/kernelgym-reward-only/.venv/bin/activate
```

The script validates `redis-server`, `torch.version.cuda == "12.9"`, `nvcc`, and Nsight Compute from CUDA 12.9. It also validates Compute Sanitizer when `ENABLE_COMPUTE_SANITIZER=true`; with the default `false`, that check is reported as skipped. Common overrides are not needed: it creates and activates the node-local venv with Python 3.12 when missing, then checks the CUDA tools under `/usr/local/cuda-12.9/bin` directly. `KERNELGYM_LOCAL_VENV_DIR`, `KERNELGYM_OFFLINE_WHEEL_DIR`, and `KERNELGYM_OFFLINE_REDIS_DIR` may override the defaults; all must remain absolute paths and the venv must be on local storage. `bash scripts/ensure_redis.sh --verify-bundle` validates checksums, package metadata, platform compatibility, and offline apt dependency resolution without installing or starting anything.

Use `--profile v1`:

```bash
python -m kernelgym.cli.service start-local --profile v1
```

Override CPU compile-worker count at startup when the profile default is not appropriate. `--cpu-workers` is accepted as a short alias for `--cpu-compile-workers`.

```bash
python -m kernelgym.cli.service start-local --profile v1 --cpu-compile-workers 8
bash deploy_node.sh --cpu-compile-workers 8
```

To stop the running service (kills the API, monitor, GPU/CPU workers, and shuts down local Redis without saving):

```bash
python -m kernelgym.cli.service stop --profile v1
# or with the installed entrypoint:
kernelgym-service stop --profile v1
# or the convenience wrapper (mirrors deploy_node.sh):
bash stop_node.sh
```

A typical restart cycle inside the container is `bash stop_node.sh && bash deploy_node.sh`. For a cold restart that also removes local Redis persistence and KernelGym compile/work caches before launching, use `bash deploy_node.sh --clear-cache`.

The deployment convenience script is container-only. It runs `set_env.sh`, ensures the pinned Redis packages are present from the offline bundle, sources the node-local venv, and validates the runtime. It does not create or install the venv; run `ensure_venv.sh` once when bootstrapping a container or when packages need repair. It always stops existing KernelGym worker processes before starting worker-only nodes.

## Mode 1: Physical Host, Then Docker

Use this mode for external reward nodes such as `192.168.16.39` and `192.168.16.40`, where the operator starts
from the physical host. Host-level duties happen before starting the container:

1. Stop old reward services if needed.
2. Lock GPU clocks on the host.
3. Start or replace the Docker container.
4. Enter the container and ensure the node-local venv plus Redis there with CUDA 12.9.
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
source /root/kernelgym-reward-only/.venv/bin/activate
python -m kernelgym.cli.service start-local --profile v1
```

The same startup can be run with:

```bash
bash deploy_node.sh
```

Worker-only multi-node deployment uses `deploy_node.sh` from inside each container after the node-local venv exists.

## Mode 2: Already Inside A Container

Use this mode when the operator is already in the runtime container. Do not start Docker from inside this
container. From the repo root, create the node-local venv and start services directly:

```bash
bash set_env.sh
bash ensure_venv.sh --recreate
source /root/kernelgym-reward-only/.venv/bin/activate
python -m kernelgym.cli.service start-local --profile v1
```

After the node-local venv exists, the single-node convenience entrypoint is:

```bash
bash deploy_node.sh
```

Worker-only nodes follow the same rule: use `bash deploy_node.sh --join <primary>` from inside each container.

## Convenience Scripts

Single node (local-only, no remote workers):

```bash
bash deploy_node.sh
```

Cluster — the size is dynamic, so there is no node count to declare:

```bash
bash deploy_node.sh --cluster              # on the primary, ready for joins
bash deploy_node.sh --join 192.168.16.40   # on each worker node (primary's address)
```

The script is intended to run from inside containers, one invocation per host. `--cluster` makes the primary accept remote workers; `--join <primary>` brings a node in with a server-allocated id (no rank). Each node auto-detects the logical CUDA devices visible inside its container, so 8-card and 4-card containers can use the same command. After all current workers registered under the local hostname are fully healthy (a transient `degraded_check` is not ready; offline or stale registrations are ignored), the launcher submits a unique `force_refresh=true` CUDA-Agent request with correctness, performance timing, and CUDA profiling explicitly affined to that hostname. This makes each node warm its own CPU compile and GPU execution path before deployment returns; the HTTP wall timeout defaults to 1800 seconds and can be changed with `--startup-warmup-timeout`. The server task budget follows that value with 60 seconds reserved for HTTP completion, so increasing the option also accommodates slower cold compilation. `--no-startup-warmup` is an explicit diagnostic escape hatch and is required when intentionally launching with `--cpu-compile-workers 0`, because an affined split compile cannot run without local CPU capacity. Use `--gpu-devices 0,1` only to select a subset; `--cpu-compile-workers N` / `--cpu-workers N` overrides CPU compile capacity. Add `--block-terminal` to any role when the deploy command should stay in the foreground; after startup and warmup succeed, Ctrl-C, SIGTERM, or a terminal hangup stops this node's KernelGym services before the command exits. The older `--nnodes N --node-rank R --master-addr <ip>` form still works but is deprecated in favor of `--cluster` / `--join`.

## Multi-Node Tutorial

This tutorial assumes two physical reward nodes, `192.168.16.40` as the primary and `192.168.16.39` as the worker-only node. Replace the addresses and add more nodes the same way for larger clusters. The primary hosts Redis and the HTTP API on fixed ports; every worker-only node connects back to the primary and registers its local GPU/CPU workers.

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

### 2. Prepare the node-local environment in each container

Run this on the primary container and every worker-only container:

```bash
bash set_env.sh
bash ensure_venv.sh --recreate
```

`deploy_node.sh` activates `/root/kernelgym-reward-only/.venv` itself, so you do not need to keep the shell activated after this step.

### 3. Start the primary node first

On the primary, start it cluster-ready:

```bash
bash deploy_node.sh --cluster
```

`--cluster` enables Redis remote access on port `20110` (forwards `--redis-remote-access` to the service CLI), which disables Redis protected mode inside the container so worker nodes can connect to the primary Redis. A plain `deploy_node.sh` (single node) does not enable this.

Wait until the primary reports API readiness:

```text
API ready: http://127.0.0.1:20111/health
```

### 4. Start each worker node

On every non-primary node, point it at the primary — no rank, no count:

```bash
bash deploy_node.sh --join 192.168.16.40
```

Add as many nodes as you like the same way; each gets a stable, server-allocated id. If a worker node should use fewer CPU compile workers than the profile default, pass the override on that node:

```bash
bash deploy_node.sh --join 192.168.16.40 --cpu-workers 8
```

(The deprecated `--nnodes N --node-rank R --master-addr <ip>` form is still accepted for all of the above.)

### 5. Verify from the primary

Run the checks inside the primary container. `check_node.sh -v` shows local and remote GPU workers with fresh heartbeats, health/admission state, quarantine scope, and CPU workers from each node. Any Redis worker/device quarantine produces `WARN`, increments `gpu_workers_quarantined`, and adds a detail table with scope, fault class, and reason. If the quarantine scan is incomplete, `quarantine_scan` reports `incomplete`, the count is `unknown`, and the overall status fails closed to `WARN`.

```bash
bash check_node.sh -v
bash test_reward.sh
```

Expected high-level signals:

```text
api_status:          healthy
gpu_workers_fresh:   <all>/<all>
gpu_workers_ready:   <all>/<all>
gpu_workers_quarantined: 0
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

## Hot-Plugging Nodes

Worker nodes can join or leave a running cluster without restarting the primary and without interrupting in-flight tasks. Everything is done with `deploy_node.sh` / `stop_node.sh` per host; you do not edit Redis by hand. A joining node's workers start pulling from the shared queues within one heartbeat; a leaving node's workers drop out of the load balancer within the heartbeat timeout.

### Start the primary so nodes can join

Start the primary with `--cluster` so its Redis accepts remote workers (a plain `deploy_node.sh` binds Redis locally and no remote node can connect). There is **no node count** — the cluster size is dynamic:

```bash
bash deploy_node.sh --cluster   # primary, ready for nodes to join
```

The primary runs standalone until others join. (`--cluster` simply enables remote Redis access; the old `--nnodes 2 --node-rank 0 --master-addr <self>` form still works but is deprecated.)

### Add a worker node

On the new node, inside the container, after the node-local venv exists, point it at the primary:

```bash
bash deploy_node.sh --join 192.168.16.40    # 192.168.16.40 = primary address
```

No rank or count: the server auto-allocates a stable per-hostname node id (idempotent across re-joins). The primary keeps serving; the node registers its GPU/CPU workers and they appear in `workers/status` within a heartbeat. Verify from the primary with `bash check_node.sh -v`. (Legacy `--nnodes N --node-rank R --master-addr <ip>` is still accepted.)

### Remove a worker node

On that node, stop its local services:

```bash
bash stop_node.sh
```

This stops only that node's workers (it is host-local). They unregister and leave the primary's `workers/status`; any that were mid-task drop from the load balancer within the heartbeat timeout. The rest of the cluster is unaffected.

### Stop the whole cluster

`stop_node.sh` is **host-local** — it kills the local KernelGym processes and clears the Redis state it can reach; it does **not** reach across the network to other hosts. Running it on the primary therefore stops the primary and wipes shared Redis but leaves worker-node processes running (orphaned, no longer heartbeating). To tear a cluster down cleanly, run `stop_node.sh` on each worker node first, then on the primary (this order is also why the [restart step](#6-restart-or-stop) stops workers before the primary). There is no built-in single-command, cluster-wide stop.

### Advanced: add or remove one worker without a node restart

`deploy_node.sh` operates at node granularity (it restarts all of a node's workers). On the **primary** only, the `worker_monitor --persistent` reconciles a desired-state set `kernelgym:expected_workers` every `WORKER_MONITOR_INTERVAL` (30 s), so you can add or retire a single GPU/CPU worker without disturbing the node's other workers:

```bash
# add one GPU worker on cuda:3 (monitor launches it within ~30 s)
redis-cli -p 20110 SADD kernelgym:expected_workers worker_gpu_3
redis-cli -p 20110 HSET kernelgym:expected_worker:worker_gpu_3 device cuda:3 hostname "$(hostname)" node_id v1

# retire it: drop from desired state FIRST (else the monitor respawns it), then stop the process
curl -s -X POST 'http://127.0.0.1:20111/worker/evict_from_lb?worker_id=worker_gpu_3'   # optional drain
redis-cli -p 20110 SREM kernelgym:expected_workers worker_gpu_3
redis-cli -p 20110 DEL  kernelgym:expected_worker:worker_gpu_3
PID=$(redis-cli -p 20110 HGET kernelgym:worker_process:worker_gpu_3 pid); [ -n "$PID" ] && kill -TERM "$PID"
```

`POST /worker/evict_from_lb?worker_id=<id>` alone just drains a worker from the load balancer (no kill, no respawn) to quarantine a flaky one. This expected-worker set exists only on the primary; do **not** add worker-only-node ids to it, or the primary would try to launch them locally with the wrong device.

## Profiler Timestamp Policy (CUPTI TSC Bug)

Background and design: `docs/design-doc/PROFILER_EMPTY_CAPTURE.md`. CUDA 12.6u2–13.0 CUPTI can drop CUDA kernel records from `torch.profiler` (empty captures, `time_coverage=0`) when Kineto registers its TSC timestamp callback.

**Default deployment needs no manual step.** Profile `v1` sets `KERNELGYM_CUPTI_TSC_SHIM=true`: at startup the service builds a version-gated `LD_PRELOAD` shim (`kernelgym/native/cupti_tsc_shim.cpp`, artifact under `.native/`), preloads it into all service processes, and sets `KINETO_TSC_FIXED=true`, so profiling runs a single candidate forward per profiler context. Every gate fails open to the legacy 10-forward workaround: shim build failure skips injection, and workers that detect an expected-but-unengaged shim retry empty captures with the legacy count.

What the operator should check after a deploy:

1. `deploy_node.sh` output (from `scripts/validate_runtime.py`) contains a `=== Validate CUPTI TSC shim ===` block ending in `shim_probe=OK`, with `shim_state=1` (engaged, CUDA 12.6–13.0) or `shim_state=2` (passthrough on fixed CUPTI 13.1+). Any `WARNING:` line there means the legacy workaround is active — the service still works, just with slower profiling.
2. Profiling-enabled results should show `kg_kernel_perf_num_profile_trials: 1` and `cupti_tsc_shim_state: 1` inside `metadata.profiling`.
3. Watch the empty-capture rate: `kg_kernel_profiling_empty_initial` / `kg_kernel_profiling_retries_used` / `kg_kernel_profiling_empty_final` in result metadata, and `[Profiling] empty-capture:` warnings in the worker logs (e.g. `logs/v1/workers.log`). Keep `PROFILING_RETRY_COUNT=1`.

Manual operations:

- **Disable the shim** (rollback to the always-on 10-forward workaround):

  ```bash
  export KERNELGYM_CUPTI_TSC_SHIM=false   # ambient env deliberately overrides the profile for this key;
                                          # edit kernelgym/deployment_profiles.py for a permanent change
  bash stop_node.sh && bash deploy_node.sh --cluster
  ```

  or force the legacy count directly with `export NUM_PROFILING_TRIALS=10` and restart. The timestamp callback is process-level state chosen before CUPTI activity enable, so a restart is mandatory for any of these switches; never flip them on a running service.

- **After deploying a Kineto/torch build that version-gates the TSC callback in source**: set `KERNELGYM_CUPTI_TSC_SHIM=false` and `KINETO_TSC_FIXED=true` in the profile env and restart; the declaration is then trusted without the shim.

- **After upgrading the node to a matched CUDA/CUPTI 13.1+ stack** (driver must be >= 580; do not swap `libcupti.so` alone): no config change — the shim passes registration through to the real CUPTI, and auto resolution independently detects the fixed CUDA version. Once the fleet is on 13.1+, remove `KERNELGYM_CUPTI_TSC_SHIM` from the profile.

## Verification

Run lint and tests from the node-local CUDA 12.9 venv:

```bash
source /root/kernelgym-reward-only/.venv/bin/activate
ruff format .
ruff check .
pytest
```

On a GPU runtime with CUDA 12.9, `tests/kernelbench/backends/test_cuda_agent_gpu.py` compiles, loads, and runs a minimal
CUDA-Agent extension. Without GPU, torch, nvcc, or executable `/dev/shm`, that test skips.
