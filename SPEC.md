# KernelGym Reward Node Spec

This file records run-specific deployment details. Keep stable repo conventions in
`AGENTS.md`; keep volatile node, port, and runtime topology details here.

## Reward Nodes

| Node | Host | Host SSH | Runtime | GPUs | Intended use |
| --- | --- | --- | --- | --- | --- |
| node 39 | `192.168.16.39` | `ssh chenshuailin@192.168.16.39` | External physical host | 8 | Existing reward node |
| node 40 | `192.168.16.40` | `ssh chenshuailin@192.168.16.40` | External physical host | 8 | New reward testing node |

These are SSH commands for the external physical hosts. They land on the host, not inside the reward
container. Port `23452` is not valid for these nodes.

Runtime profile `v1` is defined by `kernelgym/deployment_profiles.py`; `auto` is an alias for `v1`.
Profile `v1` currently sets `DEFAULT_BACKEND=auto`, so omitted API backend fields are resolved by the reward service rather than defaulting to Triton. Worker processes on `.40` were checked with:

```text
KERNELGYM_CORRECTNESS_GPU_INPUTS=true
KERNELGYM_CORRECTNESS_MAX_WALL_S=20
KERNELGYM_CORRECTNESS_PASS_ON_BUDGET=true
KERNELGYM_CORRECTNESS_BUDGET_MIN_PASS_TRIALS=2
```

CUDA 12.9 runtime dependencies are pinned in `requirements-cuda129.txt`, including `apache-tvm-ffi==0.1.11`.
That file intentionally does not specify public index URLs; internal deployments must provide package indexes through
image, pip/uv config, or environment.

External physical hosts require GPU clock locking and container startup before
the reward service starts. The current container image is:

```text
192.168.14.129:80/library/slime:nightly-dev-20260430b
```

## Fixed Runtime Ports

| Service | Port |
| --- | --- |
| Redis | `20110` |
| API | `20111` |
| Metrics | `20112` |
