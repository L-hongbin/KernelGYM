# KernelGym Reward Node Spec

This file records run-specific deployment details. Keep stable repo conventions in
`AGENTS.md`; keep volatile node, port, and runtime topology details here.

## Reward Nodes

| Profile | Host | SSH | Runtime | GPUs | Intended use |
| --- | --- | --- | --- | --- | --- |
| `reward-39` | `192.168.16.39` | `ssh -p 23452 root@192.168.16.39` | External physical host | 8 | Existing reward node |
| `reward-40` | `192.168.16.40` | `ssh -p 23452 root@192.168.16.40` | External physical host | 8 | New reward testing node |

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
