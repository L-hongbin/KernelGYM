# 共享盘与节点本地 Python 环境的效率差异

## 结论

KernelGym 的源码、日志和长期证据可以继续放在共享盘，但频繁创建的 GPU 子进程不应从共享盘虚拟环境加载 Python、Torch 和 CUDA 依赖。2026-08-26 的隔离 A/B 实验显示，把依赖环境从共享 NFS 移到节点本地磁盘后，warm `import torch` 从 15.91 秒降到 1.15 秒，4 路并发导入从 22.45 秒降到 1.27 秒。

生产拥堵并非只有一个原因：共享环境使每个新子进程本身启动较慢，而当时每节点仅允许同时创建两个子进程（c2），又把单次慢启动放大成数十到数百秒的 spawn-slot 和 warm-pool 等待。最终方案因此同时采用节点本地环境和 c8；源码、`logs/`、`py_logs/`、编译产物和证据仍保留在共享 checkout。

## 实测差异

下表均来自同一台 4 × A800 隔离 debug 容器中的 fresh-process 测量；两侧使用 CPython 3.12.3、`torch==2.11.0+cu129` 和 CUDA 12.9。

| 场景 | 共享 NFS 环境 | 节点本地环境 | 改善 |
| --- | ---: | ---: | ---: |
| warm 串行 `import torch` | 15.91 s | 1.15 s | 13.8× |
| 4 路并发 `import torch` | 22.45 s | 1.27 s | 17.7× |
| 8 个真实 worker constructor，c2 | 63.10 s | 8.41 s | 7.5× |
| 8 个真实 worker constructor，c8 | 16.42 s | 4.72 s | 3.5× |
| 4 个 pool × 6 个零间隔任务，c2 | 254.87 s | 33.35 s | 7.6× |
| 4 个 pool × 6 个零间隔任务，c8 | 79.52 s | 16.25 s | 4.9× |

迁移后的生产复盘覆盖约 12.9k 次 spawn-slot 获取：等待时间 p99 为 0.00 秒、最大 0.01 秒，child READY-after-containment 平均 1.69 秒。迁移前出现过的数百秒 `_get_idle_worker` 饥饿没有再次出现。

## 可以确认和不能确认的原因

可以确认的是，warm 共享环境即使 block reads 和 major faults 都为零，导入仍显著慢于本地环境，因此普通数据页缓存不能消除这个差异；本地 smoke 中 CUDA init、set-device、allocation 和 synchronize 合计只有约 0.35 秒，CUDA 初始化本身也不是十几秒延迟的主体。

NFS metadata/RPC、动态加载和运行时等待是最合理的解释，但原实验没有采集 syscall 或 NFS RPC trace，不能给这些机制分别定量。共享与本地实验也没有逐批完全交错执行，时间变化的 NFS 负载仍是未完全排除的混杂因素。因此这里报告可重复观察到的路径差异，不把未经 tracing 的机制当成已证实根因。

## 最小复现

使用 [`scripts/reproduce_runtime_import_latency.py`](../scripts/reproduce_runtime_import_latency.py)。脚本只依赖 Python 标准库，不启动 KernelGym 服务，不初始化 CUDA，也不修改两个环境。它让两个解释器分别启动 fresh child，交错执行串行导入，并按 ABBA 顺序执行同环境并发批次；摘要写到 stderr，包含每次原始记录和模块来源的 JSON 写到 stdout。

在隔离 debug 容器中准备两个包版本一致的环境：一个位于 NFS 等共享文件系统，另一个位于容器本地 `overlay` 或本地块设备。不要为了复现重新启用生产服务的废弃共享 `.venv`。先确认解释器所在文件系统和 Torch 版本：

```bash
findmnt -T /nfs/path/to/shared-env/bin/python
findmnt -T /root/path/to/local-env/bin/python
/nfs/path/to/shared-env/bin/python -I -c 'import torch; print(torch.__version__, torch.__file__)'
/root/path/to/local-env/bin/python -I -c 'import torch; print(torch.__version__, torch.__file__)'
```

执行最小对比：

```bash
python3 scripts/reproduce_runtime_import_latency.py \
  --shared-python /nfs/path/to/shared-env/bin/python \
  --local-python /root/path/to/local-env/bin/python \
  --module torch \
  --warmup-runs 1 \
  --serial-runs 6 \
  --parallel-runs 8 \
  --parallelism 4 \
  > /tmp/kernelgym_import_latency.json
```

判断结果时优先看 `summaries.<label>.<mode>.child_import_s`，并检查 `module_origins` 确认两侧确实来自预期环境。`parent_wall_s` 还包含解释器启动和父进程调度开销。如果只验证脚本本身，可把两个解释器都指向同一个 Python，并用 `--module json` 做无 Torch smoke；这不会复现存储差异，只验证测量流程。

这个脚本有意只复现最主要且最容易隔离的 fresh-process import 差异。需要复现真实 `PersistentWorker`、CUDA READY handshake 或 warm-pool replenishment 时，使用完整的 [`scripts/benchmark_worker_spawn.py`](../scripts/benchmark_worker_spawn.py) 和原始实验参数。

## 文档与证据分层

本文是唯一推荐的阅读入口。其余文件保留为不同阶段的原始证据，而不是四份平级说明文档：

| 证据 | 角色 |
| --- | --- |
| [`docs/evidence/performance/20260826_spawn_debug/README.md`](evidence/performance/20260826_spawn_debug/README.md) | 完整 A/B benchmark、限制和原始 JSON 索引。 |
| [`docs/evidence/performance/20260826_local_venv_c8_migration/README.md`](evidence/performance/20260826_local_venv_c8_migration/README.md) | 锁定环境、安装、测试和 constructor/pool 部署前验证。 |
| [`logs/deploy_evidence/20260826_local_venv_c8_restart/summary.md`](../logs/deploy_evidence/20260826_local_venv_c8_restart/summary.md) | node 20/21 的实际重启、warmup 和最终进程状态。 |
| [`docs/evidence/performance/20260827_post_migration_status_and_capacity.md`](evidence/performance/20260827_post_migration_status_and_capacity.md) | 14 小时生产回归以及独立的 GPU 容量分析。 |

对应实现提交为 `4812975`（`fix(runtime): use local offline environments and c8 spawning`），部署文档提交为 `6811715`。
