# KernelGym 优化记录

## 适用范围

这份文档记录当前 KernelGym 在 CUDA kernel 评测链路上的优化项，重点覆盖：

- `/evaluate`
- `split_compile_and_execute=true`
- CUDA-agent backend
- `cpp_extension.load` 与 `manual_ninja` 两条编译路径
- GPU worker 与 CPU compile worker 协同
- Redis/local cache 对编译阶段的影响

文档同时记录：

- 当前推荐配置
- 可调参数和作用边界
- 已实现优化
- 已修复问题
- 需要复测的 benchmark 方向

## 当前推荐配置

当前默认目标是让 CUDA-agent 走 `manual_ninja`，并开启可复用的编译 object cache。

```bash
export CUDA_BUILD_BACKEND=manual_ninja
export CACHE_INDEX=redis
export COMPILE_ARTIFACT_CACHE_INDEX=redis
export MANUAL_NINJA_OBJECT_CACHE=true
export MANUAL_NINJA_OBJECT_CACHE_INDEX=redis
export DETAILED_COMPILE_TIMING=false
export WORKER_USE_BLOCKING_TASK_POLL=true
export WORKER_TASK_POLL_BLOCK_TIMEOUT_SEC=1
export GPU_WORKER_CLASS=legacy
export WORKER_SPLIT_COMPILE_MODE=background_compile
export WORKER_BACKGROUND_COMPILE_LIMIT=2
export WORKER_COMPILE_POOL_SIZE=2
export WORKER_COMPILE_MAX_TASKS_PER_WORKER=10
export GPU_WORKERS_POLL_CPU_TASKS=true
export WORKER_POOL_SIZE=1
export MAX_TASKS_PER_WORKER=1
```

请求侧建议：

```json
{
  "split_compile_and_execute": true,
  "enable_compile_artifact_cache": true
}
```

说明：

- `manual_ninja` 是当前默认编译 backend。
- `MANUAL_NINJA_OBJECT_CACHE=true` 是主要优化项。
- `enable_compile_artifact_cache` 仍可用于整段 compile artifact 缓存实验，但不是当前默认推荐项。
- `DETAILED_COMPILE_TIMING=false` 是推荐默认值；详细耗时统计属于分析开关，不属于性能优化开关。

## auto_configure 输出

`scripts/auto_configure.sh` 会生成 `.env.<hostname>`，并输出当前优化相关选项。

### env 按 hostname 区分

KernelGym 的本机 env 文件默认按 hostname 隔离：

```bash
.env.<hostname>
```

例如：

```bash
.env.ai-16-21
.env.ai-11-229
```

脚本选择规则：

- 如果显式设置 `ENV_FILE=/path/to/.env.xxx`，优先使用该文件。
- 否则 `start_all_with_monitor.sh` 优先读取当前机器的 `.env.<hostname>`。
- 如果 `.env.<hostname>` 不存在，`auto_configure.sh` 会为当前 hostname 生成一份。
- `.env` 只作为 legacy fallback，不建议多机共享使用。

这样做的原因：

- 不同机器的 `API_HOST/API_PORT/REDIS_PORT/GPU_DEVICES` 可能不同。
- 不同机器的 `TORCH_CUDA_ARCH_LIST`、CUDA/PyTorch 版本、临时目录策略也可能不同。
- 按 hostname 区分可以避免 A 机器的端口、GPU 或 `/dev/shm` 配置覆盖 B 机器。

当前会输出：

- `CUDA_BUILD_BACKEND`
- `MANUAL_NINJA_OBJECT_CACHE`
- `TORCH_CUDA_ARCH_LIST`
- `CPU_COMPILE_WORKERS`
- `KERNELGYM_TMP_ROOT`
- `KERNELGYM_TMPDIR`
- `/dev/shm` 建议项和容量信息
- `CACHE_INDEX`
- `COMPILE_ARTIFACT_CACHE_INDEX`
- `MANUAL_NINJA_OBJECT_CACHE_INDEX`
- `DETAILED_COMPILE_TIMING`

`KernelGym optimization options:` 标题在交互式终端中用蓝色标识。

### 临时目录

新增参数：

```bash
KERNELGYM_TMP_ROOT=/tmp
KERNELGYM_TMPDIR=${KERNELGYM_TMP_ROOT}/kernelgym_${HOSTNAME}
```

默认使用 `/tmp`。如果机器上存在 `/dev/shm`，`auto_configure.sh` 会提示：

```bash
KERNELGYM_TMP_ROOT=/dev/shm may be faster for temporary compile files
```

并打印 `/dev/shm` 的总大小和可用大小。

注意：

- `/dev/shm` 可能更快，但容量和 tmpfs 限额依赖机器配置。
- 大并发编译时要确认 `/dev/shm` 空间足够。
- `start_all_with_monitor.sh` 和 `stop_all.sh` 都已识别 `KERNELGYM_TMP_ROOT`。

## 主要参数开关

### 1. 编译 backend

```bash
CUDA_BUILD_BACKEND=manual_ninja|cpp_extension_load
```

含义：

- `cpp_extension_load`：使用 `torch.utils.cpp_extension.load`。
- `manual_ninja`：显式生成 `build.ninja` 并调用 PyTorch 的 ninja build helper。

当前状态：

- 默认推荐 `manual_ninja`。
- `cpp_extension_load` 保留作为对照和 fallback。
- 两条路径都需要保持编译结果一致。

### 2. manual_ninja object cache

```bash
MANUAL_NINJA_OBJECT_CACHE=true|false
MANUAL_NINJA_OBJECT_CACHE_INDEX=redis|fs
```

含义：

- 缓存并复用 `manual_ninja` 产生的 `.o`。
- 当前可复用对象包括：
  - `binding_registry.o`
  - `generated_binding.o`
  - `generated.cuda.o`
- `binding.o` 当前不作为通用对象复用，因为它包含模块级 `PYBIND11_MODULE`，和具体扩展模块名绑定。

cache key 维度包括：

- object name
- 编译 rule
- 源文件内容 digest
- 相关 header digest
- 归一化后的 `build.ninja`
- Python/PyTorch/CUDA 版本信息

当前状态：

- 默认开启。
- 默认 index 使用 Redis。
- object 文件仍在本地文件系统，Redis 存 index/metadata。

### 3. 统一 cache index

```bash
CACHE_INDEX=redis|memory
COMPILE_ARTIFACT_CACHE_INDEX=${CACHE_INDEX}
MANUAL_NINJA_OBJECT_CACHE_INDEX=${CACHE_INDEX}
```

含义：

- `CACHE_INDEX` 是默认 cache index 配置。
- compile artifact cache 和 manual ninja object cache 可以分别覆盖。

当前状态：

- 默认推荐 `redis`。
- Redis 管理 metadata/index；大文件产物仍保留在本地路径。
- public API 返回会隐藏 `cache_key`、`cache_index`、`cache_scope`、`work_dir`、`so_path` 等内部字段。

### 4. compile artifact cache

```json
{
  "enable_compile_artifact_cache": true
}
```

含义：

- 缓存完整 compile artifact。
- 命中后可以跳过实际编译，直接进入 load/execute。

当前状态：

- 支持 memory/redis index。
- Redis value 存 artifact metadata，并校验 `so_path/work_dir` 是否仍存在。
- 这是更激进的 cache，适合重复 payload 测试。
- 对真实 RL 生成任务，kernel 内容经常变化，命中率依赖样本重复度。

### 5. 详细编译耗时

```bash
DETAILED_COMPILE_TIMING=true|false
```

含义：

- `false`：只保留整体编译耗时。
- `true`：额外读取 `.ninja_log`，拆分 nvcc/C++/link 等阶段耗时。

当前状态：

- 默认 `false`。
- 这是分析开关，不是优化项。
- 开启后会多读统计文件，适合定位瓶颈，不建议长期默认开启。

### 6. TORCH_CUDA_ARCH_LIST

```bash
TORCH_CUDA_ARCH_LIST=8.0
```

含义：

- 控制 PyTorch CUDA extension 编译目标架构。
- A800 对应 `8.0`。

当前状态：

- `auto_configure.sh` 会检测本机可见 GPU 的 compute capability。
- 如果检测到多个 arch，会输出推荐值。
- `device_info.compute_capability` 会优先使用指定的 `TORCH_CUDA_ARCH_LIST`，使不同机器指定同一 arch 时返回信息一致。

### 7. GPU worker 类型

```bash
GPU_WORKER_CLASS=legacy|background_compile
```

含义：

- `legacy`：使用原始 `GPUWorker`。
- `background_compile`：使用 `BackgroundCompileGPUWorker`。

当前状态：

- 当前默认仍建议 `legacy`。
- `BackgroundCompileGPUWorker` 适合定向实验。
- `WORKER_SPLIT_COMPILE_MODE=background_compile` 控制 split compile 模式下的后台 compile 行为。

### 8. CPU compile worker

```bash
CPU_COMPILE_WORKERS=<int>
GPU_WORKERS_POLL_CPU_TASKS=true|false
```

含义：

- CPU compile worker 处理 compile-stage 任务。
- GPU worker 在 GPU 队列空闲时可以轮询 CPU/compile 队列。

当前状态：

- 对 `split_compile_and_execute=true` 有价值。
- 对单体 `split=false` 基本没有收益。
- `GPU_WORKERS_POLL_CPU_TASKS=true` 可以避免没有 CPU compile worker 时 GPU 空转。

### 9. worker pool 参数

```bash
WORKER_POOL_SIZE=1
MAX_TASKS_PER_WORKER=1
WORKER_COMPILE_POOL_SIZE=2
WORKER_COMPILE_MAX_TASKS_PER_WORKER=10
WORKER_BACKGROUND_COMPILE_LIMIT=2
```

含义：

- `WORKER_POOL_SIZE/MAX_TASKS_PER_WORKER` 控制 execute subprocess。
- `WORKER_COMPILE_POOL_SIZE/WORKER_COMPILE_MAX_TASKS_PER_WORKER` 控制 compile subprocess。
- `WORKER_BACKGROUND_COMPILE_LIMIT` 控制后台 compile 并发上限。

当前状态：

- execute 仍保持强隔离：`WORKER_POOL_SIZE=1`、`MAX_TASKS_PER_WORKER=1`。
- compile subprocess 可以复用更多任务，降低进程重启开销。

## 已实现的优化项

### 1. compile / execute 拆分

已实现。

效果：

- compile 和 GPU execute 可以分阶段执行。
- CPU worker 可以分担 compile。
- GPU worker 可以更聚焦在 kernel correctness/performance 验证。

### 2. manual_ninja 编译路径

已实现。

效果：

- 可以显式控制 ninja build。
- 可以在 ninja 层做 object cache。
- 可以对比 `cpp_extension.load` 和 `manual_ninja` 的编译结果与耗时。

当前状态：

- 默认使用 `manual_ninja`。
- 保留 `cpp_extension_load` 作为 fallback 和对照。

### 3. manual_ninja object cache

已实现。

效果：

- 相同 object 可复用。
- cache miss 时正常并行编译，并在编译成功后写入 cache。
- cache hit 时改写 link 输入，直接链接已缓存的 `.o`。

当前状态：

- 支持 Redis index。
- 支持本地 fs index。
- Redis 只管理 key/index/metadata，不直接存 `.o` 二进制内容。

### 4. compile artifact cache

已实现。

效果：

- 完整 compile artifact 可复用。
- 命中后可跳过 compile stage。

当前状态：

- 支持 memory/redis index。
- public response 隐藏内部路径和 cache key。
- 主要用于重复样本或 benchmark，对真实 RL 任务命中率依赖生成重复度。

### 5. 阻塞式任务轮询

已实现。

效果：

- 降低 Redis 忙轮询开销。

当前状态：

- `WORKER_USE_BLOCKING_TASK_POLL=true` 是默认推荐。

### 6. 编译耗时统计

已实现。

当前默认：

- 不开启 detail 时，只保留整体编译耗时。
- 开启 detail 时，额外输出：
  - `cpp_extension_load_wall_sec`
  - `manual_ninja_build_wall_sec`
  - `manual_ninja_import_wall_sec`
  - `copy_so_sec`
  - `total_wall_sec`
  - ninja `.ninja_log` 解析出的 object/link 阶段信息

### 7. device_info 元数据

已实现。

返回字段：

```json
{
  "device_info": {
    "gpu_name": "...",
    "compute_capability": "8.0",
    "cuda_version": "...",
    "driver_version": "..."
  }
}
```

当前状态：

- `hardware` 已改为 `device_info`。
- 顶层 `metadata.gpu_name` 已移除。
- compile 失败也会尽早返回 `device_info`。

### 8. kernel 使用检测字段收敛

已实现。

当前 public 字段：

```json
{
  "is_kernel_used": true
}
```

说明：

- 旧的 `custom_kernel_used`、`profiler_used`、`cuda_profiler_used`、`triton_profiler_used` 不再作为 public 字段返回。
- 读取旧 Redis 结果时会兼容映射到 `is_kernel_used`。
- `profiler_matches` 只有在仍代表当前有效检测结果时才返回；如果最终结论被 performance profiling 覆盖，会移除该字段。

## 已修复问题

### 1. `forced_compile/lock` 错误

问题：

- `torch.utils.cpp_extension` 会创建 `build_directory/lock`。
- 父目录不存在时会失败。

修复：

- 进入 torch extension 编译前先创建 `build/forced_compile`。

### 2. PyTorch `_prepare_ldflags` 签名不兼容

问题：

- PyTorch 不同版本的私有函数 `_prepare_ldflags` 签名不同。
- 某些版本包含 `with_sycl`，PyTorch 2.8 等版本没有。
- 固定传 5 个位置参数会导致：

```text
_prepare_ldflags takes 4 positional arguments but 5 were given
```

修复：

- `manual_ninja` 路径按运行时签名动态传参。
- 有 `with_sycl` 时传入，没有时自动跳过。

### 3. split compile 后 execute 使用 sanitized artifact 失败

问题：

- public response 会隐藏 `so_path/work_dir/cache_key` 等内部字段。
- 如果 execute 阶段误用 sanitized artifact，会找不到 `.so`。

修复：

- 内部使用 `_internal_compile_artifact` 传递完整 artifact。
- public response 继续只返回 sanitized `compile_artifact`。

### 4. stop/start 脚本过早退出

问题：

- `stop_all.sh` 使用 `set -euo pipefail`。
- env 缺字段、Redis 已停止、cache 目录不存在等 best-effort cleanup 场景可能导致非 0 退出。
- `start_all_with_monitor.sh` 在启动前调用 stop 脚本，因此会被阻断。

修复：

- `stop_all.sh` 对 env 读取、Redis/cache/compile cleanup 做 best-effort。
- `start_all_with_monitor.sh` 对 stop 脚本非 0 返回只 warning，继续启动。
- `stop_all.sh` 末尾显式 `exit 0`。

### 5. Redis 残留

问题：

- stop 后可能还有本地 Redis 进程残留。

修复：

- `stop_all.sh` 会尝试 `redis-cli SHUTDOWN`。
- 失败时 fallback kill 当前端口上的 `redis-server`。

## 当前 benchmark 状态

已有历史结论：

- `split=true` 下，GPU+CPU compile worker 通常优于 GPU-only。
- `BackgroundCompileGPUWorker` 在部分配置下和 legacy 持平，但没有稳定领先。
- compile artifact cache 对重复 payload 有明显收益。

需要重新复测：

- `manual_ninja cold` vs `cpp_extension.load`，包含 GPU execute。
- `manual_ninja object cache` cold/no-cache 对比。
- Redis index 和 fs index 对 object cache 管理开销的影响。
- `KERNELGYM_TMP_ROOT=/tmp` vs `/dev/shm` 对 compile wall time 的影响。

旧 benchmark 数字不再作为当前推荐依据，因为当前默认 backend、cache 方式、metadata 处理和 timing 统计方式都已更新。

## 复测命令

### manual_ninja cold vs cpp_extension.load

```bash
ENV_FILE=/nfs/FM/lihongbin/CODE/KernelGYM_V3/.env.ai-16-229 \
SERVER_URL=http://<host>:<port> \
GPU_WORKERS=4 \
CPU_WORKERS=4 \
SCENARIOS=gpu_cpu_true \
WAIT_READY_TIMEOUT=300 \
RESULTS_ROOT=/nfs/FM/lihongbin/CODE/KernelGYM_V3/logs/manual_ninja_vs_cpp_load \
CUDA_BUILD_BACKEND=manual_ninja \
MANUAL_NINJA_OBJECT_CACHE=true \
CACHE_INDEX=redis \
test_script/run_evaluate_torch_cuda_arch_benchmark.sh --limit 20 --concurrency 8
```

对照：

```bash
CUDA_BUILD_BACKEND=cpp_extension_load
```

### object cache no-cache/cold 对比

no-cache：

```bash
MANUAL_NINJA_OBJECT_CACHE=false
```

cold：

```bash
MANUAL_NINJA_OBJECT_CACHE=true
MANUAL_NINJA_OBJECT_CACHE_INDEX=redis
```

测试前清理 cache：

```bash
./stop_all.sh
```

如需清理本地 compile/cache 目录，在提示时输入 `y`。

### `/dev/shm` 临时目录实验

```bash
KERNELGYM_TMP_ROOT=/dev/shm \
ENV_FILE=/nfs/FM/lihongbin/CODE/KernelGYM_V3/.env.$(hostname) \
bash scripts/auto_configure.sh --force
```

然后重启：

```bash
./start_all_with_monitor.sh
```

对照默认：

```bash
KERNELGYM_TMP_ROOT=/tmp
```

## 下一步优化候选

### 1. binding 编译复用

当前状态：

- `generated_binding.o` 可通过 hash 复用。
- `binding.o` 由于模块名绑定，当前不做通用复用。

后续方向：

- 继续评估固定 launcher/fallback 的可行性。
- 但 submission 当前仍要求写 `APPLY_BINDINGS`，固定 ABI 方案灵活性不足。

### 2. execute subprocess 复用

当前问题：

- `MAX_TASKS_PER_WORKER=1` 隔离性强。
- 但 subprocess 重启成本高。

后续方向：

- 定向测试更高的 `MAX_TASKS_PER_WORKER`。
- 需要关注 CUDA context、模块卸载和跨任务状态污染。

### 3. loaded `.so` / session 复用

当前状态：

- compile artifact 和 object cache 已实现。
- `.so` load/session 级别复用还没有实现。

后续方向：

- 增加进程内 `artifact -> loaded handle/session` cache。
- 需要谨慎处理 cleanup、CUDA context 生命周期和模块名冲突。

### 4. cache 管理进一步统一

当前状态：

- `CACHE_INDEX=redis` 作为默认 index。
- object 文件和 `.so` 文件仍在本地文件系统。

后续方向：

- Redis 继续作为 index/metadata 管理中心。
- 不建议直接把 `.o` 大二进制放 Redis，除非有明确的容量、淘汰和跨节点共享策略。
