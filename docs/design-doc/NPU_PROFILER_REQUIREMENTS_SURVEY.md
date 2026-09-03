# 自研 NPU Profiler 功能优先级与返回信息 Survey

## 文档目的

本文面向内部自研 NPU、运行时和编译器团队，用于明确 profiler 应优先支持哪些功能，以及每项功能至少需要返回什么数据。目标不是复制某个现有 GPU 工具的 UI，也不讨论上层评测服务如何接入；目标是让算子、kernel、编译器和大模型训练团队能够用同一套 profile 证据回答四个问题：

1. 时间花在 host、runtime、排队、通信、搬运还是 NPU 计算上？
2. 哪个 framework op、编译器 op、device task 或 kernel 位于关键路径？
3. 热点 task 为什么慢：工作量不足、计算单元未吃满、内存、依赖、同步、资源占用还是通信？
4. 应该修改哪一层的什么内容，并如何验证修改确实改善了瓶颈？

本文把功能分为 P0、P1、P2。P0 是第一版必须形成闭环的能力；P1 是能够支持严肃 kernel/算子调优的深度能力；P2 是自动诊断、能效和大规模集群等增强能力。

## 给 NPU 团队的直接结论

建议按下面顺序建设，而不是先做“大而全的 counter 面板”：

| 顺序 | 优先级 | 能力 | 第一版必须回答的问题 |
| --- | --- | --- | --- |
| 1 | P0 | 统一时钟与全链路 correlation | 一个用户/framework op 最终触发了哪些 runtime dispatch、device task 和 engine 活动？ |
| 2 | P0 | Host + runtime + device engine timeline | 时间花在哪里，哪里在排队、空闲、同步或搬运？ |
| 3 | P0 | Top task、关键路径、overlap 和统计 | 哪些 task 最热；compute/DMA/communication 是否重叠；大量小 task 是否被提交开销支配？ |
| 4 | P0 | 最小硬件指标集 | 热点是 compute、memory、latency/resource 还是工作量不足？ |
| 5 | P0 | Marker、capture range、筛选与迭代控制 | 能否只采稳定 step、指定 op/kernel/rank，而不是 profile 整个进程？ |
| 6 | P0 | 稳定机器可读导出、capability query 和采集质量 | 上层工具能否可靠解析；哪些指标未采到；数据是否受丢事件、复用或回放影响？ |
| 7 | P1 | 内存层级、pipeline/stall、occupancy/resource limiter | 为什么 task 内部没有达到峰值，具体卡在哪个硬件单元或等待原因？ |
| 8 | P1 | Framework -> graph/IR -> binary -> instruction/source 映射 | 应该修改哪段源代码、IR、tiling 或调度？ |
| 9 | P1 | Roofline、profile diff、动态内存与分布式深度分析 | 优化空间多大；新版本改善/回退了什么；通信和内存是否成为瓶颈？多卡训练的基础通信 trace 应前移到 P0 |
| 10 | P2 | 自动诊断、能效、集群规模化和在线分析 | 能否稳定地产生有证据的建议，并服务大规模持续优化？ |

最关键的产品判断是：先让 trace 的身份链、时间线和导出正确，再增加深度 counter。没有可靠 correlation 的硬件指标只能告诉用户“某时刻 NPU 利用率不高”，无法告诉他应该改哪个 op；没有 capture/filter 的全量 counter 则会带来高开销和不可用的数据量。

## 建议的分析工作流

成熟 profiler 普遍采用分层工作流：

```text
Trace / Overview
  -> 找到关键 step、空洞和热点 device task
  -> 对少量热点运行 targeted counter profile
  -> 定位 memory / compute / stall / resource 原因
  -> 映射回 framework op、IR、source 或 instruction
  -> 修改后用非 profiler benchmark 验证，再做 profile diff
```

因此建议产品至少提供四种模式，而不是一个 `full=true`：

| 模式 | 采集内容 | 预期成本 | 典型用途 |
| --- | --- | --- | --- |
| `trace` | host/runtime/device task、memory copy、marker、communication 时间线 | 低 | 日常第一步、生产问题定位 |
| `triage` | trace + 少量可单 pass/sampling 的核心硬件指标 | 低到中 | 自动判断 compute/memory/idle/launch-bound |
| `kernel_deep_dive` | 对指定 task/kernel/iteration 收集详细 counter，允许 replay/multi-pass | 高但受控 | kernel 和算子微架构调优 |
| `memory` / `distributed` | 动态内存或跨卡通信的专项数据 | 按需 | OOM/fragmentation、collective/straggler/overlap |

P0 的成功标准不是“能导出很多字段”，而是 `trace + triage` 已经能稳定缩小问题范围，并精确选出需要 deep dive 的少量 task。

### 主流工具能力对照

| 工具 | 第一层定位 | 深度分析 | 对 NPU 需求的主要参考 |
| --- | --- | --- | --- |
| NVIDIA Nsight Systems + Compute | Host/API/kernel/memory/NVTX 时间线与统计 | 单 kernel launch、occupancy、compute/memory throughput、warp stall、source、roofline、rules | 系统 trace 与 kernel counter 分层；先选 hotspot 再深挖 |
| AMD ROCprofiler-SDK + Compute Profiler | Runtime API、marker、kernel dispatch、memory、queue/correlation trace | 可查询的 basic/derived counters、multi-pass、memory chart、roofline、baseline diff | capability query、开放输出格式、按 dispatch/kernel/device 筛选 |
| OpenXLA XProf | Framework/HLO/module/host/device/step 统一语义视图 | op/kernel stats、roofline、静态/动态 memory、communication、较新 TPU 的细粒度 counter | 编译器 sideband、framework-to-device correlation 和模型级分析 |
| Ascend msprof / MindStudio Insight | Python/CANN/runtime/NPU/communication/overlap 时间线 | AI Core pipeline/arithmetic/memory/cache/resource conflict、source/instruction、operator details | 比较接近自研 NPU 所需的 engine、片上 memory、通信和指令视角 |
| Intel VTune XPU/GPU analysis | Host-to-XPU task correlation、NPU DDR/NOC bandwidth、top tasks | GPU 侧 occupancy、stall、memory hierarchy、source/basic-block 分析 | 统一 XPU offload 视角，以及 NPU 最小 trace+bandwidth 能力边界 |

共同点不是具体 metric 名，而是五层数据同时存在：统一 trace、稳定 identity/correlation、可筛选硬件 counter、compiler/source metadata、机器可读离线结果。

## P0：第一版必须支持

### P0-1 统一时间轴和全链路 correlation

这是最高优先级。每个 device task 必须尽可能关联到发起它的 host/runtime 活动，并进一步关联到编译器和 framework 语义。

推荐的身份链：

```text
user range / training step
  -> framework op / Python stack
  -> graph or compiler IR op / fusion group
  -> executable or module
  -> runtime API / command / dispatch
  -> queue or stream
  -> device task / kernel
  -> matrix, vector, scalar, DMA, network or control engine activity
```

每个 trace event 至少返回：

| 字段 | 要求 |
| --- | --- |
| Identity | `event_id`、`parent_id`、`correlation_id`、`flow_id`；ID 在一次 profile 内唯一且稳定 |
| Scope | host/process/thread、rank、device、die/core/engine、queue/stream |
| Semantic identity | framework op ID/name、graph/IR op ID、fusion ID、executable/module ID、kernel/task ID、binary hash；不可用时返回 missing reason |
| Time | 原始 clock domain、统一后的 `start_ns/end_ns/duration_ns`、时钟换算参数和同步误差 |
| Category | framework、compiler、runtime API、queue、compute、DMA、communication、allocation、sync、marker 等 |
| Arguments | shape、dtype、layout、bytes、direction、collective type、launch/resource 参数等按 category 扩展 |
| Provenance | `measured/derived/static/heuristic`；不能把编译器静态估算伪装成运行时实测 |

OpenXLA XProf 的时间线已经把 framework op、XLA op、module、host offload、scalar unit 和同步活动放到同一个 viewer，并明确标注哪些 track 是实测、哪些依赖 compiler sideband 或 heuristic。[XProf Trace Viewer](https://openxla.org/xprof/trace_viewer) Ascend MindStudio 也提供 Python -> CANN/Runtime -> NPU task 的分层时间线与 host-device 关联。[MindStudio Timeline](https://www.hiascend.com/document/detail/en/mindstudio/830/practicalcases/GeneralPerformanceIssue/toolsample6_022.html) 这说明语义 correlation 不是 UI 附加项，而是异构 profiler 的基础数据模型。

验收时应准备一个包含 framework fusion、多个 runtime dispatch、DMA、两类 compute engine 和显式同步的合成 workload，从用户 marker 能逐层跳转到全部 device task；不能只靠名称字符串猜关联。

### P0-2 Device task 和 engine timeline

时间线至少覆盖：

- host API、runtime/driver API、command submission、queue wait、device execution 和 completion；
- NPU compute task，按实际架构暴露 matrix/cube、vector、scalar、control 等 engine；
- H2D、D2H、D2D、片上/片外 DMA 或等价搬运任务；
- collective、point-to-point、片间/卡间 communication；
- barrier、event、stream/queue synchronize、依赖等待；
- user marker、step/iteration、request/batch/token 等业务 range；
- clock/frequency 和必要的 device-wide 状态。

每个 device task 返回：名称和稳定 ID、开始/结束/时长、device/core/engine、queue/stream、前驱/后继依赖、提交时间、queue delay、launch/resource 参数和 correlation IDs。

engine 名称必须同时提供架构原始名和稳定的通用类别。例如内部硬件叫法可以变化，但上层仍能识别 `matrix/vector/scalar/dma/network/control`。不要只返回一个全局 `NPU Utilization`；它无法区分 matrix engine 饱和但 vector/DMA 空闲，或 device 因 host 提交不足而整体空闲。

NVIDIA Nsight Systems、AMD `rocprofv3 --runtime-trace`、Intel VTune XPU Offload 和 Ascend MindStudio 都把 runtime API、kernel/task、memory operation、marker 和 host-device 关联作为第一层能力。[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)，[ROCprofiler-SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)，[Intel XPU Offload](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-1/xpu-offload-view.html)

### P0-3 Top task、关键路径、overlap 和空洞分析

仅有可视化时间线仍会让用户手工量图。第一版就应提供机器可读聚合：

| 分析 | 最小返回信息 |
| --- | --- |
| Top task | task/kernel-op pair、occurrences、total/mean/median/p95/min/max/std duration、占 device busy sum 和 step wall 的不同分母比例 |
| Step/request breakdown | host compute、compile、launch/submit、queue wait、device compute、DMA、communication、sync、idle |
| Critical path | 关键路径 event 序列、每段时长、依赖边、不可重叠原因 |
| Overlap | compute-compute、compute-DMA、compute-communication 的 union/overlap 时长和比例 |
| Idle gaps | device/core/engine 空闲区间、gap 前后 task、是否由 host、queue、dependency、communication 或未知原因造成 |
| Small-task analysis | 小于配置阈值的 task 数、总时长、提交/间隙开销、潜在 fusion 组 |
| Load balance | core/engine 间工作量、active time、最大/最小/P95 差异 |

必须明确统计分母和并发语义：`sum(task_duration)` 在并发时可能大于 wall time，不能把它误称为 utilization；应同时返回 `busy_union`、`duration_sum` 和 `range_wall`。

XProf Overview 把 step 拆成 input、compile、kernel launch、host compute、device compute、collective communication 等类别；GPU Kernel Stats 按 kernel-framework op pair 返回次数和总/平均/最小/最大时长。[XProf Overview](https://openxla.org/xprof/overview_page)，[XProf GPU Kernel Stats](https://openxla.org/xprof/gpu_kernel_stats) 这类聚合应是导出数据的一部分，而不是只存在于 GUI。

### P0-4 最小硬件指标集

P0 不要求一次覆盖全部 PMU counter，但必须提供一套跨架构尽量稳定的 `triage` 指标，让用户能把热点初步分为 compute-bound、memory-bound、latency/resource-bound、workload/launch-bound 或 mixed。

每个被选中 task/kernel 至少返回：

| 类别 | P0 指标 |
| --- | --- |
| Basic | elapsed cycles/time、有效 core 数、frequency、dispatch/launch dimensions、task 工作量 |
| Engine utilization | matrix/vector/scalar/load-store/DMA active cycles 或 `% of peak/available cycles`；必须说明 denominator |
| Compute | 主要 dtype 的 operation/instruction count 或 achieved throughput、理论峰值和 `% of peak`；区分 matrix 与 vector/scalar |
| Memory | HBM/DDR read/write bytes、time、bandwidth 和 `% of peak`；有低成本 L2/片上 memory 指标时一并返回 |
| Residency | active tasks/waves/warps/blocks 的等价指标、theoretical/achieved occupancy 或并行驻留度 |
| Resource usage | register/accumulator/local or scratch/on-chip shared buffer、barrier/slot 等资源用量和主要 limiter |
| Scheduler health | eligible/issued/active work 的等价核心指标，或至少返回 engine starvation / no-ready-task 比例 |

不能只给派生结论而没有 raw counter，也不能只给硬件 raw name 而没有稳定语义。建议同时返回：

```json
{
  "normalized_name": "matrix_engine_active_pct",
  "value": 73.2,
  "unit": "percent_of_available_cycles",
  "scope": "device_task",
  "raw_counter": "ARCH_XYZ_MX_ACTIVE_CYCLES",
  "aggregation": "sum_over_cores_div_available_cycles",
  "collection_mode": "exact",
  "pass_id": 0,
  "availability": "collected"
}
```

AMD ROCprofiler 明确区分 basic counter 和 derived metric，要求先查询目标 GPU 可用 counter，并在 counter 超出单 pass 容量时使用 multi-pass；其输出保留 dispatch、queue、timestamps、counter name/value。[ROCprofiler counter collection](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) NPU profiler 也需要从第一版就暴露 capability 和 collection mode，不能让上层假设所有芯片都有相同指标。

### P0-5 Capture、marker、filter 和 repeat 控制

必须支持：

- programmatic start/stop、context manager 或等价 API；
- user marker/range，支持嵌套和 metadata；
- delay、duration、warmup skip、iteration/step range、launch skip/count；
- framework op、graph/IR op、kernel/task exact/regex、process、rank、device、queue/stream、engine filter；
- 只采 trace、只采某 counter set、或 trace 后自动选择 top-K task；
- 多进程和 child process 的明确支持边界；
- buffer 大小、flush、最大文件和 dropped-event 行为可配置。

这是可用性和开销控制的关键。NCU、ROCprofiler 和 XProf 都支持按 kernel、iteration、range 或 programmatic window 缩小采集范围；缺少这些能力时，深度 profile 很容易因数据量和 replay 成本无法用于真实模型。

### P0-6 稳定导出、capability query 和采集质量

第一版必须同时提供 CLI/SDK 和机器可读结果。GUI 可以后置，数据协议不能后置。

推荐输出：

| Artifact | 要求 |
| --- | --- |
| Native raw report | 保留最完整原始数据，用于同版本工具复查 |
| Trace | Perfetto-compatible trace 或文档化等价格式；另提供 JSON/DB 便于程序查询 |
| Summary | CSV/JSON/DB，包含 top task、step breakdown、overlap、idle、communication 等 |
| Counters | 长表格式：task/dispatch identity + raw/normalized metric + value/unit/scope/aggregation/pass |
| Metadata | resolved config、tool/driver/firmware/compiler/runtime versions、device/topology、clock、capture range、quality |
| Capability | 当前设备支持的 trace domains、counter、counter group、单位、scope、约束、互斥和最大单 pass 组合 |

AMD ROCprofiler 当前能输出 SQLite、CSV、JSON、Perfetto 和 OTF2，并导出 resolved configuration；Nsight Systems 支持 native report、SQLite 和 CSV/JSON stats；Ascend `msprof` 支持 timeline JSON、summary CSV 和 DB。[ROCprofiler output](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofv3.html)，[Nsight Systems Analysis](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)，[Ascend profile export](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/profiling/atlasprofiling_16_0026.html) 内部工具至少应达到“无需 GUI、无需解析 stdout 就能完成离线分析”。

每次 profile 必须返回采集质量：

- `complete/partial/failed` 和分 domain 状态；
- dropped event、buffer overflow、truncated data、clock sync error；
- counter pass 数、replay/multiplex/sampling 配置和采样覆盖率；
- profiler wall overhead、是否调整 clock/cache、是否检测到其他 workload；
- unsupported/missing counter 的明确原因，不能用 `0` 代替；
- tool error code、phase、message、retryable 和 fallback。

## P1：支持严肃 kernel/算子调优

### P1-1 内存层级和数据搬运分析

应按实际 NPU memory hierarchy 返回数据，而不是只给 HBM bandwidth：

- HBM/DDR、L2/LLC、片上 SRAM/UB/VMEM/SMEM/L1/L0、register/local/scratch 等层级；
- 每层 read/write bytes、requests/transactions、time、bandwidth、peak percentage、hit/miss/eviction；
- request size、sectors/transactions per request、合并/对齐效率；
- bank conflict、port/queue conflict、replay、local spill；
- engine 间数据流向，例如 DMA -> SRAM、SRAM -> matrix/vector；
- H2D/D2H/D2D/片间传输及与 compute overlap；
- memory chart：逻辑数据流与物理存储层级关联。

Ascend AI Core metrics 已覆盖 pipeline/arithmetic utilization、UB/L1/L2/main memory、L0、resource conflict 和 L2 cache；MindStudio Memory Chart 把带宽和数据流映射到硬件层级。[Ascend AI Core metrics](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/ascendgraphapi/atlasgeapi_07_0149.html)，[MindStudio Memory Chart](https://www.hiascend.com/document/detail/en/mindstudio/600/msug/msug_000222.html) AMD Compute Profiler 和 Intel VTune 也都提供 memory hierarchy/chart。这说明片上 memory 可见性是算子 profiler 的核心能力，不是高级附加项。

### P1-2 Pipeline、stall 和 resource limiter

需要一套互斥或至少可解释重叠关系的 stall taxonomy。建议至少覆盖：

- no ready work / scheduler starvation；
- input/data dependency、long/short memory wait；
- execution pipeline busy；
- barrier/synchronization；
- queue/scoreboard/full buffer backpressure；
- instruction fetch/decode；
- bank/port/resource conflict；
- output/writeback/network wait；
- branch/control-flow divergence（架构适用时）。

每类返回 cycles、占 eligible/active/elapsed 的哪一种比例、scope、是否为 sampled counter。诊断规则只有在 scheduler 无法 issue 或 engine 没吃满时才应把 stall 提升为瓶颈；“某 stall 比例高”本身不等于优化收益大。

同时返回 occupancy/residency 的 resource limiters：register/accumulator、片上 memory、barrier、thread/wave/task slot、block/core mapping、tile shape 和并行度。用户需要知道的不只是“occupancy 低”，而是被什么限制以及调整哪个参数可能释放驻留度。

### P1-3 Framework、compiler IR、binary、instruction 和 source 映射

对自研 NPU，这一项通常比再增加几十个 counter 更能提升调优效率。

至少支持：

- framework op、shape/dtype/layout、Python/C++ source/stack；
- graph op、fusion group、前后 graph 邻居；
- 编译器 lowering 后的 IR op、tiling、schedule、pipeline stage；
- executable/module、kernel symbol、binary hash、compile flags；
- device task 与 instruction bundle/PC range；
- source/IR/instruction heatmap：cycles、stall、traffic、instruction count；
- 静态估算和运行时实测明确分开。

OpenXLA XProf 可以从热点 HLO op 跳到 graph 和用户 source，并把静态 FLOPs/bytes 与实测时间组合；Ascend MindStudio 提供 operator source、instruction heatmap 和 workload detail；Intel VTune 支持 source/basic-block memory latency 分析。[XProf Graph Viewer](https://openxla.org/xprof/graph_viewer)，[MindStudio Insight](https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0002.html)，[Intel GPU analysis](https://www.intel.com/content/www/us/en/developer/articles/technical/optimize-applications-for-intel-gpus-with-intel-vtune-profiler.html)

P0 需要建立稳定 ID 和最小 source/IR attribution；P1 再完成 instruction-level heatmap。若编译器没有 sideband metadata，profiler 团队应把这件事列为跨团队接口需求，而不是在 UI 中用 kernel name 猜测。

### P1-4 Roofline、峰值标定和 profile diff

Roofline 至少支持 program、step、op/task 三个 scope，并返回：

- operations、bytes 和各自来源是 hardware counter 还是 compiler static estimate；
- arithmetic/operational intensity；
- achieved compute throughput 和各 memory level bandwidth；
- calibrated/theoretical peak、ridge point；
- compute-bound、HBM/L2/on-chip-memory-bound 或 underutilized 分类；
- 到对应 roof 的 headroom。

OpenXLA XProf 和 AMD Compute Profiler 都提供 per-op/per-kernel roofline，并区分不同 memory level；AMD 还提供 baseline comparison。[XProf Roofline](https://openxla.org/xprof/roofline_model)，[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/what-is-rocprof-compute.html)

Profile diff 应能比较同一 task 的两个版本，检查 device、输入、binary、tool/counter set、clock 等可比条件，返回 duration、engine utilization、memory traffic、occupancy、stall 和 findings 的 delta。最终性能仍应由不带 deep profiler 的 benchmark 验证，因为 counter replay/multiplex 会扰动时序。

### P1-5 动态内存、OOM 和 fragmentation

需要同时支持编译器静态 memory plan 和 runtime 动态 memory profile：

- allocation/free 时间线、地址空间、size、alignment、memory type、allocator；
- current/peak/lifetime peak、reserved/committed/free；
- fragmentation 和最大连续空闲块；
- buffer lifetime、alias/reuse、padding overhead；
- allocation 关联 framework/graph/IR op、source stack；
- OOM 前后的 top buffers 和未释放对象。

XProf 区分静态 Memory Viewer 和动态 Memory Profile，并提供 peak allocation、allocation/deallocation、fragmentation、op/shape/dtype attribution。[XProf Memory Profile](https://openxla.org/xprof/memory_profile)，[XProf Memory Viewer](https://openxla.org/xprof/memory_viewer) 对 NPU 编译器管理大块静态内存的场景，这种静态+动态双视角同样必要。

### P1-6 分布式通信：训练场景应前移到 P0

如果该 NPU 主要服务大模型多卡训练，下面的基础项应直接列为 P0，而不是等待单卡 profiler 完成：

- rank、process、device、communication group、collective ID；
- collective type、algorithm/protocol、message bytes、start/end/duration；
- link/plane/topology、理论/实测 bandwidth；
- enqueue、等待、传输、completion 分段；
- compute-communication overlap、exposed communication time；
- rank 间同一 collective 的 P50/P95/max 和 straggler；
- 跨 host/device/rank 时钟同步误差和 correlation；
- 通信失败、重试、拥塞、链路降级。

P1 再增加拓扑热图、关键 collective、root-cause 和自动建议。XProf Megascale Stats、Ascend communication/overlap timeline、Intel 多 GPU topology 都提供了类似维度。[XProf Megascale Stats](https://openxla.org/xprof/megascale_stats)，[MindStudio Timeline](https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0034.html)，[Intel GPU Offload](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-1/gpu-offload-analysis.html)

## P2：增强能力

### P2-1 有证据的自动诊断和建议

自动诊断输出应是结构化 finding，而不是一段无法验证的自然语言：

```json
{
  "finding_id": "memory.hbm_bandwidth_bound",
  "origin": "rule_set_v3",
  "scope": {"task_id": "task-42"},
  "severity": "high",
  "confidence": 0.92,
  "evidence": [
    {"metric": "hbm_bandwidth_pct_of_peak", "value": 91.3},
    {"metric": "matrix_engine_active_pct", "value": 34.8}
  ],
  "diagnosis": "The task is limited by HBM bandwidth rather than matrix compute.",
  "recommendations": [
    {"action": "Increase on-chip reuse or fuse the producer.", "rationale": "This reduces HBM bytes per output."}
  ]
}
```

需要区分 `hardware_fact`、`derived_metric`、`rule_inference` 和 `llm_inference`，记录规则/model/prompt version。KernelPro 把 NCU/NSYS/SASS 证据转成 severity、root cause 和排序建议，其实验显示结构化 micro-tools 比直接给模型 raw counters 更有效；KEET 也把 NCU profile、source 和运行配置聚合成可审查报告。[KernelPro](https://arxiv.org/html/2606.26453v2)，[KEET](https://arxiv.org/html/2605.04467v1) 这类语义层有价值，但依赖 P0/P1 数据可靠，不能反过来替代底层数据。

### P2-2 Power、energy 和 thermal

支持 device/rail/engine 可行粒度的 power、energy、temperature、frequency、throttling timeline，并返回 sampling frequency、integration window、归因范围和误差。微秒级 task 若采样覆盖不足，只能返回 device-wide sample，不能伪造 per-task energy。能效比较必须同时约束性能和正确性。

### P2-3 大规模、在线和持续分析

- 百卡/千卡 trace 的分片、索引、按需加载和 DB 查询；
- remote capture、attach/detach、ring buffer 和异常触发保留；
- 在线低频 PMU sampling 与离线 deep dive 结合；
- profile 数据配额、压缩、TTL、脱敏和权限；
- CI regression：自动比较关键 workload 的 profile signature；
- 可插拔 parser/rule SDK，允许算子团队定义自有 counter group 和检查器。

## 统一返回数据契约

无论底层架构如何，建议至少稳定以下四类 record。UI、CLI 和上层自动分析都从这些记录派生。

### 1. `TraceEvent`

```text
event_id, parent_id, correlation_id, flow_id
domain, category, name
process_id, thread_id, rank_id
device_id, core_id, engine_type, engine_raw_name, queue_id, stream_id
start_ns, end_ns, duration_ns, original_clock_domain
framework_op_id, graph_op_id, ir_op_id, executable_id, kernel_id
provenance(measured/derived/static/heuristic)
args(shape/dtype/layout/bytes/direction/launch/collective/...)
```

### 2. `TaskProfile`

```text
task/kernel identity and binary hash
launch dimensions, tiling and resource usage
duration/cycles/frequency
engine utilization and compute throughput
memory hierarchy traffic/bandwidth/cache
occupancy/residency and resource limiters
scheduler/pipeline/stall breakdown
source/IR/instruction references
raw counter references and findings
```

### 3. `MetricValue`

```text
raw_name, normalized_name, value, unit
scope(device/core/engine/task/instruction)
aggregation and denominator
timestamp or measurement window
collection_mode(exact/sample/multiplex/replay/derived/static)
pass_id, sampling_frequency, coverage
availability and missing_reason
```

### 4. `ProfileMetadata`

```text
profile ID and resolved collection config
tool/runtime/driver/firmware/compiler versions
device SKU/revision/topology/peak specs
clock/power/cache policy
input/step/range identity
counter capability and selected set
clock synchronization/calibration
overhead, replay/pass count, dropped/truncated data
artifact manifest and errors/warnings
```

字段名称可以内部协商，但语义、scope、单位、分母、采集方式和 missing reason 不应省略。

## P0 验收建议

### 功能 canary

| Workload | 必须观察到的证据 |
| --- | --- |
| 大量 1–5 us 小 task | host submit、queue、device task 的完整 correlation；small-task 和 gap 聚合 |
| 单个 matrix-heavy task | matrix engine active、主要 dtype throughput、资源配置和 task duration |
| 单个 bandwidth-heavy vector/copy task | HBM read/write bytes 和 bandwidth，compute 利用率较低 |
| DMA 与 compute 双缓冲 | 两类 engine timeline、overlap union 和依赖边正确 |
| 显式 barrier/sync | host/runtime/device wait 的因果链与等待时长 |
| 多 stream/queue 并发 | duration sum 与 busy union 正确区分，最大并发度正确 |
| framework fusion | 一个 framework/graph fusion op 能关联多个或一个 device task，关联不靠名字猜测 |
| unsupported counter | 返回 `availability=unsupported` 和原因，不返回 0 |
| buffer overflow | profile 标记 partial、给出 dropped count；不能静默输出不完整时间线 |
| 多进程/多 rank（若支持） | ID 不冲突，跨进程/卡时钟误差可见，collective 能对齐 |

### 质量和开销目标

以下是建议目标，可根据硬件能力修订，但必须发布测量方法和实际结果：

- trace-only 对代表性、持续至少 1 秒的 workload，端到端 overhead 中位数目标 `<=3%`；短 task 另行报告，不用平均数掩盖；
- 在声明的 event-rate/buffer 配置范围内 dropped event 为 0；超出范围必须明确报错或 partial；
- device duration 与硬件基准 timer 在 10 us 以上 task 上误差目标 `<=2%`，同时发布 timestamp resolution；
- host-device clock synchronization error 可查询；多卡场景发布跨卡误差；
- deep counter 模式不设虚假的低 overhead 目标，但必须返回 pass/replay/multiplex 次数、被 profile 的 task 数和总 wall cost；
- profiler 采集失败不能导致目标进程、driver 或其他 device context 残留异常；
- native report 能离线重放/解析，稳定 schema 至少跨一个工具小版本兼容。

XProf 公布的典型采集开销是 TPU `<1%`、GPU `<5%`，可作为 trace 层的行业参考，而不是直接照搬的验收数字。[XProf](https://openxla.org/xprof)

## 不应接受的“已支持 profiling”定义

以下任一情况都不足以称为可用的 NPU profiler：

- 只有 GUI，没有稳定 JSON/CSV/DB 和解析文档；
- 只有 framework op aggregate，没有 device task timeline 和 correlation；
- 只有 device-wide utilization，不能关联到 task/kernel/engine；
- 只有 kernel duration，没有 queue delay、DMA、sync 和 dependency；
- 返回数百个 raw counter，但没有 unit、scope、denominator、aggregation 和 availability；
- 不支持 marker/filter，只能采整个训练进程；
- unsupported/missing 指标用 0 填充；
- 混合 static estimate、sample 和 exact counter，却不标 provenance；
- replay/multi-pass 改变了执行时序，却把 profile duration 当 benchmark；
- framework/source/IR 无法映射到 binary/device task；
- trace 丢事件、clock 不同步或采集被截断时仍报告 success；
- 自动建议没有 evidence path，用户无法复核。

## 建议交付拆分

### Milestone 0：数据底座

- trace event/correlation ID/clock sync 数据模型；
- marker 和 capture/filter API；
- host/runtime/device/DMA 基础 trace；
- native + Perfetto/JSON/DB 导出；
- capability query、resolved config、quality/error metadata；
- 公开字段说明和最小 parser SDK。

### Milestone 1：可用的 `trace + triage`

- framework/compiler sideband ID；
- top task、step breakdown、overlap、idle/queue-gap；
- matrix/vector/scalar/DMA、HBM、occupancy/resource 的最小 counter set；
- targeted task selection 和多 pass 元数据；
- 若产品以多卡训练为主，加入 collective 基础 trace、rank 对齐和 compute-communication overlap；
- P0 canary 与 overhead 报告。

### Milestone 2：kernel deep dive

- memory hierarchy/chart、pipeline/stall、resource limiter；
- source/IR/instruction heatmap；
- roofline、baseline diff；
- static/dynamic memory profile；
- 分布式通信的拓扑、straggler、拥塞与 root-cause 深度分析。

### Milestone 3：自动化和规模化

- 结构化 findings/rules；
- power/energy；
- cluster-scale DB、remote/live capture；
- CI regression 和可插拔分析 SDK。

## 给 NPU 团队评审时建议确认的问题

1. 当前硬件有哪些 timestamp source，host/device/多卡如何同步，误差是多少？
2. 哪些 task/engine 活动可 trace，是否都能获得稳定 correlation ID？
3. P0 最小 counter 哪些能单 pass，哪些需要 replay/multiplex，是否会改变执行状态？
4. framework/runtime/compiler 各自能提供哪些 sideband metadata，谁维护 ID 生命周期？
5. 能否按 marker、iteration、op/kernel、device/rank 精确过滤？
6. 对架构版本差异，如何 query capability、表达 unsupported 和维护 normalized metric？
7. 多进程、多卡、collective 和跨机 clock sync 的支持边界是什么？
8. 原始数据格式是否文档化、可离线解析、可跨版本兼容？
9. trace event-rate、buffer、文件大小、overhead 和丢事件上限是多少？
10. profiler 故障如何隔离，是否可能影响目标进程或 device 状态？

## 相关工具与资料

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NVIDIA Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [AMD ROCprofiler-SDK / rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [AMD ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/what-is-rocprof-compute.html)
- [OpenXLA XProf](https://openxla.org/xprof)
- [Ascend MindStudio Insight](https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0002.html)
- [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0033.html)
- [Intel VTune XPU Offload](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-1/xpu-offload-view.html)
- [KernelPro](https://arxiv.org/html/2606.26453v2)
- [KEET](https://arxiv.org/html/2605.04467v1)
