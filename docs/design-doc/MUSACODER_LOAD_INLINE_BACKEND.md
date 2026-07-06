# MusaCoder `load_inline` 后端

## 背景问题

MusaCoder 产出的 kernel 是 **stock-KernelBench 格式**：一个自包含的 Python 模块——
若干 import、一个或多个 `torch.utils.cpp_extension.load_inline(...)` 调用（即时编译
自定义 CUDA），以及一个 `class ModelNew(nn.Module)`，其 `forward` 调用编译出来的扩展。
模型的原始回复把这个模块包在推理（`<think>...</think>`）、一个 ```` ```python ````
代码块、有时还有结尾的 chat 特殊 token（如 `<|im_end|>`）里。

KernelGym 的自动后端识别（`detect_kernel_backend`）此前只认 TVM-FFI 标记，其余一律
默认到 **`cuda_agent`**——而 `cuda_agent` 的 parser 期望三段式
`### CUDA_KERNELS` / `### APPLY_BINDINGS` / `### MODEL_NEW` 加 `binding_registry.h` /
`REGISTER_BINDING(...)` 的 pybind 脚手架。MusaCoder 的提交完全没有这些，于是每个样本都
在解析/precheck 阶段失败，从未被真正评测。

## 关键认识

评测机器本身是 **后端无关的，而且已经支持自包含模块**：

- `kernelgym/toolkit/kernelbench/loading.py::load_custom_model` 会在代码前面插入
  `os.environ['TORCH_EXTENSIONS_DIR'] = <每任务 build_dir>` 然后 `exec` 这个模块。
  exec 时会执行顶层的 `load_inline(...)` → 即时把 CUDA 编译进每任务的 build 目录 →
  定义出 `ModelNew`。这正是 `KernelBenchCudaBackend` 已经在用的路径。
- correctness 通过 **在同一个 `set_seed(seed_num)` 下分别构造** reference `Model` 和
  `ModelNew` 来对齐权重（`pipeline.py` 在 ~688 行构造 reference、~874 行构造 custom
  model；`correctness.py` 在每个 trial 前再 reseed）。因此被保留的相同 `nn.Module`
  子模块在同一 seed 下会初始化出**完全相同的权重**，不需要 state_dict 传递。（这也正是
  prompt 要求 `ModelNew.__init__` 保留原始子模块的原因。）
- timing / speedup / reference 缓存都基于 model 实例和源码，与 kernel 怎么编译出来无关。

所以 MusaCoder 唯一的特殊工作是：**从原始回复里恢复出干净模块**，并**把它路由到现有的
CUDA 评测路径**。

## 适配方案

三处小的、纯增量的改动（不改动任何现有后端的行为）：

1. **`kernelgym/toolkit/kernelbench/binding_detection.py`**
   - 新增 `extract_model_code(text, entry_point="ModelNew")`：剥掉推理区（保留最后一个
     `</think>` 之后的文本）；在 ```` ```python ```` 代码块里取**最后一个**定义了
     `class ModelNew` 的块（兜底：最后一个含 `load_inline` 的块，再兜底最后一个块，再兜底
     用剥完推理的纯文本）；去掉结尾的 chat 特殊 token。对已经干净的模块是幂等的。
   - `detect_kernel_backend`：现在按优先级匹配标记——TVM-FFI → cuda_agent
     (`### CUDA_KERNELS`/`### APPLY_BINDINGS`/`### MODEL_NEW`/`binding_registry.h`/
     `REGISTER_BINDING(`/`cuda_extension`) → **`load_inline`（`load_inline` 这个 token）**
     → 默认 `cuda_agent`。现有 cuda_agent/tvm 提交不受影响：它们的标记先被匹配，且它们
     从不出现 `load_inline`。

2. **`kernelgym/backend/kernelbench/load_inline_backend.py`**（新增）
   - `KernelBenchLoadInlineBackend(KernelBenchCudaBackend)`：`compile()` 先对原始提交跑
     `extract_model_code`，再委托给 CUDA 后端（validate → `build_compile_cache` →
     `load_custom_model` exec → load_inline 即时编译进每任务目录）。`load()` 防御性地
     再抽取一次再委托。把 artifact/handle 标为 `backend="load_inline"`。其余
     （`create_model`、`run`、`cleanup`、correctness、timing）全部原样继承。
   - `load()` 还会跑一次静态 decoy 分析（见下），把结论挂到 handle 的 `profiling_hints`
     上，供 pipeline 取用。

3. **`kernelgym/backend/kernelbench/dispatcher.py`**
   - 注册新后端，并把 `load_inline` / `inline` / `inline_extension` 路由过去。

## Decoy 检查（reward-hacking 防御）

`kernelgym/toolkit/kernelbench/load_inline_decoy.py::detect_load_inline_decoy` 是一个
纯静态（AST）检查：当一个提交**用 `load_inline(...)` 编译了扩展，但 `ModelNew` 从未引用
它**（既没引用扩展变量、也没引用绑定到它的 `self` 属性、也没有用到它的模块级 helper）时，
判为 decoy——这种样本里 `forward` 其实退化成了 torch 算子，kernel 编译了却没用上。判定
**保守、力求零误报**：只要在 `load_inline` 赋值语句之外有任何对扩展的真实引用就不判 decoy。

支持的"使用"形态（均不判 decoy）：模块级 `ext = load_inline(...)` 后 `ext.fn(...)`、
`self.ext = ext`/`self.ext = load_inline(...)`（直接 self-attr）后 `self.ext.fn(...)`、
模块级 helper 里引用、以及 `import load_inline as <alias>` 别名。"无 load_inline 调用"
的判定基于 AST（不是裸文本正则），所以注释/字符串里的 `load_inline(` 不会误导它。

整合（`pipeline.py::_run_load_inline_decoy_step`）：只对 `backend == "load_inline"`
且 correctness 通过的样本生效（不正确的样本已被拒绝；只有"正确但作弊"才是有意义的 decoy）。
命中时设 `decoy_kernel=True`、`runtime=-1.0`，并把结论写进 metadata。

Codex review 驱动的修正（见 NO-GO → 修复）：原版有两个会**误判正确样本为 decoy**的假阳性
路径——(1) 直接 `self.ext = load_inline(...)`（无中间变量）未被跟踪；(2) `load_inline` 被
import 别名时漏识别 → 走"无扩展"误判。均已修复并加单测（`test_load_inline_decoy.py` 共
10 例）。这对 avg@8 权威跑很关键：误判 decoy 会把正确样本算成不正确、压低 correct rate。
保守取舍（已知、可接受的假阴性）：扩展只要在死代码/未调用的 helper 里被引用过也会清除 decoy
标记——对非对抗性的 MusaCoder 生成足够，优先保证零假阳性而非抓全所有混淆 decoy。

注意：与 `cuda_agent`/`tvm_ffi` 不同，load_inline 暂不做基于 profiler 的自定义 kernel
**覆盖率**统计（命名 kernel 覆盖率需要后端解析出的 kernel 名）。decoy 用上面的静态检查兜底；
覆盖率指标是后续工作。

## 如何调用

- **显式**：`/evaluate` 请求里传 `backend="load_inline"`。
- **自动**：传 `backend="auto"`（`v1`/`auto` profile 的默认），识别器会把提交归类为
  `load_inline`。

注意：slime 调用方目前显式传 `backend="cuda_agent"`（`KERNEL_BACKEND=cuda_agent`），这会
绕过自动识别。要评测 MusaCoder load_inline dump，调用方需改传 `backend="load_inline"`
（或 `auto`）。这是 slime 侧的改动，独立于本后端。

## 已知限制 / 说明

- **覆盖率指标**：见上，load_inline 暂无 profiler 覆盖率（decoy 已用静态检查覆盖）。
- **dtype**：生成的 kernel 通常假设 `float32`（`data_ptr<float>()`）；非 float 的
  reference 输入会在 correctness 阶段以 runtime error 暴露。
- **split compile/execute**：`load_inline` 把 build + dlopen 耦合在模块 `exec` 里。CUDA
  路径的 build 发生在 `compile`（不启动 kernel），所以在 CPU compile worker 上可行；但若在
  无 GPU 的 worker 上 dlopen CUDA 扩展出问题，对 load_inline 任务传
  `split_compile_and_execute=false`。

## 测试 / 证据

`scripts/test_load_inline_musacoder.py` 跑 KernelBench `add` 例子（已知正确的自测）、一个
合成 decoy（编译了 add kernel 但 `forward` 用 `torch.add`），以及来自 slime dump 的真实
MusaCoder-27B 生成（含有权重的转置卷积 problem 57），全部走
`eval_kernel_against_ref(backend="load_inline", detect_decoy_kernel=True)`。

单测：`tests/test_load_inline_detection.py`（识别+抽取，7 例）、
`tests/test_load_inline_decoy.py`（decoy，7 例）。

在 `192.168.16.19`（A800，torch 2.11.0+cu129）的结果：

| case | 题目 | detected | compiled | correct | decoy | 说明 |
|---|---|---|---|---|---|---|
| add 自测 | elementwise add | load_inline | yes | yes | no | 已知正确 |
| add decoy 自测 | elementwise add | load_inline | yes | yes | **yes** | forward 用 torch.add，正确判为 decoy |
| dump idx 0 | 1 square matmul | load_inline | yes | yes | no | 朴素 kernel 慢于 cuBLAS |
| dump idx 216 | 28 HardSigmoid | load_inline | yes | yes | no | |
| dump idx 280 | 36 RMSNorm | load_inline | yes | yes | no | |
| dump idx 448 | 57 conv_transposed_2D（**有权重**） | load_inline | yes | **no** | no | 见下 |

真实生成全部 compiled，且**未被误判为 decoy**；合成 decoy 被正确命中。

conv（有权重）这一例最关键：权重对齐诊断证明评测正确、非 bug——`ModelNew` 保留了
`self.conv_transpose2d = nn.ConvTranspose2d(...)`，在 `set_seed(42)` 下 reference 与
`ModelNew` 的 `conv_transpose2d.weight` **逐位一致**；直接调用被保留的子模块能复现参考
输出（maxdiff `0.0`）；模型自己的自定义转置卷积 kernel 与参考差 maxdiff `5.4e-4`，刚好
超过 KernelBench 的 fp32 容差（`1e-4`），因此 `correctness=False` 是正确判定——是生成的
kernel 精度不够，不是评测缺陷。

## 重打分（re-scoring）流程与新增适配

不走 slime 框架，直接从 slime dump 抽取已生成的回答喂给 KernelGym 评分。两个脚本：

- `scripts/score_one_sample.py`：在**单独进程**里评一个样本（读 reference + 原始 response
  文件，跑 `eval_kernel_against_ref(backend="load_inline", detect_decoy_kernel=True)` +
  `eval_reference_only`，打印结果 JSON）。
- `scripts/rescore_musacoder_dump.py`：编排器。加载 dump（`eval_0.pt`，800 样本，每题 8 个，
  按 problem_id 排序），取前 N 题 × 每题 K 个样本（索引 `(problem_id-1)*8 + k`），**每样本
  起一个子进程** `score_one_sample.py` 评测，聚合 sample 级 / problem 级（problem 只要有一个
  样本 correct 且非 decoy 即记 correct）的 compile/correct/decoy/fast@p，并把每样本的
  reference/response/result 落到 review 目录便于人工检查。

**新增适配（为什么这么设计）**：

- **每样本一子进程隔离**：`load_inline` 按 `name=` 缓存编译产物，同一进程跨样本会出现同名
  扩展冲突（100 题时大量 kernel 重名）。生产服务靠每任务子进程隔离；这里用"每样本一子进程"
  复刻，配合 cuda backend 的每任务 build 目录，彻底避免冲突。
- **GPU 并行**：编排器用线程池 + GPU 池给并发样本分配卡（子进程内 `CUDA_VISIBLE_DEVICES`
  指定，进程内看到的就是 `cuda:0`）。`--workers == num_gpus`（如 8）时每样本独占一张卡，
  数字最干净（权威跑用这个）。**关键瓶颈**：load_inline 每样本要现场 nvcc 编译（~30–90s，
  CPU），这段时间它占的 GPU 是 **0% 空转**的（实测 8 worker 时 8 卡全 0%、8 个 nvcc 在跑）。
  所以可**超额并发**：`--workers 16`（8 卡每卡 2 个样本）让一部分样本编译（CPU）时另一部分
  执行（GPU），吞吐约翻倍。代价：同卡 2 样本可能把最大的题 OOM（假失败）、且 perf/speedup
  更吵——correctness（即 avg@8）不受影响。子进程 + 每任务 build 目录 + GPU 池保证隔离安全。
- **健壮性**：每样本 `--per-sample-timeout` 超时保护；worker 崩溃/非 JSON 输出被隔离记为
  失败，不影响整批；large/OOM 题的子进程错误同样被隔离。

**容差确认（重要）**：KernelGym 的 fp32 correctness 容差是 `atol=rtol=1e-4`，已核对**与
官方 KernelBench 一致**（`src/kernelbench/eval.py` 的
`PRECISION_TOLERANCES = {torch.float32: 1e-4, float16/bfloat16: 1e-2}`）。因此像
problem 4（matvec，K=1048576，相对误差 `1.9e-4`）、problem 57（conv，`5.4e-4`）这类
"差一点点"的失败是**在与 MusaCoder/KernelBench 相同的容差下真实不达标**，不是评测过严，
也不需要改。

**首轮（前 10 题，均为 matmul 族）**：10/10 compiled，9/10 correct，0 decoy；
fast@1.0=0（朴素 kernel 在这些大 shape 上慢于 cuBLAS，符合预期）。唯一失败的 problem 4
经诊断为下文的 fp32 大规约精度近失，判定正确。

**第二轮（前 50 题，含 matmul/激活/norm/pooling/reduction/conv/batchnorm）**：
50/50 compiled，44/50 correct（88%），0 decoy，fast@1.0=8（最高 Softsign 2.98×、
AvgPool1D 1.90×、pooling 多个 >1.3×）。6 个失败逐一人工核对，**全部确认是模型 kernel 自身
问题、非评测 bug**：

| pid | 题目 | 失败类型 | 诊断 |
|---|---|---|---|
| 4 | Matrix-vector (K=1048576) | Output mismatch | fp32 大规约精度近失，rel `1.9e-4` |
| 16 | Matmul transposed A | Output mismatch | 转置处理逻辑错，rel `10%`（gross） |
| 33 | BatchNorm | Output mismatch | kernel 计算完全不对，rel `~100%`（gross） |
| 37 | FrobeniusNorm | CUDA illegal memory access | kernel 越界访问（真实崩溃） |
| 45 | Average Pooling 2D | CUDA illegal memory access | kernel 越界访问（真实崩溃） |
| 50 | conv_standard_2D | Output mismatch | 卷积 kernel 不达标 |

两个 CUDA 越界崩溃验证了"每样本一子进程"隔离的必要性：崩溃被限制在该子进程内，其余 44 个
样本不受影响（若同进程批量评测会被污染）。weighted 题（conv/batchnorm）的 ~10%/~100% 总
误差也排除了"权重未对齐/train-eval 模式"这类评测侧假阴性——是模型 kernel 真的算错。

**第三轮（全 100 题，每题 1 个样本 = 8 个里的第 0 个，temp=0.7）**：
**compile 98/100，correct 60/100，decoy 0，fast@1.0=15，fast@1.2=11**（problem 级同此，
因每题 1 样本）。40 个失败按类型：32 output_mismatch、3 CUDA 越界、1 binding 参数不匹配、
1 invalid launch config、1 invalid arg、1 undefined symbol 链接失败、1 超时
（pid97 attention：单 block 串行做全注意力，O(seq·head_dim)/query，极慢但非死循环，600s
超时正确判失）。全部为模型 kernel 自身问题，无评测 bug。加速赢家合理：GELU 7.21×、
TripletMargin 3.83×、KLDiv 3.22×、Softsign 3.15×、MSE 2.14×、多个 pooling/reduction >1×。

### 双侧人工核对结论（防假阴性 + 防假阳性）

- **失败侧（防假阴性）**：逐一核对，全部是模型 kernel 真错——精度近失（rel 1.9e-4）、
  逻辑错（matmul-T rel 10%、BatchNorm rel ~100%）、CUDA 越界崩溃、binding/launch/链接错、
  pathological 超时。无一是评测误判。
- **正确侧（防假阳性）**：不只看 `return` 行和 allclose，**通读了 4 个"正确"样本的整段
  CUDA kernel**（matmul tiled、MaxReduce、MaxPool1D、AvgPool1D），专门排查 allclose 看不见
  的 latent bug——max/min reduction 是否误用初值 `0`（会被 `torch.rand≥0` 蒙混）、矩阵转置/
  索引错、未初始化内存读、形状特化、池化除数/边界——**均未发现**（MaxReduce/MaxPool 都用
  `-FLT_MAX`，matmul 索引与同步正确）。并独立 GPU 复跑确认：correct 的 weighted conv
  与参考**逐位一致**（maxdiff 0、权重对齐）、GELU 赢家 maxdiff 1.2e-7。

### 待办 / 与论文对比的口径

单样本是 60%（每题 8 个 temp=0.7 样本里的第 0 个）。完整 800 样本的结果见下。

### 第四轮（全量 800 = 100 题 × 8 样本，avg@8）

avg@8 = 全部 800 样本的平均（每题 8 个的均值再对题平均，因每题恰好 8 个 = sample-level 率）。

| 指标 | 值 |
|---|---|
| **avg@8 correct**（490/800） | **61.25%** |
| avg@8 compile（785/800） | 98.12% |
| **best@8 correct**（每题 8 个任一正确，70/100） | **70%** |
| best@8 compile | 100% |
| decoy（全 800） | **0** |
| fast@1.0 / fast@1.2（avg@8） | 13.4% / 9.75% |

每题"8 个里正确几个"的分布（双峰，合理）：`8/8`=39 题、`7/8`=15、`6/8`=5、`5/8`=5、
`4/8`=2、`3/8`=3、`1/8`=1、`0/8`=30。即模型能稳定做对的简单算子（elementwise/激活/池化/
简单规约/部分 matmul）多在 8/8；难题（多数 conv 变体、batchnorm、attention）多在 0/8；中间
是有方差的题。30 个 0/8 的题全部仍能编译（compile≥1），只是 kernel 算错。全 800 样本
**0 decoy**，验证了修复后的 decoy 检测器在规模上无假阳性。

口径说明：容差 = KernelBench 标准 fp32 1e-4；decoy 检测开启（KernelBench 原版无此项，故
"纯 KernelBench correctness" = correct + decoy；本轮 decoy=0，两者一致）。论文若报 greedy
pass@1 用单样本(60%)对比、若报 best@k 用 best@8(70%)对比、若报平均用 avg@8(61.25%)对比。

### 失败分类 + "为什么 correct 偏低"（关键发现）

按题型：activation/elementwise/attn 98.3%、matmul 93.1%、norm 87.5%、reduction 81.2%、
pooling 70.8%、loss 50.0%、**conv 16.4%**。30 个 0/8 全错的题里 **29 个是卷积**。非卷积
correct=444/520=85.4%，卷积=46/280=16.4%——单 conv 一类把整体从 ~85% 拉到 61%。

把 310 个 incorrect 按错误类型细分：output mismatch 257（其余：CUDA 越界 14、binding
签名 11、编译/链接 10、invalid config 7、其它/超时 ~11）。再把 257 个 output mismatch
重跑测 maxdiff 细分：

| 类别 | 数量 | 说明 |
|---|---|---|
| near-miss（≤1e-2，实测 212/214 在 rel<1e-3） | 214 (83%) | kernel 数值上几乎对，只是超了 1e-4 |
| gross（>1e-2） | 25 (10%) | 逻辑真错（matmul-T 10%、BatchNorm 100% 等） |
| shape mismatch | 18 (7%) | 输出形状错，真 bug |

**结论（更正：不是"累加顺序"）**：near-miss 几乎全是 conv（conv mismatch 96% 是
near-miss，rel<1e-3）。根因有两个，都是 **KernelBench 标准评测自带的**（已实测）：

1. **参考实现跑在 TF32**。`torch.backends.cudnn.allow_tf32` 在 Ampere 上默认 True，且
   KernelBench 的 `src/kernelbench/eval.py` 不关它（KernelGym 同样不设），所以参考
   `nn.Conv2d` 实际用 TF32（~10 位尾数）。实测参考 conv 自己的 fp32-vs-fp64 误差：
   TF32 开 = 3.38e-4、关 = 3.65e-7——**参考本身就超 1e-4**。模型的真 fp32 kernel
   （`data_ptr<float>`）和 TF32 参考比必然差 ~3e-4 → conv near-miss 主因。**非 kernel
   逻辑错、非评测 bug，KernelBench 自己也会这样判。**（matmul 的 TF32 在 torch 2.x 默认
   关，所以 matmul 参考是真 fp32。）
2. **朴素大规约**。好算法（cuBLAS/cuDNN 分块/pairwise）换顺序只差 ~1e-7（K=1e6 实测
   2.4e-7，所以"顺序"本身无害）；但模型写的单累加器朴素 fp32 求和误差随 K 增长，实测
   K=1,048,576 时 1.54e-4，对应 matvec(problem 4) 的 1.9e-4。

容差对数字影响巨大：

| | 1e-4（KernelBench 标准，TF32 开） | rel<1e-3 / TF32 关 |
|---|---|---|
| avg@8 correct | 61.25% | 88.0% |
| conv | 16.4% | 90.7% |

**所以 61.25%（1e-4、TF32 开）才是忠实于 KernelBench 标准的数字**——TF32 的 conv 惩罚是
KernelBench 自带的，标准 KernelBench 的论文数字也带同样的伪影，故应与 61.25% 对比；88%
（1e-3 或关 TF32）是"更公平的 fp32 上限"。**不要为了好看去关 TF32**，那会偏离 KernelBench/
论文。与论文对齐前需确认论文的 eval 配置（是否标准 KernelBench：TF32 开 + 1e-4）。证据：
`/tmp/rescore_800.json`、`/tmp/mismatch_subdiv.json`、`/tmp/ts_nearmiss_1e3.json`。
