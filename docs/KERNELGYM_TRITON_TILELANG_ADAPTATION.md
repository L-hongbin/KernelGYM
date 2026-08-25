# KernelGYM Triton 与 TileLang 适配总结

本文记录 2026-08-24 完成的 KernelGYM 适配工作。本次修改主要增加 Triton、TileLang 的统一支持，并补齐与 CUDA 类似的评测、profiling 和分阶段执行链路。

## 功能清单

| 功能模块 | 新增或调整内容 | 主要文件 |
|---|---|---|
| TileLang 后端 | 新增独立的 `KernelBenchTileLangBackend`，支持源码 artifact、加载、模型创建和运行 | `kernelgym/backend/kernelbench/python_dsl_backend.py`、`kernelgym/backend/kernelbench/tilelang_backend.py` |
| Triton 后端重构 | 将原 Triton 实现迁移到共享 Python DSL 后端，减少 Triton 与 TileLang 的重复逻辑 | `kernelgym/backend/kernelbench/triton_backend.py`、`kernelgym/backend/kernelbench/python_dsl_backend.py` |
| 统一 Python DSL artifact | CPU 阶段校验并序列化源码，GPU 阶段加载源码并触发 Triton 或 TileLang JIT | `kernelgym/backend/kernelbench/python_dsl_backend.py` |
| 后端路由 | Dispatcher 可明确区分和路由 `triton`、`tilelang`、`cuda`、`cuda_agent`、`tvm_ffi` | `kernelgym/backend/kernelbench/dispatcher.py` |
| 自动语言识别 | `backend=auto` 时，可通过 import、decorator 等标志识别 Triton、TileLang 和 TVM-FFI | `kernelgym/toolkit/kernelbench/binding_detection.py` |
| API 类型支持 | `Backend` 枚举及 HTTP API 新增 `tilelang` | `kernelgym/common.py`、`kernelgym/server/api/models.py` |
| 编译参数传递 | API、任务模型、workflow、toolkit 和 pipeline 增加 `compiler_options` 字段 | `kernelgym/server/api/models.py`、`kernelgym/schema/task.py`、`kernelgym/workflow/kernelbench_helpers.py`、`kernelgym/toolkit/kernelbench/toolkit.py`、`kernelgym/toolkit/kernelbench/pipeline.py` |
| Split compile/execute | Triton、TileLang 支持 CPU compile 阶段生成源码 artifact，再交给 GPU worker 执行 JIT | `kernelgym/backend/kernelbench/python_dsl_backend.py`、`kernelgym/toolkit/kernelbench/pipeline.py` |
| Artifact 元数据 | 增加 DSL 类型、JIT 标志、cache key、语言和 profiling hints 等信息 | `kernelgym/backend/kernelbench/python_dsl_backend.py` |
| Profiling 支持 | TileLang 使用提取出的自定义 kernel 名称进行 profiler 匹配和 kernel coverage 统计 | `kernelgym/toolkit/kernelbench/pipeline.py` |
| Decoy 检测 | TileLang 接入具名 kernel coverage 和疑似绕过检测链路 | `kernelgym/toolkit/kernelbench/pipeline.py` |
| GPU 设备环境 | Triton、TileLang 在加载和运行阶段统一处理 CUDA device/runtime 环境 | `kernelgym/backend/kernelbench/base.py` |
| 依赖配置 | 新增 `tilelang==0.1.8`，提供 `dsl` optional dependency，并加入 CUDA 12.9 环境依赖 | `pyproject.toml`、`requirements-cuda129.txt` |
| 单元测试 | 覆盖后端路由、源码 artifact、torch-only 拒绝、自动识别及编译参数贯通 | `tests/kernelbench/backends/test_python_dsl_backends.py` |
| 跨语言 GPU matrix | 新增统一 runner，通过真实 HTTP API 验证 CUDA、Triton、TileLang | `benchmarks/run_language_matrix.py` |
| CUDA fixtures | 新增通过 TVM-FFI 执行的 CUDA 测试用例集合 | `benchmarks/kernels/cuda/` |
| Triton fixtures | 新增覆盖奇数 shape、FP16/BF16、RMSNorm、RoPE、KV gather 等场景的用例 | `benchmarks/kernels/triton/` |
| TileLang fixtures | 新增与 Triton 对应的推理算子、dtype、shape 和非连续输入用例 | `benchmarks/kernels/tilelang/` |
| Negative cases | 增加 torch-only submission，验证 Triton/TileLang 后端能够拒绝未使用目标 DSL 的代码 | `benchmarks/kernels/triton/cases.py`、`benchmarks/kernels/tilelang/cases.py` |
| 文档 | 更新支持的 backend、运行命令、语言矩阵、profiling 和 reward-hacking 说明 | `README.md`、`docs/HTTP_API.md`、`benchmarks/README.md` 等 |

## 验证状态

| 检查项 | 结果 |
|---|---|
| Backend/workflow pytest | 47 passed，1 skipped |
| Triton/TileLang contract tests | 全部通过 |
| CUDA、Torch、CUPTI 启动检查 | 通过 |
| 部署健康检查 | 2/2 GPU ready，24/24 CPU compile workers online |
| 启动 profiling warmup | HTTP 200，编译成功，正确性通过 |
| 完整跨语言 GPU matrix | 尚未在本次 review 中完整执行 |

## 当前边界

- `compiler_options` 已贯通 API、任务和 artifact，但目前主要用于元数据及 cache key，尚未统一应用到各 DSL 的实际 JIT 调用。
- Triton 和 TileLang 的 CPU compile 阶段生成可移交的源码 artifact；真正的设备代码 JIT 在 GPU 执行阶段发生。
- 完整语言矩阵依赖运行中的 KernelGym 服务和真实 GPU，因此与常规 pytest 分开执行。

## 运行跨语言验证

```bash
python benchmarks/run_language_matrix.py \
    --api http://127.0.0.1:20111 \
    --languages cuda,triton,tilelang \
    --concurrency 2
```

验证 CPU compile 到 GPU execute 的 artifact 移交：

```bash
python benchmarks/run_language_matrix.py \
    --languages triton,tilelang \
    --split \
    --include-negative
```

## 功能代码讲解

以下示例省略了异常处理和无关参数，只展示每项功能的核心调用关系。

### 1. TileLang 后端

```python
class KernelBenchTileLangBackend(KernelBenchPythonDslBackend):
    name = "kernelbench.tilelang"
    backend_name = "tilelang"
    dependency = "tilelang"
    accepted_import_roots = frozenset({"tilelang"})
```

TileLang 继承共享的 Python DSL 后端，只声明自己的名称、依赖和允许的 import。源码校验、artifact 生成、加载及执行均复用父类逻辑。

### 2. Triton 后端重构

```python
from .python_dsl_backend import KernelBenchTritonBackend

__all__ = ["KernelBenchTritonBackend"]
```

原来的 Triton 专用实现改为导出共享模块中的实现。这样 Triton 和 TileLang 使用同一套源码 artifact 生命周期，同时仍保留原有导入路径。

### 3. 统一 Python DSL artifact

```python
return {
    "compiled": True,
    "artifact_type": "python_jit_source",
    "jit_compile_on_execute": True,
    "code": code,
    "entry_point": entry_point,
}
```

CPU compile 阶段不生成 GPU 二进制，而是校验源码并生成可序列化 artifact。GPU worker 收到 artifact 后加载源码，第一次调用 kernel 时由 DSL runtime 完成 JIT。

### 4. 后端路由

```python
def _select(self, name):
    backend = self._resolve_backend_name(name)
    if backend == "triton":
        return self._triton
    if backend == "tilelang":
        return self._tilelang
    if backend == "tvm_ffi":
        return self._tvm_ffi
    if backend == "cuda":
        return self._cuda
    raise ValueError(f"Unsupported KernelBench backend '{name}'")
```

请求中的 backend 被解析为具体实现。未知名称直接报错，避免错误地回退到 CUDA 后端。

### 5. 自动语言识别

```python
def detect_kernel_backend(text: str) -> str:
    if TVM_FFI_MARKER_RE.search(text):
        return "tvm_ffi"
    if TILELANG_MARKER_RE.search(text):
        return "tilelang"
    if TRITON_MARKER_RE.search(text):
        return "triton"
    return "cuda_agent"
```

当请求使用 `backend=auto` 时，系统通过源码中的 import、decorator 或绑定标志选择后端。例如 `import tilelang` 会路由到 TileLang。

### 6. API 类型支持

```python
class Backend(str, Enum):
    CUDA = "cuda"
    TRITON = "triton"
    TILELANG = "tilelang"
    TVM_FFI = "tvm_ffi"
    AUTO = "auto"
```

API 模型使用该枚举校验请求。因此客户端可以直接提交 `"backend": "tilelang"`，非法值会在进入执行链路前被拒绝。

### 7. 编译参数传递

```python
result = eval_kernel_against_ref(
    custom_model_src=task.kernel_code,
    backend=task.backend,
    compiler_options=task.compiler_options,
)
```

`compiler_options` 从 HTTP 请求进入任务模型，再依次经过 workflow、toolkit 和 pipeline 到达 backend artifact。目前该字段主要作为元数据和 cache key 的一部分保留。

### 8. Split compile/execute

```python
# CPU worker
artifact = backend.compile(code, backend="tilelang")

# GPU worker
handle = backend.load(artifact, device="cuda:0")
session = backend.open_session(handle, device="cuda:0")
```

Split 模式将源码检查和 artifact 生成安排在 CPU worker，把 artifact 交给 GPU worker加载和运行，从而复用现有的分阶段任务调度机制。

### 9. Artifact 元数据

```python
artifact = {
    "build_backend": "tilelang_jit",
    "compile_artifact_cache_key": cache_key,
    "profiling_hints": {
        "custom_kernel_names": kernel_names,
        "language": "tilelang",
    },
}
```

artifact 除了源码，还携带构建方式、cache key 和 profiler 提示。后续 pipeline 无需重新分析整个请求即可获得关键上下文。

### 10. Profiling 支持

```python
custom_kernel_names = backend_profiling_hints.get("custom_kernel_names", [])
coverage = compute_named_kernel_coverage(
    custom_kernel_names,
    profiling_metrics,
)
```

TileLang 后端从 decorator 下的函数定义提取 kernel 名称。Profiler 完成采样后，pipeline 用这些名称匹配实际 CUDA events 并计算覆盖率。

### 11. Decoy 检测

```python
_apply_coverage_metadata(
    metadata=metadata,
    kernel_exec_result=result,
    coverage_result_dict=coverage,
    coverage_backend="tilelang",
    detect_decoy_kernel=True,
)
```

具名 kernel 覆盖率会写入评测 metadata，并参与疑似 decoy 判断，用于发现提交代码声明了自定义 kernel、实际却主要走其他计算路径的情况。

### 12. GPU 设备环境

```python
device = self._normalize_device("cuda:0")
self._maybe_set_cuda_device(device)
self._maybe_set_triton_env(device)
```

后端在加载和运行模型前统一规范设备参数、设置当前 CUDA device，并准备 Triton/TileLang 所需的 GPU 可见性环境。

### 13. 依赖配置

```toml
[project.optional-dependencies]
dsl = ["tilelang==0.1.8"]
```

TileLang 被放入独立的 DSL 可选依赖，使用者可以按需安装；CUDA 12.9 的部署 requirements 同时固定相同版本，保证节点环境一致。

### 14. 单元测试

```python
def test_dispatcher_routes_triton_and_tilelang_without_cuda_fallback():
    backend = KernelBenchBackend()
    assert isinstance(backend._select("triton"), KernelBenchTritonBackend)
    assert isinstance(backend._select("tilelang"), KernelBenchTileLangBackend)
```

该测试确认两种 DSL 都会进入自己的后端，未知 DSL 也不会静默落到 CUDA，从而覆盖最基础的路由契约。

### 15. 跨语言 GPU matrix

```python
LANGUAGE_MODULES = {
    "cuda": "benchmarks.kernels.cuda.cases",
    "triton": "benchmarks.kernels.triton.cases",
    "tilelang": "benchmarks.kernels.tilelang.cases",
}
```

统一 runner 按语言加载用例模块，通过线程池并发调用真实 HTTP API，最终汇总 compiled、correctness、decoy 和 profiler kernel 数量。

### 16. CUDA fixtures

```python
payload = {
    "kernel_code": cuda_source,
    "backend": "tvm_ffi",
    "precision": "fp32",
}
```

矩阵中的 CUDA 用例走生产使用的 TVM-FFI 绑定路径，用来与 Triton、TileLang 在相同 API 和评测口径下对比。

### 17. Triton fixtures

```python
@triton.jit
def add_kernel(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n
    tl.store(y + offset, tl.load(x + offset, mask=mask) + 1.25, mask=mask)
```

Triton fixtures 包含可真实启动的 kernel，并覆盖 odd shape、不同精度和典型推理算子，而不只是验证源码能否被 import。

### 18. TileLang fixtures

```python
@tilelang.jit(out_idx=[-1], target="cuda")
def make_kernel(n):
    @T.prim_func
    def add(a: T.Tensor((n,), "float32"), b: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for thread in T.Parallel(256):
                index = block * 256 + thread
                if index < n:
                    b[index] = a[index] + T.float32(1.25)
    return add
```

TileLang fixtures 与 Triton 使用相近的输入和算子场景，使 correctness、性能与 profiling 结果可以按语言对照。

### 19. Negative cases

```python
TORCH_ONLY = """
import torch
class ModelNew:
    def forward(self, x):
        return x + 1
"""

artifact = KernelBenchTileLangBackend().compile(TORCH_ONLY)
assert artifact["compiled"] is False
```

仅使用 PyTorch、没有导入目标 DSL runtime 的提交会被拒绝，防止请求声明 `backend=tilelang` 或 `backend=triton`，实际完全绕过对应 DSL。

### 20. 文档更新

```markdown
| `backend` | `auto` | One of `cuda`, `triton`, `tilelang`, `cuda_agent`, `tvm_ffi`, `auto`. |
```

HTTP API、README、benchmark 指南和 reward-hacking 设计文档同步记录新增能力，确保接口说明与实现保持一致。

## Hacking 检测策略

当前 KernelGYM 的 hacking 防护分为静态拦截、正确性防护、运行时行为检测以及性能与覆盖率诊断四层。

### 策略总览

| 层级 | 策略 | 防御目标 | 当前处理 |
|---|---|---|---|
| 静态 | Fallback/bypass 检测 | 使用 `try/except/pass` 隐藏失败或切换备用实现 | 直接拒绝 |
| 静态 | 计时函数篡改检测 | 重写 CUDA Event、`synchronize` 或系统计时函数 | 直接拒绝 |
| 静态 | 线程/进程注入检测 | 后台提前计算或隐藏异步耗时 | 直接拒绝 |
| 静态 | Lazy/Fake Tensor 检测 | 伪造 Tensor 或延迟真正计算 | 直接拒绝 |
| 静态 | PyTorch/ATen compute 检测 | 使用框架算子代替自定义 kernel | 直接拒绝 |
| 静态 | FP32 降精度检测 | FP32 任务内部偷偷使用 FP16 | 直接拒绝 |
| 静态 | 非默认 CUDA stream 检测 | 使用其他 stream 隐藏真实执行时间 | Warning |
| 正确性 | Reference 优先执行 | 防止 candidate 先修改全局 CUDA/PyTorch 状态 | 固定执行顺序 |
| 正确性 | 每轮 RNG 对齐 | 防止利用 reference/candidate 随机状态差异 | 重置相同 seed |
| 正确性 | Reference alias clone | 防止 candidate 原地修改输入后影响 reference 输出 | 检测 alias 后 clone |
| 正确性 | Reference cache poison | 防止未写入的 `empty` tensor 复用 reference 缓存数据 | 零值 scratch 污染缓存块 |
| 正确性 | 多轮输出比较 | 防止只针对单次输入、首次调用或固定状态作弊 | Shape 和数值硬校验 |
| 正确性 | 每轮 ATen profiling | 防止前几轮走 kernel、后续轮次切换 PyTorch fallback | 每个 candidate trial 都检测 |
| 运行时 | ATen allowlist gate | 捕获动态调用的 PyTorch 高层计算 | 非 allowlist 算子标记 decoy |
| 运行时 | Triton usage hook | 声明 Triton 但实际未启动 Triton kernel | 标记 decoy |
| Profiling | 自定义 kernel 名称匹配 | 判断声明的 kernel 是否真实执行 | 记录匹配结果 |
| Profiling | 自定义 kernel 时间覆盖率 | 判断主要计算时间是否绕开自定义 kernel | 低于 30% 标记 suspected decoy |
| Profiling | 极低覆盖率候选 | 自定义 kernel 时间占比接近零 | 低于 0.1% 标记 hard candidate，但不拒绝 |
| Profiling | 空采样重试 | 避免 profiler 偶发漏采样造成误杀 | 自动重试，最终 fail-open |
| Timing | CUDA Event 计时 | 防止 CPU wall time遗漏异步 GPU 执行 | Event 包围 candidate forward |
| Timing | 强制 CUDA synchronize | 防止 kernel 未结束就完成计时或比较 | 各关键阶段同步 GPU |

### 1. 静态代码检查

`validate_kernel_static()` 默认启用以下硬拦截：

```python
DEFAULT_FORBIDDEN_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",
    "lazy_eval",
    "framework_compute",
    "precision_downgrade",
]
```

主要检测：

- `try/except/pass` fallback；
- 修改 `torch.cuda.Event.record`、`elapsed_time` 或 `torch.cuda.synchronize`；
- 修改 `time.perf_counter`、`time.time`；
- 创建线程、进程、线程池或进程池；
- 自定义 Tensor subclass 或构造 Fake/Lazy Tensor；
- 调用 `torch.mm`、`torch.sum`、`at::matmul` 等框架计算；
- FP32 任务中显式使用 half、FP16 或快速 FP16 计算模式。

显式 CUDA stream 操作目前只产生 warning：

```python
DEFAULT_WARNING_CHECKS = ["stream_injection"]
```

这是因为部分合法 wrapper 也可能需要 stream 操作，直接禁止容易造成误杀。

### 2. Reference 执行顺序保护

每轮正确性检查固定先执行可信 reference，再执行 candidate：

```python
output = reference_model(*inputs)
output_new = candidate_model(*inputs)
```

该顺序防止 candidate 在 reference 之前修改 TF32、cuDNN、默认 dtype、默认 device、输入内容等进程级状态。当前两者仍在同一进程，尚未实现完整的全局状态隔离。

### 3. RNG 对齐

每个 trial 生成独立 seed，并在两次 forward 前恢复相同 RNG 状态：

```python
set_seed(trial_seed)
output = reference_model(*inputs)

set_seed(trial_seed)
output_new = candidate_model(*inputs)
```

这样随机算子会从相同状态开始，避免调用顺序导致的随机差异，也减少利用 RNG 状态作弊的空间。

### 4. Reference 输出 alias 防护

如果 reference 输出与输入共享 storage，candidate 原地修改输入可能同时改变已保存的正确答案。当前实现会检测 alias 并 clone：

```python
if _output_aliases_inputs(output, inputs):
    output = _clone_output_on_device(output)
```

### 5. CUDA allocator cache poison

已知的一种攻击是 candidate 返回未写入的 tensor：

```python
def forward(self, x):
    return torch.empty_like(x)
```

如果它恰好复用 reference 已释放的显存，残留数据可能与正确答案相同。当前在 reference forward 后分配同结构零值 tensor，覆盖可能被复用的缓存块：

```python
poison_scratch = _zero_poison_like(output)
torch.cuda.synchronize()
del poison_scratch
```

这是一种实用型防御，但不是严格的 allocator 隔离保证。

### 6. 多轮正确性校验

每轮都会重新生成输入并检查输出 shape 和数值：

```python
if output.shape != output_new.shape:
    return KernelExecResult(correctness=False)

outputs_close = torch.allclose(output, output_new, atol=atol, rtol=rtol)
```

多 trial 可降低只记忆固定输入、只在第一次调用返回正确结果或依赖持久状态的作弊概率。

### 7. 运行时 ATen fallback 检测

静态正则可能被别名、封装或动态调用绕过，因此每个 candidate correctness forward 都会启用 CPU-side ATen profiling：

```python
with aten_operator_profiling_context(True) as profiler:
    output_new = model_new(*inputs)
```

允许 view、allocation、copy、dtype/device 转换等 tensor plumbing 操作。如果捕获到 `aten::mm`、`aten::convolution`、`aten::sum` 等非 allowlist 计算，即使数值正确，也会设置：

```text
decoy_kernel=true
policy_violation=true
policy_violation_reason=DISALLOWED_ATEN_COMPUTE
```

若 profiler 本身初始化或提取失败，则记录 `aten_detection_valid=false`，但不会直接判为 decoy，避免基础设施故障导致误杀。

### 8. Triton 实际使用检测

对声明为 Triton 的提交，系统通过 runtime launch hook 检查是否真的启动了 Triton kernel：

```python
used, matches = detect_triton_usage_for_module(model_new, *inputs)

if not used and backend == "triton":
    result.decoy_kernel = True
```

因此仅写 `import triton`、实际使用其他计算路径无法通过。TileLang 当前没有完全对应的 runtime hook，主要依赖具名 kernel profiling 和通用 ATen gate。

### 9. 自定义 kernel coverage

不同后端通过不同方式获得预期 kernel 名称：

- Triton 使用 runtime launch hook；
- TileLang 从 `@tilelang.jit`、`@T.prim_func` 提取；
- CUDA-Agent 和 TVM-FFI 由 backend artifact 提供 profiling hints。

随后将预期名称与 profiler 捕获的 CUDA events 匹配，计算：

```text
custom kernel CUDA time / total CUDA time
```

当前阈值如下：

| 时间覆盖率 | 处理 |
|---:|---|
| 大于或等于 30% | 正常 |
| 低于 30% | `suspected_decoy=true` |
| 低于 0.1% | 额外设置 `hard_decoy_coverage_candidate=true` |
| Profiler 未捕获 kernel | 标记 `coverage_unavailable`，不判 decoy |

低覆盖率目前只用于诊断：

```text
suspected_decoy_enforced=false
suspected_decoy_effect=DIAGNOSTIC_ONLY
```

原因是 extension 内部可能合法调用 cuBLAS/cuDNN，profiler 中显示的是库 kernel 名称。当前缺少可靠的调用来源归因，若仅按名称覆盖率硬拒绝，会误杀合法实现。

### 10. Profiler 空采样重试

Profiling 已启用但没有捕获到 CUDA kernel 时，pipeline 会根据配置进行重试。所有重试仍为空时，将其视为 profiler 可靠性问题，而不是 hacking 证据：

```text
coverage_measurement_valid=false
coverage_unavailable=true
decoy_kernel=false
```

### 11. GPU 计时保护

性能阶段使用 CUDA Event 包围 candidate forward，并在读取时间前同步设备：

```python
start.record()
candidate_forward()
end.record()
torch.cuda.synchronize()
runtime_ms = start.elapsed_time(end)
```

结合静态禁止修改 Event 和 synchronize，可避免 GPU 异步执行造成的虚假低耗时。

### 强制策略与诊断策略

| 策略 | 是否强制 | 结果 |
|---|---|---|
| 静态 forbidden pattern | 是 | 拒绝验证或编译 |
| Correctness shape/数值不匹配 | 是 | `correctness=false` |
| Candidate 执行异常 | 是 | `correctness=false` |
| 禁止的 ATen compute | 是 | `decoy_kernel=true` |
| Triton 未真实使用 | 是 | `decoy_kernel=true` |
| 自定义 kernel 覆盖率低于 30% | 否 | 仅 `suspected_decoy` 诊断 |
| 自定义 kernel 覆盖率低于 0.1% | 否 | 仅 hard-candidate 标记 |
| CUDA stream injection | 否 | Warning |
| Profiler 无数据或异常 | 否 | Fail-open 并记录错误 |

当前 hard decision 主要依赖静态禁止模式、正确性检查、动态 ATen fallback 检测和 Triton usage 检测。具名 kernel coverage 因 vendor library 调用归因尚不完整，暂时只承担观测和预警作用。
