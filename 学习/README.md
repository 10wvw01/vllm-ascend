# 310P × Qwen3.6 × vLLM-Ascend 系统学习地图

> 面向第一次接触 vLLM、昇腾推理、投机推理和融合算子的学习者。
>
> 本目录以分支 `feat/310p-memfabric-public-api` 当前实现为准。编写基线：`90a4d1f8fd0ce0100d032bafa484bbd9a20dcbc9`。

## 1. 最终要学会什么

学完以后，不只是“知道代码在哪”，而应该能独立回答：

- vLLM 是什么，vLLM-Ascend 又补了什么？
- 一条请求从 HTTP/Engine 进入，到 310P 上真正执行算子，中间经过哪些层？
- Qwen3.6-35B-A3B-w8a8 的 W8A8、MoE、TP=2 分别意味着什么？
- Prefill、Decode、KV Cache、Batch、Graph、投机推理分别解决什么问题？
- 310P 为什么需要单独的 `vllm_ascend/_310p`、patch、AscendC 算子？
- DFlash 投机推理为什么能更快？draft、verify、accept/reject 到底在干什么？
- 为什么本次 MM+AR 选择 `self_attn.o_proj` / `linear_attn.out_proj` 作为融合点？
- 普通路径 `MatMul -> HCCL AllReduce` 和当前融合路径到底差在哪里？
- 为什么 8 个 AI Core 全部参与 MM，core0 还要兼任 signal owner？
- 为什么要有 ready、generation、credit、arena、wave、batch、quiet？
- 为什么小 M 又改成 N-split？为什么尾批可以带“脏的补齐行”却仍然正确？
- 当前方案是不是最优？如何通过 profiler 和性能模型判断下一步应该优化哪里？

真正的目标是获得一套“看懂系统、找瓶颈、选融合点、做实验、证明优化有效”的方法，而不是背当前实现。

---

## 2. 建议学习顺序

请严格按下面顺序读。前面章节是在给后面的融合算子铺路。

| 顺序 | 文档 | 读完应掌握 |
|---|---|---|
| 1 | [01_先看懂全局地图.md](01_先看懂全局地图.md) | vLLM、vLLM-Ascend、310P、模型、算子之间是什么关系 |
| 2 | [02_一次请求端到端怎么跑.md](02_一次请求端到端怎么跑.md) | 一次请求从调度到 NPU 执行的完整数据流 |
| 3 | [03_Qwen3.6模型_TP与W8A8.md](03_Qwen3.6模型_TP与W8A8.md) | 模型结构、MoE、TP、RowParallelLinear、W8A8 与 FP16 例外 |
| 4 | [04_310P定制能力是怎么接进去的.md](04_310P定制能力是怎么接进去的.md) | 310P 定制代码为什么是“纵向切片”，patch/runner/quant/ops 怎样协同 |
| 5 | [05_投机推理_DFlash怎么工作.md](05_投机推理_DFlash怎么工作.md) | draft + target verify 的原理，以及 310P 下 DFlash 的特殊实现 |
| 6 | [06_为什么选择o_proj做MM_AR融合.md](06_为什么选择o_proj做MM_AR融合.md) | 如何判断一个融合点“合法、值得、可落地” |
| 7 | [07_MM_AR融合算子完整设计.md](07_MM_AR融合算子完整设计.md) | 当前融合算子的数学、内存、同步、流水线和失败语义 |
| 8 | [08_从Python到AscendC逐层读代码.md](08_从Python到AscendC逐层读代码.md) | 从模型层一路跟进到 AscendC kernel 的真实调用链 |
| 9 | [09_MemFabric_同步_流水线与Graph.md](09_MemFabric_同步_流水线与Graph.md) | MemFabric public API、cache coherence、credit、Graph 生命周期 |
| 10 | [10_性能分析与下一步优化方法论.md](10_性能分析与下一步优化方法论.md) | 如何判断当前是不是最好，以及怎样找下一步优化方向 |
| 11 | [11_实战学习路线与自测题.md](11_实战学习路线与自测题.md) | 用实验把“看懂”升级为“会做” |
| 12 | [12_术语表_小白版.md](12_术语表_小白版.md) | 随时查不懂的专业名词 |

---

## 3. 先记住一张总图

```text
用户请求
  |
  v
vLLM 调度器 / Engine
  |  决定本轮哪些请求、多少 token 一起跑
  v
vLLM Model Runner
  |
  v
vLLM-Ascend 310P Model Runner / Patch
  |
  +-------------------------+
  |                         |
  | 普通模型计算             | 投机推理 DFlash
  |                         | draft -> verify -> accept/reject
  v                         v
Qwen3.6 模型层
  |
  | Full Attention / GDN / MoE / Linear ...
  v
RowParallelLinear(o_proj / out_proj)
  |
  | 当前定制命中后改走 MemFabricMmArLinearMethod310
  v
Python custom-op wrapper
  |
  v
torch.ops._C_ascend.memfabric_mm_ar_allreduce
  |
  v
C++ Runtime
  |  arena / wave / batch / graph / lifecycle
  v
AscendC kernels
  |  8-core MM -> cache clean -> ready
  |                     |
  |                     v
  |              MemFabric SDMA signal/wait
  |                     |
  +-------------> FP16 local add
                        |
                        v
                已完成 TP=2 规约的输出
```

这条链就是整个学习目录的“主轴”。后面所有章节都只是把其中一层放大。

---

## 4. 当前实现最重要的事实

### 4.1 这不是简单地把两个函数粘在一起

当前所谓“MM+AR 融合”，更准确地说是**系统调度边界上的融合**：

- 对模型层来说，只调用一次 custom op；
- custom op 内部统一控制 MM、跨卡交换、等待、规约和生命周期；
- 设备侧仍有 producer / wait / add / protocol 等不同 kernel；
- 性能收益来自减少通用路径开销，并让计算与通信形成流水，而不是强行把所有逻辑塞进一个 AscendC kernel。

理解这一点非常重要：**融合的本质是消除不必要的边界，并创造可重叠的数据流，不等于“kernel 数量必须等于 1”。**

### 4.2 W8A8 模型不代表每一层都是 INT8

本次命中的：

```text
self_attn.o_proj
linear_attn.out_proj
```

在当前目标 checkpoint 的 ModelSlim 路由中属于被跳过量化的 FLOAT 层，310P 上按 FP16 执行。因此当前融合合同是：

```text
activation FP16
weight FP16/NZ
partial output FP16
MemFabric exchange FP16
local reduce FP16
```

不要看到模型名字里的 `w8a8` 就默认所有 Linear 都是 W8A8。

### 4.3 当前 q 默认值以代码为准

历史设计文档里曾出现过默认 `q=2`，但当前 `vllm_ascend/envs.py` 和融合目录 README 的实现合同是：

```text
baseM = 256
q ∈ {1,2,4}
default q = 1
batch_m = 256 * q
```

学习源码时要建立一个习惯：**运行时代码 > 当前模块 README > 历史设计文档**。文档也会过期。

### 4.4 当前方案不是“终极答案”

代码里已经包含多轮实机发现后形成的优化，例如：

- eager ready 用 generation 避免每波清 32 KiB；
- tail scratch 只初始化一次，避免每尾批大 memset；
- small-M 从 M-split 改成 N-split；
- lookahead=1 让 MM、SDMA、wait/add 尝试重叠。

这些选择都来自特定硬件、特定 shape、特定版本上的测量。它们是“当前证据下的工程答案”，不是永恒定律。后续每次改 CANN、MemFabric、模型 shape 或 TP，都要重新验证。

---

## 5. 阅读源码时的三个原则

### 原则一：先问“数据是什么”，再问“函数叫什么”

例如 `M/K/N` 不清楚时直接看 kernel 会非常痛苦。先写清楚：

```text
X_rank: [M, 2048]
W_rank: [2048, 2048]
Y_partial: [M, 2048]
TP=2 final: Y = Y_rank0 + Y_rank1
```

函数名马上就会变得好懂。

### 原则二：所有同步都问“它在保护什么资源”

- ready：保护“8 个 core 都把这个 batch 写完了”；
- wait mail：保护“peer 数据真的已经到 recv arena”；
- quiet：保护“本波发出的 SDMA 已经 drain”；
- credit：保护“下一波不会过早覆盖当前 arena”；
- Graph warmup：保护“capture 里不会发生 malloc / 首启 / host rendezvous”。

不要只记 API 名字。

### 原则三：优化必须回答“省了哪段时间”

任何“更快”的方案都应该能映射到时间线：

```text
MM | 通信 | reduce | launch | sync | format/layout | memset/copy | graph overhead
```

如果说不清楚优化减少的是哪一段，通常就还没有形成可以验证的性能假设。

---

## 6. 建议同时打开的源码

学习本目录时，建议 IDE 里长期固定打开：

```text
vllm_ascend/_310p/ops/memfabric_mm_ar.py
vllm_ascend/_310p/quantization/modelslim_config.py
vllm_ascend/_310p/model_runner_310p.py
vllm_ascend/_310p/spec_decode/dflash_proposer_310.py
vllm_ascend/patch/worker/patch_idex_310.py

csrc/_310P/memfabric_mm_ar/memfabric_mm_ar_binding.cpp
csrc/_310P/memfabric_mm_ar/memfabric_mm_ar_runtime.cpp
csrc/_310P/memfabric_mm_ar/memfabric310p_adapter_api.h
csrc/_310P/memfabric_mm_ar/memfabric310p_adapter.cpp
csrc/_310P/memfabric_mm_ar/memfabric310p_device.asc

vllm_ascend/envs.py
```

读到一个陌生概念，先查 [12_术语表_小白版.md](12_术语表_小白版.md)，再回源码。

---

## 7. 什么叫“真的学会了”

最终你应该能不看文档，自己画出以下两张图：

1. **一次 Qwen3.6 请求的端到端执行图**：scheduler → runner → model → attention/GDN/MoE → linear → custom op → NPU；
2. **一个 MM+AR wave 的设备时间线**：P0 → P1/W0/A0 → P2/W1/A1 → drain → quiet → ack。

并且能解释每一条边为什么存在、删掉会发生什么、如果 profiler 显示某段变成瓶颈下一步怎么做。

做到这一点，才算从“会用项目”进入“能继续优化项目”的阶段。
