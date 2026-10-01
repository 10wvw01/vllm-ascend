# 310P × Qwen3.6 × vLLM-Ascend 系统学习地图

> 目标不是“看懂几个名词”，而是最终能够从源码解释：一条请求为什么这样跑、310P 为什么这样定制、DFlash 为什么这样改、MM+AR 为什么选这个节点、当前实现为什么这样写、哪里仍可能不是最优。
>
> **先读 `00_主线故事_从一次Qwen请求到MM_AR融合.md`。** 后面的 01～20 不是平级知识点，而是同一条工程故事线上逐步出现的问题与解法。

---

## 0. 先抓住唯一主线

整套材料只围绕一个故事：

```text
要在 310P3 上高效服务 Qwen3.6
    ↓
vLLM 先解决在线调度/KV/Continuous Batch
    ↓
310P 和通用平台不同，需要 _310p 纵向定制
    ↓
Decode 太贵，引入 DFlash 一次验证多个 token
    ↓
进入 Qwen3.6 层后，TP=2 的 o_proj 产生 partial output
    ↓
partial 必须 SUM，形成 MM -> AllReduce 串行边界
    ↓
选择 o_proj/out_proj 做 MM+AR 融合
    ↓
朴素融合又遇到 8 核协作、cache 可见性、ready、generation
    ↓
为了流水，又引入 arena / wave / credit / lookahead
    ↓
为了 Graph，又必须解决 warmup / 固定地址 / generation 生命周期
    ↓
真实 Decode 小 M 又暴露 M-split 的 weight-stream 固定成本
    ↓
演进出 small-M N-split
    ↓
最后用 profiler 决定 q、threshold、lookahead、chunk 等下一代优化
```

主线文档：

[00_主线故事_从一次Qwen请求到MM_AR融合.md](00_主线故事_从一次Qwen请求到MM_AR融合.md)

以后每读到一个机制，都问：

```text
上一阶段到底遇到了什么问题，才逼出这个设计？
```

如果回答不了，就先不要背 API 或源码。

---

## 1. 学习完成后的能力标准

最终你应该可以不看文档独立回答下面这些问题，而且能指出源码位置：

```text
vLLM Scheduler 到底给 Model Runner 什么？
Prefill / Decode 的 M 为什么不同？
Qwen3.6 hybrid layer 中 full attention 和 GDN 怎么分流？
W8A8 模型为什么本次 o_proj 还是 FP16？
TP=2 RowParallelLinear 为什么天然产生 [M,N] partial + SUM？
为什么 self_attn.o_proj 是一个适合做 MM+AR 的边界？
为什么 8 个 AI Core 全都做 MM，core0 又能安全 signal？
为什么 cache clean 必须发生在 ready 前？
为什么 ready cell 每个占 64B？
为什么 eager generation 可以省掉每 wave 32KiB clear？
为什么 Graph 又必须 clear？
为什么 batch 和 wave 是两种不同粒度？
为什么 quiet 必须在 wave 尾，不能放进每个 wait？
为什么 small-M 改 N-split 后每 core weight 只有 1MiB？
为什么 blocked layout [8][T][256] 还可以一次 SDMA？
为什么 tail 的脏 padding 行不会污染有效输出？
DFlash 的 slot_mapping 为什么 block_size 错一个值就可能 acceptance≈0？
怎样用 profiler 判断 q 应该变大还是变小？
怎样证明“边搬边规约”值得做，而不是凭直觉？
```

如果只能回答“是什么”，还没到目标；要能回答：

```text
为什么这样设计
不这样会出什么具体错误/性能问题
替代方案是什么
为什么当前没选替代方案
以后什么证据出现时应该改
```

---

## 2. 第一阶段：先建立系统地图（01～12）

这部分面向第一次接触 vLLM / 昇腾 / speculative decode 的读者，作用是建立共同语言。

| 顺序 | 文档 | 作用 |
|---|---|---|
| 0 | [00_主线故事_从一次Qwen请求到MM_AR融合.md](00_主线故事_从一次Qwen请求到MM_AR融合.md) | 先知道整套项目为什么一步步演进到当前形态 |
| 1 | [01_先看懂全局地图.md](01_先看懂全局地图.md) | 分清 Qwen、vLLM、vLLM-Ascend、310P、Python/C++/AscendC 各自职责 |
| 2 | [02_一次请求端到端怎么跑.md](02_一次请求端到端怎么跑.md) | 建立 Scheduler → Runner → Model → NPU 的一条请求主线 |
| 3 | [03_Qwen3.6模型_TP与W8A8.md](03_Qwen3.6模型_TP与W8A8.md) | 先理解模型几何、MoE、TP、量化 |
| 4 | [04_310P定制能力是怎么接进去的.md](04_310P定制能力是怎么接进去的.md) | 看懂 `_310p` 为什么是一个纵向平台适配，而不只是几个 kernel |
| 5 | [05_投机推理_DFlash怎么工作.md](05_投机推理_DFlash怎么工作.md) | 建立 draft / verify / accept 的基本概念 |
| 6 | [06_为什么选择o_proj做MM_AR融合.md](06_为什么选择o_proj做MM_AR融合.md) | 融合点选择导读 |
| 7 | [07_MM_AR融合算子完整设计.md](07_MM_AR融合算子完整设计.md) | MM+AR 总体机制导读 |
| 8 | [08_从Python到AscendC逐层读代码.md](08_从Python到AscendC逐层读代码.md) | 先建立源码文件地图 |
| 9 | [09_MemFabric_同步_流水线与Graph.md](09_MemFabric_同步_流水线与Graph.md) | 同步/通信/Graph 的基础认知 |
| 10 | [10_性能分析与下一步优化方法论.md](10_性能分析与下一步优化方法论.md) | 性能优化思路导读 |
| 11 | [11_实战学习路线与自测题.md](11_实战学习路线与自测题.md) | 建立练习顺序 |
| 12 | [12_术语表_小白版.md](12_术语表_小白版.md) | 随查术语 |

**注意：06～10 现在应当视为“导读”，真正的源码深度在下面第二阶段。**

---

## 3. 第二阶段：源码精读与设计推导（13～20）

### 13｜一层 Qwen3.6 到底怎么执行

[13_源码精读_Qwen3.6一层到底怎么执行.md](13_源码精读_Qwen3.6一层到底怎么执行.md)

直接跟：

```text
Qwen3_5DecoderLayer.forward
 -> full attention / linear attention
 -> o_proj/out_proj
 -> ModelSlim quant route
 -> MemFabricMmArLinearMethod310
 -> torch custom op
```

重点解释：

```text
global K=4096 为什么 kernel local K=2048
W8A8 模型为什么目标层是 FP16
reduce_results=False 为什么是 correctness 所有权转移
M=10 与 M=512 为什么进入不同硬件路径
```

### 14｜DFlash 一轮 decode 的真实数据结构

[14_源码精读_310P_DFlash一次decode.md](14_源码精读_310P_DFlash一次decode.md)

直接跟 `set_inputs_first_pass()` 和 310P override，重点拆：

```text
context/query
1+K query length
block table
slot mapping
64/128 kernel block-size mismatch
per-layer cache layout
Graph persistent buffer
acceptance 为什么会被地址错误打到接近 0
```

### 15｜从数学和 critical path 真正推导融合点

[15_深度推导_为什么选o_proj做MM_AR.md](15_深度推导_为什么选o_proj做MM_AR.md)

不是列“o_proj 很适合”的优点，而是比较：

```text
o_proj
qkv_proj
residual
整 Attention
MoE communication
TP=2 vs TP>2
```

并解释融合的真正收益来自哪里、理论上限怎么估。

### 16｜AscendC producer 内核逐行精读

[16_AscendC内核逐行精读_MM_ready_signal.md](16_AscendC内核逐行精读_MM_ready_signal.md)

会手算：

```text
q=1/2/4 的 batch_m / batch_bytes
每 core A/C 地址偏移
每核写多少 KiB
为什么 B 完整共享
为什么 ready 64B/cell
为什么 clean -> ready -> signal 顺序不能变
```

### 17｜Host Runtime 状态机

[17_Runtime深度精读_arena_wave_credit_graph.md](17_Runtime深度精读_arena_wave_credit_graph.md)

把：

```text
96MiB pool
arena_rows
batch/wave
credit
lookahead=1
tail scratch
Graph warmup
single stream
poisoned context
shutdown lifecycle
```

放进一个明确状态机里。

### 18｜Small-M N-split 为什么有效

[18_SmallM_Nsplit深度推导.md](18_SmallM_Nsplit深度推导.md)

从小 M 的完整 weight-stream 固定开销出发，推到：

```text
M-split -> N-split
8MiB weight -> 8×1MiB slices
T stair
[8][T][256] blocked layout
blocked add
payload 缩小
```

并解释什么情况下 N-split 反而不适合。

### 19｜MemFabric 协议正确性证明

[19_MemFabric协议与Graph正确性.md](19_MemFabric协议与Graph正确性.md)

从 happens-before 角度证明：

```text
MM -> cache clean -> ready -> signal
peer signal -> wait -> add
wave reads -> quiet -> ack -> next gate
```

并解释 Graph generation、GVA geometry、mail strict validation、fail-stop。

### 20｜性能实验手册

[20_性能建模与实验手册.md](20_性能建模与实验手册.md)

不再泛泛说“可以调 q”，而是给出：

```text
M sweep 点位
q=1/2/4 实验矩阵
overlap ratio
small-path threshold 曲线
T staircase 实验
ready clear A/B
tail memset A/B
chunk-size microbenchmark
lookahead depth 实验
TP>2/低精度通信的收益上限判断
```

---

## 4. 不要按编号顺读：按“故事五幕”学习

### 第一幕：先让 Qwen3.6 成为一个可服务系统

```text
00 -> 01 -> 02 -> 03 -> 04
```

你要回答：请求如何变成一次 NPU forward？310P 为什么需要自己的 Runner/Graph/算子路径？

### 第二幕：让 Decode 一次推进更多 token

```text
05 -> 14
```

你要回答：DFlash 到底改了什么数据结构？为什么它会改变 MM+AR 看到的 M 分布？

### 第三幕：从真实模型关键路径发现融合机会

```text
13 -> 06 -> 15
```

你要回答：为什么这里天然是 `partial MM + SUM`？为什么选 o_proj，而不是随便选一个 Linear？

### 第四幕：把一个数学上简单的融合做成硬件上真正正确且可流水

```text
07 -> 16 -> 17 -> 19 -> 18
```

你要回答：为什么会依次出现 8 核 M-split、ready、cache clean、generation、arena、credit、Graph 生命周期和 small-M N-split？

### 第五幕：证明当前方案值不值得，并决定下一代怎么演进

```text
10 -> 20
```

你要回答：怎样用真实 workload 和 profiler 推翻或验证 q、lookahead、threshold、chunk 等设计假设？

`08 / 11 / 12` 是工具型章节：源码地图、练习题、术语索引，贯穿全程反复查。

---

## 5. 阅读每一段源码都问六个问题

### 1）上一步遇到了什么真实问题？

这是现在新增的第一问，也是最重要的一问。没有问题背景，后面的机制都会变成抽象名词。

### 2）这段代码的输入 tensor/地址到底是什么？

先写 shape，不要先看函数名。

例如融合入口：

```text
x_rank      [M,2048]
weight_rank [2048,2048]
partial     [M,2048]
```

### 3）它保护的资源是什么？

例如：

```text
ready 保护一个 batch 的 8-core C 完成
credit 保护整个 wave 的 arena 复用
```

### 4）如果删掉这段，会出现“错数、hang、还是只变慢”？

这能帮你区分 correctness 机制和 performance 机制。

### 5）为什么放在这一层实现？

```text
模型层？quant route？C++ runtime？AscendC？MemFabric adapter？
```

### 6）有没有替代方案，为什么当前没选？什么证据出现时应该换？

例如：

```text
small M 为什么不继续 M-split？
为什么不是通信专核？
为什么不是每 batch quiet？
为什么不是整个 Attention mega-kernel？
```

能回答第六个问题，才算开始具备继续演进设计的能力。

---

## 6. 以后判断文档和代码冲突时，以什么为准

优先级：

```text
1. 当前运行时代码
2. 当前模块 README / contract
3. 当前测试与 benchmark
4. 历史设计文档
5. 历史聊天结论
```

例如 q 默认值历史上出现过 q=2，但当前 `envs.py` 与 runtime contract 是默认 q=1。学习时必须养成“以当前 commit 的执行代码为准”的习惯。

---

## 7. 最终你要能讲出的不是三张孤立图，而是一部完整电影

第一张是请求如何进入模型：

```text
Scheduler -> Runner -> Qwen Layer -> Attention/GDN/MoE -> Linear -> NPU
```

第二张是为什么走到融合：

```text
TP=2 RowParallel
 -> partial MM
 -> 必须 SUM
 -> stock MM+HCCL 边界
 -> batch 化 MM+transport+add
```

第三张是融合内部如何保证正确又试图重叠：

```text
MM -> clean -> ready -> signal -> wait -> add -> quiet -> ack -> next gate
```

第四张是它如何继续演进：

```text
大 M M-split
 -> profiler 暴露 small-M weight-stream floor
 -> N-split
 -> 再用 profiler 判断下一步
```

当你能把这四段首尾连起来，并解释“为什么下一段必然从上一段的问题里长出来”，才真正形成了这套项目的系统认知。
