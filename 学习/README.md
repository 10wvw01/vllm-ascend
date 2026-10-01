# 310P × Qwen3.6 × vLLM-Ascend 系统学习地图

> 目标不是“看懂几个名词”，而是最终能够从源码解释：一条请求为什么这样跑、310P 为什么这样定制、DFlash 为什么这样改、MM+AR 为什么选这个节点、当前实现为什么这样写、哪里仍可能不是最优。
>
> 本目录分成两层：**01～12 是入门/地图层，13～20 是源码精读/设计推导层。** 如果你的目标是“真正吃透”，01～12 只算预备知识，必须继续读 13～20。

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

## 4. 建议的真正学习顺序

不要从 01 一口气顺读到 20。

推荐：

```text
第一轮：01 -> 02 -> 03 -> 04 -> 05
目的：知道系统里有哪些角色

第二轮：13 -> 14
目的：用真实源码把“模型执行”和“投机推理”落地

第三轮：06 -> 15
目的：先看融合点导读，再看真实推导

第四轮：07 -> 16 -> 17 -> 18 -> 19
目的：从总设计进入 kernel/runtime/protocol

第五轮：10 -> 20
目的：从“知道有哪些优化方向”升级到“会设计实验判断方向”

最后：08 / 11 / 12 当源码地图、练习题和术语索引反复使用
```

---

## 5. 阅读每一段源码都问五个问题

### 1）这段代码的输入 tensor/地址到底是什么？

先写 shape，不要先看函数名。

例如融合入口：

```text
x_rank      [M,2048]
weight_rank [2048,2048]
partial     [M,2048]
```

### 2）它保护的资源是什么？

例如：

```text
ready 保护一个 batch 的 8-core C 完成
credit 保护整个 wave 的 arena 复用
```

### 3）如果删掉这段，会出现“错数、hang、还是只变慢”？

这能帮你区分 correctness 机制和 performance 机制。

### 4）为什么放在这一层实现？

```text
模型层？quant route？C++ runtime？AscendC？MemFabric adapter？
```

### 5）有没有替代方案，为什么当前没选？

例如：

```text
small M 为什么不继续 M-split？
为什么不是通信专核？
为什么不是每 batch quiet？
为什么不是整个 Attention mega-kernel？
```

能回答第五个问题，才算开始具备继续演进设计的能力。

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

## 7. 最终你要能画出的三张图

### 图一：一次请求端到端

```text
Scheduler -> Runner -> Qwen Layer -> Attention/GDN/MoE -> Linear -> NPU
```

### 图二：一次 MM+AR wave

```text
P0
P1 + SDMA0
W0/A0 + SDMA1
P2 ...
quiet
ack
```

### 图三：correctness happens-before

```text
MM -> clean -> ready -> signal -> wait -> add -> quiet -> ack -> next gate
```

不仅能画，还要能解释：

```text
每条边保护什么资源
删掉后会发生什么
哪条边是性能可调、哪条边是 correctness 不能动
```

做到这一层，才接近“吃透融合算子怎么玩”。
