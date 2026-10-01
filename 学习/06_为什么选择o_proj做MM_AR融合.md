# 06｜为什么选择 o_proj 做 MM+AR 融合：怎样判断一个融合点值不值得做

## 1. 融合不是“看到两个相邻算子就合并”

做融合前至少要过三关：

```text
第一关：数学上能不能融合？
第二关：工程上能不能接管？
第三关：性能上值不值得融合？
```

本次 `o_proj MM + TP AllReduce` 恰好三关都比较好。

---

## 2. 第一关：数学上为什么合法

TP=2 Row Parallel Linear：

```text
rank0: Y0 = X0 @ W0
rank1: Y1 = X1 @ W1
final: Y = Y0 + Y1
```

MM 和 reduction 之间没有：

- 非线性激活；
- 数据依赖的条件分支；
- 必须先看到完整 Y 的其它操作。

因此：

```text
先完整算 Y0/Y1，再 AllReduce
```

可以重排成：

```text
分块算 partial Y
分块交换 partial Y
分块 local add
```

只要每一行/每一块最终做的仍然是：

```text
Y0 + Y1
```

数学结果就相同。

这给了 pipeline 的空间。

---

## 3. 为什么“紧邻”很重要

如果两个算子中间还有：

```text
MM -> activation -> AllReduce
```

一般不能直接把 partial output 先 reduce 或随意重排，因为：

```text
f(Y0) + f(Y1)
通常不等于
f(Y0 + Y1)
```

而这里是：

```text
MM -> SUM reduction
```

中间没有破坏线性的操作。

所以它是非常干净的融合边界。

---

## 4. 第二关：工程上为什么容易接管

当前 `o_proj/out_proj` 本来就是 `RowParallelLinear` 语义。

框架默认大致是：

```text
LinearMethod.apply()
    -> partial output
RowParallelLinear
    -> if reduce_results: tensor_model_parallel_all_reduce()
```

所以只要新的 Linear Method 能直接返回最终结果，就可以：

```text
新的 apply(): MM + reduce
layer.reduce_results = False
```

模型 forward 本身不需要重写。

这是一个非常理想的接入方式：

- 输入输出 tensor 形状不变；
- 上层模型不知道通信实现变了；
- feature off 能回退；
- correctness reference 很清楚：stock MM + HCCL AllReduce。

---

## 5. 第三关：性能上为什么有潜力

普通路径可以粗略写成：

```text
T_stock = T_MM + T_collective + T_boundary
```

其中 `T_boundary` 包括可能的：

- op dispatch；
- stream/collective 协调；
- 通用通信层固定开销；
- 中间 tensor 生命周期。

如果按 batch 分块并让 MM/SDMA 重叠，理想化以后更像：

```text
T_fused ≈ pipeline_fill
        + Σ max(T_MM_batch, T_SDMA_batch, T_reduce_batch的可重叠部分)
        + pipeline_drain
```

这并不保证一定更快，但至少有一个明确的性能假设：

> 把通信从“MM 完了以后再整体做”改成“partial output 一出来就开始搬”，从而缩短关键路径。

---

## 6. 为什么 `o_proj` 特别适合

### 特点一：它后面天然就需要 TP reduction

不是为了融合而人为加通信。

### 特点二：输出就是下一阶段要消费的最终 hidden

把 reduction 一起做完以后，上层可以直接继续，不需要额外转换接口。

### 特点三：目标 shape 稳定

当前两类目标 projection 都满足：

```text
local K = 2048
N = 2048
TP = 2
FP16
```

固定几何使得：

- tiling 好定；
- arena payload 好规划；
- 测试空间可控；
- small-M 专门优化容易做。

### 特点四：它是频繁出现的层

Transformer 多层重复执行。同一层级优化哪怕每层只省几十微秒，乘上几十层、数千 decode step 后也可能很可观。

### 特点五：当前目标 projection 是未量化 FLOAT

这让第一阶段不用同时解决 INT8 partial reduction 的 scale 语义，明显降低设计复杂度。

---

## 7. 为什么不是先融合 q_proj/k_proj/v_proj

不是说它们永远不能优化，而是作为“MM+AR”第一候选没有这里自然。

Q/K/V projection 的通信语义和后续 attention 数据布局不同，不一定紧跟一个和当前完全相同的 TP SUM。

如果没有明确的：

```text
MM partial -> 立即 reduce -> final tensor
```

那么就不能直接复用当前的两 rank exchange + add 设计。

优化点应该由真实数据依赖决定，而不是因为某个 Linear 名字看起来重要。

---

## 8. 为什么不是一上来融合整个 Attention

融合范围越大，理论优化空间可能越大，但工程风险也会指数上升：

```text
QKV
RoPE
KV Cache write/read
attention core
softmax / normalization
output projection
TP communication
Graph
动态 sequence length
```

如果第一次就做“大一统 Attention+通信超级算子”：

- correctness 定位困难；
- shape 空间巨大；
- Graph 更难；
- 一个局部问题可能导致整个路径回退。

当前方案选择一个**局部但高价值、数学边界清晰**的节点，是更稳健的优化迭代方式。

---

## 9. 为什么不是先融合 MoE 通信

MoE 当然也很可能是大头，但它的通信一般涉及：

```text
router
all-to-all / dispatch
不同 expert token 数
expert MM
combine
```

动态性和并行维度比固定 TP=2 `o_proj` 更复杂。

如果目标是先建立一套 310P 计算通信融合方法论，固定 TP=2 Row Parallel 是更干净的试验场。

后续瓶颈转移到 MoE 后，再针对 EP/dispatch/combine 做另一类融合更合理。

---

## 10. 一个可复用的“融合点筛选表”

以后你自己找新融合点，可以按下面打勾，而不是凭感觉。

### A. 数学合法性

```text
[ ] 两个阶段是否连续依赖？
[ ] 能否分块而保持数学等价？
[ ] 中间是否有非线性或全局依赖？
[ ] 浮点重排会不会导致不可接受误差？
```

### B. 数据流价值

```text
[ ] 前一算子的输出是否立即被后一算子消费？
[ ] 能否避免中间 GM 写回/再读？
[ ] 能否提前启动 DMA/通信？
[ ] 能否减少格式转换？
```

### C. 调度价值

```text
[ ] 当前两阶段是否串行？
[ ] 是否存在不同硬件 engine 可重叠？
[ ] 是否能减少 launch / host sync / collective 固定开销？
```

### D. 形状和覆盖率

```text
[ ] 该 shape 在真实 workload 中出现频率高吗？
[ ] shape 是否足够稳定便于专门优化？
[ ] Prefill/Decode 是否需要不同策略？
```

### E. 工程接入

```text
[ ] 能否保留原输入输出合同？
[ ] 能否 feature flag 回退？
[ ] correctness reference 是否明确？
[ ] 生命周期/Graph 能否管理？
```

### F. 可观测性

```text
[ ] profiler 能看出 MM / DMA / reduce 时间吗？
[ ] 能定义端到端基线吗？
[ ] 出错时能定位协议状态吗？
```

只有“相邻”这一条，远远不够。

---

## 11. 本次融合真正想消除什么

很多人听“融合算子”会想到：

```text
少一次 GM 中间 tensor
```

但本项目的主要价值更偏系统层：

### 目标一：把通用 collective 换成 TP=2 特化数据流

```text
peer SDMA exchange + local add
```

### 目标二：让 partial MM 和通信按 batch 形成流水

```text
P(n+1) before W/A(n)
```

### 目标三：Runtime 拥有完整生命周期

从一次 Python op 内统一控制：

```text
gate -> producer -> wait -> add -> quiet -> ack
```

### 目标四：针对 small-M 做不同并行方式

通用 Linear+AllReduce 路径未必会为这个固定模型 shape 做如此窄的优化。

---

## 12. 什么情况下这个融合可能不划算

必须接受一个事实：融合不是天然更快。

可能亏损的情况：

### 小 M

如果固定 batch MM 大量 padding，计算浪费可能大于通信收益。当前已经因此增加 N-split small path。

### 通信本来不是瓶颈

如果 MM 占 95%，AllReduce 很短，花很大工程量改通信可能收益有限。

### overlap 根本没发生

如果 MM 和 SDMA 在目标硬件上争同一关键资源，代码虽然流水化，timeline 仍然串行。

### 额外协议开销太大

ready、cache clean、wait、credit 等固定开销若大于 HCCL 原本开销，会适得其反。

### shape 覆盖太低

只优化极少出现的 M/层，端到端收益会被其它层淹没。

所以最终裁判永远是：

```text
真实 workload + profiler + end-to-end KPI
```

---

## 13. 选点时要分清“理论吞吐”和“服务收益”

假设某融合 kernel benchmark 快 30%，但它只占整模型 5% 时间。

Amdahl 定律告诉我们端到端最大收益有限。

反过来，某个算子只快 10%，但它每层、每 step 高频出现，也可能更有价值。

因此选融合点前应该先做：

```text
真实模型 profiler
  -> 按 op / layer 聚合时间
  -> 找高频关键路径
  -> 看前后数据依赖
  -> 再决定融合
```

而不是先写 kernel，再找理由。

---

## 14. 为什么当前实现从 full-attention o_proj 又扩到 GDN out_proj

这是一种很健康的演进方式：

第一步先验证一个明确场景；然后发现另一类层满足同样合同：

```text
local K=2048
N=2048
TP=2
FP16
RowParallel SUM
```

于是扩“资格路由”，而不是复制另一份 kernel。

这说明可复用的真正抽象不是：

```text
这是 full attention
```

而是：

```text
这是满足相同 MM+TP-reduction contract 的 projection
```

---

## 15. 当前融合是否已经“最好”

不能这样下结论。

当前代码已经体现多轮实测优化，但仍有很多可研究问题：

- q 是否应该按 M 自动选择，而不是进程固定？
- lookahead=1 是否最优？
- small-M 的 N-split stair 是否还能调？
- local add 能否和接收/后继算子进一步重叠？
- 是否能降低 ready/cache clean 固定开销？
- weight slice 8 MiB/layer 的内存换时间是否永远值得？
- FP16 payload 是否是最佳通信格式？
- TP>2 时应该继续 P2P 特化还是回到 collective？

“好的优化工程”不是证明当前方案完美，而是清楚知道：

```text
当前方案解决了什么
代价是什么
尚未解决什么
下一步如何验证
```

---

## 16. 本章自测

1. 为什么 `MM -> SUM` 比 `MM -> activation -> SUM` 更适合流水化融合？
2. 为什么 `reduce_results=False` 是融合正确性的一部分？
3. 选择 `o_proj` 的数学、工程、性能理由分别是什么？
4. 为什么“整个 Attention 全融合”不一定是第一阶段最好的工程选择？
5. 如果另一个层也有 `[M,2048]@[2048,2048]`，能否直接接当前融合？还必须检查哪些语义？
6. 你会怎样用 profiler 证明某个候选融合点值得做？

下一章开始完整拆解当前 MM+AR 设计本身。
