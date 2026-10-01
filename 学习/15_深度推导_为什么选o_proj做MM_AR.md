# 15｜深度推导：为什么选 `o_proj/out_proj` 做 MM+AR，而不是别的节点

> 这一章不是“列优点”，而是按真实优化工程的方法，从模型数据流、TP 数学、通信边界、硬件可重叠性、实现风险和收益上限一步步筛出当前融合点。

---

## 1. 先定义我们要找的不是“两个相邻算子”，而是“关键路径边界”

所谓融合点，真正应该满足的是：

```text
上游阶段产生的数据
   |
   | 这个数据如果不在这里被消费，就必须落地/同步/通信
   v
下游阶段
```

我们想优化的，是这条边界本身的成本。

例如：

```text
MatMul -> HCCL AllReduce
```

边界成本可能包含：

```text
MatMul partial output 写 GM
PyTorch / dispatcher 返回
RowParallelLinear 决定 reduce
collective launch
HCCL 组织通信
通信完成
后续算子才能消费 final output
```

真正的问题不是：

```text
能不能把 MM 和 AR 写进一个函数？
```

而是：

> 能不能让 partial output 在“刚产生”时就进入通信，并且让通信和后续 partial MM 重叠，从而缩短 critical path？

这才是本次融合的设计出发点。

---

## 2. 从 `o_proj` 的 TP 数学开始，不先谈性能

当前目标 Linear 是 RowParallel。

全局权重可按 K 维写成：

```text
W = [W0]
    [W1]
```

输入同样按 K 切分：

```text
X = [X0 X1]
```

完整结果：

```text
Y = X @ W
  = X0 @ W0 + X1 @ W1
```

所以两张卡：

```text
rank0: P0 = X0 @ W0
rank1: P1 = X1 @ W1
```

最后：

```text
Y = P0 + P1
```

这有三个非常重要的性质。

### 性质 1：partial output 的 shape 与 final output 完全相同

```text
P0: [M,N]
P1: [M,N]
Y : [M,N]
```

当前 N=2048。

### 性质 2：最终操作是逐元素 SUM

没有 softmax，没有 activation，没有 normalization。

### 性质 3：不同 row 之间互不依赖

对于任意一行 m：

```text
Y[m,:] = P0[m,:] + P1[m,:]
```

所以可以按 M 分块：

```text
batch0 的 partial
一出来就交换/加
不需要等待 batch1
```

这三个性质共同构成“可流水化 MM+AR”的数学基础。

---

## 3. 什么叫“可以分块保持等价”

假设：

```text
M = 1024
```

我们分成 4 个 256-row batch：

```text
X = [X^0]
    [X^1]
    [X^2]
    [X^3]
```

因为矩阵乘法每一行独立：

```text
P_rank = concat(
  X^0_rank @ W_rank,
  X^1_rank @ W_rank,
  X^2_rank @ W_rank,
  X^3_rank @ W_rank
)
```

而 TP SUM 也是逐元素：

```text
Y^i = P0^i + P1^i
```

于是：

```text
先算完全部 MM 再 AR
```

和：

```text
MM0 -> exchange0 -> add0
MM1 -> exchange1 -> add1
...
```

在数据依赖上是等价的。

这给 runtime 一个非常重要的自由：

> 可以重新排列执行时间，而不改变数学语义。

所有计算通信 overlap，本质都建立在这种“允许重排”上。

---

## 4. 为什么 `MM -> activation -> AR` 就不是同一个问题

假设中间有非线性 f：

```text
rank0: f(P0)
rank1: f(P1)
然后 SUM
```

结果是：

```text
f(P0) + f(P1)
```

而如果先 SUM：

```text
f(P0 + P1)
```

通常：

```text
f(P0)+f(P1) != f(P0+P1)
```

因此真正能把通信推进到 partial 阶段的前提，是 reduction 前没有破坏线性的全局操作。

`o_proj` 后的 TP SUM 恰好是非常干净的线性边界。

---

## 5. 再看模型位置：为什么 `o_proj` 是一个高价值位置

Full Attention 大致：

```text
hidden
 -> qkv_proj
 -> q/k processing
 -> attention
 -> gate
 -> o_proj
 -> residual path 后续
```

`o_proj` 的输出不是某个临时中间量，而是 attention 子层最终要交给 decoder layer 的 hidden 表示。

因此：

```text
o_proj partial 不做完 TP SUM
后面的 residual / layernorm / MLP 都不能把它当完整 hidden 使用
```

也就是说 AllReduce 直接位于 critical path。

通信不是“后台慢慢做也行”的附属任务，而是后续计算的强依赖。

所以只要能隐藏其中一部分时间，端到端就有机会直接受益。

---

## 6. 为什么 GDN `linear_attn.out_proj` 也能复用同一 kernel

一开始从 full attention `self_attn.o_proj` 出发很自然。

但真正可复用的抽象不是“它叫 o_proj”，而是它满足同一个合同：

```text
RowParallel SUM
TP=2
K_global=4096
K_local=2048
N=2048
FP16 unquantized path
```

GDN 的 `linear_attn.out_proj` 也满足这组条件。

所以最终融合路由不是只检查字符串后缀，还检查真实 `layer_types[layer_idx]`。

这说明一个成熟优化应该从：

```text
为某个具体层写 special case
```

逐步抽象成：

```text
为满足同一数学/shape/硬件合同的一类节点提供实现
```

但又不能抽象过头：当前 kernel 仍然只支持这一组固定几何。

---

## 7. 选择融合点时，先算“理论上最多能省多少”

假设 stock 时间线：

```text
T_stock = T_MM + T_AR + T_boundary
```

其中：

- `T_MM`：完整 o_proj matmul；
- `T_AR`：generic allreduce；
- `T_boundary`：dispatcher / launch / framework / 同步固定成本。

如果完全串行，比如：

```text
T_MM=400us
T_AR=200us
T_boundary=50us
```

总共：

```text
650us
```

最理想的流水融合也不可能把它变成 0。

如果通信与计算完全重叠，理论下界更接近：

```text
max(T_MM, T_exchange)
+ reduce/drain/fill 固定成本
```

假设 exchange 200us，MM 400us，则通信最多可隐藏约 200us。

也就是说：

```text
潜在收益上限主要由“可重叠的通信比例”决定
```

如果 profiler 显示 stock AllReduce 只有 10us，而 MM 500us，就算工程上能融合，优先级也可能很低。

所以正确流程应该是：

```text
profiler -> 找 critical boundary -> 建上限模型 -> 再写 kernel
```

而不是反过来。

---

## 8. 为什么不是先优化 qkv_proj

`qkv_proj` 当然很重，但它后面直接进入：

```text
split q/k/v
norm
RoPE/MRoPE
attention
```

这里的并行语义通常不是“刚做完一个 RowParallel partial，马上必须 SUM 成同 shape final tensor”。

也就是说，它和当前 AllReduce 之间没有一个完全相同的：

```text
P0[M,N] + P1[M,N]
```

边界。

如果想优化 qkv_proj，更可能是另一类问题：

```text
projection + split + norm + rope 融合
```

或者数据格式/内存访存优化。

不能因为“也是 Linear”就强行套当前 MM+AR 方案。

---

## 9. 为什么不是先把 `o_proj + residual` 一起融合

这看起来很诱人：既然 o_proj 后面很快就会进 residual，是否可以：

```text
MM + AR + residual add
```

一起做？

理论上有可能，但需要进一步确认：

```text
residual tensor 是否每 rank 相同？
residual add 的 dtype/format 是什么？
上层 fused layernorm 是否已经有别的优化？
Graph 输入输出合同会不会改变？
是否能避免额外 GM round-trip？
```

当前项目先选择停在：

```text
完成 RowParallelLinear 的完整语义
```

这是一个天然模块边界。

这样上层看到的仍是“正常 Linear 输出”，接入风险最低。

以后若 profiler 证明 residual/layernorm 边界仍占明显成本，可以再扩大融合边界。

这是一种非常重要的优化策略：

> 先占领一个语义清晰的局部边界，再基于数据决定是否继续向上下游扩张。

---

## 10. 为什么不是一开始就做整个 Attention mega-kernel

整个 Attention 涉及：

```text
QKV projection
RoPE
KV cache read/write
attention score
softmax
value aggregation
gate
o_proj
TP reduction
```

理论上融合越大，中间落地越少。

但工程复杂度不是线性增加，而会快速放大：

```text
动态 sequence length
prefill/decode 差异
KV cache layout
Graph shape
不同 layer type
量化路径
TP/CP
异常定位
```

如果最终输出不对，你很难判断问题来自：

```text
RoPE？
KV 地址？
softmax？
MM？
通信？
```

而 `o_proj + AR` 的 reference 极其清晰：

```text
reference = stock unquantized matmul + HCCL allreduce
```

这使 correctness 和性能 A/B 都非常容易做。

对于第一次建立 310P 计算通信协作能力，这是非常重要的工程价值。

---

## 11. 为什么不是先做 MoE dispatch/combine

Qwen3.6 MoE 的通信潜力可能更大，但问题形态完全不同。

MoE 通信通常受：

```text
router 结果
每个 expert token 数
EP world size
all-to-all / dispatch
expert MM
combine
```

影响。

token 分布是动态的，负载可能不均衡。

而当前 `o_proj`：

```text
TP=2 固定
N/K 固定
每 rank payload shape 可预测
SUM 语义固定
```

所以它是一个更容易先把下面这些基础能力做扎实的试验场：

```text
peer SDMA
cache visibility
multi-core rendezvous
arena reuse
credit
Graph capture
fail-stop
```

等这些基础设施成熟后，再把方法迁移到更复杂的 MoE 通信，风险更低。

---

## 12. 为什么 TP=2 特别适合做“互换 + 本地加”

AllReduce SUM 对 2 个 rank：

```text
rank0 有 P0
rank1 有 P1
```

只需要：

```text
rank0 收 P1 -> P0+P1
rank1 收 P0 -> P1+P0
```

每边只需要得到 peer 的一个 partial。

这可以被简单映射成：

```text
one peer SDMA exchange
+
local add
```

如果 TP=4：

```text
Y = P0+P1+P2+P3
```

问题立刻变复杂。

可以做：

```text
3 个 peer payload
```

但流量和同步都会增加；也可以设计 ring/tree reduce-scatter/allgather，但那已经重新接近一个 collective library 的问题。

所以当前设计从 TP=2 开始，不只是产品约束，更是因为它让通信算法可以极度特化。

---

## 13. 为什么当前融合的本质不是“一个 kernel”

很多人会问：

```text
既然叫融合算子，为什么里面还有 producer/wait/add 多个 kernel？
```

因为这里融合的是**框架语义与调度所有权**。

模型上层原来看到：

```text
Linear op
AllReduce op
```

现在只看到：

```text
memfabric_mm_ar_allreduce custom op
```

这个 op 内部自己控制：

```text
MM producer
SDMA
wait
local add
quiet
credit
```

如果为了追求“只有一个 AscendC kernel”而把所有阶段硬塞进同一个核，反而可能：

```text
让 AI Core 等通信
让资源无法独立调度
降低 overlap
增加生命周期复杂度
```

所以判断“融合程度”的标准不应该是 kernel count，而应该是：

> 有没有消除无用边界？有没有获得原先拿不到的调度自由？

---

## 14. 从数据生命周期看，当前融合真正改变了什么

### stock 路径

```text
x
 |
 v
stock matmul
 |
 v
partial tensor
 |
 v
框架返回 / RowParallel 控制
 |
 v
HCCL AllReduce
 |
 v
final tensor
```

### fused 路径

```text
x
 |
 v
custom runtime
 |
 +--> batch MM -> send arena -> peer SDMA
 |                       |
 |                       v
 +-------------------- wait -> local add -> output
```

这里没有完全消除 partial output 的 GM 存储，因为 peer SDMA 仍然需要从可见的 send arena 读。

所以当前融合的主要收益不是：

```text
彻底去掉 partial output GM
```

而是：

```text
1. 取消 generic HCCL 路径的一部分固定成本
2. 获得 batch 粒度的通信提前启动
3. 让 MM(n+1) 与 SDMA(n) 有机会重叠
4. 对 TP=2 使用更简单的 peer exchange
5. 对 small-M 使用固定模型专用 N-split
```

这个判断非常重要，否则你会对“融合后为什么还有 send arena 写回”产生误解。

---

## 15. 如果未来要继续扩大融合，应该从哪里看

不要先凭想象画超级算子。

应该拿 profiler 依次回答：

```text
1. fused MM+AR 后，critical path 最大剩余块是什么？
2. local add 是否变成明显占比？
3. 下一个 consumer 是否紧跟 output？
4. output 是否被再次读 GM？
5. 是否能通过 producer-consumer fusion 消掉这次读写？
6. 扩大边界会不会破坏 Graph / fallback / 通用性？
```

例如如果发现：

```text
local add 80us
后面 residual add 30us
而两者都读取同一 output
```

才有依据研究：

```text
peer reduce + residual 融合
```

如果 profiler 显示 add 只有 5us，就没必要为了“看起来更融合”增加复杂度。

---

## 16. 一个通用的融合点评分方法

以后找新节点，可以按 6 个维度做分析，但不要简单打总分，而是逐项证明。

### 数学可分块性

必须证明：

```text
分块执行 + 重排
与原数学等价或误差可接受
```

### critical-path 强度

下游是否必须等它？

### 数据量

中间 tensor 是否大到值得优化？

### 异构 overlap

前后阶段是否能利用不同 engine：

```text
Cube / Vector / SDMA / network
```

### shape 稳定性

越固定，越适合 aggressive specialization。

### 工程可回退性

必须有清晰 reference 和 feature-off 路径。

当前 `o_proj/out_proj` 的价值就在于这 6 项都比较干净。

---

## 17. 本章自测

1. 为什么 `o_proj` 的 TP SUM 能按 M 分 batch 重排？
2. 为什么 partial output 和 final output shape 一样对融合很有利？
3. 当前融合最大的收益来源到底是不是“省一次 GM 写”？
4. 为什么 TP=2 可以简化成 peer exchange + local add？TP=4 为什么不能直接照搬？
5. 为什么 qkv_proj 不能机械复用当前 MM+AR 方案？
6. 为什么 mega-attention kernel 不一定比局部融合更适合作为第一步？
7. 如果 local add 成为下一阶段瓶颈，你会先收集什么证据再决定扩融合？

能把这些问题回答成一条因果链，而不是罗列关键词，才算掌握“怎么选融合节点”。
