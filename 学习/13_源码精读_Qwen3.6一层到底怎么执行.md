# 13｜源码精读：Qwen3.6 的一层在这个仓库里到底怎么执行

> 目标：不再停留在“Qwen3.6 有 Attention / MoE / TP”这种概念层，而是沿着当前仓库真实代码，把一层模型从 Python forward 一直追到本次 MM+AR custom op 的入口。
>
> 本章建议同时打开：
>
> - `vllm_ascend/patch/worker/patch_qwen3_5.py`
> - `vllm_ascend/_310p/quantization/modelslim_config.py`
> - `vllm_ascend/_310p/ops/memfabric_mm_ar.py`
> - `csrc/_310P/memfabric_mm_ar/memfabric_mm_ar_binding.cpp`

---

## 1. 先明确：Qwen3.6 在代码里为什么叫 `qwen3_5`

当前目标 checkpoint 属于 Qwen3.5/3.6 这一套模型实现，HF text config 的 `model_type` 在本分支里仍按：

```text
qwen3_5_moe_text
```

做识别。

因此看到：

```python
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
```

不要误判成“这不是 Qwen3.6”。这里的名称是框架内部模型协议名，不是我们学习时要纠结的产品版本号。

真正应该关心的是：

```text
这一层的结构是什么？
这一层的 Linear 形状是什么？
这一层是否被量化？
这一层在 TP=2 下需要什么通信？
```

---

## 2. 一层 DecoderLayer 的真实骨架

`patch_qwen3_5.py` 里把上游 `Qwen3_5DecoderLayer.forward` 替换成 `AscendQwen3_5DecoderLayer.forward`。

核心结构可以直接从源码抽象成：

```text
hidden_states
   |
   v
input_layernorm
   |
   +------------------------------+
   |                              |
   | layer_type=full_attention    | layer_type=linear_attention
   v                              v
self_attn                      linear_attn (GDN)
   |                              |
   +--------------+---------------+
                  |
                  v
        attention output hidden
                  |
                  v
post_attention_layernorm
                  |
                  v
                 MLP / MoE
                  |
                  v
           本层 hidden_states
```

这里有一个非常关键的概念：

> Qwen3.6 的层不是“每层都完全一样的普通 Attention”。当前模型是一种 hybrid layer 结构，某些层走 full attention，某些层走 linear attention/GDN。

这就是为什么本次融合最终同时覆盖：

```text
full attention:  self_attn.o_proj
linear attention: linear_attn.out_proj
```

它们虽然来自不同 attention 类型，但最后都存在“把 attention 结果投影回 hidden size”的 RowParallel Linear。

---

## 3. full-attention 分支里，`o_proj` 在哪里出现

`AscendQwen3NextAttention.forward` 的主要逻辑是：

```text
hidden_states
  -> qkv_proj
  -> q/k norm + RoPE / MRoPE
  -> attention core
  -> optional gate
  -> o_proj
```

源码最后非常直接：

```python
attn_output = self.attn(q, k, v)
...
out, _ = self.o_proj(attn_output)
```

也就是说，本次融合的前一个数据来源不是“任意 tensor”，而是：

```text
Attention 已经算好的输出 attn_output
```

随后 `o_proj` 做的是一个线性变换。

如果先忽略 TP：

```text
Y = X @ W^T
```

本项目 kernel 采用的内部约定更适合写成：

```text
A[M,K] @ B[K,N] -> C[M,N]
```

当前融合合同固定：

```text
K_local = 2048
N       = 2048
```

所以每张 TP 卡真正接到融合 op 的输入是：

```text
x_rank: [M, 2048]
weight_rank: [2048, 2048]
partial_output: [M, 2048]
```

其中 M 不是模型固定维度，它是“这次 forward 有多少 token 行”。

这点很重要：

```text
K/N 是模型几何
M 是运行时工作量
```

Prefill 的 M 往往较大；Decode / speculative verify 下的 M 可能很小或是 batch×query-token 数。

---

## 4. 为什么 TP=2 后，`o_proj` 不是一个普通 MatMul

当前 `o_proj` 是 Row Parallel Linear。

先用一个非常小的例子理解。

假设原始输入 hidden dimension 是 4：

```text
X = [x0 x1 x2 x3]
```

TP=2 后，输入维度按 K 切开：

```text
rank0 拿 [x0 x1]
rank1 拿 [x2 x3]
```

权重也按对应 K 分片：

```text
W = [ W0 ]
    [ W1 ]
```

每张卡只算自己的部分：

```text
rank0: Y0 = X0 @ W0
rank1: Y1 = X1 @ W1
```

最终完整线性层等价于：

```text
Y = Y0 + Y1
```

因为矩阵乘法对 K 维求和，本来就可以拆成两个 K 分片的部分和。

所以 RowParallelLinear 的本质不是：

```text
两张卡各算一半输出列
```

而是：

```text
两张卡都得到完整 N 列的 partial output，最后在相同位置做 SUM。
```

这解释了为什么当前两张卡上：

```text
partial output 都是 [M, 2048]
```

而不是 `[M,1024]`。

也解释了为什么通信语义是 **AllReduce SUM**。

---

## 5. 为什么模型名是 W8A8，这里却进入 FP16 kernel

这是本项目最容易被误读的地方之一。

目标模型名：

```text
Qwen3.6-35B-A3B-w8a8
```

很容易让人下意识认为：

```text
所有 Linear 的 weight 和 activation 都必须是 INT8
```

实际上 ModelSlim 的量化描述允许某些层被跳过量化。

当前 `modelslim_config.py` 的逻辑是：

```python
if isinstance(layer, LinearBase):
    if self.is_layer_skipped_ascend(prefix, packed):
        ...
        return AscendUnquantizedLinearMethod()
```

而 MemFabric 定制就挂在这个“被跳过量化”的分支里：

```python
eligible = should_enable_memfabric_mm_ar(layer)
if eligible:
    return MemFabricMmArLinearMethod310()
```

所以顺序其实是：

```text
这是 Linear 吗？
  -> 是
它是不是 ModelSlim 跳过量化的层？
  -> 是
它又是不是本次融合目标 o_proj/out_proj？
  -> 是
那就用 MemFabricMmArLinearMethod310
```

也就是说，本次融合不是去改 W8A8 matmul kernel，而是在 W8A8 模型里**命中少数保留 FLOAT 的 projection**。

`memfabric_mm_ar.py` 还继续检查：

```python
layer.params_dtype == torch.float16
```

因此当前融合合同明确是：

```text
A: FP16 ND
B: FP16 NZ
C partial: FP16 ND
MemFabric payload: FP16
local reduction: FP16
```

这也是为什么以后你若想把融合扩到真正的 W8A8 Linear，不能只把 dtype 检查删掉：

你还必须重新回答：

```text
INT8 MM 的输出是什么 dtype？
scale 在哪里？
两个 rank 的 partial 是否可在同一 scale 下直接 SUM？
是先 dequant 再通信，还是量化通信？
误差怎么验证？
```

这已经是另一套设计问题。

---

## 6. `should_enable_memfabric_mm_ar()` 为什么检查这么多条件

很多初学者看到一堆 if，会觉得只是“防御式编程”。

实际上它是在定义一个非常窄的 **kernel ABI 合同**。

当前检查包含：

```text
feature flag 开启
model_type == qwen3_5_moe_text
prefix 是 self_attn.o_proj 或 linear_attn.out_proj
prefix 对应的真实 layer_type 必须匹配
TP size == 2
input_size == 4096
input_size_per_partition == 2048
output_size == 2048
params_dtype == FP16
quant method 必须是 unquantized 路径
```

为什么 `input_size=4096`，但 kernel K=2048？

因为：

```text
全局 Linear K = 4096
TP=2 RowParallel
每 rank K_local = 4096 / 2 = 2048
```

kernel 只看到本 rank 的局部分片，所以 kernel 里的：

```cpp
MF310P_MM_AR_K = 2048
```

完全正确。

这个区别以后一定要形成条件反射：

```text
模型层 global shape != 单 rank kernel local shape
```

---

## 7. 为什么还要根据 `layer_types[layer_idx]` 再校验一次

仅凭后缀：

```text
.self_attn.o_proj
.linear_attn.out_proj
```

理论上已经很具体。

但代码还会：

```python
layer_idx = ...
return layer_types[layer_idx] == expected_layer_type
```

这相当于做“双因素验证”：

```text
名字说你是 full attention o_proj
配置也必须说这一层确实是 full_attention
```

为什么要这么保守？

因为融合 kernel 是极窄 shape/语义特化。万一模型结构、prefix 规则或上游实现以后变化，仅靠名字误命中可能直接造成静默错数。

对于“定制融合算子”，一个重要工程原则就是：

> 宁可漏优化，不要误优化。

漏优化只是回到 stock path；误优化可能返回形状正确但数值错误的 tensor。

---

## 8. `configure_memfabric_mm_ar()` 真正接管了什么

命中以后有一句非常关键：

```python
layer.reduce_results = False
```

这句是整个融合接管的“法律手续”。

原始 RowParallelLinear 的逻辑语义是：

```text
LinearMethod.apply()
   -> 产生本 rank partial output
RowParallelLinear 外层
   -> 如果 reduce_results=True
      再做 tensor_model_parallel_all_reduce
```

而融合路径的 `apply()` 已经返回：

```text
两 rank partial 已经合并好的 final output
```

如果不把 `reduce_results` 关掉，就会变成：

```text
fused MM + reduce
        |
        v
又做一次 generic all-reduce
```

TP=2 情况下结果会接近：

```text
2 * 正确结果
```

所以 `reduce_results=False` 不是性能开关，而是 correctness 所有权转移：

```text
以前：RowParallelLinear 外层负责 reduce
现在：MemFabric custom op 负责 reduce
```

以后做任何算子融合，都要问同样的问题：

> 原来后处理是谁负责？融合后是否出现“双执行”？

---

## 9. `MemFabricMmArLinearMethod310` 为什么继承 `AscendUnquantizedLinearMethod`

它没有重新实现完整权重加载逻辑，而是：

```python
class MemFabricMmArLinearMethod310(AscendUnquantizedLinearMethod):
```

这是一个很好的工程选择。

因为目标层原本就是 unquantized path，原有 method 已经负责：

```text
weight parameter 管理
load 后处理
310P 所需的 NZ format/cast
普通 fallback matmul
```

融合实现只想替换：

```text
apply 阶段的“MM + TP reduction”
```

而不是重写所有 Linear 基础设施。

所以代码：

```python
def process_weights_after_loading(...):
    super().process_weights_after_loading(layer)
    configure_memfabric_mm_ar(layer)
```

先让 stock path 把 weight 处理正确，再挂融合标记。

这体现一个可复用原则：

> 定制代码尽量只接管真正需要改变的最小边界，不要复制整套框架逻辑。

否则上游 weight format、loader、quant router 一变，你的定制代码就很容易漂移。

---

## 10. 一次真实融合调用的 Python 调用链

模型执行到：

```python
self.o_proj(attn_output)
```

如果该 layer 已在 weight loading 后配置成功，可以把调用链理解成：

```text
Qwen Attention.forward
  |
  v
RowParallelLinear.forward
  |
  v
MemFabricMmArLinearMethod310.apply
  |
  | x = x.contiguous()
  | rank = get_tensor_model_parallel_rank()
  v
memfabric_mm_ar_allreduce(layer, x, rank)
  |
  | 检查 dtype / shape / rank
  | 读取 q -> batch_m
  v
torch.ops._C_ascend.memfabric_mm_ar_allreduce(
    x,
    layer.weight.data,
    tp_rank,
    batch_basem_count,
)
```

到这里才离开 Python。

注意 custom op 的参数里**没有显式传 M**。

因为 M 直接来自：

```cpp
num_tokens = x.size(0)
```

也没有传 N/K，因为 ABI v7 当前是极窄定制：

```text
K=N=2048 已经编译进 kernel contract
```

这种写法性能和实现都简单，但可复用性低。

以后若想泛化到其他 hidden size，需要重新考虑：

```text
静态多模板？
运行时 tiling？
kernel cache？
ABI 中是否传 K/N？
哪些 shape 真值得支持？
```

---

## 11. custom op binding 为什么很薄

`memfabric_mm_ar_binding.cpp` 只做 PyTorch 注册：

```text
memfabric_mm_ar_allreduce(Tensor x, Tensor weight, int tp_rank, int batch_basem_count)
```

然后映射到 C++ 实现。

它没有把大量业务逻辑堆在 binding 层，这是对的。

建议以后读自定义算子时固定分三层：

```text
binding: PyTorch 怎么找到这个 op
runtime: op 内部怎么组织资源和 launch
kernel: AI Core 具体怎么算
```

把三层职责混在一起，会很难维护，也很难 Graph 化。

---

## 12. 用一个具体 M=10 的 Decode 例子串起来

假设当前某轮进入这个 `o_proj` 的 token 行数：

```text
M = 10
TP = 2
q = 1
batch_m = 256
```

每个 rank 上：

```text
x_rank      [10, 2048] FP16
weight_rank [2048,2048] FP16 NZ
```

Python 命中融合以后不会先执行 stock matmul。

直接把 x/weight 交给 custom op。

C++ runtime 看到：

```text
num_tokens = 10
10 < min(batch_m=256, small-path ceiling=256)
```

因此后续不会走普通 256-row M-split batch path，而是进入 small-M N-split。

也就是说，模型层代码完全不知道：

```text
这个 M=10 最终在 kernel 里不是按 M 分 8 核，而是按 N 分 8 核
```

这就是好的抽象边界：

```text
模型语义固定
运行时根据 shape 选择不同硬件执行策略
```

---

## 13. 用一个 M=512 的例子串起来

仍然：

```text
q=1 -> batch_m=256
M=512
```

C++ runtime 会得到：

```text
batches = ceil(512 / 256) = 2
```

每个 batch 的 kernel 几何：

```text
A [256,2048]
B [2048,2048]
C [256,2048]
```

8 core 按 M 切：

```text
每 core 32 行
A_core [32,2048]
C_core [32,2048]
B 仍共享完整 [2048,2048]
```

rank0 算出：

```text
C0_batch0
C0_batch1
```

rank1 算出：

```text
C1_batch0
C1_batch1
```

每个 batch 交换以后本地：

```text
out_batch = local_partial + peer_partial
```

最终 output 仍然是模型期待的：

```text
[512,2048]
```

上层 attention 完全不需要知道里面分成了两个 batch。

---

## 14. 这一章最重要的三个“边界”

### 边界一：模型语义边界

```text
Attention -> o_proj -> final hidden
```

模型只关心最终 hidden 正确。

### 边界二：TP 语义边界

```text
本 rank MM partial + TP SUM = final Linear result
```

这是为什么可以把 MM 和 AllReduce 放进同一个 custom op。

### 边界三：硬件执行边界

```text
M 大：M-split batch pipeline
M 小：N-split small path
```

这个选择只属于 runtime/kernel，不应该污染模型层。

能分清这三个边界，你以后就更容易判断：

```text
某个优化应该放模型代码？
放 quant method？
放 runtime？
还是放 AscendC？
```

---

## 15. 自测：不要只回答“是什么”，要回答“为什么”

1. `input_size=4096` 为什么 kernel K 却是 2048？
2. RowParallelLinear 为什么每个 rank 的输出仍然是 `[M,2048]`？
3. 为什么 `reduce_results=False` 是 correctness 必需，而不是性能优化？
4. 为什么融合 method 继承 `AscendUnquantizedLinearMethod`，而不是从零写一个新 Linear？
5. 为什么 W8A8 模型里这个融合仍然是 FP16？
6. M=10 和 M=512 最终为什么会走两套不同 kernel 策略，但模型层不需要知道？
7. 如果未来把 `o_proj` 改成真正 INT8 quantized，当前 ABI 哪些假设会立刻失效？

如果这些问题能从代码路径解释，而不是背结论，就已经开始真正读懂这个项目了。
