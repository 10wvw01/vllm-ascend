# 03｜Qwen3.6 模型、TP 与 W8A8：先把融合发生的数学位置看懂

## 1. 为什么学融合算子之前一定要先看模型

融合算子不是凭空设计出来的。你必须先回答：

```text
前一个算子输出什么？
这一层数学是什么？
后一个算子为什么存在？
TP 怎么切？
权重到底是什么 dtype / format？
```

否则很容易出现“kernel 能跑，但数学语义已经错了”的情况。

本项目的目标是 `Eco-Tech/Qwen3.6-35B-A3B-w8a8`。从融合视角，不需要一开始掌握模型每个细节，但必须理解 Attention、GDN/Linear Attention、MoE、Linear、TP 和量化路由。

---

## 2. 先把一个 Transformer block 想简单

可以先把一层理解成：

```text
hidden_states
   |
   +--> Attention 或 Linear Attention
   |       |
   |       v
   |    output projection
   |       |
   +-------+--> residual
   |
   +--> MoE / FFN
           |
           v
        residual
```

Qwen3.6 这一模型族不是每一层都完全同构。当前融合代码会根据 `text_config.layer_types` 检查当前层究竟是：

```text
full_attention
或
linear_attention
```

对应命中的 projection 分别是：

```text
full attention:   .self_attn.o_proj
linear attention: .linear_attn.out_proj
```

这不是只看名字后缀就盲目融合。代码还会取 layer index，再和模型配置里的 layer type 对上，避免“同名但不是目标语义”的误命中。

源码：

```text
vllm_ascend/_310p/ops/memfabric_mm_ar.py
```

---

## 3. `o_proj` 到底是什么

Attention 前面通常会产生多个 head 的输出。最后要把这些 attention 结果映射回模型 hidden dimension，这一步就是 output projection。

先抽象成：

```text
attention_result [M, K]
        @
W_o              [K, N]
        =
output            [M, N]
```

本项目目标层的全局 shape：

```text
K_global = 4096
N        = 2048
```

在 TP=2 下，本地 K 变成：

```text
K_local = 2048
```

所以每张卡实际做：

```text
[M, 2048] @ [2048, 2048] -> [M, 2048]
```

这就是当前 AscendC kernel 里固定的 K=N=2048 的来源。

不是 kernel 作者“喜欢 2048”，而是模型结构 + TP 切分推导出来的。

---

## 4. 什么是 Tensor Parallel

大模型的一层权重太大或希望两张卡并行计算，可以把同一个矩阵乘拆给多张卡。

常见有 Column Parallel 和 Row Parallel。

### Column Parallel：按输出列拆

简化：

```text
W = [W0 W1]
```

两张卡分别算不同输出列，最后可能 concat。

### Row Parallel：按输入 K 方向拆

本项目的 `o_proj` 重点是这个：

```text
       [W0]
W  =   [W1]

X = [X0 X1]
```

于是：

```text
X @ W
= [X0 X1] @ [W0]
              [W1]
= X0@W0 + X1@W1
```

rank0 算：

```text
Y0 = X0 @ W0
```

rank1 算：

```text
Y1 = X1 @ W1
```

最后：

```text
Y = Y0 + Y1
```

因此 Row Parallel Linear 的后面天然接着一次 reduction。

这就是为什么 `o_proj MM + AllReduce` 是一个很自然的融合候选。

---

## 5. AllReduce 在 TP=2 下到底做什么

AllReduce 是通用集体通信语义：所有 rank 贡献数据，做 reduce，然后每个 rank 都得到结果。

这里是 SUM，因此：

```text
rank0 输入 Y0
rank1 输入 Y1

最终 rank0 得 Y0+Y1
最终 rank1 得 Y0+Y1
```

当 world size 只有 2 时，可以把这个语义更具体地实现成：

```text
rank0 把 Y0 发给 rank1
rank1 把 Y1 发给 rank0

rank0 local add: Y0 + received(Y1)
rank1 local add: Y1 + received(Y0)
```

当前 MemFabric 路径就是利用这个 TP=2 特化，把“通用 collective”变成“peer exchange + local reduce”。

注意：这不是说 HCCL 的数学错或一定慢，而是当前场景足够固定，可以设计更窄、更可控的流水。

---

## 6. W8A8 是什么

W8A8 常见含义：

```text
Weight      8-bit
Activation  8-bit
```

目的主要是：

- 减少权重带宽；
- 降低显存/内存占用；
- 利用硬件 INT8 计算吞吐。

但一个量化 checkpoint 并不代表**所有层都必须量化**。

某些层可能因为：

- 精度敏感；
- 算子支持；
- 模型量化配置；
- 收益有限；

被明确保留为 FLOAT。

---

## 7. 为什么这个 W8A8 模型的融合却是 FP16

这是本项目最容易产生误解的地方。

当前 310P ModelSlim 路由会先判断一个 Linear 是否被量化配置 skip：

```text
is_layer_skipped_ascend(...)
```

如果没有 skip，走正常量化 scheme：

```text
AscendLinearMethod(scheme)
```

如果被 skip，则默认走：

```text
AscendUnquantizedLinearMethod
```

而本次目标的 `o_proj/out_proj` 正是在这种未量化路径中，再进一步判断是否满足 MemFabric 融合合同：

```text
if eligible:
    MemFabricMmArLinearMethod310
else:
    AscendUnquantizedLinearMethod
```

所以路线是：

```text
整个 checkpoint：W8A8 为主
        |
        +--> 大部分量化 Linear：INT8/量化路径
        |
        +--> 某些 skip 的 FLOAT Linear
                  |
                  +--> 目标 o_proj/out_proj
                            |
                            v
                       FP16 MM+AR
```

当前代码还明确要求：

```text
layer.params_dtype == torch.float16
x.dtype == torch.float16
```

因此本次融合并不是“W8A8 MM+AR kernel”，而是“W8A8 模型中特定未量化 projection 的 FP16 MM+AR”。

这个表述以后和别人讨论时一定要准确。

---

## 8. 为什么权重要转 NZ

昇腾矩阵计算经常使用适合 Cube/矩阵单元的数据布局，例如 FRACTAL_NZ。

可以粗略理解成：

```text
人类直观看的普通二维连续矩阵 ND
      ↓ 重新排布
硬件矩阵单元更喜欢的分块格式 NZ
```

这不是改变数学值，而是改变内存里的摆放方式。

当前融合 kernel 的 Matmul 类型里：

```text
A: ND FP16
B: NZ FP16, transpose flag
C: ND FP16
```

`MemFabricMmArLinearMethod310` 继承 `AscendUnquantizedLinearMethod` 的一个重要原因，就是复用既有未量化层的正确权重加载和 NZ 处理，而不是重新复制一套权重转换逻辑。

这是很典型的工程设计原则：

> 只替换自己必须改变的执行部分，权重语义和已有稳定处理尽量复用。

---

## 9. 为什么 full attention 和 GDN 可以共用一个 MM+AR kernel

虽然前面的 attention 算法不同：

```text
full attention
vs
linear attention / GDN
```

但到了最终 projection，这两个目标层恰好共享当前融合所需的合同：

```text
TP=2
K_global=4096
K_local=2048
N=2048
FP16
无 bias
RowParallel reduction
```

于是从融合 op 视角看，它们已经“长得一样”。

这是一条很重要的优化方法：

> 不要按模型层名字判断能否复用 kernel，而要抽取真正影响 kernel 的 shape、dtype、layout、parallel semantics 合同。

如果两个层在这些合同上完全一致，就有机会共用同一实现。

---

## 10. MoE 又是什么，和本次融合有什么关系

MoE = Mixture of Experts。

简单理解：不是每个 token 都经过同一个 FFN，而是 router 为 token 选择少量 expert：

```text
token
  |
router
  |
  +--> expert 3
  +--> expert 17
  ...
```

MoE 的主要性能问题通常包括：

- routing；
- token dispatch/combine；
- expert matmul；
- EP 通信；
- 小 batch expert 利用率。

本次 MM+AR 融合并不直接融合 MoE。它优化的是 attention/linear-attention 后面的 RowParallel projection。

但在端到端优化时，两者会相互影响：如果你把 o_proj 优得非常快，模型瓶颈可能自然转移到 MoE。优化永远要重新看完整 timeline。

---

## 11. 当前融合资格检查为什么这么严格

`should_enable_memfabric_mm_ar()` 不是只判断 feature flag，而是锁定：

```text
model_type
layer prefix + layer_types
TP size == 2
input global == 4096
input local == 2048
output == 2048
params dtype == FP16
unquantized linear method
```

为什么不“尽可能多地支持 shape”？

因为第一版高性能定制最怕把语义边界做模糊。

窄合同的好处：

- 容易证明正确；
- kernel tiling 可完全针对固定 shape；
- 测试空间小；
- 出问题容易 fail fast；
- 不会错误接管其它层。

等证据充分以后再扩合同，比一开始做“万能融合 op”更稳妥。

---

## 12. 如果以后想做量化 MM+AR，要重新考虑什么

当前 FP16 方案不能直接把类型改成 INT8 就算完成。

至少要重新回答：

```text
1. 输入 activation 的量化 scale 在哪里？
2. MM accumulator 是 INT32 还是其它类型？
3. partial output 在跨卡前是什么 dtype？
4. 两个 rank 的 scale 是否相同，能否直接整数域相加？
5. 先 dequant 再传，还是压缩传输？
6. 精度误差是否可接受？
7. NZ/量化权重 layout 如何适配？
8. 通信节省是否大于新增量化/反量化开销？
```

这就是“模型量化”和“计算通信融合”交叉以后复杂度会迅速增加的原因。

---

## 13. 本章自测

你应该能解释：

1. Row Parallel 为什么是 K 方向拆分并最终做 SUM？
2. 当前每个 rank 的 MM shape 为什么是 `[M,2048] @ [2048,2048]`？
3. 为什么一个叫 W8A8 的 checkpoint 仍然能存在 FP16 Linear？
4. ModelSlim 路由如何把目标未量化层再切换成 MemFabric method？
5. full attention 和 GDN 为什么前面算法不同，却可以共用当前 MM+AR kernel？
6. 为什么融合资格检查越窄，第一阶段通常越容易做对？

下一章开始看 310P 定制体系是怎样“插进”原本 vLLM-Ascend 的。
