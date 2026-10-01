# 05｜投机推理与 DFlash：为什么“先猜几个”反而能更快

## 1. 普通 Decode 慢在哪里

普通自回归生成有一个天然限制：

```text
第 n+1 个 token 依赖第 n 个 token
```

因此最朴素的 target model decode 是：

```text
跑一次大模型 -> 生成 1 token
跑一次大模型 -> 再生成 1 token
跑一次大模型 -> 再生成 1 token
...
```

即使一次 target forward 已经优化得很好，生成 1000 个 token 仍然要经历大量串行轮次。

投机推理想解决的不是“单次矩阵乘更快”，而是：

> 能不能让一次昂贵的 target model forward，确认多个 token？

---

## 2. 最直观的投机推理

假设有一个更便宜的 draft 模型。

它先猜：

```text
A B C D
```

然后 target model 一次性验证这些位置。

如果 target 判断：

```text
A 对
B 对
C 对
D 错
```

那这一轮可以一次接受：

```text
A B C
```

而不是只推进 1 个 token。

所以性能收益来自：

```text
更便宜的 draft 成本
+
一次 target verify 的成本
<
逐 token 跑多次 target 的成本
```

---

## 3. 三个角色：Propose、Verify、Accept/Reject

### Propose

Draft proposer 提出若干候选 token。

### Verify

Target 模型对这些候选位置计算自己的概率/logits。

### Accept / Reject

根据投机算法判断 draft token 能否接受。

一旦某一位置不再接受，通常后面的 draft suffix 也不能原样保留，因为自回归条件已经变化。

所以可以把一轮理解为：

```text
已有上下文
   |
Draft：猜 K 个 token
   |
Target：一次验证 K 个位置
   |
接受最长合法 prefix
   |
可能再得到一个 target bonus token
   |
进入下一轮
```

---

## 4. 为什么不是 K 越大越好

假设 draft 一次猜 15 个，但经常第 2 个就错：

```text
猜了 15 个
真正接受 1~2 个
```

后面很多 draft 和 verify 工作就浪费了。

反过来 K 太小：

```text
每轮只猜 2 个
虽然命中率高，但摊不薄 target 的固定开销
```

因此 spec decode 的收益取决于至少四件事：

```text
1. draft 有多便宜
2. draft 的接受率有多高
3. target 一次 verify K 个 token 有多贵
4. K 对 Graph / kernel shape / KV 管理的额外成本
```

真正调优时要看“accepted tokens per target step”，不能只看 draft 吞吐。

---

## 5. DFlash 在这个仓库里是什么

仓库通用 speculative decoding 代码在：

```text
vllm_ascend/spec_decode/
```

可以看到多种 proposer：

```text
dflash_proposer.py
dspark_proposer.py
eagle_proposer.py
medusa_proposer.py
ngram_proposer.py
...
```

310P 又有一套平台特化：

```text
vllm_ascend/_310p/spec_decode/
```

当前主要包括：

```text
dflash_model_310.py
dflash_mrope.py
dflash_proposer_310.py
dflash_vocab.py
llm_base_proposer_310.py
```

这里最重要的理解不是先背 DFlash 论文细节，而是：

> 上层仍然遵守 vLLM speculative decode 的 proposer/verify 协议，但 310P 对 draft 输入构造、KV、position、Graph 和设备算子做了平台化重写。

---

## 6. 为什么 310P 不能直接照搬其它平台 DFlash

`dflash_proposer_310.py` 文件开头就给了非常关键的信息：

```text
310P 没有通用实现所依赖的 Triton 路径
```

因此原来用 Triton 做的：

```text
copy_and_expand_dflash_and_dspark_inputs...
```

在 310P 上换成 AscendC custom op：

```text
npu_copy_and_expand_dflash_inputs
```

这是一个非常典型的平台适配案例：

```text
算法语义没变：还是要构造 draft inputs
实现工具变了：Triton -> AscendC
```

所以学平台适配时，不要把“算法不同”和“实现手段不同”混在一起。

---

## 7. 为什么投机推理会把 KV Cache 搞复杂

普通 decode 的新 token 位置比较直观：

```text
position = 已有长度
```

但 draft 一次可能扩出 K 个候选位置。

此时系统要同时知道：

```text
每个候选属于哪个 request？
绝对 position 是多少？
应该写到哪个 KV block？
物理 slot 是多少？
验证失败以后哪些数据还能保留？
```

所以 `_310p/spec_decode/dflash_proposer_310.py` 里会出现大量：

```text
block table
physical block size
slot mapping
position
request id
layout conversion
```

这些看起来不像“AI 算法”，但它们决定 draft/verify 是否读到正确上下文。

KV 地址错一位，模型也许不会 crash，却会产生完全错误的 token，这类 bug 非常难查。

---

## 8. 为什么同一个 draft cache 里还可能有不同 block size

当前 310P DFlash 代码专门处理了：

```text
不同 attention layer 的真实 KV cache 物理 block size 可能不同
```

也就是说不能粗暴假设：

```text
全模型所有 draft KV layer 都用一个 block_size
```

代码会按 layer 查询真实 cache shape，再建立对应 slot。

这是一个很值得学习的工程原则：

> 配置里的逻辑 block size，不一定等于 kernel 最终看到的物理布局。

性能框架经常会为了 kernel 对齐、分组或硬件约束重新切块。做底层优化时一定要看“真正分配出来的 tensor”。

---

## 9. 310P 为什么最多 15 个 speculative tokens

当前代码定义：

```text
MAX_SUPPORTED_NUM_SPEC_TOKENS_310P = 15
```

原因不是随便限制。

代码注释给出的硬件/算子约束是：

```text
310P recurrent GDN kernel
最大 recurrent query length = 16
```

而 target verification 还要额外包含一个 bonus token，因此：

```text
1 bonus + K draft <= 16
=> K <= 15
```

这类约束特别值得学习，因为它体现：

```text
算法参数上限
不是来自算法论文
而是由下游 kernel buffer/shape contract 决定
```

如果未来 GDN kernel 支持更长 query，这个上限才有重新评估空间。

---

## 10. 什么是 bonus token

先用直觉理解。

如果 target 验证 draft 候选后发现前 K 个都能接受，系统仍然希望 target 给出“再往后一个”的 token，避免下一轮完全从空开始。

这个由 target 额外提供的 token 常被称作 bonus token。

因此 verify query 不是简单“只有 K 个 draft token”，而可能有：

```text
K speculative tokens + 1 target bonus position
```

这就是很多 shape 和 buffer 限制里出现 `K+1` 的原因。

---

## 11. 为什么 mRoPE 也要单独处理

Position Encoding 告诉模型“这些 token 在序列中的位置”。

投机推理一次创建多个未来位置，而且某些模型使用更复杂的 RoPE/mRoPE 规则。

所以 310P 有：

```text
vllm_ascend/_310p/spec_decode/dflash_mrope.py
```

你可以把它理解成：

```text
普通 decode：位置一格一格往后走
DFlash：一次临时构造一小段未来位置
```

如果 position 和 KV slot 不一致，即使 token id 对，attention 语义也会错。

---

## 12. 为什么投机推理和 Graph 关系很深

投机推理让本轮 shape 变得更多样：

```text
请求数不同
K 不同
accepted length 不同
full attention / GDN 不同
```

而 Graph 喜欢稳定 shape。

所以 310P 分支里有很多：

```text
dflash_full.py
dflash_piecewise.py
dflash_full_and_piecewise.py
dflash_full_decode_contract.py
...
```

这些文件本质是在解决：

> 怎样把动态的 speculative decode 工作负载，整理成 310P 上可以稳定捕获/重放或分段捕获的执行形态。

“FULL / PIECEWISE” 不要先死记。先记住 Graph 的根矛盾：

```text
服务输入动态
vs
Graph 希望执行图稳定
```

各种模式都是在两者之间找平衡。

---

## 13. 投机推理会怎样影响 MM+AR

虽然 MM+AR 和 DFlash 是两个不同子系统，但端到端上它们会碰到一起。

原因：投机 verify 会改变 Linear 看到的 `M`。

普通 decode 可能：

```text
M ≈ 活跃请求数
```

DFlash target verify 可能一次为每个请求处理多个候选位置：

```text
M 变大，并且离散 shape 更多
```

这会影响 MM+AR：

- 是走 small-M N-split 还是 batch path；
- tail 比例有多大；
- q=1/2/4 哪个更合适；
- Graph 需要 capture 哪些 shape；
- MM/SDMA overlap 能否充分展开。

所以不能分别优化完 DFlash 和 MM+AR 就假设端到端一定最优。

必须测组合场景。

---

## 14. 怎样判断投机推理到底赚没赚

不要只看“每秒 draft token”。建议至少记录：

```text
accepted_tokens / speculative_step
acceptance_rate
speculative K
Draft time
Target verify time
Rejection/sampling time
每个最终输出 token 的平均设备时间
端到端 ITL / throughput
```

可以建立一个很粗的判断模型：

```text
平均每轮接受 A 个 token
一轮总成本 = T_draft + T_verify + T_other

每个有效 token 成本 ≈
(T_draft + T_verify + T_other) / A
```

只有这个值低于普通 target decode 每 token 成本，投机推理才真正带来收益。

---

## 15. 读 DFlash 源码的推荐顺序

不要直接从 4 万行级别 proposer 细节开始。

推荐：

```text
1. vllm_ascend/spec_decode/dflash_proposer.py
   先看通用语义

2. vllm_ascend/_310p/spec_decode/__init__.py
   看 310P 暴露了哪些替代能力

3. vllm_ascend/patch/worker/patch_idex_310.py
   看 310P 版本怎么接进去

4. vllm_ascend/_310p/spec_decode/dflash_proposer_310.py
   再看输入/KV/Graph 特化

5. model_runner_310p.py
   最后看它怎样参与整轮 execution
```

读每个函数时始终问：

```text
它是在改变 speculative 算法？
还是只是在 310P 上实现同一个协议？
```

---

## 16. 本章自测

1. 投机推理为什么有机会一次推进多个 token？
2. K 为什么不是越大越好？
3. Draft、Target verify、Accept/Reject 各做什么？
4. 310P DFlash 为什么要把 Triton 输入构造换成 AscendC？
5. 为什么 speculative decode 会让 KV slot 管理更复杂？
6. 为什么当前最多 15 个 draft token？
7. DFlash 为什么可能改变 MM+AR 最常见的 M 分布？

下一章正式进入本项目核心：**为什么偏偏选择 o_proj 后的 MM+AR 做融合，而不是随便挑两个相邻算子。**
