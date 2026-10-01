# 14｜源码精读：310P 上一次 DFlash speculative decode 到底怎么跑

> 这一章不讲“投机推理就是小模型猜、大模型验”这种一句话解释，而是直接跟当前源码，回答一次 DFlash round 里到底有哪些 tensor、为什么要重新构造输入、为什么 310P 要单独做 block table / slot mapping / Graph 适配。
>
> 建议同时打开：
>
> - `vllm_ascend/spec_decode/dflash_proposer.py`
> - `vllm_ascend/_310p/spec_decode/dflash_proposer_310.py`
> - `vllm_ascend/patch/worker/patch_idex_310.py`
> - `vllm_ascend/_310p/model_runner_310p.py`

---

## 1. 先把一次 speculative round 画成“数据生产链”

普通 autoregressive decode：

```text
target model
  -> 预测 1 token
  -> 再跑一次
  -> 再预测 1 token
```

假设 target 一次 decode 要 1 ms，生成 8 token 大概要做 8 次完整 target forward。

投机推理想把它改成：

```text
draft 先便宜地猜 K 个 token
        |
        v
target 一次性验证这 K 个候选
        |
        v
接受连续正确前缀
        |
        v
下一轮
```

但真正工程上的关键不是“猜”这个词，而是：

> 怎样把 K 个候选组织成一次 target 能验证的输入，同时正确维护 KV Cache、position、slot mapping、采样位置和被拒绝 token 的状态。

当前 310P DFlash 的大量代码，恰恰都在做这些“输入和状态的重新组织”。

---

## 2. `num_speculative_tokens` 和真正 query 长度不是同一个数

当前 310P 代码明确限制：

```python
MAX_SUPPORTED_NUM_SPEC_TOKENS_310P = 15
```

注释解释原因：310P recurrent GDN kernel 支持最大 query length 16，而 DFlash verification 还会包含一个 bonus token。

因此：

```text
最大 draft token K = 15
真正 verify query = 1 + K <= 16
```

共享 proposer 里也直接写：

```python
num_query_per_req = 1 + self.num_speculative_tokens
```

这就是为什么代码里经常同时出现：

```text
num_speculative_tokens
num_query_per_req
```

不要把两者当成同义词。

例如：

```text
K = 6
num_query_per_req = 7
batch_size = 10
num_query_total = 70
```

这 70 行 query 才是 draft/verify 相关 kernel 实际要处理的 query token 数。

---

## 3. DFlash 为什么还需要 `context` 和 `query` 两类 token

共享 `AscendDflashProposer.set_inputs_first_pass()` 的注释已经说得很清楚：

```text
DFlash cross-attention:
context K/V from target hidden states
Q from query embeddings (bonus + mask tokens)
```

先用人话说：

DFlash 不是简单把 target 的全部输入原封不动再跑一个“小模型”。它要把 target 已经算出来的 hidden states 当成 context 信息，再构造本轮要生成/验证的 query token。

因此第一轮会同时有：

```text
context:
  target 已有 hidden states

query:
  每个 request 的 bonus + speculative token positions
```

共享代码：

```python
num_context = target_token_ids.shape[0]
num_query_per_req = 1 + self.num_speculative_tokens
num_query_total = batch_size * num_query_per_req

self._dflash_hidden_states[:num_context] = target_hidden_states
```

这意味着：

```text
context 数量由 target 本轮已有 token 数决定
query 数量由 batch_size × (1+K) 决定
```

两者完全不是同一个维度。

---

## 4. 为什么要有 `set_inputs_first_pass()`

普通模型 runner 已经有一套：

```text
input_ids
positions
query_start_loc
seq_lens
slot_mapping
block_table
```

为什么 DFlash 还要重新做？

因为 draft 模型看到的逻辑序列与 target 当前 forward 的逻辑序列不完全相同。

例如 target 这一轮也许每个 request 只刚产生 1 个新 token，但 DFlash 想构造：

```text
1 个 bonus + K 个 speculative query
```

所以必须把原来的 metadata 改写成 draft 模型能理解的新 batch。

共享实现里 `set_inputs_first_pass()` 会重新设置：

```python
cad.query_start_loc = new_query_start_loc
cad.seq_lens = effective_seq_lens + num_query_per_req
cad.num_actual_tokens = num_query_total
cad.max_query_len = num_query_per_req
cad.slot_mapping = query_slot_mapping
cad.causal = False
cad.attn_state = AscendAttentionState.ChunkedPrefill
```

这些不是“辅助字段”，而是在重定义 draft attention 这一轮看到的世界。

---

## 5. 为什么 DFlash 这里把 attention state 设成 `ChunkedPrefill`

直觉上这是 decode，为什么不是普通 decode state？

因为从 attention kernel 的角度看，本轮不是每个 request 只有 1 个 query token。

而是：

```text
每 request 一次处理 1+K 个 query
```

它更像一个短的 multi-token query chunk。

所以代码把：

```python
cad.attn_state = AscendAttentionState.ChunkedPrefill
```

这说明一个很重要的学习方法：

> “业务上叫 decode”不代表底层 attention backend 一定使用名为 Decode 的执行形态。

底层真正关心的是 tensor 结构和 query length。

---

## 6. 其他平台为什么用 Triton，310P 为什么要换 AscendC

共享实现直接调用：

```python
copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[...]()
```

它来自：

```text
vllm_ascend.ops.triton.spec_decode.utils
```

但 310P 定制文件顶部明确写：

```text
310P has no Triton
```

所以 310P 把这一步替换成：

```python
torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs(...)
```

对应的是 AscendC custom op。

这就是典型的“算法相同、硬件实现不同”：

```text
算法需求：构造 DFlash query/context inputs
其他平台：Triton kernel
310P：AscendC custom op
```

因此学习 310P 适配时，不应该把它理解成“华为重写了一套投机推理算法”，而是：

```text
共享 speculative decode 语义
+
310P 专属的数据准备 / cache / graph / kernel 实现
```

---

## 7. `slot_mapping` 到底是什么，为什么它会成为 310P 大坑

先用最简单的 KV Cache 地址模型理解。

假设 KV Cache 按 block 分页：

```text
block_size = 64 token
```

某个逻辑位置：

```text
position = 130
```

那么：

```text
逻辑 block index = 130 // 64 = 2
block 内 offset   = 130 % 64  = 2
```

`block_table[request, 2]` 告诉你这个逻辑 block 实际映射到了哪个物理 block。

如果物理 block id = 17：

```text
slot = 17 * 64 + 2
```

这个 slot 就是 KV cache 中这个 token 应该读写的位置。

当前 310P 代码里的核心公式就是：

```python
block_ids * block_size + positions.remainder(block_size)
```

所以：

> slot_mapping 本质上是“逻辑 token position -> KV cache 物理槽位”的地址翻译结果。

如果这个映射错了，程序通常不会直接 crash，而是会去读另一个合法地址里的错误 K/V。

这类 bug 极其危险，因为 shape 全都对，但 attention 数值错。

---

## 8. 为什么当前 310P 会出现 128 vs 64 的 block-size 问题

310P 定制代码专门记录了一个真实问题：

共享 proposer 可能把：

```text
kernel_block_size = 128
```

当成默认值。

但实际分配出来的 draft KV Cache 可能使用：

```text
64
```

例如 draft spec 的一个大 block 被进一步拆成 kernel 使用的小 block。

如果你用 128 去算：

```text
slot = block_table[pos//128] * 128 + pos%128
```

但真实 KV cache 是按 64 组织，得到的 slot 就不是实际 cache 地址。

代码注释给出的后果非常严重：

```text
SplitFuse cross-attention 读空 block
输出接近全 0
acceptance 接近 0
```

所以 `_ensure_kernel_block_size_matches_cache_310()` 会从真实已分配 cache tensor 的 shape 反查 physical block size，并修正：

```python
proposer.kernel_block_size = cache_block_size
```

这不是“调个参数提高性能”，而是修正**地址语义**。

---

## 9. 为什么不同 draft layer 甚至可能有不同 block size

代码 `_draft_cache_block_sizes_310()` 没有返回一个 int，而是：

```python
dict[layer_name, block_size]
```

注释给出实际场景：某些 hybrid draft model 前几层可能用 64，后几层用 128。

因此不能偷懒写成：

```text
整个 draft 模型只有一个 block_size
```

否则：

```text
适用于 layer0 的 slot mapping
未必适用于 layer5
```

310P 于是维护：

```text
per-layer block table
per-layer query slot mapping
per-layer context slot mapping
```

这就是为什么 speculative decode 工程代码看起来比“draft + verify”复杂几个数量级：

真实系统里每一个候选 token 都必须落到正确 cache 物理地址。

---

## 10. `_convert_block_table_layout_310()` 在做什么

假设 physical page 能容纳更大的物理区，而不同 kernel 希望用不同逻辑 block size。

代码要求：

```text
physical_block_size % source_block_size == 0
physical_block_size % target_block_size == 0
```

然后把同一个 physical page 重新展开成不同数量的 logical block id。

例如为了帮助理解，假设：

```text
physical block = 256 token
source logical block = 128
source ratio = 2

target logical block = 64
target ratio = 4
```

一个物理 page 在 source table 里可能对应：

```text
[10, 11]
```

转成 target 逻辑布局后可能需要表达成：

```text
[20,21,22,23]
```

核心不是数字本身，而是：

```text
同一块物理 KV 内存
不同 kernel 按不同 block 粒度索引
```

所以要转换的是“索引视图”，而不是重新搬一份 KV 数据。

---

## 11. 为什么 context slot 还要再单独重算一次

这是一个非常值得学习的真实 bug 修复思路。

310P AscendC input builder 会重新计算 query slots，但 context slots 可能直接沿用 `cad.slot_mapping`。

如果 `cad.slot_mapping` 是按 128 block 构造，而 draft cache 实际是 64：

```text
query slot = 64 语义
context slot = 128 语义
```

结果 cross-attention 的 Q 和 context K/V 实际根本没有落在同一套物理地址体系。

因此 `_recompute_context_slots_310()` 强制用与 query 完全相同的：

```text
block_table + kernel_block_size + position
```

重新计算 context slot。

这是一个非常典型的 correctness 推理：

```text
只修 query 不够
因为 cross-attention 两侧必须对同一个 cache layout 达成一致
```

学会这种“地址合同一致性”思维，比记函数名重要得多。

---

## 12. 为什么有一条 int32 地址数学路径

代码里：

```python
_uses_int32_draft_address_math_310(...)
```

会在某些 310P Graph 路径下避免 dynamic int64 address arithmetic。

人话解释：

有些表达式数学上：

```text
int64 当然更大、更安全
```

但特定硬件 / Graph / 动态地址生成路径上，int64 可能触发不对齐、额外 kernel 或不支持的执行行为。

于是只要确认：

```text
真实 cache 地址索引范围不会超过 int32
```

就可以选择 int32 地址运算来规避硬件问题。

这提醒我们：

> dtype 不只是“数值精度问题”，在系统代码里还可能决定地址计算、Graph 可捕获性和 kernel 支持路径。

---

## 13. `token_indices_to_sample` 为什么要重新构造

Draft 模型通常一次生成/输出多个位置，但最后不是所有 logits 都拿来采样。

需要告诉后处理：

```text
从这一大块输出里，哪些 token 位置对应我们真正要取的 speculative proposal
```

所以代码分配：

```python
token_indices_to_sample = torch.empty(
    batch_size * self.num_speculative_tokens,
    dtype=torch.int32,
)
```

注意它的数量是：

```text
batch_size * K
```

而不是：

```text
batch_size * (1+K)
```

因为那个额外 bonus query 并不是 K 个 speculative token 之一。

这也是为什么理解 tensor 数量比背流程更重要。

---

## 14. 被 reject 的 token 为什么影响下一轮 seq_len

共享代码：

```python
if has_num_rejected:
    effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
```

原因是 speculative proposal 并不是全部都会正式进入最终序列。

假设上一轮：

```text
猜 6 个
只接受 3 个
```

另外 3 个如果已经临时进入某些 draft 状态，就必须在逻辑 seq_len 上扣回去。

否则下一轮会把“已被 target 拒绝的 token”当成真实历史。

所以 speculative decode 最重要的不变量之一是：

```text
最终序列状态 = target 已确认接受的 token
```

所有 draft 临时状态都必须服从这个事实。

---

## 15. 为什么 DFlash 要 `precompute_and_store_context_kv`

共享代码 `build_model_inputs_first_pass()`：

```python
self.model.precompute_and_store_context_kv(
    self._dflash_hidden_states[:num_context],
    self._context_positions_buffer[:num_context],
    self._context_slot_mapping_buffer[:num_context],
)
```

它把 target hidden states 转成 draft cross-attention 需要的 context K/V，并写到 draft cache 对应槽位。

可以理解成：

```text
target 已经替我们算出了很贵的上下文表征
          |
          v
DFlash 把这些表征加工成自己的 context KV
          |
          v
后续 draft query 直接 cross-attend 这些 context KV
```

因此 DFlash 的“便宜”并不是凭空出现，它通过复用 target 已有信息，减少 draft 自己重新建模全部 context 的成本。

---

## 16. Graph 为什么让这一切更复杂

如果只做 eager，每轮 Python 可以：

```text
算 shape
建 metadata
分配临时 tensor
调用 kernel
```

Graph replay 希望很多东西地址和执行序列固定。

因此当前代码预先分配大量 persistent buffer：

```text
_context_slot_mapping_buffer
_slot_mapping_buffer
_context_positions_buffer
positions
_dflash_hidden_states
各种 per-block-size buffer
```

目的不是代码风格，而是：

```text
把动态创建
变成固定地址上的原地更新
```

Graph 优化的核心思想可以记成：

> replay 时最好只更新“内容”，不要改变“对象和地址结构”。

这也是为什么后续 MM+AR custom op 也要求 Graph capture 前完成 context / scratch / warmup。

---

## 17. 为什么 dummy/profile run 也必须和真实路径一致

很多 Graph bug 都来自：

```text
真实请求走 A
capture dummy run 走 B
```

结果图捕获到的根本不是线上执行形态。

310P DFlash 为 dummy run 专门做：

```text
RoPE flag
full graph buffer binding
hybrid capture scope
metadata owner 对齐
```

原因就是：

```text
Graph 捕获期使用的 tensor / metadata / kernel 路径
必须和 replay 期兼容
```

这不是单纯“预热一下算子”。

---

## 18. 用 batch=2、K=3 做一轮手工推演

假设：

```text
batch_size = 2
num_speculative_tokens K = 3
```

于是：

```text
num_query_per_req = 1 + 3 = 4
num_query_total = 2 * 4 = 8
```

逻辑上可以想成：

```text
request0: [bonus, q1, q2, q3]
request1: [bonus, q1, q2, q3]
```

新的：

```text
query_start_loc = [0,4,8]
```

`token_indices_to_sample` 只需要 6 个 speculative token：

```text
request0: q1 q2 q3
request1: q1 q2 q3
```

因此 size：

```text
2 * 3 = 6
```

每个 query position 再根据：

```text
position
block_table
kernel_block_size
```

算出它的 KV slot。

这 8 个 query 交给 draft forward 后，产生 proposal logits；随后 target 对 proposal 进行 verify；accept/reject 结果又反馈成下一轮的 rejection/seq_len 状态。

这才是一轮 speculative decode 的真正闭环。

---

## 19. 为什么“acceptance rate”不能单独代表 DFlash 是否优化成功

假设两种方案：

```text
A: acceptance 90%，draft 每轮 1.0 ms
B: acceptance 75%，draft 每轮 0.2 ms
```

A 不一定更快。

真正应该看：

```text
每个 target verify round 平均接受几个 token
÷
整个 round 的总耗时
```

近似可以写：

```text
effective tokens/s
≈ E[accepted_tokens_per_round]
  / (T_draft + T_prepare + T_target_verify + T_accept)
```

而 310P 代码里这些 block-table、input builder、Graph、GDN 优化，很多就是在降低：

```text
T_prepare
T_draft
```

所以做 speculative decode 性能优化时，不能只盯 acceptance。

---

## 20. 你以后排 DFlash bug，应按什么顺序

### 第一步：先看 proposal 是否有意义

```text
logits 是否全 0？
proposal token 是否异常固定？
```

### 第二步：如果输出异常，先查地址而不是先怀疑模型

```text
block_size
block_table
slot_mapping
position
context/query 是否用同一 layout
```

### 第三步：再查 Graph/eager 差异

```text
仅 Graph 错？
仅 replay 错？
地址是否固定？
dummy run 和真实 run metadata 是否一致？
```

### 第四步：最后才看 acceptance/rejection 数学

因为如果前面的 KV 地址已经错，后面 rejection sampler 再正确也没有意义。

---

## 21. 本章自测

1. 为什么 `K=6` 时 query length 是 7？
2. DFlash 的 context 和 query 分别来自哪里？
3. 为什么业务上是 decode，attention state 却可能是 `ChunkedPrefill`？
4. 为什么 128/64 block-size 不一致会导致 acceptance 接近 0？
5. `slot_mapping = block_id * block_size + offset` 里的每一项分别是什么？
6. 为什么 query slot 修对了还不够，context slot 也必须用同一布局重算？
7. 为什么 per-layer draft KV cache 可能需要不同 block table？
8. Graph 为什么要求 persistent buffer？
9. 为什么 acceptance rate 高不代表端到端一定快？

把这些问题讲清楚以后，你才算真正理解“310P 上的 DFlash 适配”是在解决什么，而不是只知道有个 `spec_decode/` 目录。
