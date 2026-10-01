# 08｜从 Python 到 AscendC：沿着一次 MM+AR 调用逐层读代码

这一章只做一件事：给你一条稳定的源码阅读路线。以后代码更新，也可以沿同样方法重新追。

## 1. 总调用链先画出来

```text
ModelSlim quant route
  |
  v
MemFabricMmArLinearMethod310.apply
  |
  v
memfabric_mm_ar_allreduce(...)
  |
  v
torch.ops._C_ascend.memfabric_mm_ar_allreduce
  |
  v
binding.cpp
  |
  v
memfabric_mm_ar_allreduce_impl
  |
  +--> adapter API
  |      |
  |      +--> MemFabric host public APIs
  |      +--> AscendC kernel launch
  |
  +--> direct producer / small producer
  +--> wait
  +--> add
  +--> quiet / ack / gate
```

读代码时不要横向漫游，先把这一条追通。

---

## 2. 第一站：`modelslim_config.py` —— 为什么这个 Linear 会被接管

文件：

```text
vllm_ascend/_310p/quantization/modelslim_config.py
```

先找 Linear 的 quant method 选择逻辑。

核心思路可以简化成伪代码：

```python
if module is LinearBase:
    if this layer is skipped by quant config:
        if should_enable_memfabric_mm_ar(...):
            return MemFabricMmArLinearMethod310()
        return AscendUnquantizedLinearMethod()
    else:
        return AscendLinearMethod(quant_scheme)
```

你应该从这里确认三件事：

### 1. 本次目标层先是“未量化层”

所以融合 method 的基类是 unquantized method。

### 2. 融合是正常 method selection 的一个分支

不是在模型 forward 里硬改。

### 3. feature 不满足时自然回到 stock 路径

这给了清晰的 A/B baseline。

---

## 3. 第二站：`memfabric_mm_ar.py` —— 所有资格条件集中在哪里

文件：

```text
vllm_ascend/_310p/ops/memfabric_mm_ar.py
```

先不要从头逐行读。按下面顺序找函数。

### `should_enable_memfabric_mm_ar(...)`

这是路由总闸门。

你要检查它验证了：

```text
feature flag
model_type
layer prefix
真实 layer type
TP=2
input/output size
FP16 params
unquantized method
```

学习重点：**任何一个高风险专用 kernel 都应该有类似的窄合同 guard。**

### `configure_memfabric_mm_ar(...)`

重点看：

```text
assert layer.reduce_results is True
layer.reduce_results = False
```

它告诉你融合 op 已经接管了 reduction 语义。

### `MemFabricMmArLinearMethod310`

重点看两个生命周期：

```text
process_weights_after_loading
apply
```

前者把 layer 配成 fused；后者真正决定调用 fused custom op 还是 warmup fallback。

### `memfabric_mm_ar_allreduce(...)`

这里是 Python 到 C++ 的边界。

它还会检查输入：

```text
FP16
2D
shape[-1] == 2048
rank 0/1
```

然后调用：

```text
torch.ops._C_ascend.memfabric_mm_ar_allreduce(...)
```

---

## 4. 为什么 Python 层还要重复做 shape 检查

你会发现 C++ 里也有检查。

这不是完全多余。

Python 检查的价值：

- 尽早给框架开发者可读错误；
- 不让明显错误输入进入底层状态机；
- 测试可以直接验证 route contract。

C++ 检查的价值：

- custom op 可能被其它调用者直接调用；
- 防止 Python guard 未来修改后破坏底层不变量；
- 内存和通信代码必须自己保护自己。

底层高风险模块适当的 defense in depth 是合理的。

---

## 5. 第三站：`memfabric_mm_ar_binding.cpp` —— PyTorch 怎么找到 C++

文件：

```text
csrc/_310P/memfabric_mm_ar/memfabric_mm_ar_binding.cpp
```

你会看到类似：

```text
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
```

以及 schema：

```text
memfabric_mm_ar_allreduce(
    Tensor x,
    Tensor weight,
    int tp_rank,
    int batch_basem_count
) -> Tensor
```

然后把 PrivateUse1/NPU backend 映射到 C++ 实现。

这一步的本质：

```text
Python 的 torch.ops 名字
       ↓
PyTorch dispatcher
       ↓
C++ 函数指针
```

此外还注册：

```text
memfabric_mm_ar_shutdown()
memfabric_mm_ar_debug_snapshot()
```

说明这个子系统不只有计算 op，还有明确生命周期和诊断接口。

---

## 6. 第四站：`memfabric_mm_ar_runtime.cpp` —— 整个融合的“大脑”

文件很长，不建议线性读。

按以下块读。

### A. 常量和 `RuntimeState`

先找到：

```text
kMmArWidth = 2048
kBaseM = 256
kMaxArenaRows = 8192
small path threshold
RuntimeState
```

把状态字段按类别分：

```text
MemFabric context/layout
配置：rank/q/batch_m
内存：scratch/weight slices
warmup flag
protocol flag
graph state
wave generation
poisoned error state
```

看到 state 字段时问：

> 它是一次调用临时变量，还是进程生命周期资源？

这是理解 runtime 的关键。

### B. `init_context_locked(...)`

这里回答：

```text
context 什么时候创建？
q 什么时候锁定？
local pool 多大？
arena_rows 怎样从内存预算算出来？
为什么 world_size 固定 2？
```

特别注意：同一个 worker 生命周期里 rank/q 不能随便变，否则 runtime 直接报错。

这说明 q 当前是**进程配置**，不是每次调用动态 autotune 参数。

### C. warmup helpers

看：

```text
warm_producer_locked
warm_small_locked
warm waiter/add/protocol
```

理解为什么同一 kernel symbol 会 launch 两次，以及为什么 graph capture 前必须完成。

### D. `ensure_weight_slices_locked(...)`

这是 small-M N-split 的常驻优化。

看它如何：

```text
按 weight pointer cache
首次分配 8 MiB slices
Graph capture 内禁止首次构建
最多限制 entry 数
```

然后问一个优化问题：

> 这个时间换空间的选择，在几十层模型上会占多少额外内存？值不值得？

这正是第 10 章要训练的思考方式。

---

## 7. 第五站：直接定位 `memfabric_mm_ar_allreduce_impl(...)`

这是主入口。

建议自己在编辑器里把它折叠成这几段：

```text
1. validate input
2. get stream / capture state
3. init context
4. init scratch
5. allocate output
6. warm/init protocol
7. if small_path: ...
8. else batch waves: ...
9. return output
```

只要先看到这个骨架，长函数就不再可怕。

---

## 8. 主入口前半段为什么先判断 capture

逻辑类似：

```text
if capturing:
    context 必须已经存在
else:
    可以初始化/warmup
```

之后 scratch 也是一样：

```text
if scratch 不存在:
    capture 中禁止分配
    eager 中 aclrtMalloc + zero once
```

这说明当前 Graph 兼容策略是：

> 所有动态生命周期工作在 eager warmup 阶段做完，capture 只记录稳定设备动作。

---

## 9. small path 分支怎么读

条件：

```text
small_path_enabled()
&& M < min(batch_m, 256)
```

然后按语义标注每行：

```text
prepare_wave    -> 清必要控制状态
 gate           -> 等资源信用
 ensure slices  -> 取 N-split 权重缓存
 memcpy         -> 把 M 行放进 staging scratch
 generation     -> 本轮 ready 版本
 producer       -> 8-core N-split MM + signal
 wait           -> 等 peer blocked result
 add_small      -> add + unblock
 quiet          -> drain outbound SDMA
 ack            -> 归还 wave credit
```

如果这样注释过一遍，函数调用名字就不再需要背。

---

## 10. batch path 分支怎么读

先看两层切分：

```text
M
 -> wave (受 arena_rows 限制)
 -> batch (每个 batch_m 行)
```

再看两个 lambda/逻辑块：

```text
enqueue_producer(batch)
enqueue_wait_add(batch)
```

### producer 做什么

```text
算 valid_rows
如果 tail：copy 到 scratch
确定 x_batch
确定 generation
mf310p_direct_producer_async
```

### wait_add 做什么

```text
mf310p_wait_batch_async
算 output offset
mf310p_add_batch_async(valid_rows)
```

最后真正决定流水的只有很短几行：

```text
producer(0)
for batch=1..:
    producer(batch)
    wait_add(batch-1)
wait_add(last)
quiet
ack
```

这说明性能架构通常隐藏在几十行“编排代码”里，而不是最长的验证逻辑里。

---

## 11. 第六站：`memfabric310p_adapter_api.h` —— 先读接口，再读实现

不要直接打开 adapter.cpp。

先读 header，因为它就是模块合同。

先把接口分组：

### 生命周期

```text
mf310p_create
mf310p_destroy
mf310p_get_layout
```

### 一次性协议初始化

```text
control_barrier
exchange_geometry
init_credit
```

### 大 M batch path

```text
direct_producer_async
wait_batch_async
add_batch_async
```

### small-M

```text
build_weight_slices
small_producer_async
wait_small_async
add_small_async
```

### wave protocol

```text
prepare_wave_async
gate_async
quiet_async
ack_async
```

### warmup

各种 `warmup_*`

理解 API 分组以后，再进 `memfabric310p_adapter.cpp` 看每个接口具体怎样桥接 public MemFabric + kernel launch。

---

## 12. `mf310p_layout_t` 为什么值得单独画图

结构体里包括：

```text
pool_base
own_segment
peer_segment
send_arena
recv_arena
peer_recv_arena
ack_slot
peer_ack_slot
arena_bytes
batch_bytes
arena_rows
batch_m
max_batches
```

建议手画：

```text
symmetric pool
+---------------------------+
| rank0 segment             |
|  send arena               |
|  recv arena               |
|  ack/control              |
+---------------------------+
| rank1 segment             |
|  send arena               |
|  recv arena               |
|  ack/control              |
+---------------------------+
```

然后标出 rank0 眼里的：

```text
own send
own recv
peer recv
```

只要地址关系不清楚，后面所有 signal/wait 都会显得神秘。

---

## 13. 第七站：`memfabric310p_device.asc` —— 先读 producer

先跳过所有辅助函数，找到：

```text
Mf310pDirectProducerKernel
```

把它按 7 步读：

```text
1. block = GetBlockIdx()
2. 算每 core row offset
3. 配 A/B/C GlobalTensor
4. mm.IterateAll(cGm)
5. clean 本 core C region
6. write ready
7. if core0: wait 8 ready -> signal
```

你会发现其实核心很短。

难的不是矩阵乘 API，而是保证“何时可以安全 signal”。

---

## 14. 再回头看 tiling 宏

当前 batch path 有：

```text
M=256
M=512
M=1024
```

每个模板里的 singleCore M 是总 batch M / 8。

基础 tiling：

```text
baseN=256
baseK=64
CONFIG_NORM
```

注释中还记录 `CONFIG_MDL` 在目标小 per-core M 上不可用/表现有问题，因此选择 NORM。

这些“为什么不用另一个模式”的注释非常宝贵，因为性能代码最大的维护风险就是：后来的人只看见结果，不知道失败过哪些路线，又重新踩坑。

---

## 15. 再读 small producer

找到：

```text
Mf310pNsplitProducerKernel
```

对比 direct producer：

### direct

```text
8 core 分 M
每 core 全 N
共享完整 B
输出正常 row-major
```

### N-split

```text
8 core 分 N
每 core 全 T rows
每 core 自己 256-column weight slice
输出 [8][T][256]
```

对比着读，是理解 kernel 设计最有效的方法。

---

## 16. 最后读 wait 和 add

### `mf310pWaitBatchKernel`

核心不是计算，而是协议验证：

```text
smem_shm_sdma_wait
check status
check dst
check len
check imm
```

### `mf310pAddKernel`

核心是：

```text
连续切片
双 VECIN queue
vector Add
VECOUT
```

### `mf310pAddBlockedKernel`

在 Add 基础上多做：

```text
blocked [8][T][256]
 -> 正常 [M,2048]
```

这三类 kernel 分别代表：

```text
协议控制
普通数据规约
规约+layout transform
```

---

## 17. 读源码时如何判断“为什么这样写”

每遇到一段奇怪代码，用下面五问：

```text
1. 它在保证 correctness 还是 performance？
2. 如果删掉，会错数、死锁，还是只是变慢？
3. 它是硬件约束，还是当前实现选择？
4. 它是一次性成本，还是每 token/每 batch 成本？
5. 它在 eager 和 graph 下语义是否不同？
```

例如 generation：

```text
correctness：防 stale ready
performance：eager 避免 32 KiB clear
实现选择：可以有其它 epoch/doorbell 设计
成本：每 wave 少量控制
Graph：固定 generation + clear
```

这么分析，比背注释更能形成自己的判断力。

---

## 18. 推荐自己做一份“调用链笔记”

每次代码更新后，用下面模板重新确认：

```text
层名：
输入 shape/dtype/layout：
输出 shape/dtype/layout：

Python route：
Custom op schema：
C++ entry：
Adapter API：
Device kernel：

MM 切分：
通信粒度：
同步条件：
Graph 约束：
Fallback：
错误语义：
```

只要这张表仍然能填完整，你就没有被实现细节淹没。

---

## 19. 本章自测

1. 融合路径第一次从 stock 路径分叉发生在哪个文件？
2. `reduce_results=False` 在哪一层设置，为什么在那里设置合适？
3. PyTorch dispatcher 怎样把 Python `torch.ops` 连到 C++？
4. 为什么 runtime 主入口比 device producer 复杂得多？
5. 为什么先读 adapter header 再读 cpp？
6. direct producer 的 8 个关键动作你能按顺序说出来吗？
7. small producer 和 direct producer 最核心的并行维度差别是什么？

下一章单独把 MemFabric、同步、流水和 Graph 串起来。
