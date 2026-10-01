# 17｜Host Runtime 深度精读：arena、wave、credit、lookahead 和 Graph 状态机

> 主文件：`csrc/_310P/memfabric_mm_ar/memfabric_mm_ar_runtime.cpp`
>
> 如果说 `memfabric310p_device.asc` 解决“一个 batch 怎么算和发”，那么 runtime 解决的是：“很多 batch、很多请求、Graph replay、进程生命周期下，怎样保证地址不撞、顺序不乱、还能尽量 overlap”。

---

## 1. RuntimeState 不是“缓存几个变量”，而是整个融合协议的进程内状态机

核心结构：

```cpp
struct RuntimeState {
    std::mutex mutex;
    mf310p_context_t* ctx;
    mf310p_layout_t layout;
    int tp_rank;
    int64_t batch_basem_count;
    uint32_t batch_m;
    bool poisoned;
    std::string failure_reason;
    uint64_t producer_scratch;
    std::vector<WeightSlicesEntry> weight_slices;
    bool producer_warmed;
    bool waiter_warmed;
    bool add_warmed;
    bool small_warmed;
    bool protocol_warmed;
    bool protocol_initialized;
    aclrtStream eager_stream;
    bool graph_capture_seen;
    uint32_t wave_seq;
};
```

建议不要逐字段背，而要按 5 类理解。

### A. 通信资源

```text
ctx
layout
```

### B. 固定运行合同

```text
tp_rank
batch_basem_count
batch_m
```

### C. 长生命周期内存

```text
producer_scratch
weight_slices
```

### D. warmup / Graph 准备状态

```text
producer_warmed
waiter_warmed
add_warmed
small_warmed
protocol_warmed
protocol_initialized
```

### E. correctness 状态

```text
poisoned
failure_reason
eager_stream
graph_capture_seen
wave_seq
```

所以 runtime 本质上是：

> 一个带资源所有权、协议阶段和失败状态的有限状态机。

---

## 2. 为什么 runtime 要加全局 mutex

`memfabric_mm_ar_allreduce_impl()` 一进入就：

```cpp
std::lock_guard<std::mutex> guard(state.mutex);
```

原因不是“C++ 写法保守”，而是当前 runtime 设计本身是**单上下文、单 eager stream、共享 arena**。

如果两个 host thread 同时对同一个 RuntimeState enqueue：

```text
thread A: wave0 batch0
thread B: wave0 batch0
```

它们会竞争：

```text
same send arena slot
same ready cells
same wave generation
same credit
```

因此当前 contract 明确选择：

```text
host 侧串行进入 runtime
设备内部再利用 MM/SDMA/vector overlap
```

这和“支持多请求并发”并不矛盾。vLLM 的请求并发最终会被 scheduler 合并成某轮 tensor，进入一次模型 forward；这里保护的是一个 worker 进程内的 fusion runtime 协议，不是用户请求队列。

---

## 3. `batch_m` 为什么一旦 context 创建后就不能在线改

初始化时：

```cpp
batch_m = 256 * batch_basem_count
```

如果 `state.ctx` 已存在，再次调用会检查：

```text
rank 不能变
batch_basem_count 不能变
```

为什么改 q 要重启 worker？

因为 q 不只影响一个循环变量，它决定：

```text
batch_m
batch_bytes
arena_rows 对齐
max_batches
producer kernel 模板
scratch 大小
mail len
ready/control 使用方式
```

也就是说 q 是 runtime memory/protocol geometry 的一部分。

如果在进程中途从 q=1 改 q=4，却继续复用旧 context，双方 rank 对同一地址空间和 batch 长度的理解可能不一致，属于协议级错误。

---

## 4. arena_rows 是怎样从内存预算推出来的

当前常量：

```text
N = 2048
FP16 = 2 bytes
row_bytes = 2048*2 = 4096 bytes
local pool 默认 = 96 MiB
headroom = 1 MiB
max arena rows = 8192
```

runtime 计算：

```cpp
rows_by_budget =
    (local_pool_bytes - headroom - 8)
    / (2 * row_bytes)
```

为什么分母有 `2 * row_bytes`？

因为每一行 arena capacity 要同时预留：

```text
send arena 一行
recv arena 一行
```

即每 rank 需要两份 `[arena_rows,2048]` FP16 数据区。

随后：

```text
arena_rows = min(rows_by_budget, 8192)
arena_rows 向下对齐到 batch_m 的整数倍
```

这保证：

```text
一个 wave 的 arena 可以被完整切成若干 batch slot
```

不会出现最后一个 slot 只有半块物理空间。

---

## 5. 96 MiB 默认预算下，大致能放多少行

先粗算：

```text
可用于 send+recv 的字节
≈ 96MiB - 1MiB
≈ 95MiB
```

每一行双份：

```text
2 * 4096B = 8192B = 8KiB
```

理论 rows：

```text
95MiB / 8KiB ≈ 12160 rows
```

但代码有：

```text
kMaxArenaRows = 8192
```

所以最终最多 8192 rows，再按 batch_m 对齐。

例如：

### q=1

```text
batch_m=256
8192/256=32 batches per wave
```

### q=2

```text
batch_m=512
8192/512=16 batches per wave
```

### q=4

```text
batch_m=1024
8192/1024=8 batches per wave
```

注意：

> q 变大并没有让 arena 总行数按比例变大，只是每个 batch 变大、每 wave 的 batch 数变少。

这就是当前“arena 按内存预算定容，而不是按 batch payload 扩张”的设计。

---

## 6. batch 和 wave 的关系，用 M=20000 举例

假设：

```text
arena_rows=8192
q=1
batch_m=256
M=20000
```

runtime 外层：

```cpp
for (wave_start = 0; wave_start < M; wave_start += arena_rows)
```

于是：

```text
wave0 rows: 8192
wave1 rows: 8192
wave2 rows: 3616
```

每 wave 内再切 batch：

```text
wave0: 32 batches
wave1: 32 batches
wave2: ceil(3616/256)=15 batches
```

所以：

```text
batch = 计算/通信流水粒度
wave  = arena 安全复用粒度
```

两者绝对不能混淆。

---

## 7. 为什么需要 fixed credit

设想两个 TP rank 共享“逻辑上对称”的 send/recv arena。

rank0 wave0 发完数据后，如果立刻开始 wave1，可能重新覆盖：

```text
sendArena slot0
```

但 rank1 也许还在用上一波对应的数据做 local add。

于是可能出现：

```text
rank1 正在读 old peer partial
rank0 已经把同一物理 slot 写成 next wave partial
```

这是典型 producer-consumer reuse race。

credit 的作用可以理解成：

```text
一张“允许复用整块 arena”的令牌
```

每波：

```text
gate
  -> 拿到本波复用许可
...
quiet
  -> 确认本 rank 发出的 SDMA 全部落地
ack
  -> 告诉 peer：我已经不再依赖这一波 arena，可给下一波
```

因此 ready 是**batch 内 8 核同步**，credit 是**wave 间两 rank 资源同步**。

这是两层完全不同的协议。

---

## 8. `prepare_wave` 为什么 eager 和 Graph 参数不同

调用：

```cpp
mf310p_prepare_wave_async(
    ctx,
    capturing ? 1u : 0u,
    stream)
```

第二个参数决定是否 clear wave control。

### eager

```text
clear_wave_control = 0
```

因为 eager generation 单调变化，不需要每波清 32KiB ready cells。

### Graph

```text
clear_wave_control = 1
```

因为 Graph replay 复用固定 generation，需要每 replay 把历史 ready 清掉。

这就是同一套 control memory 在两种执行模式下采取不同 correctness 策略。

---

## 9. 为什么 Graph capture 前必须把几乎所有东西准备好

`require_capture_ready_locked()` 检查：

```text
ctx 已创建
protocol initialized
producer scratch 已分配
producer warm
waiter warm
add warm
small path warm
protocol warm
```

原因是 Graph capture 里不适合出现：

```text
aclrtMalloc
host control barrier
geometry exchange
首次 kernel 符号 lazy init
需要 host sync 的 warmup
```

这些行为要么无法 capture，要么 replay 语义不稳定。

所以当前设计把生命周期分成：

```text
Eager warmup 阶段：
  把动态初始化做完

Graph capture：
  只捕获稳定 device work

Replay：
  重放固定图
```

这其实和很多高性能系统里的原则一致：

> 初始化可以复杂，hot path 必须稳定。

---

## 10. 为什么 kernel 每个符号要 warm 两次

代码对 producer/wait/add/protocol/small templates 都：

```cpp
for (int i=0; i<2; ++i) launch warmup
```

当前 device 文件注释记录一个目标机特性：

```text
freshly loaded .so 中某 kernel symbol 第一次 launch 可能 silent no-op
```

因此只 warm 一次不够，第二次才确保该 symbol 真正进入可用状态。

尤其 small-M 有 5 个模板：

```text
T=16/32/64/128/256
```

每个模板是不同 kernel symbol，所以要全部提前 warm。

如果你只测 T=16，然后 Graph 首次 replay T=128，可能出现非常难理解的“Graph 里第一次不工作”问题。

---

## 11. 为什么 small-M weight slice 必须在 capture 前构建

small path 需要把完整 NZ weight 重排成：

```text
8 × [256,2048] column slice
```

总 8 MiB。

`ensure_weight_slices_locked()` 如果 cache 没命中，会：

```text
aclrtMalloc 8 MiB
8 次 device-to-device strided copy/re-layout
```

这些显然不是理想 Graph replay 内容。

所以 capturing 时发现 slice 不存在，直接报错：

```text
must be built by eager warmup before ACL Graph capture
```

这属于 Graph 的“资源前置化”。

---

## 12. eager 为什么强制单 stream

`validate_execution_stream_locked()`：

```text
第一次 eager 调用记录 stream
后续 eager 必须是同一个 stream
```

为什么？

因为 runtime 依赖 stream 内顺序来证明很多事情：

```text
scratch copy 在 producer 前
producer 在 wait/add 的指定顺序前
quiet/ack 在 wave 尾
weight slice build 在第一次 small producer 前
```

如果允许两个 eager stream 并发使用同一 arena/context，就必须额外设计跨 stream event/barrier。

当前代码选择简单、可证明的合同：

```text
eager single-stream
Graph capture side stream 作为框架管理的特殊情况单独允许
```

这不是说 310P 不能多 stream，而是本 fusion runtime 没有声明支持共享 context 的任意多 stream 并发。

---

## 13. 为什么 Graph capture stream 可以例外

`torch.npu.graph()` 会切到内部 capture stream。

runtime 不把它当成“随机第二条 eager stream”，因为 Graph 框架在切换前有自己的 quiescent handoff 语义。

所以 capturing 时：

```text
state.graph_capture_seen = true
return
```

不强制等于 eager_stream。

这里体现的是：

```text
框架控制的 stream transition
!=
用户随意多 stream 并发
```

---

## 14. 普通 batch path 的完整 enqueue 顺序

当前 runtime 注释直接给出：

```text
lookahead=1
```

代码：

```cpp
enqueue_producer(0);
for batch=1..N-1:
    enqueue_producer(batch)
    enqueue_wait_add(batch-1)
enqueue_wait_add(last)
quiet
ack
```

例如 4 个 batch：

```text
P0
P1
W0 A0
P2
W1 A1
P3
W2 A2
W3 A3
Q
ACK
```

这里 `P` 不是纯 MM：

```text
P = 8-core MM + cache clean + ready rendezvous + signal
```

所以 P0 完成 signal 后，SDMA0 可以独立推进。

runtime 随后先排 P1，而不是立刻 W0/A0，目的就是给：

```text
SDMA0 || MM1
```

制造时间窗口。

---

## 15. 为什么 lookahead=1 能形成两段 overlap

理想时间线：

```text
时间 --->

AI Core/Cube:  [MM0][MM1][wait/add0][MM2][wait/add1]...
SDMA:               [copy0]      [copy1]      [copy2]...
```

更准确地说 wait kernel 自身很小，add 用 vector/MTE，而 producer MM 用 Cube；不同阶段是否真重叠还取决于设备调度与资源争用。

代码只是创造**可重叠的依赖图**。

是否真的 overlap，必须用 profiler 验证。

这点很重要：

> enqueue 顺序支持 overlap，不等于硬件一定给你 overlap。

如果 profiler 最终仍串行，问题可能在：

```text
同 stream launch serialization
engine dependency
SDMA 与 GM 带宽竞争
kernel occupancy
MemFabric 内部调度
```

---

## 16. 为什么不是 `P0 -> W0/A0 -> P1`

这种顺序虽然正确，但几乎把 pipeline 写死成：

```text
MM0
等通信0
reduce0
MM1
等通信1
reduce1
```

SDMA 期间 Cube 可能空闲。

当前 `P1 before W0/A0` 的核心想法是：

```text
先把下一份可独立计算的工作塞进去
再回来消费上一份通信结果
```

这是经典 software pipeline。

---

## 17. 为什么不是 lookahead=2 或更多

理论上可以尝试：

```text
P0 P1 P2 W0 A0 ...
```

但深度增加会改变：

```text
同时 in-flight 的 SDMA 数
mail arrival 顺序压力
arena slot 占用时间
wait 延迟
Cube/Vector 调度关系
```

而且如果 SDMA 本身已经能完全被一个 MM batch 隐藏，继续加深没有收益。

所以 lookahead depth 应该是 profiler 参数，不是“越大越先进”。

当前 1 是一个简单、协议风险低的起点。

---

## 18. 为什么 `wait` 后可以马上 add，而不用 quiet

`wait(batch i)` 的语义是：

```text
peer 发到我 recv slot i 的数据已经可消费
```

因此本地 add 只依赖：

```text
local send slot i 已经存在
peer recv slot i 已到
```

它并不要求：

```text
我发给 peer 的其他 batch 全部已经完成
```

所以：

```text
wait i -> add i
```

可以和后续 outbound SDMA 同时存在。

`quiet()` 如果放这里，就把不相关的“本 rank outbound drain”强行变成 add 的前置依赖。

这会破坏 overlap。

---

## 19. 为什么 quiet 必须最终存在

不在每 batch quiet，不代表永远不 quiet。

wave 结束前必须证明：

```text
本 rank 这一波发出去的 SDMA 都已经完成
```

否则 ack 后 peer 可能允许下一波复用地址，而当前 outbound DMA 仍在读旧 send slot。

所以 wave 尾：

```text
quiet -> ack
```

顺序不能反过来。

如果：

```text
ack -> quiet
```

peer 收到 credit 后可能提前覆盖你还在被 SDMA 读取的源数据。

---

## 20. tail batch 为什么只 copy valid rows，不每次 memset

假设：

```text
batch_m=256
最后只剩 valid_rows=33
```

producer kernel仍要求完整 256-row input。

runtime 有一个 `producer_scratch`，分配时只清零一次。

尾批：

```text
copy 前 33 行
后 223 行保持旧值/零
```

然后照样算完整 256 行、照样发送完整 1MiB。

为什么结果仍正确？

因为矩阵乘法按 row 独立：

```text
C[row] 只依赖 A[row]
```

33..255 行的脏输入只污染对应脏输出行，不会影响 0..32 行。

而 add 阶段只写：

```text
valid_rows=33
```

所以脏行永远不会进入最终 output。

这叫 garbage row localization。

它用“多算无用行”换掉了“每尾批一次大 memset”的固定开销。

---

## 21. 为什么这种尾批技巧不能随便推广到所有算子

关键前提是：

```text
输出第 i 行只依赖输入第 i 行
```

Linear 满足。

但如果是：

```text
softmax across rows
layer 之间跨 token reduction
某种 global normalization
```

脏 padding 行可能参与全局统计，影响 valid rows。

所以“脏 padding 可以不清”不是通用技巧，而是建立在数据依赖证明上的优化。

---

## 22. 为什么异常后 runtime 直接 poisoned

host 任何 wave 内异常：

```cpp
state.poisoned = true;
state.failure_reason = exc.what();
throw;
```

下一次进入：

```cpp
check_runtime_healthy(state)
```

直接拒绝继续。

原因是通信协议错误后，host 很难可靠知道：

```text
peer 收到了几封 mail
哪个 SDMA 已经 in-flight
credit 是否已消费
arena 哪些 slot 被覆盖
```

要实现在线恢复，需要一个完整的 distributed reset protocol。

当前没有这套协议，所以 fail-stop 是更正确的选择。

---

## 23. 为什么 Graph-used context 在 shutdown 时宁可泄漏到进程退出

如果 `graph_capture_seen=true`，shutdown 会跳过 `mf310p_destroy()`。

代码给出的理由是：Graph replay 可能在 host 函数之外执行，runtime 无法证明所有 replay stream 已经 quiescent。

如果此时释放 pool：

```text
某个 graph 仍可能持有这些设备地址
```

后果是 use-after-free。

当前策略：

```text
不能证明安全销毁
=> 让资源跟进程一起死
```

这是典型的安全生命周期权衡：

```text
可控进程级泄漏
优于
不确定的异步 use-after-free
```

---

## 24. runner hook 为什么把 device-wide sync 改成 current-stream sync

`memfabric_mm_ar_runner_hooks.py` 里：

```python
if memfabric_mm_ar_pool_started():
    torch.npu.current_stream().synchronize()
else:
    torch.npu.synchronize()
```

为什么 pool 启动后不能随便 device-wide sync？

MemFabric 自己有长期运行的 supervised epoch kernel/stream。

device-wide synchronize 可能把这个长期任务也当成“必须结束”的工作，从而一直等。

所以融合启用后，runner 只 drain 模型当前 stream。

这说明外部通信 runtime 接入模型框架时，影响的不只是一个 custom op，还会改变“什么叫设备空闲”的定义。

---

## 25. 一张完整状态图

可以把 runtime 简化成：

```text
UNINITIALIZED
   |
   | first eager fused call
   v
CREATE_CTX
   |
   v
ALLOC_SCRATCH
   |
   v
WARM_KERNEL_SYMBOLS
   |
   v
CONTROL_BARRIER + EXCHANGE_GEOMETRY
   |
   v
INIT_CREDIT
   |
   v
READY
   |
   +---- eager wave ----+
   |                    |
   | prepare/gate       |
   | P/W/A pipeline     |
   | quiet/ack          |
   |                    |
   +--------------------+
   |
   +---- graph capture/replay ----+
   |                               |
   | require all resources ready   |
   | capture fixed device sequence |
   | replay                        |
   +-------------------------------+
   |
   +---- protocol failure ---> POISONED
   |
   +---- shutdown (no graph) ---> QUIET/SYNC/DESTROY
   |
   +---- shutdown (graph used) --> keep process-lifetime
```

这张图比函数列表更重要。

---

## 26. 本章自测

1. 为什么 batch 是计算/通信粒度，wave 是 arena 复用粒度？
2. 96MiB pool 为什么默认最多只取 8192 rows？
3. q=4 时一波最多几个 batch？
4. ready 和 credit 分别保护什么资源？
5. `quiet -> ack` 为什么不能交换顺序？
6. 为什么 Graph capture 前必须提前做 weight slice build 和 kernel warmup？
7. 为什么 eager 强制单 stream？
8. 为什么 tail 可以保留 stale padded rows？成立的数学前提是什么？
9. 为什么协议失败后不尝试继续复用 context？
10. 为什么 Graph-used context 宁愿进程级保留，也不立即 destroy？

如果你能把这些回答成“如果不这样做，会出现什么具体 race/错误”，就已经真正理解 runtime 了。
