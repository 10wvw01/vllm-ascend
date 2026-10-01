# 09｜MemFabric、同步、流水线与 Graph：通信代码真正难的是“什么时候安全”

## 1. 先把 MemFabric 当成搬家公司

对这个项目来说，不需要一开始理解 MemFabric 内部实现。

你只需要先掌握 public contract：

```text
host：创建 context / 获得布局 / 初始化控制面

device：
signal -> 发起一段 SDMA 搬运
wait   -> 等一封到达 mail
quiet  -> 确保本地发出的搬运已经完成/drain
```

当前设计刻意不去读取 MemFabric 私有 ring、mailbox、orchestrator 内部状态。

这意味着：

> vLLM 负责“我要搬什么、何时搬、搬完如何规约”；MemFabric 负责“底层怎么可靠搬过去”。

---

## 2. 为什么还需要 adapter

外部 MemFabric public API 和 vLLM 想要的高层动作并不完全是同一种粒度。

所以 `memfabric310p_adapter_api.h` 建立一个稳定边界：

```text
vLLM C++ Runtime
      |
      | mf310p_* ABI v7
      v
310P adapter
      |
      | MemFabric public host/device API
      v
MemFabric
```

Adapter 还有一个重要作用：明确**谁拥有哪块内存、谁负责生命周期**。

这样 runtime 不需要知道 transport 私有数据结构。

---

## 3. 对称内存为什么适合 peer 通信

如果 rank0 和 rank1 每次都临时交换“我这次 buffer 地址是多少”，开销和协议复杂度都会增加。

初始化阶段建立 symmetric pool 后，双方能提前得到稳定的：

```text
send_arena
recv_arena
peer_recv_arena
ack slot
```

以后数据面只需要：

```text
slot index + 固定几何
```

非常适合 Graph 和高频 Decode。

---

## 4. 控制面和数据面要分开理解

### 控制面

负责：

```text
context create
双方 rendezvous
交换 geometry
初始化 credit
生命周期 teardown
```

这些事情不应该每个 token 都做。

### 数据面

每一 wave 高频执行：

```text
gate
MM
signal
wait
add
quiet
ack
```

性能优化的重要原则之一就是：

> 尽量把昂贵的控制面工作搬到初始化阶段，让 hot path 只剩稳定设备动作。

---

## 5. 三个“完成”千万不要混

通信代码里最容易犯错的是把不同层次的“完成”当成一回事。

### MM 完成

某个 AI Core 已经算完自己的 C。

### 数据可被 SDMA 看到

MM 算完不等于 peer engine 已经能看到最新内存，所以还要 cache clean。

### SDMA 完成

signal 只是发起/排队，不代表所有 outbound 数据已经到达并安全结束。

所以正确顺序中存在多个 barrier/状态：

```text
MM done
 -> cache visible
 -> 8 cores ready
 -> signal queued
 -> peer wait sees expected mail
 -> local add consumes recv
 -> wave quiet/drain
 -> ack resource safe to reuse
```

以后任何协议优化都要明确自己想省的是哪一个步骤，不能模糊地说“少一个同步”。

---

## 6. ready 是核内协作，不是跨卡通信

`ready[batch][core]` 解决的问题是：

```text
同一张卡上的 8 个 AI Core
是不是都完成这个 producer batch？
```

它和 peer rank 无关。

core0 等 8 个 ready 后才是唯一 signal producer。

因此可以分成两层同步：

```text
卡内：8-core ready rendezvous
跨卡：MemFabric signal / wait / credit
```

分清这两层以后，很多“是否需要跨卡 barrier”的问题会简单很多。

---

## 7. 为什么单 producer signal 并不等于独占一个 core

当前设计：

```text
8 个 core 都参加 MM
core0 算完自己那份以后
额外等待其它 7 个 ready
然后执行一次 signal
```

“single producer”描述的是：

```text
只有一个执行者负责发布一次通信事件
```

而不是：

```text
必须保留一个 core 整轮不做计算
```

这是并发语义里很容易混淆的两个概念。

---

## 8. generation 如何解决 ABA/stale 状态问题

假设 ready cell 一直被复用：

```text
wave0 写 1
wave1 也想写 1
```

wave1 开始时如果还没覆盖旧值，core0 可能误判“已经 ready”。

generation 让状态变成：

```text
wave0 -> 100
wave1 -> 101
wave2 -> 102
```

判断条件不是“非零”，而是：

```text
cell == current_generation
```

旧状态不会被误认。

当前 eager generation 还故意放在高范围 `0x40000000` 以后，与 Graph capture 使用的小固定 generation 分离。

这是为了防止 eager 和 captured wave 交替时出现碰撞。

---

## 9. 为什么 Graph 路径反而要 clear ready

Graph replay 的特点是：录制下来的参数/命令希望稳定。

如果 capture 时 generation=1，replay 时仍然可能是 1。

上一轮也留下 1，就无法靠递增版本规避 stale。

所以 captured flow 把控制区 clear 作为图内动作的一部分，再使用固定 generation。

这说明 eager 最优策略和 Graph 最优策略可以不同。

不要为了“代码统一”强迫两条路径完全相同。

---

## 10. credit 为什么是 wave 级，而不是 batch 级

如果每个 batch 都等对方确认 arena slot 可复用：

```text
P0 -> ack0
P1 -> ack1
P2 -> ack2
```

协议开销会很高，而且切断流水。

当前 arena 本身可以容纳多个 batch，因此把资源保护提升到整 wave：

```text
进入 wave 前 gate 一次
wave 内多个 batch 流水
wave 末尾 quiet + ack 一次
```

这样 credit 的固定成本被整波摊薄。

这也是“batch 粒度计算 + wave 粒度资源管理”的分层。

---

## 11. 为什么 `wait` 不等于 `quiet`

可以用寄快递类比：

```text
wait：我收到对方给我的第 n 个包裹了吗？
quiet：我自己发出去的所有包裹都已经处理完了吗？
```

两者方向不同。

如果每次收到一个 peer batch 就把自己 outbound 全部 quiet，会导致正在飞的后续 SDMA 被迫频繁 drain。

所以当前 wait kernel 只消费一封期望 mail，不做 wave drain。

---

## 12. lookahead=1 的时间线应该怎么看

假设 4 个 batch，host enqueue：

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

不要把这个理解成严格的设备串行执行图。

真正希望看到的 profiler 近似：

```text
AI Core : [MM0][MM1][A0][MM2][A1][MM3][A2][A3]
SDMA    :      [D0]    [D1]    [D2]    [D3]
```

实际 overlap 受硬件调度、资源竞争和 wait 条件影响。

所以“代码具备 overlap 结构”与“实机真的 overlap”是两个不同结论。

只有 profiler 能证明后者。

---

## 13. 为什么 batch 太大也不一定好

q 增大：

```text
q=1 -> 1 MiB / 256 rows
q=2 -> 2 MiB / 512 rows
q=4 -> 4 MiB / 1024 rows
```

大 batch 的潜在优点：

- 减少 batch 数；
- 减少 signal/wait/add launch 次数；
- 单次 MM 更大，core 利用率可能更好。

潜在缺点：

- 第一个 batch 更晚才能开始通信；
- pipeline 粒度变粗；
- tail 浪费更大；
- Decode 常见 M 小时更容易落入特殊路径；
- arena 同样内存能容纳的 batch 数更少。

所以 q 是一个典型的：

```text
固定开销 amortization
vs
pipeline granularity / padding
```

权衡。

不能凭直觉说 q 越大越好。

---

## 14. arena_rows 为什么由内存预算推导

当前默认 local pool 预算约 96 MiB。

runtime 不是简单写死：

```text
arena_rows = 8192
```

而是根据：

```text
local_pool_bytes
headroom
send + recv 每行字节数
batch_m 对齐
最大上限 8192
```

推导。

好处：

- q 变了仍能保持内存安全；
- 不会把 2 MiB/4 MiB payload 误当协议常量；
- arena capacity 和实际 pool 大小一致。

这是“配置参数应该从物理资源推导，而不是复制魔法数字”的例子。

---

## 15. Graph capture 为什么最怕动态生命周期

Graph 录制后希望 replay 的设备操作稳定。

这些事情非常不适合 capture 内第一次发生：

```text
malloc/free
建立 TCP/store rendezvous
host barrier
动态加载首次 kernel symbol
构建新的 weight slice cache
改变 q/arena geometry
```

所以正确生命周期是：

```text
Eager warmup
  -> create MemFabric
  -> exchange geometry
  -> allocate scratch
  -> build needed weight slices
  -> warm kernels
  -> initialize credit

然后 capture
  -> 只记录稳定设备工作

最后 replay
```

这就是“capture-ready state”的含义。

---

## 16. 为什么每个 kernel symbol 都要提前 warm

当前目标环境记录了 fresh `.so` 首次 kernel symbol launch 可能 silent no-op 的平台行为。

因此：

```text
producer 256/512/1024 模板
small T=16/32/64/128/256 模板
add
blocked add
wait/protocol
```

只要后面可能用到，就应该在正式 Graph 前把相关 symbol 预热。

否则可能出现很诡异的问题：

```text
某 shape 第一次 replay 没有真正执行
第二次却好了
```

把这种 quirks 系统化 warmup，比在业务层重试安全得多。

---

## 17. shutdown 为什么也要非常谨慎

通信系统退出时如果还有未完成 SDMA，直接 free pool/context 会造成 use-after-free 风险。

因此 shutdown 会尝试：

```text
quiet
stream sync
再 free scratch/slices/context
```

如果 quiet/sync 已经失败，当前设计宁可：

```text
标记 poisoned
资源留到进程退出
```

也不冒险 destroy。

这体现一个底层生命周期原则：

> 泄漏到进程退出通常比异步设备仍在访问时提前释放更安全。

---

## 18. 怎样分析一个可能的死锁

如果 TP 两个 worker 卡住，建议按状态机排：

```text
1. 两边是否都创建了 context？
2. control barrier 是否双方到达？
3. geometry 是否交换一致？
4. initial credit 是否种下？
5. 本 wave gate 是否一边永远等 credit？
6. producer 8 ready 是否全部到达？
7. core0 signal 是否成功？
8. peer wait 期待的 batch/dst/len/imm 是否一致？
9. quiet 是否能 drain？
10. ack 是否发回？
```

不要只看“卡在 wait”，因为根因可能是上游 producer 没 signal，甚至更早的某 core ready 永远没写。

---

## 19. 怎样分析 silent wrong result

比 deadlock 更危险的是“不报错但答案错”。

优先检查：

```text
ready 是否早于 cache clean？
mail dst/len/imm 是否严格验证？
尾批 valid_rows 是否正确？
wave slot 是否被下一波提前覆盖？
rank0/rank1 的 batch 划分是否完全相同？
weight slice 是否对应正确 layer weight pointer？
Graph replay 是否使用了 stale control cell？
```

当前协议里的大量 validation 就是在把 silent corruption 尽量变成显式失败。

---

## 20. MemFabric 未来还能怎么演进

在保持 public API 边界前提下，可以探索：

```text
更深 lookahead
更多 in-flight batch
更细/更粗 payload autotune
减少 ready polling 开销
硬件事件/doorbell 替代部分轮询
add 与后继算子融合
不同 arena 双缓冲
TP>2 的通信拓扑
压缩/低精度 payload
```

但每项演进都必须重新证明：

```text
不会覆盖 arena
不会错配 mail
不会破坏 cache visibility
不会让 Graph 生命周期动态化
不会让异常恢复变得不可控
```

同步优化的第一目标永远是正确，然后才是少几微秒。

---

## 21. 本章自测

1. ready、wait、quiet、credit 分别保证什么？
2. 为什么 ready 是卡内同步而 credit 是跨卡资源协议？
3. signal 成功为什么不代表整 wave 通信已经结束？
4. 为什么 Graph 和 eager 的 generation 策略不同？
5. q 变大有什么两个好处和三个潜在坏处？
6. 为什么 shutdown 失败时宁愿泄漏到进程退出？
7. 如果 wait 卡死，你会从哪些上游状态开始排查？

下一章把所有这些机制放回性能工程：怎样判断下一步该改哪一个。
