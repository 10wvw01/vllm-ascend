# 07｜MM+AR 融合算子完整设计：从一个公式走到 8 核、SDMA 和流水线

这一章不按文件讲，而是按“设计问题”讲。先理解为什么需要每个机制，再去看源码。

## 1. 先从最简单的正确版本开始

TP=2：

```text
rank0: Y0 = X0 @ W0
rank1: Y1 = X1 @ W1
Y = Y0 + Y1
```

最直接实现：

```text
rank0 MM -> Y0
rank1 MM -> Y1
        |
        v
两边交换 Y0/Y1
        |
        v
各自做 FP16 add
```

如果只考虑数学，这就够了。

真正工程复杂的原因是还要满足：

```text
性能
并发执行
缓存一致性
内存复用
Graph replay
异常处理
动态 M
小 M 效率
进程生命周期
```

下面逐个增加机制。

---

## 2. 为什么要用 arena，而不是每次 malloc 一个输出

每轮都：

```text
malloc send
malloc recv
通信
free
```

会有：

- host/runtime 固定开销；
- Graph 无法安全 capture 动态 malloc；
- 通信双方地址不稳定；
- 更难做流水和信用控制。

因此初始化时建立固定 symmetric pool，并从中规划：

```text
send_arena
recv_arena
peer_recv_arena
ack_slot
control/status
```

之后每个 batch 只用：

```text
slot = arena_base + batch_index * batch_bytes
```

这把动态问题变成了固定地址上的循环复用。

---

## 3. baseM 和 q 是什么

当前原生 MM 设计以：

```text
baseM = 256
```

作为通信/计算批次基础。

配置：

```text
q ∈ {1,2,4}
batch_m = 256 * q
```

于是：

```text
q=1 -> batch_m=256
q=2 -> batch_m=512
q=4 -> batch_m=1024
```

每行输出 N=2048，FP16 2 bytes，因此 payload：

```text
batch_bytes = batch_m * 2048 * 2
```

得到：

```text
q=1: 1 MiB
q=2: 2 MiB
q=4: 4 MiB
```

这里的“通信 batch”不是一个永远固定 2 MiB 的概念，而是跟 `baseM*q` 绑定。

当前默认 q=1。

---

## 4. 为什么 MM 用 8 个 AI Core

目标硬件上当前 producer 设计使用：

```text
blockDim = 8
```

并且不是留一个 core 专门通信，而是：

> 8 个 core 全部参与 MM，core0 在完成自己那份 MM 后再兼任 signal owner。

原因很直观：

如果为 signal 独占一个 core：

```text
7 核算 MM + 1 核等通信
```

MM 计算资源白白少 1/8。

而 signal 只是 MM 结束后的短控制动作，没有必要长期占用一个 AI Core。

所以当前原则是：

```text
8 核都算
core0 也算
所有核 ready 后
core0 单 producer signal
```

---

## 5. 为什么普通 batch path 按 M 切 8 份

AscendC classic Matmul API 在这里按单 core 语义工作，不会自动把一个大 MM 分给 8 个 block。

因此代码显式做：

```text
batch_m rows
     |
按 M 等分给 8 core
```

例如 q=1：

```text
batch_m = 256
每 core = 32 行
```

每个 core：

```text
A: [32,2048]
B: [2048,2048] 共享完整权重
C: [32,2048]
```

8 个 core 合起来才得到完整：

```text
[256,2048]
```

q=2/4 同理，只是每 core 的 M 增大。

---

## 6. 为什么权重是共享完整 B，而不是也切 B

大 M batch path 选择 M-split：

```text
每个 core 负责不同输出行
```

这样每个 core 需要完整的 B，因为：

```text
一行完整输出有 N=2048 列
```

优点是输出直接是正常 ND 连续行，通信和 reduce 简单。

缺点是小 M 时，每个 core 分到的行太少，而每个 core 仍然需要访问完整 B，这会出现明显的权重流固定成本。

这正是 small-M 后来改 N-split 的原因。

---

## 7. 为什么 MM 完成后不能马上 signal

假设 core0 自己算完了，就立刻告诉 SDMA：

```text
“这个 batch 可以搬了”
```

但此时 core1~7 可能还没把自己的 C 写完。

SDMA 会读到：

```text
一部分新数据 + 一部分旧数据
```

所以必须有一个 8-core 完成协议。

当前使用 generation-tagged ready cells。

每个 core：

```text
完成自己的 MM
 -> clean 自己写的 C cache lines
 -> ready[batch][core] = generation
```

core0：

```text
等待 8 个 ready 都等于本轮 generation
 -> signal
```

---

## 8. 为什么 ready 要带 generation

如果 ready 只是：

```text
0 = 没完成
1 = 完成
```

上一波留下的 `1` 很可能让下一波误以为 core 已经完成。

最简单办法是每波先 memset ready 区域为 0，但当前实测清 32 KiB 会产生可见固定开销。

所以 eager 路径改成：

```text
wave 1: generation = 0x40000000
wave 2: generation = 0x40000001
wave 3: generation = 0x40000002
...
```

core0 只认“等于当前 generation”的 cell。

旧值即使存在，也不会误判。

这是很经典的并发设计：

```text
用版本号避免反复清状态
```

Graph capture 则因为要 replay 相同命令，继续使用固定 generation，并把 clear 放在图内。

---

## 9. 为什么写完 C 还要 cache clean

AI Core 写 GM 后，最新数据可能仍在缓存层级里。

而 peer SDMA 直接从内存地址搬数据。如果不保证可见性，可能发生：

```text
AI Core 认为写完了
SDMA 却读到旧内存内容
```

因此每个 core 在发布 ready 前，会对自己负责的 C 区域做 64B cache line clean/invalidate。

顺序非常重要：

```text
MM write
 -> cache clean
 -> ready
 -> core0 signal
```

不能变成：

```text
ready
 -> cache clean
```

否则 ready 就失去“数据已经可被 SDMA 正确读取”的含义。

这叫 happens-before 关系：ready 不只是完成标记，还隐含了前面的数据可见性已经成立。

---

## 10. signal 到底做什么

core0 调用 public device API：

```text
smem_shm_sdma_signal(...)
```

可以把它理解成：

```text
请 SDMA 把我的 send slot
搬到 peer 的 recv slot
并附带 batch_index 等 mail 信息
```

MemFabric transport 内部如何组织 ring/mailbox，对 vLLM 是黑盒。

这是当前设计的重要边界：

```text
vLLM 只依赖 public signal/wait/quiet
不读取 MemFabric private internals
```

这样外部库升级时耦合更小。

---

## 11. peer 到达后为什么还要 wait 并严格校验 mail

`wait` 不是只问“有没有一条消息”。

还要检查：

```text
status
目标 dst
长度 len
imm / batch index
```

因为流水线中可能同时存在多个 batch。

如果 batch0 的 reduce 错拿了 batch1 的 recv 数据，shape 仍然一样，很可能不会 crash，却会静默算错。

所以协议校验属于 correctness，不是调试附加项。

---

## 12. local add 为什么自己写 AscendC kernel

收到 peer partial 后：

```text
out = local_send + peer_recv
```

理论上直接用 PyTorch：

```python
out = a + b
```

也能完成数学。

但当前实机观察到通用 elementwise add 对变化 output shape 可能产生昂贵的首次 GE compile 固定开销，代码注释记录约 90ms 量级的历史测量。

因此实现了 shape-agnostic 的：

```text
mf310pAddKernel
```

由 8 个 block 切连续元素，使用双 buffer queue 做：

```text
MTE2 load
 -> vector Add
 -> MTE3 store
```

这说明高性能工程中经常要优化的不是数学复杂度，而是**框架首次编译/动态 shape 固定成本**。

历史数字要在当前软件栈重新测，不能当永恒事实。

---

## 13. 为什么不等整个 M 算完再一次通信

假设 M=2048，q=1，可以切成 8 个 256-row batch。

如果串行：

```text
MM batch0
MM batch1
...
MM batch7
然后通信全部数据
然后 add
```

前面 batch0 算完以后一直在等。

更合理的是：

```text
batch0 算完 -> 马上让 SDMA 搬
同时 AI Core 去算 batch1
```

于是把 M 变成流。

这就是 batch pipeline 的真正价值。

---

## 14. 当前为什么是 lookahead=1

当前实际 enqueue：

```text
P0
P1
W0 A0
P2
W1 A1
...
Wlast Alast
quiet
ack
```

`P` = producer：MM + signal
`W` = wait peer mail
`A` = local add

之所以先 enqueue `P(n+1)` 再 enqueue `W/A(n)`，是希望：

```text
P(n) signal 以后 SDMA(n) 进行中
AI Core 开始 P(n+1)
```

然后 P(n+1) signal 后，设备又可推进：

```text
W/A(n) 与 SDMA(n+1)
```

这是一种计算、通信、reduce 的交错。

为什么不是 lookahead=2/3？不是数学不允许，而是：

- arena 占用；
- mail/credit 复杂度；
- wait 时机；
- 硬件实际 overlap；
- 过度排队可能没有收益。

当前选择是工程平衡，后续可以 profiler 驱动继续研究。

---

## 15. 为什么 quiet 只在 wave 尾部做一次

如果每个 batch 都：

```text
signal
quiet
wait
add
```

`quiet` 会把通信流水反复 drain，导致后续 batch 无法与前面计算重叠。

当前 hot path 的 wait 刻意不包含 quiet。

整波完成后才：

```text
quiet
ack
```

于是中间可以让多个 outbound SDMA 保持流水状态。

这是“把同步从 batch 粒度提升到 wave 粒度”的核心优化之一。

---

## 16. wave 和 batch 有什么区别

### batch

一次 producer/MM/communication 的粒度：

```text
batch_m = 256*q 行
```

### wave

一次可以安全占用当前 arena 的一组 batch。

arena 有固定行容量，例如最多到 8192 rows。

如果 M 超过 arena capacity：

```text
M
 -> wave0
 -> wave1
 -> ...
```

每 wave 内再切多个 batch。

wave 边界承担资源复用和 credit 保护。

---

## 17. credit 是解决什么问题

想象 rank0 已经开始下一波，并重新写自己的 send/recv arena，但 rank1 还在读上一波。

就会发生覆盖：

```text
rank1 正在读 old batch
rank0 把 slot 写成 new batch
```

结果数据竞争。

所以需要一个波级信用协议：

```text
只有确认 peer 已经完成上一波对共享 arena 的使用
才允许下一波复用
```

当前抽象成：

```text
gate
...
quiet
ack
```

可以把 credit 想成“一张 arena 使用许可证”。

---

## 18. 为什么尾批不再每次清零整个 scratch

如果最后 batch 只有 33 行，但 batch_m=256，需要一个完整形状给固定 MM kernel。

最直观：

```text
scratch 256 行全 memset 0
copy 33 行有效数据
做 256 行 MM
最后只 reduce 前 33 行
```

但每次 memset 大块设备内存会成为固定开销。

当前优化：

```text
scratch 初始化时只清零一次
之后尾批只覆盖有效前缀
后面的行允许保留 0 或历史脏值
```

为什么正确？

矩阵乘每一输出行只依赖对应输入行：

```text
C[row] = A[row] @ B
```

脏的 padding 行只会污染自己的输出行。

而 add 只消费：

```text
valid_rows
```

所以这些无效行永远不会进入最终 output。

这是一类非常有价值的优化思维：

> 不要为了“内存看起来干净”做无必要工作；只要能证明无效数据不会流入可观察结果，就可以避免清零。

---

## 19. 为什么 small-M 需要另一条路

假设 M=16，batch_m=256。

普通 M-split batch path 会算 256 行，其中 240 行无效。

更大的问题是每个 core 只算很少 M 行，却仍访问完整 2048×2048 权重。

当前实测发现这有明显的权重流固定地板。

因此 small path 改为 N-split：

```text
N=2048
 -> 8 core × 256 columns
```

每个 core：

```text
处理全部 T 行
只读取自己的 [256,2048] 权重 slice
得到 [T,256]
```

权重预先重排成：

```text
8 个连续的 256-column NZ slice
总计约 8 MiB / weight
```

这牺牲额外常驻内存，换取小 M 下更低的 weight-stream 成本。

---

## 20. 为什么 small-M 用 T 阶梯而不是完全动态 M

当前模板：

```text
T ∈ {16,32,64,128,256}
```

例如：

```text
M=9   -> T=16
M=25  -> T=32
M=70  -> T=128
```

这是静态高性能 kernel 和动态服务 shape 的折中：

- 完全动态 M：kernel tiling/Graph 更复杂；
- 永远 T=256：小 M 浪费太多；
- 多个模板阶梯：代码有限，同时降低 padding。

这是常见的 bucketization 思路。

---

## 21. small path 为什么输出是 blocked layout

N-split 时每个 core 产生：

```text
[T,256]
```

8 core 输出自然排成：

```text
[8][T][256]
```

为了避免 producer 之后再做一次昂贵重排，通信直接搬这个 blocked layout。

最后由 `mf310pAddBlockedKernel` 一边 local+peer add，一边把它写回正常：

```text
[M,2048]
```

这叫把“解块/unblock”融合进 reduce。

原则是：

> 中间格式只服务中间阶段，不必强迫每一步都回到最终标准格式。

---

## 22. 为什么 kernel 首次要 warm 两次

当前代码记录目标环境存在一个特殊行为：fresh `.so` 中每个 kernel symbol 第一次 launch 可能 silent no-op。

因此 runtime 在正式使用前，对相关 symbol 做两次 warmup并同步。

这不是算法设计，而是目标平台运行时现实。

重点学习的是应对方式：

```text
把不可控的首次行为提前移出 hot path
并且移出 Graph capture
```

不要把这种平台 quirks 隐藏成偶发 correctness bug。

---

## 23. 为什么失败后要 poisoned

分布式通信协议一旦中途失败，不能简单 catch 然后继续下一轮。

例如：

```text
rank0 认为 batch2 已发
rank1 wait 失败
arena / mail / credit 状态可能已经不同步
```

继续复用 context 可能把一次显式错误变成后续静默错数。

所以 RuntimeState 记录：

```text
poisoned = true
failure_reason = first error
```

以后直接 fail fast，要求重启 worker。

这是分布式底层代码很重要的安全原则：

> 当协议状态是否还能恢复无法被严格证明时，宁可停止复用，也不要猜。

---

## 24. 把整个大 M wave 连起来

最终可以画成：

```text
Host/C++ Runtime
  |
prepare_wave
  |
gate  ----------------------------- 等上一波资源可复用
  |
  +--> P0: 8-core MM -> clean -> ready -> core0 signal
  +--> P1: 8-core MM -> clean -> ready -> core0 signal
  +--> W0 -> A0
  +--> P2
  +--> W1 -> A1
  |    ...
  +--> Wlast -> Alast
  |
quiet ------------------------------- drain outbound SDMA
  |
ack   ------------------------------- 告诉 peer 本波资源已安全消费
  |
output [M,2048]
```

small M 则：

```text
prepare -> gate
 -> stage M rows
 -> 8-core N-split producer
 -> one signal
 -> wait
 -> blocked add/unblock
 -> quiet -> ack
```

这两条路共享的是同一个数学语义，不同的是如何映射硬件。

---

## 25. 本章自测

1. 为什么 core0 不能自己 MM 完就 signal？
2. ready 前为什么必须 cache clean？
3. generation 为什么可以让 eager 省掉 ready clear？
4. 为什么 `wait` 不应该每次隐含 `quiet`？
5. wave credit 保护的是哪个资源竞争？
6. 尾批保留脏 padding 行为什么仍然正确？
7. small-M 为什么从 M-split 改成 N-split？
8. blocked add 为什么也是一种融合？
9. poisoned 状态为什么比“失败后继续试”安全？

下一章按真正调用链逐文件读代码。
