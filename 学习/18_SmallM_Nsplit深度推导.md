# 18｜Small-M 为什么要从 M-split 改成 N-split：从性能现象到内存布局推导

> 这一章专门回答一个最容易被一句话带过、但实际上很关键的问题：
>
> **为什么大 M 用 8 核按 M 切，而小 M 反而要按 N 切？**
>
> 主文件：
>
> - `csrc/_310P/memfabric_mm_ar/memfabric_mm_ar_runtime.cpp`
> - `csrc/_310P/memfabric_mm_ar/memfabric310p_device.asc`

---

## 1. 先看 direct path 在小 M 下发生了什么

大 M 路径固定：

```text
8 核 M-split
```

例如 q=1，batch_m=256：

```text
每 core 负责 32 行
```

这在 M>=256 时很自然。

但假设真实 M=10。

如果仍然强行走 256-row batch：

```text
真实有效行 = 10
计算行       = 256
```

计算放大：

```text
256 / 10 = 25.6 倍
```

也就是说，绝大多数 MM 都在算最终会丢弃的 padding row。

这还不是全部问题。

---

## 2. M-split 在小 M 下还有“完整权重流”固定成本

Direct path 每个 core：

```text
A_core [Mcore,2048]
B      [2048,2048] 完整共享
C_core [Mcore,2048]
```

即使 Mcore 很小，每个 core 仍然需要围绕完整 B 做 Matmul 数据流。

目标机注释记录过实测现象：

```text
full-B M-split
每 core 有约 150us 量级 weight-stream floor
```

而且这个固定底噪对很小 M 不会按行数线性下降。

为什么？

因为小 M 时 arithmetic work 下降得很快，但：

```text
weight 访问
Matmul pipeline setup
NZ tile traversal
kernel 固定开销
```

并没有同比例消失。

所以性能从“算力受限”逐渐变成“权重流/固定开销受限”。

---

## 3. 一个简单算术：M=16 时算术强度为什么很差

假设单 core 如果做完整 N=2048、K=2048 的 16 行：

MAC 数量：

```text
16 * 2048 * 2048
≈ 67 million MACs
```

但 weight 大小仍接近：

```text
2048 * 2048 * 2B
= 8 MiB
```

如果按 8 核 M-split，8 核都面对完整 B 的工作语义。

即使硬件 cache/L1/L2 能减少部分物理 DRAM 重读，执行结构仍然对小 M 非常不友好：

```text
每核算的 row 很少
但 N/K traversal 几乎完整
```

这就是小 M 需要改变并行维度，而不是只调 basic tile 的原因。

---

## 4. N-split 的核心想法

小 M 时不要让 8 个核分别算不同 M row，而是：

```text
每个 core 都处理全部 M
但只算 1/8 的 N 列
```

当前：

```text
N = 2048
cores = 8
N_per_core = 256
```

因此每 core：

```text
A [T,2048]
B_slice [2048,256]（物理 NZ slice）
C [T,256]
```

8 核拼起来：

```text
[T,2048]
```

其中 T 是覆盖真实 M 的模板阶梯：

```text
16 / 32 / 64 / 128 / 256
```

例如 M=10：

```text
T=16
```

每 core 只流 1/8 的 weight。

---

## 5. 为什么 weight slice 大小是 1 MiB/core

完整 weight FP16：

```text
2048 * 2048 * 2B
= 8 MiB
```

按 N=256 列切成 8 份：

```text
2048 * 256 * 2B
= 1 MiB/core
```

8 份总大小仍是 8 MiB。

所以 small path 做一次性的 weight relayout：

```text
full NZ weight
  -> 8 个 contiguous per-core NZ slice
```

不是多复制 8 倍 weight，而是重新排成 8 个总计仍约 8 MiB 的分片布局。

runtime 的 `kWeightSlicesBytes = 8 MiB` 正是这个来源。

---

## 6. 为什么需要 weight re-layout，而不能每次从完整 NZ 里跳着取列

理论上每 core 可以对完整 NZ B 做 strided 访问，只算自己 256 列。

但这会有几个问题：

```text
地址计算复杂
访存不连续
Matmul API 的 B view/tiling 不一定高效
每次 kernel 都要承担 gather-like 成本
```

所以当前选择在首个 eager 使用时做一次 D2D re-layout：

```text
B_full -> B_slice0 ... B_slice7
```

后续每次 small-M 都直接消费连续 slice。

这是典型的：

```text
一次性 preprocessing 成本
换取大量 decode 热路径重复收益
```

适合“weight 进程生命周期稳定、请求很多次复用”的 serving 场景。

---

## 7. 为什么 slice cache 按 weight pointer 做 key

不同层有不同 `o_proj/out_proj` 权重。

因此不能全模型只建一份 slice。

runtime 维护：

```text
(weight_ptr -> slices_ptr)
```

首次某层 small-M：

```text
查不到
 -> malloc 8MiB
 -> build slices
 -> cache
```

以后同一个 layer weight 再进 small-M：

```text
直接复用
```

这依赖一个 serving 假设：

```text
模型 weight pointer 在进程生命周期内稳定
```

对于正常 inference worker 这是合理的。

---

## 8. 为什么 slice cache 还要限制 64 条

每层约：

```text
8 MiB
```

64 条：

```text
512 MiB
```

如果不设上限，某些病态调用或不断产生新 weight pointer 的场景会无限吃 NPU 内存。

所以当前选择：

```text
最多 64 个 weight slice entry
```

这是一种“正常 serving 足够、异常使用有硬上限”的工程保护。

---

## 9. T 为什么不是直接等于 M

AscendC Matmul 静态模板更适合有限集合的编译期 shape。

如果每个 M=1..255 都单独实例化：

```text
255 个 kernel symbol
```

编译、binary、warmup、维护都很重。

所以当前用 staircase：

```text
M 1..16    -> T=16
17..32     -> T=32
33..64     -> T=64
65..128    -> T=128
129..255   -> T=256
```

这在两个目标之间折中：

```text
减少 padding 浪费
减少模板数量
```

例如 M=17：

```text
T=32
浪费 15 行
```

而不是像 batch path 一样直接算 256 行。

---

## 10. 为什么 M=255 反而 small path 的 padding 优势不大

M=255：

```text
T=256
```

只浪费 1 行。

此时 small path 仍可能有价值，主要不再来自“少算 padding”，而来自：

```text
N-split 让每 core 只消费 1/8 weight slice
```

但是当 M 接近 256 时，M-split 每 core 的计算利用率也开始变好。

因此：

> small-path threshold=256 并不是理论必然最优点，而是当前实测和实现复杂度下的选择。

如果 CANN/Matmul 性能变化，threshold 应重新扫。

---

## 11. 当前代码注释里的实测数字告诉了我们什么

设备代码注释记录：

```text
full-B M-split:
约 150us/core weight-stream floor

1MiB N-slice:
MM 约 53us
```

还记录过一个尝试：

```text
baseN=16, baseK=1024 的 full-32K-slot tiling
测得约 90/241us，反而更差
```

重点不是记住 53、150、90、241 这些数字。

重点是优化方法：

```text
发现小 M 慢
 -> 提出“可能是完整 weight stream floor”假设
 -> 改并行维度减少每核 weight 工作集
 -> 实测
 -> 再尝试更激进 tiling
 -> 实测更差
 -> 拒绝那个方案
```

这就是正确的性能工程闭环。

---

## 12. small path 的输出为什么变成 blocked layout `[8][T][256]`

每个 core 负责一个 N block：

```text
core0 -> N[0:256]
core1 -> N[256:512]
...
core7 -> N[1792:2048]
```

为了让每个 core 连续写自己的结果，send slot 排成：

```text
[core][row][n_in_block]
= [8][T][256]
```

这样每个 core 的 C：

```text
[T,256]
```

是连续的。

8 个 core block 又首尾放在同一 slot，整个：

```text
T*2048*2 bytes
```

仍是一块连续 SDMA payload。

所以 N-split 改变了 slot 内 layout，但没有破坏“一次 signal 搬一整块连续内存”的通信要求。

---

## 13. 为什么不能把 `[8][T][256]` 直接当 `[T][2048]` 使用

因为内存顺序不同。

正常 row-major `[T,2048]`：

```text
row0: N0..N2047
row1: N0..N2047
...
```

blocked `[8][T][256]`：

```text
core0: row0 N0..255, row1 N0..255, ...
core1: row0 N256..511, row1 N256..511, ...
...
```

所以收到 peer payload 后不能直接用普通 contiguous add kernel。

必须做“unblocking reduce”。

这就是 `mf310pAddBlockedKernel` 存在的原因。

---

## 14. blocked add 的地址公式怎么理解

最终输出仍要求：

```text
out[row, col]
```

对于某个 256-column segment：

```text
nBlock = colBase / 256
```

在 blocked source 里的 offset：

```text
srcOff = nBlock * T * 256
       + row * 256
```

这正好定位：

```text
第 nBlock 个 core 的第 row 行 256 列
```

然后：

```text
send[srcOff:srcOff+256]
+
recv[srcOff:srcOff+256]
```

写回正常 row-major output 对应 segment。

所以 add kernel 同时做两件事：

```text
TP SUM
+
blocked -> normal output layout 转换
```

这是一个很好的融合点：既然 anyway 要读 send/recv 做加法，就顺便在写 output 时恢复布局，不需要单独 transpose kernel。

---

## 15. 为什么 small path 仍然用一个 signal，而不是 8 个 core 各发一段

虽然结果由 8 个 N block 组成，但 `[8][T][256]` 整体在内存中连续。

因此：

```text
core0 等 8 ready
 -> signal(slot, length=T*2048*2)
```

仍然只需要一封 data mail。

这保留了和 batch path 类似的 transport 协议：

```text
一个 wave/batch identity
一个连续 payload
一个 wait
```

避免把 MemFabric 协议复杂度乘 8。

---

## 16. small path 为什么当前是 serial wave，不做多 batch pipeline

small-M 定义本身：

```text
M < 256
```

最多只有一个 T<=256 的工作块。

没有多个 M batch 可以流水。

所以顺序自然是：

```text
prepare
 gate
 stage A
 N-split producer
 wait peer
 blocked add
 quiet
 ack
```

这条路径优化目标不是 pipeline depth，而是降低：

```text
单次 tiny-M 固定开销
```

因此它和大 M path 优化重点不同。

---

## 17. 为什么 input staging 仍然需要 scratch

真实 x：

```text
[M,2048]
```

kernel 模板要求：

```text
[T,2048]
```

所以把 M valid rows copy 到 producer scratch。

T 之后的行可以是 stale 数据，原因与 tail 相同：

```text
每个输出 row 只依赖自身 input row
```

blocked add 最终只 reduce：

```text
validRows=M
```

因此 `[M,T)` 这些脏行永不进入最终输出。

这再次利用 row-locality 避免每次 memset。

---

## 18. small path 的通信 payload 也会 padding，但比 batch path 小得多

M=10：

```text
T=16
payload = 16*2048*2
= 65536B
= 64KiB
```

如果走 q=1 batch path：

```text
payload = 256*2048*2
= 1MiB
```

差：

```text
16 倍
```

所以 N-split 不只是减少 MM padding，也明显减少 peer SDMA payload。

对于 decode 小 M，这一点很重要。

---

## 19. 用 M=63 做完整数字推演

M=63：

```text
T=64
```

每 core weight slice：

```text
1 MiB
```

每 core output：

```text
64*256*2
= 32768B
= 32KiB
```

8 core 总 slot：

```text
256KiB
```

通信 payload：

```text
64*2048*2
= 256KiB
```

相比 q=1 batch path 1MiB：

```text
通信量缩小 4 倍
```

计算 padding：

```text
64/63 ≈ 1.016倍
```

而不是：

```text
256/63 ≈ 4.06倍
```

所以这类 M 正是 N-split 特别有吸引力的区域。

---

## 20. weight slicing 的一次性成本什么时候值得

如果某层只执行一次，先花时间：

```text
malloc + 8 次 D2D re-layout
```

可能根本不划算。

但 LLM serving 中同一个 layer 会在：

```text
每个 decode step
每个 request
```

反复执行。

因此只要：

```text
首次 slice build 成本
<<
后续累计 small-M 节省
```

就值得。

可以写一个简单 break-even：

```text
T_build / (T_old_small - T_new_small)
```

得到需要多少次调用才能回本。

例如：

```text
T_build=2ms
每次省 100us
```

20 次调用后回本。

真实 serving 通常远超这个调用次数。

---

## 21. 为什么不能简单说“N-split 永远比 M-split 好”

大 M 时 N-split 会带来：

```text
每 core 都读完整 A
输出 blocked layout
需要 weight slice
需要 unblocking add
```

而 M-split：

```text
每 core 只读自己 A rows
C 天然 row-major
直接普通 add
```

所以并行维度选择本质是一个 shape-dependent tradeoff。

可以粗略理解：

```text
小 M：减少 weight 工作集更重要 -> N-split
大 M：充分利用每核行计算 + 简单输出布局 -> M-split
```

这正是高性能 kernel 常见的“按 shape dispatch 多策略”。

---

## 22. 后续怎么继续优化 small-M

真正值得实验的变量包括：

```text
threshold 是否真应固定 256
T staircase 是否需要更多/更少档
N_per_core 是否固定 256
是否全部 8 core 都适合 M=1/2/4
weight slice layout 是否最优
blocked add 是否成为新瓶颈
A staging copy 是否可以避免
payload 是否可以只传 valid M 而不是 T
```

但每个方向都要问：

```text
节省的是什么？
增加的是什么？
```

例如“只传 valid M”会减少 SDMA bytes，但 kernel/协议需要处理任意长度和对齐；也许 transport 固定成本才是主导，少几十 KiB 没收益。

所以必须 profiler 驱动。

---

## 23. 本章自测

1. 小 M 下 M-split 的两个主要浪费分别是什么？
2. 为什么 N-split 每 core 只需要 1 MiB weight slice？
3. T staircase 解决了什么工程问题？
4. `[8][T][256]` 与 `[T][2048]` 内存顺序有什么本质区别？
5. `mf310pAddBlockedKernel` 为什么同时承担 reduce 和 unblocking？
6. M=63 时 small path payload 是多少？q=1 batch path 又是多少？
7. 为什么 stale padded rows 不影响 valid output？
8. 为什么 N-split 不适合直接推广到所有大 M？
9. 如何用 break-even 思维判断 weight re-layout 是否值得？

把这些讲明白，就真正理解了 small-M 路径是“从性能证据推导出的另一套并行算法”，而不是一个临时补丁。
