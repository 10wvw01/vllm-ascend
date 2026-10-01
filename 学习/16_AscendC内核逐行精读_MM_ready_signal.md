# 16｜AscendC 内核逐行精读：8 核 MM、ready、cache clean、signal 到底怎么拼起来

> 主文件：`csrc/_310P/memfabric_mm_ar/memfabric310p_device.asc`
>
> 目标：看到 `Mf310pDirectProducerKernel` 时，不只是知道“8 核做 MM”，而是能自己算出每个 core 读哪段 A、写哪段 C、为什么 B 不切、为什么 ready cell 是 64B、为什么 core0 可以安全 signal。

---

## 1. 先把 kernel 的输入翻译成人话

核心 producer：

```cpp
Mf310pDirectProducerKernel(
    gva,
    xGm,
    wGm,
    sendArena,
    peerRecvArena,
    producerControl,
    protocolStatus,
    batchIndex,
    generation,
    batchBytes)
```

可以按职责分成四组。

### 计算输入

```text
xGm = 当前通信 batch 的 A
wGm = 当前 rank 的 B/weight
```

### 通信地址

```text
sendArena     = 本 rank partial output 写入区
peerRecvArena = peer rank 的接收区 GVA
```

### 同步/错误状态

```text
producerControl = ready cells
protocolStatus  = fail-stop 错误码
```

### 当前 batch 身份

```text
batchIndex
current generation
batchBytes
```

所以这个 kernel 不是一个纯 MM kernel，它的职责是：

```text
算 partial C
-> 让 C 对 SDMA 可见
-> 证明 8 核全部完成
-> 发起这一个 batch 的 peer SDMA
```

---

## 2. 先理解 AscendC Matmul 这里为什么需要“手工 8 核切 M”

代码注释明确指出：

```text
classic AscendC Matmul API is single-core semantics
```

这句话非常关键。

很多人看到：

```cpp
rt.usedCoreNum = 8;
```

会误以为 Matmul 库自动把一个 `[256,2048] @ [2048,2048]` 大矩阵分给 8 个核。

当前实现不是这样。

每个 block 如果都给相同 A/B/C base，它们会从自己的视角执行同一个单核 tiling。

因此代码必须显式给每个 block 不同的 A/C 起始地址。

核心：

```cpp
constexpr uint32_t kRowsPerCore = BATCH_M / 8;
const uint64_t rowOff = block * kRowsPerCore;
```

然后：

```text
A base += rowOff * K
C base += rowOff * N
B base 不变
```

这才真正形成“8 核合作完成一个 batch”。

---

## 3. q=1 时每个 core 到底算什么

当前：

```text
baseM = 256
q=1
BATCH_M = 256
K = 2048
N = 2048
cores = 8
```

因此：

```text
rowsPerCore = 256 / 8 = 32
```

block0：

```text
A rows [0,32)
C rows [0,32)
```

block1：

```text
A rows [32,64)
C rows [32,64)
```

...

block7：

```text
A rows [224,256)
C rows [224,256)
```

每个 core 看到的逻辑 MM：

```text
A_core [32,2048]
B      [2048,2048]
C_core [32,2048]
```

8 个 C_core 在内存上首尾相接，组成：

```text
C_batch [256,2048]
```

这就是 M-split。

---

## 4. q=2/q=4 只是每核 M 增大，不改变 N/K

### q=2

```text
BATCH_M = 512
rowsPerCore = 64
```

每核：

```text
[64,2048] @ [2048,2048] -> [64,2048]
```

### q=4

```text
BATCH_M = 1024
rowsPerCore = 128
```

每核：

```text
[128,2048] @ [2048,2048] -> [128,2048]
```

所以 q 调节的是：

```text
每次 producer launch 处理多少 M
```

它同时改变：

```text
MM 粒度
SDMA payload
ready/signal 次数
wait/add 次数
pipeline batch 数
```

这就是为什么 q 不是“通信参数”或“MM 参数”中的单独一种，而是整个流水粒度。

---

## 5. 为什么 B 不按 M-split 一起切

每个输出元素：

```text
C[m,n] = sum_k A[m,k] * B[k,n]
```

当我们按 M 分时，一个 core 只负责某些 m，但它仍然要计算完整的 n=0..2047。

所以它必须看到完整 B 的所有 N 列和 K 行。

因此 direct path：

```text
A 按行切
C 按行切
B 完整共享
```

这有一个性能副作用：

> 即使某个 core 只有很少的 M 行，它仍然面对完整 2048×2048 weight 的访问模式。

这正是 small-M 路径后来改 N-split 的根因之一。

---

## 6. `MatmulType` 四个模板参数在表达什么

代码：

```cpp
Mf310pANd  = GM, ND, FP16
Mf310pBNzT = GM, NZ, FP16, true
Mf310pCNd  = GM, ND, FP16
```

可以简化理解成：

```text
A: GM 里的普通二维 ND 排布
B: GM 里的 FRACTAL_NZ 排布，并按 Matmul API 的 transpose 语义使用
C: GM 里的普通 ND 排布
```

为什么 B 是 NZ？

310P 上 unquantized Linear 的 weight 在 load 后处理阶段已经转换成适合 NPU Cube/Matmul 的 NZ 格式。

所以 fused method 继承 stock `AscendUnquantizedLinearMethod` 很重要：

```text
weight format 的责任仍由原有 Linear infrastructure 承担
kernel 直接消费已经处理好的 NZ weight
```

如果自己绕过这套权重后处理，kernel 即使 shape 数字一样，也可能因为 layout 不同算错。

---

## 7. tiling 为什么是 `{Mcore, 2048, 2048}` + basic `{Mcore,256,64}`

宏：

```cpp
MatmulShapeParams {
    MB/8,
    2048,
    2048,
    MB/8,
    256,
    64
}
```

可以理解成：

```text
单 core 总问题：
M = BATCH_M/8
N = 2048
K = 2048

基础块：
basicM = 单 core 全 M
basicN = 256
basicK = 64
```

例如 q=1：

```text
Mcore=32
basicM=32
basicN=256
basicK=64
```

N 方向需要：

```text
2048 / 256 = 8 个 basicN 块
```

K 方向需要：

```text
2048 / 64 = 32 个 basicK 块
```

所以一个 core 的工作本质是沿 K 方向累加、沿 N 方向铺满 2048 列。

当前代码使用：

```text
CONFIG_NORM
```

而不是 `CONFIG_MDL`，注释记录在目标 dav-2002 上，小 Mcore 场景使用 MDL 会出现 on-chip MTE buffer overrun，因此这里不是随便选的模板模式，而是实机约束后的结果。

---

## 8. `SetOrgShape()` 为什么还要再告诉一次形状

代码：

```cpp
mm.SetOrgShape(
    kRowsPerCore,
    2048,
    2048,
    2048,
    2048);
```

静态 tiling 描述“怎样切”，OrgShape 描述这次 operand 的真实逻辑矩阵关系。

对于初学者可记成：

```text
static tiling = 编译期工作模板
org shape    = 当前这块 A/B/C 的矩阵语义
```

两者共同让 Matmul API 知道地址跨度和运算维度。

---

## 9. C 写到哪里：把地址公式完整算一遍

当前 batch slot：

```cpp
slot = sendArena + batchIndex * batchBytes
```

core 的 C 起点：

```cpp
slot + rowOff * N * sizeof(FP16)
```

q=1，N=2048，FP16=2B。

每一行大小：

```text
2048 * 2 = 4096 B = 4 KiB
```

每 core 32 行：

```text
32 * 4096 = 131072 B = 128 KiB
```

因此：

```text
core0 C offset =   0 KiB
core1 C offset = 128 KiB
core2 C offset = 256 KiB
...
core7 C offset = 896 KiB
```

整个 batch：

```text
256 * 4096 = 1,048,576 B = 1 MiB
```

这正好等于 q=1 的 `batchBytes`。

所以 8 核各写连续 128 KiB，恰好拼成一块连续 1 MiB SDMA payload。

这个“输出天然连续”是 M-split direct path 很重要的好处。

---

## 10. 为什么 MM `IterateAll` 返回后还不能 signal

从“这个 core 的代码执行完”到“peer SDMA 一定读到新数据”，中间还有 cache visibility 问题。

AI Core 对 GM 写入后，数据可能仍停留在本地 cache 层级。

如果马上：

```text
signal SDMA
```

SDMA 可能从更下层内存看到旧值。

因此每个 core 先：

```cpp
Mf310pCleanRegion(自己写的 C 区域)
```

然后才发布 ready。

正确顺序是：

```text
MM 完成
  -> clean C
  -> write ready
  -> core0 确认所有 ready
  -> signal
```

这条顺序构成一个非常重要的 happens-before 关系：

> “ready == current generation” 不仅表示 core 算完，还表示这个 core 的 C 已经被清到 SDMA 可见层级。

---

## 11. `Mf310pCleanRegion` 为什么按 64B line 循环

代码每次：

```text
64 bytes
```

做 single-cache-line clean+invalidate。

q=1 时每 core C=128 KiB：

```text
128 KiB / 64 B = 2048 条 cache line
```

8 核合计：

```text
16384 条 line
```

为什么不直接 flush 全 cache？

因为全 cache flush 会破坏更多仍有价值的数据，并扩大固定成本。

当前实现只 clean：

```text
这个 core 真正写过的连续 C rows
```

这叫 ownership-scoped cache maintenance。

它的思路是：

```text
谁写的数据，谁负责让它对通信 engine 可见
```

而不是 core0 替所有 core 清整块 1MiB。

---

## 12. ready cell 为什么每个占 64B，不是 8B

逻辑上 generation 只需要：

```text
uint64_t / uint32_t
```

几字节就够。

但布局：

```text
ready[batch][core] 每 cell 独占 64B
```

地址：

```cpp
cell = (batchIndex * 8 + core) * 64
```

目的不是节省内存，而是隔离 cache line。

如果 8 个 core 的 ready 紧挨着放在同一个 cache line：

```text
core0 写 byte0
core1 写 byte8
...
```

多个 core 会对同一 cache line 做更新/clean，产生 false sharing 和可见性复杂度。

现在：

```text
一个 ready = 一条独占 cache line
```

每个 core 只 flush 自己的 line，协议简单很多。

代价是空间浪费：

```text
一个 8-core batch ready = 8*64 = 512B
```

最大 64 batch：

```text
64*512 = 32768B = 32KiB
```

这正是代码里经常提到“32 KiB ready 区”的来源。

---

## 13. generation 为什么比 boolean ready 更好

如果 cell 只写：

```text
1 = done
```

下一波复用同一 cell 时，上一波的 1 仍然存在。

core0 可能在其他 core 还没开始算时就看到：

```text
ready == 1
```

从而提前 signal。

最直接修法：

```text
每波 memset ready=0
```

但 32KiB 控制区清零在当前 GVA pool 上有明显固定开销。

所以 eager 路径使用单调 generation：

```text
waveA = 0x40000000
waveB = 0x40000001
...
```

core0 判断：

```text
cell == current generation
```

旧 cell 即使是任意历史值，也不会误判。

这实际上是一个版本化 barrier。

---

## 14. 为什么 Graph generation 又不能直接沿用 eager 逻辑

Graph capture 后 replay 的 launch 参数需要满足图的稳定执行合同。

当前 graph path 使用：

```text
generation = batch + 1
```

这是每次 replay 都一样的小值。

既然 generation 会重复，历史 ready 就可能冲突，所以 graph wave 需要把 ready clear 操作捕获进图。

于是代码刻意把两个值域分开：

```text
Graph: 1..max_batches
Eager: 从 0x40000000 起
```

目的就是避免：

```text
Graph 留下的 ready 值
恰好被 eager wave 当成当前值
```

这是一个非常典型的“Graph replay 与 eager 共用控制内存”问题。

---

## 15. 为什么只有 core0 调 `signal()`

8 个 core 最终共同写一整块 C batch。

如果每个 core 都 signal：

```text
同一个 batch 会产生 8 个 transport 请求
```

还需要定义每个 core 搬哪一段、peer 如何拼接、mail 如何匹配。

当前公共 MemFabric API 和协议更适合一个完整连续 payload：

```text
slot [batch_m,2048]
```

所以设计是：

```text
8 核都算
8 核都 clean
8 核都 ready
只有 core0 等待全部 ready 后 signal 整块
```

core0 不是“通信专核”。

它先和其他 7 核一样完成自己的 MM，然后额外做一个很短的 coordinator 角色。

这样不会损失 1/8 MM 计算资源。

---

## 16. core0 等待 ready 的循环为什么还要反复 clean/invalidate cell

读取普通 GM 地址时，core0 自己的 cache 可能保留旧 ready 值。

所以 `Mf310pWaitReady()` 每轮对 cell 做 line clean/invalidate，再读取 volatile 值。

目的不是把数据写出去，而是强迫自己重新观察共享内存的最新状态。

你可以把它理解成：

```text
其他 core 写 + flush
core0 invalidate + reload
```

双方共同建立可见性。

如果只写 `volatile` 而不解决 cache coherency，volatile 只能约束编译器，不等价于跨 core cache 一致性协议。

---

## 17. signal 的 dst 为什么是 `peerRecvArena + batchIndex*batchBytes`

本 rank batch i 写：

```text
sendArena[i]
```

希望 peer 收到同样 i 的 partial：

```text
peer.recvArena[i]
```

所以：

```cpp
dst = peerRecvArena + batchIndex * batchBytes
```

mail 的 immediate 也传：

```text
imm = batchIndex
```

于是 receiver wait 时可以同时验证：

```text
这条 mail 的地址是不是 batch i 的 slot？
长度是不是 batchBytes？
imm 是不是 i？
```

这就是为什么协议即使 payload shape 都一样，也不会只凭“来了一封 mail”就直接消费。

---

## 18. wait kernel 为什么很小，但非常重要

`mf310pWaitBatchKernel` 只有一个 block 真正工作：

```text
smem_shm_sdma_wait(gva)
```

然后检查：

```text
status
m.dst
m.len
m.imm
```

它不做 `quiet()`。

这个选择是性能设计的核心：

```text
wait = 等 peer 这个 batch 已到，可开始读 recv
quiet = 等自己之前发出的请求全部 drain
```

两者语义不同。

如果每次 wait 前都 quiet，自身后续 SDMA 无法保持在飞状态，pipeline 会被人为串行化。

---

## 19. add kernel 为什么不直接让 core0 做

收到 peer partial 后需要：

```text
out = send + recv
```

一个 batch q=1 有：

```text
256*2048 = 524288 FP16 elems
= 1 MiB input A
+ 1 MiB input B
+ 1 MiB output 写
```

这是明显的 vector/memory 工作，不应该只让单核做。

`mf310pAddKernel` 最多用 8 block，把 element range 切成多个连续 slice。

每个 block 使用双 buffer queue：

```text
MTE2 load send
MTE2 load recv
Vector Add
MTE3 store out
```

双 buffer 的目标是让：

```text
搬下一 tile
与
算当前 tile / 写上一 tile
```

尽可能流水。

所以整个 MM+AR 实际上已经在利用多种硬件单元：

```text
Cube -> MM
SDMA -> peer copy
Vector/MTE -> local reduce
```

计算通信协作优化的本质，就是安排这些 engine 的时间关系。

---

## 20. fail-stop 为什么直接 `Trap()`

协议异常：

```text
READY_TIMEOUT
SIGNAL_FAILED
WAIT_FAILED
MAIL_MISMATCH
...
```

发生后，继续返回一个 tensor 是非常危险的。

因为你无法保证：

```text
send/recv arena 哪部分是新的
mail 队列处于什么状态
peer 是否已经进入下一阶段
credit 是否还对称
```

所以 device 先写 `protocolStatus`，flush 后：

```cpp
AscendC::Trap()
```

host runtime 同时会把 context 标记 poisoned。

这是一种明确的选择：

```text
宁可 fail-stop
不要用可能污染的通信状态“尽量继续跑”
```

对于分布式协议，这通常比局部恢复更安全。

---

## 21. 读这份 kernel 时应该建立的因果链

不要背函数名，应该能顺着下面讲：

```text
为什么 8 核？
因为一个 batch MM 要并行。

为什么手工 rowOff？
因为当前 classic Matmul 是单核语义，不会自动切 M。

为什么 B 完整共享？
因为 direct path 是 M-split，每个 row 仍需完整 N/K weight。

为什么 C 连续？
因为每核负责连续 row，方便拼成一个 SDMA payload。

为什么 clean？
因为 SDMA 必须看到最新 C。

为什么 ready？
因为 core0 不能在其他核还没完成时发送整块。

为什么 generation？
因为 slot 会跨 wave 复用，不能把旧 ready 当新 ready。

为什么 core0 signal？
因为一次 batch 只需要一条连续 peer copy。

为什么 wait 校验 mail？
因为流水中多个 batch shape 相同，错配不会自动 crash。

为什么 wait 不 quiet？
因为要保留 SDMA overlap。
```

如果能完整说出这条链，就已经不是“看懂代码行”，而是理解了设计。

---

## 22. 自测计算题

### 题 1

q=2 时：

```text
batch_m=?
batch_bytes=?
每 core rows=?
每 core C bytes=?
```

答案应自己推：

```text
512
2 MiB
64
64*2048*2 = 256 KiB
```

### 题 2

q=4，一个 batch 的 ready cell 控制区实际仍占多少？

提示：ready 数量按 `batch × 8 core`，不是按 payload bytes。

一个 batch仍然：

```text
8*64B = 512B
```

### 题 3

如果把 ready 写在 cache clean 之前，会发生什么？

正确回答必须涉及：

```text
core0 观察到完成
但 SDMA 尚不能保证观察到最新 C
```

而不是只说“可能有 race”。

---

下一章会继续把 host runtime 的 arena、wave、credit、lookahead 与这里的 kernel 对接起来。只有把 host enqueue 和 device kernel 放在同一条时间线上，才能真正理解这个“融合算子”为什么这样设计。
