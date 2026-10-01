# 19｜MemFabric 协议与 Graph 正确性：把“不会错”证明出来

> 这一章专门做一件事：不再只说“有 ready / credit / wait / quiet”，而是把它们放进一套明确的 happens-before 关系里，解释为什么当前协议能避免读旧数据、错 batch、arena 覆盖和 Graph replay 污染。
>
> 主文件：
>
> - `csrc/_310P/memfabric_mm_ar/memfabric310p_adapter_api.h`
> - `csrc/_310P/memfabric_mm_ar/memfabric310p_adapter.cpp`
> - `csrc/_310P/memfabric_mm_ar/memfabric310p_device.asc`
> - `csrc/_310P/memfabric_mm_ar/memfabric_mm_ar_runtime.cpp`

---

## 1. 先区分三种“同步”

本项目里至少有三类同步，名字看起来都像“等一下”，但保护对象不同。

### 1）核内/核间完成同步

```text
ready[batch][core]
```

保护：

```text
这个 batch 的 8 份 C 是否都写完并对 SDMA 可见
```

### 2）rank 间数据到达同步

```text
signal / wait
```

保护：

```text
peer 的 partial 是否已经到本 rank recv slot
```

### 3）wave 级资源复用同步

```text
quiet / credit gate / ack
```

保护：

```text
整个 arena 是否可以安全被下一 wave 重写
```

很多协议 bug 都来自把这三层混成一个“同步”概念。

---

## 2. MemFabric 在当前设计里被当成什么

当前 ABI v7 的原则是：

```text
MemFabric = opaque transport
```

vLLM-Ascend 不去读取：

```text
private request ring
mailbox 内部布局
head/tail
reserved slot
SQE
orchestrator 私有状态
```

只使用公共边界：

### Host public/control

```text
create/destroy
get_layout
control_barrier
exchange_geometry
```

### Device public data path

```text
smem_shm_sdma_signal
smem_shm_sdma_wait
smem_shm_sdma_quiet
```

为什么这很重要？

因为一旦 vLLM 依赖 private ring layout，MemFabric 内部版本升级就可能直接破坏融合算子。

当前 adapter 的作用就是把外部库封装成一个稳定的内部 ABI。

---

## 3. 为什么还要有 `memfabric310p_adapter_api.h`

初学者容易问：

```text
既然 MemFabric 有 public API，vLLM 为什么不直接到处 include 它？
```

因为 vLLM 侧还需要一个自己的稳定合同。

`mf310p_layout_t` 暴露的是**本融合真正需要知道的应用级几何**：

```text
send_arena
recv_arena
peer_recv_arena
ack_slot
peer_ack_slot
arena_bytes
batch_bytes
expected_credit_dst
expected_recv_base
arena_rows
batch_m
max_batches
```

它不把 MemFabric 的 private internal structs 传播到 runtime。

这样：

```text
外部 MemFabric public API 变化
      |
      v
adapter.cpp 调整
      |
      v
vLLM runtime 尽量不变
```

这就是 adapter layer 的真正价值：

> 隔离依赖边界，而不是为了多写一层代码。

---

## 4. 为什么 ABI version 要显式检查

runtime 初始化时：

```cpp
mf310p_adapter_abi_version()
== VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION
```

如果不检查，最危险的情况不是链接失败，而是：

```text
函数名相同
struct 大小/字段语义变了
代码还能跑
但地址或协议理解已经不一致
```

这会变成静默错数或随机 hang。

所以 ABI version 是“双方协议解释必须一致”的硬门槛。

---

## 5. 一次 batch 从 rank0 到 rank1 的 correctness 链

假设 batch i。

rank0 的 core c：

```text
1. MM 写 C0_i,c
2. clean 自己写的 C cache lines
3. ready[i][c] = generation
4. flush ready cell
```

rank0 core0：

```text
5. invalidate/reload ready[i][0..7]
6. 确认全部 == generation
7. signal(send_i -> rank1.recv_i)
```

rank1：

```text
8. wait() 得到 mail
9. 校验 dst/len/imm/status
10. local add(rank1.send_i, rank1.recv_i)
```

从正确性上，我们要证明：

```text
rank1 add 读取到的 recv_i
一定对应 rank0 本轮 batch i 的完整新 partial
```

下面逐条证明。

---

## 6. 为什么 ready 必须在 cache clean 之后

我们希望建立：

```text
C 写入并可见
happens-before
ready 发布
```

于是 core0 看到 ready 后，才能把它当成“C 可安全发送”的证据。

如果顺序反了：

```text
ready
 -> cache clean
```

core0 可能：

```text
看见 ready
立即 signal
SDMA 开始读 C
```

而真正的新 C 还没被 clean 到 SDMA 可见层级。

这会产生：

```text
控制状态是新的
数据却是旧的
```

这是最典型、也最难排的并发错误。

因此 ready 的语义不是：

```text
“计算函数返回了”
```

而是：

```text
“数据已经满足下一阶段可见性条件”
```

---

## 7. 为什么 core0 自己也必须写 ready 再参与等待

有人可能觉得：

```text
core0 自己当然知道自己已经完成
没必要等 ready[core0]
```

理论上可以特判。

但当前统一协议让每个 core 都做：

```text
clean -> ready
```

core0 再检查 8 个 cell，包括自己。

优点是：

```text
完成条件统一
可见性条件统一
没有“core0 走另一套语义”的隐藏分支
```

对于并发协议，统一状态机往往比省一个 cell 读取更重要。

---

## 8. 为什么 mail 要校验四个字段

`wait` 后检查：

```text
status == OK
dst    == expected slot
len    == expected batch bytes
imm    == batch index
```

### status

确认 transport 本身成功。

### dst

确认数据写到了当前 rank 预期的 recv 地址。

### len

确认不是别的 payload shape。

### imm

确认 mail 的逻辑 batch identity。

为什么 dst 和 imm 两个都要？

因为它们验证不同维度：

```text
dst = 物理地址身份
imm = 逻辑 batch 身份
```

两个同时匹配，才能降低“地址算对但逻辑顺序错”或“batch id 对但目标地址错”的风险。

---

## 9. 为什么不能只依赖 FIFO 顺序

假设当前代码“相信 transport mail 一定严格按 signal 顺序到达”，于是只写：

```text
wait 一封
就当 batch0
再 wait 一封
就当 batch1
```

如果未来 transport 实现、并发路径、异常恢复导致顺序与假设不同，payload shape 又完全一样：

```text
batch0 和 batch1 都是 1MiB
```

程序不会因为 size 不一致而 crash。

它只会把 token 行错位相加，产生静默错数。

因此当前协议把 batch identity 编进 mail，并做 strict validation。

这是“不要让 correctness 依赖隐含 transport 行为”的设计。

---

## 10. GVA geometry exchange 解决什么问题

`mf310p_exchange_geometry()` 的注释指出：

```text
mail dst fields carry sender-space GVAs
receiver validation uses peer mapping learned here
```

简单理解：

通信两边看到的是对称/映射后的全局虚拟地址空间，但接收方验证 mail 的 dst 时，不能只拿“本地普通指针”盲比。

双方在初始化阶段通过 control network 交换：

```text
各自 arena / peer mapping 几何
```

然后得到：

```text
expected_recv_base
expected_credit_dst
```

后续热路径只做数值校验，不再 host rendezvous。

这再次体现：

```text
复杂的地址协商放初始化
热路径只消费稳定几何
```

---

## 11. 为什么 `control_barrier` 只应该在初始化做

如果每 wave 都做 host barrier：

```text
rank0 host
rank1 host
```

必须同步到同一点后才能继续。

这会直接破坏模型执行流水，并给每层每次 forward 增加巨大固定开销。

当前只在 protocol initialization：

```text
prepare
control_barrier
exchange_geometry
init_credit
sync
```

之后完全靠 device-side data path 和 wave credit 运行。

这是高性能通信系统很常见的原则：

> control plane 慢但低频；data plane 必须快且少 host involvement。

---

## 12. fixed credit 的安全证明

我们定义：

```text
credit 存在
=> peer 允许当前 rank 开始复用下一 wave arena
```

一波末尾当前 rank 执行：

```text
所有 wait/add 已 enqueue
quiet
ack
```

为什么 `ack` 可以表示安全？

因为在同一 stream 中：

```text
add 对 recv/send 的读取
排在 ack 前
```

并且：

```text
quiet 保证本 rank outbound SDMA 已完成
```

因此 ack 发出时：

```text
本 rank 已不再需要上一波 peer 写入的 recv 数据
本 rank 的 outbound transport 也不再依赖上一波 send 数据
```

所以 peer 收到 ack 后，才可以开始下一波覆盖对应 arena。

这就是 credit 的资源所有权转移语义。

---

## 13. 如果把 ack 提前，会出现什么具体错误

错误顺序：

```text
... producer
ack
quiet
```

peer 看到 credit 后开始下一 wave：

```text
覆盖 send/recv slots
```

但当前 rank 的 SDMA 可能仍在读取上一波 send slot。

于是 DMA 读到一半旧、一半新，或者完全新数据。

这不是“性能变差”，而是协议失效。

因此：

```text
quiet -> ack
```

属于 correctness ordering。

---

## 14. Graph replay 为什么会挑战 generation 协议

Eager 每 wave generation 都变：

```text
G0,G1,G2...
```

但 Graph replay 是“把捕获时的命令原样再执行”。

如果 producer kernel 参数里 generation 是 capture 时的常数：

```text
batch0 generation=1
batch1 generation=2
```

每次 replay 都还是 1/2。

上一 replay 结束后 ready cell 也许仍是 1/2。

下一 replay 若不 clear：

```text
core0 一启动就可能看到历史 ready==1/2
```

于是提前 signal。

所以 Graph path 不能只依赖 versioning，必须在图内先 clear ready region。

---

## 15. 为什么 eager generation 从高位值域开始

Graph 使用小值：

```text
1..max_batches
```

如果 eager 也从 1 开始单调：

```text
某次 eager generation=1
```

可能与 Graph 历史 cell 冲突。

因此 eager 从：

```text
0x40000000
```

开始，和 Graph 值域完全分离。

这样即使 eager 与 Graph 共用同一块 ready memory，历史值也不会被另一模式误认。

这叫 namespace separation：

```text
同一物理状态存储
不同执行模式使用不相交逻辑版本空间
```

---

## 16. 为什么 Graph capture 中不能重新做 host barrier / geometry exchange

Graph replay 是设备执行重放。

如果 capture 内含“host 端另一个 rank 必须同时进入某函数”的语义，replay 并不会自动让 host 再执行一遍同样 rendezvous。

所以：

```text
host control-plane 操作
```

必须在 capture 前完成。

Graph 中只能保留：

```text
固定地址上的 device-side prepare/gate/producer/wait/add/quiet/ack
```

这也是 `require_capture_ready_locked()` 如此严格的原因。

---

## 17. 为什么 Graph-used context 销毁这么难

Graph replay 可能不再经过 `memfabric_mm_ar_allreduce_impl()` 的 host 逻辑。

也就是说 runtime 看不到每一次 replay 开始/结束。

进程退出前若调用 destroy：

```text
host 认为模型 stream 空了
```

但另一个 Graph replay stream 是否还持有 pool 地址，runtime 无法可靠证明。

要安全 destroy，理论上需要：

```text
统一 replay stream tracking
或者 device-wide quiescence guarantee
```

当前两者都没有建立完整证明，所以选择不 destroy。

这是生命周期正确性优先于资源“看起来干净”的例子。

---

## 18. 为什么 `torch.npu.synchronize()` 可能和 MemFabric 长生命周期 kernel 冲突

MemFabric pool 启动后可能有自己的 supervised epoch kernel 长期驻留在独立 stream。

如果框架做：

```python
torch.npu.synchronize()
```

语义接近“等整个 device 所有 stream 工作结束”。

但长期驻留 kernel 的目标本来就是不结束。

于是 device-wide sync 可能永久等待。

当前 runner hook 改成：

```python
torch.npu.current_stream().synchronize()
```

只等模型 stream。

这揭示一个系统级事实：

> 引入 persistent communication runtime 后，原本框架中“同步设备”的假设可能不再成立。

---

## 19. fail-stop 需要两层状态：device status + host poisoned

Device 异常时：

```text
protocolStatus = error code
flush
Trap
```

Host 捕获异常后：

```text
state.poisoned=true
failure_reason=...
```

为什么两层都要？

### device status

用于知道实际 protocol kernel 在哪类错误上停了。

### host poisoned

防止下一次 Python 调用重新进入同一通信 context。

如果只有 device trap，没有 host poisoned：

```text
下一次请求可能再次尝试复用半损坏状态
```

如果只有 host exception，没有 device protocol status：

```text
难以定位到底是 ready、wait、mail 还是 credit 出错
```

两层配合才完整。

---

## 20. 如何画当前协议的 happens-before 图

一份最重要的图：

```text
rank0 core0..7                     rank1
---------------                   -----
MM C0_i
   |
cache clean C0_i
   |
ready_i,c
   |
core0 all-ready
   |
signal -------------------------> peer recv_i
                                    |
                                    wait mail i
                                    |
                                    validate
                                    |
                                    add(local_i, recv_i)
                                    |
                                  wave reads done

rank0 outbound all done
   |
quiet
   |
ack ----------------------------> rank1 next-wave gate
```

需要牢牢记住：

```text
ready 建立“计算 -> 发送”的顺序
wait 建立“peer 写入 -> 本地读取”的顺序
credit 建立“上一 wave 使用 -> 下一 wave 覆盖”的顺序
```

这三条就是整个 correctness 骨架。

---

## 21. 如果以后做“边搬运边规约”，协议要改什么

假设未来不是一个 batch 完整 SDMA 后再 add，而是：

```text
chunk0 到 -> reduce chunk0
chunk1 到 -> reduce chunk1
...
```

那现有 mail 粒度可能不够。

必须重新定义：

```text
每个 chunk 的 identity
每个 chunk 的 dst/len
什么时候允许 add
什么时候允许复用源 chunk
quiet/credit 是否仍按 wave
```

还要考虑：

```text
partial recv cache visibility
chunk mail 数量的固定开销
更多 signal/wait 是否抵消 overlap 收益
```

所以“边搬边规约”不是在 add kernel 里加个循环，而是通信协议粒度变化。

---

## 22. 为什么当前 batch_m 必须和协议一起考虑

q 改变：

```text
batch_bytes
mail 数量
signal 数量
wait 数量
ready rendezvous 数量
SDMA 单请求长度
```

因此性能调 q 实际是在调一个 protocol granularity。

q 小：

```text
更早启动通信
更多 mail/signal/wait 固定开销
```

q 大：

```text
更少协议固定开销
但更晚才能发第一块
pipeline 粒度更粗
small tail 浪费可能更大
```

这也是为什么 q 必须通过端到端 profiler 选择，而不是只看单个 MM kernel。

---

## 23. 本章自测

1. ready、wait、credit 各保护什么？
2. 为什么 `ready` 必须在 C cache clean 之后？
3. 为什么 mail 同时校验 dst 和 imm？
4. 为什么 control barrier 不应该进入热路径？
5. `quiet -> ack` 的 correctness 证明是什么？
6. Graph 为什么需要 ready clear，而 eager 可以不 clear？
7. 为什么 eager generation 要避开 Graph 的小值域？
8. 为什么 persistent MemFabric runtime 会改变框架的 device sync 策略？
9. 为什么边搬边规约属于“协议设计变化”，而不只是 kernel 优化？

如果能把每个问题回答成明确的“资源所有权 + happens-before”关系，你就真正开始掌握通信融合的正确性设计了。
