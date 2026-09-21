# 310P3 TP=2 FP16 o_proj + MemFabric 当前架构

> 本文只描述**当前 HEAD 的实现结构与不变量**，不记录 bring-up 历史。  
> 固定目标见 [需求与验收合同](310p_memfabric_o_proj_requirements.md)；  
> 当前完成度、证据和下一步任务见
> [AI-native 开发状态与执行计划](310p_memfabric_o_proj_development_plan.md)。

## 1. 当前实现一句话

目标 Qwen3.6 full-attention `o_proj` 被路由到一个 opaque custom op。
该 op 在单条 torch NPU stream 上执行：

```text
first wave:
  stream sync -> MemFabric control barrier

later wave:
  gate(previous peer ack)

producer (8 blocks):
  FP16 matmul chunk
  -> direct write send arena
  -> 64B line clean
  -> ordered MemFabric mailbox request
  -> AICPU epoch SDMA

waiter:
  quiet(local posted transfers)
  -> wait(peer arrival mails x chunks)

reduce:
  repo-owned multi-block FP16 add(send, recv -> output)

ack:
  signal peer ack slot (imm = wave index)
```

后续 wave 不再有 per-wave host barrier。

## 2. Model-side 路由

入口文件：

```text
vllm_ascend/_310p/ops/memfabric_o_proj.py
vllm_ascend/_310p/quantization/modelslim_config.py
```

只有同时满足以下条件才命中：

```text
feature gate enabled
model_type == qwen3_5_moe_text
prefix == *.layers.N.self_attn.o_proj
layer_types[N] == full_attention
tp_size == 2
global K == 4096
local K == 2048
N == 2048
unquantized 310P linear route
params_dtype == FP16
```

命中后：

```text
layer._ascend_310p_memfabric_o_proj = True
layer.reduce_results = False
MemFabricOProjLinearMethod310.apply()
  -> torch.ops._C_ascend.memfabric_direct_o_proj_allreduce(...)
```

因此 generic RowParallelLinear 不会再次发起 HCCL allreduce。

## 3. PyTorch op 与 host runtime

注册：

```text
csrc/memfabric_o_proj_binding.cpp
```

公开的本功能 op：

```text
memfabric_direct_o_proj_allreduce(Tensor x, Tensor weight, int tp_rank, int tile_m)
memfabric_o_proj_shutdown()
memfabric_o_proj_debug_snapshot()
```

主 runtime：

```text
csrc/memfabric_o_proj_runtime.cpp
```

process-persistent 状态包括：

- MemFabric context / symmetric pool layout；
- `producer_scratch`：尾块 staging；
- producer bucket warmup bitmap；
- waiter/add/ack-gate warmup 状态；
- `posted_seqs`：本 rank 已占用的 request-ring 全局序号；
- `wave_count`：ack/gate wave generation；
- poison flag + first failure reason。

任一 wave 中途抛错后 runtime 被 poison，后续 forward 直接失败，要求双 rank 重启。

## 4. 内存布局

默认 `tile_m=32`：

```text
chunk_bytes = 32 * 2048 * 2 = 128 KiB
arena_bytes = 64 * chunk_bytes = 8 MiB
local physical contribution = 32 MiB
```

每个 rank segment：

```text
+0 MiB          send arena, 8 MiB
+8 MiB          recv arena, 8 MiB
+16 MiB         ack slot, 8 bytes
...             currently unused gap
segment tail    V5 SDMA reserved region, 48 KiB
```

对端使用相同 offset，因此本端可直接计算 `peer_recv_arena` /
`peer_ack_slot`。

send 与 recv 不 alias。reserved tail 完全由 MemFabric mailbox-ring epoch
协议管理，host 不应把它当普通 workspace 使用。

## 5. Direct producer

设备代码：

```text
csrc/memfabric_o_proj/external/memfabric310p_device.asc
```

producer 使用静态 M bucket：

```text
16, 32, 64, 128, 256, 512, 1024, 2048, 4096
```

每个 chunk 的核心顺序：

```text
SetTensorA/B
-> IterateAll
-> result direct to send slot
-> Mf310pCleanRegion(64B lines)
-> Mf310pPostSlotOrdered(...)
```

P6 后 producer 采用最多 8 blocks。block `b` 处理交错 chunk：
`b, b+B, b+2B, ...`。每个 chunk 仍使用同一已验证的单核 GEMM recipe。

多个 block 不直接无序写 request ring。host 给每次 launch 传入
`first_seq = posted_seqs`，kernel 在 `Mf310pPostSlotOrdered` 中等待：

```text
REQ_TAIL == seq
and
REQ_TAIL - REQ_HEAD < 64
```

满足后才发布该 slot，再推进 tail。这样保持 FIFO，不需要原子化 ring writer。

## 6. Wait / reduce / ack-gate

### Waiter

```text
smem_shm_sdma_quiet_at(res)
for each chunk:
    smem_shm_sdma_wait_at(res)
```

`quiet` 证明本 rank 的 posted transfer 已落到 peer；`wait` 按 FIFO 消费本
rank 收到的 peer data mails。

### Reduce

当前不是 `at::add_out`。实际使用 repo-owned：

```text
mf310pAddKernel
```

它是 shape-agnostic、多 block、TQue 双缓冲的 FP16 vector add，避免
torch_npu elementwise add 对新 output shape 首次触发约 90 ms GE compile。

### Ack/gate

每个 wave 的 data mails 后追加一封 ack mail：

```text
[data x N, ack]
```

接收端消费顺序：

```text
waiter(data x N)
gate(ack)
```

ack 在本 rank reduce 完成后发往 peer ack slot，`imm = wave_index`。
下一 wave 的 gate 收到该 ack 才允许 producer 继续，从而证明 peer 已结束上一
wave 对 recv arena 的读取。

只有 `wave_count == 0` 时保留一次：

```text
aclrtSynchronizeStream
-> smem_shm_control_barrier
```

用于 pool-creation rendezvous。

## 7. Tail chunk

当 `M % tile_m != 0`：

1. 选择覆盖有效行数的最小 power-of-two bucket；
2. 清零 process-persistent scratch；
3. D2D copy 有效 rows 到 scratch；
4. 必要时清零 send slot 未覆盖部分；
5. tail producer 从 scratch 读取，禁止越界读原始 `x`。

因此 tail 不依赖未初始化数据。

## 8. Kernel warmup

dav-2002 上每个新 kernel symbol 的首次 launch 可能是 silent eager-load no-op。
当前 runtime 对 producer bucket、waiter、add、ack/gate 都执行两次
side-effect-free warmup shape，并只做 stream-level sync。

这是当前平台合同的一部分，不能在没有实机证据时删除。

## 9. Adapter 边界

```text
csrc/memfabric_o_proj/external/memfabric310p_adapter_api.h
csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp
```

adapter ABI 当前为 **v4**。它把定制 MemFabric 类型隔离在 adapter 内部，
主 runtime 不 include `smem*` 头。

CMake：

```text
cmake/memfabric_310p.cmake
```

只从显式 `VLLM_ASCEND_310P_MEMFABRIC_ROOT` 使用 V5 split install 布局，
并自动以 `bisheng --npu-arch=dav-2002` 编译
`memfabric310p_device.asc`。生产运行必须显式提供
`MF_SDMA_ORCH_JSON=<root>/hybm/aicpu_kernel/libmf_sdma_orch_v6.json`。

## 10. 当前平台硬约束

当前服务器上的 V5 epoch 模式仍有三个生产 blocker/约束：

1. pool 存活期间 device-wide synchronize 会等待常驻 epoch task；
2. 当前 epoch 自限与 launch timeout 组合导致约 25 秒生命周期风险；
3. teardown 后 HDC 状态不可靠，不应继续执行设备操作。

因此硬件 benchmark 会把 HCCL reference 放在建池前，建池后只做
stream-level sync，并在 shutdown 后直接退出。

这些是当前平台事实，不是最终产品设计。对应根治任务见 AI-native 状态文档。

## 11. 调试

`memfabric_o_proj_debug_snapshot()` 可读取双 rank 关键 mailbox 状态：

- request head/tail；
- quiet / arrival stamps；
- arrival mail image；
- own/peer arena 首字；
- ack/gate 协议异常残留。

出现 hang 时优先判断“数据面没有推进”还是“数据面已完成但宿主 device-sync 卡住”。

## 12. 源码地图

```text
vllm_ascend/_310p/ops/memfabric_o_proj.py
  eligibility / reduce_results takeover / fused op call

vllm_ascend/_310p/quantization/modelslim_config.py
  unquantized linear dispatch

csrc/memfabric_o_proj_binding.cpp
  torch op registration

csrc/memfabric_o_proj_runtime.cpp
  persistent state / wave orchestration / tail / poison

csrc/memfabric_o_proj/external/memfabric310p_adapter_api.h
  repo-owned ABI v4

csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp
  installed MemFabric V5 bridge / layout

csrc/memfabric_o_proj/external/memfabric310p_device.asc
  producer / waiter / add / ack / gate kernels

cmake/memfabric_310p.cmake
  custom MemFabric build/link contract

tests/ut/_310p/test_memfabric_o_proj_source.py
  source invariants

benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py
  TP=2 single-layer correctness + latency

benchmarks/scripts/analyze_310p_memfabric_overlap.py
  V5 overlap evidence analysis
```

## 13. 文档事实优先级

发生冲突时按以下顺序处理：

1. 实际 HEAD 源码；
2. 310P3 实机行为；
3. 定制 MemFabric 当前 ABI/source；
4. 本架构文档；
5. commit message / 历史讨论。

历史 P0-P6 记录不再作为当前实现说明。
