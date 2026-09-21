# 310P3 TP=2 FP16 o_proj + MemFabric 融合设计

> 本文只描述当前 public-API ABI v6 设计。
>
> 需求基线：[310p_memfabric_o_proj_requirements.md](310p_memfabric_o_proj_requirements.md)
>
> 实机计划：[310p_memfabric_o_proj_development_plan.md](310p_memfabric_o_proj_development_plan.md)
>
> 使用说明：[310p_memfabric_o_proj_usage.md](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)

## 1. 总体架构

普通 RowParallelLinear：

```text
local o_proj MM
    -> generic TP allreduce
```

目标路径：

```text
Python eligibility
    -> one torch custom op
        -> persistent runtime
            -> 7 MM workers + 1 coordinator
                -> MemFabric public signal/wait/quiet
            -> local FP16 add
```

命中目标层后 `reduce_results=False`，generic HCCL allreduce 被融合 op 取代。

## 2. 代码地图

```text
vllm_ascend/_310p/ops/memfabric_o_proj.py
  eligibility / layer configuration / torch-op call

vllm_ascend/_310p/quantization/modelslim_config.py
  unquantized target-layer dispatch

csrc/_310P/custom_memfabric_o_proj/
  README.md
  memfabric_o_proj_binding.cpp
  memfabric_o_proj_runtime.cpp
  memfabric_o_proj_torch_adpt.h
  memfabric310p_adapter_api.h
  memfabric310p_adapter.cpp
  memfabric310p_device.asc

cmake/memfabric_310p.cmake
  public MemFabric package + dav-2002 device build

tests/ut/_310p/test_memfabric_o_proj_source.py
  source-level architecture guard
```

本功能所有定制 C++/AscendC 实现都隔离在
`csrc/_310P/custom_memfabric_o_proj/`。

## 3. MemFabric public contract

Host 使用公开 API 完成：

- init / shm init；
- symmetric pool create/destroy；
- symmetric size 查询；
- SDMA readiness 查询；
- 初始化时 control barrier。

Device 数据面只使用：

```cpp
smem_shm_sdma_signal(gva, src, dst, len, imm)
smem_shm_sdma_wait(gva)
smem_shm_sdma_quiet(gva)
```

vLLM 不读取 MemFabric private transport state。

## 4. Persistent context 与应用内存

默认 local pool 为 32 MiB。vLLM 在公开 symmetric rank segment 的应用区域内
使用：

```text
send arena
recv arena
8B ack slot
```

peer recv 地址由 `pool_base + peer_rank * symmetric_size + recv_offset`
计算，`symmetric_size` 来自 public API。

普通 `aclrtMalloc` 额外分配：

- producer ready control；
- protocol status；
- tail scratch。

这些不是 MemFabric internal workspace。

## 5. 7+1 producer pipeline

最多 8 个 AICore block：

```text
block 0     communication coordinator
block 1..7  MM workers
```

worker 对 logical chunk 交错分配：

```text
worker0: 0, 7, 14, ...
worker1: 1, 8, 15, ...
...
worker6: 6, 13, 20, ...
```

worker：

```text
MM(chunk)
-> send slot
-> cache clean send slot
-> ready[logical_chunk] = 1
-> clean ready cache line
```

ready 每个 slot 独占 64B cache line。

coordinator：

```text
for logical chunk in order:
    wait ready[chunk]
    signal(send_slot -> peer_recv_slot, imm=physical_chunk)
```

只有 block0 调 public `signal()`，不依赖 multi-producer 行为。

`signal()` 只负责异步提交，所以前面已提交 chunk 的 SDMA 可以与其它 worker 的
后续 MM 重叠。

## 6. Wait 与 mail validation

producer kernel 后，同 stream 启动 waiter：

```text
quiet()
for c in wave chunks:
    mail = wait()
    validate status/dst/len/imm
```

数据 mail 预期：

```text
dst = recv_arena + c * chunk_bytes
len = chunk_bytes
imm = c
status = OK
```

任一 mismatch 都进入 fail-stop。

## 7. Local reduction

waiter 完成后启动 repo-owned multi-block FP16 add：

```text
out = send_arena + recv_arena
```

output 是普通 NPU Tensor，不与 symmetric arena alias。

## 8. Fixed-credit wave protocol

recv arena 跨 layer/call/wave 复用。为防止 peer 在本端 add 尚未读完时覆盖
recv arena，使用应用级 fixed credit。

初始化：

```text
control barrier
rank0 signal CREDIT -> rank1 ack slot
rank1 signal CREDIT -> rank0 ack slot
quiet
stream sync
```

每个 wave：

```text
prepare status
-> gate: wait CREDIT
-> producer
-> waiter
-> add
-> ack: signal CREDIT
```

credit 使用固定 `MF310P_CREDIT_TAG`。correctness 由 fixed credit 保证，因此 graph replay 不需要额外 host generation 状态。

credit mail 同样严格校验 `status/dst/len/imm`。

## 9. Fail-stop

vLLM-owned protocol status 区分：

- READY_TIMEOUT
- SIGNAL_FAILED
- QUIET_FAILED
- WAIT_FAILED
- MAIL_MISMATCH
- CREDIT_MISMATCH
- ACK_FAILED

异常路径：

```text
write protocol status
-> cache clean
-> AscendC::Trap()
```

不得用不完整 recv 数据继续 add/ack。

debug snapshot 只读应用层 layout/arena/status，不读取 MemFabric internals。

## 10. Tail 与 multiple waves

`tile_m` 默认 32，可配置为 [16,4096] 内 2 的幂。

- full chunk：直接读取输入；
- tail：persistent scratch zero-pad 后用相应 static M bucket；
- 每 wave 最大 64 chunks；
- 超过容量时 runtime 分 multiple waves；
- 每个 wave 都执行完整 fixed-credit protocol。

## 11. ACL Graph

Graph capture 允许 torch-npu 管理的 capture side stream，但 capture 内禁止：

- 首次 create MemFabric context；
- 首次分配 scratch；
- 首次初始化 fixed credit；
- lazy producer bucket warmup；
- warmup sync/barrier。

这些必须在 eager/profile 阶段完成。

普通 eager 仍要求单 execution stream，避免同一 context 上多个
coordinator/ack 并发使用 public `signal()`。

Graph-used context 当前按 process lifetime 处理，因为 replay 可能运行在 host
函数不可见的 stream 上，shutdown 不能仅凭 eager stream 证明全部 replay 已
quiescent。

## 12. Teardown

纯 eager context：

```text
public quiet
-> synchronize eager stream
-> free vLLM scratch/control
-> smem_shm_destroy
```

quiet/sync 失败时不强行 destroy 可能仍有 in-flight SDMA 的 pool。

## 13. Build

feature-on 时 `cmake/memfabric_310p.cmake`：

1. 从 `MEMFABRIC_HYBRID_HOME_PATH` 查 public host/device headers；
2. 查 `libmf_smem.so`；
3. bisheng `--npu-arch=dav-2002` 编译当前
   `memfabric310p_device.asc` 为 `libmf310p_device.so`；
4. 将 adapter 编进 `vllm_ascend_C`；
5. 链接 public MemFabric library。

仓库不保存本功能的预编译 object。

## 14. 当前验证状态

当前代码已经完成 source-level 架构审查和静态门禁，但必须在目标 310P3 上对
**当前 commit** 重新完成 build、correctness、long-run、overlap、Qwen eager
和 ACL Graph 验收。

