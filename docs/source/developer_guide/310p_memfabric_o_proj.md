# 310P3 TP=2 BF16 o_proj + MemFabric AllReduce 融合算子设计文档

> 状态：研发中，当前分支已经具备 Phase-1 代码骨架与服务器 bring-up 基础设施，但尚未完成 Ascend 310P3 实机编译和数值/性能验收。
>
> 目标分支：`feat/310p-w8a8-o-proj-memfabric-ar`
>
> 关联文档：
>
> - [研发计划与当前状态](310p_memfabric_o_proj_development_plan.md)
> - [编译、部署与模型启动说明](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)
> - [OpenCode 后续研发提示词](310p_memfabric_o_proj_opencode_prompt.md)

## 1. 背景与目标

目标是在一张 Ascend 310P3 双 die 卡上，以 TP=2 运行 `Eco-Tech/Qwen3.6-35B-A3B-w8a8` 时，将 full-attention 的 `o_proj` 与其紧随其后的 Tensor Parallel AllReduce 做成一条面向 310P 的融合流水。

普通 `RowParallelLinear` 路径为：

```text
BF16 local o_proj matmul
        |
        v
TP all-reduce
```

本项目目标路径为：

```text
MM tile[t]
    -> local partial result
    -> MemFabric publish
    -> AICPU/SDMA peer exchange
    -> local reduce

同时 MM tile[t+1] 继续执行，不等待 tile[t] 的通信完成。
```

最终目标不是简单地用另一个通信 API 替换 HCCL，而是让计算和通信形成生产者/消费者流水，实现“边算边搬”。

## 2. 固定适用范围

当前实现刻意保持窄范围，避免影响其他模型和其他算子。

- 硬件：Ascend 310P3，单物理卡、双 die。
- Tensor Parallel：严格 TP=2。
- 模型：Qwen3.5/Qwen3.6 MoE text trunk；本项目实机目标为 `Eco-Tech/Qwen3.6-35B-A3B-w8a8`。
- 目标层：仅 `full_attention` 的 `self_attn.o_proj`。
- 不命中：`linear_attn` / GDN / MLP / MoE expert linear / 其他 RowParallelLinear。
- 量化：未量化 BF16（2026-09-18 负责人裁决 A，实机核查目标 checkpoint 的
  full-attention o_proj 为 FLOAT 条目；原 W8A8 契约废弃，见研发计划 4.11）。
- 输出 dtype：BF16。
- 通信：定制 `wgm-dev-310p` MemFabric Hybrid 的 AICore/AICPU/SDMA 路径。
- 融合算子内部不使用 HCCL/MC2；HCCL 仅可作为 correctness reference。

Qwen3.6 目标几何：

```text
global attention o_proj input K = 16 heads * 256 = 4096
TP = 2
local K = 2048
hidden size / N = 2048

per rank:
A[M, 2048] x W[2048, 2048] -> Y_local[M, 2048]
```

## 3. 非目标

当前阶段明确不做以下泛化：

- TP>2；
- 多物理卡通信；
- 任意模型/任意 hidden size；
- W8A8/INT8 量化路径回归（除非负责人再次裁决）；
- FP16/INT32 通信路径泛化；
- linear-attention/GDN 融合；
- 通用 collective 库替代品；
- 在未证明数值等价前传输 INT32 accumulator。

窄契约是为了先在 310P3 上把正确性、同步语义和 overlap 做实，再决定是否泛化。

## 4. 与 vLLM-Ascend 的集成点

### 4.1 RowParallelLinear 语义

上游 `RowParallelLinear` 通常执行：

```text
quant_method.apply(...) -> output_parallel
if reduce_results and tp_size > 1:
    tensor_model_parallel_all_reduce(output_parallel)
```

本功能只对严格符合目标契约的 `o_proj` 做以下处理：

1. 在权重 post-process 阶段识别目标 layer；
2. 将该 layer 标记为 `_ascend_310p_memfabric_o_proj=True`；
3. 设置 `layer.reduce_results=False`，阻止 generic RowParallelLinear 再发起一次 TP AllReduce；
4. 310P W8A8 scheme 的 `apply()` 进入 MemFabric 特殊路径；
5. 特殊路径返回的 Tensor 已经是 TP=2 reduce 后结果。

这样可以避免修改整个 Qwen 模型实现，也不会改变非目标 linear 的行为。

### 4.2 精确 eligibility

只有同时满足以下条件才允许进入融合路径：

- feature gate `VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1`；
- `model_type == "qwen3_5_moe_text"`；
- layer prefix 为 `*.layers.N.self_attn.o_proj`；
- `config.layer_types[N] == "full_attention"`；
- `tp_size == 2`；
- `input_size == 4096`；
- `input_size_per_partition == 2048`；
- `output_size == 2048`；
- 310P 未量化 BF16 路由（`AscendUnquantizedLinearMethod`，modelslim
  description 中该层为 FLOAT 条目）；
- `params_dtype == torch.bfloat16`。

Qwen hybrid decoder 的 linear-attention 路径使用 `linear_attn` 命名，因此不会被 `self_attn.o_proj` 条件命中；额外检查 `layer_types[N]` 用于防止未来上游命名变化导致误匹配。

## 5. MemFabric 依赖关系

本项目依赖的是独立编译安装的定制 MemFabric：

```text
GDD_ESCC/memfabric_hybrid:wgm-dev-310p
              |
              | build + install
              v
custom 310P MemFabric install prefix
              |
              | headers + libraries
              v
vllm_ascend_C
```

不是：

```text
official/upstream MemFabric + 额外插件
```

也不存在单独部署的 `vllm_ascend_memfabric310p_adapter.so`。仓库内 `mf310p_*` adapter 是源码隔离层，直接编译进 `vllm_ascend_C`。

CMake 只在显式给出的 `VLLM_ASCEND_310P_MEMFABRIC_ROOT` 内查找：

- `smem.h`
- `smem_shm.h`
- `smem_shm_aicore_sdma.h`

并使用 `NO_DEFAULT_PATH`，防止误链接系统中的官方/上游 MemFabric。

## 6. 核心流水设计

### 6.1 单向依赖

正确的依赖关系是：

```text
MM[t] -> publish[t] -> SDMA[t] -> arrival[t] -> REDUCE[t]
  |
  +--------------------------------------------> MM[t+1]
```

关键约束：

- `MM[t+1]` 不等待 `SDMA[t]`；
- `REDUCE[t]` 等待 peer arrival；
- op 最终返回前，必须等待本次调用全部 MM、SDMA、arrival 和 local reduce 完成。

### 6.2 TP=2 AllReduce 简化

因为 world size 固定为 2，不需要 ring/tree collective。

每个 rank 都执行：

```text
rank0.send[t] --SDMA--> rank1.recv[t]
rank1.send[t] --SDMA--> rank0.recv[t]

rank0.final[t] = rank0.send[t] + rank0.recv[t]
rank1.final[t] = rank1.send[t] + rank1.recv[t]
```

通信 payload 为已经完成 W8A8 dequant 的 BF16 tile。

## 7. 内存布局与 race 规避

不能让 MM 直接写一个随后被 local reduce 原地覆盖、同时又可能被 peer SDMA 读取的 buffer，否则会出现：

```text
peer SDMA read(send)  <->  local reduce write(send)
```

因此每个 rank 使用两个不重叠的 symmetric arena：

```text
send arena
  保存 local MM partial result
  在本 wave 完成前保持 immutable

recv/final arena
  peer SDMA 写入
  peer arrival 后执行 local add
  得到该 rank 的最终 reduced result

arrival flags
  peer SDMA 到达通知

workspace
  MemFabric SDMA mailbox / AICPU orchestration
```

Phase-1 当前每个 wave 最多 64 个 chunk，默认：

```text
tile_m = 32
chunk = BF16 [32, 2048]
chunk_bytes = 32 * 2048 * 2 = 128 KiB
```

128 KiB 与已提供的 310P AICore/AICPU/SDMA 示例量级保持一致。

## 8. Phase-1：staged correctness/overlap baseline

当前代码首先复用标准 BF16 matmul（310P 实机已验证可用）：

```python
torch.mm(x_tile, weight)  # weight [K_local, N] BF16
```

每个 M tile 的执行顺序：

```text
BF16 matmul(x_tile)
        |
        v
copy local result -> symmetric send[t]
        |
        v
AICore cache clean
        |
        v
smem_shm_sdma_notify(mailbox[t], src, words)
        |
        +----> AICPU/SDMA 搬到 peer.recv[t]

与此同时 compute stream 可以 enqueue MM[t+1]
```

当前 Phase-1 的两个已知额外成本：

1. matmul result -> symmetric send 的一次本地 copy；
2. 每个 M tile 一次 `npu_quant_matmul` launch。

它的意义是先验证通信协议和 overlap，而不是最终性能形态。

## 9. Phase-2：最终 direct tiled producer

最终目标是一个 AscendC/CATLASS 级 tiled BF16 producer：

```text
Cube BF16 MM tile[t]
    -> rank0-only bias (if any)
    -> direct write symmetric send[t]
    -> DataCacheCleanAndInvalid
    -> smem_shm_sdma_notify(t)
    -> continue tile[t+1]
```

这样移除 Phase-1 的 local copy 和 per-tile ACLNN matmul 调度成本。

对于大 M/prefill，优先 M-row tiling；对于 decode 小 M，M-only tiling 很可能没有足够流水深度，后续需要评估 N-panel tiling。

## 10. BF16 数值语义

融合算子必须保持未量化路径语义：

- 输入/权重均为 BF16，本地 matmul 数值与 `torch.mm` 一致；
- output dtype 保持 BF16；
- o_proj 无 bias；若未来带 bias，只在 TP rank 0 应用一次（TP sum 后不重复）；
- 本地 reduce 为 BF16 -> F32 add -> BF16 round（与 allreduce FP32 求和对齐）。

## 11. Runtime 生命周期

MemFabric 资源必须是 process-persistent：

- `smem_init` / `smem_shm_init`；
- symmetric SHM；
- SDMA workspace；
- arrival flags；
- reduce stream；
- stable GVA。

禁止每个 forward 创建/销毁 pool。这既是性能要求，也是 ACL graph stable-address 要求。

当前 runtime 还实现了 failure poison：一旦某个 wave 在 submit/publish/wait/reduce 中途失败，不允许静默复用同一 context，因为可能仍有 AICPU polling、SQE 或 stale flag 存活。bring-up 阶段遇到此类错误应重启对应 worker。

## 12. Wave 重用与同步

重复推理时，arrival flag 和 mailbox 必须安全重用。

理想 wave-boundary 协议为：

```text
rank0/rank1 均结束上一 wave
        -> 双 rank rendezvous
        -> 清本地 mailbox + arrival flags
        -> 双 rank rendezvous
        -> 新 wave submit / publish
```

但是当前已经拿到的 example 只证明：

- `smem_shm_sdma_notify`
- `smem_shm_sdma_poll_flag`
- `smem_shm_sdma_submit`
- `smem_shm_sdma_wait`
- `smem_shm_sdma_get_workspace`
- `smem_shm_sdma_get_result`

示例 host 侧使用 `MPI_Barrier`，还没有证明定制 MemFabric 存在 `smem_shm_control_barrier()` 或等价 API。因此 `mf310p_prepare_wave()` 的最终 barrier 实现必须以 `wgm-dev-310p` 源码为准，不能继续假设私有 ABI。

## 13. ACL Graph 考虑

310P model runner 会使用 ACL graph capture/replay，因此生产版本必须满足：

- symmetric memory 地址稳定；
- workspace/flag 地址稳定；
- 不在 replay 中重新创建 MemFabric runtime；
- wave generation/reset 语义可重复；
- 不把 host barrier 错放进每个 tile 的关键路径。

实机 bring-up 顺序建议先 `--enforce-eager` 验证 correctness，再去掉 eager 单独验证 graph capture/replay。

## 14. 代码地图

```text
vllm_ascend/_310p/ops/memfabric_o_proj.py
  model-side eligibility + Phase-1 tiled producer orchestration

vllm_ascend/_310p/quantization/methods/w8a8_static.py
  310P W8A8 dispatch integration

csrc/memfabric_o_proj_binding.cpp
  PyTorch custom op registration

csrc/memfabric_o_proj_runtime.cpp
  process-persistent runtime state / arena wrapping / wave lifecycle

csrc/memfabric_o_proj/external/memfabric310p_adapter_api.h
  repo-owned narrow C ABI

csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp
  direct calls into installed wgm-dev-310p MemFabric

csrc/memfabric_o_proj/external/memfabric310p_device.asc
  AICore publish + correctness-baseline reduce

cmake/memfabric_310p.cmake
  explicit custom MemFabric build/link contract

benchmarks/scripts/bench_310p_memfabric_o_proj.py
  TP=2 MemFabric communication/reduce hardware bring-up harness

tests/ut/_310p/test_memfabric_o_proj_source.py
  import-free source regression tests
```

## 15. 剩余私有 MemFabric 信息

要删除最后的 ABI 假设，需要在 310P 服务器上从 `wgm-dev-310p` 确认：

1. `smem_shm_aicore_sdma.h` 的真实内容；
2. `smem_shm_sdma_submit/wait/get_workspace/get_result` 的 host declaration；
3. `smem_shm_sdma_submit` 背后的 AICPU mailbox/SQE 编排；
4. chunk 的 src/dst offset 递增规则；
5. arrival flag 写入值、顺序和复用规则；
6. 是否有定制 host/device barrier/rendezvous API；
7. example 08 的 `.asc` 实际编译命令、object 格式和链接方式；
8. 定制 MemFabric install 后的真实 library 名称和依赖顺序。

## 16. 验收标准

最终完成至少需要满足：

1. TP=2 bilateral MemFabric exchange + local reduce 数值正确；
2. 连续重复 wave 不出现 stale flag、丢 notify、reset race；
3. Phase-1 结果与现有 `torch.mm + TP allreduce`（FP32 math）对齐；
4. profiler 能证明 `MM[t+1]` 与 `SDMA[t]` 有实际重叠；
5. Qwen3.6 full-attention `o_proj` 才命中，其他层保持原路径；
6. eager 与 ACL graph 均能稳定重复运行；
7. Phase-2 direct producer 数值通过；
8. 端到端模型 generation correctness 通过；
9. 性能数据至少包含 TTFT/prefill latency、decode latency/TPOT、tokens/s、融合 op latency 和 overlap timeline；
10. feature gate 关闭时普通 vLLM-Ascend 行为不变。
