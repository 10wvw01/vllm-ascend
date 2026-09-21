# 310P3 Qwen3.6 未量化 o_proj + MemFabric 融合算子需求目标

> 本文是本功能的**需求基线和最终验收合同**。
>
> 它回答的是“最终必须做成什么”，不是“当前代码怎么实现”。设计方案、研发计划和代码都允许根据 310P 实机结果调整，但不得在没有明确重新评审需求的情况下改变本文定义的目标和约束。
>
> 当不同资料出现冲突时，按以下优先级处理：
>
> 1. 本需求目标文档；
> 2. Ascend 310P3 实机行为以及 `wgm-dev-310p` 定制 MemFabric 的真实源码/ABI；
> 3. [设计文档](310p_memfabric_o_proj.md)；
> 4. [研发计划](310p_memfabric_o_proj_development_plan.md)；
> 5. 当前 feature branch 中的代码实现和历史假设。
>
> 如果实机或定制 MemFabric 真实接口证明当前设计不可行，应修改设计和实现以继续满足本文需求，而不是降低需求来适配现有代码。

> **修订记录（2026-09-18，项目负责人裁决）**：原需求假设目标模型 full-attention
> `o_proj` 为 static W8A8 量化层。实机核查（本地与 modelscope 上游
> `quant_model_description.json` 一致）证明 Qwen3.6-35B-A3B-w8a8 的全部 10 个
> full-attention `o_proj` 均为**未量化 BF16（FLOAT）**；且本机 CANN 9.1 的
> `npu_quant_matmul` 不支持 BF16 输出（仅 INT8/FP16），原"W8A8 + BF16 输出"
> 组合在本机不可实现。经项目负责人裁决（方案 A），融合目标**按真实检查点
> 重定基线为未量化 BF16 o_proj**；W8A8 相关语义条目（activation INT8、
> FRACTAL_NZ、deq_scale、quant_bias）整体移除，其余目标（计算通信重叠、
> MemFabric SDMA、tile 级流水、ACL Graph、性能收益）不变。310P 实机已验证
> BF16 matmul 可用。详见研发计划 4.11。

> **修订记录二（2026-09-21，实机平台事实）**：本机 CANN 9.1.0 的 BF16
> NZ linear 路径不可用（310P 上该 checkpoint 无 torch_dtype，vLLM 以 FP16
> 运行；dav-2002 AICore 标量环境亦无 `bfloat16_t`），融合管线的
> matmul/交换/规约**实际以 FP16 执行**。通信与数值语义目标（FP32 求和
> 语义的单次 round、与 reference bit-exact、计算通信重叠、MemFabric
> SDMA、tile 级流水、ACL Graph、性能收益）不变。同日 V5（epoch API）
> 迁移完成：V4 kfc/workspace 协议相关条款由研发计划 §9.2 记录的实际
> V5 契约取代。

## 1. 项目背景

目标模型 `Eco-Tech/Qwen3.6-35B-A3B-w8a8` 在 Ascend 310P3 单卡双 die、Tensor Parallel=2 场景中，full-attention 的 `self_attn.o_proj` 为 RowParallel **未量化 BF16** 线性层（实机核查定论，见顶部修订记录）。

正常路径可以抽象为：

```text
local FP16 o_proj matmul
    -> materialize local partial output
    -> tensor-parallel allreduce
    -> final o_proj output
```

本项目希望利用 310P3 单卡双 die 的拓扑和定制 `wgm-dev-310p` MemFabric AICore/AICPU/SDMA 能力，把 o_proj 计算与 TP=2 数据交换/归约变成一个专用融合流水，减少独立 collective 带来的串行等待和调度开销，并实现计算通信重叠。

## 2. 一句话最终目标

在 **Ascend 310P3 单卡双 die、TP=2** 上，为 **Qwen3.6-35B-A3B-w8a8 full-attention `self_attn.o_proj`** 提供一个由 vLLM-Ascend 调用的专用融合算子，使其同时拥有 BF16 o_proj 本地矩阵乘和 TP=2 reduction，并通过 **`wgm-dev-310p` 定制 MemFabric** 实现 tile 级“边算边搬”，最终用实机 correctness、profiler overlap 和性能数据证明有效。

## 3. 固定适用范围

第一版生产目标只要求支持以下组合：

- Hardware: Ascend 310P3；
- Topology: 单物理卡、双 die；
- Tensor Parallel: **固定 TP=2**；
- Model: `Eco-Tech/Qwen3.6-35B-A3B-w8a8`；
- vLLM model type: 当前对应 `qwen3_5_moe_text`；
- Layer: **仅 full-attention `self_attn.o_proj`**；
- Quantization: **未量化（BF16 权重，checkpoint FLOAT 条目）**；
- Output dtype: BF16；
- full-attention global input K: 4096；
- TP=2 local input K: 2048；
- output hidden size N: 2048。

当前目标不是做成一个通用 allreduce 库，也不是一次性覆盖其它模型、其它 TP size、其它芯片或全部 RowParallelLinear。

## 4. 必须满足的功能需求

### R1. 只融合目标 full-attention o_proj

融合路径必须只对目标 Qwen3.6 full-attention `self_attn.o_proj` 生效。

至少需要同时满足：

```text
model_type == qwen3_5_moe_text
layer prefix == *.layers.N.self_attn.o_proj
config.layer_types[N] == full_attention
TP == 2
local K == 2048
N == 2048
未量化 BF16 linear（310P AscendUnquantizedLinearMethod 路由）
BF16 output
```

`linear_attention` / GDN 的 `linear_attn` 路径不得误命中。

任何非目标层必须保留原有 vLLM-Ascend 行为。

### R2. 融合算子拥有计算和 TP reduction

对于命中层，最终逻辑不能再是：

```text
fused/local matmul
    -> generic tensor_model_parallel_all_reduce
```

目标融合算子本身必须拥有：

```text
W8A8 local matmul
+ TP=2 peer exchange
+ local reduction
+ final completion/join
```

因此目标层不能在融合结果之后再次执行 generic TP allreduce。

从 vLLM 上层视角，应表现为一个完整的 o_proj+reduction 融合调用，而不是要求模型代码显式拼接一组通信算子。

### R3. 通信必须使用 310P 定制 MemFabric

本功能的生产通信路径必须使用：

```text
GDD_ESCC/memfabric_hybrid : wgm-dev-310p
```

要求：

- **不编译、不安装、不依赖官方/upstream MemFabric 来实现本功能**；
- 先编译安装 `wgm-dev-310p`；
- vLLM-Ascend 直接 include/link 该定制安装产物；
- 不把架构做成“官方 MemFabric + 额外 310P plugin”；
- 不引入额外独立部署的 adapter `.so`；
- 不通过 runtime `dlopen` 绕过真实 build/link 关系；
- 不允许凭经验猜 upstream library 名、soname、ABI 或同步语义。

HCCL 可以用于 reference correctness/latency 对比，但**不得成为目标融合算子的内部通信实现**。

### R4. 必须实现真正的 tile 级计算通信 overlap

目标依赖关系是：

```text
MM[t]
  -> publish[t]
  -> SDMA[t]
  -> peer arrival[t]
  -> local reduce[t]

MM[t] -> MM[t+1]
```

关键硬约束：

```text
MM[t+1] 不得等待 SDMA[t] 或 REDUCE[t]
```

允许的等待点是：

- producer 在复用同一有限 ring slot 之前等待该 slot 可安全复用；
- operator 最终返回之前等待所有计算、发送、接收和 reduction 完成；
- wave boundary 为保证 repeated-wave correctness 所需的双 rank rendezvous。

不允许：

- 每 tile host synchronize；
- 每 tile 两 rank barrier；
- `MM[t] -> wait communication[t] -> MM[t+1]` 的串行流水。

最终是否真正 overlap，必须由 310P profiler timeline 证明，不能仅根据代码“没有显式 wait”推断。

### R5. TP=2 allreduce 数学语义必须正确

对两个 TP rank：

```text
rank0 local = Y0
rank1 local = Y1
```

两端最终都必须得到：

```text
Y = Y0 + Y1
```

TP=2 可以专门化为双向 exchange + local sum，不要求实现通用 ring/tree allreduce。

双方行为必须对称，不能沿用示例中仅为演示而设计的单向 role4/role5 业务语义。

### R6. 必须避免 send-read / reduce-write race

本地 matmul/发送数据和接收/最终结果必须有明确内存所有权。

最低要求：

```text
send/local buffer : peer SDMA 可能读取期间保持 immutable
recv/final buffer : peer 数据落地后供 local reduce 写最终结果
```

禁止在 peer SDMA 仍可能读取本地 partial output 时，对同一地址进行 local reduction 覆盖。

生产实现可以使用 full arena、double buffer 或 ring buffer，但必须证明复用安全。

### R7. repeated-wave 必须可靠

同一 worker 中连续执行融合 op 不能依赖一次性 flag 状态。

必须解决：

- stale arrival flag；
- mailbox slot reuse；
- late clear；
- peer 新 notify 被本地旧 wave memset 擦除；
- SQE/AICPU 上一 wave 尚未真正完成；
- graph replay 后 generation 混淆。

最少要求提供 repeated-wave stress 证据，默认验收门槛为 **连续 1000 次**无 hang、无 mismatch、无 stale flag 触发。

具体 barrier/epoch/flag 协议必须依据 `wgm-dev-310p` 真实实现确定，不能猜不存在的 MemFabric barrier API。

## 5. 必须保持的 BF16 数值语义

融合不能以牺牲现有未量化路径语义为代价。

必须保持：

- 本地 matmul 与 `torch.mm`/`torch.nn.functional.linear` 在相同 BF16 权重上的数值语义一致（同为 BF16 输入、BF16 累加精度行为按 aclnn matmul 默认）；
- output dtype = `layer.params_dtype`（BF16）；
- o_proj 无 bias（Qwen 系列该层无 bias）；若未来权重带 bias，bias 只允许 rank0 应用一次；
- 通信 payload 为本地 partial BF16 result（与对端相加前不做其它数值变换）。

本地 reduce kernel 语义为 BF16 -> F32 add -> BF16 round（与 TP allreduce 的 FP32 求和后回 BF16 对齐，实机已验证 allclose）。除非通过数值推导和实机验证证明完全等价，否则不要为了减少通信量擅自改成其它压缩 payload。

## 6. 分阶段实现目标

### Phase 1：实机 correctness 和 overlap 基线

允许继续使用现有：

```text
per-tile BF16 matmul (torch.mm / F.linear)
    -> copy to symmetric send arena
    -> clean/notify
    -> MemFabric SDMA
    -> peer arrival
    -> local reduce
```

Phase 1 的目的：

- 打通真实 `wgm-dev-310p` build/ABI；
- 证明 TP=2 通信正确；
- 证明 repeated-wave 正确；
- 证明 W8A8 数值正确；
- 证明 tile 级 MM/SDMA 可以真实重叠；
- 跑通 Qwen3.6 eager inference。

Phase 1 是必要里程碑，但**不是最终融合形态**。

### Phase 2：最终 direct tiled producer

最终性能形态需要消除 Phase 1 的主要临时开销。

目标 producer：

```text
AscendC/CATLASS-level BF16 tiled matmul
    -> rank0-only bias (if any)
    -> direct write symmetric send tile
    -> cache clean
    -> smem_shm_sdma_notify(tile)
    -> continue compute next tile
```

需要尽量消除：

- `BF16 matmul output -> send arena` 的额外本地 copy；
- 每个 M tile 单独 ACLNN matmul launch 的调度开销。

优先复用当前 CANN/CATLASS/vLLM-Ascend 中真实可用的 310P BF16 matmul primitive，不要求为了形式上的“自研”从零手写 Cube matmul。

## 7. Decode 与 Prefill 需求

### Prefill

M 较大时，优先使用 M-row tiling，让：

```text
MM row tile[t+1]
```

与上一 row tile 的 SDMA/reduce 重叠。

### Decode

M 很小时，单纯 M-row tiling 可能退化成 1 个 tile，无法获得有效 overlap。

因此在大 M 路径稳定之后，需要评估 N-panel 或其它适合小 M 的 tiling。

是否实施 N-panel、panel size 取多少，必须由 310P 实机 benchmark/profiler 决定，不在需求层预设固定数值。

## 8. ACL Graph / vLLM runtime 兼容需求

最终实现不能只在 eager 模式可用。

由于 310P vLLM runtime 使用 ACL Graph，生产验收还要求：

- MemFabric symmetric pool/process context 生命周期稳定；
- graph capture/replay 期间关键 device address 稳定；
- workspace/flags 不在每 forward 重建；
- repeated graph replay 不读旧 generation；
- 不同 graph/M bucket 行为正确；
- graph 模式下仍保持正确完成语义。

第一次 bring-up 可以使用 `--enforce-eager` 降低变量，但 eager 通过不代表项目完成。

## 9. Feature gate 和回退要求

本功能必须默认关闭。

关闭 feature 时：

- 原有 310P W8A8 路径保持不变；
- 原有 generic TP allreduce 保持不变；
- 不要求系统安装定制 MemFabric；
- 不得影响其它模型和硬件构建。

打开 feature 时：

- 如果定制 MemFabric build inputs 缺失，应 fail fast；
- 不允许静默回退到 HCCL 并假装融合已启用；
- runtime 发生半 wave 失败后，必须防止危险状态被静默复用。

## 10. 明确非目标

以下事项不是第一版需求，不应让研发偏离主线：

- TP>2 的通用 allreduce；
- 多机 MemFabric collective；
- Ascend 910B/910C/950 支持；
- 非 Qwen3.6 模型通用化；
- linear-attention/GDN 融合；
- 所有 RowParallelLinear 自动融合；
- 替换整个 vLLM distributed subsystem；
- 为了“统一抽象”重构大量无关 vLLM-Ascend 代码；
- 把 upstream/official MemFabric 重新引入本功能依赖；
- 先做漂亮框架、后补实机 correctness。

如果某项工作不能直接推进本文的 correctness、overlap、性能或生产可用性，应降低优先级。

## 11. 验收要求

### A. Build 验收

必须在真实 310P 服务器记录并可重复：

- `wgm-dev-310p` commit；
- 定制 MemFabric build/install command；
- install prefix；
- 实际 headers；
- 实际 libraries 和 link order；
- `.asc` 的真实编译规则和产物形式；
- vLLM-Ascend 最终 build command。

要求 feature-on build 成功，feature-off build 不依赖 custom MemFabric。

### B. 通信 correctness 验收

独立 MemFabric benchmark 至少覆盖：

```text
M = 1, 8, 32, 64, 128, 512, 2048
```

要求：

- 两 rank 结果均与 TP=2 HCCL reference allclose；
- tail chunk 正确；
- repeated-wave >= 1000 次稳定；
- 无 hang、无随机 mismatch。

### C. BF16 单层 correctness 验收

对目标 o_proj 比较：

```text
reference = torch.mm(local, weight) + TP allreduce(FP32 math)
fused     = tiled BF16 MM + MemFabric exchange + local reduce
```

至少覆盖：

- 多种 M；
- tail M；
- repeated calls；
- feature on/off；
- full-attention 命中与 linear-attention 不命中。

误差门槛应根据现有 BF16/reference 行为在实机确定并记录，不允许通过人为放宽 tolerance 掩盖算法错误。

### D. Qwen3.6 E2E 验收

模型：

```text
Eco-Tech/Qwen3.6-35B-A3B-w8a8
```

至少要求：

- TP=2 eager 可以启动；
- full-attention o_proj 确实走融合路径；
- linear-attention/GDN 不走融合路径；
- 短 prompt 正常生成；
- prefill 正常；
- 多轮请求稳定；
- feature-off baseline 正常；
- 去掉 `--enforce-eager` 后 ACL Graph capture/replay 正常。

### E. Overlap 验收

必须保存 profiler/timeline 证据，能够观察到至少一个代表性 workload 中：

```text
MM[t+1]
```

与：

```text
SDMA[t] / peer receive / local reduce[t]
```

存在真实设备侧时间重叠。

没有 profiler 证据时，不得把该需求标记为完成。

### F. 性能验收

必须同时记录 baseline 和 fused：

- communication microbenchmark；
- target o_proj microbenchmark；
- 代表性 prefill；
- 代表性 decode；
- 至少一组 Qwen3.6 end-to-end 指标。

项目目标是取得**可重复、可解释的正向性能收益**。本文不预先虚构百分比目标；最终收益阈值由实机 baseline 数据和项目评审确定，但如果没有正收益，应继续定位瓶颈，不能只因 correctness 通过就宣告性能项目完成。

## 12. 最终交付物

项目完成时，feature branch/PR 至少应包含：

- 可编译的融合算子实现；
- 精确 model hook；
- `wgm-dev-310p` direct build/link 集成；
- TP=2 MemFabric runtime；
- repeated-wave 安全协议；
- Phase-2 direct tiled W8A8 producer；
- 可接受的 local reduce 实现；
- ACL Graph 兼容；
- source/unit regression；
- 310P hardware correctness test/benchmark；
- profiler overlap 证据或其采集说明；
- baseline/fused 性能数据；
- 完整设计文档；
- 研发计划/状态文档；
- 编译部署与模型启动使用说明。

## 13. OpenCode / 后续开发者的目标锁定规则

后续自动化开发代理或工程师每次开始新阶段前，都必须重新确认：

```text
我当前做的事情是否直接推进以下最终目标？

Qwen3.6-35B-A3B-w8a8
+ Ascend 310P3 single-card dual-die
+ TP=2
+ full-attention self_attn.o_proj only
+ unquantized BF16 -> BF16 semantics preserved
+ custom wgm-dev-310p MemFabric
+ o_proj matmul and TP reduction owned by one fused path
+ MM[t+1] overlaps communication/reduce[t]
+ repeated-wave safe
+ eager + ACL Graph correct
+ profiler-proven overlap
+ measurable positive performance result
```

如果答案是否定的，应停止扩散工作，回到上述主目标。

不得因为当前代码更容易实现而擅自：

- 把目标改成普通 HCCL allreduce；
- 放弃 overlap；
- 只做通信 demo 不接模型；
- 只在 eager 模式工作；
- 扩大范围去支持 TP>2/其它模型；
- 把 Phase 1 correctness prototype 当成最终性能实现。

任何需要改变本文硬需求的决定，都应由项目负责人显式确认并同步修改本文，而不是由开发代理自行改变。