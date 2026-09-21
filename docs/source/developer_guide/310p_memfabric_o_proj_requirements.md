# 310P3 TP=2 FP16 o_proj + MemFabric 融合需求与验收合同

> 本文只定义**本次融合需求的固定边界、硬约束和完成标准**。  
> 当前实现状态、下一步任务和实机证据统一维护在
> [AI-native 开发状态与执行计划](310p_memfabric_o_proj_development_plan.md)。  
> 设计实现见 [当前架构](310p_memfabric_o_proj.md)，构建与运行见
> [使用手册](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)。
>
> 分支名中的 `w8a8` 是历史遗留。当前真实实现合同是：
> **Qwen3.6 full-attention 未量化 o_proj，310P 运行时 FP16 matmul/交换/reduce。**

## 1. 唯一目标

在一张 Ascend 310P3 单物理卡的双 die 上，以 TP=2 运行
`Eco-Tech/Qwen3.6-35B-A3B-w8a8` 时，只对 full-attention
`self_attn.o_proj` 提供一个专用融合路径：

```text
local FP16 o_proj matmul
  -> tile/chunk 直接写 symmetric send arena
  -> custom MemFabric V5 SDMA 双向交换
  -> 本地 FP16 reduce
  -> final o_proj output
```

目标不是“把 HCCL 换成另一个 allreduce API”，而是让
`MM[t+1]` 与 `SDMA[t]`/对端接收发生真实设备侧重叠，并减少独立 collective
的调度、同步和格式转换开销。

## 2. 固定适用范围

第一版只支持以下组合：

- Hardware: Ascend 310P3 / dav-2002；
- Topology: 单物理卡、双 die；
- Tensor Parallel: 固定 TP=2；
- Model: `Eco-Tech/Qwen3.6-35B-A3B-w8a8`；
- vLLM model type: `qwen3_5_moe_text`；
- Layer: 仅 `*.layers.N.self_attn.o_proj` 且
  `layer_types[N] == "full_attention"`；
- Quantization: 目标 checkpoint 中该层为 FLOAT/未量化；
- Runtime dtype: FP16 activation / FP16 weight / FP16 output；
- Geometry: global K=4096，TP local K=2048，N=2048；
- Bias: 当前目标层无 bias；
- Communication: 仅定制 `wgm-dev-310p` MemFabric V5 mailbox-ring epoch API。

任何不满足上述条件的 layer 必须走 stock 路径。

## 3. 功能合同

### R1. 融合路径必须拥有 matmul + TP reduction

上层只调用一个 opaque fused op。命中目标层后必须关闭 generic
`RowParallelLinear.reduce_results`，避免融合结果之后再次执行 HCCL allreduce。

### R2. TP=2 结果语义必须正确

每个 rank 计算本地 partial，并双向交换：

```text
rank0.final = rank0.local + rank1.local
rank1.final = rank1.local + rank0.local
```

结果 dtype 保持 FP16。单层 reference 使用本地 `F.linear` +
TP=2 FP32-math sum 后单次 round 到 FP16；不得通过放宽 tolerance 掩盖错误。

### R3. 生产通信必须走定制 MemFabric

融合路径内部不得静默回退到 HCCL/MC2。HCCL 只允许用于建池前的 correctness
reference/baseline。

### R4. 必须存在真实计算通信重叠

至少一个代表性 prefill workload 必须有设备侧证据证明：

```text
MM[t+1] overlaps SDMA[t] / peer arrival[t]
```

仅凭“代码没有 host wait”不能算完成。证据可以是 profiler timeline，或在 V5
平台限制下可复核的 task-time + 带宽/串行下界论证。

### R5. repeated-wave 必须安全

send/recv arena 复用不得出现 stale、覆盖、丢件或跨 wave race。首 wave 可进行
一次 pool-creation rendezvous；后续 wave 应使用设备侧协议，不允许恢复成
per-wave host barrier。

### R6. 失败后不得危险复用

任一 half-wave/协议错误发生后，本进程 runtime 必须进入 poison 状态并要求重启
对应 TP workers；不得尝试在未知 mailbox/SQE 状态上继续 forward。

### R7. 最终必须支持 ACL Graph

`--enforce-eager` 只允许用于 bring-up。最终必须验证 capture/replay、多轮请求、
persistent GVA、不同 M bucket 及 repeated-wave 语义。

### R8. Feature gate 必须隔离影响

默认关闭。关闭时：

- 普通 vLLM-Ascend 构建不依赖定制 MemFabric；
- 非目标模型/层行为不变；
- generic TP allreduce 行为不变。

打开但依赖缺失时必须 fail fast。

## 4. 当前算子输入输出合同

```text
x      : [M, 2048], FP16, contiguous
weight : logical [2048, 2048], FP16, 310P runtime NZ layout
output : [M, 2048], FP16

tile_m : power-of-two in [16, 4096], default 32
chunk  : [tile_m, 2048] FP16
max chunks per wave : 64
```

M 可以小于 `tile_m`；尾块必须通过 scratch/padding 保证不越界读且结果确定。

## 5. 非目标

本次需求不包含：

- TP>2；
- 多物理卡/多机 collective；
- 其它模型或任意 hidden size 泛化；
- W8A8/INT8 o_proj 融合；
- BF16 runtime 数据通路；
- linear-attention/GDN 融合；
- MLP/MoE expert linear 融合；
- 通用 MemFabric collective 库；
- 大范围重构 vLLM distributed subsystem；
- 为了抽象统一而修改无关 310P 代码。

只有当实机数据证明当前窄目标已经稳定完成后，才讨论泛化。

## 6. 最终验收

### A. Build

- 定制 MemFabric V5 可从明确 commit 重复 build/install；
- `.asc` 设备 kernel 可由仓库 CMake 自动编译；
- feature-on extension 可链接并加载；
- feature-off 不依赖定制 MemFabric。

### B. 单层 correctness

至少覆盖：

```text
M = 1, 8, 32, 33, 64, 128, 512, 2048, 4096
```

要求目标层 fused 与 reference bit-exact 或满足已记录且可解释的严格数值合同；
必须包含 tail、单 wave、multi-wave、repeated calls。

### C. 路由正确性

- full-attention o_proj 命中；
- linear-attention/GDN 不命中；
- 非目标 Linear 不命中；
- feature-off 回到 stock 路径。

### D. Eager E2E

当前 HEAD 的 Qwen3.6 TP=2：

- 可加载；
- 目标 10 个 full-attention o_proj 走融合；
- 短 prompt/pre-fill/decode 正常；
- 多轮请求稳定；
- feature-off baseline 正常。

历史旧实现跑通过不替代当前 HEAD 的验收。

### E. ACL Graph

移除 `--enforce-eager` 后 capture/replay 正常，重复请求无 stale/hang，
不同 M bucket 的资源复用正确。

### F. Overlap

保存可复核的设备证据和分析命令，证明代表性 workload 的真实 overlap。

### G. 性能

同时保存 baseline/fused：

- 单层 o_proj + reduction latency；
- TTFT / prefill latency；
- decode TPOT；
- prompt tokens/s；
- output tokens/s；
- 至少一组完整 Qwen3.6 E2E；
- 必要时补充 SDMA 带宽、CPU/AICPU overhead。

要求收益可重复、可解释。correctness 通过本身不能宣告性能项目完成。

### H. 长稳

MemFabric runtime 生命周期必须跨过当前平台已知的 epoch timeout 风险，不依赖
“在约 25 秒内主动结束进程”的 benchmark workaround。必须完成真实服务生命周期
下的稳定性验证。

## 7. Definition of Done

只有 A-H 全部完成，项目才能从 WIP 转为完成。

禁止以下“伪完成”：

- 只完成通信 demo；
- 只在 eager 工作；
- 只证明单层正确；
- 只证明 overlap 但没有 E2E；
- 通过 HCCL fallback 规避 MemFabric 问题；
- 用旧 commit 的 E2E 结果替代当前 HEAD；
- 通过缩短测试时间绕过 epoch 生命周期问题。

## 8. 需求变更规则

本文件不记录开发历史。任何改变固定范围、dtype、TP、通信后端或验收标准的决定，
必须由项目负责人显式确认，并在同一变更中同步：

1. 本需求合同；
2. 当前架构；
3. AI-native 状态/任务；
4. 对应 source regression / hardware acceptance。

没有上述同步，不得由 AI 或开发者自行降低目标。
