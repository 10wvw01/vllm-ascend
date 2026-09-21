# 310P3 Qwen3.6 FP16 o_proj + MemFabric 融合需求基线

> 本文只定义**当前最终需求和验收合同**。历史实现、废弃 API、旧平台现象不属于
> 本文内容。出现冲突时，以目标 310P3 当前实机行为、当前
> `wgm-dev-310p` public API 和本文需求为准。

## 1. 最终目标

在 Ascend 310P3 单物理卡双 die、TP=2 上，为
`Eco-Tech/Qwen3.6-35B-A3B-w8a8` 的 full-attention
`self_attn.o_proj` 提供专用融合路径：

```text
FP16 local o_proj MM
        +
TP=2 peer exchange
        +
local reduction
```

通信必须使用 `wgm-dev-310p` MemFabric public SHM/SDMA API，并实现真实的
chunk 级计算通信 overlap。

## 2. 固定适用范围

第一版生产目标固定为：

- Ascend 310P3 / dav-2002；
- 单物理卡、双 die；
- TP=2；
- model type: `qwen3_5_moe_text`；
- 仅 full-attention `*.self_attn.o_proj`；
- global K=4096；
- TP local K=2048；
- output N=2048；
- 目标 layer 为未量化 FLOAT 条目；
- FP16 input / weight compute path / exchange / output。

不要求支持 TP>2、多物理卡、其它 hidden size、linear-attention/GDN、
MLP/MoE expert linear 或通用 collective。

## 3. 模型路由

只有同时满足以下条件才能启用：

```text
VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
model_type == qwen3_5_moe_text
prefix == *.layers.N.self_attn.o_proj
layer_types[N] == full_attention
TP == 2
global K == 4096
local K == 2048
N == 2048
params_dtype == torch.float16
unquantized linear route
```

命中后必须设置 `layer.reduce_results=False`。融合 op 返回值已经具有 TP=2
allreduce 语义，不能再执行 generic HCCL allreduce。

## 4. MemFabric 边界

MemFabric 是只读外部黑盒。vLLM-Ascend：

- 只消费安装包公开 host/device headers 与 `libmf_smem.so`；
- host 只调用公开 lifecycle/readiness/control API；
- device 数据面只调用 `smem_shm_sdma_signal/wait/quiet`；
- 不修改 `memfabric_hybrid`；
- 不读取或复制 mailbox/ring/head/tail/reserved/orchestrator/SQE 等内部实现；
- 不新增依赖私有布局的“适配 API”。

如果 public API 或目标平台存在限制，必须通过当前实机验证记录为依赖约束，
不能通过侵入 MemFabric 内部规避。

## 5. 计算通信 overlap

同一 chunk：

```text
MM(chunk n)
  -> cache clean
  -> ready[n]
  -> signal(n)
  -> SDMA(n)
```

同时其它 worker 可以继续：

```text
MM(chunk n+1 / n+k)
```

硬约束：

- signal 必须等待本 chunk MM 和 clean；
- 后续 MM 不得等待前一 chunk SDMA 完成；
- 不允许每 tile host synchronize/barrier；
- op 返回前必须完成 outbound、peer arrival 和 local reduction；
- overlap 必须由 CANN profiler timeline 证明。

## 6. TP=2 数学语义

两 rank partial 为 `Y0`、`Y1`，两端最终都必须得到：

```text
Y = Y0 + Y1
```

当前 reference：

```text
local = F.linear(x, local_weight)
reference = FP32 TP sum -> FP16
fused = fused local MM + MemFabric exchange + local FP16 add
```

当前 ABI v6 必须在目标 310P3 上取得独立 correctness 证据。

## 7. 内存与复用安全

必须明确区分：

- send arena：本地 partial，peer DMA 读取期间保持 immutable；
- recv arena：peer partial 落点；
- output：普通 NPU Tensor；
- producer ready/status：vLLM-owned device memory；
- ack slot：symmetric segment 内的应用级 credit payload。

下一 wave 覆写 peer recv arena 前，必须确认 peer 已完成上一 wave 的 local add。
当前协议使用 FIFO fixed credit，不依赖 host wave index。

## 8. repeated-wave 与 ACL Graph

每个 wave：

```text
prepare protocol status
-> gate(fixed credit)
-> MM/coordinator pipeline
-> quiet + strict wait
-> local add
-> ack(fixed credit)
```

初始化只预置一枚 fixed credit。

要求：

- repeated eager >=1000 次无 hang/mismatch/stale；
- Graph replay 不 create pool、malloc scratch、lazy warmup 或执行 host barrier；
- capture 前完成 persistent context/protocol/bucket warmup；
- repeated Graph replay correctness 稳定。

## 9. 错误模型

以下异常不得静默返回正常结果：

- ready timeout；
- public signal failure；
- quiet failure；
- wait 非 OK；
- data mail `dst/len/imm` mismatch；
- credit mail mismatch；
- ack signal failure。

device 必须 fail-stop；失败 context 不得被当作健康状态继续复用。

## 10. Stream 与生命周期

- 普通 eager：一个 process-local context 只允许一个 eager execution stream；
- ACL Graph：允许 framework-managed capture side stream，前提是 capture 前完成初始化/预热；
- eager destroy 前执行 public quiet + stream synchronize；
- 无法证明 quiescent 时不得强行 destroy 可能仍有 in-flight SDMA 的 pool。

## 11. Build / feature gate

feature 默认关闭。

feature-off：

- 不依赖 MemFabric 安装；
- 普通 310P 路径不变。

feature-on：

- 仅支持 `ascend310p*`；
- `MEMFABRIC_HYBRID_HOME_PATH` 必须来自当前 MemFabric 安装环境；
- public host/device headers 和 `libmf_smem.so` 缺失时 CMake fail fast；
- device kernel 从当前 `.asc` 源码编译；
- 仓库不使用预编译 `.o` 作为正式输入。

## 12. 验收矩阵

### Build

- feature-off build；
- feature-on CMake；
- bisheng dav-2002 compile；
- `vllm_ascend_C` load；
- public MemFabric dependency resolution。

### 单层 correctness

至少覆盖：

```text
M = 1, 8, 32, 33, 64, 128, 512, 2048, 4096
```

验证两 rank、tail、64-chunk 边界、multiple waves 和 repeated calls。

### 稳定性

- >=1000 fused calls；
- >60s；
- 10min；
- 无 hang、protocol trap、mail mismatch、stale data。

### ACL Graph

- minimal fused-op capture/replay；
- 多次 replay；
- Qwen ACL Graph；
- capture 内无 lazy init/warmup。

### 性能

同时记录 stock 与 fused：

- local MM；
- stock HCCL allreduce；
- fused total；
- producer/wait/add/gate/ack；
- TTFT、prefill、TPOT/ITL、tokens/s；
- CANN timeline 的 MM/SDMA overlap。

项目完成要求 correctness、Graph、stability 全部通过，并取得可重复、可解释的
正向性能收益。
