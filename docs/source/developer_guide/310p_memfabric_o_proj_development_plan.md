# 310P3 TP=2 FP16 o_proj + MemFabric 当前实机验证计划

> 本文只记录**当前代码下一步需要完成的 Gate、风险和完成定义**，不记录已经
> 废弃的历史实现阶段。
>
> 设计：[310p_memfabric_o_proj.md](310p_memfabric_o_proj.md)
>
> 需求：[310p_memfabric_o_proj_requirements.md](310p_memfabric_o_proj_requirements.md)
>
> 使用：[310p_memfabric_o_proj_usage.md](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)

## 1. 当前代码状态

当前源码已完成：

- MemFabric 只读黑盒化，只使用 public API；
- 定制 C++/AscendC 代码隔离到
  `csrc/_310P/custom_memfabric_o_proj/`；
- adapter ABI v6；
- 7 MM workers + 1 communication coordinator；
- `MM -> clean -> ready -> signal` chunk pipeline；
- strict data/credit mail validation；
- fixed credit wave protocol；
- protocol status + `AscendC::Trap()` fail-stop；
- real `aclrtGetDevice()` device id；
- eager single-stream 与 ACL Graph capture-side-stream 合同；
- eager destroy 前 final public quiet；
- feature build 只消费当前 MemFabric public package；
- 无仓库内预编译 device object；
- source regression 与目录结构门禁。

当前版本**尚未完成目标 310P3 的当前 commit 实机验收**。

## 2. 实机 Gate 0：固定环境

每轮测试记录：

```text
Date
vLLM-Ascend SHA
MemFabric SHA
CANN
Driver/Firmware
PyTorch
torch-npu
vLLM
SOC_VERSION
MEMFABRIC_HYBRID_HOME_PATH
model path
feature env
test command
result
profiler artifact
```

所有 BUG、性能结论和回归比较都必须绑定这些版本信息。

## 3. Gate 1：Build

依次确认：

- feature-off build 成功；
- feature-on CMake 成功；
- bisheng dav-2002 编译 `.asc` 成功；
- `vllm_ascend_C` link/load 成功；
- custom ops 注册成功；
- public `libmf_smem.so` 正确解析；
- build 不引用旧目录或预编译 object。

Build 未通过时不进入 correctness。

## 4. Gate 2：最小双 rank bring-up

配置 TP=2，先跑：

```text
M=1
single fused call
```

确认：

- pool create；
- SDMA readiness；
- initial fixed credit；
- producer / waiter / add / ack 完成；
- 两 rank 无 trap；
- protocol status = OK；
- output 与 reference 一致。

## 5. Gate 3：单层 correctness

覆盖：

```text
M = 1, 8, 32, 33, 64, 128, 512, 2048, 4096
```

重点：

- M=33 tail；
- 64-chunk 边界；
- >64 chunks multiple waves；
- 两 rank output；
- repeated calls；
- feature-on/off。

出现数值差异时先定位 MM、exchange、wait、add 或 tail，不通过放宽 tolerance
掩盖问题。

## 6. Gate 4：Repeated / long-run

至少：

- representative M repeat >=1000；
- mixed-M loop；
- >60s；
- 10min。

监控：

- READY_TIMEOUT；
- SIGNAL_FAILED；
- QUIET_FAILED；
- WAIT_FAILED；
- MAIL_MISMATCH；
- CREDIT_MISMATCH；
- ACK_FAILED；
- stale arena；
- deadlock；
- shutdown。

## 7. Gate 5：平台共存行为

在 MemFabric context 存活时单独验证当前 CANN/当前 MemFabric：

- `aclrtSynchronizeStream`；
- `torch.npu.synchronize()`；
- `aclrtSynchronizeDevice()`；
- HCCL barrier；
- HCCL all_reduce；
- destroy/recreate。

这里只记录**当前实机结果**。不得把旧 bring-up 的 device-sync、生命周期或
orchestrator 现象直接当作当前依赖事实。

## 8. Gate 6：Profiler overlap

对 512 / 2048 / 4096 等代表性 M 采 CANN timeline。

必须看到或通过时间下界证明：

```text
MM worker execution
       overlaps
SDMA of earlier chunks
```

同时记录：

- gate；
- producer；
- waiter；
- add；
- ack；
- fused cycle；
- p50 / p95 / p99。

如果 overlap 或总时延不理想，优先调：

1. `tile_m`；
2. worker 数；
3. wave/chunk granularity；
4. MM tiling；
5. cache clean 成本。

不得为了性能重新耦合 MemFabric private ring。

## 9. Gate 7：Qwen eager

模型：

`Eco-Tech/Qwen3.6-35B-A3B-w8a8`

确认：

- 只有 full-attention o_proj 命中；
- linear-attention/GDN 不命中；
- 无 duplicate generic allreduce；
- prefill/decode correctness；
- repeated requests；
- feature-off baseline 正常。

## 10. Gate 8：ACL Graph

顺序：

1. eager/profile 预先完成 context/protocol/bucket warmup；
2. minimal fused-op capture/replay；
3. 多次 replay；
4. 不同 graph bucket；
5. Qwen ACL Graph；
6. repeated requests。

检查 capture 内没有 create/malloc/lazy warmup/sync/barrier。

## 11. Gate 9：性能验收

A/B：

```text
A: stock o_proj + HCCL
B: fused ABI v6 coordinator + MemFabric
```

记录：

- 单层 latency；
- TTFT；
- prefill latency / tokens/s；
- decode TPOT / ITL；
- output tokens/s；
- peak memory；
- CANN overlap timeline。

只有 correctness、Graph、long-run 全通过且存在可重复、可解释的正向收益，
项目才进入完成状态。

## 12. 当前风险

### R1：ready flag 跨 AICore cache 可见性

当前每 chunk 独立 64B line，并使用 clean/invalidate polling。必须在 dav-2002
实机确认无 stale visibility。

### R2：public signal 单 producer

当前仅 block0 调 signal，规避 multi-producer 不确定性。需要 64-chunk、
multiple-wave 和长稳压力验证。

### R3：Trap 错误传播

需要确认目标 CANN 上 `AscendC::Trap()` 能可靠使当前 kernel/stream 失败，
并验证错误后的 worker 重启/teardown 行为。

### R4：ACL Graph replay

capture side stream 已被显式识别；Graph-used context 当前 process-lifetime。
需要实机验证 replay、退出和重复请求。

### R5：小 M 性能

M=1/8/32 的固定调度/通信开销比例高。若 correctness 稳定但 decode 无收益，
只根据 profiler 决定是否增加 small-M 专门路径。

## 13. 完成定义

必须全部满足：

- 当前 public-API build 可重复；
- correctness matrix 全通过；
- repeated/10min 稳定；
- protocol 无异常；
- Qwen eager 正确；
- ACL Graph 正确；
- profiler 证明 MM/SDMA overlap；
- 相对 stock 有明确正收益；
- 文档中的命令与最终实机环境一致。
