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
- source regression 与目录结构门禁；
- peer-space 地址校验：`mf310p_exchange_geometry` 经控制网 allgather 交换
  pool base，gate/waiter 按期望发送方地址校验（替代仅在同基址巧合下成立
  的接收方等值断言）；
- M 阈值路由：小于 `VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M`（默认
  4096）的批次走 stock NZ matmul + HCCL all-reduce，decode 不经过融合
  路径；
- warmup fallback：dummy-run 走 stock，池创建推迟到首个真实请求；
- 池存活期间 `_sync_device` 收敛为流级同步，profile/step 同步不再等
  待 epoch kernel；
- env 门控的设备级慢 op 监测（`[mf310p-slow]`，500ms 阈值）与段内
  worker 心跳；
- layer bench `--diverge-va` 回归守卫（rank1 建池前 4GB 预分配，覆盖
  跨进程 GVA 基址分叉场景）。

### 实机验收进度（2026-09-22，MemFabric v9 = 2026-09-21 19:12 构建）

MemFabric v9 修复了 epoch 28s 自退缺陷（墙钟自限 18s），数据面行为不变。

| Gate | 状态 | 结果 / 剩余 |
|---|---|---|
| 0 环境 | PASS | 版本记录齐备；融合仅支持 `ASCEND_RT_VISIBLE_DEVICES=0,1`（R7） |
| 1 build | PASS | feature on/off 双构建稳定可重复 |
| 2 M=1 bring-up | PASS | bare + diverge-va 全 PASS，bit-exact |
| 3 correctness 矩阵 | PASS | M=1/33/512/2048/4096，含 tail 与多波 |
| 4 repeated/long-run | 部分 | repeat 与 >60s 通过；10min 稳态未跑（R6 + 设备占用） |
| 5 平台共存 | PASS | stream/device 同步、HCCL 共存、destroy/recreate |
| 6 profiler overlap | PASS | M=2048：producer 1967µs，waiter 暴露 161µs，周期 2406µs < 串行下界 2661µs |
| 7 Qwen eager | PASS | TP=2 serve 正确推理；M 路由下 TTFT 与 stock 持平 |
| 8 ACL Graph | 部分 | minimal capture/replay bit-exact；Qwen 级 graph 未测 |
| 9 性能验收 | 部分 | 见下 |

### Gate 9 当前数据

- 单层（tile_m=128）：M=4096 与 stock 持手（3.241 vs 3.294 ms）；M>=6144
  融合约 +15%（6144：4.603 vs 5.318；8192：6.022 vs 6.934）；M<=2048
  stock 占优。
- serve 级（Qwen3.6-35B-A3B-w8a8，32 请求 × 8192-token，并发 4）：无
  D3 冻结时 E2E 与 stock 持平（o_proj 融合收益 ~0.1% E2E，被 10/40 全
  注意力层与 MoE 前向稀释）。
- **未满足完成定义中的「相对 stock 有明确正收益」（E2E 级）**；R6 未
  解除前融合路径不可上生产。

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

### R1：ready flag 跨 AICore cache 可见性（已验证）

diverge-va、多波、serve 1041 邮件压力下未出现 stale visibility；
clean/invalidate 轮询方案维持。

### R2：public signal 单 producer（已验证）

64-chunk、multiple-wave、serve 长负载下零错序。

### R3：Trap 错误传播（部分验证）

实机观察到 Trap 生效（缺陷定位期间复现过全部校验状态码）；错误后
poison + 进程级重启路径已实现，未做专项演练。

### R4：ACL Graph replay（部分验证）

minimal capture/replay bit-exact。graph-used context 退出时保守跳过
destroy，进程退出有 SIGABRT 噪声（依赖 MemFabric 提供「只停线程不
销毁池」的关停 API）。Qwen 级 graph 待测。

### R5：小 M 性能（已裁决）

M 阈值路由（MIN_M=4096）落地：decode/小批走 stock，无需 small-M
专门路径。

### R6：D3——fused serve 随机 ~20s 引擎整体冻结（新，阻塞生产化）

约 0.75 次/5min；同批并发请求 e2e 齐平 +20s，输出仍正确。已证明冻结
不在融合 op 内（设备级事件监测零慢 op、host trace 干净）；嫌疑 epoch
18s 自退后重launch 与 aicpu 调度器/运行时锁的偶发交互（MemFabric
内部）。等上游定位；最小复现（剥离 vLLM 的 pool+epoch+普通负载）待
设备空闲后执行。

### R7：MetaSetDeviceAccess 逻辑 deviceId 授权（新，部署约束）

MemFabric meta 区的设备侧授权使用逻辑 device 号，仅
`ASCEND_RT_VISIBLE_DEVICES=0,1`（逻辑==物理）时成立；2,3/4,5 上必现
507899。等上游修复或确认语义后放开设备对选择。

另：设备侧内核部署有 `test -f` 即不重部署的缓存语义，launch json 的
版本号递增（v6→v9）是唯一击穿手段；升级 MemFabric 内核后必须换用新
版本号 json，且注意 `/usr/local` 安装件可能滞后于 repo 构建。

## 13. 完成定义

必须全部满足：

- 当前 public-API build 可重复：**已满足**；
- correctness matrix 全通过：**已满足**（含 diverge-va）；
- repeated/10min 稳定：**部分**（repeat/60s 已过，10min 待 R6 解除后补）；
- protocol 无异常：**已满足**；
- Qwen eager 正确：**已满足**；
- ACL Graph 正确：**部分**（minimal 已过，Qwen 级待测）；
- profiler 证明 MM/SDMA overlap：**已满足**；
- 相对 stock 有明确正收益：**部分**（op 级 M>=6144 约 +15%；E2E 级持平，
  受模型结构稀释，且 R6 未解除）；
- 文档中的命令与最终实机环境一致：**已满足**（usage 已同步 tile 定容、
  MIN_M/WARMUP_FALLBACK/TRACE）。

剩余关键路径：R6（上游 D3 定位）→ 10min 稳态 + D3 最小复现 → Qwen 级
ACL Graph → （可选）M 自适应 tile 选择与 8-worker 自签名 producer 重设计。
