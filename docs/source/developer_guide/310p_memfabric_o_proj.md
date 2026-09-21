# 310P3 TP=2 FP16 o_proj + MemFabric AllReduce 融合算子设计文档

> 状态：P5 完成（V5 epoch API 迁移 + direct producer 实机 bit-exact 验收通过，
> 2026-09-21）。P6 性能优化进行中。
>
> 目标分支：`feat/310p-w8a8-o-proj-memfabric-ar`
>
> 关联文档：
>
> - [研发计划与当前状态](310p_memfabric_o_proj_development_plan.md)
> - [编译、部署与模型启动说明](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)
> - [需求目标（验收合同）](310p_memfabric_o_proj_requirements.md)

## 1. 背景与目标

目标是在一张 Ascend 310P3 双 die 卡上，以 TP=2 运行 `Eco-Tech/Qwen3.6-35B-A3B-w8a8` 时，将 full-attention 的 `self_attn.o_proj` 与其紧随其后的 Tensor Parallel AllReduce 做成一条面向 310P 的融合流水。

普通 `RowParallelLinear` 路径为：

```text
FP16 local o_proj matmul
        |
        v
TP all-reduce (HCCL)
```

本项目目标路径（V5 已实现形态）：

```text
fused AscendC producer kernel（单 kernel 循环）:
  FP16 cube matmul chunk[t] -> 直接写 symmetric send arena
      -> 64B 行级 cache clean -> mailbox signal[t]
      -> AICPU epoch kernel 代投 SDMA（与下一 chunk 的 matmul 重叠）
  ...chunk[t+1]...
waiter kernel: quiet（本端信号全落地）-> wait x N（按序收件）
add_out: final = send + recv（stream 序跟随 waiter）
```

最终目标不是简单地用另一个通信 API 替换 HCCL，而是让计算和通信形成生产者/消费者流水，实现"边算边搬"。

## 2. 固定适用范围

当前实现刻意保持窄范围，避免影响其他模型和其他算子。

- 硬件：Ascend 310P3（dav-2002），单物理卡、双 die。
- Tensor Parallel：严格 TP=2。
- 模型：Qwen3.5/Qwen3.6 MoE text trunk；本项目实机目标为 `Eco-Tech/Qwen3.6-35B-A3B-w8a8`。
- 目标层：仅 `full_attention` 的 `self_attn.o_proj`。
- 不命中：`linear_attn` / GDN / MLP / MoE expert linear / 其他 RowParallelLinear。
- 量化：未量化（2026-09-18 负责人裁决 A，实机核查目标 checkpoint 的
  full-attention o_proj 为 FLOAT 条目；原 W8A8 契约废弃，见研发计划 4.11）。
- 输出 dtype：FP16（checkpoint 无 torch_dtype，310P 以 FP16 运行；BF16 NZ
  linear 在本机 CANN 不支持）。
- 通信：定制 `wgm-dev-310p` MemFabric V5（origin/wgm-dev-310p，
  mailbox-ring epoch API）的 AICore/AICPU/SDMA 路径。
- 融合算子内部不使用 HCCL/MC2；HCCL 仅可作为 correctness reference
  （且必须在建池之前使用，见 §12）。

Qwen3.6 目标几何：

```text
global attention o_proj input K = 16 heads * 256 = 4096
TP = 2
local K = 2048
hidden size / N = 2048

per rank:
A[M, 2048] x W[2048, 2048] -> Y_local[M, 2048]   (W 为 FRACTAL_NZ 运行时布局)
```

## 3. 非目标

当前阶段明确不做以下泛化：

- TP>2；
- 多物理卡通信；
- 任意模型/任意 hidden size；
- W8A8/INT8 量化路径回归（除非负责人再次裁决）；
- BF16/INT32 通信路径泛化；
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
4. 310P 未量化路由（`MemFabricOProjLinearMethod310`，modelslim 派发）的
   `apply()` 进入融合路径；
5. 融合路径返回的 Tensor 已经是 TP=2 reduce 后结果。

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
- 310P 未量化路由（`AscendUnquantizedLinearMethod` 或其 MemFabric 派发
  子类，modelslim description 中该层为 FLOAT 条目）；
- `params_dtype == torch.float16`。

Qwen hybrid decoder 的 linear-attention 路径使用 `linear_attn` 命名，因此不会被 `self_attn.o_proj` 条件命中；额外检查 `layer_types[N]` 用于防止未来上游命名变化导致误匹配。

## 5. MemFabric 依赖关系

本项目依赖的是独立编译安装的定制 MemFabric V5（origin/wgm-dev-310p）：

```text
GDD_ESCC/memfabric_hybrid @ origin/wgm-dev-310p
              |
              | build (cmake -DXPU_TYPE=NPU -DBUILD_PYTHON=OFF)
              | install 到 <prefix>（如 /opt/memfabric-wgm-dev-310p-v5）
              v
V5 split 布局:
  <prefix>/smem/include/host/        smem.h, smem_shm.h（host adapter）
  <prefix>/smem/include/device/      smem_shm_aicore_base_sdma.h（.asc）
  <prefix>/smem/lib64/libmf_smem.so
  <prefix>/hybm/lib64/libmf_hybm_core.so
  <prefix>/acc_links/lib64/libacc_tcp_net.so
  <prefix>/hybm/aicpu_kernel/        libmf_sdma_orch_v6.json（epoch kernel）
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

CMake 只在显式给出的 `VLLM_ASCEND_310P_MEMFABRIC_ROOT` 内查找 V5 布局
（host/device 头文件 + epoch launch json），并使用 `NO_DEFAULT_PATH`，防止
误链接系统中的官方/上游 MemFabric。运行期 `MF_SDMA_ORCH_JSON` 必须显式
指向 v6 json（split 布局下自动发现失效，kfc 回退通道在本机不可用）。

## 6. 核心流水设计

### 6.1 单向依赖

正确的依赖关系是：

```text
matmul[t] -> clean[t] -> signal[t] -> (AICPU epoch) SDMA[t] -> arrival mail[t]
      |
      +------------------------------------------------> matmul[t+1]
```

关键约束：

- `matmul[t+1]` 不等待 `SDMA[t]`（signal 只入环即返回，AICPU 异步搬运）；
- 本端 `waiter` 的 `quiet` 等待本端全部信号落地（send arena 可安全 reduce）；
- 本端 `waiter` 的 `wait x N` 按序收齐对端 N 个 chunk（recv arena 可读）；
- `add_out` 在同一 stream 上跟随 waiter，op 返回前全部完成。

### 6.2 TP=2 AllReduce 简化

因为 world size 固定为 2，不需要 ring/tree collective。

每个 rank 都执行：

```text
rank0.send[t] --SDMA--> rank1.recv[t]
rank1.send[t] --SDMA--> rank0.recv[t]

rank0.final[t] = rank0.send[t] + rank0.recv[t]
rank1.final[t] = rank1.send[t] + rank1.recv[t]
```

通信 payload 为 FP16 tile（`at::add_out` 的 FP16 add 语义与 FP32 求和后单次
round 等价，实测与 reference bit-exact）。

## 7. 内存布局与 race 规避

每个 rank 的 symmetric segment（默认 32 MiB 物理贡献）内：

```text
+0        send arena (8 MiB, 64 chunk 槽)
+8 MiB    recv/final arena (8 MiB)
+16 MiB   (保留；P6 计划用作 ack slot)
段尾 48 KiB  V5 SDMA mailbox 保留区（memfabric 托管，host/用户不可直接使用）
```

race 规避：

- send arena：wave 内 immutable；wave 间复用由本端 `quiet` 保护（全部信号
  落地后才允许下一 wave producer 覆写，stream 序保证）；
- recv arena：由 wave 边界 join（见 §12）保护——双 rank 都完成上一 wave 的
  `add_out` 读取后才允许任何一方的新 signal 覆写对端 recv；
- chunk 槽固定 128 KiB（tile_m=32），partial tail 由 host 侧 scratch
  memset+memcpy + slot 尾部 pad memset 保持确定性。

默认参数：

```text
tile_m = 32（须为 [16, 4096] 内 2 的幂，kernel 按 M-bucket 静态特化）
chunk = FP16 [32, 2048]
chunk_bytes = 32 * 2048 * 2 = 128 KiB
max_chunks_per_wave = 64（请求环容量；单 kernel 无 wave 硬上限，背压自旋）
```

## 8. Direct producer（P5 已实现）

单 AscendC kernel（`Mf310pDirectProducerKernel`，M-bucket 模板特化）循环：

```text
SetTensorA/B -> SetOrgShape -> IterateAll（直接写 send arena 槽）
    -> Mf310pCleanRegion（64B 行级 clean，ex08 姿势）
    -> smem_shm_sdma_signal_at(res, slot, peer_recv_slot, chunk_bytes, imm=chunk)
    -> 下一 chunk（复用同一 Matmul/TPipe/UB）
```

dav-2002 位级正确 matmul 配方（对照 `F.linear`，m=1..4096 全 bucket
bit-exact）见研发计划 §9.1。partial tail chunk 用 `bucket_of(rows_last)` 的
第二实例，从 host 准备的 scratch 读取避免 x 越界读。

每个 kernel symbol 首次 launch 是静默哑弹（binary eager-load no-op），必须
以零副作用 shape（chunk_count==0 / chunks==0）warmup 两次。

waiter kernel（`mf310pWaitKernel`）：

```text
smem_shm_sdma_quiet_at(res)           // 本端信号全落地
for c in chunks: wait_at(res)         // 按序收件；status != OK 早退
```

## 9. FP16 数值语义

融合算子必须保持未量化路径语义：

- 输入/权重均为 FP16（NZ 布局权重，`F.linear` 同源），本地 matmul 数值与
  `F.linear` bit-exact（已验收）；
- output dtype 保持 FP16；
- o_proj 无 bias；若未来带 bias，只在 TP rank 0 应用一次（TP sum 后不重复）；
- 本地 reduce 为 `at::add_out(send, recv)` 的 FP16 add（等价 FP32 加 + 单次
  round，实测与 reference bit-exact）。

## 10. Runtime 生命周期

MemFabric 资源必须是 process-persistent：

- `smem_init` / `smem_shm_init`（config store，rank0 起 server）；
- symmetric SHM pool（`SMEMS_DATA_OP_SDMA`）+ stable GVA；
- V5 host 唯一数据面查询：`smem_shm_sdma_get_workspace(shm) != nullptr`
  （epoch kernel 就绪判定）；
- producer scratch（tail 用）+ 每 bucket warmup bitmap + waiter warmup bit。

禁止每个 forward 创建/销毁 pool。这既是性能要求，也是 ACL graph stable-address 要求。

当前 runtime 实现 failure poison：一旦某个 wave 中途失败，不允许静默复用同一
context（mailbox/SQE 状态不可恢复），bring-up 阶段遇到此类错误应重启对应
worker。

LIFO atexit teardown：`memfabric_o_proj_shutdown()` 在 torch_npu 析构前销毁
context（否则静态析构期 SIGSEGV，实机验证）。

## 11. Wave 协议（V5）

每个 wave（op 调用内按 64-chunk 容量分批）在单条 torch stream 上：

```text
join    = aclrtSynchronizeStream(当前流) + smem_shm_control_barrier(pool)
          （跨 wave recv 覆写竞态的唯一保护，见 §12）
warmup  = 懒激活 bucket/waiter symbol（哑弹 x2 + sync）
produce = full producer（tile_m bucket）+ 可选 tail producer（小 bucket, scratch 源）
wait    = waiter kernel（quiet + wait x chunks）
reduce  = at::add_out(out[wave], send[wave], recv[wave])
```

wave 内 chunk 之间没有任何 host 同步（signal 背压 + stream 序自洽）。

## 12. V5 平台约束（本机实测，集成必读）

V5 的常驻 epoch kernel 设计在本机引出三个硬约束（详细证据见研发计划 §9.3）：

1. **device 级同步死锁**：`torch.npu.synchronize()` 及 torch_npu 的 HCCL
   集合通信（barrier/allreduce，内部同样 device-sync）会等待永续 epoch 任务
   → 永久挂死，约 25-28s 后 epoch 被击杀并砖化 AICPU（507901）。池存活
   期间只能用 stream 级同步；
2. **epoch 25s launch-timeout 击杀窗口**：池寿命超过 ~25s 必然触发（本机
   usleep 粒度），单进程内池生命周期必须控制在窗口内；根治需 memfabric
   kernel 侧缩短自限并重部署（厂商跟进）；
3. **teardown HDC 静默断连**：`smem_shm_destroy` 后任何设备操作都可能失败，
   shutdown 之后不应再有设备/集合操作。

诊断工具：`torch.ops._C_ascend.memfabric_o_proj_debug_snapshot()` 同步 D2H
转储双端 mailbox 协议字（reqHead/Tail、quiet/arrival 戳、邮件像、arena 首
字）。

## 13. ACL Graph 考虑

310P model runner 会使用 ACL graph capture/replay，因此生产版本必须满足：

- symmetric memory 地址稳定；
- workspace/mailbox 地址稳定；
- 不在 replay 中重新创建 MemFabric runtime；
- wave 协议语义可重复；
- 不把 host barrier 错放进每个 tile 的关键路径（P6 计划以设备侧
  ack/gate 替代 host join）。

实机 bring-up 顺序建议先 `--enforce-eager` 验证 correctness，再去掉 eager 单独验证 graph capture/replay。

## 14. 代码地图

```text
vllm_ascend/_310p/ops/memfabric_o_proj.py
  model-side eligibility + 融合 op 单入口路由

vllm_ascend/_310p/quantization/modelslim_config.py
  310P 未量化 linear 的 MemFabric 派发

csrc/memfabric_o_proj_binding.cpp
  PyTorch custom op 注册（融合 op / shutdown / debug snapshot）

csrc/memfabric_o_proj_runtime.cpp
  process-persistent runtime state / arena wrapping / wave 编排 / poison

csrc/memfabric_o_proj/external/memfabric310p_adapter_api.h
  repo-owned narrow C ABI (v3)

csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp
  direct calls into installed wgm-dev-310p V5 MemFabric

csrc/memfabric_o_proj/external/memfabric310p_device.asc
  AscendC direct producer + waiter kernels

cmake/memfabric_310p.cmake
  explicit custom MemFabric build/link contract（V5 split 布局）

benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py
  TP=2 单层 bit-exact + latency benchmark

tests/ut/_310p/test_memfabric_o_proj_source.py
  import-free source regression tests
```

## 15. 验收标准

最终完成至少需要满足（当前状态）：

1. TP=2 bilateral MemFabric exchange + local reduce 数值正确（**已达成**，
   9/9 rows bit-exact）；
2. 连续重复 wave 不出现 stale/丢件/reset race（**已达成**，repeat 压测通过）；
3. 融合结果与 `F.linear + TP allreduce`（FP32 math）bit-exact（**已达成**）；
4. profiler 能证明 `matmul[t+1]` 与 `SDMA[t]` 有实际重叠（V4 时代已验证，
   V5 待 P6 重新量化）；
5. Qwen3.6 full-attention `o_proj` 才命中，其他层保持原路径（**已达成**，
   UT + modelslim 路由）；
6. eager 与 ACL graph 均能稳定重复运行（P7）；
7. 端到端模型 generation correctness 通过（待 P6 后执行）；
8. 性能数据至少包含 TTFT/prefill latency、decode latency/TPOT、tokens/s、
   融合 op latency 和 overlap timeline（P6）；
9. feature gate 关闭时普通 vLLM-Ascend 行为不变（**已达成**）。
