# 310P3 TP=2 W8A8 o_proj + MemFabric AllReduce 研发计划

> 本文用于项目推进和服务器交接，记录整体研发计划、当前已完成事项、待办事项、实机验证顺序和验收门槛。
>
> 设计文档：[310p_memfabric_o_proj.md](310p_memfabric_o_proj.md)
>
> 使用说明：[310p_memfabric_o_proj_usage.md](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)
>
> OpenCode 提示词：[310p_memfabric_o_proj_opencode_prompt.md](310p_memfabric_o_proj_opencode_prompt.md)

## 1. 最终目标

在 Ascend 310P3 单卡双 die、TP=2 条件下，对 `Eco-Tech/Qwen3.6-35B-A3B-w8a8` 的 full-attention `self_attn.o_proj` 实现：

```text
W8A8 o_proj matmul + TP=2 reduction
```

融合为一条计算/通信 overlap 流水：

```text
MM[t] -> publish[t] -> SDMA[t] -> peer arrival[t] -> local reduce[t]
  |
  +--------------------------------------------------> MM[t+1]
```

目标是让 `MM[t+1]` 不等待 `SDMA[t]`，最终以定制 `wgm-dev-310p` MemFabric 实现双 die 数据搬运。

## 2. 研发阶段总览

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| M0 | 明确模型、硬件、TP、量化和通信契约 | 已完成 |
| M1 | vLLM model-side 精确挂接和 feature gate | 已完成 |
| M2 | 定制 MemFabric build/link + persistent runtime 骨架 | 基本完成，待实机编译 |
| M3 | TP=2 bilateral SDMA + repeated-wave correctness | 代码骨架完成，私有 ABI 待核对 |
| M4 | Phase-1 tiled W8A8 MM/communication overlap | 已实现代码，待 310P 实机验证 |
| M5 | Phase-2 direct AscendC/CATLASS tiled producer | 未开始实机实现 |
| M6 | ACL graph + 端到端 Qwen3.6 correctness/performance | 未验证 |

## 3. 当前已完成事项

### 3.1 模型侧挂接

已完成：

- 仅允许 `qwen3_5_moe_text`；
- 仅允许 `*.layers.N.self_attn.o_proj`；
- 额外要求 `config.layer_types[N] == "full_attention"`；
- TP 必须为 2；
- local K 必须为 2048；
- output N 必须为 2048；
- 310P static W8A8；
- BF16 output；
- 命中后 `RowParallelLinear.reduce_results=False`，避免 generic TP allreduce 再执行一次。

### 3.2 Build/feature gate

已完成：

- feature 默认关闭；
- `wgm-dev-310p` 独立编译安装；
- vLLM-Ascend 直接 include/link 定制 MemFabric；
- 不使用运行时 `dlopen`；
- 不部署第二个 adapter `.so`；
- CMake 只从显式指定的 custom install prefix 找头文件；
- feature 打开但 MemFabric root/libs/device object 缺失时 configure 直接失败；
- `.asc` device object 当前作为外部 build input 传入。

### 3.3 Runtime/内存所有权

已完成代码骨架：

- process-persistent MemFabric context；
- symmetric send arena；
- symmetric recv/final arena；
- arrival flags；
- SDMA workspace；
- dedicated reduce stream；
- send/recv 不 alias；
- failed wave poison runtime，防止残留 polling/SQE/flag 被继续复用。

### 3.4 Phase-1 pipeline

已完成代码骨架：

```text
npu_quant_matmul(tile)
    -> copy to symmetric send tile
    -> AICore cache clean
    -> smem_shm_sdma_notify
    -> AICPU/SDMA
    -> peer flag
    -> local BF16 reduce
```

compute stream 不在 tile 间等待通信，wave 最后统一 join。

### 3.5 测试基础设施

已完成：

- import-free source regression；
- eligibility/fallback regression；
- no-dlopen/direct-link regression；
- non-aliasing arena regression；
- failure poison regression；
- TP=2 MemFabric exchange/reduce correctness/latency benchmark：
  `benchmarks/scripts/bench_310p_memfabric_o_proj.py`。

## 4. 当前最高优先级待办：P0 编译打通

服务器 bring-up 首先只解决“能正确编译和链接”，不要同时优化性能。

### P0.1 确认定制 MemFabric install 产物

在 `wgm-dev-310p` 编译安装后记录：

- install prefix；
- `smem.h` 路径；
- `smem_shm.h` 路径；
- `smem_shm_aicore_sdma.h` 路径；
- 真实生成的 `.so/.a`；
- transitive link dependencies；
- runtime `LD_LIBRARY_PATH` / RPATH 要求。

将实际 library list 写入：

```bash
VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES
```

不要猜官方 MemFabric library 名。

### P0.2 确认 `.asc` 编译规则

找到 `wgm-dev-310p` example 08 实际使用的 device build 规则，用同一套 310P toolchain 编译：

```text
csrc/memfabric_o_proj/external/memfabric310p_device.asc
```

得到可被 `vllm_ascend_C` 链接的 object，并设置：

```bash
VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT=/abs/path/memfabric310p_device.o
```

如果当前 `.asc` 语法、launch stub 或 object 形式不兼容，应优先按 example 08 的实际编译方式修改，不要引入另一套猜测工具链。

### P0.3 修复所有编译错误

优先级：

1. header/API signature；
2. namespace/type；
3. device launch stub；
4. linker symbol；
5. RPATH/runtime loader。

每解决一个错误，记录原始错误信息和修复 commit。

## 5. P1：MemFabric 通信协议正确性

编译打通后，不立刻启动 35B 模型，先运行独立通信 benchmark。

### P1.1 核对 host API

确认真实签名和语义：

- `smem_shm_sdma_submit`
- `smem_shm_sdma_wait`
- `smem_shm_sdma_get_workspace`
- `smem_shm_sdma_get_result`

重点确认 `submit(..., chunks)` 是否：

- 按 mailbox 顺序消费；
- 每个 chunk 自动推进 src offset；
- 自动推进 dst offset；
- 每个 chunk 对应哪个 flag；
- flag 写入发生在 SDMA 数据对 peer 可见之后。

### P1.2 核对 wave barrier/reset

当前最大同步风险是 repeated wave。

需要从定制分支确认：

- 是否存在 MemFabric 自带 barrier/rendezvous；
- 如果没有，应使用 vLLM worker 现有 process-group barrier、store rendezvous 或其他可靠 host 协议；
- 清 flag/mailbox 前必须确保 peer 不再使用上一 wave；
- 新 notify 不得被另一 rank 的 late clear 擦掉。

禁止在没有源码依据时继续假设 `smem_shm_control_barrier()` 一定存在。

### P1.3 独立 benchmark 验证

至少测试 M：

```text
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024+
```

验证：

- bilateral send；
- peer receive；
- local add；
- rank0/rank1 结果一致；
- 与 HCCL reference 对齐；
- 连续 1000+ wave 无 stale flag；
- tail chunk 正确；
- process restart 后可重新初始化。

## 6. P2：Phase-1 W8A8 correctness

通信独立验证通过后，再接入真正 `o_proj`。

### P2.1 单层数值对比

参考路径：

```text
npu_quant_matmul(local)
    -> tensor_model_parallel_all_reduce
```

融合路径：

```text
tiled npu_quant_matmul
    -> MemFabric exchange
    -> local add
```

对比：

- max abs error；
- max relative error；
- BF16 tolerance；
- rank0-only quant_bias；
- tail M；
- 多次重复。

### P2.2 检查所有 fallback

确认以下情况仍走原始路径或明确报错：

- feature gate=0；
- TP!=2；
- 非 full_attention；
- 非目标 shape；
- 非 W8A8；
- 非 BF16；
- 其他模型。

## 7. P3：端到端 Qwen3.6 bring-up

建议先使用：

```text
--tensor-parallel-size 2
--quantization ascend
--enforce-eager
```

第一阶段不启用额外 EP、sequence-parallel MoE 或其他新优化，减少变量。

验证：

- 模型可加载；
- 目标 full-attention layer 被正确标记；
- 非目标 layer 不命中；
- 短 prompt generation 正常；
- 长 prompt/prefill 正常；
- 多轮请求无 hang；
- worker 退出/重启行为正常。

建议至少保留一份 feature gate 关闭时的 baseline 日志用于对比。

## 8. P4：Profiler 验证 overlap

数值稳定后再看性能。

需要在 timeline 中证明：

```text
MM[t+1]
```

与：

```text
SDMA[t] / peer receive / local reduce[t]
```

有实际时间重叠，而不是 host 逻辑上异步但设备实际串行。

关注：

- compute stream；
- publish kernel；
- AICPU orchestration；
- SDMA；
- reduce stream；
- `finish()` 等待区间。

如果 overlap 不明显，优先排查：

1. stream dependency；
2. hidden sync；
3. `npu_quant_matmul` 是否阻塞 host；
4. cache clean/notify 是否造成 device-wide serialization；
5. tile 大小是否不合适。

## 9. P5：Phase-2 direct producer

Phase-1 correctness 和通信协议完全稳定后，才开始 Phase-2。

研发目标：

- 使用 AscendC/CATLASS 或可复用 310P Cube matmul primitive；
- 直接写 symmetric send arena；
- 保持 W8A8 dequant 语义；
- rank0-only quant_bias；
- tile 完成后立即 clean+notify；
- 一个 producer kernel 内继续下一 tile；
- 移除 local copy；
- 减少 per-tile launch。

对于 decode M 很小的场景，评估 N-panel tiling。

## 10. P6：reduce 优化

当前 device reduce 是 correctness baseline，不应在通信协议未稳定前同时重写。

后续优化顺序：

1. BF16 vector/UB add；
2. 多 AICore；
3. double buffer；
4. 与 arrival flag 的分块消费；
5. 评估 reduce 是否成为新的 critical path。

每一步都必须保留 reference correctness test。

## 11. P7：ACL graph

Eager 路径通过后：

1. 去掉 `--enforce-eager`；
2. capture；
3. replay 多轮；
4. 检查 persistent GVA；
5. 检查 wave flag generation/reset；
6. 检查 graph replay 是否重复执行 host-only barrier；
7. 检查不同 M bucket/graph 的资源复用。

如果 graph 失败，不要先修改算法；先最小化复现为一个融合 op graph capture case。

## 12. 性能验收指标

至少保存以下 baseline/fused 数据：

- 单层 `o_proj + allreduce` latency；
- prefill latency；
- TTFT；
- decode TPOT；
- output tokens/s；
- prompt tokens/s；
- peak memory；
- SDMA bandwidth；
- overlap ratio/timeline；
- CPU/AICPU overhead。

测试至少覆盖：

- decode 小 M；
- 中等 batch；
- 大 prefill M；
- 长时间稳定运行。

## 13. 当前风险清单

### R1：私有 ABI 与示例不完全一致

风险最高。必须以 `wgm-dev-310p` 源码和实际编译结果为准。

### R2：wave flag 重用 race

可能导致偶发 hang/错误结果。连续 wave stress test 是强制验收项。

### R3：Phase-1 overlap 可能被底层同步破坏

即使 Python/C++ 没有显式 wait，也必须用 profiler 证明设备侧真实 overlap。

### R4：小 M 没有足够流水深度

decode 可能需要 N-panel，而不是只按 M 分块。

### R5：ACL graph

外部 MemFabric runtime 与 host rendezvous 可能不天然 graph-safe，需要单独设计 capture/replay 边界。

### R6：generic `memfabric_hybrid` Python dependency

仓库 `requirements.txt` 仍存在通用 `memfabric_hybrid`。服务器 bring-up 必须确保使用的是定制 `wgm-dev-310p` 安装，不要让 pip 自动下载一个上游版本后误以为已经满足本项目依赖。

## 14. 每次服务器迭代建议记录

每轮提交至少记录：

```text
Date:
Commit:
CANN:
Driver/Firmware:
SOC_VERSION:
PyTorch:
torch-npu:
vLLM:
vLLM-Ascend branch:
MemFabric commit:
MemFabric install prefix:
MemFabric libraries:
Device .asc compile command:
Feature env:
Test command:
Result:
Profiler artifact:
Known issue:
Next action:
```

建议把关键日志、编译命令和性能结果持续回填到本文或 PR #1，避免后续交接重复踩坑。

## 15. 完成定义

该项目可以从“WIP”转为“完成”时，应至少满足：

- custom MemFabric 从源码可重复编译安装；
- vLLM-Ascend 可一条明确流程编译；
- TP=2 communication stress test 通过；
- Phase-1 单层 correctness 通过；
- 端到端 Qwen3.6 generation 通过；
- ACL graph replay 稳定；
- profiler 证明 overlap；
- Phase-2 direct producer correctness 通过；
- 性能相对 baseline 有明确收益或至少能解释瓶颈；
- 文档中的编译、部署和启动命令经过服务器实测并更新为最终命令。
