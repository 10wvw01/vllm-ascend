# T2：原生 MM + MemFabric AllReduce（不融合）对照实验

> **实施状态（2026-10-18）**：实验已完成，结果与 7 项平台缺陷修复见 `T2_AR_ONLY_REPORT.md`。与本文设计的主要差异：① B 的 staging 由 D2D `aclrtMemcpyAsync` 改为自定义 `V2ArStageKernel`（拷贝引擎读与 AICore 写不透明、MTE1 可读到陈旧值，见报告 BUG-3/6）；② 为规避 MTE3 空闲窗首发射丢弃，每调用 2 次牺牲发射 + 1 次有效发射（BUG-5）；③ 守护 stream-sync 改为 device-sync 的 64 深度地址集合（BUG-1/2）；④ B 在生产者与 staging 之间强制 host `stream.synchronize()` 并计入 B 计时（BUG-6）。本文其余协议与判据不变。
>
> **当前为实验代码，尚未经过 310P3 编译和测试。** 目标不是替代已通过的 T0/T1，而是把 MemFabric 传输价值与算子融合价值拆开。V1 与正式 vLLM 模型路由不修改。

## 统一三组定义

| 方案 | 计算 | 通信及加法 | 当前对比口径 |
|---|---|---|---|
| A / Stock | torch_npu 原生 `F.linear`，FP16 | HCCL FP16 AllReduce | 唯一性能收益基线 |
| B / Non-fused | **同一个** `F.linear`，FP16 | D2D 入对称区 → MemFabric SDMA → FP16 Add | 检查只替换通信是否已超过 Stock |
| C / V2 fused | 自定义 Ascend C MM | 与 MM 位于同一个 Kernel 的 MemFabric SDMA + Add | 判断融合在 B 上还能节省多少 |

**重要：B 没有调用 V1/V2 的自定义 MM，也没有融合 MM + AR。** B 的 D2D 拷贝是必要且必须计时的真实代价，不能在性能报告中把它排除。现有 MemFabric 公开 `signal` 接口可接收指针，但尚未证明普通 torch_npu 分配的本地 Tensor 能直接充当 SDMA 源，因此**当前 B 的发送源只使用池内对称缓冲**。普通 Tensor direct-src 是另一个后续可选探针，不能未经验证直接启用。

### B 的实际数据流

```text
原生 torch_npu F.linear(x, W_NZ) -> partial[M,2048] 普通 NPU Tensor
                                     |
              同一 ACL Stream 的 aclrtMemcpyAsync (D2D)
                                     v
                    symmetric ownC[M,2048] (ND)
                                     |
                     单独 AR-only AICore Kernel
                        core0: signal -> wait
                        1/2/4/8 核: FP16 Add
                        core0: quiet -> peer DONE -> wait DONE
                                     |
                              y[M,2048] (ND)
```

- 同一个 Kernel 内**没有 MM**，只有 MemFabric SDMA + FP16 Add 和同步协议。
- 对称池由现有 `mf310p_v2_trial_create` 建立；Host API 新增 `mf310p_v2_trial_run_ar_only(handle, native_partial, y, m, add_cores, stream)`。
- `m=1..8192` 精确行数；同一个 wave 内最多 8192 行；chunkRows 候选 256/512/1024，每条 SDMA 消息最长 4MiB（沿用 V1 经验证的长度上界）。
- 每 chunk `signal→wait→Add`；一波结束才 `quiet + DONE`。Add 核数 1/2/4/8 独立测，所有同步仍用官方 IBSet/IBWait 和 MemFabric 公共 API。
- 不支持同一 context 的并发调用或多个 stream；跨 Rank 调用必须严格同序同形状；一旦报错后禁止复用池。

## 编译与运行

使用已安装 CANN 9.1、dav-2002 bisheng、MemFabric `wgm-dev-310p` 和同一分支的两个开关：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1
python setup.py build_ext --inplace

ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/bench_ar_only.py \
  --rows 1 16 64 128 255 256 257 512 1024 2048 4096 8192 \
  --chunk-rows 1024 --add-cores 1 2 4 8 --repeat 30 --fused
```

- 若仍使用 T0/T1 报告里 KFC 控制面损坏的 dev0/dev1，可**仅在诊断环境显式设置** `MF_KFC_BYPASS=1`；此开关沿用 T1 工具的临时 KFC 文件绕过，不能视为健康部署，也不得在生产启用。应尽量在健康设备完成最终验收。
- 另外记录原生 MM-only 延迟，帮助判断 B 的额外 D2D + MemFabric 成本；不能将这种差值误称为 MemFabric 纯链路延迟。
- `--fused` 加测当前上下文下可运行的 V2 T1（M<=256）和 T0（M 为 chunkRows 倍数）。为确保比较到最优 T0，建议另以 `--chunk-rows 256/512` 测 M=512/1024；T0 的 c1024 overlap 已验证大 M 最优。
- **A 与 FP32 正确性参考必须在 MemFabric 建池之前运行**；这不等价于 HCCL/MemFabric 同一推理进程的交替运行已安全，混合路由须另设 G3 共存验收。
- 样本计时边界 = 框架 Native MM 调用入队 → HCCL 或 B 的 D2D+SDMA+Add 入队 → stream sync；两个 Rank 以较慢 Rank 的耗时作为 TP 完成时延。报告 median、P95、首次调用（单独记录，不能混入稳态）、FP16 正确性。
- `--fused` 的 T0 首次 Kernel 可能触发既有 `aclrtSynchronizeDevice` 0–15s workaround，因此 T0 首次值只能单列，不许把预热开销隐藏为「已经修复」；当前没有 Graph 验收。
- 禁止拿 `F.linear + FP32 HCCL all_reduce + FP16 cast` 的延迟冒充 Stock：那只可用于严格 FP16 融合正确性参考。真实性能基线是原生 FP16 MM + FP16 HCCL AR。

## 验收条件

1. **构建**：AR-only Kernel + C ABI 导出在 dav-2002 成功编译链接；无任何原生 MM 被编进新 Kernel。
2. **正确性**：M={1,2,4,8,16,32,64,128,255,256,257,512,1024,2048,4096,8192}；chunk=256/512/1024；Add 核数 1/2/4/8；两卡输出与 `F.linear(...).float() -> HCCL sum -> FP16` 位精确，协议 status=0；边界 M=1、257、8192 和多 chunk 切分必须覆盖。
3. **压力**：重复 1000 次、不同 M、两 Rank 偏斜、动态 y 指针与复用；严禁在存在首写丢弃/对端超时的情况下标记 PASS。
4. **性能**：至少三次独立测试会话，每档报告 max-rank median、max-rank P95、A/B/C 的真实时延及 `(A-B)/A`、`(A-C)/A`；大于测试波动才能称收益。
5. **分解**：记录 native MM、D2D stage、MemFabric signal/wait、Add、quiet/DONE，分析 B 的主要固定开销；如果 B 慢于 A，**不能断言 MemFabric 传输本身一定比 HCCL 慢**，因为 B 包含额外 D2D/Kernel/协议成本。
6. **回归**：G0/G1/G2 与 T1 保持通过，不修改 V1、现有默认行为以及 V2 小/大 M 正确性。

**首写保护说明：** AR-only 路径目前仅以一次新输出地址的 `aclrtSynchronizeStream` 作实验性保护。对新 AR-only Kernel 是否真正有效，必须实机确认；该方式并未解决 T0 已知的 MTE3 丢写根因，不能宣称 Graph/生产可靠。

报告名称：`T2_AR_ONLY_REPORT.md`，包含原始完整性能矩阵和所有 BUG 修复/回归证据。
