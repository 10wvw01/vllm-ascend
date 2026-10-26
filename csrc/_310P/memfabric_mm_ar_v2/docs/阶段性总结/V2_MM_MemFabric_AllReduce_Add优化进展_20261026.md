# V2 MM + MemFabric AllReduce 融合算子 Add 优化进展

> **归档日期：**2026-10-26  
> **优化周期：**2026-10-22～2026-10-23（D2 设计、D3 编码、D4 实机验收）  
> **运行平台：**Ascend 310P3 单卡双 Die，TP=2，CANN 9.1，BiSheng dav-2002  
> **范围：**V2 正式算子内部 Vector Add 的单变量优化；不是重建 Runner、DFlash、ModelState 或通信协议  
> **结论：**V2 Kernel TP2 配对 Event p50 耗时减少 **6.26%（170.6 μs）**；Qwen3.6 Eager 小样本对照中输出吞吐 **+1.86%**。**已有 MemFabric 稳定性 P0 未关闭；不能据此宣称整体超过 Stock 或生产可用。**

## 一、阶段性结论（适合直接用于项目汇报）

本轮在 V2 MM + MemFabric AllReduce 融合 Kernel 中，将 Vector Add 单次处理粒度由 **1024 个 FP16 元素扩大至 2048 个 FP16 元素**，使规约循环次数减半。在 MM 计算布局、8 个 AI Core 参与 MM、4 Chunk、Overlap=1、跨 Die SDMA/IB/DONE 协议和 Host/Serving ABI 均保持不变的条件下，获得以下实机结果：

- **算子内部 A/B**：配对 TP2 Event p50 从 **2726.9 μs** 降至 **2556.3 μs**，节省 **170.6 μs**，执行时延降低 **6.26%**；4 组 ABBA 同期对照均达到 **5.50%～7.16%** 的时延改善。
- **功能正确性**：8 组 Trial 形状×双 Rank 逐样本位精确；60 次 Serving 混合路由×双 Rank 均位精确、协议状态为 0。
- **Qwen3.6 小规模 Eager**：TPOT p50 从 **0.241 s** 到 **0.236 s**（-1.98%）；输出吞吐从 **37.96** 到 **38.67 tok/s**（+1.86%）；TTFT p50 从 **12.110 s** 到 **12.084 s**（-0.21%，噪声级）。
- **工程裁决**：Add Tile2048 **单算子有条件 GO**，E2E **性能方向性正向**；但 G3 阶段已经存在的低概率双 Die 挂死 P0 在本轮仍有复发，V2 整体**不具备生产放行条件**。

> **数字口径提示：**这里的 **6.26%** 是“V2 Tile1024 → V2 Tile2048”的内部时延降幅，**不是**“融合 V2 比 Stock MM+HCCL 快 6.26%”。模型 +1.86% 也是同一 V2 体系内两版 Add Tile 的小样本对照，**不是**相对于 Stock 的增益。

## 二、为什么优化 Add

V2 的核心数据流是本端 MM → MemFabric 传送另一 Die 的局部结果 → 本端 FP16 Add 规约。此前 Add 采用 1024-FP16 元素 Tile，循环中反复搬入两端数据、执行向量加法、搬出结果。

**设计目标**不是改变跨 Die 通信算法，而是用更大的向量操作批次减少同样输出数据所需的循环调度及重复指令开销；这样可单独衡量收益、易于回滚，并避免改变复杂的缓存可见性和通信正确性约束。

### 具体实现

仅对正式 Kernel `src/op_kernel/mf310p_v2_kernel.asc` 做生产专用调整：

- 定义 `kProductionAddTile = 2048`；`kProductionAddBytes = 4096`。
- 三个 Vector 队列 `VECIN/VECIN/VECOUT` 的单队列 Buffer 各从 **2048B → 4096B**，总计从 **6144B → 12288B**；队列深度仍为 1。
- 同时修改 Chunk 中间规约以及最后一个 Chunk 收尾规约的 `DataCopy / Add / DataCopy` 处理长度及循环步长。
- Vector UB 占用多 **6144B**，因此从 MM 临时 UB 预算中对应扣减；保留独立 IB 的 **32B** 预算。
- 诊断 Kernel 与共享 `mf310p_v2_kernel_common.h` 仍用旧 `kAddTile=1024`，保证诊断对照不被生产优化污染。

| 项目 | 优化前 Tile1024 | 优化后 Tile2048 | 变化 |
|---|---:|---:|---|
| 每次 Add 元素 | 1024 FP16 | 2048 FP16 | ×2 |
| 每个 Vector 队列 Buffer | 2 KB | 4 KB | ×2 |
| 三队列占用 UB | 6144 B | 12288 B | +6144 B |
| 每 Core / CHUNK=1024 的 Add 循环 | 256 次 | 128 次 | **-50%** |
| 每 Core / M=4096 的 Add 循环 | 1024 次 | 512 次 | **-50%** |
| MM 临时 UB | 255968 B | 249824 B | -6144 B |
| MM Core / CHUNK / overlap | 8 / 1024 / 1 | 8 / 1024 / 1 | 不变 |
| MM、SDMA、IB、DCCI、Signal、DONE | 原流程 | 原流程 | **未改动** |

**UB 资源验证**：310P3 256KB UB（262144B）下，Tile2048 的 MM 可用临时空间为 **249824B**。BiSheng/CANN 9.1 对相关 Matmul tiling 的最坏需求为 **131072B**，剩余裕量 **118752B**，通过设备编译期证明。初版 D3 曾将 `TOTAL_UB_SIZE` 的 `static_assert` 放在不支持该常量的命名空间作用域，D4 将断言移入设备 Kernel 函数体，保持断言语义不变，修复了编译问题。

## 三、优化前后实测效果

### 3.1 V2 融合算子内部性能（D4-3，主验收）

**对照对象**：V2 Add Tile1024（基线 `edebac40d`）与 V2 Add Tile2048（候选 `5adbf1810`），单变量仅为 Add Tile/UB 预算及等价编译断言修正。  
**条件**：M=4096、CHUNK=1024、Overlap=1、TP=2；无 Profiler 的设备 Event 计时；每个双 Rank 样本取 `max(rank0, rank1)`；8 会话 ABBA（每会话 3 次预热 + 30 次有效样本），切换设备 SO 并验证 SHA。

| 指标 | Tile1024 | Tile2048 | 差值/变化 |
|---|---:|---:|---:|
| **配对 Event p50** | **2726.9 μs** | **2556.3 μs** | **-170.6 μs / -6.26%** |
| Wall p50 | 2816 μs | 2647 μs | -169 μs |
| 四组同期 A/B 的时延改善 | — | 5.50%～7.16% | 全部达到 ≥5% 门槛 |
| 跨会话 p50 离散 | 约 1.1% | 约 1.2% | 低于 D1-A 3.3% 噪声包络 |
| ≥6ms 停顿 / 数值错误 / 协议异常 | 0 / 0 / 0 | 0 / 0 / 0 | 无新增异常 |

**解释**：通过把规约循环次数从 1024 次减少到 512 次，整体 Kernel 关键路径节约约 171μs；循环次数减半**不意味着**整个 Kernel 耗时减半，因为 MM、通信等待、同步和其他执行阶段均未改变。

### 3.2 Qwen3.6 模型级 A/B（D4-4，方向性证据）

**条件**：`Eco-Tech/Qwen3.6-35B-A3B-w8a8`、TP=2、Eager、输入4096 / 输出512、20 prompts、并发10；相同 V2 Serving 栈仅替换设备 SO。由于原 die0/1 被外部租户占用，本次模型回归改到 dev6/7；采用完成且无挂死的干净会话，基线 **n=2**，候选 **n=3**。

| 指标 | Tile1024 | Tile2048 | 观察到的变化 |
|---|---:|---:|---:|
| TTFT p50 | 12.110 s | 12.084 s | -0.21%，噪声级 |
| TTFT p95 | 27.839 s | 27.866 s | +0.10%，噪声级 |
| TPOT p50 | 0.241 s | 0.236 s | **-1.98%** |
| 输出吞吐 | 37.96 tok/s | 38.67 tok/s | **+1.86%** |

**归因边界**：这组数据说明**整体性能方向性偏正**，但样本较少，而且 V2 的目标层只覆盖 M∈[2048,4096] 的 `o_proj/out_proj`；常见 Decode 小 M 由 Stock 回退执行。因此不能直接将 TPOT 的全部改善归因于 Add，更不能把这组输出512的结果当作要求中的 **输出2048、50请求**完整配置的证明。

### 3.3 与 Stock MM+HCCL 的关系：不能混淆两个基线

| 比较对象 | 已有证据 | 能够得出的结论 |
|---|---|---|
| **V2 Tile1024 vs V2 Tile2048** | D4 同环境 ABBA | 优化后的 V2 Kernel 时延下降 **6.26%** |
| **V2 Tile1024 vs 真实 Serving Stock MM+HCCL** | 早期 Prefill 归因：Stock MM约795μs + HCCL AR约1149μs；当时 V2 Kernel约2573μs，额外 contiguous 拷贝约66μs/站点 | 旧 V2 在该 Serving 工况**尚慢于 Stock** |
| **V2 Tile2048 vs 真实 Serving Stock MM+HCCL** | 本轮 D4 没有进行严格同环境、同工作负载的全路径 Stock 三方性能验收 | **不能仅由 D4 的 6.26% 推断新 V2 已超过 Stock** |

后续必须在已正常运行的 Stock+DFlash 框架上，保持其它条件不变，分别统计真实被替换的 MM+AR、融合算子命中次数、双 Rank 耗时和模型端到端指标。

## 四、正确性与工程验证

| 验收项目 | 实机结果 |
|---|---|
| BiSheng diag ON/OFF 编译 | 基线与候选全部 PASS |
| Matmul Workspace / UB 资源检查 | PASS（最坏需求131072B，可用249824B） |
| Trial 直调 8 形状 × 双 Rank | 全部位精确、status=0 |
| Serving 混合路由 | 60 次/Rank 全部 PASS（36 次 V2、24 次 Stock 回退） |
| 非整 Chunk M=2049/3073/4095 | PASS（Serving stage 补齐后输出真实行） |
| 输出缓冲复用、新地址首写 | PASS |
| 单算子 ABBA 性能 | 四组均 ≥5% 改善，无 ≥6ms 停顿 |
| 真实模型 Eager 性能 | 小样本方向性正向 |
| V2 整体长期稳定性 | **未通过生产放行，既有 P0 未关闭** |

**稳定性风险不可省略**：D4-4 默认 epoch 预算下，共有 7 个会话发生 **2 次双 Die AICore 超时挂死**，分别在 Tile1024 基线和 Tile2048 候选各出现一次。相同问题在 Add 优化前的 G3 阶段已存在；现有取证认为是旧的跨 Die 同步/负载相位竞态问题，而非 Add Tile 改动引入。但**根因尚未彻底关闭**，不能用单次正确性或吞吐改善替代长期稳定性验收。

## 五、阶段价值及下一步

**本轮实际完成的工作**：

1. 在不改通信协议和 MM 架构的条件下，通过 Add Tile 翻倍将规约循环次数减半。
2. 通过设备编译、UB 资源证明、双 Rank 位精确验证和 ABBA 交叉测试，证实 V2 Kernel 局部时延改善 **6.26%**。
3. 在 Qwen3.6 的小样本 Eager A/B 中观察到 **+1.86% 输出吞吐**；同时明确性能传导幅度受目标算子覆盖率和非目标计算耗时限制。
4. 清楚识别出后续必须补齐的两项验收：**相对真实 Stock MM+HCCL 的完整收益**、**跨 Die 稳定性 P0**。

**接下来仅沿融合算子主线推进**：大 M 继续优化 MM 与 SDMA/规约的重叠、IB等待和协议控制成本；小 M 研究同 Kernel 合并 MM 与 MemFabric 所带来的固定开销消减；通过现有 Stock+DFlash 推理路径做薄接入对照。**不重建 V2 Runner、DFlash 或 Hybrid ModelState。**

完整目标负载继续固定为：**TP=2、输入4096、输出2048、并发10、50请求、DFlash K7**。在获得同版本/同环境的 Stock、V2 Tile1024、V2 Tile2048 三方实测前，本轮仅报告已完成的 D4 A/B 结果，不延伸新的性能数值。

## 六、证据与版本追溯

- [D2/D3 设计、源码变更与两轮静态审查](../V2_D2_D3_ROUND1_ADD_TILE2048_20261023.md)
- [D4 Add Tile1024→2048 实机验证原始报告](../V2_D4_ROUND1_ADD_TILE2048_ONDEVICE_REPORT_20261023.md)
- [D1-A 设备计时噪声基线](../V2_D1A_PROFILER_PERTURBASELINE_REPORT_20261022.md)
- [真实 Prefill Stock vs V2 归因报告](../V2_PREFILL_ATTRIBUTION_REPORT_20261021.md)
- [正式 V2 Kernel 实现](../../src/op_kernel/mf310p_v2_kernel.asc)
- [项目 ADR 和当前实施边界](../DECISIONS.md)
- 基线提交：`edebac40d226a8aeb08460b5a33a3283a842bba5`；优化候选提交：`5adbf1810`。
- D4 入库脱敏证据目录：`csrc/_310P/memfabric_mm_ar_v2/evidence/d4_20261023/`（ABBA 样本、正确性、E2E 结果、监控和故障摘要）。

---

**一句话汇报**：V2 MM+MemFabric AllReduce 在不调整计算通信协议的条件下，将 Vector Add 粒度由1024提升至2048，使规约循环减少50%，实测融合 Kernel 时延下降 **6.26%**；在 Qwen3.6 小样本实际推理中观察到输出吞吐约 **+1.86%** 的方向性改善。**相对 Stock 的最终收益和既有稳定性 P0 仍需独立验证。**
