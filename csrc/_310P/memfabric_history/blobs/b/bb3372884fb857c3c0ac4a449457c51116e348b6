# V2 MTE2/Scalar 专项 Gate A：Prefill 证据与统计口径审计（2026-10-22）

> **2026-10-22 现场核验更新**：本审计的待办项已由 `V2_MTE2_SCALAR_GATE_AB_VERIFICATION_REPORT_20261022.md` 在服务器上执行并关闭（拷贝归因结论被修正、跨 Rank 校时裁定为真实偏斜）。

> 状态：**GitHub 公开证据离线审计完成；服务器运行态核验待现场执行**。不把静态审计标成新一轮实机 PASS。  
> 审计分支：`diag/310p-v2-mte2-scalar-gate-ab-20261022`，从 `perf/310p-v2-prefill-attribution-20261021@5701ec3f` 建立。  
> 性能证据来源：`evidence/v2_prefill_attribution_20261021/`，以及该分支中的 `V2_PREFILL_ATTRIBUTION_REPORT_20261021.md`。  
> 正式服务端基线：`feat/310p-memfabric-public-api`；Eager、TP2、M=4096、chunk=1024、slots=4、MemFabric KVER11。Graph 不在本项目范围。

## 1. 判定范围与证据原则

Gate A 的目标是建立**可以审计、可重复计算、可复用的真实模型性能归因基线**，不是证明 SDMA/IB 是唯一瓶颈。本轮没有连接 310P3 运行环境，无法单靠仓库材料验证服务器当前被加载的二进制或原始 Trace 的完整性。

- **observed**：原始 `sites/*.csv`、`provenance.json`、日志节选、`kernel_internal_breakdown.json` 实际记录的值；
- **derived**：从这些值明确计算出的汇总量及其假设；
- **unknown**：真实加载路径、跨 Rank 硬件时钟基准、Kernel 内部通信/IB 的独占时间、单次长尾根因。
- **不得混合**：T2/W1 Host 同步调用完成时间、Serving kernel_details 设备执行时间、TP 两 Rank 的累加时间。

## 2. Host SO / KVER 版本核对：矛盾已有明确来源

| 证据批次 | Host core SHA256 前缀 | KVER11 JSON SHA256 前缀 | 说明 |
|---|---|---|---|
| G1-D 原装/恢复态记录（§2.2/§9） | `88d39285` | `25e16b0a` | G1-C/G1-D 原装恢复态 |
| W1/W4 ABBA 诊断部署 `sessions/m4096_r12/provenance.json` | `6e9dc2d6` | `25e16b0a` | G1-D 诊断库曾处于在位状态；与恢复态不同 |
| **Prefill 本轮 `provenance.json`** | **`88d39285`** | **`25e16b0a`** | 与原装恢复态一致，不等于 W1/W4 诊断部署态 |

**裁定 A-01（已关闭的表面矛盾）**：此前“Prefill 的 Host SHA 不同于 W1/W4”是事实，但 Prefill 与 G1-D 原装恢复态的一致性也是事实；不能因为两个实验使用不同 Host SHA 就宣布 Prefill A/B 失效。**同时也禁止宣称与 W1/W4 的运行环境逐字节相同**。Prefill Stock/V2 同一会话对照仍须现场核验各自实际加载的包内 `libmf_hybm_core.so`、`/usr/local` 两个路径、`/proc/<PID>/maps` 和 Plugin SO。

设备侧 AICPU v11 的 SHA、`MF_SDMA_ORCH_JSON` 实际环境、是否安装过诊断 v12，都不能通过此 JSON 的一项字段间接证明；现场须验证并归档。

## 3. 逐站点重算：完整稳态步优先

从 GitHub 的 `sites/stock_r{0,1}_sites.csv`、`sites/v2_r{0,1}_sites.csv` 重算，四份均为 **5 步 × 40 站点 = 200 行**。V2 第 0 步有一个 74ms 异常，故本次稳态 V2 以**完整的第 1～4 步**作为主口径；Stock 用第 0～4 步近似对照，并将会话未配对的局限写明。

| 对照值（每 Rank 每个 Forward，ms） | Rank0 | Rank1 |
|---|---:|---:|
| Stock MM+AR 逐站点时间之和，五步均值 | 约 77.85 | 约 93.20 |
| V2 融合 Kernel，完整无异常四步均值 | **约 105.92** | **约 109.99** |
| 净差（**不含额外拷贝**，仅设备事件时间求和） | **+28.06** | **+16.78** |
| 第一站点到下一步第一站点的稳态周期差 | 约 **+31.3** | 约 **+31.8** |

- 以上“站点之和”不是可直接相加的因果关键路径，不表示相邻算子必然无重叠。
- Rank1 的站点和周期差距比 Rank0 大，**不能用站点汇总直接宣布双 Rank 归因闭合**。Rank 偏斜、TP 等待和事件配对需保持待查。
- 单步周期是第一目标站点锚定的**模型步周期代理指标**，未包含“首站点之前/最后站点之后”的完整 Model Forward 边界，不应改名为严格 Forward device latency。
- 上轮报告 `final_attribution.json` 将异常时长 `>=10000us` 的一个 V2 Kernel 剔除，但 `v2_sum_per_step_us = sum(199)/5`，实际形成“缺一个站点按 0 算”的近似。若需要评价整步成本，应使用 **完整无异常的四步**，或明确补齐估计的模型和误差。
- V2 74ms 事件属于窗口第一步第 9 个左右的目标站点（确切位置以 `site_idx` 复查），**不可用“无故障”替代对长尾的解释**。

用于复算的标准库脚本：`test/perf/audit_prefill_gate_a.py`，输出独立 `gate_a_audit.json`，不覆盖历史证据。

## 4. 跨 Rank 时钟校准脚本有符号错误

历史 `evidence/.../scripts/compare2.py` 定义 `off = median(t_rank0 - t_rank1)`，然后以 `t1c = t_rank1 - off` 将 Rank1 时间换成 Rank0 时间。按这个定义正确方向应为：

```python
off = median(t_rank0 - t_rank1)
t_rank1_on_rank0_clock = t_rank1 + off
skew = t_rank0 - t_rank1_on_rank0_clock
```

**裁定 A-02**：该脚本给出的任何“跨 Rank 绝对时间偏斜”都不宜直接用于结论。修符号后，仍须验证 Rank0/Rank1 对应 HCCL 事件确实是同一代调用、没有丢行/多行，并检验偏移是否随时间漂移。每 Rank 内事件时长（原始 duration）不因此失效。

## 5. contiguous 拷贝结论存在待消歧证据

上一轮报告写“V2 每站点增加 `x.contiguous()` 约 66us；Stock 零额外拷贝”，但同份 `analysis/host_api_summary.json` 中：
- **Stock** 的 `aclnnContiguous` count = **200**；
- **V2** 的同名 Host API 没出现在 Top 列表，设备时间线另报有约 199 个 D2D 拷贝。

**裁定 A-03（待核验）**：仅凭名称计数不能推断 Stock 的 200 次 `aclnnContiguous` 都属于目标层，也不能据此否定 V2 的 D2D；但“Stock 无任何 contiguous”现在不是已经交叉验证的事实。必须用原始 trace 的 **关联算子、输入 stride、拷贝前后设备 Stream、buffer 地址或 PyTorch operator 栈**确认两臂真正多出的净拷贝成本。当前将“V2 净 +3.6ms”列为**条件估计**，不作为 Gate B 优化验收条件。

## 6. MTE2 / Scalar 语义边界

Rank0 计数器 p50（V2 vs Stock MM）为：
- MAC：458.9us vs 456.6us；
- MTE2：1711.4us vs 335.9us；
- Scalar：872.8us vs 186.5us；
- MTE3：191.1us vs 91.5us。

这些是各执行单元活动/利用率相关的硬件统计，**不是 Kernel 源码中“SDMA signal 到 wait 返回”或“IBWait 独占耗时”的直接计时**；并行单元时间不具有朴素可加性。可以把传输、缓存维护、IB/消息轮询列为候选，但不能把 `mte2_time` 直接命名为远程 SDMA 等待，或将 `scalar_time` 直接叫协议轮询税。

`GetSystemCycle/PrintTimeStamp` 是否支持 310P3 也必须独立核对；CANN9.1 公开 API 文档对“Atlas 推理系列 AI Core”列为不支持，禁止直接套用 A2/A3 的设备内埋点代码：
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/基础API/工具接口/系统资源与变量/GetSystemCycle_ISASI.md
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00002.html

## 7. 微基准比较与 Serving 负载的保留限制

1. T2/W1 同步调用耗时 `~3.27ms` 不等于 Serving Profiler MM+AR 设备核之和 `~1.94ms`：不可直接用差值宣称“HCCL 在 Serving 达到最佳工况”。
2. Stock 和 V2 原始 50×(4096/2048) ×10 吞吐复测仅一对 AB，方向支持，但统计强度有限；切勿把 TPOT 全部分位变化归结为 Prefill。
3. Profiler 请求使用重复 Token `[64] * 4096`，原 G3 使用随机 token；MoE 路由分布不相同，结果外推须有限定。
4. 当前归因工作专注**Eager M=4096**，不含 Graph、M=4095 或 Gate E 长稳。

## 8. Gate A 状态与下一步准入条件

| 项目 | 当前状态 | 现场确认后才能关闭 |
|---|---|---|
| GitHub 版本来源审计 | **PASS（版本含义已澄清）** | 现场 Host / Device 实际加载身份仍待核 |
| 双 Rank 站点 CSV 算术复算 | **PASS** | 保留原始 trace CSV 与每步来源校验 |
| V2 74ms 异常剔除口径 | **已纠正** | 不修改原报告原始数据，新增独立结果 |
| Rank 时钟符号审计 | **发现缺陷** | 用原始 Rank0/Rank1 同代事件重算，核实漂移 |
| contiguous 净拷贝归因 | **待核实** | 关联原始 trace 进行算子/地址级对照 |
| Pipe 计数器 → 内部阶段耗时 | **尚不能映射** | 进入 Gate B 的能力选择，不做虚假定量 |

**Gate A 离线审计完成；设备加载、精确跨 Rank 时间和拷贝归属的最终关闭仍需服务器原始 Trace。** 若上述阻塞项未关闭，可讨论 Gate B 的设计，但不进入声称“精确分段时间已知”的阶段。

## 9. 现场最小补证（仅一次环境/文件核验，无需再跑长压）

- 对 Stock/V2 两个原始会话或保全的启动记录核对实际 `CDLL` 路径、`/proc/<pid>/maps`（进程尚存在时），以及 `/usr/local` 和 site-packages 的 SHA256；如进程已结束，只能验证**当前在位文件**，不能倒推历史实际加载路径。
- 核对原始两臂、双 Rank `kernel_details.csv`、`operator_details.csv` 与 trace 目录 SHA256、大小及导出时间；不运行覆盖原目录的分析。
- 关联 Stock `aclnnContiguous` 200 次与 V2 D2D 199 次的具体操作；不得简单抵消计数。
- 修复离线 Rank 时钟变换并检查事件同代匹配；不能校准则只使用每 Rank 的局部时间统计。
- 记录未解决的 G1-D Gate E P0（通信邮件可见性/写入缺失分支），**性能取证不等于稳定性结案**。
