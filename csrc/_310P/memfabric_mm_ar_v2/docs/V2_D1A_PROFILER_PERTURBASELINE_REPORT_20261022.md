# V2 D1-A：Profiler 扰动排除与稳态性能基线实机报告（2026-10-22）

> 状态：**实机执行完成**。12 个 ABBA/交叉顺序短窗口会话（formal/loop4 × external/profile/aic × 2 重复）全部通过逐样本正确性门禁；零设备侧停顿；建立三口径可信基线与异常剔除规则。
> 分支：`diag/310p-v2-gate-c-controlled-isolation-20261022`（a15013c75 之上，未改动任何正式 Kernel / MemFabric 协议 / KVER11 部署）。
> 上游输入：Gate C 实机报告（`V2_MTE2_SCALAR_GATE_C_VERIFICATION_REPORT_20261022.md`）及其证据目录。原始 Trace / msprof 数据仅存服务器（SHA256 见 `evidence/gate_d1a_20261022/trace_provenance.json`，44 件）。

## 0. 执行摘要（三个必答问题）

| 问题 | 结论 |
|---|---|
| **Profiler 扰动是否被证实？** | **未证实为稳态扰动**：在核验空闲条件下，Profiler（含/不含 AIC PipeUtilization 指标）对 formal/loop4 的清洁中位影响 ≤3.3%（与无 Profiler 外部计时重合）；Gate C 的 23–45ms 共模停顿在 D1-A 全部 12 会话 384 rank-样本中**零复现**（含 64 个 AIC 模式样本）。唯一疑似 Profiler 会话（loop4_aic_r1）经 wall−event 差值分解**归因为外部租户宿主侧 CPU 抖动**，已隔离出基线。 |
| **哪些数据可用于正式 V2 优化验收？** | §4 基线表全部 11 个清洁会话：formal kernel p50 2526–2642us / event p50 2656–2812us；loop4 kernel p50 3048–3130us / event p50 3184–3305us；跨会话离散 <3.3%；与 Gate C 清洁中位（formal 2556–2634）定量吻合。交叠收益（裸路径口径）r0 473–592us / r1 388–564us。验收规则见 §5。 |
| **哪些仍是 unknown？** | ① Gate C 23–45ms 配对停顿的**机制**（历史强 AIC 相关 + 本轮零复现 + 裸路径零停顿，无法进一步收窄；寄存器/调度器级定位未做）；② formal 会话级 2.6↔3.4–3.6ms 双态的**确切触发器**（D1-A 6 个 formal 会话全部落于 2.66–2.81ms，3.4–3.6ms 态未复现；mild-band 家族与宿主抖动链路已在 loop4_aic_r1 证实，但 Gate C formal_m10 无环境记录可回溯）；③ §6 各项。 |

## 1. D0 环境准入与 G1-D 残留核验

- **Host SO**：双路径 `libmf_hybm_core.so` 均为 `88d39285…`（Gate A/B 核验身份 = Gate C 结束态）；无 G1-D tail 残留。
- **KVER11**：JSON `25e16b0a…` 未变；设备侧 v11 双 die 探针 rc=3（原装在位，未触发重部署；探针二进制 `849f93c0…` 与 Gate C 同一）。
- **插件 SO**：`8eb65de9…`（Gate C 诊断 ON 构建，工作树 clean@a15013c75，本轮零重编译、零源码改动正式路径）。
- **环境变量**：ambient shell 无 MF_*/ASCEND_WORK 残留；执行器显式指定 KVER11 绝对路径；`MF310P_V2_NOWARMUP` unset（driver fail-closed）。
- **共机外部负载事件（重要）**：15:47–16:26 全程空闲（块 1 顺利）；**16:32 起另一租户宿主侧作业持续占用 die1**（AICore 60–99%、内存峰值 +14GB、busy 20–90s/gap 7–20s 循环，容器内不可见 PID 2112358/2114435），17:53、18:07 出现两个可用窗口完成块 2。全部被阻断的尝试保全于 `_contaminated_*` 目录（见 idle 门记录）。die0 全程空闲。
- 详见 `d1a_environment_record.json`、`idle_gate_records.txt`。

## 2. 实验设计与执行

| 项 | 设定 |
|---|---|
| 固定形状 | M=4096、chunk=1024、TP2（ASCEND_RT_VISIBLE_DEVICES=0,1）、KVER11，与 Gate C 完全一致 |
| case | formal（overlap=1 正式核）、loop4（4 Chunk 串行诊断核）；一会话一 case（Gate C 纪律） |
| 三模式 | `external`（无任何 Profiler）/ `profile`（torch_npu CPU+NPU，无 AIC 指标）/ `aic`（+PipeUtilization+Level1，与 Gate C 同参） |
| 顺序 | 块 1 `f-ext,l-ext,f-prof,l-prof,f-aic,l-aic` → 块 2 反序（ABBA），formal/loop4 交错；每会话 warmup 3 + iters 16 |
| 外部计时口径 | **wall_us**=perf_counter 包住(launch+stream sync)，含宿主发射+全设备执行+同步返回；**event_us**=双 NPU Event 夹 Kernel 的设备流区间，不含宿主发射延迟、含 event 开销。两者均**非** msprof Task Duration，禁止混同。三模式均记录同口径 wall/event，Profiler 会话另存 chrome trace + op_summary |
| 门禁 | 每样本逐次 `torch.equal` 位精确 + 64B status=0（失败即停）；会话前后 5664 双 die AICore=0 空闲门；trace 核数 = iters 精确计数（fail-closed）；拒绝覆盖既有证据 |
| 新增工具 | `test/perf/gate_d1a_perturbation.py`（驱动）、`analyze_gate_d1a.py`（fail-closed 汇总）、`run_gate_d1a_sessions.py`（ABBA 批调度 + 空闲门 + 污染隔离）；CPU UT `test_gate_d1a_source_contract.py` 4/4 |

## 3. 停顿普查与 Profiler 扰动判定

**全 12 会话 384 rank-样本：event ≥6ms 停顿 0、kernel ≥6ms 停顿 0、wall ≥6ms 仅 2**（formal_profile_r1 双 rank 样本 0，8589/8626us；其 kernel 2534/2643us 不受影响——Profiler 启动的宿主开销，非设备停顿）。

- Gate C 的 23–45ms 严格配对停顿（历史仅在 AIC 会话出现，5/约12 会话）本轮 **0/128 AIC rank-样本、0/128 profile rank-样本零复现**。结合两轮：现象**非确定性**；裸路径（external）两轮合计 160 rank-样本零停顿（95% CI 上界 <1.9%）。
- **模式等价性（清洁 p50，隔离 loop4_aic_r1 后）**：

| case | 指标 | external | profile | aic | 跨模式最大离散 |
|---|---|---|---|---|---|
| formal | event r0/r1 | 2656–2731 / 2708–2796 | 2711–2786 / 2729–2812 | 2744–2763 / 2720–2789 | 3.0% / 3.3% |
| formal | kernel r0/r1 | — | 2526–2559 / 2573–2642 | 2557–2570 / 2547–2629 | 1.3% / 3.2% |
| loop4 | event r0/r1 | 3204–3247 / 3184–3271 | 3264–3304 / 3285–3305 | 3280 / 3305 | 3.1% / 3.8% |
| loop4 | kernel r0/r1 | — | 3048–3110 / 3106–3130 | 3099 / 3107 | 2.0% / 0.7% |

（单位 us；区间为 2 个重复会话 p50 的范围；loop4 aic 仅 r2 有效——r1 已隔离，见下。）

- **event−kernel 差值稳定在 156–227us**（8 个 Profiler 会话 16 rank 全部），是外部计时与设备时长的稳定换算锚点；唯一离群 loop4_aic_r1（r0=321/r1=625us）正是其被隔离的证据。

### 3.1 loop4_aic_r1 异常会话分解（本轮唯一的疑似 Profiler 扰动，最终归因为环境）

现象：r0 kernel p50 4013us（较组内 3048–3110 +32%），隔样本交替带 3.1–3.3/4.2–4.9ms，mte2+scalar 吸收、mac 恒 458（Gate C 停顿签名家族的温和幅度版）。

分解：① 该会话 17:20 执行，处于租户 die1 活动期（前后空闲门通过但会话内 die1 内存 −123MB 漂移）；② **r1 wall−event 差值中位 419us、波动 105–1430us**，其余 11 会话全部 100–170us 包络内——r1 宿主发射路径抖动；③ r1 kernel 不膨胀（3136）而 r1 event 膨胀（+577us）= 发射延迟进流区间；④ r0 在 IB 等待 r1 迟到的 signal → r0 kernel 任务内等待膨胀。**链条：宿主 CPU 抖动 → 发射延迟 → 对端等待相膨胀**，无需 PMU 干预假设即可完全解释；且同模式重复（loop4_aic_r2，核验干净窗口）**未复现**（3280/3305，与 external 重合）。

处置：按"清洁种群"规则从基线剔除，全部数据保全（`_killed_`/`_contaminated_*` + 会话目录），作为**宿主抖动→等待相膨胀**链路的取证样本。该链路同时提示 Gate C"隔样本交替共振"与 formal_m10 mild-band 的一个可检验候选机制（见 §6）。

### 3.2 sample-0 规则

16 个 rank-会话中 15 个的 sample-0 event 高于会话中位 +75..+1781us，而 sample-0 **kernel 全部不高**（部分低于中位）——纯 Profiler 启动的宿主/流间隙效应。规则：Profiler 会话的 event/wall 基线统计剔除 sample-0；kernel 口径无需剔除。external 会话无此效应。

## 4. 稳态基线（正式 V2 优化验收可用数据）

**口径声明**：kernel_us=msprof Task Duration（设备侧）；event_us=外部流区间（≈kernel+156–227us 固定锚点）；wall_us≈event+100–170us（发射+同步）。验收对比必须同口径。

| 基线 | rank0 | rank1 | 跨会话离散 |
|---|---|---|---|
| formal kernel p50 | 2526–2570us | 2547–2642us | ≤3.2% |
| formal event p50（裸路径） | 2656–2731us | 2708–2796us | ≤3.3% |
| loop4 kernel p50 | 3048–3110us | 3106–3130us | ≤2.0% |
| loop4 event p50（裸路径） | 3204–3247us | 3184–3271us | ≤2.7% |
| 交叠收益（loop4−formal，裸 event） | 473–592us | 388–564us | — |
| formal 管道（aic r1/r2 p50） | mac 458–459 / mte2 1666–1765 / scalar 849–886 | mac 458–459 / mte2 1609–1751 / scalar 884–1080 | mac 恒定 |

- formal kernel 基线与 Gate C 清洁中位（2556/2587、2588/2634）**定量吻合**（差 <1.3%）——跨日、跨 Profiler 模式、跨计时方法三方独立重合，基线可信度高。
- loop4 kernel 3048–3130 落于 Gate C 清洁范围（3020–3609）低段；r0>r1 不对称在 D1-A 仅 ~30–60us（Gate C 串行会话曾见 ~0.4–0.6ms）。
- 交叠收益与 Gate C 方向性结论一致（本轮 0.39–0.59ms，Gate C 0.5–1.1ms；两轮区间重叠）。

## 5. 验收与异常剔除规则（后续正式 V2 优化直接引用）

1. **正确性门（不可豁免）**：每样本 `torch.equal` 位精确 + status=0 + IB probe 先行；任何失败即停机保全（507014/507018 同）。
2. **空闲门**：会话前后双 die AICore=0；会话内内存漂移 >100MB 记录并单列（本轮唯一触发即 loop4_aic_r1）。
3. **停顿剔除**：≥6ms 样本不入清洁统计，单列留痕（Gate C 阈值沿用）。
4. **sample-0 剔除**：Profiler 会话的 event/wall 剔除样本 0（§3.2）；external 会话保留。
5. **会话级交叉验证**：同组重复会话清洁 p50 离散 >5% 或 event−kernel 差值偏离 156–227us 锚点 → 会话隔离（quarantine）+ 干净窗口重跑，不销毁证据。
6. **验收口径**：优化前后对比必须同 case 同口径；最终验收数字以**裸路径 external event** 为准（无 Profiler），Profiler kernel 口径用于归因分析。
7. **幅度门槛建议**：formal 层级优化的可测信号需 >3.3%（当前跨会话噪声包络）才视为真实；建议优化目标以 ≥5%（~130us）为最小可验收幅度。

## 6. 未解决（unknown）

1. **23–45ms 配对停顿机制**：零复现不等于不存在；其"双 Rank 设备任务时长同时膨胀且配对"的形态仍无宿主抖动可解释（宿主抖动只能膨胀等待侧，见 §3.1 r1 kernel 不膨胀）。候选（PMU 缓冲回写与 AICPU 编排核/HybmSdmaOrchEpoch 干扰）仍无直接证据。判定实验：需要一轮**裸路径长窗口计时普查**（本任务禁止长压测，未执行）。
2. **formal 2.6↔3.4–3.6ms 双态触发器**：D1-A 六会话全落 2.66–2.81ms，未捕获 3.4–3.6ms 态。Gate C formal_m10 的清洁样本呈 2.5ms 与 ~4.8ms 混合（中位被拉高）——与 §3.1 mild-band 同族；但 Gate C 无环境记录可回溯，"宿主抖动"归因对该会话不可检验。
3. **loop4 r0 交替带在干净 AIC 窗口是否存在**：仅 1 个干净 AIC 会话（r2，无带）；样本量不足以给发生率置信区间。
4. **74ms**：未观测（与 Gate C 一致，仅记未重现）；Gate E P0 独立维持未关闭。
5. msprof 解析耗时 3s–8.9min 剧烈波动（宿主 CPU 争用），不影响测量窗口内数据，但影响批量取证节奏。

## 7. 结论

1. **Profiler 扰动（稳态）未证实**：清洁条件下三种模式定量重合；formal/loop4 的可信基线已建立（§4），可直接用于正式 V2 优化验收（规则 §5）。
2. Gate C 23–45ms 停顿与 formal 3.6ms 态均未复现；前者机制仍 unknown，后者最可能属于宿主抖动 mild-band 家族（formal_m10 不可回溯验证）。
3. 共机租户作业是当前最大环境风险：本轮已证明其宿主侧效应可伪装成"Profiler 扰动"（loop4_aic_r1 案例完整取证）；§5 规则 2/5 可检出该类污染。
4. 正式 Kernel、MemFabric 协议、KVER11 部署零改动；未运行 diag1/NoIB；未做长压测。

## 8. 证据索引

| 文件（`evidence/gate_d1a_20261022/`） | 内容 |
|---|---|
| `d1a_environment_record.json` | D0 准入：SO/KVER11/插件哈希、设备侧 v11 探针、环境卫生、租户负载记录 |
| `d1a_session_stats.json` | 12 会话 × 双 Rank 全量统计：wall/event/kernel p50/p90/p99/min/max、逐样本序列、停顿/配对普查、aic 管道逐样本 |
| `d1a_verdicts.json` | 模式等价性、隔离记录、offset 锚点、sample-0 规则、停顿普查结论 |
| `idle_gate_records.txt` | 23 条会话前后空闲门原始记录（含被阻断尝试） |
| `trace_provenance.json` | 服务器侧 44 件原始 Trace/op_summary/prof 目录 SHA256 清单（不入库） |

服务器原始数据（保留本机，不入库）：`/tmp/opencode/gate_d1a/sessions/`（12 会话完整数据 + `_contaminated_*`/`_killed_*` 隔离证据 + npu_smi 快照）、`/tmp/opencode/gate_d1a/c0/`。
