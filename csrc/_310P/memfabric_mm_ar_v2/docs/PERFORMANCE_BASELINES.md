# 310P3 MM+MemFabric AR｜唯一性能口径与证据基线

> 控制于 `dev/310p-memfabric-mm-ar`；2026-10-25。本文件记录**已经验证的实验结果及其适用范围**，不是为了看起来好而汇总“最大正收益”。动态优先级回到 [PROJECT_CONTROL.md](PROJECT_CONTROL.md)。

## 三层完全不同的性能账本

| 层 | 正确比较 | 必须记录 | 禁忌 |
|---|---|---|---|
| **L0：V2内部优化** | 同一Kernel/服务环境下旧V2 vs 新V2 | Tile/Chunk/Overlap、ABBA、双Rank Event与bit-exact | **不能将L0增益宣称为超越Stock** |
| **L1：完整算子价值** | 原生FP16 **Stock MM+HCCL AR** vs 最优正式 V2 **MM+MemFabric AR** | 每M的K/N/权重格式/输入Stride、真实额外contiguous/Host/Device/全同步、两Rankp50/p90/p99、无Profiler交叉、机理Trace | 不以Stock单独MM、T2 AR-only或模型Profiler的阶段直接加减代替L1 |
| **L2：真实Qwen模型传导** | 同模式同模型同编译版本的 **Stock全链路 vs 新融合全链路** | 准确40站点命中、M分布、Graph真正命中、真实模型TTFT/TPOT/ITL/吞吐及Step关键路径 | 不把普通Eager与投机推理混算，也不把Stock-vs-Stock Graph假阳性当V2对比 |

### 已入账数值

| 来源/时间 | 测试对象 | 真实结果 | 限制 |
|---|---|---|---|
| [D4 2026-10-23](V2_D4_ROUND1_ADD_TILE2048_ONDEVICE_REPORT_20261023.md) | L0，M4096，V2 Tile1024→2048 | 双Rank配对Event p50 **2726.9→2556.3µs**，等效速率 **+6.26%**；8会话×30 | **内部V2对比**，不能称L1 |
| [Prefill 2026-10-21](V2_PREFILL_ATTRIBUTION_REPORT_20261021.md) | 历史L1，真实M4096 Prefill；老V2 | 融合站点约**2573µs** vs Stock MM795µs+HCCL1149µs = **1944µs**，即当时融合约慢32%；额外`x.contiguous()`约**66µs** | 历史旧V2/Profiler/真实Serving路径；**不得挪为D4当前数值** |
| [SEL-01 2026-10-24](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-sel01-three-arm-eager-c2-audit-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/SEL01_STOCK_V1_V2_EAGER_4096_2048_TP2_BATCH10_REPORT_20261024.md) | L2：4096/2048/TP2/batch10/Eager | Stock 4/4有效、**45.98 tok/s**；V2 4/5有效、**46.49 tok/s（+1.12%）**，每轮增益正负不一；V1 0/3自然尝试有效 | V2**没有证明≥5%稳定提升**，还有507014长稳风险；V1不能宣称“更慢”，因为无有效完成场 |
| [F1-C 2026-10-25](../evidence/f1c_20261025/F1C_ONDEVICE_REPORT.md) | 小M L1：M=10、原生FP16 Stock vs T1诊断融合 | Stock **308.2µs**，T1 **425.2µs**，T1等效速率 **−27.5%**，耗时增加约38% | 只证实**旧T1 NO-GO**，不证明下一代小M单Kernel没有潜力；旧F1-1 Host初筛+20%已不可信 |
| [F2 2026-10-24](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-f2-mm-sdma-overlap-sweep-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/F2_ONDEVICE_REPORT_20261024.md) | L0，M2048/4096的D4同SO Chunk参数 | 1024优于512/256；Overlap1优于0，18/18 PASS | **仅参数选择结果**，不是Stock对照，不能终止大M整体优化 |
| [C3 2026-10-24](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-v2-c3-cleanlines-unroll4-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/V2_C3_CLEANLINES_UNROLL4_ONDEVICE_REPORT_20261024.md) | L0，CleanLines unroll4 | 名义约**+1.70%**、配对有负有正 | 未跑真实Qwen，NO-GO，不合入 |

### L1正式测试协议

1. **同一310P3单卡双Die/TP2**，固定K/N=2048（目标attention投影真实形状）、FP16和模型实际Weight格式，确认引用相同逻辑输出及数值标准；若候选要求NZ而Stock原生可直接用非contiguous输入，要把必要转换/暂存作为候选完整成本。
2. 独立进程隔离HCCL和MemFabric正式对照，控制热态、CANN/kernel SO、Host、KVER、AICPU部署版本、核频及占用；必要时用安全的混合路由实机验证复核，但不擅自宣称HCCL+MemFabric同进程Graph安全。
3. NPU Event窗口、Host Wall和每Rank分别采，按样本`max(rank0[i],rank1[i])`后计算分位数；验证Event区间真正覆盖HCCL异步通信，Event**不等于无Host空档的纯Kernel时间**。
4. 一次形状筛选可以少次采样，但正式收益要双轮ABBA以及稳定性尾部。每个失败或慢样本留证，失去正确性或进程挂死即停止、不筛掉后统计。
5. **当前大M没有在同一次严格对照实验中给出“D4 Tile2048 完整融合 vs 原生Stock完整MM+HCCL”的确定胜负数字；首个新正式实验必须补这个缺口。** 在没有新结果前，L1大M标记**UNKNOWN**。

### L2普通模型与投机推理分账

**普通Eager历史强制验收**：
- 模型`Eco-Tech/Qwen3.6-35B-A3B-w8a8`，输入4096、输出2048、TP2、max-num-seqs/batch上限10、并发10、50请求、ignore-eos、Eager；记录真实调度batch和当步模型输入M；Stock vs融合同场交叉，失败会话不能充作吞吐值。
- 输出吞吐、TTFT/TPOT/ITL（median+tail）、总执行时间、首调正确性、目标`o_proj/out_proj`站点逐层命中情况、真实HCCL与MF Kernel trace/日志。明确记录Prefill有效站点及Decode回退。
- 对站点的“节省时间”不能简单累加到端到端：要计算Prefill与Decode关键路径中的实际重叠和外部算子/Host调度变化。旧Prefill归因显示目标站点占比有限（当时报告约3.3%），只有同次对照真实阶段Trace才可确定当前比例。
- **Graph模式另列**：以Kernel/编译产物和replay hit取证，过去Graph出现stock-vs-stock假阳性，必须防止。

**DFlash投机推理新增第二场景**：
- 参照指定外部分支`HaonanWu1/vllm-ascend:dev_v24/feat-310p-dflash`，锁定确切SHA；在**我方项目主线**实现融合薄适配。
- 同投机配置的Stock vs投机融合；额外记录verify宽度`num_spec_tokens+1`、active requests、logical/physical/padded M、Graph FULL/PIECEWISE/NONE真实选择与命中、draft/target forward次数、接受率及服务级输出tok/s。不可仅凭`batch×(k+1)`理论值冒充实测M，也不得把投机收益记到融合算子。
- 默认仍是Stock；融合路由如果不支持Graph/某些shape，显式记录降级与覆盖率，不把降级样本纳入“融合命中”收益。

### 下一轮报告固定模板

1. 当前主线/候选SHA + 外部DFlash只读SHA + 实际SO/KVER/CANN指纹。
2. L0内部改善、L1完整算子Stock胜负、L2模型传导：**分别列结果**，无数据写UNKNOWN。
3. 两Rank正确性、协议状态、长尾/错误、数据形状与独立诊断trace。
4. 为什么只传导这些：真实调用数量、Prefill/Decode/Verify覆盖、输出Layout转换、等待/重叠、Graph回退、其它主要Kernel占比。
5. 一个明确的下轮Kernel改动/优化方向、rollback、实机G0准入和需要的报告链接。

## 2026-10-25 R1现场G0：NO-GO且无新模型/算子性能（R1-B整合）

- [R1原始现场报告](R1_G0_ONDEVICE_NOGO_REPORT_20261025.md)：旧基线preflight在实机发现6项门禁阻塞，四象限 Stock/V2+DFlash Eager/Graph均NOT_RUN，无L0/L1/L2数值；同期修复版`static_go=true`只说明诊断工具与该环境更匹配，不是模型已经启动。
- 旧SEL-01、D4、F1-C等性能基线保持原数值和局限；新的投机Stock/V2吞吐、TPOT、接受率、真实M/融合Kernel覆盖率仍**UNKNOWN**。

## 2026-10-25 R1-B与V2 Triton绕过：性能仍UNKNOWN

- [R1-B现场报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-r1b-stock-eager-smoke-nogo-20261025/csrc/_310P/memfabric_mm_ar_v2/docs/R1B_STOCK_EAGER_SMOKE_NOGO_REPORT_20261025.md)：G0通过，Stock+DFlash Eager加载Target后Draft初始化FAIL；V2 Eager、完整业务和Graph均NOT_RUN，L0/L1/L2无新数字。
- 历史曾引入310P3 V2 Triton条件豁免，但它只影响配置验证且从未产生真实V2 Runner性能；**2026-10-26依据ADR-013已经撤销**。D4/SEL-01/F1-C数据仍保持原口径。

## 2026-10-26｜废止Runner重建，不影响历史L0/L1/L2基线

- R2-00～R2-21的V2 Runner迁移任务和“同V2 Runner先跑Stock再融合”的前置框架工程**全部取消**；源计划已从主线删除，参见[ADR-013](DECISIONS.md)。
- **唯一合法L2**：同一已可运行的Stock+DFlash推理路径，功能开关分别关闭/启用**已有**V2融合LinearMethod；输入4096、输出2048、并发10、50请求、TP2/K7；记录真实M与每层命中。原有L0/L1仍保持不变，不新增未经实测的性能数据。

## R1-B现场证据归档（R2-00，仅事实，非新性能）

- [R1-B原始报告](R1B_STOCK_EAGER_SMOKE_NOGO_REPORT_20261025.md)和[evidence/r1b_20261025](../evidence/r1b_20261025/) 已由历史独立取证分支选择性导入唯一主线。G0 PASS；Stock+DFlash加载Target后因草稿混合Attention架构校验失败，**V2 Eager与两组full / Graph全部NOT_RUN**；新增输出速率、TTFT、ITL、acceptance、physical M / fusion hit均UNKNOWN。

## ADR-015：大 M 的 Prefill→TTFT 四层传导验收合同（2026-10-27）

**唯一最终性能目标：**真实 Qwen3.6 Prefill 的客户端 **TTFT** 正向改善（同业务配置Stock vs V2，报告p50/p90/p99和逐场差异，不预设未经确认的提高比例）。L0 V2内部性能改善只作工程指标，L1是**原生FP16完整Stock MM+HCCL AllReduce vs 正式V2完整MM+MemFabric AR**，最终以模型TTFT判定，不以输出吞吐或TPOT代替。

| 层 | 必须核对及归档 | 判定边界 |
|---|---|---|
| L1 原生算子完整链 | 同卡双Die、同M/K/N、原生FP16 Stock `F.linear+HCCL AR`的实际完成时延 vs D4及后续正式V2；双Rank原始样本、真实HCCL跨流Event/route审计、输入与weight布局、必要同步 | Stock MM-only、只计融合Kernel、旧W1/W4同步税或逐调用HCCL竞争都不能代表真实Prefill完整链 |
| L2 真实40站点 | `o_proj`10 + `out_proj`30每Rank的真实路由/次数/物理M和stride/padding；目标算子的执行流、`x.contiguous()`、Stage与输出可被下游消费的实际完成时间 | 40站点是既有目标范围，**每次请求实际命中必须实测**，不能假设总是40；不可把两Rank、相互重叠的Kernel时间直接相加 |
| L3 完整Prefill步 | 同一请求各Prefill调度步的起止时间、双Rank同步、目标站点到下一消费算子的依赖、模型其它算子和设备重叠、Step关键路径变化 | 单请求旧Profiler目标站点约占3.3%仅为历史证据，不是当前D4的实际占比或TTFT收益上限 |
| L4 客户端首Token TTFT | 正式工作负载Stock/V2同卡同环境交叉、逐请求排队/Prefill执行/首Token返回时间、p50/p90/p99及失败率；必要时另外做隔离Prefill诊断 | TTFT包含排队、Continuous Batching/Chunked Prefill及返回；**输出token吞吐/TPOT不是本大M项目的最终验收主指标** |

**测试负载不得混算：**(A) 单请求辅助归因：input4096/output1/并发1、热态Eager、TP2，仅用于区分Prefill执行路径，不替代业务TTFT；(B) **正式E2E**：input4096/output2048/TP2/并发10/50请求/Eager/ignore-eos，双方同版本同卡、同负载、记录逐请求TTFT以及真实M/融合命中；DFlash/Graph另按同模式对照，不能混进普通Eager账本。两种负载均须记录是否有首次池初始化/冷首写/首次校验，热稳性能不得和冷态混算。

**收益传导报告强制回答：**① L1真正省了多少µs；② 40站点真实命中和额外布局/拷贝后净省多少；③ 双Rank关键路径上完整Prefill步缩短多少；④ 客户端TTFT为何只变化这些（排队、调度、噪声）；⑤ 是否已能用同场证据解释差额。禁用“每层p50 × 40 = TTFT节省”的未经时序验证的推算。若没有同步的真实Profiler/无Profiler证据，则相应阶段记`UNKNOWN`。2026-10-21旧Prefill报告W1 Stock同步税与W4通信连发竞争已实证影响微基准，最新D4的真实完整L1仍`UNKNOWN`，本决策不增加任何实机数值。


1. **同一310P3单卡双Die/TP2**，固定K/N=2048（目标attention投影真实形状）、FP16和模型实际Weight格式，确认引用相同逻辑输出及数值标准；若候选要求NZ而Stock原生可直接用非contiguous输入，要把必要转换/暂存作为候选完整成本。
2. 独立进程隔离HCCL和MemFabric正式对照，控制热态、CANN/kernel SO、Host、KVER、AICPU部署版本、核频及占用；必要时用安全的混合路由实机验证复核，但不擅自宣称HCCL+MemFabric同进程Graph安全。
3. NPU Event窗口、Host Wall和每Rank分别采，按样本`max(rank0[i],rank1[i])`后计算分位数；验证Event区间真正覆盖HCCL异步通信，Event**不等于无Host空档的纯Kernel时间**。
4. 一次形状筛选可以少次采样，但正式收益要双轮ABBA以及稳定性尾部。每个失败或慢样本留证，失去正确性或进程挂死即停止、不筛掉后统计。
5. **当前大M没有在同一次严格对照实验中给出“D4 Tile2048 完整融合 vs 原生Stock完整MM+HCCL”的确定胜负数字；首个新正式实验必须补这个缺口。** 在没有新结果前，L1大M标记**UNKNOWN**。

### 历史结论与现版缺口（只读证据盘点）

- 2026-10-21旧V2：Rank0历史目标站点设备时长Stock约77.85ms/步，旧V2 Kernel约105.91ms/步（未包含另外约3.6ms/步`x.contiguous()`）；同轮设备完整Prefill步周期2364.4→2395.7ms、孤立4096/1单请求时延约+34ms。无Profiler另一场正式4096/2048/50/并发10 TTFT mean11341.9→11578.6ms（旧V2退化），不同负载不可相减估排队。
- 2026-10-22已经从提交的双Rank逐站点CSV复算并标记含74ms的异常Step；原始Ascend profiler大文件仅历史服务器保存；Host SO报告SHA前缀`6e9dc2d6`与同轮`provenance.json`前缀`88d39285`不一致：`PROVENANCE_CONFLICT`，不能跨版本混合归因。
- D4 Tile2048独立大M L0 M4096 **2726.9→2556.3µs、内部加速+6.26%**不能称为比Stock快；SEL-01全模型Eager的输出tok/s中位+1.12%方向不稳，**不能证明TTFT改善**。最新D4完整L1、真实40层L2、完整Step L3和正式L4均`UNKNOWN/NOT_RUN`。
- **当前研发循环**：A1实现留于Draft [PR #9](https://github.com/10wvw01/vllm-ascend/pull/9)，先通过CPU/G0、独立完整Stock L1，随后真实无Profiler L3热态Screen与跨Serve ABBA（L2短Profiler按需归因校准）；只有`L3_GO`且取得独立许可才执行正式4096/2048/TP2/并发10/50请求 L4。当前无Profiler真实设备Step采集工具合同尚未校准，不能拿4096/1客户端时延充当L3；不强设L3≥5%。详见[主线实验方案](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)。

## 2026-10-27｜A1-R1现版D4：L1 row-major虚高，真实NZ-Stock及L3均未获益

**来源/冻结身份**：[A1-R1独立复盘](A1_R1_实机复盘与优化优先级_20261027.md)、[原始报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/A1_R1_BATCH_REPORT_20261027.md)，卡16384/dev2,3，CANN9.1.0，Host`1dc46d7d…`、D4`2a19390c…`、Plugin`8eb65de9…`、KVER11`25e16b0a…`；A1仅Draft，不改Stock默认。性能数值都为**同一R1实验批次**，但L1/U1与L2/L3的测量形态不同，不能混成直接同负载对照。

| 指标 | Stock | V2 D4 | 结论 |
|---|---:|---:|---|
| L1 TP2 U1 M2048，row-major Stock对照 | 1649.0µs | 1345.8µs | V2 +22.5%；**非真实Serving MM基线** |
| L1 TP2 U1 M4096，row-major Stock对照 | 3043.8µs | 2513.7µs | V2 +21.1%；**非真实Serving MM基线** |
| L2真实Qwen M4096，Rank0：NZ MM + HCCL /融合kernel | 751.9+1390.9≈2142.8µs | 2510.8µs | V2慢~368µs/站点（近似、独立p50相加） |
| L2真实Qwen M4096，Rank1：NZ MM + HCCL /融合kernel | 752.1+1165.5≈1917.6µs | 2429.7µs | V2慢~512µs/站点 |
| L3无Profiler Prefill input4096/output1，S1→F1 | 2346.08ms | 2372.20ms | V2慢26.12ms |
| L3无Profiler Prefill input4096/output1，S2→F2 | 2343.25ms | 2370.96ms | V2慢27.70ms |
| L4 input4096/output2048/TP2/10并发/50请求 | — | — | **NOT_RUN_BY_DESIGN**，本轮不可声明任何正式TTFT增益 |

- **方法论P0**：U1普通row-major `F.linear`的MM-only M4096约1869µs，而真实模型的`aclnnMatmulWeightNz`约752µs。修订前面“当前D4完整L1 UNKNOWN”：现已完成**row-major版L1**，但**模型同源NZ L1对照仍欠缺**。下一轮先将`Stock_NZ`设为主参照，保留row-major仅诊断。
- **L3 P0**：经过L2 40真实站点和同步依赖证据校准，四Serve ABBA各6预热+30成功请求，Stock场间漂移仅2.83ms；V2双窗口稳定负26～28ms、p90 2451～2517ms出现重尾，当前D4为`L3_INCONCLUSIVE_OR_NO_GAIN`，不可称TTFT已经恶化或改善1.1%；**正式L4未跑**。
- **归因P0**：MM/NZ基线与通信等待可能共同贡献，不可把V2多出耗时全部直接认作MM或把40×站点p50加成“额外Host12ms”。先用隔离低侵入短Trace把MM/SDMA/IB/Pool/Add、长尾原因测清，再做Kernel单变量优化。服务器原始Profiler CSV仍为SERVER_ONLY，GitHub归档有20份按站点切片与SHA清单。

## 2026-10-27｜A1-R2真实NZ U1 Screen、受污染Confirm与D4机制诊断

> **地位：**R2 attempt3实机归档，A1工作分支运行身份`2718f41dc`（报告后追加归档至`e1c3b3d5`），固定Host `1dc46d7d…`、D4 Device `2a19390c…`、插件 `8eb65de9…`、KVER11 `25e16b0a…`，目标物理卡16384/dev2,3。权重NZ调用现与Serving统一启用`allow_internal_format=True`和`jit_compile=False`。Route真实CANN Kernel四组（Rank0/1×M2048/4096）各6次`aclnnMatmulWeightNz`+相邻`hcom_allReduce`；[路由机器证据](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_route/nz_route_verified.json)。切勿把旧R1行主序正收益与R2真实NZ混在一个基线结论中。

| 阶段/形状 | 真实NZ-Stock MM+HCCL 完整链p50 | D4完整链p50 | V2相对Stock加速率 | 正式有效性 |
|---|---:|---:|---:|---|
| **R2 Screen M1024（诊断）** | 625.677µs | 936.759µs | −33.208% | SCREEN_ONLY；非主业务形状，未单独核2048/4096同规格NZ路由证明 |
| **R2 Screen M2048** | 1057.474µs | 1367.878µs | −22.692% | SCREEN_ONLY；不能当八场Confirm |
| **R2 Screen M4096** | 2006.133µs | 2539.831µs | −21.013% | SCREEN_ONLY；与R1 L2真实NZ比Stock更快的方向一致 |
| **R2 8场Confirm M1024** | 1275.072µs | 1158.639µs | +10.049% | **INVALID / INCONCLUSIVE_DRIFT**，Stock跨场漂移2080.483% |
| **R2 8场Confirm M2048** | 1741.615µs | 1439.896µs | +20.954% | **INVALID / INCONCLUSIVE_DRIFT**，Stock跨场漂移79.675% |
| **R2 8场Confirm M4096** | 2506.341µs | 2600.430µs | −3.618% | **INVALID / INCONCLUSIVE_DRIFT**，Stock跨场漂移663.527% |

**采用顺序**：同批`nz_screen/u1_analysis.json`为有效的**初步方向判断**；八场`nz_confirm/u1_analysis.json`虽有完整Rank原始JSON，但多场污染、无独立确认审计SHA（`audit_sha256=null`），**不能签发任何正式NZ L1速比**。不能挑选看起来更好的Session重算正式Confirm。R2报告正文将Screen粗写为672/1067/2023µs，**与机器源JSON的625.677/1057.474/2006.133µs不完全一致**；本基线优先引用机器原件（[Screen](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_screen/u1_analysis.json)、[Confirm](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_confirm/u1_analysis.json)）。

**D4机制实测**：[双Rank机器报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/mechanism/diag_assessment.json) Formal 2635.925/2617.149µs、Loop4 3234.206/3213.112µs、Diag0 6378.946/412.448µs、Diag4 27033.241/26994.908µs、Diag5 1021.224/1055.573µs；[短Profiler派生摘要](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/mechanism/sp_trace_summary.json)正式`V2TrialKernel<tiling1024,1024>` Rank0/1约2343/2355µs。**机制模式非等价，不可以差值当设备内MM/SDMA/IB/Add真实阶段耗时，INNER_PHASE_UNKNOWN**；短Profiler原始Chrome仍SERVER_ONLY。

**Epoch & Serving**：本轮首个E15A模型启动可用显存27.33GiB小于90%利用率所需38.92GiB，受到外部进程占用导致VLLM初始化失败；后续5/5/15与L3、L4均无新有效数据。原R1周期尖峰仍是待验证假设，不得写“Epoch原因已确定”。[R2最终报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_FINAL_REPORT_20261027.md)。


## 2026-10-28｜经主线审阅的E1/E2/E3真实NZ完整L1对照（来源不改写）

> **主线唯一索引**：[E3证据审核及E4门禁](E3_实机证据审核与E4分层流水准入_20261028.md)。本节记录以同卡、同批和双Rank成对Event为有效统计边界，**不同实验批次的具体微秒数不能混算成百分比**；原始报告与JSON仍置于对应PR工作分支。

| 实验批次和形状 | Stock NZ+HCCL (µs) | E1 (µs) | 新候选 (µs) | 新候选相对E1 | 数据有效性 |
|---|---:|---:|---:|---|---|
| E2 attempt3 M2048 | 1087.533 | 1236.081 | E2-R1268.731 / E2-P1328.021 / E2-P512 1581.211 | 均负 | `E2_FULL_CHAIN_TIMING_VALID`；E2-P的attempt2 507014保留独立缺陷 |
| E2 attempt3 M4096 | 2022.624 | 2285.456 | E2-R2292.578 / E2-P2383.007 / E2-P512 2756.868 | 均负 | 同上 |
| E3 M2048 | 1078.412 | 1222.611 | E3-G1451.002 / E3-T1483.932 | G方向负（−15.74%仅描述性） | `INCONCLUSIVE_DRIFT`，E1漂移6.192%>5%；不得签完整批次 |
| **E3 M4096** | **2022.553** | **2271.836** | **E3-G2119.506** / E3-T2267.090 | **E3-G +7.187%（少152.330µs）** | M4096所有臂≤2.02%漂移；E3-G仍比Stock慢**96.953µs**，全批仍`INCONCLUSIVE_DRIFT` |

- E3完整L1真实NZ路由4/4、G/T修复后bit-exact/status0、八场16份正式Rank JSON和前后冻结SO身份已在[E3机器评估](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-e3-chunk2048-20261028/csrc/_310P/memfabric_mm_ar_v2/evidence/e3_chunk2048_20261028/e3_assessment.json)与[实机报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-e3-chunk2048-20261028/csrc/_310P/memfabric_mm_ar_v2/docs/A1_E3_CHUNK2048_实机报告_20261028.md)保存。E3-G M4096 p99两场2209.469/2229.079µs，E1为2379.161/2406.195µs；不能只凭p99个别数据认定长稳问题根除。
- **固定开销归因限制**：E3-G MM调用和SDMA消息都未减少，减少的是2048会合组数量及每组生产/接收DCCI、Ready、AddRelease的次数；`INNER_PHASE_UNKNOWN`。AddTile仍为2048 **FP16元素**，不是2048行。E4 AddTile和Packet级Add均为未实测待验。
- **业务状态**：E3/E4 L2 真实40站点、L3 Prefill设备Step和L4客户端TTFT全`NOT_RUN`；本主线正式Stock默认不变，既有历史SEL-01/Eager等结果不迁移为E3成绩。


**E3关键证据已归档主线（2026-10-28后续补充）**：[52份主线原始归档](../evidence/e3_chunk2048_20261028/MAINLINE_ARCHIVE_MANIFEST.md)含[E3机器评估](../evidence/e3_chunk2048_20261028/e3_assessment.json)与[原版实机报告](A1_E3_CHUNK2048_实机报告_20261028.md)，16份正式双Rank Event采样及G0/NZ/SHA逐文件与来源Git blob一致。此举**不改变数值或漂移判定**，仅消除PR#20为唯一证据存放位置的工程风险；其深层Profiler和未上传的PLog/构建日志不视作已归档。


## 2026-10-28｜E4 Attempt1：编译/正确性准入进展，**无新增有效L1性能**

- [E4主线归档原始23文件](../evidence/e4_layered_20261028/MAINLINE_ATTEMPT1_ARCHIVE.md)，原试验`work/310p-mm-ar-e4-layered-pipeline-20261028` @ `1e20d3522`，报告归档`5b727e9f`。G0 CPU80+9条、E4合同9条、BiSheng四实例含AddTile8192一次编译PASS，原生NZ Stock四组路由PASS；E3-G M2048/4096双RankSmoke bit-exact/status0 PASS（**只判正确性**）。
- 运行中宿主PID506263竞争占卡，E3-G M4096六条配对Event约2.3ms/6.5～6.8ms交替，**绝对不得作为E3-G新版性能或Stock对照**。E4-T2的torchrun被执行工具SIGTERM打断无设备执行；E4-T4/T8/PacketAdd完全未上机，全部E4候选的`Smoke=NOT_RUN / Screen=NOT_RUN / Confirm=NOT_RUN`（Confirm另受未授权约束）。**本节不新增E4性能表或速度增益**。
- 上轮可用E3-G（M4096 2119.506µs）与真NZ Stock（2022.553µs）差96.953µs仍是**旧E3批次的参考目标，不可视为E4同批对照**。E4-B全2049形状规划仅CPU验证，变长设备执行/路由未知。


## 2026-10-28｜E4 Attempt2七臂Screen（方向性，不签正式性能）与Attempt3零新数据

> attempt2（HEAD `5b727e9f9`，30样本/臂、rank max配对p50，`SCREEN_ONLY_NO_SIGNED_PERFORMANCE`）：M4096 Stock 2037.7 / E1 2261.9 / E3-G 2133.9 / e4t2 2131.7 / e4t4 2112.5 / **e4t8 2076.2** / e4p 2105.3µs——**e4t8相对E3-G −2.71%（−57.7µs），对Stock差距从96.2µs缩至38.5µs（回收约60%）**；e4t2与E3-G差0.10%（排除重建漂移）；e4p −1.34%（受"Core0须等两包"结构性限制）；M2048全部E4候选与E3-G持平（±1.3%）、仍落后Stock约32-34%。E3-G/Stock两锚与E3-Confirm认证值高度一致（96.2 vs 96.953µs），方向可信度高，但单场Screen不可签正式结论。
>
> attempt3（保护机制验收轮）**零新性能样本**：16384被宿主级外部负载分时占用（最长安静窗口约8分钟），五臂Smoke复测与七臂Screen复测均BLOCKED；候选SO与attempt2逐字节一致（`08f73f2c…`），attempt2 Screen方向维持原判。e4t8/e4t4的正式速比、漂移与p90/p99尾部判定须待授权的十四场Confirm，且须先获得连续双Die独占租约（P0）。原始Screen JSON与守护器证据见[主线归档](../evidence/e4_layered_20261028/attempt2_3_mainline_archive/)。

## 2026-10-28｜E4 Attempt3b换卡38144受守护Screen（4/7场，方向性，不签正式性能）

> 卡38144（用户授权换卡）、全程受守护、单场×30样本。M4096配对p50/p90/p99（µs）：stock 2025.8/2053.9/2058.6、e1 2340.7/2395.8/2426.4、e3g 2142.3/2197.8/2227.2、e4t2 2175.4/2236.2/2264.5；M2048 e3g落后Stock 33.6%（跨卡复证单组结构性劣势）。**守护器无性能扰动**：vs attempt2（16384、无守护）stock锚+0.58%、e3g锚+0.40%、p99近乎逐值复现。同构对照e4t2-vs-e3g差−1.55%（attempt2 −0.10%）标定单场噪声带±1.5%→e4t8的−2.71%（attempt2）仅约1.8倍噪声带，**正式裁决必须十四场Confirm**。e4t8/e4t4/e4p三场因外部调度扩散至38144未跑成，e4t8方向维持attempt2原判。详见[attempt3b报告（E4分支）](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-e4-layered-pipeline-20261028/csrc/_310P/memfabric_mm_ar_v2/evidence/e4_layered_20261028/A1_E4_attempt3b_换卡38144报告_20261028.md)。


## 2026-10-29｜V3-only短测使用历史对照：无新增设备数值

用户要求不重复运行V2和真实NZ Stock性能臂。V3工具读取E3 `e3_assessment.json`、
E4 attempt2 `attempt2_e4_screen_summary.json`及本表R2 D4 Screen记录，保留来源SHA、
`INCONCLUSIVE_DRIFT`/`SCREEN_ONLY`标签，输出每个V3形状相对每份历史基线的独立差值。
双Rank逐样本取max后计算V3 p50/p90/p99，前后两轮检查漂移；不同版本/卡/KVER的差值仅供方向筛选。
**V3 CANN编译/实机/性能NOT_RUN，没有任何V3收益数字或正式GO。**
历史数据不得被重新归类为本次同批对照；脚本不自动重跑旧臂，不启动模型/TTFT。

## 2026-10-29｜V3首次实机：功能通过，性能目标失败

来源：V3分支`6144af16`的[完整原始记录](../../memfabric_mm_ar_v3/experiments/quick-20261029-r1r2r3/README.md)；
执行`f2c3eaf0`，MemFabric`db38d75e`/KVER28，38144 dev4/5。
正确性五关双RankPASS，4配置×2轮×30样本，先逐样本取Rank max再算分位数，复算一致。

| M | V3 group1024 | V3 group2048（较快） | 历史E4/E3-G | 历史E4/NZ Stock | 最佳V3时延增加：V2 / Stock |
|---|---|---|---|---|---|
| 2048 | 1897.604 | 1709.941 | 1457.929 | 1088.372 | 17.29% / 57.11% |
| 4096 | 3343.041 | 2883.197 | 2133.945 | 2037.734 | 35.11% / 41.49% |

单位微秒；V3列为两场配对p50的平均，跨轮漂移0.28%～1.37%。
时延增加不等于吞吐下降同一百分比，来源为跨批Screen，不升级历史原判定或签同批正式GO。
研发裁决为当前V3无收益、停止推进集成，按chunk512/1024/2048重排流水；
新设计详见[复核报告](../../memfabric_mm_ar_v3/PERFORMANCE_REVIEW_20261029.md)，V3.1性能UNKNOWN。
用户明确旧V2/真实NZ Stock不重跑，继续复用以上历史数据；Stock默认不变。

## 2026-10-29｜V3.2方案A编码，无新增实机数值

新路径为单Y pool-return，完整窗口必须包含原流同步、双Rank输出槽协商、Tensor包装、
Kernel和最终完成；保留Host Wall与配对Rank max分位数。不将无末尾copy的结果冒充旧out接口收益。
CANN/Storage/设备门禁及性能NOT_RUN；继续复用历史V3.1、V2 E3-G及真实NZ Stock，
不重跑旧性能臂，不增加独立归因轮。[准入单](../../memfabric_mm_ar_v3/V32_ADMISSION.md)。

## 方案A2修复：无新增实机数据

新增Storage-ready prepare Host allgather，计入pool-return事件窗口及Host Wall，
不得从完整性能口径中删除。引用回收/关闭一致性是可靠性修复，不宣称加速；
历史基线不变，新增设备性能NOT_RUN。

## A3编码交付：无新增设备性能数值

A3完整计时包括本地资源/Storage、完成事件提交、设备START、NZ MM、显式seed、ADD及排空。
计算候选compute512/1024与通信comm512/1024/2048分别记录；默认六配置固定compute1024。
比较历史A2/V3.1/V2/真实NZ Stock，不把新增S/seed成本排除，不签跨批正式性能GO。
设备编译/数值/生命周期/性能NOT_RUN，见[A3_ADMISSION](../../memfabric_mm_ar_v3/A3_ADMISSION.md)。

## A3首READY修订：无新增性能数值

profile3302减少一封正常控制消息、计算前跨Rank等待、本地fan-out和独立ACL memset任务。
首READY由8B扩为64B携带必要校验；其余数据、seed、事件及最终排空全部计入完整窗口。
实际节省微秒数未测量，不承诺收益；新设备编译/门禁/性能NOT_RUN。


## A3-2/A3-tail候选：无新设备数值

同批原A3前/七核专职/末尾席位/原A3后，各64/128行MM一组，共八会话。
维持完整pool-return窗口、双Rank配对max及Host Wall；每臂跨轮和A3前后漂移>5%不签稳定收益。
历史Stock仍只作跨批方向比较，不重跑旧性能臂；当前设备编译/数值/性能全部NOT_RUN。
详见[席位设计和执行合同](../../memfabric_mm_ar_v3/A3_SEAT_SCHEDULE.md)。


## 统一计时/归因代码：无新增实机数字

新合同ND_X_NZ_W_TO_READY_ND_Y_V1，统一Stock/V2/V3/A3 Tensor调用至输出完成，
所有每次必需成本保留；无打点性能与Host/Device打点分别报告。旧历史结果不重标为新合同。
11会话同批对照与V2冻结见[UNIFIED_TIMING](../../memfabric_mm_ar_v3/UNIFIED_TIMING.md)。
打点扰动与漂移检查、每核ticks占比不等于全局串行分解；新设备编译/测量NOT_RUN。


## 2026-10-30｜历史实验与数据集中整合，无新增性能结论

各分支V1/V2/V3、E1～E4、G1取证、A1真实模型、A3系列的脚本及数据进入主线；
原始blob、来源HEAD和NO-GO/共享卡/漂移/NOT_RUN限制保留，不改写旧报告。
索引见[整合清单](../../memfabric_history/manifest.json)。算法及实机推进已按用户要求暂停。
最新共享卡统一计时报告的摘要应结合配对JSON阅读：原A3对A3-2实际9/12配置更快，
Host打点存在约+6.3%个例；本整合不重算或替换原始数据，也不签正式稳定收益GO。


## 2026-10-30｜A3管线Profiler原始证据归档（b34901ba）

新增来源为用户现场授权采集后提交的A3管线计数，包含双Rank CSV与脚本；
[原始报告及数据](../../memfabric_mm_ar_v3/experiments/a3-unified-shared-card-20261029-r1/README.md)。
本次只保全原文与原始数据，不重算、修订或升级其因果结论。管线计数存在并发重叠，
不能由归档本身证明MM/seed/wait的互斥墙钟占比，也不改变共享卡、未测SDMA及非发布GO限制。
版本与逐文件Git blob校验见[增量记录](../../memfabric_history/INCREMENTAL_ARCHIVES.json)。

## 2026-10-30｜新统一L1统计合同与历史分段证据整理

新合同L1_TENSOR_IO_V2用完整Tensor输入到ND输出ready，另加真实下游残差ADD的chain场景；
Stock为NZ FP16 MM+HCCL完整输出。相同W/N/R，逐同下标双Rank max后算术均值，所有样本保留。
无打点性能与带前后控制的Host/Profiler诊断区分，独占检查跨轮/候选重复漂移，共享保留漂移而不签稳定GO。
[统一入口](../../../../csrc/_310P/tools/L1_test/README.md)、[36行历史Host阶段数值](L1_PERFORMANCE_EVIDENCE.md)。
后者保持旧合同p50范围，不变成新均值。Device时钟、SDMA真实占用与核内互斥阶段仍UNKNOWN。
本轮无CANN/NPU新数值，不改写既有性能结论；不能由新CPU工具或文档推导任何新正收益。

## 2026-10-30｜统一L1框架已直接纳入主线，无新增实机数值

按用户指令将PR #23内容纳入`dev/310p-memfabric-mm-ar`，后续实验直接从主线固定SHA执行并归档。
428项CPU测试及36个子测试不代表设备性能；新框架CANN/NPU仍NOT_RUN，既有历史数值与限制保持不变。
新的性能记录仍必须保全双Rank原始样本、运行身份、独占/共享标签、完整输入输出合同及前后控制。

## 2026-10-30｜L1框架目录迁移，无新增性能数值

统一入口迁至`csrc/_310P/tools/L1_test/l1_bench.py`，计时边界、预热/样本数、双Rank配对均值、独占/共享与采集语义不变。
此轮只验证目录迁移及调用链，CANN/NPU无新增执行；已有性能数值和UNKNOWN限制保留。

## 2026-10-30｜V1/V2封版选型仅复用历史

已有four-batch机器评估及源码身份归入[审核包](../../releases/v1-v2-20261030/README.md)，没有新性能数值。
five-arm/E2评估在主线历史blob内可读取，不再要求服务器重复提供。旧p50、漂移/Screen标签和形状差异完整保留。
V1 ABI7最新配置与后续稳定性证据待找；V2 E4-T8的2076.211µs/M4096仍是单场Screen，不升级为正式全形状winner。
本轮不重跑、不拼新混合路由、不修改生产默认；完整数值与取舍见[RELEASE_SELECTION](RELEASE_SELECTION.md)。


## 2026-10-30｜大M性能优先选择与原始Wall复算，无新设备数据

用户确认M2048～4096，V1只从成功大M记录选。旧报告R2/R3中q4/batch_m1024 M4096约2.36ms，
优于q2约2.73ms、q1约3.31ms；选q4，仅补准确源码/二进制身份，不混比不同批次V2耗时。
V2同一E4 attempt2的14份原始Rank JSON已抽出校验；每臂30样本，按每序号Rank max后聚合：

| M | 性能选择 | 完整Wall算术均值µs | Wall p50µs | 历史Event p50µs |
|---|---|---:|---:|---:|
| 2048 | E1 | 1394.627 | 1398.185 | 1279.596 |
| 4096 | E4-T8 | 2175.690 | 2156.940 | 2076.211 |

全部样本保留，无跨批次拼排名；没有把旧p50改名为均值。T4/M4096 Wall均值2175.818µs，
仅比T8高0.128µs，实际近乎持平；本次按观测值选T8，保留单轮Screen不确定性，不要求Confirm重跑。
D4为现有正式构建，非本轮性能选择：five-arm M4096同批D4=2512.995µs、E1=2274.167µs，
E1耗时少9.50%，D4/E1漂移0.476%/0.129%；M2048 D4漂移5.685%原判保留，不挑样本。
完整Wall含调用/提交/流完成；原件、来源、计时批次实际Host/Device及源码身份见../../releases/v1-v2-20261030/。
V2实测点已选，区间内部未知；该代内部选择不表示超过Stock，生产默认不变，无新增实机测量。
