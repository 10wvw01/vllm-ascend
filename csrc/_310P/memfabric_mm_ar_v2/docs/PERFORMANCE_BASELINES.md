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
### ADR-015：大 M 的 Prefill→TTFT 四层传导验收合同（2026-10-27）

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

## 2026-10-26｜A1大M完整算子重测启动：新结果暂为空

- 2026-10-21真实Prefill profiler的Stock约1944μs（MM795+HCCL1149）与V2约2573μs、额外contiguous约66μs，是**历史同场景归因**；D4的V2 2726.9→2556.3μs（内部-6.26%）属于另一无Profiler设备微基准，**严禁跨环境相减**。
- 新工作分支`work/310p-mm-ar-large-m-a1-20261026`以`A1_LARGE_M_BASELINE_AND_BOTTLENECK_20261026.md`和纯CPU`a1_large_m_assess.py`重建M2048/4096完整FP16 Stock+HCCL与最新D4 V2+MemFabric同环境对照合同。**截至当前未取得新真实Stock/V2双RankABBA样本，差距=UNKNOWN、硬件子相位耗时=UNKNOWN、模型收益=UNKNOWN。**
- 旧U1工具只作设备侧G0后才能被授权执行；M1024只可作诊断形状，不当作正式Serving融合命中；Screen不授予GO，Confirm有独立跨流HCCL/路由证明也只可能得到算子候选结论。

## 2026-10-27｜历史L1～L4复用账与D4补测缺口（无新实机数字）

- [完整复用审计与最小TTFT实验组](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)：旧版2026-10-21 Prefill已留双Rank×200站点/臂CSV、Step周期及单请求时延；2026-10-22已从CSV独立复算，**原始Profiler仅服务器保留**，不要再次将历史CSV当新的独立实验。历史L1仅是真实Serving中MM与HCCL的阶段Trace和V2核时延，**并非同卡同热态无Profiler的严格完整Stock ABBA**。
- 历史离线复算完整干净Step（独立于原报告约30.9ms含拷贝口径）：Rank0 Stock 77.8525ms、V2 kernel 105.9149ms，未计输入Copy净增加28.0624ms；Rank1对应93.2048→109.9853ms，增加16.7805ms；旧报告Rank0 Prefill步周期约+31.3ms，客户端4096/1稳态约+34ms。上述均**旧版本旧负载**，不是当前D4性能，也不能简单跨Rank绝对时钟相减。
- 历史版本身份未解决：报告正文Host SHA `6e9dc2d6...`与证据`provenance.json`的`88d39285...`冲突；如需作严格跨轮比较，先核查原始部署/装载证据。历史2026-10-21正式4096/2048/50/10回归TTFT mean旧Stock11341.9ms→旧V2 11578.6ms；可证明旧版方向，不足以精确拆出queue/dispatch增量。
- **当前D4 L1严格全链、40目标站点真实D4拷贝/路由成本、双RankPrefill步、同场正式TTFT = UNKNOWN/NOT_RUN**。沿ADR-015仅补A1既有Stock确认、2短Profiler归因与4Serve正式Stock/V2 ABBA；新轮次暂未获人工NPU许可，所有数值必须标历史。

