# 310P3 单卡双 Die MM + MemFabric AllReduce 融合算子｜项目唯一入口

> **代码目录规则（2026-10-30）：310P融合算子相关实现、工具与实验代码集中在`csrc/_310P`；统一L1框架、入口与测试的唯一目录为`csrc/_310P/tools/L1_test`。**

> **2026-10-30用户最新指令：后续直接在`dev/310p-memfabric-mm-ar`开发、修复、执行实机实验并归档。** 实验从主线固定完整SHA，使用隔离构建/结果目录；不再默认创建工作分支或等待PR合并才允许测试。G0、CPU合同、私有编译、短Smoke、正式性能及故障STOP要求继续有效。以下旧分支流转规定仅作历史背景，以本条为准。

> **权威研发主线：`dev/310p-memfabric-mm-ar`。** 本文件及本分支下的 `PROJECT_CONTROL.md` 是当前状态的唯一权威入口；其他 `feat/`、`perf/`、`diag/`、`docs/` 分支为历史/取证/待评审资产，不得按名称推断最新状态。更新日：2026-10-28（UTC+8）。
>
> **请新 ChatGPT 会话、OpenCode 及新接手开发人员先读这里，再读下列链接，不要仅凭聊天记忆或旧版 PLAN.md 制定计划。**

## 2026-10-30｜用户授权V1/V2/V3研发资产主线整合；算法推进暂停

当前唯一工作入口为`dev/310p-memfabric-mm-ar`。V1/V2/V3及系列候选、测试工具、
实验原始数据已集中整合；54条项目分支的源码版本及冲突历史完整保全，
[整合目录与恢复入口](csrc/_310P/memfabric_history/README.md)、
[冻结分支清单](csrc/_310P/memfabric_history/manifest.json)。
Serving默认及V1/V2现有构建入口保持不变，历史失败/未准入方案不因合并自动启用。
本轮只做分支管理、冲突处理及CPU/源码验证，不推进新算法或实机实验。

## A3-2 / A3-tail新候选：同批对照待实机

新增7核MM+Core0专职通信与Core0末尾席位实现，保留原A3首READY版为同批基线。
S/Y按ND完整行连续，NZ为权重格式。[设计与八会话执行入口](csrc/_310P/memfabric_mm_ar_v3/A3_SEAT_SCHEDULE.md)。
当前设备编译/正确性/性能NOT_RUN；不启用Serving，MemFabric依赖不变。

## 2026-10-29｜V3.2方案A2服务器准入与短测完成：正确性全过，性能仍不过

OpenCode按[执行单](csrc/_310P/memfabric_mm_ar_v3/OPENCODE_EXECUTION.md)在38144双Die独占窗口完成A2一次性验证，
[实验记录](csrc/_310P/memfabric_mm_ar_v3/experiments/quick-20261029-v32a2-plan-a/README.md)。
起点`213cfcedb`，两个执行BUG修复后HEAD `0f638d511`（CANN ACL头路径；`SetDesc`未从libtorch_npu
导出，改经torch 2.10 `c10::make_storage_impl`注册工厂+按SetDesc语义填`npu_desc_`，未退回普通
from_blob）。MemFabric锁定`db38d75e`未改动。CPU102条、私有SDK/SO/Storage适配冻结、
G0与双Die连续独占、六门禁双Rank全PASS（含单边持有busy close、单chunk排空、环回绕、下游ND消费）。
六配置两轮短测漂移1.17%~2.69%，最佳M2048/4096均chunk1024：2018.7/2727.1µs（双Rank max、
完整接口窗口含acquire同步/双Host协商/Storage包装）。**相对历史V3.1最佳慢21.9%/8.4%，
相对V2 E3-G慢39.1%/28.7%，相对真实NZ Stock慢87.2%/34.8%——A2方向性判定未达性能目标**，
接口成本为主因结构（未做归因轮）。KVER28自拷贝差异仍在，不签长稳/发布GO；Stock默认不变。
详见控制台§39。

## 2026-10-29｜V3实机已完成，性能失败，进入V3.1流水重排

已审核V3分支`6144af16`的[原始R1/R2/R3证据](csrc/_310P/memfabric_mm_ar_v3/experiments/quick-20261029-r1r2r3/README.md)：
执行`f2c3eaf0`、MemFabric修复`db38d75e`/KVER28；CPU87条、CANN编译、五关双Rank正确性全部PASS。
四配置两轮30样本全部完成；最佳group2048在M2048/4096为1709.941/2883.197微秒，
相对历史V2/E3-G仍慢17.29%/35.11%，相对同批历史NZ Stock慢57.11%/41.49%。
**当前V3性能目标失败，不推进集成；下方开发期NOT_RUN只保留为历史状态。**
研发[复核与新设计](csrc/_310P/memfabric_mm_ar_v3/PERFORMANCE_REVIEW_20261029.md)
固定chunk512/1024/2048，沿用V2对应MM tiling，重排MM/Reduce/Gather并减少包、同步和MTE串行。
V3.1代码与CPU88项已完成，CANN编译及实机仍NOT_RUN；[编译准入单](csrc/_310P/memfabric_mm_ar_v3/V31_ADMISSION.md)。
继续只测V3并复用历史V2/Stock，Serving默认不变。
OpenCode负责执行、执行BUG修复、数据/证据与远程提交；方案、协议与结果审查由研发负责。

## 2026-10-29｜V3 SDMA ADD开发与V3-only快速历史对照

用户已批准V3：移除本地Vector Add，采用单Kernel MM + ADD Reduce-Scatter + COPY All-Gather。
独立工程见[csrc/_310P/memfabric_mm_ar_v3/README.md](csrc/_310P/memfabric_mm_ar_v3/README.md)，
工作分支`work/310p-mm-ar-v3-sdma-add-20261029`从主线`54d5518d`派生。
用户最新要求是**只跑V3短测、复用历史V2/真实NZ Stock，不重新跑旧性能臂**；
该快测只提供历史方向对照，不签同批正式收益，不执行Serving/TTFT/Confirm。
代码包含原语/协议/融合/串行与流水/尾块/Torch薄适配、私有构建和一键有界测试。
当前仅CPU与静态检查，**BiSheng/310P3实测NOT_RUN**；Stock默认不变。

上游MemFabric已同步：GitCode `ba3e7f42` → GitHub镜像 `b9d46f97`（整树一致，commit不同）。
V3最新依赖和OpenCode两仓私有构建/部署流程见[固定执行单](csrc/_310P/memfabric_mm_ar_v3/OPENCODE_EXECUTION.md)。


## 2026-10-28 最新权威状态（先读，避免把历史NOT_RUN当现状）

**E3实机已经完成，E4编码已开工；二者的准入阶段不能混淆。** 真实NZ Stock M4096=2022.553µs、E1=2271.836µs、E3-G=2119.506µs、E3-T=2267.090µs。E3-G较E1提升7.187%（缩短152.330µs），但**仍慢Stock 96.953µs**；M2048 E1两场漂移6.192%，所以E3整批仍`INCONCLUSIVE_DRIFT`，不挑样本签正收益；真实M2049～4095尚未做融合实机性能。E3修复过一次E3-T L0C满载507015 AICore异常，E2-P曾遇507014且根因未闭环。**Stock一直是Serving默认**，E3/E4的L2真实Qwen站点、L3 Prefill Step和L4客户端TTFT皆`NOT_RUN`。

- **最新且唯一的方案/证据入口**：[E3实机证据审核与E4分层流水准入](csrc/_310P/memfabric_mm_ar_v2/docs/E3_实机证据审核与E4分层流水准入_20261028.md)、[ADR-020](csrc/_310P/memfabric_mm_ar_v2/docs/DECISIONS.md)、[主线PROJECT_CONTROL](csrc/_310P/memfabric_mm_ar_v2/docs/PROJECT_CONTROL.md)、[主线PERFORMANCE_BASELINES](csrc/_310P/memfabric_mm_ar_v2/docs/PERFORMANCE_BASELINES.md)。
- **E3原始证据已在主线保全52份、非只靠PR #20**：[主线归档Manifest](csrc/_310P/memfabric_mm_ar_v2/evidence/e3_chunk2048_20261028/MAINLINE_ARCHIVE_MANIFEST.md)、[机器评估](csrc/_310P/memfabric_mm_ar_v2/evidence/e3_chunk2048_20261028/e3_assessment.json)、[实机报告](csrc/_310P/memfabric_mm_ar_v2/docs/A1_E3_CHUNK2048_实机报告_20261028.md)，包括16份双Rank原始JSON、真实NZ路由、G0及SO身份，**逐文件Git blob SHA与来源一致**；[E3 PR #20](https://github.com/10wvw01/vllm-ascend/pull/20)仅作实验Kernel与重型Profiler补充，不合入正式源码。
- **唯一活跃研发实现**：[E4 PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)，短分支`work/310p-mm-ar-e4-layered-pipeline-20261028`，首版AddTile与Packet分半组Add候选、2049个M的**纯CPU**混合分组/安全路由规划器。**E4 CPU/CI、CANN编译、310P实机、性能尚NOT_RUN；连续M设备Kernel及生产自适应路由均未实现**。
- **下一步**：E4源码协议深审与CPU/独立BiSheng→G0/双Rank短Smoke→七臂短Screen→有收益再经单独批准十四场Confirm；之后实现连续M混合尾块设备路径，最终和真实Qwen业务TTFT层级分别验证。禁止用户未授权时启用自动Serving路由或直接merge实验分支代码。

### E4 2026-10-28第一轮设备执行结论（ADR-021）

- E4 **G0、原仓CPU80条、E4合同9条、BiSheng四变体（包括AddTile8192）、原生NZ路由4/4、E3-G正确性Smoke**已经完成；尚无任何E4新变体设备Smoke与性能。运行中目标16384 dev2/3遭跨四卡宿主PID506263抢占；E4-T2进程还独立遭执行工具SIGTERM中止，未运行Kernel。**七臂Screen NOT_RUN，正式十四场Confirm未授权/NOT_RUN，Stock默认无变化**。E3-G受污染的约2.3/6.5～6.8ms交替Smoke不允许作性能比较。
- 主线已独立保全[23份E4 Attempt1原始证据](csrc/_310P/memfabric_mm_ar_v2/evidence/e4_layered_20261028/MAINLINE_ATTEMPT1_ARCHIVE.md)、[中文原始实机报告](csrc/_310P/memfabric_mm_ar_v2/evidence/e4_layered_20261028/A1_E4_第一轮实机验证报告_20261028.md)、[STOP事件报告](csrc/_310P/memfabric_mm_ar_v2/evidence/e4_layered_20261028/stop_incident/STOP_INCIDENT_ANALYSIS.md)，拷贝文件Git blob SHA与来源一致；E4实现仍在[Draft PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)。
- **下一步先修实机运行治理，才复测算法**：现场必须取得连续双Die独占窗口，补强G0后实时外部进程出现即STOP（现30秒巡检只记录而不阻断）并让OpenCode以独立session启动及轮询消除工具超时SIGTERM；完成CPU负面回归、两轮静审后重G0核对SO/HEAD，再做E4-T2/T4/T8/P正确性Smoke与七臂Screen。不得伪称E4本轮有性能增益或已支持任意M设备执行。

**E4测试保护实施进展（2026-10-28）**：已在原[E4 PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)实现`e4_stage_guard.py`主动占卡STOP与`e4_detached.py`独立Session启动/status/退出码，全部四处torchrun接入守护，新增纯CPU故障注入合同，详见[主线PROJECT_CONTROL第28节](csrc/_310P/memfabric_mm_ar_v2/docs/PROJECT_CONTROL.md)和[现场操作指南](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-e4-layered-pipeline-20261028/csrc/_310P/memfabric_mm_ar_v2/docs/A1_E4_最小实机测试保护与复跑操作_20261028.md)。**这是工具代码完成，不代表新的CPU/CI或310P3实机PASS。** 必须在新的独占租约下重G0、核HEAD、重编私有SO，再对四候选Smoke/Screen；Stock生产默认与Confirm授权仍不变。

## 从这里恢复上下文（推荐阅读顺序）

1. [项目控制台：当前全项目状态、技术合同和决策](csrc/_310P/memfabric_mm_ar_v2/docs/PROJECT_CONTROL.md)——全项目权威状态。
2. [大M Prefill TTFT实验总账：目标、流程、任务清单、状态、证据和进度日志](csrc/_310P/memfabric_mm_ar_v2/docs/LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)——**本轮大M实验唯一进度追踪入口**；新会话和OpenCode开始A1/L3之前必须读取。
3. [工作分支与合入治理](csrc/_310P/memfabric_mm_ar_v2/docs/BRANCH_GOVERNANCE.md)——唯一主线、实验分支归档、代码来源与禁止事项。
4. [统一性能基线与证据索引](csrc/_310P/memfabric_mm_ar_v2/docs/PERFORMANCE_BASELINES.md)——严格区分单算子 V2 内部收益、完整 V2 vs Stock、模型实际传导。
5. [项目决策记录 ADR](csrc/_310P/memfabric_mm_ar_v2/docs/DECISIONS.md)——ADR-013 废止重建Runner；ADR-015锁定TTFT；ADR-016主线管方向；**ADR-020记录E3审签事实和E4技术门禁**。
6. [正式 V2 融合工程及薄接入](csrc/_310P/memfabric_mm_ar_v2/docs/README.md)——只改 MM+MemFabric 算子、选层与必要的 Stock 回退。
7. [DFlash K7 Stock/V2四象限历史实机准入](csrc/_310P/memfabric_mm_ar_v2/docs/DFLASH_STOCK_V2_R1_GO_GATE_20261025.md)——历史Stock/V2/DFlash取证，不代替当前大M实验总账。
8. [工程准入与310P3实机交接门禁](csrc/_310P/memfabric_mm_ar_v2/docs/ENGINEERING_GATES.md)——区分纯CPU、Kernel、发布级验证；锁定源码/二进制/装载三重身份。
9. [310P 融合算子工程入口](csrc/_310P/memfabric_mm_ar_v2/docs/README.md) 与 [目录内自动执行约束](csrc/_310P/memfabric_mm_ar_v2/AGENTS.md)。

## 不可偏离的当前目标

> **最新权威研发方向（2026-10-27，ADR-015/ADR-016）：主线管方向，工作分支管实施。** 大M以真实Qwen3.6 **Prefill TTFT**为最终指标，L1唯一完整基线是原生FP16 Stock MM+HCCL AllReduce，优化聚焦每Rank实际命中的40个`o_proj/out_proj`目标站点双Rank设备关键路径。阶段顺序为 **L1/L2归因与调优 → L3真实无Profiler Prefill Step可重复稳定正向收益 → L4原始4096/2048/TP2/并发10/50请求客户端TTFT**；L3未达到稳定正收益则继续优化，不启动L4长输出验收。L3设备Step尚缺可审计无Profiler采集与校准，不能以4096/1客户端时延代替。正式方向、[L1～L4历史证据复用及实验合同](csrc/_310P/memfabric_mm_ar_v2/docs/L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)、[ADR](csrc/_310P/memfabric_mm_ar_v2/docs/DECISIONS.md)保存在本主线；[A1 PR #9](https://github.com/10wvw01/vllm-ascend/pull/9) Draft只负责尚未验收的实验实现。ADR-015/016当时的文档调整不构成新实机测试；其后E1/E2/E3已有可追溯的独立设备试验，详见上方2026-10-28状态。Stock默认及G0/安全停止条件不变。

- **Ascend 310P3 单卡双 Die / TP=2**；同 Kernel 完成 FP16 MM + MemFabric SDMA AllReduce + Add，Stock MM+HCCL 是唯一有效的完整算子最终性能对照。
- **大 M 必须持续向硬件可证实的极限优化**，不再每轮讨论“值不值得做”；每次输出完整算子 vs Stock，以及 Qwen 集成真实收益及差距原因。
- **小 M 专攻单 Kernel 带来的固定开销消减**，覆盖普通 Decode 和 DFlash 投机验证的实际 M；旧 T1 M=10 已经严格 NO-GO，不能复活其错误的 +20% 结论。
- **与 Qwen3.6 / DFlash 投机推理集成同步演进**。指定外部只读参照：`HaonanWu1/vllm-ascend:dev_v24/feat-310p-dflash`；**绝不向该仓库提交代码、推送分支或发 PR**；全部工作只落在我们的 `10wvw01/vllm-ascend`。
- 生产/Serving **Stock 默认不变**。任何未完成实机正确性、路由真实性、Graph/生命周期与稳定性验证的融合候选不能启用为默认。

## 项目状态速览

| 事项 | 状态 |
|---|---|
| E3-G完整算子 M4096 | **相对同批E1 +7.187%**，但**仍慢真NZ Stock 96.953µs**；仅L1，M2048整体漂移不签；原始PR #20、主线已审签 |\n| E4分层流水及连续M | [PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)源码+CPU逻辑规划进行中；所有E4设备实验NOT_RUN，生产Stock默认 |\n| D4 Tile2048 正式大 M Kernel | V2 内部相对前版本 +6.26%（不是相对 Stock），保留基线 |
| 4096 输入/2048 输出/TP2/batch10/Eager 模型 | SEL-01：V2 vs Stock 输出速率中位 +1.12%，方向不一致、未达目标 |
| 小 M（M=10） | F1-C：T1 425.2µs vs 原生 Stock 308.2µs，T1 等效速率约 −27.5%；NO-GO |
| F2 Chunk 缩小 | 256、512 均不优于1024；保留 D4 Chunk1024 / Overlap1 |
| MemFabric 长稳与 Graph | 507014 Gate E 尚未根治；Graph 不能仅依据输出正确就宣称真正命中融合 |
| DFlash/Stock 集成 | 保留 Stock/DFlash 基线与 R1-B 历史证据（当时指定组合失败）；**重建 V2 Runner/DFlash 兼容体系正式废止**。融合只做现有模型路径的 LinearMethod 薄接入与 Stock 回退，实际可运行环境和融合命中须分别验证 |
| U1 统一诊断工具 | 已纳入 GitHub 纯CPU源码/合成数据合同回归；**U1真实设备运行仍NOT_RUN**，不是研发主目标 |
| 统一主线 | **本分支**；已有[专用CI](https://github.com/10wvw01/vllm-ascend/actions/runs/40126962427)：治理6/6 + V2源码合同54/54 PASS；正式Kernel未因治理改变 |

## 每次新会话请这样开始

> 阅读 `10wvw01/vllm-ascend` 的 `dev/310p-memfabric-mm-ar:PROJECT_310P_MEMFABRIC.md` 和其链接的 `PROJECT_CONTROL.md`、`LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md`、`DECISIONS.md`、`PERFORMANCE_BASELINES.md`、`BRANCH_GOVERNANCE.md`；核查远程主线 HEAD 和未合入变更。**先报告当前有效结论和下一步动作，再继续我本次的需求。禁止把历史实验报告当作新决策，禁止改动 HaonanWu1 仓库。** 如文档与代码冲突，用源码和新证据定位冲突，并同步更新控制台，不能自行静默改变方向。

## 变更纪律

- 每个有实质意义的工作：在本主线或由其派生的短命任务分支完成，审查后回到本主线；不要把新的实验分支当长期主线。
- 每次完成实机、性能决策、接口变更或外部 DFlash 基线变化，**必须同一批提交更新** `PROJECT_CONTROL.md`，必要时补 ADR/性能报告/分支表。未更新项目控制台的任务视为**未完成归档**。
- 不把文档的“预计收益”“静态通过”“本地测试通过”写成实机 PASS；环境 SO、KVER11 和运行时真正命中的 Kernel 都须有证据。

> **2026-10-26 纠偏硬规则：** `work/310p-v2-r2-implementation-20261025`、PR #7、R2-00～21、T01～T31/46项通用Runner兼容矩阵全部废止，不得从中合入 Runner、DFlash、Hybrid ModelState、GDN/采样/多KV重建代码。唯一目标：保持可运行的Stock+DFlash框架不变，仅替换合格RowParallelLinear的 MM+HCCL 为 V2 MM+MemFabric，真实设备测试按G0/安全门禁执行。历史R1-B失败只属于当时的冻结配置，不得断言其它已可运行Stock环境也失败。

### 2026-10-26｜新增四卡动态择优实机原则（ADR-014）

310P3服务器共有4张卡，实机应优先选**健康且独占的一张双Die卡**；卡级Alarm/占用则记录并换卡重新G0，运行中507014/507899/IB/KFC/错算则STOP保全现场，不自动轮测四卡。换卡需核查`ASCEND_RT_VISIBLE_DEVICES`逻辑/物理映射、双Host SO的SetAccess物理DeviceID修复和KVER11；**KFC由MemFabric在获准首次建池时自动按需部署，G0只读阶段不预部署**。同一Stock/V2 ABBA不得跨卡混合。详见[310P3四卡动态选卡与KFC标准](csrc/_310P/memfabric_mm_ar_v2/docs/310P3_四卡动态选卡与KFC自动部署准入规范_20261026.md)。未进行新实机测试。

## 2026-10-29｜V3.2方案A编码交付，尚未完成服务器准入

最新实现及准入见[V32_ADMISSION](csrc/_310P/memfabric_mm_ar_v3/V32_ADMISSION.md)：
现有阻塞API、单Y pool-return、独立producer、双槽Storage生命周期。
V3.1历史测试通过不覆盖此版新Kernel/TorchNPU Storage；本轮CANN编译/设备/性能NOT_RUN。
MemFabric公共非阻塞收件仅提供[需求原稿](csrc/_310P/memfabric_mm_ar_v3/MEMFABRIC_TRY_WAIT_REQUEST.md)，
由用户提交给负责团队，非方案A依赖。详见控制台§37，Stock默认不变。

## A3最新研发状态：设计审核与代码交付，设备待准入

用户定稿完整S固定席位、独立Y、NZ MM计算优先及连续就绪批量通信。
[A3_DESIGN](csrc/_310P/memfabric_mm_ar_v3/A3_DESIGN.md)与
[A3_ADMISSION](csrc/_310P/memfabric_mm_ar_v3/A3_ADMISSION.md)：profile3302、首READY参数校验、
Direct ADD无Gather、无逐波Host集合/同步、同流事件回收；seed完整计时。
新设备编译/正确性/性能NOT_RUN；A2证据不能替代A3准入。控制台§41，Stock默认不变。

## 2026-10-30｜统一L1框架与VN-M设计文档交付，待设备验证

用户恢复三项工程整理任务。新增[独立Stock/版本对照/统一采集入口](csrc/_310P/tools/L1_test/README.md)，
统一预热、相同N、逐样本双Rank max后算术均值、独占/共享及真实下游残差消费；不恢复算法调优或改Serving默认。
[V1版本索引](csrc/_310P/memfabric_mm_ar/docs/VERSION_INDEX.md)、
[V2版本索引](csrc/_310P/memfabric_mm_ar_v2/docs/VERSION_INDEX.md)、
[V3版本索引](csrc/_310P/memfabric_mm_ar_v3/docs/VERSION_INDEX.md)共35份设计/诊断说明，保留旧名称与来源。
[历史Host阶段数值与瓶颈证据边界](csrc/_310P/memfabric_mm_ar_v2/docs/L1_PERFORMANCE_EVIDENCE.md)。
新框架CANN/NPU为NOT_RUN；历史二进制未匹配则阻止执行，不将注册或CPU通过写成实机GO。

## 2026-10-30｜统一L1框架已纳入主线，后续直接在主线实机

PR #23内容（1415d479）按用户最新指令直接纳入`dev/310p-memfabric-mm-ar`。
后续开发、实机执行中的修复、实验数据和报告直接提交本主线，不再默认另起工作分支或等待PR合入。
实验服务器使用SSH获取主线，核对并固定每轮完整SHA，保留隔离构建/结果目录和原始双Rank证据。
详见[最新治理规则](csrc/_310P/memfabric_mm_ar_v2/docs/BRANCH_GOVERNANCE.md)。
G0→CPU→私有编译/短Smoke→正式性能及故障STOP要求继续执行；新框架设备验证仍NOT_RUN。

## 2026-10-30｜V1/V2规整与封版审核：复用历史，不重跑

[审核与源码锁定包](csrc/_310P/releases/v1-v2-20261030/README.md)已整理37个源码/构建输入及四批历史机器评估。
V1登记的批流水、小M、单波、generation/Graph是当前ABI7能力或模式，六个核心文件与public-api fa1c4217逐blob一致。
V2 D4/E1/E3/E4源码SHA与历史构建manifest匹配；M2048倾向E1，M4096已有E3-G稳定改进证据和E4-T8最快Screen。
最终唯一最优版尚缺V1配置/后续稳定性及E4后续复核原件，不把候选与生产正式封版混为一谈。
[委托OpenCode只读查历史](csrc/_310P/releases/v1-v2-20261030/OPENCODE_HISTORY_LOOKUP.md)；不运行新实验，不改正式构建/Serving默认。


## 2026-10-30｜V1/V2大M按性能选型，范围2048～4096

用户确认范围2048～4096及“谁性能好，选谁”，不重跑旧版。V1选历史成功q4/batch_m1024，
仅余成功记录与准确源码/二进制映射，不再要求最新全M或Decode/模型长稳修复作为本次前置。
V2 M2048选E1、M4096选E4-T8，源码及实测点选型已封存；不因D4现有正式身份或E3-G复核更多而选慢版。
14份原始Rank JSON复算完整Wall均值为1394.627/2175.690µs；T8/T4 M4096均值近乎持平，保留Screen标签。
[性能选择与源码封存](csrc/_310P/releases/v1-v2-20261030/README.md)、
[仅查V1成功身份的OpenCode任务](csrc/_310P/releases/v1-v2-20261030/OPENCODE_HISTORY_LOOKUP.md)、ADR-038为准。
未测区间点不冒称最优，不新拼路由、不改Serving默认；本轮仅离线复算与CPU/源码验证。
