# 310P3 MM+MemFabric AR｜项目正式决策记录（ADR）

> 这是**追加式**项目决策账本：保留决策的前因后果，不因后续数据变化静默重写当时的结论。当前有效优先级与状态以 [PROJECT_CONTROL.md](PROJECT_CONTROL.md) 为准。时间均为Asia/Taipei日期。

## ADR-001｜范围与原则｜有效
- **范围**：Ascend 310P3单卡双Die TP=2，目标Qwen3.6已确认未量化的RowParallel o_proj/out_proj；同Kernel FP16 MM、MemFabric SDMA AllReduce与本地Add。
- **唯一完整算子基线**：原生FP16 Stock MM+HCCL AR；Stock继续产品默认。MM-only、AR-only或旧V2只能用于诊断或V2内部改进。
- **约束**：8核尽可能参与MM，合法同步并保证数据可见性；固定Core0负责signal的设计可以在充分证明后修改，但不允许通过删cache clean/credit/wait来获得不安全加速。
- **禁区**：不要把研发扩展成MoE/GDN/Attention一般性能优化；它们属于Qwen背景，不是本期融合算子工作。

## ADR-002｜大M继续攻坚｜2026-10-25｜有效
- Tile2048 / Chunk1024 / Overlap1作为已证实的V2候选基线；C3 unroll4不合入。
- **不再讨论大M优化是否有价值**，目标是用可复核的硬件关键路径分析持续改善到当前物理限制，重点优化正式MM效率、input.contiguous/内存布局、SDMA与MM更充分交叠、同步/规约关键路径。
- 每轮必须直接报告当前V2**完整算子 vs Stock**的差距及模型真实性能传导原因，D4内部+6.26%不能当作Stock收益。
- 源码版本优化与真实性能验证分开记录，不因单轮微改善小于3%停止整条大M研发线。

## ADR-003｜小M单Kernel固定成本｜2026-10-25｜有效
- F1-C M10旧T1路径原生FP16 Stock约308.2µs vs T1约425.2µs，旧T1 **NO-GO**；此前不公平Stock含FP32/Host初筛“+20%”作废。
- 不推断所有单Kernel小M方案均失败，目标是减少Stock分算子链路的固定开销，覆盖普通Decode与DFlash实际Verify M。需要独立专用MM+MemFabric AR单波路径设计和公平全链路对比。

## ADR-004｜DFlash同步集成、只改我方仓库｜2026-10-25｜有效
- 指定外部只读架构参照：`HaonanWu1/vllm-ascend:dev_v24/feat-310p-dflash`（检查时HEAD `1989b04c04e6…`），当前自己的DFlash核心文件已基本对齐，优先保留而非重写Model Runner。
- 仅在 `10wvw01/vllm-ascend` 自己的项目主线开发融合Kernel/LinearMethod/Graph/lifecycle薄适配；**禁止给外部主开发仓推代码/PR**。
- 投机验证logical M=active requests×验证宽度只是示例；需审计physical padding M与真实Graph命中，Graph的伪Stock路径不可算融合成功。
- 普通Eager input4096/output2048/TP2/batch10保留历史验收，投机推理另设同模式Stock对照，不能混算。

## ADR-005｜性能与稳定性采用分级门槛｜有效
- 性能研究可以在控制范围先筛选、再重点对照；但每一条被上机的路径必须先通过G0版本/设备独占/Host+Device+KVER、CPU合同与正确性冒烟。
- P0 Gate E的507014、MTE3首写静默损坏、Graph路由固化Stock仍为发布阻塞；**不能以性能优先为由忽略数值错误、数据面异常或未经验证的绕过**。无阻断前可进行隔离的性能研究，正式发布前集中关问题。
- 真实Stock vsV2用完整算子、双Rank配对、ABBA、路由真实性和尾部，任何失败样本保全。

## ADR-006｜唯一项目主线与GitHub上下文落盘｜2026-10-25｜有效
- 建立 `dev/310p-memfabric-mm-ar` 作为此项目**唯一长期研发控制主线**；仓库默认 `main` 不变，外部仓库完全只读。
- 主线取自已有U1资产分支，保留D4 Kernel和F1-C/U1工具；U1未执行CPU/NPU事实严格保持UNKNOWN；创建主线不启用融合默认路由。
- 新任务只从主线派生短期工作分支，经过代码/结果审查后回合主线；历史33条分支按用途冻结、暂不删除。
- 新会话入口为仓库根 `PROJECT_310P_MEMFABRIC.md`；**任何完成任务必须更新`PROJECT_CONTROL.md`**，涉及技术决策再追加本ADR，性能则同步 `PERFORMANCE_BASELINES.md`；ChatGPT记忆、聊天记录与旧实验README不能取代GitHub权威状态。
- 历史诊断/性能分支不能未审核整体合入，否则旧 C3/T1/C4 的代码和判断可能污染项目路线。

## ADR-007｜工程交付门禁自动化与实机身份核对｜2026-10-25｜有效
- 决策：将 `dev/310p-memfabric-mm-ar` 的上下文落盘从文字约定扩展成纯CPU自动变更检查；正式源码/测试/证据更新同步 `PROJECT_CONTROL.md`，性能证据同步 `PERFORMANCE_BASELINES.md`。
- 原因：过往阶段分支分散，消息结论和真实代码/装载二进制混用；仅靠新对话提示词无法强制执行。
- 取舍：不将这一CI当作CANN/NPU正确性或发布验收；不在治理提交中修改 Kernel/Host/Serving；不改变GitHub默认 `main`，仓库保护由所有者另行确认。
- 落地：`.github/workflows/310p_mm_ar_project_gate.yml`、`tools/310p_mm_ar/project_gate.py`、`ENGINEERING_GATES.md`；实际运行结果与上机G0另行取证。
- 安全：MemFabric源码SHA、Host/Device安装SO SHA256、进程实际装载地址及KVER应分别核对；未获用户授权不自动切换服务器worktree或替换运行库。

## ADR-008｜DFlash K7的Stock/V2受控实机四象限｜2026-10-25｜有效
- 本次独立模型组：Stock+DFlash Eager和V2+DFlash Eager；都锁定目标Qwen3.6-35B-A3B-w8a8、草稿Qwen3.6-35B-A3B-DFlash、K7、TP2/单卡双Die，按上游TP2验收惯例指定draft TP=2，不照搬K15。
- 必须G0→Stock Eager smoke→V2 Eager smoke→同口径Stock/V2 4096输入/2048输出/batch10/50请求；Graph仅在Eager证据人工审阅后分别做Stock和V2短烟测，V2 Graph不当成已支持或已通过。
- 正式V2只在2048～4096融合，Verify/Decode大多仍为Stock，禁止将开启V2称为解码融合成功；任何Graph收益必须有原生capture/replay+双Rank设备融合调用证据。
- 新脚本仅增加测试/诊断与协议文档，Kernel/Host/Serving和上游DFlash源码保持不变。所有执行受G0 JSON、设备独占、恢复审批和Graph单独授权约束。
- 上游310P Docker主动卸载Triton与Triton-Ascend，pip check仅对明确的两类“未安装”缺包提供窄例外，不容忍其余异常。
- 详见 [R1 DFlash K7 准入协议](DFLASH_STOCK_V2_R1_GO_GATE_20261025.md)。G0现场、Eager和Graph目前均NOT_RUN；发布Gate E及首写风险仍未关闭。

## ADR-009｜R1-G0取证整合与R1-B环境基线｜2026-10-25｜有效

- 触发：旧R1 `work/310p-mm-ar-r1-g0-nogo-20261025@4505650` 实机原始G0 NO-GO，Stock/V2+DFlash全组NOT_RUN；报告与原始JSON已归档。
- 决策：以已按用户命令修订的主线 `592e708` 为基底，选择性移植vLLM 0.27.1/torch-npu post4/transformers 5.17的真实服务器G0取证及模拟真正Serve导入顺序的探针；不得以旧R1分支全量覆盖新版Runner。
- 决策：已知`pip check`告警仅在明确声明者+缺包及版本冲突完全一致时豁免并记录，任何新增/未知/空失败输出需NO-GO。为模型实机执行增加导入探针、模型路径、相同Python、editable worktree和V2 SO存在性门禁，不以静态结果充当模型PASS。
- 参数：Target和Draft都以310P服务器核验路径 `/home/models/Qwen/` 为准，K7、TP2；保持用户host0.0.0.0（先网络隔离）、max-num-seqs16、GPU0.90和固定Bench4096/2048/50/10。Eager必须`cudagraph_mode=NONE`，Graph独立研究。
- 不更新通用`requirements.txt`的过时引脚来强制所有架构升级，也不执行设备安装/改SO；另立打包元数据债务，待确认部署矩阵再改。
- 只有现场重G0、Stock/V2 Eager及双Rank真实Kernel证据分别通过，才允许后续Graph诊断；对外部HaonanWu1仓库保持只读。


## ADR-010｜R1-B现场NO-GO证据主线补档｜2026-10-25｜有效（原始结论不变）

- 证据来源：`work/310p-r1b-stock-eager-smoke-nogo-20261025@f51aa689`。仅引入 `R1B_STOCK_EAGER_SMOKE_NOGO_REPORT_20261025.md` 与相应 `evidence/r1b_20261025/` 原始blob，保留完整server.log.gz；不引入该旧分支的控制台或工程代码，避免覆盖PR#5及R2计划。
- 实机裁决：G0 PASS、Stock+DFlash Eager FAIL（原模型混合sliding/full attention被V1门禁拒绝）；V2、full、Graph NOT_RUN；未出现507014/507899，设备现场恢复；没有任何新的融合性能。
- ADR-011/012后续决策是研发V2方案，不得回写改称R1-B执行通过。

## ADR-011｜310P3 V2 Triton验证条件豁免｜2026-10-25｜**已由ADR-013废止**

- 决策：按所有者指示优先走310P3 V2 Model Runner，停止V1多KV组扩展；在自有仓库实现**310P3+vLLM0.27.1+TP2+Eager+显式双开关**条件下仅豁免上游V2 Triton存在性检测，不修改vLLM源码、全局HAS_TRITON、非Triton V2其它校验、现有Stock生产默认。
- 触发：R1-B现场G0 PASS、Stock+DFlash混合Draft门禁FAIL和`VLLM_USE_V2_MODEL_RUNNER=1`时`Model Runner V2 requires Triton`证据；并非设备推理已成功。
- 安全：本轮Worker仍是V1派生，V2选择必须在触碰设备前显式`NOT_IMPLEMENTED`，防止V1伪命中；**CONFIG-GATE-ONLY**。后续按V2 Worker/Triton替代/API多KV组分阶段推进、静审后上机。Graph和MemFabric仍独立门禁。
- 验证：CPU-only参数作用域、旧逻辑回退、版本漂移、Graph、TP、非DFlash Spec限制及fail-closed行为单测，CI报告不等于NPU PASS。


## ADR-012｜R2 V2 Runner全量范围与分段验收锁定｜2026-10-25｜**已由ADR-013废止**

- 决策：以 R2总执行计划（**已从当前主线删除**；仅可在废止前历史commit `a37c0212f422`追溯，不是执行入口） 的 R2-00～R2-21 + R3-01～R3-04 作为下一阶段完整任务锁定，范围是 310P3 无Triton V2 Model Runner Eager（Target Stock→DFlash Stock→MM+MemFabric融合）；先CPU/两轮静审再一轮分段实机；Graph、Gate E 与大/小M研究单独列账。
- 边界：只允许 our `10wvw01/vllm-ascend` 增量，继续保留0.27.1环境版本、模型、K7、TP2、Stock默认、原主线和不替换未授权SO；V2校验豁免目前仍CONFIG-GATE-ONLY。绝不改V1多KV、模型layer_types或伪造Triton。
- 验收：每一组由可追溯代码SHA/装载二进制/HCCL/两个Rank/Token正确性确认；L1完整算子与L2同V2 Runner的Stock/融合配对，S0 V1 Runner不能参与融合收益归因；Graph独立门槛。
- 变更：发现新必经API/Kernel必须补计划与UT/静审、风险评估；扩大为核心Scheduler大改或不具备修复路径时，须经新ADR与所有者裁决，不可静默扩大。

## ADR-013｜2026-10-26｜重大纠偏：剔除重建 V2 Runner / DFlash 兼容体系｜**有效**

- **决策**：立即废止ADR-011/ADR-012及R2-00～R2-21通用V2 Runner、Triton、Hybrid ModelState/GDN/多KV/采样/Device Copy系统的研发与合入；PR #7 不采纳，工作分支不再作为开发/集成基础。历史取证与日志继续留档，禁止误作未来实施计划。
- **原因**：原始目标仅是使正式`V2 MM+MemFabric AllReduce`通过模型已有`RowParallelLinear`在Stock+DFlash链路中选择性替换原生`MM+HCCL`，不需要重建DFlash或其ModelState。扩大工作违反ADR-001/ADR-004、拖慢单算子性能主线。
- **落地**：从当前唯一主线剔除R2 Runner开发计划与Triton验证豁免；恢复既有env-only Stock/V1选择；**保留V2未实现时的Worker提前拒绝**。正式融合Kernel、Host ABI、已有LinearMethod、Stock默认与历史性能账本均不动。
- **后续合法范围**：大M算子优化到可证硬件瓶颈；小M同kernel固定开销削减；融合算子与现有Stock+DFlash的最小选层/生命周期/数值/Graph共存适配。先取证用户提供的可运行Stock+DFlash实际源码和部署，保证同一Runner/同一模型/TP2/K7做Stock/V2 A/B。
- **历史证据不重写**：R1-B指定旧组合中Draft构造失败为历史真实NO-GO，不能据此反证用户当前可能已运行的Stock+DFlash配置。**本ADR不代表有新设备运行或融合性能结果**。
- **禁令**：不得复活R2工作分支、PR #7、46项Runner泛化准入、Device-Native GDN Copy Consumer等旧任务；未来任何框架级需求需另行由用户明确批准，不可默认为融合算子依赖。

## 后续决策的写入模板

`ADR-014｜YYYY-MM-DD｜有效/被替代`

- 决策：
- 触发证据（SHA、报告、原始JSON或现场路径）：
- 取舍与未采纳的方案：
- 代码和当前默认路由影响：
- 验证门槛、回退方案和需要更新的控制台章节：

## ADR-014｜2026-10-26｜四卡动态选卡、故障换卡及KFC自动部署｜**有效**

- **决策**：服务器有4张310P3物理卡，项目单卡双Die TP2实验必须每次动态选择健康空闲同卡双Die；G0健康/占用失败时允许换另一卡重新G0，而非卡号绑定停工；四卡全不合格则NO-GO。同一Stock/V2 ABBA不得跨卡混算。
- **MemFabric条件**：换卡不能只改`ASCEND_RT_VISIBLE_DEVICES`；历史实机验证CANN VMM`aclrtMallocPhysical`逻辑号、`aclrtMemSetAccess`物理号分裂，非identity映射要审查Host双路径的物理号修复版本。AICPU KFC v11在**授权首次初始化**按需自动部署、核验每Die，不预先人工部署四卡，不使用伪`MF_SDMA_ORCH_DEPLOY_HELPER`或诊断常驻模式。
- **异常分级**：G0单卡Alarm或租户占用→排除并换卡；运行中507014/507899/错算/超时/IB或KFC异常→先STOP保全现场，排查系统性问题，后续换卡必须重新取得人工准入，禁止自动轮试四张卡。
- **依据/合同**：[四卡动态选卡及KFC规范](310P3_四卡动态选卡与KFC自动部署准入规范_20261026.md)与[2026-10-23 KFC/DeviceID实机记录](MEMFABRIC_310P_KFC_DEPLOY_AND_DEVICE_PIVOT_20261023.md)。本项不变更正式Kernel/Host/Serving/Stock默认，也不构成新设备测量/部署的PASS。

## ADR-015｜2026-10-27｜大M最终目标锁定真实Qwen3.6 Prefill TTFT与40站点关键路径｜**有效**

- **用户裁决**：大M最终性能指标是**真实Qwen3.6 Prefill对应的客户端首Token TTFT下降**；不再以独立Kernel微基准加速或整模型输出tok/s/TPOT作为大M最终成功标准。持续调优至可证实的硬件/协议瓶颈原则（ADR-002）仍有效，单算子优化作为实现路径而非最终交付。
- **L1唯一基线**：TP2/310P3单卡双Die，原生FP16 `F.linear + HCCL AllReduce`完整链，与正式V2 MM+MemFabric AR（包含必要布局/拷贝/同步）公平同卡热态对照；实际HCCL异步跨Stream尾部、双Rank耗时/bit-exact、SO/KVER身份和异常均需留证。D4内部Tile2048 +6.26%不能冒充Stock优势，最新D4 M2048/4096相对Stock的L1仍UNKNOWN。
- **唯一重点优化对象**：已选择的Qwen3.6 attention/GDN 40个 `o_proj/out_proj` 目标层（每Rank）在**真实Prefill设备关键路径**上的可回收时间，包括真实M和路由命中、权重/输入布局、`x.contiguous()`、SDMA/IB/DONE、双Rank同步和下游消费时刻；不转为MoE/GDN/Attention模型本体普遍优化。40站点不是对所有请求实际命中的无条件保证。
- **验收四环**：① L1完整Stock vs V2单算子；② 真实Serving目标站点净耗时；③ Prefill步的双Rank关键路径净变化；④ 客户端TTFT含排队、分块/混合调度、首Token返回的逐请求p50/p90/p99。同场同模式证据说明为什么算子µs不能全传到TTFT；不以各层p50简单累加、TTFT变化大于/小于推算或当前旧Profiler 3.3%占比直接作GO。
- **正式业务合同不变**：Qwen3.6-35B-A3B-w8a8普通Eager，input4096/output2048/TP2/并发10/50请求/ignore-eos；独立input4096/output1/并发1热态Eager仅用于隔离Prefill归因，不能代替正式TTFT验收。投机DFlash/Graph保持单独同模式账本及既有薄适配边界（ADR-013），不受本次焦点收敛而废止。
- **取舍、安全及状态**：只更新项目文档、测量和后续验收目标，不更改正式Kernel/Serving、Stock默认与MemFabric协议；发布级Gate E/507014及冷首写/bit-exact仍为硬门禁。A1 PR#9目前仅待原仓CPU合同、四卡只读G0；Smoke/Screen/Confirm/Profiler与新增E2E TTFT **全部NOT_RUN**。后续分阶段人工审批，出现设备异常STOP保全现场；不能由本决策自动授权实机运行。同步更新`PROJECT_CONTROL.md`、`PERFORMANCE_BASELINES.md`、A1工作方案及仓库入口，主线仅经评审合入。

**ADR-015执行策略补充｜2026-10-27｜L3-first前置性能门槛（与原目标一致）**

- 大 M 每一代新候选按真实Qwen3.6**完整Prefill Step双Rank关键路径**的可重复改善决定是否进入客户端正式L4；L1完整Stock仍是算子诊断基础，L2真实40站点拷贝/形状/同步归因为优化对象。
- L3应在冻结输入4096/output1、并发1、TP2、Eager且无Profiler的真实模型路径上，依靠**具有设备完成语义**、无每次Device Sync引入额外税的Prefill Step计时；少量Profiler用于独立核执行/依赖时序交叉校验，不可将带Profiler时延或单请求客户端TTFT冒充纯L3。现有工具尚未被证实具有无Profiler Prefill设备Step计时能力，未静态/实测验证前一律标`L3_MEASUREMENT_NOT_READY`。
- L3先轻量Stock/V2 Screen，方向正向且现场准入后再做无Profiler跨Serve的Stock–V2–V2–Stock重复Confirm，按预先冻结的Stock漂移/测量不确定性门槛判断`L3_GO`；未获稳定正向时`L3_NO_GAIN/INCONCLUSIVE`返回L1/L2继续优化，**不启动正式4096/2048/50/10 L4**；不能按不适当的5% L3门槛阻断真实小幅正收益。L3即使通过也不保证混合调度L4正收益，L4独立验证正式TTFT。
- L3测量研发不构成通用Runner/DFlash状态机改造许可；仅允许测试路径可审计的低侵入采样，Stock产品默认、G0/首写/507014/异常STOP及每阶段人工许可不变。对应[历史复用与L3-first专项方案](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)。本修订没有产生新310P3测量。

## ADR-016｜2026-10-27｜主线管方向、工作分支管实施｜**有效**

- **正式裁决**：`dev/310p-memfabric-mm-ar` 是唯一项目权威控制主线。稳定的研发目标、ADR、工作负载与性能验收合同、历史证据归因、当前实验顺序/准入、安全门禁及真实状态**必须优先落盘主线**。新会话与OpenCode一律先读取主线入口，不以未合入工作分支替代项目方向。
- **实施隔离**：A1现有工作分支`work/310p-mm-ar-large-m-a1-20261026`（Draft PR #9）只保留未审完的诊断/测试代码、实机执行准备、后续具体Kernel候选及CPU/NPU待验内容；不得把该分支整体merge进主线。今后工作分支不承担唯一最新版ADR或长效项目状态。
- **选择性合入策略**：本次只审查、同步ADR-015和L3-first历史与正式实验合同、项目入口、性能基线与分支治理。现有正式Kernel/Host/Serving、Stock默认、上游DFlash和MemFabric协议**零改动**；不复制未验证的A1测试脚本。主线文档合入后，A1 PR #9需按更新的主线核对差异，后续只合入审查合格的实施资产，禁止回流旧版治理文件覆盖主线。
- **闭环规定**：任一实机报告经过证据审核后，在同一批主线变更中更新`PROJECT_CONTROL.md`（状态）、`PERFORMANCE_BASELINES.md`（事实和适用工况），产生新决策则追加ADR；CI/静态审查与实机准入分开标识，不以PR建立或文档合并暗示性能/安全PASS。
- **关联规范**：[主线分工治理](BRANCH_GOVERNANCE.md)、[ADR-015 L1～L4历史归因与L3-first合同](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)。此决议不改变ADR-013与ADR-014，实机G0/Smoke/Screen/Confirm/L3/L4均维持原独立授权和异常STOP要求。


## ADR-020｜2026-10-28｜E3证据归主线与E4分层流水/连续M实验授权｜**有效；E4设备NOT_RUN**

- **治理决定**：E3工作分支[PR #20](https://github.com/10wvw01/vllm-ascend/pull/20)的`run_id=e3-chunk2048-20261028-001`已经完成G0、修复后BiSheng、Smoke、八场原始双RankL1对照；本决策**选择性**将实验事实/归因边界/当前下阶段策略写入唯一主线，不合并E3正式Kernel/Host、原始Profiler大型文件或过期分支管控文档。主线[证据审核与E4合同](E3_实机证据审核与E4分层流水准入_20261028.md)成为正式索引，E3来源PR仍Draft只读证据。
- **可信E3结果**：M4096真实NZStock2022.553µs，E1 2271.836µs，E3-G2119.506µs，E3-T2267.090µs；E3-G对E1 **+7.187%（净少152.330µs）**，仍慢Stock **96.953µs**。M4096所有臂漂移≤2.02%；M2048同批Stock1078.412/E1 1222.611/E3-G1451.002/E3-T1483.932µs但E1跨场漂移6.192%>5%，**整批总体INCONCLUSIVE_DRIFT**，不可硬签M2048的精确速比；E3-G负向方向稳健。E3-T首版L0C满载首个真实MM引发507015，改`basicM 256→128`并重新双Rank验证PASS，不等于507014根因已闭环。
- **归因裁决**：E3-G相对E1保持4次1024 MM子调用、4条4MiB数据消息和Add总元素数不变；每4096行producer/consumer缓存维护、ReadyTree及Add release会合从4组→2组，兼有流水与每Core输出布局变化。152.330µs是**净整体效果**，各内部阶段独立耗时未知，不允许归到某一单点。E2-R/READY预失效/512和E3-T继续冻结负收益证据。
- **E4范围**：只新增一个隔离工作分支`work/310p-mm-ar-e4-layered-pipeline-20261028`，基于E3已验证代码；优先 E4-0 AddTile2048/4096/8192单变量、E4-A Packet0/1分半组Add、E4-B连续M的可审计规划器/混合2048+1024与尾块。不能提前发未完成包或把新增IB/缓存费用藏在流水里。真实Serving路由维持Stock，全部E4硬件/性能`NOT_RUN`。
- **必要审查**：一切E4设备内同核通信仍遵守Core0唯一mail消费者、每包dst/len/imm、DataCache生产/接收可见性、最终8核AddDone、quiet+DONE和首写、双Rank安全。原生真NZ Stock完整链是唯一性能基线，M区间连续测试在合法Padding及真实额外copy开销下逐点衡量。历史Qwen L2/L3/L4在E3/E4下均NOT_RUN。


**ADR-020落实追加记录（2026-10-28，主线原始证据迁移）**：已将[E3主线证据52份归档清单](../evidence/e3_chunk2048_20261028/MAINLINE_ARCHIVE_MANIFEST.md)、[E3机器评估](../evidence/e3_chunk2048_20261028/e3_assessment.json)、双Rank原始采样、真NZ路由、G0及原报告全部原样保存在`dev/310p-memfabric-mm-ar`；每份源blob SHA与归档blob一致。此后工作分支PR#20不再是关键机器数据唯一备份。**不因证据迁移授予任何E3实验Kernel/Host的生产合入资格**，Stock默认不变；大型Profiler及原未上传的PLog/build日志仍留待补证。


## ADR-021｜2026-10-28｜E4 Attempt1被外部占卡阻断，连续独占和主动STOP守护升为P0｜**有效**

- **机测事实**：run_id `e4-layered-20261028-001`，执行HEAD `1e20d3522cc9dd8040b257fae8d887e7d12f3b5e`，归档HEAD `5b727e9f9153e8292c6022a14b63e974dfb0db03`。G0、原仓80+E4合同9条UT、2049M CPU规划、CANN9.1 BiSheng四实例隔离编译、原生NZ四组路由、E3-G双RankSmoke均PASS。但E4-T2048启动即被执行工具SIGTERM打断无Kernel执行；外部宿主PID506263随后被查证在四卡持续占用，原始首份06:29:39Z巡检已有该PID。**全部E4新候选设备Smoke和短Screen NOT_RUN**；十四场Confirm从未授权。不能给出E4与Stock/E3-G的任何测速结论。
- **治理故障**：`e4_campaign.sh`的30s巡检仅记录而不作即时安全决策；G0单次独占不保证整轮持续独占。外部进程出现后E3-G Smoke发生同下标Rank同时2.3ms/6.5～6.8ms交替，属于被占卡污染的正确性取证，不可用于L1对照。原工具进程组SIGTERM也是独立执行工艺问题，不得解释成E4-T Kernel失败。
- **强制P0**：下一轮继续沿用当前E4隔离分支/PR #21，不另开E5或重跑上一轮CPU验收；先获取真实连续单卡双Die独占，再完成具有**阻断新阶段**能力的宿主级只读外部占用检测（不能仅打印30s日志），与独立session启动+进程完成状态轮询的终止保证，补CPU模拟/负面测试和两轮静审。发生异常全批STOP保全，无自动重试、换卡、杀租户PID或清KFC；每次重用构建须核实执行HEAD与manifest以及SO SHA完全一致，不兼容则重建。任何性能Screen都必须G0/运行中独占、原生NZ、全部E4 Smoke PASS。
- **性能合同不变**：E3-G已验证M4096 2119.506µs vs真正Stock2022.553µs，仍差96.953µs；`INNER_PHASE_UNKNOWN`。E4-T2/T4/T8/PacketAdd增益`UNKNOWN`，E4-B仍仅纯CPU连续M规划。Stock Serving默认不变、E4 L2/L3/L4 NOT_RUN、十四场Confirm需单独授权。
- **权威原件**：[本主线E4 Attempt1 23份逐文件同SHA归档](../evidence/e4_layered_20261028/MAINLINE_ATTEMPT1_ARCHIVE.md)，[原中文报告](../evidence/e4_layered_20261028/A1_E4_第一轮实机验证报告_20261028.md)，[STOP根因取证](../evidence/e4_layered_20261028/stop_incident/STOP_INCIDENT_ANALYSIS.md)，[E4 PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)（实施非主线）。


**ADR-021执行补充｜2026-10-28｜最小保护工具代码就绪，尚待实际CPU/设备验收**：原E4分支[PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)已补`e4_stage_guard.py`（每次torchrun前/中/后核目标双Die健康和可观测跨卡新PID，异常自动STOP并保留原始npu-smi证据，只终止本run进程组），`e4_detached.py`（独立Session启动/status/退出码及Screen-only限权），新增CPU负面测试与CI；`e4_campaign.sh`四处设备入口均接入守护并强制当前HEAD/Build绑定。**测试代码仍NOT_RUN，工具没有取得真实持续独占租约也不能保证测试环境安全**；新HEAD必须先CPU合同和G0，通过后重编私有SO，方可进入E4四候选设备Smoke，七臂短Screen仍非正式性能Confirm。



## ADR-022｜2026-10-29｜V3用SDMA ADD求和；本轮只测V3并复用历史对照

- 用户批准V3独立开发：MM写本地部分结果，READY之后以ADD_FP16执行owner分片Reduce，COPY All-Gather，保留独立输出与全部同步/首写保护。
- 不允许全量双端原地ADD互改发送源；标签改为40位并拒绝epoch溢出；ADD不允许不确定重试。
- MemFabric固定`beaf490b`/KVER28，V3与旧KVER11头/库不混用；当前源码占位自拷贝差异记录为长稳未闭环项。
- 用户明确要求快速判断算子相对历史V2/真实NZ Stock的收益方向：默认只跑V3，旧臂不重测；
  允许生成带来源和历史原判定的跨批差值，但不签同批正式性能GO，不自动补测。
- 新工程[README](../../memfabric_mm_ar_v3/README.md)、[两轮审查](../../memfabric_mm_ar_v3/REVIEW.md)及一键短测入口统一归本项目主线管理。
- 代码/CPU完成不等于设备PASS；私有BiSheng编译和G0后才能执行ADD→协议→融合→短测。Stock默认及原模型准入边界不变。


## ADR-023｜2026-10-29｜上游同步与V3两仓私有部署执行合同

- 权威GitCode来源`ba3e7f42873b6f2a4e8d118c03a14d1f43d39ab4`，镜像`b9d46f97955b11a8b7a892e9fd8d516e66346a65`，完整tree相同；V3改锁后者。
- 采用GitHub插件进行源码树快照同步，保留原GitHub历史及上游来源，不宣称commit身份相同。
- OpenCode只执行[固定执行单](../../memfabric_mm_ar_v3/OPENCODE_EXECUTION.md)，不承担协议/源码修正或依赖选择。
- 只更新我方MemFabric及V3两个新隔离源码树；不改上游/外部HaonanWu1/运行中的框架和Serving。
- private SDK显式包含上游打包器漏掉的SDMA AICPU三件套，之后编V3私有SO；不执行共享安装器、不预部署四卡。
- 默认仍为V3-only快速历史对照，无旧性能臂、模型或Confirm；服务器编译/设备验证仍NOT_RUN。

## ADR-024｜2026-10-29｜V3首轮性能失败，按V2颗粒度重排三段流水

- 依据：V3`6144af16`归档的R3原始双Rank采样，最佳group2048仍较历史V2/E3-G
  慢17.29%/35.11%（M2048/4096）；正确性通过不构成性能达标。
- 决策：保留SDMA ADD路线，重新设计MM/Reduce/Gather排布；固定chunk512/1024/2048，
  沿用对应V2 MM tiling，区分chunk与packet，禁止把只重命名pipeline当作已有交叠。
- 首要改动：当前Reduce提交与上一组Gather/输出解耦，合法交错邮件统一分发；
  验证owner半片段单包及双UB MTE，逐项替代同步依赖；保持精确一次ADD、缓存可见性、
  首写、输出生命周期及最终quiet/DONE。不改变独立ND输出语义来制造收益。
- 新设计与最小验证合同：[V3.1复核](../../memfabric_mm_ar_v3/PERFORMANCE_REVIEW_20261029.md)。
  先同V3内部最小诊断/逐项消融，再完整算子六配置两短轮；不重跑V2/Stock、不跑模型/TTFT。
- 职责按用户最新规定：OpenCode负责执行、执行BUG修复、结果汇总、证据采集和远程提交；
  研发负责方案/协议、实现审查和后续规划。覆盖ADR-023旧“只执行不修复”描述。
- 本次为文档/设计裁决，V3.1未实现、CPU/CANN/设备均NOT_RUN，Stock默认不变。

## ADR-025｜V3.1实现与有条件设备准入

实现三段推进、合法交错邮件分发、双UB copy与4MiB包，chunk固定512/1024/2048。
CPU88项PASS（含实际C++调度器480场异步模拟及负例），设备CANN编译/NPU均NOT_RUN。
没有设备编译证据前不宣称实机准入已通过；服务器先编译再按G0和门禁执行。
完整条件与交接见[准入单](../../memfabric_mm_ar_v3/V31_ADMISSION.md)。Stock默认不变，不重跑历史V2/Stock性能臂。

## ADR-026｜2026-10-29｜V3.2排布从最终输出与源消费依赖反推

采用[单Y输出槽与独立producer设计](../../memfabric_mm_ar_v3/V32_LAYOUT_SCHEDULE.md)。
TP2的ADD完成→Gather天然保护被覆盖源，否决不必要的额外半份S；
主改动是池内输出Storage持有、去最终copy、按整行带推进及公共非阻塞收件。
不假设310P支持异步WaitIterateAll/GetTensorC，不照搬GPU warp/TMA。
保持chunk512/1024/2048、八核MM、完整ND输出及既有数值标准。
用户要求不新增独立归因设备轮，直接实现后验证完整链；新代码和性能均未验证。

**2026-10-29接口约束追加**：MemFabric由其他团队维护。V3.2主方案A仅使用现有signal/wait/quiet，
采用固定R/D/G发送与阻塞接收顺序，保留单Y输出及独立producer；1024场CPU协议交错排空通过。
公共try_wait及逐片段自驱改为备选B，等待负责团队提供，不作为本轮交付依赖。
详见[V3.2排布推导](../../memfabric_mm_ar_v3/V32_LAYOUT_SCHEDULE.md)，两套协议不允许会话内混用，设备性能尚UNKNOWN。

## ADR-027｜V3.2主方案A编码，公共try_wait仅提交需求

保留现有MemFabric版本及signal/wait/quiet。独立producer采用每group/core独占缓存行，
避免将IB事件错误视为可累加通知。Core0按固定FIFO收发，取消工作核逐chunk放行等待。
双槽Y由Storage持有，下一次acquire在双方流完成后协商空闲交集；完整计时必须包含该成本。
协议模式在create协商，禁止混用31/32；generation每次只允许run一次。
新增TorchNPU格式描述与Storage、别名/单边释放/槽耗尽/下游ND消费设备门禁。
方案B接口[需求](../../memfabric_mm_ar_v3/MEMFABRIC_TRY_WAIT_REQUEST.md)供负责团队审核，未代发，
不修改对方库或以短超时伪造非阻塞。设备准入仍待新CANN与Storage验证，性能UNKNOWN。

## ADR-028｜A复核缺陷修复：准备、关闭一致性与引用回收

采用双方Storage-ready prepare后才启动数据波、双方无输出租约后才close，统一Host控制信封，
Host版本3202与旧32不兼容；MemFabric版本和设备固定FIFO顺序不变。
任何不确定异常封禁，不发新设备操作或重试ADD；Host控制和prepare开销必须完整计入性能。
上一波流完成后回收输入，新波输入保持有效。B不依赖EMPTY判断完成，仍以完整片段状态和排空作验收。
本轮CPU反例回归不替代CANN、Storage或设备准入，不增加独立设备归因测试。

## ADR-029｜A3定稿：固定S席位、NZ计算与连续通信批量解耦

TP2借鉴HCCL DoublingDirect：S全波不变，Y先seed再接对端ADD，取消Gather而非减少跨Die字节数。
保留现有64/128行NZ权重MM候选，compute与comm独立；完整成本包含seed，不假设免费双写。
热路径本地固定槽、设备START，同流与最终quiet保护跨调用S复用，不逐chunk等待源回收。
profile3301禁止旧协议混用；输入事件query回收；跨流/Graph不在本版范围。
终止性资源失败必须通知peer START ready=0，不默默覆盖；公开wait仍可能等待对端READY，
不得将A3描述成方案B的非阻塞自驱调度。设备准入仍待服务器编译和完整门禁。

## ADR-030｜A3首READY合并帧校验

独立START用于提前失败但非数学依赖；合并到首READY，在满足首范围S/Y就绪时同时确认元数据。
正常MM无需先等peer，首READY严格校验前不得提交任何ADD。失败端不覆盖S/Y，错误通知先送出再Trap。
取消初始fan-out和独立ACL状态清零任务；保留数据完成及输出生命周期所需操作。
profile3302替代3301。核内清状态不改变错误终止合同，成功普通路径不增加quiet或Host同步。


## ADR-031｜七核专职通信与Core0末尾席位同批评估

保留原A3 profile3302，新增A3-2/3303与A3-tail/3304独立会话候选。
连续通信范围为ND输出完整行；每MM座位独占S行并完成seed后发布epoch。
Core0末尾候选必须先生产被当前批次覆盖的自有行，避免等待自身；不宣称完全无阻塞。
7核负载取整可能显著增加最忙核工作，因此两种64/128行MM粒度均同批前后A3对照。
既有首READY/ADD/最终排空/输出租约不变，不修改MemFabric，不增加Host热态同步。
新设备编译与性能NOT_RUN，见[A3_SEAT_SCHEDULE](../../memfabric_mm_ar_v3/A3_SEAT_SCHEDULE.md)。


## ADR-032｜统一Tensor完整I/O计时及A3独立打点对照

Stock/V2/V3/A3必须从就绪ND输入与NZ权重到完整ND输出ready同层级计时；
必需reserve/wrap/seed/copy/事件不得剔除。Wall主指标、Event辅助，同下标双Rank max。
采用独立进程Stock/V2前后控制和A3无打点前/Host/Device/无打点后，确认扰动与漂移。
复用公开signal/wait/quiet，不改MemFabric。新A3私有device ABI增加trace地址，Host/Device同版重编。
计数未校准仅标ticks，不跨核/Die直接相减或对重叠工作求100%占比。
见[UNIFIED_TIMING](../../memfabric_mm_ar_v3/UNIFIED_TIMING.md)，设备验证NOT_RUN。
