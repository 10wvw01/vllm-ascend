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


## ADR-017｜2026-10-28｜五层完整MM+AR对照与E1性能优先重构｜**2026-10-28实机已执行：E1>D4方向确立、仍慢于Stock**

- **用户明确需求**：反对把`Chunk MM(k)→SDMA(k)→Add(k)→下个Chunk`冒充“完整串行”；必须加入原生NZ MM→MemFabric AR（B1）和**所有自定义MM结束才允许开始AR**（B2），与Stock A、现有D4 C及实际优化候选E1组成五层。所有层核对完整MM+FP16 SUM结果，不拿分段相加冒充闭环。
- **技术决策**：E1首先用每核整DCache Clean+Invalid替换D4三个生产/消费DCCI代码位置，并将每Chunk Ready从Core0等7核改成Core0等4个代表核、3个工作核作局部汇聚；共同压低逐64B循环、跨核会合及SDMA Wait暴露的关键路径；**不擅自宣称整缓存刷新快**，正式设备计算、IB、信件顺序、QUIET/DONE保留，任何差异都必须双Rank位精确、真实NZ路由与完整Stock数据验证。
- **最大性能导向**：不为大M设置ROI停止阈值；E1失败则继续下一代协议优化，不把“过度保守”当成停止调优理由。但禁止通过无依据跳过必要数据可见性/等待或绕开KFC/Gate E报告虚假收益。
- **实施边界**：仅在由主线派生的`work/310p-mm-ar-five-arm-dcci-20261028`新增独立B2/E1设备核、私有Host及五层基准/评估器，主线D4正式Kernel、Host、Serving、Stock默认、外部DFlash仓库零改动。旧A1 Draft PR #9只读取证，不整支合入/复制过时控制文件。
- **合入/现场门禁**：源码两轮深审、原仓CPU合同与CANN BiSheng编译、同卡双Die真实独占G0、NZ路由、E1/B2双Rank短Smoke后开展十场原始Event Confirm；现场事实留原始文件。任何新候选正收益先L1，再L2真实40目标站点和L3无Profiler Prefill Step稳定验证，L3_GO后才L4 input4096/output2048/TP2/并发10/50请求TTFT。
- **状态**：~~截至本决策新增代码，B2/E1设备编译、数值、性能均`NOT_RUN`，尚不具备实机GO判定。~~ **2026-10-28实机更新**：run_id `a1-five-arm-20261028-001`（16384卡dev2,3、私有装载、独占巡检、前后SHA一致）完成两轮深审→CPU62项→dav-2002隔离编译→真实NZ路由4/4→B2/E1双Rank Smoke→十场ABBA。E1 vs D4 **+7.79%（M2048）/+10.50%（M4096）**，E1仍慢于Stock（−15.29%/−11.29%）；总状态`INCONCLUSIVE_DRIFT`（D4 M2048场间漂移5.685%>5%，stage07周期长尾），M2048速比不签。按本ADR"最大性能导向"：**E1保留为下一优化基线**（仅工作分支候选），下一E代聚焦同步消息缩减、按可见性证明消除冗余清理、治理周期长尾；B2串行结构负向（+285.8/+579.5µs vs D4）维持Chunk交叠主线；B1慢归因staging成本。详见[五层实施方案](A1_五层MM_MemFabric_AR与E1关键路径优化_20261028.md)、[实机报告](A1_FIVE_ARM_ONDEVICE_REPORT_20261028.md)。


## ADR-018｜2026-10-28｜E2性能优先：两级Add放行树与双Rank READY栅栏预失效｜**开发候选，设备NOT_RUN**

- **触发证据**：A1五层真NZ Stock/E1实机`a1-five-arm-20261028-001`：E1 M2048/4096为1263.6/2274.2µs，D4为1362.0/2513.0µs，Stock为1070.4/2017.5µs；E1相对D4约+7.79%/+10.50%，但对Stock仍落后；M2048 D4漂移5.685%使整批总状态`INCONCLUSIVE_DRIFT`。严禁引用这些数字为新E2收益。
- **新方案**：E2-R原E1每Chunk 1→7 Add Release变两级1→4→7事件树；E2-P在所有8核完成接收DCache预失效后，跨Rank双方先投递并等待带独立imm的8B READY，才允许任意数据SDMA；随后接收端Vector DataCopy读GM，去掉每Chunk收件DCCI。附加512/1024分包同算法对照，严格遵守4MiB单消息已验证约束。
- **风险与回退**：READY控制信箱复用已验证DONE控制区域但必须不同imm且先READY后DONE；双Rank提前失效后不得由Scalar访问接收arena，信件必须检查dst/len/status/imm。E2-R保留逐Chunk接收DCCI作为快速隔离路径；E1原源码与正式D4/Serving/Stock默认均未变。新增Kernel不直接加入CMake或Shared SO。
- **验证次序**：CPU代码合同、BiSheng dav-2002私有SO、四卡只读G0与健康独占、Stock NZ真实路由、E2R→E2P1024→E2P512各自双Rank短Smoke，PASS后十场E1/E2/Stock配对ABBA，无Profiler Event+Wall，p50/p90/p99/漂移与双Rank原始取证；任错算/KFC/IB/507014/507899/507018立停，不自动切卡或禁防护。
- **生产验收**：E2设备完整链需跨越E1/Stock，随后按ADR-015做40站点L2/L3 Prefill关键路径，L3 GO后才做正式input4096/output2048/TP2/batch10 TTFT；此ADR不构成Serving启用或实机GO。
- **实施位置**：`work/310p-mm-ar-e2-pipeline-20261028`（以E1实验工作分支为父），[E2独立实施合同](A1_E2_两级Release与READY预失效性能攻坚_20261028.md)。状态`CODE_PENDING_CI_BUILD_NPU`。
