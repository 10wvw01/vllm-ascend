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

## 后续决策的写入模板

`ADR-010｜YYYY-MM-DD｜有效/被替代`

- 决策：
- 触发证据（SHA、报告、原始JSON或现场路径）：
- 取舍与未采纳的方案：
- 代码和当前默认路由影响：
- 验证门槛、回退方案和需要更新的控制台章节：
