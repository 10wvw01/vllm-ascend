# 310P3 MemFabric MM+AR｜分支治理、代码归属与集成策略

> **2026-10-30用户最新指令：后续直接在`dev/310p-memfabric-mm-ar`开发、修复、执行实机实验并归档。** 实验从主线固定完整SHA，使用隔离构建/结果目录；不再默认创建工作分支或等待PR合并才允许测试。G0、CPU合同、私有编译、短Smoke、正式性能及故障STOP要求继续有效。以下旧分支流转规定仅作历史背景，以本条为准。

> 基准检查：2026-10-25，远程仓库 `10wvw01/vllm-ascend` 原有33个分支；新增 `dev/310p-memfabric-mm-ar` 后为34个（如有后续新建以GitHub实时查询为准）。**不删除、不重写任何历史分支**。当前研发状态只以[PROJECT_CONTROL.md](PROJECT_CONTROL.md)为准。

## A. 唯一研发主线与分支角色

| 名称/模式 | 角色 | 允许的动作 |
|---|---|---|
| **`dev/310p-memfabric-mm-ar`** | **唯一项目主线与长期有效的项目控制台** | 主线状态/指标/代码审查、通过回归的融合实现和DFlash薄适配；新研发从它派生、评审后回到它 |
| `main` | vllm-ascend仓库默认通用分支，**不是本项目研发主线** | 禁止无明确发布决议把MemFabric实验推入main |
| `HaonanWu1/vllm-ascend:dev_v24/feat-310p-dflash` | **外部只读架构对照** | 可以读取、记录SHA、比对差异；**不得push、改写、发PR或在此仓建立工作分支** |
| 自己仓库 `dev_v24/feat-310p-dflash` | DFlash共同基础的历史参考 | 不作为融合项目主线；需要时只读对照 |
| `work/310p-mm-ar-<单一任务>` | 历史短期工作分支 | 保留来源和原始证据；2026-10-30起后续工作直接在项目主线完成 |
| `perf/` / `diag/` / `docs/` / `fix/` / `feat/` / `merge/` / `backup/` 旧分支 | 既有历史资产 | **冻结作为证据或受控回收来源**；不得将其当成新的稳定主线；不得不经检查整体merge到主线 |
| `release/310p-mm-ar-<版本>`（**未来**） | 通过正式E2E/Graph/安全门禁后的发布快照 | 现在不存在，不抢先创建 |

主线**从** `perf/310p-u1-fullchain-attribution-20261025` 派生：它继承D4已验证大M正式核、F1-C、U1测试工具、我方DFlash接入代码；与D4的`mf310p_v2_kernel.asc`逐字相同。U1工具的CPU/NPU回归仍需现场执行；选择其为资产基底**不等于认可U1是未来项目方向，也不等于正式启用T1或V2**。

## B. 已有33个远程分支按用途收编（不删分支、不丢证据）

**长期/底座与安全快照（7条）：**
- `main`：仓库默认。
- `dev_v24/feat-310p-dflash`：DFlash历史底座。
- `feat/310p-memfabric-public-api`：老MemFabric正式特性分支，**不是当前D4/U1最新代码**。
- `feat/310p-memfabric-standalone-op`、`feat/310p-w8a8-o-proj-memfabric-ar`：早期设计实现，冻结。
- `backup/310p-memfabric-public-api-before-dflash-merge-20260930`、`backup/310p-memfabric-public-api-before-reset-20260930`：安全归档，不能删除。

**阶段实验、性能与真实模型记录（10条）：**
- `perf/310p-v2-d3-add-tile2048-20261023`：D4 Tile2048成功基线。
- `perf/310p-v2-prefill-attribution-20261021`：Stock与融合站点真实Prefill分析。
- `perf/310p-v2-sync-async-value-20261021`：历史执行模式性能归因。
- `perf/310p-sel01-three-arm-eager-c2-audit-20261024`：真实4096/2048三方。
- `perf/310p-v2-c3-cleanlines-unroll4-20261024`：C3 NO-GO。
- `perf/310p-f2-mm-sdma-overlap-sweep-20261024`：F2 Chunk/Overlap初筛。
- `perf/310p-f1-smallm-fixed-cost-20261024`：F1不公平的初筛，保留负面教学。
- `perf/310p-f1c-stock-fp16-hccl-vs-memfabric-abba-20261025`：F1-C严格NO-GO。
- `perf/310p-c4-decode-cost-attribution-20261024`：扩张至MoE/GDN一般优化的**非本项目范围**实验。
- `perf/310p-u1-fullchain-attribution-20261025`：统一诊断工具准备、静态合同，尚未设备运行；主线创建基底，原分支归档。

**专项诊断/缺陷/设计（11条）：**
- `design/310p-g1c-first-stop-20261020`
- `diag/310p3-g1c-device-first-stop-20261020`
- `diag/310p3-g1d-minimal-forensics-20261021`
- `diag/310p3-gate-e-forensics-20261020`
- `diag/310p3-hccl-cache-probe-20261021`
- `diag/310p-v2-gate-c-controlled-isolation-20261022`
- `diag/310p-v2-mte2-scalar-gate-ab-20261022`
- `feat/310p-g1c-phase-trace-20261020`
- `feat/310p-g1c-v2-flight-recorder-20261020`
- `fix/310p-gate-e-verdict-20261020`
- `fix/310p-memfabric-graph-prewarm`

**旧合并及文档/独立认知资产（5条）：**
- `merge/310p-memfabric-on-dflash-20260930`
- `docs/310p-o-proj-ai-native`
- `docs/310p-repository-branch-map-20261020`
- `docs/310p-v1-v2-perf-stability-ledger-20261023`
- `docs/310p-v2-code-explain-20261022`

以上7+10+11+5=33。**这只是用途收编，不暗示每个分支无未推送或仍需保全的内容。** 若要实际清理远程分支，须先核对唯一提交、远程标签、取证路径和主线引用，由用户批准后单独操作；本轮不进行删除。

## C. 今后的编码、审核、合入规范

1. 所有任务先核对远程 `origin/dev/310p-memfabric-mm-ar` 和[项目控制台](PROJECT_CONTROL.md)；按已授权任务直接在主线实施，实验前固定完整源码SHA和工件manifest。
2. 实验、诊断、修复与结果直接追加到主线对应`test/perf/`、`experiments/`或`docs/evidence/`；使用新的私有构建和结果目录，保留每轮源码/二进制身份，不再默认另建任务分支。
3. 改动分级：
   - **文档/诊断-only**：评审正确、无需NPU也能验证的静态合同，经过至少两轮静态检查，可入主线，状态标注CPU/NPU NOT_RUN；
   - **正式Kernel / Host ABI / 通信协议 / Model Runner适配**：编译、CPU合同、双Rank正确性、已验证不绕过首写/signal/wait/credit、Stock对照，复核环境/设备SO/KVER身份后方可实机GO；实际合入需完成约定回归并记录证据；
   - **开启生产Serving**：需要Eager/Graph实际命中、模型功能/性能及长期稳定性门禁，不能仅凭算子单测或短跑。
4. 合回主线时采用经过文件级审查的提交（merge/cherry-pick均可），**绝不能把一条旧实验分支整体merge后悄悄带入未通过的Kernel或不相关旧文档**。合并记录引用报告和原始证据。
5. **每个实机报告提交后，同批更新[PROJECT_CONTROL.md](PROJECT_CONTROL.md)的状态和数字，有架构变更则更新[DECISIONS.md](DECISIONS.md)，有性能结果则更新[PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md)。** 未完成此步骤就不算交付闭环。
6. 保持Stock产品默认；性能比较“D4 vs旧V2”“V2 vs Stock”“Serving整体”三个指标严格分层，不用混合来源的时间直接推导收益。
7. 外部DFlash版本如需同步，先锁定外部commit，对照源码/ABI/Graph/lifecycle差异，**只向本仓提交最小兼容补丁**；不能在外部repo发PR。

## D. 推荐仓库日常操作

```bash
git fetch origin
git switch dev/310p-memfabric-mm-ar
git pull --ff-only
git status --short
# 读 PROJECT_310P_MEMFABRIC.md 和 docs/PROJECT_CONTROL.md
# 开始实验前固定源码：
git rev-parse HEAD
# 在主线提交修复和证据；每轮使用新的隔离构建/结果目录
```

当服务器还有未提交修改、独立worktree、已部署非当前源码的SO时，**先保全**，不得直接checkout/reset/clean覆盖证据；现场的git状态和二进制SHA不是一回事。

## E. 当前治理限制

主线不改变GitHub项目的默认分支`main`，也不替代主开发DFlash自己的分支，更不修改独立MemFabric库的发行模式。今后若需删除旧分支、设置GitHub分支保护/默认分支或建立发布工作流，必须得到单独授权和环境证据。 


## F. 2026-10-25 主线强制交付规则（增量治理）

1. 唯一主线仍是 `dev/310p-memfabric-mm-ar`；2026-10-30起所有新Kernel、适配和实机报告按用户授权直接在本主线实施与归档；本段原有短期分支/PR要求不再作为实验前置条件；不把旧 `perf/diag/docs` 一键 merge 进主线。
2. 工作流 `.github/workflows/310p_mm_ar_project_gate.yml` 只在本主线的 PR、push 或手动触发；`tools/310p_mm_ar/project_gate.py` 进行项目入口和变更台账检查，`test_project_gate.py` 是其纯CPU测试。对正式源码、目标测试或证据的变更，必须同时更新 `PROJECT_CONTROL.md`；涉及性能证据/报告还必须更新 `PERFORMANCE_BASELINES.md`。新的 ADR 必须同步当前控制台。**这是文件级遗漏检查，不验证报告真假、测试充分性或NPU正确性。**
3. 核心源码变更还需人工核对新/已有测试覆盖、2轮静态深审、G0设备/SO/KVER合同、冷首写、双Rank正确性、Stock基线、Graph真实命中与回退。具体参阅 [ENGINEERING_GATES.md](ENGINEERING_GATES.md)。
4. 当前GitHub默认 `main` 仍保留；**GitHub仓库级保护尚须所有者配置**：建议为此分支启用禁止强制推送与删除、PR审查、要求名为 `project-control` 的CI检查通过，并限定管理员绕过；保护的实际生效状态应另行在 GitHub Rulesets / Branch protection 核实。未配置时CI可以报告 FAIL，但不能阻止管理员直接push。
5. 任何GitHub CI成功均只记为“仓库元数据静态 PASS”，不表示310P服务器环境、CANN编译、模型正确性或真实性能已验证；实机测试必须由用户委托OpenCode，并在G0确认安全后执行。

## G. 2026-10-26｜偏航R2分支退出项目（ADR-013，最新裁决）

- `work/310p-v2-r2-implementation-20261025` 及关联 PR #7 **不采纳/停止合并/只读历史**。其`vllm_ascend/_310p/v2/**`、Runner/Triton/Hybrid ModelState/GDN/采样/多KV重建代码、46项准入文件和测试**一律不得cherry-pick或merge入唯一主线**。
- 主线从未合入上述R2新增模块；本次从主线撤销误入的 V2 Triton 检验豁免/测试，保留未实现V2 Worker的早拒绝及正式V2融合薄接入。GitHub不能删分支时先以关闭PR+禁用引用作为治理，**不得强制重置工作分支替代安全删除**。
- 禁止新建通用Runner研发分支。下一项开发只从本主线以`work/310p-mm-ar-<单项融合优化>`开工，限正式Kernel、定点LinearMethod与Stock回退/数值/性能/安全证据。

## H. 2026-10-27｜主线管方向，工作分支管实施（ADR-016）

- **主线的职责**：`dev/310p-memfabric-mm-ar` 是**唯一项目事实来源**，必须承载最新有效ADR、正式指标、阶段性门槛、历史数据核对与实验设计、当前经审查的结果及G0/实机安全准入。新会话/任何OpenCode现场执行从[仓库入口](../../../../PROJECT_310P_MEMFABRIC.md)与[控制台](PROJECT_CONTROL.md)恢复方向，不以工作分支的旧`PROJECT_CONTROL.md`代替主线。
- **短期分支的职责**：A1的`work/310p-mm-ar-large-m-a1-20261026`（[PR #9](https://github.com/10wvw01/vllm-ascend/pull/9)，Draft）用于尚待原仓CPU和NPU审核的测试/诊断实现、性能候选和原始证据；不得承载只有它才知道的长期正式决策，也不得因测试工具尚未验收阻止主线正式文档更新。
- **两种不同的合入门槛**：①主线方向/ADR/正式实验合同等 **docs-only** 变更，经至少两轮静态文件审查、正确引用和CI项目控制检查可合主线，明确新设备性能为`NOT_RUN`；②Kernel/Host/Serving/测量工具按安全分级CPU测试、首写/协议、G0和实机验收，再由PR选择性合入，不可将整条A1历史分支merge覆盖主线治理资料。
- **当前强制优先级**：大M [ADR-015](DECISIONS.md)以真实Qwen3.6 Prefill TTFT为最终指标，L1完整Stock作为诊断基线；先建立L3真实设备Step无Profiler可复现正向收益，**L3-GO后才批准正式4096/2048/50/10 L4 TTFT**。具体唯一合同：[历史证据复用与L3-first实验设计](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)。A1 PR #9下次合入前须与新主线对比并剔除重复/旧状态文档，保留所有未合入实施证据。
- **边界不变**：严禁复活ADR-013废止的Runner重建；不修改正式Stock默认、不改外部HaonanWu1仓库；不把docs-only合并或CI PASS冒充310P3 G0/Smoke/L3/TTFT实测成功。实机失败先STOP保全。



## 2026-10-29｜用户授权V3工作分支与MemFabric镜像同步

- V3实施分支：`work/310p-mm-ar-v3-sdma-add-20261029`，从主线`54d5518d`派生；本轮活动算法开发为V3。
- E4 PR#21保留为历史/对照证据，不将其大分支整体merge到V3或主线。
- 用户已明确授权推送V3工作分支；代码仍以Draft PR评审，不直接启用Serving或合入生产默认。
- MemFabric镜像`dev-v1.2.0`同步到`b9d46f97`，来源GitCode `ba3e7f42`，完整tree一致，历史提交身份分别保留记录。
- 服务器使用这两个仓库的新隔离克隆和detached SHA；执行者不在现有Serving worktree切分支或修改代码。
- 固定执行流程与依赖证明见[V3 OpenCode执行单](../../memfabric_mm_ar_v3/OPENCODE_EXECUTION.md)。


## 2026-10-30｜本轮用户明确授权的集中整合规则

用户要求将已实现V1/V2/V3、测试、实验及数据、系列优化全部汇入dev/310p-memfabric-mm-ar。
本轮按文件级核对整合研发资产，并以原blob快照保全冲突版本；不以先前“只留在实验分支”的规定
阻止本次已获授权的合入。仅合研发资产，不启用历史候选为Serving默认，不复活退役Runner路线。
54条源分支HEAD及合并历史纳入清单，后续唯一事实来源为主线；不创建新长期研发线、不删除源分支。
[整合目录/恢复方法](../../memfabric_history/README.md)。算法研发及实机实验保持暂停。

## I. 2026-10-30｜直接在主线进行后续实机实验（用户最新授权）

- PR #23的统一L1框架、35份版本文档和历史Host阶段表直接纳入项目主线；不以草稿PR状态阻止本次授权合入。
- 后续开发、执行中的修复、测试脚本、实验原始数据和分析直接提交`dev/310p-memfabric-mm-ar`。C/F/H节旧短分支/PR前置流程由本规则覆盖；不再默认新建实验工作分支。
- 服务器通过SSH获取主线，实验开始时记录主线完整SHA/tree及Host/Device SO、SDK/KVER、物理映射和采样配置。主线推进不追改运行中的manifest；修复后建立新一轮身份与结果目录。
- 每轮使用隔离构建/结果目录，G0→CPU合同→私有编译/短Smoke→正式性能和异常STOP继续执行。独占/共享模式按明确选择，实验模式不因分支改变。
- 原始双Rank日志、失败现场、Profiler/路由与数据同批回主线，更新控制台/基线/ADR；并发更新先核对，不强推、不覆盖其他提交。
- 本规则不启用未验收Serving默认，不删除历史分支，不更改外部HaonanWu1或MemFabric上游仓库规则。
