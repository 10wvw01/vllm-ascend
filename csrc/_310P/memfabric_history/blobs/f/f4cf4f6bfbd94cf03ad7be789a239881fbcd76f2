# 310P3 Qwen3.6 大 M 融合算子｜Prefill TTFT 实验总账

> **文档定位：本实验唯一执行与进度总账**（截至 2026-10-27，Asia/Taipei）。目标/决策的最高依据是 [ADR-015/016](DECISIONS.md)；本总账管理“做什么、为什么、依赖、状态、结果、下一步”。详细方法与历史证据仍以 [L1～L4 实验合同](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md) 为准，不重写旧实验报告。
>
> **唯一研发主线：** [dev/310p-memfabric-mm-ar](https://github.com/10wvw01/vllm-ascend/tree/dev/310p-memfabric-mm-ar)（本轮制定总账时 HEAD d5472c902c2a8576e3eb7bafb1fada0eaedce3ff；合并本总账后以实际远程 HEAD 为准）。
> **当前实施分支：** [work/310p-mm-ar-large-m-a1-20261026](https://github.com/10wvw01/vllm-ascend/tree/work/310p-mm-ar-large-m-a1-20261026)，[PR #9（Draft）](https://github.com/10wvw01/vllm-ascend/pull/9)，已核实远端 HEAD 527882246bbdbb8fc21b0b9fe6c361bb3d25316d。
> **权限边界：** 本文件只进行任务管理与实验设计，不授权 OpenCode 运行 Kernel、修改服务器 SO、执行 KFC、启动模型或跨阶段压测；每阶段均须完成相应安全审查并由用户批准。

## 0. 一眼看懂：目标、当前状态、下一步

| 项 | 当前正式结论 |
|---|---|
| **唯一最终目标** | 在 Ascend 310P3 单卡双 Die / TP=2 上，令**真实 Qwen3.6 Prefill 对应的客户端 TTFT**优于同环境 Stock，改善可重复、可归因 |
| **唯一 L1 算子级基线** | 完整原生 FP16 **Stock MM + HCCL AllReduce**，不是 MM-only、HCCL-only，也不是旧 V2 |
| **核心优化对象** | 真实模型每 Rank 预期的 **40 个未量化 RowParallel 目标层**（attention o_proj 10 + GDN out_proj 30）的完整设备关键路径；实际命中数、M/布局以本次取证为准 |
| **迭代性能门槛** | **L3-first**：先用无 Profiler、设备完成语义可靠的真实 Prefill Step A/B 证明稳定正向收益；没有 L3_GO 就不执行正式 L4 |
| **正式 L4 负载** | Qwen3.6-35B-A3B-w8a8，input=4096 / output=2048，TP2，Eager，50 请求，并发 10，ignore-eos；相同调度设置与随机输入 |
| **当前最新 D4 的完整 Stock L1** | **UNKNOWN / 新一轮 NOT_RUN**；D4 +6.26% 只是旧 V2→新 V2 的 L0 |
| **当前最新 D4 的 L3、L4** | **NOT_RUN / 未验收**；不存在已证实的当前版本 TTFT 正收益 |
| **现阶段工作** | A1 测试实现存于 PR #9 Draft；原仓 CPU 套件、310P3 四卡只读 G0 **尚未得到执行结果**；L3 无 Profiler 设备 Step 计量方案尚未完成验证 |
| **下一步行动** | ①研发侧静态核对 L3 设备计时能力并设计低侵入方案；②OpenCode 在用户授予相应许可后，仅执行 A1 原仓 CPU 检查 + 四卡只读 G0，将证据回传审核；**不可直接运行 Smoke/Screen** |
| **默认/风险** | Serving **Stock 默认不变**；Gate E/507014、首写静默损坏与未解决的设备版本/拓扑异常保持硬门槛 |

**总账记账准则：完成文档 ≠ 完成代码；代码提交 ≠ CPU 测试通过；CPU PASS ≠ G0 PASS；G0 PASS ≠ 实机许可；单算子快 ≠ L3 快；L3_GO ≠ L4 TTFT GO；性能 GO ≠ 发布 GO。**

## 1. 目标定义、成功条件、非目标

### 1.1 最终完成标准

1. **L1：**在真实形状 M=2048/4096、K=N=2048、FP16、同卡双 Rank 下，获得完整 Stock MM+HCCL 对正式 V2 MM+MemFabric AR 的正确性、时延 p50/p90/p99、跨流通信完成证明与完整版本身份；结果即便为负也如实入账。
2. **L2：**证明真实 Qwen3.6 Prefill 逐 Rank 40 个目标站点的实际融合命中、物理 M、输入/权重布局、额外 x.contiguous 拷贝、返回张量及下游可消费时刻。不能把“选择了融合方法”视作“融合 Kernel 实际执行”。
3. **L3：**通过 **不启 Profiler**、不逐调用强制设备同步的真实模型 Prefill Step 设备完成计时，在 Stock/V2 跨服务端重复 A/B 中确认**可重复、超出热态漂移/测量噪声的正向改善**。L3 不硬性要求达到 5%：历史 40 站点占比约 3.3% 仅属于旧版本/旧工况；不能拿它充当现版物理上限。
4. **L4：仅在 L3_GO 后**，以正式 4096/2048/TP2/并发10/50请求/Eager 工作负载的客户端 TTFT p50/p90/p99、原始逐请求时延和会话级重复结果作最终裁决。需解释 L1→L2→L3→L4 收益/退化的传导差额；无法直接观测的排队或跨流时段标 UNKNOWN。
5. **正确性/安全性独立门禁：**bit-exact、首写/KFC/通信协议、Host/Device/KVER11 加载版本、双 Rank 稳定性不达标即终止该实验；即使性能正向也不可启用生产默认。

**非目标：**不把本项目变为 MoE/GDN/Attention 模型本体通用优化；不复活废止的 V2 Runner/DFlash ModelState 重建；不向外部 HaonanWu1 仓库推送；小 M/DFlash/Graph 是项目其它工作线，不把它们的结果混记本轮普通 Eager 大 M TTFT 收益。

### 1.2 固定测试合同

| 维度 | L1/L2/L3 | L4 正式验收 |
|---|---|---|
| 卡/进程 | 310P3 **单物理卡双 Die TP2**，双 Rank，每次 ABBA 固定同一卡及 SO/KVER | 完全相同 |
| 形状 | L1：M=2048、4096，K=N=2048，FP16；M1024仅诊断形状；L2/3：模型实际 Forward M 必须取证 | input4096，output2048，实际 M 与命中数保留 |
| 计时 | L1：完整 MM+HCCL vs 完整 V2、真实 HCCL 完成；L3：无 Profiler 的**设备 Prefill Step 边界** | 无 Profiler 的**客户端 TTFT**、请求到达/返回及服务侧可观测阶段 |
| 模式 | Qwen3.6 Eager；L3 主筛 input4096/output1、并发1，热态 | 普通 Qwen3.6 Eager、50请求、并发10、ignore-eos、相同调度策略 |
| 样本 | L1：Smoke→Screen→按许可 Confirm 8会话；L3：轻量Screen，正向后 Confirm Stock–V2–V2–Stock，建议每场≥30稳态样本 | Stock–V2–V2–Stock 四场独立 Serve，每场正式业务相同；保留全部异常和失败样本 |
| 比较要求 | 同源 Stock、同二进制/热态、实际 M/stride/Kernel 取证；双 Rank Event 样本先配对；跨 Rank 未校准时不减绝对时间戳 | 不能把50个相关请求当50场独立实验；逐会话比较，报告 Stock 自然漂移与不确定性 |

### 1.3 优化与决策流程

```text
主线锁定目标/版本/总账
    ↓
A1 原仓 CPU 回归 + 只读四卡 G0（独立审批）
    ↓ G0_GO + 实机授权
L1 M2048 Smoke → Screen → 独立 HCCL Event/路由审计 → Confirm
    ↓
L2 实际40层路由、布局、Copy、设备关键路径审计（必要时短Profiler）
    ↓
L3 设备 Step 计时能力静审/校准（若未完成，不得臆造L3）
    ↓ 计时可用 + 安全准入 + 实机授权
L3 无Profiler轻量Stock/V2 Screen
    ├─ 无收益/不确定 ──→ 分析L1/L2与SDMA/IB/Copy/MM → 优化下一版 → 重做受影响验证
    └─ 正向信号 → L3 无Profiler Stock–V2–V2–Stock Confirm
                   ├─ 不达标 ──→ 返回优化，L4不运行
                   └─ L3_GO + 独立实机授权
                                  ↓
                      L4正式4096/2048 TTFT ABBA
                                  ↓
                L1～L4传导报告、稳定性门槛与是否合入裁决
```

**诊断性的短 Profiler 不是 L3 正式计时；客户端 4096/1 TTFT 也不是完整的设备 Prefill Step 时间。** 若现有工具无法证明无 Profiler 设备完成边界，应先解决测量能力，报告 L3_MEASUREMENT_NOT_READY，而不是把它判为 L3_NO_GAIN。

## 2. 任务清单（本总账的维护核心）

**字段约定：**任务 ID 永不复用；“状态”表示工作进度，“结果/证据”只记录实际发生的事实，不能拿计划代替实绩；一个阶段未获授权时填写 NOT_RUN，即便前置任务 READY 也不能标 GO。责任：“研发侧”=ChatGPT设计/编码/静审，“现场”=用户委托的OpenCode在310P3执行，“裁决”=用户审核许可与合入决策。

**状态值：** DONE_DOC（文档/决策已在主线）、DRAFT_CODE（代码只在工作分支）、TODO（待实施）、NOT_RUN（设备或原仓测试尚未执行）、BLOCKED（前置门禁不足）、CONDITIONAL（只在条件成立时执行）、IN_PROGRESS（有可核查进行中证据）、PASS/FAIL/INCONCLUSIVE（该项完成后的真实结果）。所有任务的结果栏可继续为 UNKNOWN。

### 2.1 目标与准备任务

| ID | 做什么；解决的问题 | 负责人 / 完成标准 | 状态（2026-10-27） | 已有结果/证据与下一动作 |
|---|---|---|---|---|
| M-01 | **锁定主目标和验收口径**；避免追逐旧V2微基准、输出吞吐而偏离Prefill TTFT | 研发侧；ADR-015、L3-first、40层范围与完整Stock定义落主线 | **DONE_DOC** | [ADR-015](DECISIONS.md)、[PR #10 已合入](https://github.com/10wvw01/vllm-ascend/pull/10)；后续遇到新证据时按ADR追加修订 |
| M-02 | **审计历史L1～L4**；避免重复采历史CSV或把不同版本拼成收益 | 研发侧；分离旧V2、D4与SEL-01证据及局限 | **DONE_DOC** | [历史归因/实验合同](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)；D4最新L1仍 UNKNOWN |
| M-03 | **建立主线/工作分支分工与本总账**；避免工作分支成为影子项目主线 | 研发侧；ADR-016、生效的主线总账/入口 | **DONE_DOC**（本次合入后） | [ADR-016](DECISIONS.md)；工作分支 PR #9 只管实现，下一步执行要读主线总账 |
| A-01 | **A1离线双Rank L1验收工具**；解决Screen被误判GO、丢Rank/假ABBA/单形状退化未拒绝 | 研发侧；原仓CPU单测、双Rank原始文件级合同、两轮静审 | **DRAFT_CODE** | [PR #9](https://github.com/10wvw01/vllm-ascend/pull/9)已有实现与隔离副本20项CPU合成测试；**原仓完整CPU套件 NOT_RUN**。下一步在实际checkout执行并提交日志 |
| A-02 | **G0只读取证工具**；先证明源码/SO/KVER/卡号身份，避免无证据上机 | 研发侧 + 现场；A1只读探针及UT、真实四卡巡检、用户审核 | **DRAFT_CODE / 现场 NOT_RUN** | 工作分支已修根目录推导并增加UT；尚未从服务器获得原仓CPU/G0数据。下一步只读巡检，不建池不启动Kernel |
| A-03 | **L3无Profiler设备Step测量设计与验证**；解决把客户端TTFT/Kernel累计误当设备关键路径 | 研发侧；证明当前vLLM可观测边界或实现仅测试路径低侵入计时，静审、CPU测试、Profiler交叉校准 | **TODO；计量能力未确认** | 现有v2_prefill_capture.sh和U1**未证明满足L3要求**；下一步先查已有Step观测接口/设备Event完成语义，严禁逐层强制同步 |

### 2.2 L1、L2、L3、L4 实验与优化任务

| ID / 阶段 | 做什么；解决的问题 | 负责人 / 验收要点 | 状态（2026-10-27） | 实际结果/证据与下一个动作 |
|---|---|---|---|---|
| E-01 重进场只读 | **服务器切过别的分支后重扫四卡、CANN/Python/真实SO与Git工作树**；防止误用旧环境、影响其它任务 | OpenCode现场，研发侧审核；`environment_inventory.json`、`card_selection.json`与四卡证据齐 | **工具已在PR#9 Draft；现场 NOT_RUN** | [A1重进场脚本](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/test/perf/a1_reentry_readonly.py)只读；下一步在用户委托后执行BATCH-R0，不自动签发G0 |
| E-02 修复溯源 | **找回2026-10-23经实机验证的MemFabric三处SetAccess物理DeviceID补丁与1dc46d7d SO**；防止新构建缺映射致507899 | OpenCode现场取证、研发侧审查；旧`mf-fix`本地diff/git/stash/二进制完整SHA或证明丢失 | **远端7分支静审完成；现场查找 NOT_RUN** | 镜像7分支未发现修复；2026-10-23修复成功的服务器源码或备份待找。兜底[MemFabric Draft PR#1](https://github.com/10wvw01/memfabric_hybrid/pull/1)不得直接部署 |
| E-03 CPU与隔离编译 | **原仓A1 CPU回归、必要时MemFabric与V2隔离staging构建和符号/ABI清单**；避免未经核验替换全局SO | OpenCode现场；CPU stdout/stderr/exit、源码SHA、编译参数和产物SHA、可恢复备份方案 | **NOT_RUN；候选源码 Draft** | 优先沿用经哈希验证的历史SO；新编译hash变化要单独审批冻结U1合同。第一批不得`pip install`到共享环境/执行KFC/Kernel |
| G-01 安全准入 | **原仓CPU合同 + 四卡只读G0**；确认双Die、独占、Host双SO/Device/KVER11、冷池恢复方案 | 现场执行，研发侧审证，用户签批；得到完整G0包 | **NOT_RUN** | **服务器已切换其它分支，历史G0全部作废。** 第一批只可执行E-01～E-03批量准备并回传，不可直接Smoke；遵循[重进场/重编译部署合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_REENTRY_BUILD_DEPLOY_AND_BATCH_GATE_20261027.md) |
| L1-01 Smoke | **M2048双Rank Stock/V2位精确与KFC首调冒烟**；排除冷首写/协议假成功 | 现场；G0_GO且**单独授权**，双臂状态/bit-exact/加载记录 | **BLOCKED / NOT_RUN** | 待G-01；不做性能GO；任何507014/507899/错算STOP |
| L1-02 Screen | **M2048/4096完整Stock vs D4首轮时延**；快速识别差距与真实优化方向 | 现场；2场既有U1大M Screen、保留双Rank原始采样/尾部和SO指纹 | **BLOCKED / NOT_RUN** | 待L1-01成功与批准；M1024仅探针，不记为正式Serving命中 |
| L1-03 路由审计 | **独立核实HCCL异步跨流尾部与V2正式D4命中**；避免W1/W4不公平Stock对照 | 研发侧+现场；Device Trace/HCCL完成Event、权重/输入布局及调用完成语义可复核 | **BLOCKED / NOT_RUN** | 待L1-02；审计失败时L1不能签发“完整链可信” |
| L1-04 Confirm | **Stock–V2–V2–Stock ×2（8场）**；拒绝单轮噪声/热态漂移/错误胜负 | 现场；另行批准、每M双Rank40样本/场、预热及独立route审计、CPU报告复算 | **BLOCKED / NOT_RUN** | 待Screen及路由审核；最终分M给出微秒差、p50/p90/p99、效速和具体证据 |
| L2-01 模型命中 | **确认真实40层 / M / stride / x.contiguous及下游就绪**；避免选择日志伪命中 | 研发侧设计、现场采样；每Rank站点和Kernel/拷贝一一对应 | **BLOCKED / NOT_RUN** | 待模型/安全准入；首选低侵入路由审计，不能从候选路由配置推断实际命中 |
| L2-02 短Profiler | **按需补Stock/V2各一次短Trace**；查SDMA/IB等待、拷贝、跨流依赖与L3计量校准 | 现场；仅在需要归因且单独授权时执行，原始双RankTrace可复核 | **CONDITIONAL / NOT_RUN** | 旧2026-10-21历史Profiler已归档但不是D4；不把Profiler结果直接充L3 GO |
| L3-01 Screen | **无Profiler的真实Qwen3.6 Prefill Step Stock/V2短A/B**；决定新候选是否值得进一步确认 | 现场；A-03设备计时合同通过、真实M和40层核实、独立许可、热态至少短窗口 | **BLOCKED / NOT_RUN** | 若无可可靠测量的设备完成Step，写L3_MEASUREMENT_NOT_READY；有负向/噪声则不启动L4 |
| L3-02 Confirm | **Stock–V2–V2–Stock四Serve无Profiler ABBA**；给出稳健的Prefill关键路径增益 | 现场+研发侧分析；Screen信号正向且另获许可；建议每场≥30稳态样本、每Rank设备Step、Stock漂移与失败全量 | **CONDITIONAL / NOT_RUN** | 仅真实重复、超出噪声并数值安全才能标L3_GO；否则L3_NO_GAIN或L3_INCONCLUSIVE |
| O-01 核心优化循环 | **针对L1/L2/L3证据调优正式V2**；优先定位SDMA到达等待、Core0/IB控制、MM/SDMA交叠、布局拷贝 | 研发侧改正式Kernel/薄接入，现场按门禁复测；每轮候选SHA+变更变量+收益归档 | **TODO（依首轮证据确定单变量）** | 历史Mac近似持平、MTE2/Scalar偏高只能给疑点排序；C3、F2负结论保留；**不得以无L3收益终止大M路线** |
| L4-01 正式TTFT | **原始4096/2048、TP2、并发10、50请求的Stock/V2/V2/Stock四会话**；检查Prefill收益能否传递到客户端 | 现场；**仅L3_GO且另获许可**；原始逐请求TTFT/故障、p50/p90/p99、同版身份与实际路由 | **CONDITIONAL / L4_NOT_RUN_BY_DESIGN** | 当前L3未通过，因此不运行L4；不存在现版可宣称的TTFT改善 |
| L4-02 四层传导 | **解释为何L1算子微秒变化没有等额传到L4**；拆出布局、重叠、双Rank与队列调度，标注不可观测段 | 研发侧；L1/L2/L3/L4同版及各工况可比，分段证据链与不足项逐一列出 | **BLOCKED / NOT_RUN** | 待四层真实可比证据；禁止直接累计40×p50或把旧历史结果混入新L4 |
| R-01 关闭/合入 | **独立评审Kernel候选、数值/发布级Gate E与Stock默认**；防止性能GO直接变成生产启用 | 研发侧审查，现场必要回归，用户裁决；变更/回滚/报告与主线同步 | **CONDITIONAL / NOT_RUN** | 发布Gate E/507014、首写及Graph门禁仍未解除；即使L3/L4正向也不得直接改默认 |

> 每个状态后面必须填写可点击的PR、提交、报告或证据目录；**没有结果就写“UNKNOWN/无新证据”**，不能把预期数值填成PASS。旧版 D4 Add Tile2048 +6.26% 已完成的是历史 **L0**，不是表中本轮 L1～L4 的完成项。

## 3. 已有历史证据（只作来源，不计为当前轮实机进度）

| 日期/来源 | 确认过的观测 | 对本轮有什么帮助 | 不可越过的边界 |
|---|---|---|---|
| [2026-10-21旧版Prefill归因](V2_PREFILL_ATTRIBUTION_REPORT_20261021.md) | M4096历史Rank0 Stock MM p50约795µs + HCCL AR约1149µs，旧V2 Kernel约2573µs；输入额外拷贝约66µs/站点；历史锚定Prefill步骤2364.4→2395.7ms，稳态单请求时延恶化约34ms | 证明旧V2在真实Serving下本来比Stock慢，老W1同步税和W4拥塞会误导 | **不是现D4**；两段独立p50之和不是严谨完整算子配对p50 |
| [2026-10-22旧CSV复算](../evidence/v2_prefill_attribution_20261021/analysis/gate_a_offline_audit_20261022.json) | 双Rank每步40目标站点已从历史逐站点CSV复算；Rank0干净Step目标站点Stock77.8525ms、旧V2核105.9149ms（未计Copy） | 旧版L2/L3已被二次复核，无需重复历史CSV分析 | 跨Rank绝对时钟未校准；报告正文Host SO前缀6e9dc2d6与证据provenance的88d39285冲突：PROVENANCE_CONFLICT |
| [2026-10-23 D4 Add优化](V2_D4_ROUND1_ADD_TILE2048_ONDEVICE_REPORT_20261023.md) | M4096 V2 Tile1024 2726.9µs → Tile2048 2556.3µs，相对旧V2 +6.26% | 当前正式D4候选参数基线 | **仅L0内部比较**；与同版完整Stock性能差距仍 UNKNOWN |
| [2026-10-24 SEL-01 Eager](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-sel01-three-arm-eager-c2-audit-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/SEL01_STOCK_V1_V2_EAGER_4096_2048_TP2_BATCH10_REPORT_20261024.md) | Stock和V2各4个有效长输出会话，吞吐中位45.98 vs 46.49 tok/s（+1.12%，跨轮不一致）；TTFT无稳定正收益 | 已知大M V2整模输出吞吐信号不足，Stock默认有依据 | 缺同期严格完整L1与L3关键路径；不能称当前L4 GO |
| [2026-10-25 F1-C小M](../evidence/f1c_20261025/F1C_ONDEVICE_REPORT.md) | M10旧T1 Stock308.2µs、T1 425.2µs | 避免用旧F1假正向和小M数据支持大M结论 | 独立小M路线，不计本轮大M里程碑 |
| [2026-10-27 主线正式裁决](DECISIONS.md) | ADR-015锁定Prefill TTFT与L3-first；ADR-016明确主线管方向、工作分支管实施 | 本总账的决策依据 | **文档完成，不产生设备性能结果** |

## 4. 阻塞/风险台账（每轮必须更新）

| 风险 ID / 优先级 | 现象与影响 | 当前状态与证据 | 解锁条件 / 责任 |
|---|---|---|---|
| B-01 / P0 | **没有最新D4完整Stock L1**，无法判断真实胜负及有效优化焦点 | UNKNOWN；A1工具在PR#9，原仓CPU及U1 NPU NOT_RUN | G0+Smoke+Screen+跨流HCCL核验+Confirm；研发侧/现场 |
| B-02 / P0 | **L3无Profiler、无额外同步的真实设备Step边界未证明可测**，阻止L3正式判定 | L3_MEASUREMENT_NOT_READY，尚未静审已有可观测接口 | 审查/实现最小诊断计量并完成CPU与短Profiler校准；研发侧 |
| B-03 / P0安全 | **Gate E/507014与首写潜在静默损坏**，会污染数据且阻止生产启用 | 历史故障待根治；与本轮L3收益判断分开，但异常即STOP | 故障取证与正确性回归；MemFabric团队协作/现场；正式发布前关闭 |
| B-04 / P0准入 | 旧43520 Die7/MCU告警、不同设备号映射和SO装载身份 | 新四卡健康/G0尚无本轮新证据 | 每次四卡只读择优、同卡双Die/独占、完整SHA/恢复、许可；现场 |
| B-05 / P1 | **历史2026-10-21 Host SO SHA记载冲突**，误导跨轮归因 | 报告6e9dc2d6 vs provenance 88d39285；未解释 | 核查旧原始安装/Trace存在性；不影响开展新同场实验；研发侧/现场 |
| B-07 / P0环境漂移 | **服务器当前跑过其它分支**，CANN/插件/Host双路径SO/KVER和卡级占用未知；MemFabric镜像7分支的三处SetAccess未包含2026-10-23非identity物理号修复 | 只读远程源码审计已做，服务器工作树/备份/旧SO尚未查证；新PR仅Draft | 先完成E-01～E-03，证明当前确切装载版本与回滚；任何构建SHA变化/共享覆盖必须独立审查批准 |
| B-06 / P1治理 | A1工作分支含主线已有的旧版治理文件，存在误覆盖风险 | PR#9 Draft，基底早于ADR-015/016主线合入 | 后续代码PR按新主线比较、选择性合入，经两轮审查；研发侧 |

## 5. 实机批次、结果表（新实验开始后在这里逐行追加）

**新一轮当前没有任何已经完成的设备批次。** 下表为录入格式，示例行明确标记 NOT_RUN，不得误认是实测。

| run_id | 日期 | 阶段/任务ID | 实际设备/Rank | Git SHA + SO/KVER指纹 | Stock/V2/负载 | 状态 | 结果摘要 | 原始证据路径 / 审核人 |
|---|---|---|---|---|---|---|---|---|
| 待创建 | — | G-01 | 待四卡只读G0选定 | UNKNOWN | CPU + 只读G0 | **NOT_RUN** | 无新现场结果 | 待上传 |
| 待创建 | — | L1-01～04 | 待G0 | UNKNOWN | M2048/4096完整Stock vs D4 | **NOT_RUN** | 当前D4完整Stock L1 UNKNOWN | 待上传 |
| 待创建 | — | L3-01～02 | 待G0和L3计时验收 | UNKNOWN | 4096/1无Profiler Prefill Step | **NOT_RUN** | L3_GO/NO_GAIN/INCONCLUSIVE均未判定 | 待上传 |
| 待创建 | — | L4-01 | 待L3_GO | UNKNOWN | 4096/2048/50/10正式TTFT | **L4_NOT_RUN_BY_DESIGN** | 未启动正式L4 | 待上传 |

### 5.1 单轮结果必须回传的文件与字段

- **身份**：run_id、工作分支与实际HEAD、所选卡/双Die物理映射、CANN/Host系统SO及Python SO/Device SO/KVER11 完整SHA、运行时真正装载路径、构建旗标、启动/恢复记录。
- **数据**：Stock与V2每场所有Rank原始样本、M/stride/weight布局、完整MM+HCCL Event与实际通信完成证明、40层命中、Copy、Prefill Step设备起止边界、真实TTFT逐请求、所有失败与异常。
- **方法**：是否Profiler、预热个数、请求Token、Eager/Graph状态、A/B顺序、跨会话漂移、是否独立进程、设备是否全程独占；若测量口径变动，视为新实验ID，不拼旧数据。
- **统计与裁决**：L1各M Stock/V2 p50/p90/p99与配对差；L3各Serve完整Step时间与漂移、L3_GO或未达原因；L4 TTFT p50/p90/p99与逐会话差值。若阶段未执行或不可观测，写 UNKNOWN/NOT_RUN，不能补造数字。
- **安全**：正确性/首写/507014/507899/KFC/IB/Graph伪路由及设备异常；发生故障先STOP保全，未经用户新授权不得自动换卡重试或跳到下一阶段。

推荐归档命名：单独的带日期和阶段的 evidence/<run_id>/（或经审查的等价目录），每次报告将链接补入任务表与本批次表。历史原始Profiler大文件仅在服务器时应标 **SERVER_ONLY**，不得捏造GitHub已上传。

## 6. 进度更新日志与每轮交接规则

| 日期 | 更新事项 | 完成/未完成的真实结果 | 关联证据 | 下一动作 |
|---|---|---|---|---|
| 2026-10-21～22（历史） | 旧V2真实Prefill归因与CSV二次审计 | 旧版L1/L2/L3/L4方向性链可解释，非现D4新测试 | [历史归因](V2_PREFILL_ATTRIBUTION_REPORT_20261021.md)和[离线复算](../evidence/v2_prefill_attribution_20261021/analysis/gate_a_offline_audit_20261022.json) | 旧证据保留，不重复计算当新结果 |
| 2026-10-23～24（历史） | D4 Tile优化、SEL-01真实Eager A/B | D4内部+6.26%；正式业务TTFT未证明稳定获益 | [D4](V2_D4_ROUND1_ADD_TILE2048_ONDEVICE_REPORT_20261023.md)、[SEL-01](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-sel01-three-arm-eager-c2-audit-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/SEL01_STOCK_V1_V2_EAGER_4096_2048_TP2_BATCH10_REPORT_20261024.md) | A1需重测现D4完整Stock |
| 2026-10-26 | A1工具/只读G0探针已写入工作分支 | DRAFT_CODE；隔离重建CPU合同20项PASS，原仓CPU/设备NOT_RUN | [PR #9](https://github.com/10wvw01/vllm-ascend/pull/9) | 补原仓CPU + 四卡只读G0并审查 |
| 2026-10-27 | ADR-015/016、历史盘点与L3-first回归主线 | 方向文档经PR#10合入；无新设备数值 | [PR #10](https://github.com/10wvw01/vllm-ascend/pull/10) | L1合同与L3计时实现分开推进 |
| 2026-10-27 | **创建本大M TTFT实验唯一总账** | 任务、依赖、历史结果、风险与本轮未跑状态已登记；**不是实机GO** | 本文件与本次docs-only主线变更 | 维护M/A/G/L/O/R系列任务和第一份实际CPU/G0结果 |

| 2026-10-27 | **服务器环境重进场与MemFabric物理号修复取证准备** | 原服务器其它分支已运行，旧G0作废；查远程镜像7分支没有物理映射修复，历史1dc46d7d实机SO仍需服务器源文件/备份取证；新增A1只读工具和批量准入合同，**没有原仓CPU/服务器实机新结果** | [A1准备文档](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_REENTRY_BUILD_DEPLOY_AND_BATCH_GATE_20261027.md)、[MemFabric Draft PR#1](https://github.com/10wvw01/memfabric_hybrid/pull/1) | 用户委托OpenCode仅执行BATCH-R0：只读盘点 + CPU + 历史源码/二进制溯源 + 隔离编译/恢复方案；回传审核后才评估部署/Smoke |

**每轮结束至少更新四处：**①本文件第2节任务状态/结果/下一步；②第5节本轮实测批次、失败与原始证据；③第6节追加一行日期更新；④若事实改变项目当前裁决，同步 [PROJECT_CONTROL.md](PROJECT_CONTROL.md)，有经审核的新性能数值同步 [PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md)，新决策再追加 [DECISIONS.md](DECISIONS.md)。不修改旧实验中的原始数字，纠错使用新记录说明原因。

**下一轮现场交接固定输出：**任务ID/开始前允许执行的阶段、实际HEAD/环境SHA、上一门禁PASS或NO-GO、操作所依据的文档、所有执行结果与异常、完成后真实状态、新结果应更新的文件、下一步是否还需要用户批准。任何“执行中”不得仅凭口头声称而改为PASS；新会话不从聊天记忆恢复状态，而从本总账与主线控制台恢复。

---

**本总账初始裁决（2026-10-27）：**目标已确定，历史证据已归档，A1工具仍在Draft实施分支；原仓CPU/四卡只读G0及最新D4 L1、L2、L3、L4尚无新实测。**现在先补A1 CPU/G0并推进L3测量能力的静态设计，不具备直接跑性能测试或进入L4的条件。**
