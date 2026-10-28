# 310P3 Qwen3.6 大 M 融合算子｜Prefill TTFT 实验总账

> **文档定位：本实验唯一执行与进度总账**（截至 2026-10-27，Asia/Taipei）。目标/决策的最高依据是 [ADR-015/016](DECISIONS.md)；本总账管理“做什么、为什么、依赖、状态、结果、下一步”。详细方法与历史证据仍以 [L1～L4 实验合同](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md) 为准，不重写旧实验报告。
>
> **唯一研发主线：** [dev/310p-memfabric-mm-ar](https://github.com/10wvw01/vllm-ascend/tree/dev/310p-memfabric-mm-ar)（本轮制定总账时 HEAD d5472c902c2a8576e3eb7bafb1fada0eaedce3ff；合并本总账后以实际远程 HEAD 为准）。
> **当前实施分支：** [work/310p-mm-ar-large-m-a1-20261026](https://github.com/10wvw01/vllm-ascend/tree/work/310p-mm-ar-large-m-a1-20261026)，[PR #9（Draft）](https://github.com/10wvw01/vllm-ascend/pull/9)，已核实远端 HEAD 527882246bbdbb8fc21b0b9fe6c361bb3d25316d。
> **权限边界（2026-10-27更新）：**用户已明确授权**一次OpenCode批次从环境恢复与M2048双RankSmoke成功后条件续跑L1 Screen/Confirm、L2和L3，L3_GO且正式负载环境可审计时续跑L4**；不再逐段返回聊天审批。前提是每阶段真实身份/数值/卡健康/恢复方案的硬门均PASS；若须影响共享SO/他人服务、出现设备异常或缺乏可靠L3计时立即STOP/保全，不得用本授权跳闸或发布默认。

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
| A-01 | **A1离线双Rank L1验收工具**；解决Screen被误判GO、丢Rank/假ABBA/单形状退化未拒绝 | 研发侧；原仓CPU单测、双Rank原始文件级合同、两轮静审 | **DRAFT_CODE / CPU_PASS** | BATCH-R0原仓CPU：`unittest`34/34、`pytest`88项+27 subtests、py_compile/bash -n均PASS；[证据](https://github.com/10wvw01/vllm-ascend/tree/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r0_reentry_20261027)。**仅CPU通过，A1代码仍PR#9 Draft** |
| A-02 | **G0只读取证工具**；先证明源码/SO/KVER/卡号身份，避免无证据上机 | 研发侧 + 现场；A1只读探针及UT、真实四卡巡检、用户审核 | **DRAFT_CODE / G0_READONLY_OBSERVED** | R0四卡实机只读已观察16384(dev2,3)健康且无可见进程；工具始终`G0_APPROVED=false`，需部署/运行身份与重新独占复核；[验收报告](A1_BATCH_R0_ACCEPTANCE_20261027.md) |
| A-03 | **L3无Profiler设备Step测量设计与验证**；解决把客户端TTFT/Kernel累计误当设备关键路径 | 研发侧；证明当前vLLM可观测边界或实现仅测试路径低侵入计时，静审、CPU测试、Profiler交叉校准 | **MEASUREMENT_IMPLEMENTED_AND_ONDEVICE_CALIBRATED** | Forward→Logits NPU Event+双Rank配对；4场ABBA各6warmup+30样本，L2逐站点完成依赖校准。不是队列/TTFT也不是跨Rank绝对时钟相减；[L3 JSON](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s5_l3/l3_result.json) |

### 2.2 L1、L2、L3、L4 实验与优化任务

| ID / 阶段 | 做什么；解决的问题 | 负责人 / 验收要点 | 状态（2026-10-27） | 实际结果/证据与下一个动作 |
|---|---|---|---|---|
| E-01 重进场只读 | **服务器切过别的分支后重扫四卡、CANN/Python/真实SO与Git工作树**；防止误用旧环境、影响其它任务 | OpenCode现场，研发侧审核；`environment_inventory.json`、`card_selection.json`与四卡证据齐 | **PASS（仅只读取证）** | [R0环境快照及四卡证据](https://github.com/10wvw01/vllm-ascend/tree/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r0_reentry_20261027)：16384(dev2,3)为唯一候选；5664/43520被占、38144残留；不得自动签发G0，R1前须重验 |
| E-02 修复溯源 | **找回2026-10-23经实机验证的MemFabric三处SetAccess物理DeviceID补丁与1dc46d7d SO**；防止新构建缺映射致507899 | OpenCode现场取证、研发侧审查；旧`mf-fix`本地diff/git/stash/二进制完整SHA或证明丢失 | **PASS：原历史源与SO找回** | 本地历史`5802f550`与两份Host SO完整SHA`1dc46d7d…`已核对；原始3个源文件**Git Blob完全一致**恢复，已[合入镜像主线PR#1](https://github.com/10wvw01/memfabric_hybrid_310p/pull/1)，新main`0670f3c7…`。历史二进制未重编或重新部署；备用异语义Draft PR已关闭 |
| E-03 CPU与隔离编译 | **原仓A1 CPU回归、必要时MemFabric与V2隔离staging构建和符号/ABI清单**；避免未经核验替换全局SO | OpenCode现场；CPU stdout/stderr/exit、源码SHA、编译参数和产物SHA、可恢复备份方案 | **CPU_PASS / HOST_BUILD_NOT_ATTEMPTED（合理）** | 原仓CPU全绿、映射15/15；已复核历史两份字节级SO与旧构建记录，无需盲目重编；下步确定隔离装载/可回滚共享部署方案，当前未安装 |
| E-04 R1A装载预审 | **以私有目录验证Host/D4/插件/KVER完整身份、动态依赖与隔离路线，不启动设备**；防止候选SO路径对但实际加载旧SO | 研发侧提供只读核验器/UT和[R1分段合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R1_ISOLATED_LOADER_PREFLIGHT_AND_SMOKE_GATE_20261027.md)；现场取证装载路径与原仓CPU | **PASS / ISOLATION_PROVEN** | 私有PYTHONPATH+LD_LIBRARY_PATH，Rank PID/maps证明正确1dc46d7d Host+D4 2a19390c+Plugin 8eb65de9；共享SO全程SHA不变，CPU108+27通过；[现场报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/A1_R1_BATCH_REPORT_20261027.md) |
| E-05 单批安全恢复+Smoke | **私有真实装载优先；首次M2048双RankStock/V2正确性与自动KFC、首写协议**，通过后不再返回聊天审批 | 现场OpenCode在用户本次条件授权下执行，真实Rank0/1 SO/Kernel/卡/数值证据强制关卡；失败STOP和回滚 | **PASS / ALLOW_SCREEN** | 双RankStock/V2 M2048 bit-exact、status=0、自动KFC/首写正常；[Smoke机器门](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s1_smoke/smoke_gate.json)；无507014/507899设备故障 |
| E-06 单批贯通L1/L2/L3 | **一次进场完成最新D4 L1原生Stock、真实Serving40站点L2和无Profiler Prefill Event L3**，减少重复KFC/模型启动和人工交接 | 研发侧现有A1门禁与新增Smoke/L2/L3验收工具、[单批操作合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R1_ONE_BATCH_THROUGH_L3_20261027.md)；现场同批条件续跑，失败保留分段证据 | **COMPLETE / L3_NO_GAIN** | 同一批次完整L1 8场、L2真实40站点两臂短Profiler、L3无Profiler四Serve 30请求/场；真实Stock NZ胜V2，L3慢26.1/27.7ms；L4按合同跳过；[独立复盘](A1_R1_实机复盘与优化优先级_20261027.md) |
| G-01 安全准入 | **原仓CPU合同 + 四卡只读G0**；确认双Die、独占、Host双SO/Device/KVER11、冷池恢复方案 | 现场执行，研发侧审证，用户签批；得到完整G0包 | **G0_PASS（本批）/ ISOLATION_PASS** | 16384(dev2,3)当场健康独占；Host/D4/Plugin/KVER实际PID加载证明，两个共享Host路径与Device安装库前后SHA不变；首轮环境PYTHONPATH遗漏tbe导致500001已记录并修正 |
| L1-01 Smoke | **M2048双Rank Stock/V2位精确与KFC首调冒烟**；排除冷首写/协议假成功 | 现场；G0_GO且**单独授权**，双臂状态/bit-exact/加载记录 | **PASS / CORRECTNESS_ONLY** | M2048双臂Rank0/1 bit-exact、V2 status=0与首写KFC；[机器验收](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s1_smoke/smoke_gate.json) |
| L1-02 Screen | **M2048/4096完整Stock vs D4首轮时延**；快速识别差距与真实优化方向 | 现场；2场既有U1大M Screen、保留双Rank原始采样/尾部和SO指纹 | **MEASURED / ROW_MAJOR_BASELINE** | Screen M2048 +15.9%、M4096 +22.2%，但Stock为row-major F.linear，不是Serving NZ；[Screen](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s2_screen/u1_screen/a1_assessment.json) |
| L1-03 路由审计 | **独立核实HCCL异步跨流尾部与V2正式D4命中**；避免W1/W4不公平Stock对照 | 研发侧+现场；Device Trace/HCCL完成Event、权重/输入布局及调用完成语义可复核 | **ROUTE_AUDITED / BASELINE_MISMATCH_FOUND** | 双臂路由及HCCL跨流完成的U1事件审计通过，但L2发现Serving Stock为aclnnMatmulWeightNz；[U1审计](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s3_confirm/u1_confirm/route_audit.json) |
| L1-04 Confirm | **Stock–V2–V2–Stock ×2（8场）**；拒绝单轮噪声/热态漂移/错误胜负 | 现场；另行批准、每M双Rank40样本/场、预热及独立route审计、CPU报告复算 | **ABBA_8_DONE / CANDIDATE_ONLY** | 正式M2048 +22.5%、M4096 +21.1%只对row-major Stock有效，不能外推模型；M1024 -11.1%仅诊断；[Confirm](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s3_confirm/u1_confirm/a1_assessment.json) |
| L2-01 模型命中 | **确认真实40层 / M / stride / x.contiguous及下游就绪**；避免选择日志伪命中 | 研发侧设计、现场采样；每Rank站点和Kernel/拷贝一一对应 | **L2_VERIFIED / ZERO_EXTRA_COPY_RECORDED** | 真实Qwen3.6每Rank每Step 10 o_proj+30 out_proj，M4096 K/N2048，D4已真命中；[20份site样本](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s4_l2/analysis/)、[L2核验](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s5_l3/l2_verified_report.json) |
| L2-02 短Profiler | **按需补Stock/V2各一次短Trace**；查SDMA/IB等待、拷贝、跨流依赖与L3计量校准 | 现场；仅在需要归因且单独授权时执行，原始双RankTrace可复核 | **DONE / STOCK_NZ_FAST** | Stock NZ MM 752µs + HCCL AR rank0 1391µs/rank1 1166µs，D4 kernel 2511/2430µs；每站点V2慢约368/512µs。原始profiler CSV服务器保留且有SHA，不在GitHub；[hash清单](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s4_l2/SERVER_ONLY_raw_traces.json) |
| L3-01 Screen | **无Profiler的真实Qwen3.6 Prefill Step Stock/V2短A/B**；决定新候选是否值得进一步确认 | 现场；A-03设备计时合同通过、真实M和40层核实、独立许可、热态至少短窗口 | **MEASURED / NO_GAIN_DIRECTION** | 热态4096/1、TP2并发1无Profiler，Stock/融合事件真实测量与L2校准；两臂均负增益，未签L3_GO |
| L3-02 Confirm | **Stock–V2–V2–Stock四Serve无Profiler ABBA**；给出稳健的Prefill关键路径增益 | 现场+研发侧分析；Screen信号正向且另获许可；建议每场≥30稳态样本、每Rank设备Step、Stock漂移与失败全量 | **ABBA_DONE / L3_INCONCLUSIVE_OR_NO_GAIN** | S1 2346.08、F1 2372.20、F2 2370.96、S2 2343.25ms；V2慢26.12/27.70ms（~1.1%），Stock漂移2.83ms，V2重尾；[原始会话+裁决](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s5_l3/l3_result.json) |
| O-01 核心优化循环 | **针对L1/L2/L3证据调优正式V2**；优先定位SDMA到达等待、Core0/IB控制、MM/SDMA交叠、布局拷贝 | 研发侧改正式Kernel/薄接入，现场按门禁复测；每轮候选SHA+变更变量+收益归档 | **R2_DEVICE_PARTIAL / NZ_SCREEN_NEGATIVE / CONFIRM_INCONCLUSIVE** | R2修复`allow_internal_format=True,jit_compile=False`后真实NZ路由两Rank×M2048/4096各6组PASS，151pytest+27子测试PASS；真NZ Screen M2048 V2−22.69%、M4096 −21.01%，但8场Confirm Stock漂移79.675%/663.527%不合格（`audit_sha256=null`）；五Case机制+短Profiler完成但INNER_PHASE_UNKNOWN；Epoch首场被外部占用挡住，NOT_RUN。[现场最终报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_FINAL_REPORT_20261027.md) |
| L4-01 正式TTFT | **原始4096/2048、TP2、并发10、50请求的Stock/V2/V2/Stock四会话**；检查Prefill收益能否传递到客户端 | 现场；**仅L3_GO且另获许可**；原始逐请求TTFT/故障、p50/p90/p99、同版身份与实际路由 | **L4_NOT_RUN_BY_DESIGN（现D4）** | 本轮L3明确未见收益，依合同不执行4096/2048/TP2/并发10/50请求；没有新正式TTFT |
| L4-02 四层传导 | **解释为何L1算子微秒变化没有等额传到L4**；拆出布局、重叠、双Rank与队列调度，标注不可观测段 | 研发侧；L1/L2/L3/L4同版及各工况可比，分段证据链与不足项逐一列出 | **L1-L3_ANALYZED / L4_ABSENT** | 当前只有行主序U1 +21%与真实NZ站点负368/512µs、L3负26~28ms；L4缺失，不得填TTFT因果拆分 |
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
| B-01 / P0 | **新D4 L1测成row-major Stock +21%，但L2真实Serving Stock NZ链更快**；不能以旧U1基线签模型GO | 新L1 8场ABBA合法但行主序F.linear；真实NZ-MM仅752µs、D4约2511µs；L2 V2每站点慢368/512µs | P0改U1正式主比较对象为Stock_NZ+HCCL、保留row-major仅诊断，复测同卡ABBA |
| B-02 / P0 | L3已获得无Profiler且有L2校准的Forward→Logits设备Event；当前融合无正向收益 | L3 ABBA S1/F1/F2/S2，V2慢26.1/27.7ms、Stock漂移2.83ms；Event不是正式4096/2048 TTFT | P0当前候选直接转O-01调优；优化后重测同版本L3，有GO再L4 |
| B-03 / P0安全 | **Gate E/507014与首写潜在静默损坏**，会污染数据且阻止生产启用 | 历史故障待根治；与本轮L3收益判断分开，但异常即STOP | 故障取证与正确性回归；MemFabric团队协作/现场；正式发布前关闭 |
| B-04 / P0准入 | 43520 Die7/MCU历史告警及卡级真实独占 | R1本批16384 dev2/3 健康且独占，初始tbe环境误配500001修复后Smoke/KFC/首写正常，无507014/507899 | 下批仍需重新G0选卡、维护旧Die7禁选和独立环境恢复，不可因一次Smoke成功注销Gate E |
| B-05 / P1 | **历史2026-10-21 Host SO SHA记载冲突**，误导跨轮归因 | 报告6e9dc2d6 vs provenance 88d39285；未解释 | 核查旧原始安装/Trace存在性；不影响开展新同场实验；研发侧/现场 |
| B-07 / P0环境漂移 | 其它租户共享环境漂移风险 | R2三次尝试发现dev2/dev3外部轮换占用；attempt2非独占并整树异常消失；attempt3 Confirm跨场污染/E15A因显存不足STOP（27.33<38.92GiB），外部进程与异常关联强但进程终止机制未定因 | **下一进场前要求管理员/调度方提供稳定的16384双Die独占租约和全程占用监视**，G0瞬间空闲不等于完整ABBA+四Serve期间独占；保留自动停止，禁止逐场挑干净样本、驱逐他人/换卡 |
| B-06 / P1治理 | A1工作分支含主线已有的旧版治理文件，存在误覆盖风险 | PR#9 Draft，基底早于ADR-015/016主线合入 | 后续代码PR按新主线比较、选择性合入，经两轮审查；研发侧 |

## 5. 实机批次、结果表（新实验开始后在这里逐行追加）

**已完成BATCH-R0现场只读设备状态/CPU取证**，但**没有任何新设备计算或性能批次**。以下首行是已完成的只读批次，后续各行仍NOT_RUN。

| run_id | 日期 | 阶段/任务ID | 实际设备/Rank | Git SHA + SO/KVER指纹 | Stock/V2/负载 | 状态 | 结果摘要 | 原始证据路径 / 审核人 |
|---|---|---|---|---|---|---|---|---|
| a1-r2-batch-20261027-001 | 2026-10-27 | NZ路由+Smoke/Screen/Confirm、五Case机制与Epoch四Serve尝试（3次attempt） | 16384(dev2/3)初入场健康，但长时间独占失效 | 运行HEAD`2718f41dc`，补充归档`e1c3b3d5`; Host`1dc46d7d`,D4`2a19390c`,Plugin`8eb65de9`,KVER11`25e16b0a` | 151CPU+27subtests PASS；4×6原生NZ-MM/HCCL路由PASS；M2048双RankSmoke PASS；Screen有效；Confirm八场原始采集但污染；5Case机制两Rank+短Profiler；E15A启动失败、后续Epoch全未测 | **R2_PARTIAL_PASS / SCREEN_V2_SLOWER / CONFIRM_INCONCLUSIVE_DRIFT / INNER_PHASE_UNKNOWN / EPOCH_NOT_RUN** | 真NZ Screen M2048 Stock1057.474 vs D4 1367.878µs、M4096 2006.133 vs2539.831µs；Confirm跨场漂移M2048 79.675%、M4096 663.527%；Formal2636/2617µs、Loop4 3234/3213µs，内部分段未知；E15A可用27.33GiB少于所需38.92GiB，外部显存占用；没有新L3/L4 | [R2现场最终报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_FINAL_REPORT_20261027.md)、[STOP报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_BATCH_STOP_REPORT_20261027.md)、[Screen JSON](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_screen/u1_analysis.json)、[Confirm JSON](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_confirm/u1_analysis.json)、[机制JSON](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/mechanism/diag_assessment.json) |
| a1-r1-batch-20261027-001 | 2026-10-27 | G0+S1 Smoke+L1 Screen/Confirm+L2+L3 | 16384(dev2,3)同卡独占 | A1 HEAD运行`26d7ea3f`+事件冲刷修复后归档`ff885ff5`; Host`1dc46d7d`,D4`2a19390c`,Plugin`8eb65de9`,KVER`25e16b0a` | U1双RankSmoke+8场ABBA+L2两臂短Profiler+L3四Serve/30请求/场 | **R1_SAFE_PASS / L1_ROW_MAJOR_ONLY / L2_VERIFIED / L3_NO_GAIN / L4_NOT_RUN** | M2048 U1+22.5%、M4096 U1+21.1%只对row-major；真实Stock NZ快于D4 368/512µs每站点；L3 V2慢26.12/27.70ms，重尾；无本轮正式TTFT | [原始报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/A1_R1_BATCH_REPORT_20261027.md)、[复盘](A1_R1_实机复盘与优化优先级_20261027.md)、[L3原始JSON](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/s5_l3/l3_result.json) |
| a1-r0-20261027-001 | 2026-10-27 | E-01/E-02/E-03，G-01只读部分 | 16384(dev2,3)为候选，未初始化 | A1`235de9f6`；修复SO`1dc46d7d`归档 /现装`88d39285`，Device在装`b8ca96f0` | 原仓CPU/四卡只读/溯源 | **R0_ACCEPTED / G0_PENDING** | CPU34/34、pytest88+27通过，修复源`5802f550`及两份Host SO在位；未部署、未运行NPU | [现场证据](https://github.com/10wvw01/vllm-ascend/tree/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r0_reentry_20261027)；[审核](A1_BATCH_R0_ACCEPTANCE_20261027.md) |
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

| 2026-10-27 | **服务器环境重进场与MemFabric物理号修复取证准备（执行前状态）** | 当时仅方案准备，尚无现场证据，旧G0作废；镜像7分支没物理映射修复；该历史准备行保留原时间节点，不作为当前最新状态 | [A1准备文档](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_REENTRY_BUILD_DEPLOY_AND_BATCH_GATE_20261027.md)、[MemFabric Draft PR#1](https://github.com/10wvw01/memfabric_hybrid/pull/1) | 用户委托OpenCode仅执行BATCH-R0：只读盘点 + CPU + 历史源码/二进制溯源 + 隔离编译/恢复方案；回传审核后才评估部署/Smoke |

| 2026-10-27 | **BATCH-R0完成并审查：CPU_PASS / MemFabric原修复找回** | A1原仓unittest34/34、pytest88+27通过；四卡只读：16384(dev2,3)候选；从本地`mf-fix`找回未推送`5802f550`及两份历史修复SO`1dc46d7d`；共享Host仍`88d39285`且D4尚未在装。**无NPU性能/Kernel执行** | [R0审核](A1_BATCH_R0_ACCEPTANCE_20261027.md)、[原始证据索引](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r0_reentry_20261027/evidence_index.json) | 准备真实修复提交长期归档，审查隔离装载优先R1方案；在用户另行许可前不覆盖生产/不启动M2048Smoke |

| 2026-10-27 | **R1源码恢复及隔离装载准备** | 已将原始`5802f550`补丁3个目标源码Git Blob字节级恢复并合入MemFabric镜像main`0670f3c7`；另仓语义不同备用Draft已关闭。A1新增R1静态装载检查器/CPU拒绝性测试及[R1分段合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R1_ISOLATED_LOADER_PREFLIGHT_AND_SMOKE_GATE_20261027.md)，**新工具原仓CPU与服务器R1A均未运行；没有实际部署/Device Smoke** | [MemFabric修复PR](https://github.com/10wvw01/memfabric_hybrid_310p/pull/1)、[R1实施分支PR#9](https://github.com/10wvw01/vllm-ascend/pull/9) | 下一次委托OpenCode执行**一次高密度CPU/只读R1A**（四卡重巡、真实动态路径、私有SO staging与恢复计划），审核其证据后再独立决定R1B首次Smoke |

| 2026-10-27 | **用户批准一次进场连续做M2048Smoke→L1→L2→L3（L3_GO后条件L4）** | A1实施分支新增Smoke硬闸、L2逐Rank真实40层Trace完整性校验、实验开关的NPU Event Forward→Logits采样、L3四场无Profiler ABBA分析与纯CPU负向测试、单会话Eager采集入口；**原仓CPU/新实机尚未执行，新增测量语义仍需现场Trace校准** | [单批执行合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R1_ONE_BATCH_THROUGH_L3_20261027.md)、[A1 PR#9](https://github.com/10wvw01/vllm-ascend/pull/9) | OpenCode一次批量执行，任一实际身份/正确性/设备异常STOP；各阶段证据独立归档，禁止将Profiler或Kernel累加替代L3设备Step |

| 2026-10-27 | **A1-R1单批S0→L3全部实机完成，现D4 L3没有收益** | G0私有装载+108CPU PASS，M2048双Rank Smoke PASS，L1 8场U1 row-major +21～22%但基线不等价于Serving；L2真实NZ Stock胜V2、40层已归档；L3 V2慢26.12/27.70ms并有周期性尾部；L4依合同未跑 | [实机报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/A1_R1_BATCH_REPORT_20261027.md)、[独立复盘](A1_R1_实机复盘与优化优先级_20261027.md) | 继续O-01；先修Stock_NZ基线、量化MM/SDMA时间及长尾来源，实机新候选再做L1/L2/L3，不提前L4 |

| 2026-10-27 | **R2 P0三项同时开展：真实NZ-Stock基线+2.5ms分段机制+周期长尾** | A1实施分支U1大M默认NZ、Row-major仅诊断；新增独立NZ MatMul/HCCL原始Profiler路由核验、R1 L2 20份站点外部预算和L3长尾复算。F1/F2异常均在index0/6/12/19/25，接近MemFabric默认15s Epoch周期（根因未证实）；**新增原仓CPU/实际310P R2试验NOT_RUN** | [R2同批实验合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R2_NZ_STOCK_25MS_周期长尾联合诊断合同_20261027.md)、[P0离线复算](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/analysis/a1_r1_p0_offline_reaudit_20261027.json) | 实机先重新G0+原仓CPU，再一批NZ路由和8场ABBA、D4机制诊断、Epoch 15s/5s单变量长尾验证；任异常STOP，不启L4 |

| 2026-10-27 | **R2三项P0联合实机工具完成静态准备，等待同批OpenCode实机** | A1已补全真NZ CANN kernel_details.csv四件取证、HCCL不可重复配对、8场U1、D4五Case/短Profiler、Epoch15→5→5→15四Serve、双Die实际`epochBudgetS`运行日志核验、原仓CPU负向UT和`a1_r2_one_batch.sh`；两轮静态审查发现并修复SHA子进程环境丢失及预算配置假阳性。**新CPU、G0和设备数据NOT_RUN；正式D4 SO未变** | [R2执行合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R2_NZ_STOCK_25MS_周期长尾联合诊断合同_20261027.md)、[A1 Draft PR#9](https://github.com/10wvw01/vllm-ascend/pull/9) | 在健康独占、真实私有SO/恢复和CPU验证后同一次OpenCode委托执行全部阶段；诊断失败即STOP，不跑L4。结果回传后再决定新Kernel优化变量 |

| 2026-10-27 | **A1-R2实机共三次尝试：真NZ路由打通，性能Screen负向，Confirm污染，Epoch被外部占用挡住** | Attempt1揭露裸torchrun的假NZ通用MatMul+TransData并修启动开关；Attempt3原生`aclnnMatmulWeightNz`确证、M2048Smoke/真实NZ Screen/五Case诊断完成；Confirm 8场漂移超标（不能签GO）、Epoch首场Serving显存不足STOP；151CPU PASS、未改生产D4。环境占用比新的算子实验更需优先解决 | [R2最终报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_FINAL_REPORT_20261027.md)、[性能基线](PERFORMANCE_BASELINES.md) | 先取得真实约80min稳定双Die独占窗口，尽量只补NZ 8场Confirm与Epoch四Serve；期间优先研发正式D4非侵入设备分段探针，后续判断MM/SDMA优先级，仍不执行L4 |

| 2026-10-28 | **A1五层+E1单批实机（工作分支five-arm-dcci）：E1显著快于D4、仍慢于Stock** | run_id `a1-five-arm-20261028-001`；16384独占全程巡检干净、私有装载前后SHA一致；NZ路由4/4、B2/E1 Smoke与十场24份Rank JSON全bit-exact/status=0。配对p50：M2048 Stock/B1/B2/D4/E1=1070.4/2653.8/1647.9/1362.0/**1263.6**µs，M4096=2017.5/5031.2/3092.5/2513.0/**2274.2**µs；**E1 vs D4 +7.79%/+10.50%**（E1仍慢于Stock −15.29%/−11.29%）；总状态`INCONCLUSIVE_DRIFT`（D4 M2048漂移5.685%>5%，stage07周期长尾），M2048速比不签。B2串行负向维持交叠主线；B1慢60%归因staging。L3/L4 NOT_RUN、Stock默认不变 | [五层实机报告](A1_FIVE_ARM_ONDEVICE_REPORT_20261028.md)、[evidence/a1_five_arm_20261028/](../evidence/a1_five_arm_20261028/)、[A1五层合同](A1_五层MM_MemFabric_AR与E1关键路径优化_20261028.md) | E1留作下一优化基线（下一E代：同步消息缩减/冗余清理消除/周期长尾治理）；正式合入待主线审PR；L1对Stock仍不闭合，不启L4 |

| 2026-10-28 | **E3 Chunk2048单批实机（工作分支e3-chunk2048）：M4096 E3-G真实超越E1 +7.19%，真2048 MM被L0C约束击败，单组M2048退化** | run_id `e3-chunk2048-20261028-001`；基准`65837dc4`+BUG修复`3e58ae38`（E3-T首版tiling 256×256 FP32累加器100%占满L0C=256KB，首个真实kernel即双Rank对称AICore异常retCode=0x26；修复basicM=128、singleCoreM=256保持一次256行MM调用语义，UT锁定+71项复跑+重建+Smoke转PASS）。G0 16384独占、NZ路由4/4、G/T双Rank Smoke与8场16份Rank JSON全bit-exact/status=0、二进制前后SHA一致。配对p50：M4096 Stock/E1/**E3-G**/E3-T=2022.6/2271.8/**2119.5**/2267.1µs（E3-G vs E1 **+7.19%/−152.3µs**、vs Stock −4.57%/−97.0µs；E3-T vs E3-G −6.51%）；M2048=1078.4/1222.6/1451.0/1483.9µs（E3-G vs E1 −15.74%：单2048组零组间交叠）。总状态`INCONCLUSIVE_DRIFT`（M2048 E1臂漂移6.19%>5%，E3-G劣于两场E1方向稳健）；M4096全臂≤2.02%稳定为首要参考。下一轮：单组组内"前1024行MM完成即发首包"混合粒度流水、producer DCCI分段、Core0串行链、周期长尾。L2/L3/L4 NOT_RUN、Stock默认不变 | [E3实机报告](A1_E3_CHUNK2048_实机报告_20261028.md)、[evidence/e3_chunk2048_20261028/](../evidence/e3_chunk2048_20261028/)、[性能基线](PERFORMANCE_BASELINES.md) | E3-G是否升为M4096主优化基线待主线裁决；下一E代聚焦混合粒度流水与97µs残差归因；仍不启L4 |

**每轮结束至少更新四处：**①本文件第2节任务状态/结果/下一步；②第5节本轮实测批次、失败与原始证据；③第6节追加一行日期更新；④若事实改变项目当前裁决，同步 [PROJECT_CONTROL.md](PROJECT_CONTROL.md)，有经审核的新性能数值同步 [PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md)，新决策再追加 [DECISIONS.md](DECISIONS.md)。不修改旧实验中的原始数字，纠错使用新记录说明原因。

**下一轮现场交接固定输出：**任务ID/开始前允许执行的阶段、实际HEAD/环境SHA、上一门禁PASS或NO-GO、操作所依据的文档、所有执行结果与异常、完成后真实状态、新结果应更新的文件、下一步是否还需要用户批准。任何“执行中”不得仅凭口头声称而改为PASS；新会话不从聊天记忆恢复状态，而从本总账与主线控制台恢复。

---

**本总账最新裁决（2026-10-27，A1-R1实机审核后）：**R1设备G0/私有装载、原仓CPU108+27、M2048双RankSmoke全部PASS，无新507014/507899。L1按行主序Stock测得M2048+22.5%、M4096+21.1%，**不可代表真实Qwen3.6 Stock**；L2真实NZ Stock约2143/1918µs而D4约2511/2430µs（每Rank分别慢368/512µs每目标站点），L3无Profiler四Serve ABBA双窗口V2慢26.1/27.7ms、有重尾，**当前D4 L3_NO_GAIN，正式L4_NOT_RUN_BY_DESIGN，Stock保持默认**。下一步先修NZ真实基线并做V2内部MM/通信/等待/周期尾部单变量归因，正式发布Gate E与历史507014仍未解除。来源：[实机报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/A1_R1_BATCH_REPORT_20261027.md)、[复盘](A1_R1_实机复盘与优化优先级_20261027.md)。

**本总账最新裁决（2026-10-27，A1-R2三次实机尝试后）：**原R1结论仍成立——当前D4真实Qwen3.6 Prefill L3没有正收益，L4仍未运行；R2发现独立U1 Stock最初并非真正NZ路径，修复Serving同源选路后证实原生MM/HCCL并取得有效Screen（M2048−22.692%、M4096−21.013% V2速度增益），但正式8场Confirm因Stock漂移与外部负载干扰被判`INCONCLUSIVE_DRIFT`；D4五Case+短Profiler对外部耗时取证完成，内部MM/SDMA/IB/Add分段仍`INNER_PHASE_UNKNOWN`；Epoch15/5/5/15未启动有效场次，`EPOCH_CAUSAL_UNKNOWN`；**绝不把Screen筛选数替代正式Confirm**。下一P0是保障物理16384双Die稳定独占窗口，再补Confirm+Epoch；独立设备内部探针开发可并行。正式Stock默认与发布Gate E历史阻塞不变。[R2原始报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_FINAL_REPORT_20261027.md)。

**本总账最新裁决（2026-10-28，A1五层+E1单批实机后）：**工作分支`work/310p-mm-ar-five-arm-dcci-20261028`完成五层完整链十场ABBA（G0独占、NZ路由4/4、B2/E1 Smoke、24份Rank JSON全bit-exact）。**E1（整核DCache+ReadyTree）对D4真实提速+7.79%（M2048）/+10.50%（M4096），方向确立并保留为下一优化基线**；但E1仍慢于原生NZ Stock（−15.29%/−11.29%），L1对Stock不闭合；B2串行结构输给D4交叠（+285.8/+579.5µs）、B1慢约60%为staging代价。总状态`INCONCLUSIVE_DRIFT`（D4 M2048漂移5.685%>5%），M2048速比不签、M4096降级为参考；正式D4/Host零改动、Stock默认不变、L3/L4 NOT_RUN。[五层实机报告](A1_FIVE_ARM_ONDEVICE_REPORT_20261028.md)。
**本总账最新裁决（2026-10-28，E3 Chunk2048单批实机后）：**E3在M4096证明粗粒度2048同步组的固定开销收益（E3-G vs E1 +7.19%，对Stock差距缩半至−4.57%/−97µs，p99更紧）；单组M2048因零组间交叠净亏228µs；"真2048 MM"被dav-2002 L0C=256KB硬限制击败（256×256累加器满载即AICore异常，修复后E3-T仍比E3-G慢6.51%，为单次大调用调度代价、非通信差异）。总状态`INCONCLUSIVE_DRIFT`（M2048 E1臂6.19%，方向稳健）。E3-G留作M4096候选基线待主线裁决；下一轮优先单组组内首包提前流水（混合粒度）、producer DCCI分段与Core0串行链压缩。正式D4/Host/Serving零改动，Stock默认不变，L2/L3/L4 NOT_RUN。[E3实机报告](A1_E3_CHUNK2048_实机报告_20261028.md)。
