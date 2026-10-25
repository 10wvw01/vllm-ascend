# 310P3 单卡双 Die MM + MemFabric AllReduce 融合算子｜项目唯一入口

> **权威研发主线：`dev/310p-memfabric-mm-ar`。** 本文件及本分支下的 `PROJECT_CONTROL.md` 是当前状态的唯一权威入口；其他 `feat/`、`perf/`、`diag/`、`docs/` 分支为历史/取证/待评审资产，不得按名称推断最新状态。更新日：2026-10-25（Asia/Taipei）。
>
> **请新 ChatGPT 会话、OpenCode 及新接手开发人员先读这里，再读下列链接，不要仅凭聊天记忆或旧版 PLAN.md 制定计划。**

## 从这里恢复上下文（推荐阅读顺序）

1. [项目控制台：当前结论、技术合同、最新决策、下一步、环境状态](csrc/_310P/memfabric_mm_ar_v2/docs/PROJECT_CONTROL.md)——**当前状态只在这里维护**。
2. [工作分支与合入治理](csrc/_310P/memfabric_mm_ar_v2/docs/BRANCH_GOVERNANCE.md)——唯一主线、实验分支归档、代码来源与禁止事项。
3. [统一性能基线与证据索引](csrc/_310P/memfabric_mm_ar_v2/docs/PERFORMANCE_BASELINES.md)——严格区分单算子 V2 内部收益、完整 V2 vs Stock、模型实际传导。
4. [R2 V2 Runner无Triton全量任务锁定](csrc/_310P/memfabric_mm_ar_v2/docs/V2_RUNNER_TRITON_FREE_MASTER_PLAN_20261025.md)——R2-00～R2-21、测试/产物/验收/止损，不再逐报错临时扩展。
5. [项目决策记录 ADR](csrc/_310P/memfabric_mm_ar_v2/docs/DECISIONS.md)——已经确定的原则及被废弃的方向。
6. [DFlash K7 Stock/V2四象限实机准入](csrc/_310P/memfabric_mm_ar_v2/docs/DFLASH_STOCK_V2_R1_GO_GATE_20261025.md)——先Stock Eager，再V2 Eager；Graph受控研究。
7. [工程准入与310P3实机交接门禁](csrc/_310P/memfabric_mm_ar_v2/docs/ENGINEERING_GATES.md)——区分纯CPU、Kernel、发布级验证；锁定源码/二进制/装载三重身份。
8. [310P 融合算子工程入口](csrc/_310P/memfabric_mm_ar_v2/docs/README.md) 与 [目录内自动执行约束](csrc/_310P/memfabric_mm_ar_v2/AGENTS.md)。

## 唯一模型与业务场景（2026-10-25 用户重新锁定；高于一般模型兼容性推断）

- **推理框架**：本仓 `10wvw01/vllm-ascend`，冻结上游 `vllm@0.27.1 / 6e448d0`；不是抽象的任意GPU V2。
- **硬件**：Ascend **310P3 单卡双 Die，TP=2、PP=1**。
- **唯一 Target 模型 ID**：**`Eco-Tech/Qwen3.6-35B-A3B-w8a8`**（[ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8)），ModelSlim/Ascend **W8A8**；现场历史Target路径`/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`，需对本地`config.json`/量化配置和ModelScope ID再核对，不可直接认为路径即版本身份证据。
- **指定负载**：输入 **4096 tokens**、输出 **2048 tokens**、**batch/并发=10**、启用 **DFlash K=7**。研发验收沿用 **50 requests**、Eager主基线、Stock独立对照；Graph为单独R3阶段。
- **DFlash Draft（与Target不是同一个模型）**：历史现场`/home/models/Qwen/Qwen3.6-35B-A3B-DFlash`、5 Sliding + 1 Full；身份/配置现场只读核验。
- 详细不可漂移合同和“310P专属DFlash究竟会不会走Triton”的分路径源码核查见 [目标场景与DFlash实路径审计](csrc/_310P/memfabric_mm_ar_v2/docs/R2_TARGET_MODEL_DFLASH_ROUTE_20261025.md)。**优先确认模型/具体Runner/实际可达函数，不能凭通用GPU源码里有Triton就继续扩充兼容代码。**

## 不可偏离的当前目标

- **Ascend 310P3 单卡双 Die / TP=2**；同 Kernel 完成 FP16 MM + MemFabric SDMA AllReduce + Add，Stock MM+HCCL 是唯一有效的完整算子最终性能对照。
- **大 M 必须持续向硬件可证实的极限优化**，不再每轮讨论“值不值得做”；每次输出完整算子 vs Stock，以及 Qwen 集成真实收益及差距原因。
- **小 M 专攻单 Kernel 带来的固定开销消减**，覆盖普通 Decode 和 DFlash 投机验证的实际 M；旧 T1 M=10 已经严格 NO-GO，不能复活其错误的 +20% 结论。
- **与 Qwen3.6 / DFlash 投机推理集成同步演进**。指定外部只读参照：`HaonanWu1/vllm-ascend:dev_v24/feat-310p-dflash`；**绝不向该仓库提交代码、推送分支或发 PR**；全部工作只落在我们的 `10wvw01/vllm-ascend`。
- 生产/Serving **Stock 默认不变**。任何未完成实机正确性、路由真实性、Graph/生命周期与稳定性验证的融合候选不能启用为默认。

## 项目状态速览

| 事项 | 状态 |
|---|---|
| D4 Tile2048 正式大 M Kernel | V2 内部相对前版本 +6.26%（不是相对 Stock），保留基线 |
| 4096 输入/2048 输出/TP2/batch10/Eager 模型 | SEL-01：V2 vs Stock 输出速率中位 +1.12%，方向不一致、未达目标 |
| 小 M（M=10） | F1-C：T1 425.2µs vs 原生 Stock 308.2µs，T1 等效速率约 −27.5%；NO-GO |
| F2 Chunk 缩小 | 256、512 均不优于1024；保留 D4 Chunk1024 / Overlap1 |
| MemFabric 长稳与 Graph | 507014 Gate E 尚未根治；Graph 不能仅依据输出正确就宣称真正命中融合 |
| R1 DFlash K7 | R1-B证据已选择性归档；现场G0 PASS、Stock Eager Draft混合注意力门禁FAIL，V2/Graph NOT_RUN；310P专属V2 Triton验证已研发性豁免，但V2 Worker未移植，仍非实机GO |
| U1 统一诊断工具 | 已纳入 GitHub 纯CPU源码/合成数据合同回归；**U1真实设备运行仍NOT_RUN**，不是研发主目标 |
| 统一主线 | **本分支**；已有[专用CI](https://github.com/10wvw01/vllm-ascend/actions/runs/40126962427)：治理6/6 + V2源码合同54/54 PASS；正式Kernel未因治理改变 |

## 每次新会话请这样开始

> 阅读 `10wvw01/vllm-ascend` 的 `dev/310p-memfabric-mm-ar:PROJECT_310P_MEMFABRIC.md` 和其链接的 `PROJECT_CONTROL.md`、`DECISIONS.md`、`PERFORMANCE_BASELINES.md`、`BRANCH_GOVERNANCE.md`；核查远程主线 HEAD 和未合入变更。**先报告当前有效结论和下一步动作，再继续我本次的需求。禁止把历史实验报告当作新决策，禁止改动 HaonanWu1 仓库。** 如文档与代码冲突，用源码和新证据定位冲突，并同步更新控制台，不能自行静默改变方向。

## 变更纪律

- 每个有实质意义的工作：在本主线或由其派生的短命任务分支完成，审查后回到本主线；不要把新的实验分支当长期主线。
- 每次完成实机、性能决策、接口变更或外部 DFlash 基线变化，**必须同一批提交更新** `PROJECT_CONTROL.md`，必要时补 ADR/性能报告/分支表。未更新项目控制台的任务视为**未完成归档**。
- 不把文档的“预计收益”“静态通过”“本地测试通过”写成实机 PASS；环境 SO、KVER11 和运行时真正命中的 Kernel 都须有证据。

### 自研DFlash保密边界与唯一模型选路（2026-10-25）

用户确认Draft是自研模型，受保密条款约束。**不得索取、上传、归档任何Draft权重、源码、原始config、逐层架构、训练资料**。新增只读脱敏枚举工具`tools/310p_mm_ar/r2_target_route_probe.py`与 [最小披露/真实路由审计合同](csrc/_310P/memfabric_mm_ar_v2/docs/R2_TARGET_ROUTE_MIN_DISCLOSURE_20261025.md)，只输出固定路径枚举，不做设备/动态import、不采集敏感材料。静态S事实部分已确认，实际运行时R还未观测；在配置与函数可达性闭环前**暂停新增泛化V2/Triton兼容算子**。唯一Target、310P3 TP2、4096/2048/batch10、DFlash K7和Stock对照不变。
