# 310P3 MM+MemFabric AR｜工程准入与实机交接合同

> **2026-10-30用户最新指令：后续直接在`dev/310p-memfabric-mm-ar`开发、修复、执行实机实验并归档。** 实验从主线固定完整SHA，使用隔离构建/结果目录；不再默认创建工作分支或等待PR合并才允许测试。G0、CPU合同、私有编译、短Smoke、正式性能及故障STOP要求继续有效。以下旧分支流转规定仅作历史背景，以本条为准。

> 所属主线：`10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`。本文件规定验证**顺序**，不宣称当前服务器满足任何门禁。最新事实以 [PROJECT_CONTROL.md](PROJECT_CONTROL.md) 为准。2026-10-25。

## 1. 四个不能混淆的身份

| 层 | 应记录 | 为什么不能互相替代 |
|---|---|---|
| 远端源码 | 仓库/分支/完整提交 SHA；MemFabric 另记其独立源码仓库 SHA | GitHub 改了不代表现场更新 |
| 本地源码 | remote、worktree、HEAD、`git status --porcelain` | 服务器可以有多个 worktree、未提交修复 |
| 构建/安装 | CANN/Bisheng/驱动版本、构建开关、Host/Device SO 的 SHA256、AICPU KVER/加载文件 | 当前实际库可能不是源码 HEAD 对应编译物 |
| 真实进程 | Python 导入位置、进程 maps 的 SO、物理设备映射、运行模式、融合命中 Trace | 构建与装载成功也不代表走了 V2 |

MemFabric 的既有 GitCode origin、自有 GitHub 仓库、服务器部署位置不得凭名称自动相互替换。现行独立定制源码管理参考 `10wvw01/memfabric_hybrid_310p`；若实机 remote/安装记录冲突，以现场证据为准先报告，再决定任何变更。**不得自动覆盖服务器 origin。**

## 2. 改动分级门禁

| 级别 | 对象 | 本地/CI准入 | 310P3实机要求 | 主线合入 |
|---|---|---|---|---|
| G-DOC | 项目控制、文档、纯CPU检查工具 | 文档路径/语义核对、独立单元测试 | **不需要**；不得写设备 PASS | 静态审查合格即可 |
| G-TEST | 未改变正式部署路径的诊断/验证工具 | CPU测试及风险审查，锁定测试合同 | 实际使用前完成 G0 和轻量 smoke | 测试资产可合入，实机状态写 NOT_RUN |
| G-KERNEL | 正式 Kernel、Host ABI、通信、LinearMethod/Runner/Graph 路由 | 两轮静态深审，构建合同、测试与回退方案明确 | G0 → 编译 → 双Rank正确性及冷首写 → 负载与长尾 → L1完整算子对比 | 实机测试结束并保全证据后决定，不能静默启用 Serving |
| G-RELEASE | 让融合成为默认或对外发布 | 上述门禁全部满足 | Eager、Graph真实命中、生命周期/异步场景、长时间稳定性与故障恢复 | 用户批准；Gate E/507014及首写风险必须关闭 |

`python3 -m unittest discover -s tools/310p_mm_ar -p 'test_project_gate.py'` 和 `python3 tools/310p_mm_ar/project_gate.py --static` **只验证项目管理约束**，不是算子 CPU 合同、CANN 编译、NPU 实测或 Graph 验收。GitHub Actions 需要真实运行并通过才算 CI PASS。

## 3. 上机前必须先出具 G0 / GO-NO-GO

1. **只读环境盘点**：`git remote -v`、`git worktree list --porcelain`、每个 worktree 的 `git status --porcelain`、`git rev-parse HEAD`，记录双Die可用性、现有进程、端口和设备占用。不执行 checkout、reset、clean、install。
2. **二进制身份**：检查源码 SHA、`sha256sum` 两处可能装载的 Host 库与 Device 库、AICPU KVER、进程 `/proc/*/maps`、Python 模块路径；不以 `version.info` 代替真实装载版本。
3. **测试隔离**：已有现场作业、未跟踪工件或诊断快照先备份并保留；新工作使用隔离 worktree/进程/输出目录。需要部署或替换 SO 时先列出变更前后 SHA、备份、恢复步骤并请用户批准。
4. **静态/CPU准入**：两轮源码审查、现有 `csrc/_310P/memfabric_mm_ar_v2/test/ut` pytest、目标场景的隔离脚本/参数与预期 FAIL 处理。U1 尚未执行的 CPU 测试不能借本次治理检查器 PASS 冒充。
5. **设备执行顺序**：G0 → 最小双Rank功能冒烟（含首写、尾 M、状态）→ 必要的正式性能；遇 507014、507899、静默数据错误、SO 不一致或未知环境立刻停止并留证，不做自动绕过。

**本轮主线治理任务是 G-DOC，完全不需要310P服务器，也未触碰正式 Kernel/Host/Serving。**

## 4. 验收结果只认两本账

**L1 完整算子**：310P3 单卡双Die、TP2、同FP16/M/K/N/权重格式/热态、真实Stock MM+HCCL AR vs 正式V2 MM+MemFabric AR，ABBA双Rank样本 `max(rank0[i], rank1[i])` 统计 p50/p90/p99，含布局转换、发起和等待的完整真实成本。必须附原始样本与设备、二进制身份；旧D4的“内部+6.26%”不代表超越Stock。

**L2 真实Qwen**：普通模式固定输入4096/输出2048/TP2/batch上限10/Eager/50请求/ignore-eos；投机模式独立对比**同配置** Stock vs 融合，含草稿接受率、Target Forward 数、实际 physical/padded M、融合层命中/Graph Replay。解释单算子收益如何或为何不能传导 TTFT/TPOT/ITL/吞吐。Graph只产出正确结果但未命中V2，不计融合 PASS。

## 5. 每次现场回传与恢复

至少保存：
- `run_id`、主线/任务分支完整 SHA、MemFabric 来源 SHA、`git status`、模型与输入参数；
- CANN/驱动、物理设备映射、Host 两处安装库与 Device SO SHA256、AICPU KVER，运行进程真实装载路径；
- 指定的构建及运行命令、环境开关、Stock/V2路由与 Graph capture/replay 真正命中证据；
- Rank0/Rank1原始 JSON/日志、ABBA 顺序和失败样本；报告中的每个 PASS/FAIL/UNKNOWN 对应证据位置；
- 是否恢复全部环境、恢复后每份 SO SHA256、尚存服务/后台内核、是否有未解决问题。

结果先提交**自有仓库**的短期任务分支，经过代码/证据核对后将报告、[PROJECT_CONTROL.md](PROJECT_CONTROL.md)、有新性能时的 [PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md)、涉及决策时的 [DECISIONS.md](DECISIONS.md) **同批合入唯一主线**。绝不向 HaonanWu1 仓库推送或建 PR。

## 6. 保护措施及边界

当前分支 GitHub 默认分支仍是 `main`，本项目不改变它。GitHub 主线保护（禁止强推/删除、要求 PR 和上述 CI 状态检查、限制直接写入）需要仓库所有者在 GitHub Rulesets 或 Branch protection 中单独启用并验证；**文件中声明规则不等于服务器已执行规则**。CI的纯CPU检查也不能替代人工评审、设备证据及发布审批。

## 7. 2026-10-26新增项目级原则：四卡动态择优、故障换卡与KFC自动部署

- 每次G0先对**全部4张310P3卡**只读巡检，确认同卡双Die、双Die+MCU健康、空闲、租户进程及Rank物理映射，再选择一张；不把历史dev0/1、dev6/7当作固定试验对象。
- 某卡G0健康/占用失败，记录原因后可切换另一张健康空闲卡并**重新G0**；执行期间出现507014/507899、错算、IB/KFC 0x2a或设备超时则先STOP保留现场，**不自动轮换**。两臂ABBA只可同卡同Die同版本，换卡重新建立run_id及热态。
- 换卡需核对`ASCEND_RT_VISIBLE_DEVICES`逻辑/物理DeviceID、MemFabric双Host SO已审核的`aclrtMemSetAccess`物理号修复、设备SO、KVER11和回滚备份。KFC v11在**得到授权的首次MemFabric初始化**按需自动部署并核验各Die，G0只读阶段不运行手工KFC/建池/替换SO，禁止假helper已部署或绕过。
- 详细流程和历史取证：[四卡动态选卡/KFC准入规范](310P3_四卡动态选卡与KFC自动部署准入规范_20261026.md)、[2026-10-23换卡实证](MEMFABRIC_310P_KFC_DEPLOY_AND_DEVICE_PIVOT_20261023.md)。本章为G-DOC，不代表新的NPU GO。
