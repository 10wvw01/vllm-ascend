# 主线A｜A1 CPU 验收与受控设备基线交接记录（2026-10-26）

> **只沿融合主线，不涉及V2 Runner/DFlash/Hybrid状态机重建。** A1 工作分支 `work/310p-mm-ar-large-m-a1-20261026`，PR#9 Draft。310P3 实机G0/Smoke/Screen/Confirm/Profiler均**NOT_RUN**，没有新增Stock/V2延迟数据。

## 1. 本期A1 CPU测试状态：实际执行与证据分开

- **已实际执行**：本助手本地容器上，以GitHub源码对应的 `build_report` 判定逻辑重建隔离Python副本，运行20项纯stdlib合成合同测试，结果 `Ran 20 tests ... OK`。覆盖：M1024路由、M2048/4096逐形状判定、Screen绝不授予GO、受审计Confirm中的单形状性能退化、伪ABBA/伪p90拒绝、身份更换、非法/NaN/负耗时、资料互不混算。**执行的是独立重建验证副本，不是直接从仓库checkout的原始测试源文件；不能标注整个A1原仓CPU验收PASS**。
- **本次修复**：A1正式 `a1_large_m_assess.py` 中的 `L1_complete_op_confirmed` 过去只看Confirm有无audit SHA，可能把M2048退化也标为确认。现必须满足实际Serving阈值下 M2048和M4096 **分别**得到 `OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS` 且审计证据存在；同时增加`L1_independent_route_evidence_audited` 独立字段，分开表示“取证审核过”和“算子性能达标”。已增3类回归：一形状NO_GO不能给全局GO、虚假的高延迟/长尾、虚假ABBA速度。
- **未完成**：GitHub PR#9尚无Actions运行；完整仓库的 `python3 -m unittest` / `python -m pytest` / `python -m py_compile` 尚无独立执行日志；原 U1 的 `evaluate()` 原始双Rank文件级集成也待复验。**验收裁决：A1核心判定算法已通过隔离CPU测试；项目原仓CPU TEST=NOT_RUN、STATIC-GO=未签发。**

## 2. G0设备只读取证工具：不触发任何Kernel

新增 `test/perf/a1_g0_readonly.py` 和纯stdlib `test/ut/test_a1_g0_readonly.py`。脚本只读取：

- 仓库Git HEAD与tracked-tree clean；两个可见物理设备ID。
- D4 Device SO、Python插件SO、MemFabric主/双路径SO、KVER11 JSON的真实文件SHA256，按冻结U1常量逐一比对；所有路径只读。
- 启动变量中的首写/KFC绕过标志，即使设为`0`也视为危险；`npu-smi info`快照及基础命令是否存在。
- 为避免构造隐式运行时加载，**不会导入torch/torch_npu/vllm插件**、不会自动检查真实Python加载SO路径，也**不证明物理双Die同卡/独占、在运行时真实加载哪个SO或KVER AICPU注册成功**。

即使静态SHA完全一致，工具仍输出`G0_APPROVED=false`、`SMOKE_APPROVED=false`；只有独立G0人工/运维取证能签发批准。脚本不设置任何`U1_*_APPROVED`变量、不执行`torchrun`、不部署替换文件、不修改设备配置。

## 3. OpenCode第一阶段：仅CPU + G0只读取证（无需启动服务）

**服务器工作顺序（核对路径再操作）**：

```bash
# 先在310P服务器选择隔离工作树；不覆盖在跑的Stock服务/主分支
git -C /path/to/vllm-ascend fetch origin work/310p-mm-ar-large-m-a1-20261026
# 在本次隔离工作树中 checkout 对应的A1 SHA；记录实际git rev-parse HEAD
# 不要从已废止的旧R2 Runner分支cherry-pick任何文件
PERF=csrc/_310P/memfabric_mm_ar_v2/test/perf
UT=csrc/_310P/memfabric_mm_ar_v2/test/ut
python3 -m py_compile "$PERF/a1_large_m_assess.py" "$PERF/a1_g0_readonly.py"
python3 -m unittest discover -s "$UT" -p 'test_a1_*.py' -v
# CPU-only全套已有U1分析合同（先扫描名称，再执行）
python3 -m pytest -q "$UT"
bash -n "$PERF/u1_run.sh"
# 以下只读取证，输出必须是全新的绝对路径；先确认文件输出位置不覆盖
python3 "$PERF/a1_g0_readonly.py" --out /独立取证目录/a1_g0_readonly.json
```

**报告回传**：代码HEAD及分支、两个CPU测试的完整stdout/stderr/退出码、pytest结果、完整`a1_g0_readonly.json`（敏感设备序列等信息可只在服务器存证）、SO/KVER和真实Plugin加载路径的独立检查结果、双Die同卡独占证据、可回滚部署备份、KFC状态、设备健康和针对507014/507899的停测流程。

本阶段**不设置**`U1_G0_APPROVED/U1_EXCLUSIVE_APPROVED/U1_RESTORE_PLAN_VERIFIED`，也**不调用**`u1_run.sh`。发生SHA/插件/设备占用不符、CPU TEST失败、异常日志，立即STOP并保全证据；不得为通过门禁而伪造SHA或禁用断言。

## 4. 第二阶段：受控大M Stock/V2基线（须第一阶段核验后单独解锁）

1. **M2048双Rank Smoke**：仅确认Stock FP16 F.linear+HCCL与正式Tile2048 V2+MemFabric的正确性、SO身份、通信隔离与硬件健康。只在G0明确通过、设备独占/恢复可用、且用户/运维明确授权后由OpenCode设置批准标记再执行。
2. **大M Screen**：沿用`u1_run.sh <绝对新目录> large screen`，M1024仅诊断、M2048/4096为正式路由；严格归档两个Rank原始sample、manifest SHA、Event+Wall、站点和切换顺序；重算`a1_large_m_assess.py`，Screen绝不给GO。
3. **独立路由/跨Stream审计**：当前Stock HCCL NPU Event是否覆盖真正异步通信尾部未确认；这项不闭环不能认定V2比Stock更快。
4. **Confirm**：Screen和真实HCCL时间线经独立审核后，再单独申请ABBA Confirm。每M两个ABBA窗口、p50/p90/p99和双Rank时间配对，不计入任何失败后重跑“幸存样本”。

**本轮交付的是静态/CPU判定补强和G0读证工具，不是新的硬件性能结果。** 以前D4 Tile1024→2048的6.26%只属于V2内部；最新Tile2048完整算子相对Stock M2048/4096仍UNKNOWN，通信等待与IB时间仍待实机安全取证。

## 5. 2026-10-26新增：四卡动态选卡及KFC自动部署硬门禁（ADR-014）

**新的执行入口**：[四卡动态选卡与KFC自动部署规范](310P3_四卡动态选卡与KFC自动部署准入规范_20261026.md)，该规范已写入唯一主线的ADR-014、ENGINEERING_GATES和PROJECT_CONTROL。本段属于旧A1工作包的补充，覆盖此前默认使用dev6/7的历史测试记录。

1. 新一轮A1实机开始前必须先对**服务器全部四张310P3卡**进行只读健康/占用巡检。每次只挑两个Die同卡、双Die及MCU状态健康、独占且无租户冲突的一张；原43520 dev6/7曾有Alarm，未提供消警证据则不选。G0卡级异常可换另一空闲卡，但必须重新生成该卡G0、完整物理DeviceID映射与新的run_id。
2. 在非0/1物理映射（历史如2/3、4/5、6/7）下，不只设置`ASCEND_RT_VISIBLE_DEVICES`，还需确认MemFabric修复后的`aclrtMemSetAccess`物理号版本，双Host SO安装路径及真实加载SHA、设备SO和KVER11。
3. **本轮只读G0不能部署KFC**。在**之后独立授权的首次M2048短Smoke**初始化时由MemFabric自己探测/自动部署v11编排核，检查每个Die核可加载；禁止遗留`MF_SDMA_ORCH_DEPLOY_HELPER`伪rc3、KFC手工删除/覆盖与非批准常驻模式。
4. Stock/V2 Screen/Confirm与ABBA必须同一物理卡、同一对Die、相同已审核SO与KVER。任何一次ABBA中途换卡都会使原ABBA样本失效，重新开实验；运行中507014/507899/错算/KFC 0x2a先STOP保全故障而非自动换卡。
5. OpenCode输出`card_selection.json`或同等四卡清单、物理卡ID/Rank映射及淘汰原因。无可用卡时停止，不因CPU PASS或单卡异常自行签发Smoke GO。

来源：主线 [2026-10-23 已实证的MemFabric部署与换卡档案](MEMFABRIC_310P_KFC_DEPLOY_AND_DEVICE_PIVOT_20261023.md)。**本条没有新的选卡实机测量、没有在新卡部署KFC、没有授予Kernel运行许可。**
