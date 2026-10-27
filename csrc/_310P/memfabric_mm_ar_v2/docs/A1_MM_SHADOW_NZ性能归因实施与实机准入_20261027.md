# A1-MM-SHADOW｜V2 D4内置MM与原生NZ-MM同构对照：实现、实机门禁与裁决

> **状态（2026-10-27）：**代码已在`work/310p-mm-ar-large-m-a1-20261026`添加影子Kernel、独立编译、双Rank测试、原始NZ/影子CANN profiler、12场交叉比较和CPU负向UT。**本变更未在310P3上编译/运行，CPU原仓CI待执行，不能称PASS或提前填写MM差值。**保留D4正式Kernel和MemFabric协议原样，Stock仍为Serving默认；A1 PR#9保持Draft。

## 1. 要回答的两个不同问题

1. **同构独立MM效率差**：`Delta_MM_only = p50(V2同构MM影子) − p50(原生aclnnMatmulWeightNz)`。这里是MM-only完整设备Event，不是正式D4运行中的“MM阶段耗时”。
2. **真实融合计算关键路径贡献**：正式D4对原生NZ-MM+HCCL的差额（R2 Screen M4096约+534µs），不能由上述MM-only差额或不同诊断模式直接做减法得出。必须配合正式Kernel的Cube/MTE占用、可用Device计数器、SDMA/IB握手等证据；不支持内部精确计时就写`FORMAL_INNER_MM_UNKNOWN`。

旧`diag0`仅跑单Chunk且占用整个UB，与正式D4的Chunk循环及Add Tile2048预留不等价，因此不能作为正式MM效率参照。本次`src/op_kernel/mf310p_v2_mm_shadow.asc`复制正式`mf310p_v2_kernel.asc`的8Core/Chunk1024、`Matmul<AT,BT,CT,BiasT,tiling1024>`、A ND、B NZ、C ND、SetTensorA/B/OrgShape、IterateAll、`TOTAL_UB_SIZE - 3*4096 - kUBForIB`以及3个Vector Queue预留。只移除IB/signal/wait/Add/DONE；影子 SO 与正式D4 SO不同，**不得直接用于推理或替换其路径**。

## 2. 独立测试分组

| 组 | 原生操作/影子操作 | 正式Event测量覆盖 | 限制 |
|---|---|---|---|
| A native | `F.linear(X,W_NZ)`，开启Serving同源`allow_internal_format=True`及`jit_compile=False` | 原生NZ MM完整操作 | 短Profiler需命中`aclnnMatmulWeightNz`，不存在`TransData`；热态不计NZ格式转换 |
| B1 nd | 影子MM，普通ND GM Tensor结果 | D4同构8核/多ChunkMM | 无MemFabric协议和Add；不等于正式D4内部精确时间 |
| B2 sym | 相同影子MM，写入MemFabric本Rank真实对称C arena | 比B1多的真实对称GM位置条件 | MemFabric建池、首写同步及CPU读回均在Event外；与B1仅可受控比较 |
| C1 sym_clean | B2 + 与正式D4一致的每Chunk`CleanLines` | 清理操作的受控增量 | 有重叠差异，不可将p50差额直接命名正式D4缓存阶段耗时 |
| D formal | 原生NZ-MM+HCCL vs 正式D4（已有U1测试） | 完整链路 | 只在八场ABBA Stock漂移≤5%、HCCL完成独立审计通过时签L1候选 |

M2048与4096、K=N2048、NZ格式29、FP16、同权重随机种子和2套X输入，双Rank分别保留100次有效原始Event。A/B1严格`native→nd→nd→native`×2；B2/C1`sym→sym_clean→sym_clean→sym`，整个运行固定16384双Die、同编译版本。独立进程（Gloo用于双Rank控制同步），**单进程绝不混合HCCL和MemFabric池**。硬件仿真/CPU合成UT不是实机性能结果。

## 3. 新增源码与责任边界

- `src/op_kernel/mf310p_v2_mm_shadow.asc`：只有影子MM和自己的`mf310p_v2_mm_shadow_launch_async`，两次enable0符号预热，输出指针普通GM/真实对称GM均仅使用调用方提供的地址。
- `test/perf/a1_mm_shadow_build.sh`：用冻结CANN9.1 BiSheng`--npu-arch=dav-2002`编译为运行目录下**独立**`libmf310p_v2_mm_shadow.so`，输出源码/头文件/产物SHA；既不覆盖`libmf310p_v2_trial_device.so`也不改变CMake默认目标。
- `test/perf/a1_mm_shadow_profile.py`：A/B1单独短Profiler；每Rank、每M真实`ASCEND_PROFILER_OUTPUT/kernel_details.csv`，至少3个原生`aclnnMatmulWeightNz`或影子`V2MmShadowKernel`，拒绝native的`TransData`路径。
- `test/perf/a1_mm_shadow_arm.py`：每臂独立双Rank进程，不启用HCCL；热态NPU Event与Wall采样、bit-exact真输出，绝不跳过首写保护；B2/C1使用独立私有`mf310p_create`、`mf310p_get_layout`取真实对称arena（Rank0 send_arena、Rank1 recv_arena），D2H仅窗口外做正确性校验。实机首调与IOMMU/布局不支持则原地STOP，不模拟对称内存。
- `test/perf/a1_mm_shadow_campaign.sh`：当前批次G0/物理独占/回滚/原SO SHA硬门后完成构建、8份真实CANN Kernel明细、12次独立Serve-less TP2短会话，任阶段错误保存原始证据STOP。
- `test/perf/a1_mm_shadow_assess.py`：读取所有12场与8份Profiler原始证据，双Rank同序号配对取max，p50/p90/p99、Stock漂移、Clean/对称内存扰动报告；高漂移`INCONCLUSIVE_DRIFT`，不剔除慢样本；**固定`real_full_d4_internal_mm_time=UNKNOWN`**，要求正式D4+HCCL由旧U1独立签核。
- `test/ut/test_a1_mm_shadow_contract.py`：CPU合成数据正负例，检验缺Rank、错SO身份、缺真实CSV、篡改哈希、少100次、漂移>5%和无通信代码隔离。

## 4. 一批310P3实机操作顺序

**设备/数据前置**：当前服务器之前有外部进程轮换占卡，必须先由管理员/调度取得约80分钟稳定双Die独占租约，并**持续只读监视**该卡的其它PID/显存。重新执行本批G0及D4真装载（Host `1dc46d7d…`、Device `2a19390c…`、插件`8eb65de9…`、KVER11 `25e16b0a…`），不得因R2旧PASS而跳过；私有LD_LIBRARY_PATH、只读共享SO的前后SHA与恢复计划须可审计。任何507014/507899/507018、Die/MCU告警、KFC首次写异常、外部进程抢占，STOP，不自动换卡或强杀他人进程。

**步骤**：

1. 在新的私有worktree checkout最新A1 SHA，运行原仓`python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut`，`bash -n`两脚本，`python3 -m py_compile`三个诊断py；任何失败先修源再重新CPU，不带病进入实机。
2. **独立BiSheng编译**影子SO，记录资源、NPU架构、真实导出符号。若`TOTAL_UB_SIZE`/SetLocalWorkspace/符号导出等失败，STOP并修诊断产物，**正式D4 SHA不变**。
3. A/B1分别做双Rank短CANN Profiler实际Kernel路由和无Profiler8场ABBA。每M各Rank20预热+100有效，保存每条Event和CPU wall，两个X变体逐次校验及至少每10次对称D2H正确性证据。
4. B2/C1单独私有MemFabric建池，应用首次写保护，在另一组独立4场ABBA完成同形状热态性能；不能正常建立安全对称arena则标`SYMMETRIC_MEMORY_NO_GO`，不得偷偷换成普通GM。
5. 对全部12场和路由原始CSV运行`a1_mm_shadow_assess.py`。若Stock或任一臂跨场漂移>5%，保留`INCONCLUSIVE_DRIFT`，不挑会话重算。
6. 有稳定连续独占余量，再按已验证的U1`U1_STOCK_LAYOUT=nz`进行独立`large screen→HCCL跨流审核→large confirm`（八场），给`Delta_Full`新证据。这个步骤不自动混入影子MM的数值判断。
7. 对A/B1/D的真实CANN设备Profiler硬件活动及寄存器/管线、Cube/MTE/Scalar/Cache统计（若310P3 CANN9.1支持），交叉看MM算法与资源效率。**活跃时间可重叠，不能简单相加或相减还原正式D4的MM阶段**；支持性不确认就输出`FORMAL_INNER_MM_UNKNOWN`。

**统一入口（示例；变量由现场重新核对）：**

```bash
# 先到物理卡/租约/环境/恢复所有真实证据已审核后再设置以下批准位
export U1_G0_APPROVED=YES U1_EXCLUSIVE_APPROVED=YES
export U1_RESTORE_PLAN_VERIFIED=YES A1_MM_SHADOW_EXCLUSIVE_APPROVED=YES
export ASCEND_RT_VISIBLE_DEVICES=2,3      # 仅示例，重新按实际物理卡确认
export A1_MM_SHADOW_STORE_PORT=18884       # 仅示例，必须验证空闲
# 其它私有绝对路径：ASCEND_HOME_PATH、MEMFABRIC_HYBRID_HOME_PATH、
# U1_DEVICE_SO_PATH、U1_CORE_SO_LIB64、MF_SDMA_ORCH_JSON
bash csrc/_310P/memfabric_mm_ar_v2/test/perf/a1_mm_shadow_campaign.sh \
     /tmp/opencode/a1-mm-shadow-new-run-20261027
```

**出场报告必须给**：原生NZ MM-only p50/p90/p99、影子ND MM-only同形状耗时与`Delta_MM_only`、对称GM/清理扰动、8核Pipe利用率/缓存行为（取得就报，不支持注明）、正式D4 vs NZ+HCCL干净Confirm独立结果以及下一优化变量。内部分段无法精确测量时明确注明，而不是计算虚构的“D4内部MM=shadow的p50”。

**试验许可**：只授权本批安全条件下的诊断及数据归档。不得擅自修改KFC/epoch、覆盖共享Host与Device SO、接入正式Serving默认路由，也不因影子MM加速便直接执行L3/L4。正式真实业务指标仍是Qwen3.6 Prefill TTFT（4096/2048，TP2，batch10），L3正向后再进行L4。
