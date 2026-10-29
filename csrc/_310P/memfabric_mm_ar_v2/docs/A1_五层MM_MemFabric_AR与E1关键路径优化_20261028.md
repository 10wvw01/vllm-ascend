# A1｜大M五层实验与E1缓存/同步关键路径优化（2026-10-28）

> 唯一权威主线：`dev/310p-memfabric-mm-ar`；本轮**短期实施分支**：`work/310p-mm-ar-five-arm-dcci-20261028`。本说明是代码实现和待验证实验合同，**不构成310P3设备GO、正确性PASS或性能收益**。正式D4 Device SO及Serving/Stock默认全部不变。

## 1. 技术判断：建立真正完整MM后AR的中间基线

用户明确要求的“串行”是全部M完成之后才开始一次逻辑AllReduce，不是D4的`overlap=0`逐Chunk MM→SDMA→Add。

| 层 | 真实执行路径 | 目标 | 等价性边界 |
|---|---|---|---|
| A/stock | 原生`F.linear(X,W_NZ)`一次完整MM → FP16 HCCL SUM | 完整算子唯一最终性能基线 | 必须CANN命中`aclnnMatmulWeightNz`且没有`TransData` |
| B1/native+MF | **相同原生完整NZ MM** → 生产者流同步 → 实际stage到对称池 → MemFabric SDMA/FP16 Add | 更换通信方案的完整真实代价 | 沿用T2已验证数据一致性修复：两次牺牲stage+一次有效stage、CacheClean与DONE均计时，不能冒充“纯MemFabric带宽” |
| B2/fullMM+MF | 自定义8核`Matmul<ND,NZ,ND>`完成所有M（内部1024 tile）→ 每核发送区CleanLines → **全体生产核只会合一次** → MemFabric完整逻辑AR（硬件每包≤4MiB，内部可分包） | 摆脱每Chunk“MM、IB、SDMA、Add交替”后，自研MM+MF串行效率 | **没有任何SDMA在全部MM完成前发出**；B2与C不是仅重排overlap标志，故差额是结构总影响 |
| C/D4 | 正式冻结Tile2048：每Chunk MM→CleanLines→IB→发送SDMA，同时等待上一Chunk并Add | 今天的正式融合比较对象 | `chunk1024, overlap=1`，不改Device SO |
| E1/optimized | 与C相同数学计算、1024 Chunk、8核/Tile2048/消息时序、IB事件总依赖、Credit/Wait/DONE；发送端及收件端逐64B DCCI改为**每核整DCache一次性Clean+Invalid**，并把Core0每Chunk的7次直接IB Ready等待改为4次树形汇聚 | **实打实的设备内关键路径优化**：每Chunk每核省去多千次标量缓存行扫描，有机会提早IB Ready/缩短等待与Add之前的阻塞 | 是独立候选Kernel，未证明Cache粒度变化的真实性能/稳定性；**若更慢或错算，保留D4且继续迭代E，不取消五层方案** |

注意B2中“完整MM”是**先完成整个M的计算阶段**，不要求禁用硬件必要的内部tiling；在310P3上全量M4096输出16MiB，而每条已验证的MemFabric包只到4MiB，不能把“一次逻辑AR”误写成“单条16MiB SDMA”。

## 2. 已经实现的源代码

- `src/op_kernel/mf310p_v2_b2_full_mm_ar.asc`：独立B2设备核。与D4保持MM几何、8核、NZ权重、DCCI、Add2048和最终quiet/DONE。原D4每Chunk一次`WaitReady`，B2改成完整MM完成后**仅一次生产端WaitReady**；其后才进入发送/接收/归约循环。每个SDMA分包的Add后执行8核`WaitAddDone`会合，防止Core0在工作核消费上包`ReleaseAdd`事件之前重用同一IB槽；整个MM阶段仍然只有一次`WaitReady`。
- `src/op_kernel/mf310p_v2_dcci_entire_candidate.asc`：E1独立设备核。D4正文复制后精确替换3个DCCI调用点（发送端1处、接收端2处），替换为每核`ENTIRE_DATA_CACHE`。将`WaitReady`的7→1扇入改为Core0等待1/2/4/6四核、Core2/4/6各等待其后一核的**两级IB Ready树**；`ReleaseAdd`/`WaitAddDone`、SDMA signal/wait、QUIET、DONE顺序保留；不能把“减少缓存维护时长”吹成“删除IB或Wait”。实际IB/Wait的时间缩短为待测假设。
- `test/perf/five_arm_private_host.cpp`：私有Host ABI读取**来自相同主线源码**的现有`mf310p_v2_trial`布局，以独立B2/E1设备符号发射；每个候选首写和输出地址变化保留设备同步防护。**绝不装入Serving默认路径**。
- `test/perf/five_arm_build_e.sh`：BiSheng CANN9.1/dav-2002将两个设备核链接进**隔离私有SO**，单独G++编译私有Host SO，记录主线/候选源及SO SHA。**不覆盖正式D4/插件/V1/Host/KVER**。
- `test/perf/five_arm_nz_route.py`：单独短Profiler CANN原始CSV，双Rank×M2048/4096验证`aclnnMatmulWeightNz`和`hcom_allReduce`各≥3次、无TransData；核实后才开始其它采样。
- `test/perf/five_arm_arm.py`：一臂一进程组、Stock独立HCCL、MF独立Gloo+私有池；真实8核B2/E1；每次交替两套输入，**每次**完整数值与FP32 Gloo再转FP16的参考位精确；独立Event计时、Wall、status，真正双Rank各100条以上样本、20热身。
- `test/perf/five_arm_campaign.sh`：G0/独占/身份/禁绕过前置，先测真实NZ路由，B2/E1先Smoke双Rank，再执行**stock,b1,b2,d4,e1,e1,d4,b2,b1,stock**平衡十场，固定输出目录、逐会话日志和前后SO SHA。
- `test/perf/five_arm_assess.py`：按第i个样本取`max(rank0[i],rank1[i])`的双Rank结束时间，再统计p50/p90/p99；五臂各有两场；任一臂同形状场间漂移>5%为`INCONCLUSIVE_DRIFT`、不挑场复算；同时计算E1/Stock和E1/D4完整链加速率，拒绝混淆`B2−C`与“纯overlap节约”。
- `test/ut/test_five_arm_code_contract.py`：源码级安全/等价范围和评估器合成正负单测。

## 3. 为什么E1具有实际优化可能，但不能先宣称成功

A1-MM-SHADOW 2026-10-27有效run2：M4096影子MM仅比原生NZ-MM慢约19.4µs，而每Chunk对称发送区DCCI受控增量约139.6µs（不同于正式D4真实可回收时长）。这些说明DCCI粒度很可能值得重做；当前D4逐64B循环，对1024行、每行4096B、8核而言，每Chunk每核**8192次**DCCI调用；每核整DCache操作理论上可减少Scalar指令数。但CANN9.1官方文档明确`ENTIRE_DATA_CACHE`本身可能较慢，单次对整体DCache的清理不能当免费；E1必须实测。

**关键路径假设**：生产端DCCI缩短→8核更早全部Ready→Core0更早signal→收件端Wait更早返回；收件端DCCI缩短→Add更早开始→下个Chunk的MM更早发起。多点收敛会共同改变真实critical path；**不是**“139.6µs×2×4直接相加”，也**不是**去掉可见性守护创造无效加速。ReadyTree增加3条核间等待的分布式依赖，是否较原7次串行等待更快必须用实机确认。E1是缓存+IB联合优化首版；若收益不足，下一E代聚焦真正缩减同步消息/按可见性证明进一步消除冗余清理。

## 4. 五臂测试与Go/No-Go纪律

必须先在普通CPU环境完成 `pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut`、`python3 -m py_compile`相关Python和 `bash -n`两条Shell，人工**两轮**审查B2与E1。当前环境没有CANN 9.1 BiSheng/310P3，**静态文件检查不等于已编译/实测**。

实机由OpenCode在310P3执行前先读唯一主线项目入口、`ENGINEERING_GATES.md`、旧A1安全runbook；核实最新Git HEAD、同卡双Die物理映射、是否外部租户占用、设备Alarms、KVER11、实际装载Host/D4/Plugin SHA、LD_LIBRARY_PATH优先级、KFC首次初始化/首写、回滚及已有其他工作树；任何状态未知先STOP，不改现有环境。确认符合G0后：

1. 从本短分支独立私有worktree读取源码、CPU准入通过；以`five_arm_build_e.sh`构建B2/E1 Device SO及Host bridge，按manifest验证SHA；**不用生产插件重新安装**。
2. 先G0、真实NZ路由，再B2/E1小样本双RankSmoke（先检查首写、status、错算和hang）；之后十场对照。CANN Profiler只运行在专用路由证明，正式十场Event**不启Profiler**。两个Rank同序号CPU Gloo对齐，不跨设备时钟相减。
3. 在实验中保持卡级只读持续占用监测；外部占卡/507014/507899/507018、KFC/IB/错算/死锁/二进制身份变化须立即停测、保留现场，不自动杀他人、换卡、绕过安全标志或剔除异常样本。
4. 原始CSV、双Rank JSON、Host日志、manifest、前后SO SHA、卡健康和原始分析JSON提交证据到本短分支；**未完成连续独占或有效路由，不签L1结论**。

构建入口：`bash csrc/_310P/memfabric_mm_ar_v2/test/perf/five_arm_build_e.sh /tmp/opencode/five-arm-build-new`，必须使用全新目录、合法CANN/MF源码头。运行入口：

```bash
# G0完成后由现场人员填写真实环境与选卡；绝不盲用历史dev2/3
export A1_FIVE_ARM_G0_PASSED=YES A1_FIVE_ARM_CARD_EXCLUSIVE=YES
export FIVE_ARM_STORE_PORT=18884  # 示例：现场须查端口真实空闲
# ASCEND_RT_VISIBLE_DEVICES、V2_HOST_LIB、U1_*、MF_SDMA_ORCH_JSON须经现场实证
bash csrc/_310P/memfabric_mm_ar_v2/test/perf/five_arm_campaign.sh \
  /tmp/opencode/five-arm-campaign-new /tmp/opencode/five-arm-build-new
```

上述不是“已批准上机”，只是可复用步骤。E1静态检查后、CANN实际编译完成、真实设备G0准入确认，才能允许跑Smoke；短Smoke通过后才准入十场性能。正式Kernel没有因本优化改变，生产/Graph/Serving不默认启用E1。

## 5. 性能决策

- 若`E1<A`：大M内核层候选成功，但仍须L2真实站点和L3无Profiler真实Prefill Step稳定变快才能正式跑L4。
- 若`E1<C`但`E1>=A`：保留E1为下一优化基线，继续压协议/调度，不声称比Stock快。
- 若`E1>=C`：**不要强行宣布整DCache更快**，保留现有D4并转向下一E候选；五层B1/B2照常保留用于归因。
- 若B1比A慢：原因可能是host sync/staging/多包AR，不能等同为MemFabric固有带宽慢。
- 若B2比C快：说明“先完整MM后AR”可能胜过当前Chunk流水，必须继续收缩交叠协议成本，而不是无条件增加深度。
- **最终项目成功仍是Qwen3.6真实Prefill TTFT；旧R2/影子证据、新五层器件Event与客户端请求TTFT不能互相冒充。**

## 6. 当前真实状态（2026-10-28实机后更新）

| 项 | 状态 |
|---|---|
| 五层合同与B2/E1源码 | 已创建并通过两轮静态深审；dav-2002隔离编译成功（候选SO `72b868c7…`） |
| 正式D4 Device SO、Host/Serving/Stock默认 | 不修改（前后SHA一致，已核实） |
| CPU语法/单测 | **PASS**：pytest 62项、py_compile、bash -n |
| B2/E1 dav-2002编译、TP2数值Smoke | **PASS**（双Rank×M2048/4096 bit-exact、status=0） |
| 五层真实NZ路由与十场ABBA | **已执行**：路由证明4/4有效；十场完成、无STOP；总状态`INCONCLUSIVE_DRIFT`（D4 M2048场间漂移5.685%>5%） |
| E1相对Stock/D4性能收益 | E1 vs D4 **+7.79%（M2048）/+10.50%（M4096）**；E1 vs Stock **−15.29%/−11.29%**（仍慢于Stock，L1不签） |
| L3/L4 Qwen真实业务 | NOT_RUN |

实机报告与全部原始证据：[A1_FIVE_ARM_ONDEVICE_REPORT_20261028.md](A1_FIVE_ARM_ONDEVICE_REPORT_20261028.md)、[evidence/a1_five_arm_20261028/](../evidence/a1_five_arm_20261028/)。
