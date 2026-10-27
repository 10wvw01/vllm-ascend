# A1-E2｜两级Add放行树 + READY栅栏接收缓存预失效

> 2026-10-28；项目唯一治理主线 `dev/310p-memfabric-mm-ar`；短期研发 `work/310p-mm-ar-e2-pipeline-20261028`，从已完成E1实机的`work/310p-mm-ar-five-arm-dcci-20261028`派生。**本文件是实施合同，未经过BiSheng编译/310P3实测，不代表性能结果。**

## 1. 为何直接开发E2而非重跑五层

2026-10-28 E1单批实机：M2048 Stock1070.4 / D4 1362.0 / E1 1263.6µs；M4096 Stock2017.5 / D4 2513.0 / E1 2274.2µs。E1相对D4约+7.79%/+10.50%，仍比真NZ Stock慢15.29%/11.29%。M2048 D4第二会话漂移5.685%，**五层整体状态INCONCLUSIVE_DRIFT**；M4096内部稳定。原始证据在父分支`evidence/a1_five_arm_20261028/`，E1已经完成双Rank短测，不再把本轮宝贵机时用在B1/B2/多轮旧D4。

本轮重在一个新设备核，提供三个候选机制，E1与Stock同批重测：
- **E1对照**：原实机E1源码零修改，使用同一BiSheng版本/优化选项重新构建并编入本次私有SO，保持复现口径。
- **E2-R/1024**：保留E1每包的完整DCache Clean+Invalid与原8→1 ReadyTree；**将每包Core0向其余7核串行ReleaseAdd改成两级ReleaseTree**，Core0只向核1/2/4/6发送，核2/4/6各把放行转递核3/5/7。发送方地址、SDMA顺序、MM/Vector Add、每核读取和DONE语义不改。
- **E2-P/1024**：E2-R基础上引入**一轮通信前的全8核接收DCache预失效 + 双Rank控制面READY消息**。两端所有核清理自己的接收缓存，内部ReadyTree→Core0双向发送并等待对端带专用imm的8B READY→Core0广播本地8核继续运行。保证**任意数据SDMA发起之前，两端各8核已失效接收缓存**；之后同一Kernel的每个Chunk不再执行接收侧`FlushCoreDCache`。所有输出仍只用AscendC Vector `DataCopy`搬到UB、`Add`并写回GM，按照CANN9.1文档此DMA路径不经过Scalar DCache；不允许引入对接收arena的Scalar读取。
- **E2-P/512**：与E2-P/1024一致，SDMA每包2MiB（vs 4MiB），换取更细计算通信流水。M2048四包，M4096八包；多包额外成本是真实测量对象，不预判收益。

以上三个候选共享`mf310p_v2_e2_pipeline.asc`，分别由私有Host模式 `kind=3/4/5`区分；Kernel编译期模板参数`PRE_INVALIDATE`区分同步语义。E1是`kind=2`，不改变原入口。

## 2. READY握手不是“无证据取消DCCI”

前置时序：

```text
每个Rank各8核：receiver whole-DCache Clean+Invalid
                  ↓
            本端 ReadyTree 8→1
                  ↓
Core0分别发出 READY(8B) → 向对端控制信箱写入
                  ↓
Core0分别等待并验证来自对端的 READY(dst,len,imm)
                  ↓
          本端 Core0 ReleaseAdd 1→7 (一次)
                  ↓
           第一Chunk MM / producer flush
                  ↓
            每Chunk ReadyTree / SDMA / Wait / ReleaseTree / Add
                  ↓
           所有AddDone → quiet → 原DONE双向握手
```

**正确性需要满足的关键不变量**：

1. 没有远端READY就没有首个数据SDMA；双方READY必须先发后等，不能双边wait死锁。
2. READY和DONE复用经过验证的8B control目标，但imm不同、顺序不同；每条信件只由Core0消费一次，必须核验`dst,len,imm,status`，错误立即Trap。
3. 本次Kernel内不允许Scalar读取或写入`ownPeerC`，MM仅写入`ownC`，Add通过Vector`DataCopy` DMA读取接收端GM；数据传输完成依旧在Core0收到SDMA对应mail并释放Add后才开始。
4. E2-R模式`PRE_INVALIDATE=false`，继续严格按E1先Wait再接收缓存刷新，是新READY路径的故障隔离对照。
5. 若 READY 消息与SDMA data mail次序或接收Buffer映射不一致、bit-exact失败、507014/507899/507018、KFC/IB死锁、SDMA投递异常，**立即STOP，保留双Rank日志**。不允许退化成跳过首写、减少硬件保护、扩大信号长度的所谓加速。

基础支持信息：CANN9.1 `DataCacheCleanAndInvalid`提供`ENTIRE_DATA_CACHE`并明确提醒全缓存模式可能较慢；DMA`DataCopy`与Scalar DCache语义不同，官方API文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E7%BC%93%E5%AD%98%E6%8E%A7%E5%88%B6/DataCacheCleanAndInvalid.md 。**这些仅证明API语义，不替代实际310P3代码和MemFabric协议正确性验证。**

## 3. 交付文件及复用

- `src/op_kernel/mf310p_v2_e2_pipeline.asc`：独立E2设备核，新增ReleaseTree、kE2TagReady和PRE_INVALIDATE编译期模式，正式D4/E1文件不修改。
- `test/perf/e2_private_host.cpp`：独立SO Host，复用经过验证的trial pool ABI，并保留地址生命周期首写同步与status清零；kind3/4/5。
- `test/perf/e2_build.sh`：BiSheng dav-2002 `E1 + E2 + B2`单独编译私有SO，记录各来源源SHA/Host桥SHA/加载SO SHA和worktree HEAD；不安装或替换共享文件。
- `test/perf/e2_arm.py`：一臂一独立TP2进程组、Stock HCCL与MemFabric池互不共存、真实NZ权重、两个交替输入、FP32 Gloo参考、每次bit-exact及status、无Profiler Event+Wall计时。
- `test/perf/e2_campaign.sh`：G0/持续独占/前后SHA门禁 → 真实NZ路由验证 → 三个E2候选顺序双Rank Smoke → 十场对照。
- `test/perf/e2_assess.py`：固定`stock,e1,e2r,e2p,e2p512,e2p512,e2p,e2r,e1,stock`平衡顺序，每臂两场，每场100测量+20热身，两个Rank逐编号取max，输出p50/p90/p99和漂移。5%场间门限任何一臂失败不重新挑样，整个Confirm降级。
- `test/ut/test_e2_pipeline_contract.py`和`.github/workflows/310p_e2_cpu_only.yml`：纯CPU源/协议及评估器正负合同，非设备通过证明。

## 4. 与Stock/E1严格对照的要求

- 所有形状 TP2 / 单卡双Die / M=2048、4096 / N=K=2048 / FP16 / NZ29，Stock使用真实`aclnnMatmulWeightNz+hcom_allReduce`且无`TransData`。不要把旧row-major Stock的3ms拿来当对照。
- E2-R/1024 vs E1/1024 单变量：ReleaseTree组织；E2-P/1024 vs E2-R/1024 单变量：跨Rank READY和接收缓存维护方式；E2-P/512 vs E2-P/1024：分包/tiling协同结构总差额，**不是纯overlap时间**。
- 真正目标是 **E2完整链同时超过E1和Stock**，首轮先尽可能缩小E1→Stock在M2048的193µs、M4096的257µs差距；不设提前终止研发的ROI阈值。
- 只在双Rank正确、编译身份一致且场间漂移<5%的前提下签E2性能；必要时单独针对长尾做短Trace，所有Profiler样本不得混入无Profiler p50。
- 严禁自行将本候选接入正式Serving/Graph、默认启用融合或对DFlash分支做任何改动。后续L1胜过Stock还需L2目标40站点及L3真实Prefill设备Step验证，稳定正向后才执行L4固定input4096/output2048/TP2/batch10/50请求。

## 5. 现场运行前门禁和回退

E2目前**仅源码已创建、尚无新CANN编译和设备测试**。OpenCode若要在310P3上运行，必须先核主线管理合同`PROJECT_310P_MEMFABRIC.md`、`docs/ENGINEERING_GATES.md`、本文件与原E1现场身份，完成静态第二轮审查/所有CPU用例；源无误才在隔离worktree构建E2私有SO，检查输出符号及manifest，确认主线/任务分支HEAD一致、二进制装载与真实物理Die映射正确。选卡必须重新四卡只读G0，不沿用过去16384 dev2/3无条件视为健康。

执行入口是`bash test/perf/e2_build.sh 全新绝对目录`，然后在确认私人环境变量及持续独占许可后调用`bash test/perf/e2_campaign.sh 全新绝对会话目录 私有build目录`。对旧E1 SO、现有V2 Host、KVER11文件和所有共享系统SO只读记录SHA，绝不覆盖。缓存协议新模式有错误则保留E1工作路径，不继续压力测验。所有原始Rank、CSV、卡状态、首写/编译/符号/Hash报告应提交工作分支。

当前完成状态：E2-R/P源码、Host桥、构建/Runner/评估/CPU合同**已提交；两轮独立静态结构/协议合同检查PASS**（含事件生产消费拓扑、READY先于数据SDMA、Private ABI、512/1024形状及证据矩阵）。GitHub Actions与原仓CPU pytest尚未取得通过记录；CANN BiSheng编译/双Rank Smoke/性能/Serving仍`NOT_RUN`。因此本轮仅可进入现场G0+CPU+编译，不可跳过直接跑性能。
