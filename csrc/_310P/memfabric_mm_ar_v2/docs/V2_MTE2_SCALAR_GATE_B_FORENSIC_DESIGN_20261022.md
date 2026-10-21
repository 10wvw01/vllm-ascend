# V2 MTE2/Scalar 专项 Gate B：真实 4-Chunk 关键路径取证设计（2026-10-22）

> **设计及两轮静态逻辑审视完成；尚未编写新的设备埋点，也未在 310P3 验证任何新时间戳 API。**  
> 基线：\`feat/310p-memfabric-public-api\` 的正式 \`src/op_kernel/mf310p_v2_kernel.asc\`；取证从本分支独立、受控构建，不更改正式默认产物。  
> 前序：\`V2_MTE2_SCALAR_GATE_A_EVIDENCE_AUDIT_20261022.md\`。待现场关闭 Gate A 的加载路径、跨 Rank 校时、拷贝归属三项缺口前，**不启动会改变内核执行语义的测量**。

## 1. 本专项要回答什么

在真实 Qwen3.6 Eager / TP2 / M4096、chunk1024（4 个 chunk，overlap=1）中，V2 设备核 p50 约 2.57ms，Stock 目标 MM+HCCL 约 1.94ms。我们需要定位差额，而不是简单让 MTE2/Scalar “降数值”。

分层问题：
1. MM 各 Core 到达 \`WaitReady\` 是否不均衡？
2. 通信发起与远端邮件就绪之间，究竟是对端 MM、SDMA 排队/执行，还是邮件读可见性在控制关键路径？
3. IB \`WaitReady / ReleaseAdd / WaitAddDone\` 的实际等待作用；
4. \`CleanLines\`、本地/对端 GM 读入、Vector Add、输出写回的贡献；
5. chunk k 的 MM 是否真正与 chunk k−1 的等待/Add 形成有效交叠；首/尾 chunk 以及 \`quiet\`+DONE 成本。

**禁令**：不假定 \`mte2_time\` 等于 \`smem_shm_sdma_wait_at\` 时间，不把 \`scalar_time\` 等于“协议轮询成本”，不把 T0/T1/T2 诊断变体的耗时差当成正式 Kernel 内分段耗时。

## 2. 正式 Kernel 的可检查执行合同（逐源码）

- 8 个 Core 全部做 MM；Core0 唯一执行当前 chunk 的 \`smem_shm_sdma_signal_at\`。
- M=4096，N=K=2048，FP16 ND×NZ，CHUNK=1024，总计 4 chunk；每次通信 chunkBytes = 1024×2048×2 = **4MiB**，全步 16MiB。
- 每 Core 每 Chunk 分配 128 行，生成本端 ownC 的 512KiB 分片（8 核合计 4MiB）。
- 对 chunk \`k=0..3\`：\`mm.IterateAll(ownC[k])\` → \`CleanLines(ownC[k])\` → \`WaitReady\` → Core0 \`signal_at(k)\`。
- \`overlap=1\` 下，\`k>0\` 的当前迭代要等待**上一个** chunk \`k−1\` 的 mail 才 \`ReleaseAdd\` → \`CleanLines(ownPeerC[k−1])\` → VECIN DataCopy / Add / VECOUT DataCopy。
- k=3 循环后处理尾 chunk3；全部完成 \`WaitAddDone\`，Core0 \`quiet_at\`，再双向 \`signal_at(DONE)\` / \`wait_at(DONE)\`。

**逻辑顺序 ≠ 时长/并发图**。特别注意 \`mm.IterateAll\` 的异步计算和 AI Core 多 Pipe 行为：源码语句前后立刻读时钟未必得到真实 MM 设备完成时刻。

源：
- \`src/op_kernel/mf310p_v2_kernel.asc\`
- \`src/op_kernel/mf310p_v2_kernel_common.h\`

## 3. 能力风险：310P3 不可直接套用 GetSystemCycle / PrintTimeStamp

CANN 9.1 公开文档的支持矩阵对“Atlas 推理系列 AI Core”标注 \`GetSystemCycle(ISASI)\` **不支持**；\`PrintTimeStamp\` 亦在该系列标为不支持。310P3 必须按具体 SoC/编译器/头文件进行验证，**默认视为不可用**。不得用 A2/A3 的 \`GetSystemCycle()/50\` 换算或未经确认的寄存器读作为测量结果。

- 官方：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/基础API/工具接口/系统资源与变量/GetSystemCycle_ISASI.md
- 官方：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00002.html

能力选择顺序：
1. **首选不改 Kernel**：复用已有 \`torch_npu.profiler\` 的完整设备时间线、Kernel 硬件 Pipe 指标、同 Stream 事件；检查是否已有设备/指令级 Trace 可以定位源代码位置。
2. 若存在 310P3 **官方支持**的低扰动 kernel 侧计时方法，先在完全独立的最小探针上编译/上机证明单核单调、单位正确、不同 Core 同步属性、空间占用；再提出是否移植到独立诊断 Kernel。
3. 若没有可信时间源，**不强行开发伪时间戳探针**。改成已有诊断变体 + 外部 Profiler 的**受控增量**试验，只报告相对成本/变化趋势，不报告“正式 Kernel 精确 SDMA 等待=xxxus”。
4. 任何插入 \`PipeBarrier<PIPE_ALL>()\` 的尝试都可能破坏 MM/SDMA/Add 的并发关系；只能作为隔离实验，不得用来代表未插桩内核的原生时序。

## 4. 可复用诊断资产审计

| 资产 | 已有功能 | 可回答 | 不能回答 |
|---|---|---|---|
| \`mf310p_v2_diagnostics.asc\` 的 MM-only | 独立 MM 对照（若干 chunk/形状） | 自有 MM 算法独立耗时 | 正式 overlap=1 下真实等待 |
| NoIB / IB / IBDONE / LOOP | 在特定形状下逐级引入协议与循环 | 模块增加后有无性能拐点 | 各模块在正式 4 chunk 中的可加“独占耗时” |
| \`mf310p_v2_trial_diagnostics.cpp\` | 单 Chunk 诊断入口、正确性和首次保护 | 确认诊断合同可运行 | 替代正式 4 chunk 的时间线 |
| G1-C/G1-D 设备飞行记录器 + Host 取证 | 507014/邮件/队列/epoch 的故障证据 | 稳定性异常、邮件可见性候选 | 可靠的 us 级 Kernel 正常工作分段时长 |
| vLLM Profiler（前轮） | 双 Rank kernel_details、PipeUtilization | 2.57ms 总核、设备差异、站点模式 | 内部 IB/mail/SDMA 的专属绝对耗时 |

G1-D 诊断线与 KVER12 及独立 Host/Device 部署相关，和本轮 KVER11 正式测量**不能混合加载**；不得未经静态兼容性审查直接将 G1-D 代码合入本项目正式生产核。

## 5. 若计时能力被证实可用：最小候选事件字典（未实现）

以下只是 \`phase_id\` 规范，不预设时间源或物理输出方式。建议对 4 个 chunk、Core0 和少量代表 Core 记录；原则上每个记录带 \`call_id, rank, core, chunk, phase, sequence, clock_domain, value\`，标明相对/绝对时钟范围。

| phase | 插入位置 | 意义和可观测限制 |
|---|---|---|
| P00 / P01 | \`mm.IterateAll\` 前 / 返回后 | 函数/提交边界；若未确认 Pipe 完成，不等于 GEMM 真正完成 |
| P02 | 本端 \`CleanLines\` 后 | 本端缓存维护完成边界 |
| P03 / P04 | \`WaitReady\` 前 / 后 | Core0 收齐 ready 的等待范围，其他核实际 \`IBSet\` |
| P05 / P06 | Core0 \`signal_at\` 前 / 后 | Host/AI Core signal API 执行区间，**非 SDMA 物理传输时间** |
| P07 / P08 | Core0 \`wait_at(prev)\` 前 / 后 | 上封消息被该核观察到的等待区间；不自动区分远端未投递与本端不可见 |
| P09 / P10 | \`ReleaseAdd\` 前 / 后 | 核间 release 屏障范围 |
| P11 / P12 | \`CleanLines(peerC)\` 前 / 后 | 对端缓存失效动作 |
| P13 / P14 | Add loop 前 / 后 | 读入 + Vector Add + 写回整个块 |
| P15 / P16 | \`WaitAddDone\` 前 / 后 | 核内/跨核 Add 完成屏障 |
| P17 / P18 | Core0 \`quiet_at\` 前 / 后 | flush/quiet API 范围 |
| P19 / P20 | DONE signal/wait 前 / 后 | 尾封协议完成 |

约束：
- **绝不**增加跨核共享的未隔离 GM 写槽；按 rank/call/core/chunk/phase 单写者、64B cache-line 分离并在导出时验证冲突；
- 不触碰 \`ownC/ownPeerC/ibWorkspace/status\` 的实际协议布局、CacheClean 顺序、KVER11 邮件环或 Done 握手；
- 不在关键路径上做 D2H、printf、逐事件 stream/device sync，不把无时钟支持的寄存器猜测当工作方案；
- 精确时戳也不能自行推断“远端邮件已投递但不可见”，必须结合 G1-D Host AICPU 槽位内容观测，另属稳定性取证；
- 做 **诊断 OFF 与 ON 的相同构建/相同真实调用路径** 对照，记录对 Kernel 外部耗时的扰动。改变流水排序的探针自动降级为功能隔离实验。

## 6. 若无支持的时间源：受控隔离矩阵（建议作为最早可执行路线）

按**不修改正式 Kernel**的前提，优先使用独立诊断入口和原有 KVER11 正式库：
1. 固定 M=4096、N=K=2048、chunk1024、TP2，预热后运行完整正式 V2 做锚点（2.57ms 量级须重现）。
2. 独立运行 MM-only 等变体，首先证明输入/权重布局、输出正确性、4 chunk 的可比性；形状或入口不同就单独分类，不外推。
3. 在可保持相同数学与缓存条件时比较 NoIB、IB、IBDONE、LOOP 的运行方向；任意“减项”如果绕过协议和正确性保护，只允许使用隔离诊断 Buffer，不允许作为生产修复，也不允许在真实 Serving 中启用。
4. 每种变体都采集**完成校验和设备 Kernel 总时长**，至少跨两 Rank，短窗口；禁止仅测 Host launch。
5. 输出“哪种阶段/依赖的增量值得深入”，**不提供不可识别的精确阶段 us 分配**。

这里已有诊断多数为**单 Chunk**入口，与正式 4 chunk overlap=1 不同；如果无法达到数据路径等价性，只报告“探针提示”，不能用来断言正式融合 Kernel 的 0.7ms 来自某一调用。

## 7. Gate C（未来上机）最小验收策略：本阶段不执行

- C0 环境门禁：Gate A 现场补证、KVER11 安装完整性、双 die 空闲、无 G1-D v12 残留、正确的正式/诊断库链接；先只加载诊断分支，原版可恢复。
- C1 正确性：OFF/ON 或正式/诊断双 Rank bit-exact，协议状态 0；任一 mail/status 不符立刻停机保全。
- C2 扰动预检：无 Profiler 的原始 Kernel 对照重复，诊断 ON 不得引起不稳定差异；建议以同环境 ABBA 和原始核耗时分布为依据设阈值（例如 Kernel p50 额外开销 >5% 则仅做定性分析，不能当原核性能）。
- C3 单真实 Prefill：原模型 Eager、M4096、1 token 输出、双 Rank 短窗口；若为隔离探针则单独说明不是等价真实 Forward。
- C4 停止条件：507014/507018、协议状态非零、首调失配、非目标 M、设备占用或外部进程、误加载 v12、原始 Trace 不完整，立即停止。不做 Gate E 长压及协议调参。

**跨 Rank 取证属于强依赖事件匹配的问题**：必须配对同代 chunk/tag、校准时钟；没有可验证校时基准，只使用每 Rank 本地时长及事件先后关系。

## 8. 第二轮审视：最危险的三项设计误差

1. **“计时能力”假阳性**：在不支持的 310P3 上误套 A2/A3 的 \`GetSystemCycle()\`，或者把 PipeBarrier 改变的流水当原生性能。对策：产品支持矩阵和独立探针前置 PASS。
2. **通信保护被弱化**：为了隔离某阶段而省略 \`CleanLines\`、IBWait、quiet/DONE，得出快但不可正确执行的假结论。对策：正式核不改，隔离诊断只验证局部代价且不得输入真实 Serving。
3. **Rank 和 Kernel 事件被错误配对**：每 Rank 的 4 chunk 时钟未统一，Host SO 在不同版本间飘移，对照失真。对策：先 Gate A，现场必须有同代配对、SHA 与校时证据。

## 9. Gate B 静态交付与 Go/No-Go

**Gate B 静态方案已完成，但“新设备埋点实现”当前 No-Go**，直到能力矩阵确认 310P3 支持合法的低扰动时间戳，或明确选择不带精确时间戳的隔离诊断方案。

后续 OpenCode 可以在 310P3 优先执行**环境核验 + 现有 Trace 二次分析 + 独立时间能力可行性验证**，不得擅自修改正式 Kernel、KVER、G1-D 设备库，也不得宣称得到了精确 SDMA/IB 毫秒数。

通过 Gate C 后的输出应包括：
- 真实模型相同 M 的 Stock/V2 设备基线；
- 可信的 per-rank、per-chunk 事件（如果能力准入通过）或清晰标注的隔离变体增量；
- 设备核整体开销与潜在回归风险；
- 至少一个能解释约 0.63–0.70ms 的、**可安全消除**的假设及验证路径；若没有，明确 No-Go，不继续无证据调参。
