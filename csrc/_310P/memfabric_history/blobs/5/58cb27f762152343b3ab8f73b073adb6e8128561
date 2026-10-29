# V2 真实 Qwen3.6 Prefill 性能归因报告（V2_PREFILL_ATTRIBUTION_REPORT_20261021.md）

> 执行：OpenCode @ 310P3（dav-2002），分支 `perf/310p-v2-prefill-attribution-20261021`
> （自 `feat/310p-memfabric-public-api` 创建，vllm_ascend/ 插件代码与基线零差异；
> 诊断分支的 G1-C/G1-D kernel 埋点**不在**本轮构建中——正式算子与通信协议零修改）。
> 机器：卡 5664 双 die（dev0/dev1），CANN 9.1.0，MemFabric wgm-dev-310p 1.2.0（KVER=11）。
> 脱敏证据包：`../evidence/v2_prefill_attribution_20261021/`（下称"证据包"）；
> 原始双 Rank Trace 仅存服务器 `/tmp/opencode/v2_prefill_captures/`。
> 关联前序结论：`V2_SYNC_ASYNC_EXECUTION_REPORT.md`（算子级 W1/W4 反转）、
> `V2_G3_QWEN_E2E_REPORT.md` §7.2（Eager serve 级 −3.51% 吞吐未归因）。

## 0. 一句话结论

**V2 在真实 Qwen3.6 Prefill 中不是"收益被稀释"，而是每站点净变慢：融合 Kernel 本体
（p50 2573us）比它替换的 Stock MM+HCCL AR（795+1149=1944us）慢 32%，叠加新增的输入
contiguous 拷贝（~66us/站点），40 个站点合计每步 +31ms；因目标站点仅占 Prefill 步的
3.3%，端到端表现为 TTFT +1.3~1.4%——与 G3 Eager A/B 观测一致，并被本轮 L3 回归
（无 Profiler、原始 4096/2048/50/10 配置）复证：吞吐 −1.10%、TTFT 全分位 +2.1~2.4%、
50/50 零失败。算子级微基准的"正收益"来自其 Stock 基线（逐调用同步 W1、连续排队 W4）
都比真实 Serving 的 Stock 条件更差；真实 Serving 中 HCCL AR 以无竞争、间隔 ~60ms 的
最优工况运行，V2 的串行融合管线没有跑赢它。**

## 1. 实验设计与方法

| 项 | 设定 |
|---|---|
| 服务端 | G3 已确认的生产参数原样使用（`tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh` 内嵌命令）：TP=2、ascend 量化、fp16、max-num-batched-tokens 4096、chunked prefill、async-scheduling、`--enforce-eager`（本轮不研究 Graph） |
| 开关 | Stock = 全部 MemFabric 变量 unset；V2 = `ENABLE_MEMFABRIC_MM_AR=1` + `MM_AR_V2_SERVE=1` + `V2_CHUNK_ROWS=1024` + `V2_OUTPUT_SLOTS=4` + store/96MB/KVER11 JSON（与 G3 完全一致） |
| 负载（采集） | 同一 4096-token 固定输入（`[64]*4096`）、`max_tokens=1`、并发 1 串行；6 次预热 + 5 次窗口请求 |
| Profiler | vLLM `--profiler-config`（torch_npu profiler，with_stack=false），`/start_profile`→5 请求→`/stop_profile`；双 Rank ascend_pt 原始导出 + 离线 `analyse()` |
| 计量方法 | **不做重叠 Kernel 耗时简单累加**。以三类独立证据三角定位：(a) 端到端请求时延；(b) 目标站点锚定的步周期（每步第 1 个目标站点到下一步第 1 个目标站点）；(c) 逐站点设备时间轴（kernel_details.csv，含 Input Shapes/Stream/Wait） |
| 停机条件 | 507014 / 数值错误 / 执行路径不符 → 立即停止保全证据（本轮未触发） |

## 2. 基线与真实性核验（全部 PASS）

| 检查项 | 结果 | 证据 |
|---|---|---|
| 代码基线 | `vllm_ascend/` 与 `feat/310p-memfabric-public-api` 零差异；`.so` 以双旗标从该源重建，`nm` 无 g1c/g1d 埋点符号（0 处），`memfabric_mm_ar_v2_allreduce` op 注册成功 | 证据包 `provenance.json` |
| 环境不变 | KVER11 JSON `25e16b0a…`、Host `libmf_hybm_core.so 6e9dc2d6…`（与 sync-async 轮一致）；/dev/shm 2G；dev0/1 空闲独占 | `provenance.json` |
| 层选择 | V2 臂双 Rank 合计 **80/80** 处 `Select MemFabricMmArV2LinearMethod310`（o_proj 10 + out_proj 30，每 Rank 40） | `captures/v2_serve_log_excerpt.txt` |
| 实际 Forward M | **4096**（单波，非分块）：目标 MM `Input Shapes "4096,2048;…"`、输出 `"4096,2048"`；对齐类（4096%1024==0，直通波） | `sites/*.csv` |
| 路由执行 | 窗口内 V2 臂 **0 个** Stock 路径目标 MM（形状签名完全消失），40×V2TrialKernel/步/Rank；Stock 臂 40×(MatMulV3+HCCL)/步/Rank | §4 站点统计 |
| 首调校验 | 双 Rank `first fused call verified bit-exact vs stock (M=4096, aligned=True)`，发生于预热请求 1（窗口开始前）；窗口内无校验税 | 日志节选 14:33:22 |
| 排除初始化/建池/预热 | 首次请求 71.29s（建池+深度扫描+首调校验）；窗口从第 7 次请求开始；预热末 5 次时延已平稳（2.400~2.408s） | `captures/*_requests.json` |
| Profiler 污染检查 | 双臂同配置同负载；持久化 `HybmSdmaOrchEpoch`（AICPU 编排核）与 stop 导出期 3.47s 设备同步已从分析中剔除（证据包 `evidence_index.md` §4） | 证据包 |

## 3. 端到端时延（含 Profiler，双臂同条件）

| 臂 | 预热（1..6） | 窗口（5 次） | 窗口均值 |
|---|---|---|---|
| Stock | 2.600 / 2.372 / 2.374 / 2.372 / 2.372 / 2.370 | 2.3784 / 2.3776 / 2.3760 / 2.3857 / 2.3723 | **2.378 s** |
| V2 | 71.29（建池）/ 2.408 / 2.401 / 2.405 / 2.401 / 2.403 | 2.4824 / 2.4168 / 2.4099 / 2.4105 / 2.4120 | **2.426 s** |

- 稳态差（剔除 V2 窗口首请求）：**+34ms（+1.4%）**；V2 窗口首请求 +104ms 中约 74ms
  为 §6 的一次性双 Rank 停顿。
- 注：单 Serve 内 5 样本的端到端差本身超出 serve 重启噪声带边缘（±0.5~0.8%），
  但与下文设备级 200×2 站点测量（分布完全不重叠）和步周期（4 步独立一致）互证后
  判为真实效应。

## 4. 设备侧归因（核心）

### 4.1 每步站点经济学（Rank0，5 步平均；Rank1 括号）

| 量 | Stock | V2 | Δ |
|---|---|---|---|
| 目标 MM（MatMulV3, [4096,2048]×[2048,2048]） | p50 **795us**，31.8ms/步（r1: 759us，30.3ms） | — | — |
| 目标 HCCL AR（hcom_allReduce, 16.8MB/rank） | p50 **1149us**，46.0ms/步（r1: 1512us，62.9ms） | — | — |
| 融合 Kernel（V2TrialKernel\<1024\>） | — | p50 **2573us**，105.2ms/步（r1: 2672us，109.6ms） | — |
| 输入 contiguous 拷贝（V2 分支 `x.contiguous()`） | 无（aclnnMatmulWeightNz 直接消费 strided 输入） | 199/199 稳态站点前有 ~16.8MB 同流 D2D MemcopyAsync（p50 66us，容量/时延与 [4096,2048] fp16 拷贝吻合），3.6ms/步 | +3.6ms |
| **目标站点合计/步** | **77.9ms**（r1: 93.2ms） | **108.8ms**（r1: 113.2ms） | **+30.9ms（r1: +20.0ms）** |
| 站点后气泡 | 0/200（下个 kernel 紧贴 AR 结束） | 0/199（下个 kernel 紧贴融合核结束） | 0 |

（V2 kernel 合计已剔除 §6 的一次性 74ms 异常；含之为 120.1ms/步。）

**单层关键路径对照（稳态，Rank0，us，时间轴近似比例）**：

```text
Stock（一层，~59ms 周期）:
  |←──── MoE/GDN/注意力等 ~55ms ────→|← MM 795 →|← AR 1149 →|← 紧贴下层 →|
                                        [MatMulV3]  [hcom_allReduce]

V2（一层，~60ms 周期）:
  |←──── MoE/GDN/注意力等 ~55ms ────→|[cp 66][←── V2TrialKernel 2573 ──→]|← 紧贴下层 →|
                                     [memcpy] [融合: 数学459+搬入等待1711+控制873+…]
```

双 Rank：r0/r1 站点结构相同；r1 的 Stock AR（1512us）与 r0（1149us）之差为快 Rank
在 HCCL 内等待慢 Rank 的到达（两路径都以各自方式吸收 ~0.1~0.4ms 偏斜）；
V2 融合核 r1 p50 2672 vs r0 2573（差 −145us 量级），步周期双 Rank 同步移动
（r0 +31.3ms / r1 +31.8ms）。

### 4.2 步周期（目标站点锚定，免受重叠累加谬误）

| 臂 | Rank0 各步周期 (ms) | 稳态均值 | Δ |
|---|---|---|---|
| Stock | 2366.9 / 2359.9 / 2371.0 / 2359.8 | **2364.4** | — |
| V2 | 2473.4* / 2395.6 / 2397.3 / 2394.3 | **2395.7**（*首步含 74ms 异常） | **+31.3ms** |

步周期差 +31.3ms ≈ 站点差 +30.9ms ≈ 端到端差 +34ms（余量为 API/调度噪声）。
**三层证据闭合。**

### 4.3 相邻算子对照（控制变量，确认无二次效应）

| 相邻 Kernel（每步合计，Rank0） | Stock | V2 | Δ |
|---|---|---|---|
| MoE QuantGroupedMatmulDequant（80/步） | 647.2ms | 648.6ms | +0.2% |
| MoeGatingTopKSoftmax（160/步） | 284.7ms | 285.2ms | +0.2% |
| GDN ChunkGatedDeltaRule*（30/步） | 268.5ms | 267.2ms | −0.5% |
| 非目标 MatmulWeightNz（181/步） | 150.9ms | 149.5ms | −0.9% |
| RmsNorm（191/步） | 97.1ms | 97.1ms | 0.0% |
| CausalConv1d / FlashAttention prefill | 89.6 / 77.4ms | 89.6 / 77.5ms | 0.0% |
| **非目标 HCCL AR（42/步）** | **48.4ms**（r1: 51.7） | **52.0ms**（r1: 50.7） | 混合 ±，无系统性变化 |

MoE/GDN/注意力/Norm 全部 ±1% 内持平：**V2 的影响被完全隔离在 40 个目标站点**，
未污染其余 96.7% 的计算，也未与保留的 42 个 HCCL AR 产生可测的资源竞争。

### 4.4 融合 Kernel 内部拆解（kernel_details 硬件计数器，p50，Rank0）

| 计数器 | V2TrialKernel | Stock MatMulV3 | 解读 |
|---|---|---|---|
| **mac_time（Cube 数学）** | **458.9us** | **456.6us** | **两路径 GEMM 数学时间相同** |
| mte2_time（搬入/等待） | **1711.4us** | 335.9us | +1375us：跨 die SDMA 到达等待+读入 |
| scalar_time（标量控制） | **872.8us** | 186.5us | +686us：协议轮询/地址/分块控制 |
| mte3_time（搬出） | 191.1us | 91.5us | +100us |
| vec_time | 127.4us | 194.9us | −68us |
| Duration | 2573.3us | 795.1us | — |
| cube_utilization | 88.6% | 82.0% | — |

**融合核的 +1778us 净增不来自数学，而来自传输与控制**（mte2/scalar/mte3 三项
合计 2.78ms，相互重叠后净增 1.78ms）。对照：Stock 的同等职能 = MM 自身搬入
336us + HCCL AR 1149us ≈ 1.49ms。即 V2 的自研 SDMA+协议通路比 HCCL ring 在该
16.8MB 载荷上贵 ~0.6ms，且全部串行在关键路径上。

### 4.5 为什么算子级"正收益"没有传导（机理）

| 条件 | Stock MM+AR 实测 | V2 融合核实测 | V2 相对 |
|---|---|---|---|
| 微基准 W1（逐调用同步，sync-async 轮） | 3.29ms/调用（含 ~1.3ms 同步税） | 2.72ms | **快 17~22%** |
| 微基准 W4（4 深连续异步） | 2.53ms/调用（AR 背靠背竞争） | 2.68ms | 慢 5~6% |
| **真实 Serving（本文）** | **1.94ms/站点**（MM 0.80 + AR 1.15，AR 间隔 ~60ms 无竞争） | **2.57ms**（与其微基准值一致，环境不敏感） | **慢 32%** |

1. 微基准的 Stock 基线在两种窗口下都被"惩罚"：W1 的显式同步、W4 的 AR 连发竞争。
   真实 Prefill 中 40 个 AR 被注意力/MoE 计算天然隔开 ~60ms，HCCL ring 以最优工况
   运行（1.15ms/16.8MB）；Eager 异步提交使 Stock 无每调用同步税。
2. V2 融合核是自定速串行管线（自有 GEMM + SDMA 分块流水 + IB 等待 + 协议），时延
   几乎不随环境变化（W1≈W4≈Serving ≈ 2.6~2.7ms）——它没有可以从"环境变好"中
   获益的自由度。
3. 因此"算子级加速"是基线选择的产品：在最接近真实条件的对比里，V2 每站点净慢
   ~0.63ms + 0.07ms 拷贝。
4. 收益天花板本身也极低：40 站点仅占步的 3.3%（即便整段删除也只省这么多）；
   一个传输效率追平 HCCL 的理想融合核每站点至多省 ~1ms，即 **Prefill 侧 TTFT
   收益的现实上限 ~1.5~2%**——而当前实现是 −1.4%。

## 5. Host 侧观察（次要，非瓶颈）

- V2 臂 `_C_ascend::memfabric_mm_ar_v2_allreduce` Host p50 56.2ms：**不是** Host 计算
  开销——Trace 显示调用内 58ms 无任何子 API/子算子，是 Host 领先设备 ~350ms（约 6 层）
  后在提交队列反压上的停靠。Stock 臂 Host 同样被反压节流，只是停靠点分布在
  aclnnNeg（均值 10.3ms）/SelfAttention（16.4ms）/aclnnInplaceZero（3.1ms）等 API。
  **双臂 Host 均为设备节奏所限，不构成 V2 的额外回归项**；enqueue 本身（memset+launch
  ~200us）远小于 kernel 时延。
- V2 分支 Python 侧每站点新增 `x.contiguous()` 检查与拷贝（§4.1 的 66us 设备拷贝即
  来源于此）；Stock 路径将 strided 输入直接交给 aclnnMatmulWeightNz。

## 6. 一次性 74ms 双 Rank 停顿（observed，机理未确认）

窗口第 9 个融合波（第 1 窗口请求内，0.465s 处）双 Rank **同时**出现 74.2/74.8ms 的
V2TrialKernel 超长执行（其余 199 个 p50 2.57ms、p90 3.0ms、最大 3.6ms）。
相邻波正常（~2.5ms）。该事件贡献了 V2 窗口首请求 +104ms 中的 ~74ms。双侧同时性
指向池/编排层事件（如 epoch 边界与波相交），本轮基线无 G1 埋点无法定位；
**与 G1-D 正在追查的 tail-sampler/epoch 现象可能同源，建议并入其取证线**。
发生频率本轮 1/200 波，不足以影响稳态结论。

## 7. L3 回归验证（无 Profiler，原始 4096/2048、50 prompts、并发 10）

配置：`bench_memfabric_mm_ar_serve.sh`（`MEMFABRIC_MM_AR_E2E_V2=1`，stock→fused 各 1 次
独立 serve，`--save-detailed` 逐请求明细已入库 bench_results）。V2 serve 真实性同样
核验：80/80 层选择、M=4096→V2(chunk=1024)、双 Rank 首调位精确校验通过；全轮 0 失败、
0 次 507014。

| 指标 | Stock | V2 | Δ |
|---|---|---|---|
| 吞吐 (tok/s) | 46.45 | 45.94 | **−1.10%** |
| TTFT mean (ms) | 11341.9 | 11578.6 | **+2.09%** |
| TTFT p50 (ms) | 11811.2 | 12091.7 | +2.38% |
| TTFT p90 (ms) | 16142.5 | 16510.5 | +2.28% |
| TTFT p99 (ms) | 27435.1 | 28075.1 | +2.33% |
| TPOT mean (ms) | 209.76 | 212.04 | +1.09% |
| TPOT p50 (ms) | 209.02 | 212.49 | +1.66% |
| TPOT p99 (ms) | 217.29 | 218.14 | +0.39% |
| 成功/失败 | 50/0 | 50/0 | — |

判读：

1. **方向与量级与 G3 Eager 三会话一致**（G3：吞吐 −3.51%/−4.53%/+2.12%，TTFT
   +2.1~+2.5%；本轮单对：吞吐 −1.10%、TTFT 全分位 +2.1~2.4%）。TTFT 是最稳健的
   信号，与本文 §3/§4 的纯 Prefill 归因（+1.4% 波内 + 排队放大）吻合。
2. **TPOT 差异非 V2 算子所致**：decode M=10 < 2048，双臂均走纯 Stock 路径（路由
   规则决定），+1.1~1.7% 属 serve 级噪声/交叠效应（G3 亦观测到同量级波动，且
   TPOT p99 仅 +0.39%）。
3. 单对复测的局限如实声明：serve 重启噪声 ±0.5~0.8%（G3 方法论结论），本轮为
   单对 AB；吞吐 −1.10% 单看在噪声带边缘，但与 TTFT 全分位一致劣化 + 三层设备级
   证据（§4）合并判读，**V2 无端到端收益的结论成立**。

## 8. 无法确认的部分（诚实边界）

1. **融合核内部 2.57ms 的更细拆分**（SDMA 到达等待 vs IB 等待 vs 协议轮询的精确占比）
   需要 kernel 内部分段计数器或 G1 埋点，本轮基线按约束未启用——只能给出 mte2/scalar
   计数器级别的分解。
2. **74ms 停顿根因**（§6）：单次发生、双侧同时、无状态错误；需 G1-D 工具链。
3. **Rank 间偏斜的净影响**：r1 的 Stock AR（1512us）天然比 r0（1149us）多等 ~360us
   （快 Rank 在 HCCL 内等待），V2 融合核 r1−r0 差 −145us——两路径都以不同方式吸收
   偏斜，无法在本数据下分离"偏斜本身"的贡献。
4. **Graph 模式**：按本轮范围排除（且 G3 报告 §4.6 的 torch.compile 固化问题未修复，
   Graph 下 V2 不会执行）。
5. M=4095（非对齐暂存补齐类）路径未单独采集：本轮输入恰为对齐类直通波；
   补齐类的额外 stage 拷贝可能使 V2 略更不利，方向与本结论一致。

## 9. 结论与下一步建议

**核心问题的回答**：V2 的算子级加速没有转化为真实 Qwen3.6 Serving 收益，不是因为
收益被稀释或被别处吃掉，而是因为**在真实 Serving 条件下该算子本身就没有加速**——
微基准的 Stock 基线（同步税/AR 连发竞争）系统性地劣于真实条件下的 Stock
（无竞争、间隔充分、厂商调优 GEMM），V2 的固定串行成本在真实对比中反超。
设备级证据（200×2 站点、分布零重叠）与端到端、步周期三层闭合。

| 优先级 | 建议 | 依据 |
|---|---|---|
| P0 | 融合核传输通路优化：mte2 1.71ms / scalar 0.87ms 是全部劣势来源（数学已打平）；对标 HCCL ring 的 1.15ms/16.8MB 找差距（分块粒度、IB 等待策略、协议轮询降频） | §4.4 |
| P0 | 消除 V2 分支的输入 contiguous 拷贝（直接消费 strided x 或复用 stage 区） | §4.1 |
| P1 | 74ms 双 Rank 停顿并入 G1-D 取证线；带 G1 埋点的复测采集 | §6 |
| P1 | 若继续算子级微基准：基线必须含"间隔充分的无竞争 AR"工况（模拟真实 Serving 间距），否则会再次高估融合收益 | §4.5 |
| P2 | Prefill 侧收益天花板 ~1.5%（40 站点×3.3% 占比）——即便优化打平也难有端到端感知；价值判断建议转向 decode/交叠工况或带宽受限拓扑 | §4.5 |
| P2 | 非对齐 M=4095 暂存路径单独采集补齐（与 G3 A/B 实际负载的对齐类存在差异） | §8.5 |

## 10. 交付物清单

| 交付物 | 位置 |
|---|---|
| 本报告 | `docs/V2_PREFILL_ATTRIBUTION_REPORT_20261021.md` |
| 脱敏证据包 | `evidence/v2_prefill_attribution_20261021/`（provenance、双臂请求时延、serve 日志节选、双 Rank 逐站点 CSV、分析 JSON、分析脚本） |
| 采集脚本（复用现有工具的最小变体） | `tests/e2e/_310p/v2_prefill_capture.sh`（已入库） |
| L3 回归数据 | 服务器 `/home/<user>/310P/bench_results/v2attrib_l3_20261021/`（逐请求明细 JSON + bench 日志；脱敏节选入证据包 `l3_regression/`，关键数字见 §7） |
| 原始双 Rank Trace（仅服务器） | `/tmp/opencode/v2_prefill_captures/{stock,v2}_prefill_v2attrib/` |
