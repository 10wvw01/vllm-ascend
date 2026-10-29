# A1五层MM+MemFabric AR与E1关键路径优化｜310P3实机报告（2026-10-28）

> run_id：`a1-five-arm-20261028-001`；分支 `work/310p-mm-ar-five-arm-dcci-20261028` @ `b6ac7330b58a810a11fd74720074f293034b2347`。
> 本报告只记录本批真实设备测量与证据，不改变正式D4/Host SO、Serving默认或任何生产路径。原始证据：本目录`../a1_five_arm_20261028/`。

## 1. 执行与身份

| 项 | 事实 |
|---|---|
| 卡 | 16384（IT21PD2G10 310P3，serial …1180，bus 0000:41:00.0），dev2/dev3 同卡双Die，TP2 |
| 卡健康 | G0时双Die+MCU Health OK、ChipFault=0、hugepages 0/0、无进程；全程30s只读巡检无外部占用、无Alarm、无507014/507899/507018 |
| 其它卡 | 5664被租户占用；38144残留~13GB+hugepages满；43520 Die7 Alarm+占用（按ADR-014排除，未触碰） |
| 环境 | CANN 9.1.0 / BiSheng clang 15.0.5；驱动 npu-smi 26.1.1；Python 3.12.13；torch 2.10.0+cpu / torch_npu 2.10.0.post4；`ASCEND_RT_VISIBLE_DEVICES=2,3` |
| 冻结装载 | Host `libmf_hybm_core.so` `1dc46d7d…`（priv-libs，含物理号修复）；D4 Device SO `2a19390c…`；插件 `8eb65de9…`；KVER11 `25e16b0a…`；私有 PYTHONPATH site + LD_LIBRARY_PATH，共享系统安装未修改 |
| B2/E1私有构建 | `libmf310p_v2_dcci_entire_candidate.so` SHA `72b868c7…`、私有Host `99403527…`，dav-2002，隔离未安装，无正式D4 ABI导出；manifest `build/manifest.json` |
| 二进制保全 | `binaries_before.sha256` 与 `binaries_after.sha256` 完全一致（性能测试期间设备/Host/插件/候选身份未变） |
| CPU准入 | `pytest test/ut` 62 passed（含five_arm合同8项）、`py_compile`、`bash -n`、两轮静态深审通过 |
| KFC | 只读G0不预部署；授权Smoke首次建池自动部署（协议status=0+双Rank bit-exact传递证明；KFC/epoch无运行时日志行，与R1/R2一致） |

## 2. 真实NZ Stock路由证明（先于一切采样）

`torchrun --standalone --nproc-per-node=2 five_arm_nz_route.py`，每Rank×M2048/4096短Profiler原生CANN CSV：

| Rank | M | aclnnMatmulWeightNz | hcom_allReduce | TransData | MM p50 | HCCL p50 |
|---|---|---|---|---|---|---|
| 0 | 2048 | 6 | 6 | 0 | 766.8µs* | — |
| 0 | 4096 | 6 | 6 | 0 | 767.4µs | 1153.3µs |
| 1 | 2048 | 6 | 6 | 0 | — | — |
| 1 | 4096 | 6 | 6 | 0 | — | — |

*路由JSON含各自精确值；4份原始CSV（`routes/rank*_m*_kernel_details.csv`）与torchrun日志已归档。路由证明仅用于Kernel选择验证，不参与十场性能计时。

## 3. B2/E1双Rank正确性Smoke

- smoke b2（stage3）：两Rank×M2048/4096×6样本全部bit-exact（FP32 Gloo参考转FP16），协议status=0。
- smoke e1（stage5）：同上全部通过；`ENTIRE_DATA_CACHE`整核Clean+Invalid在真实8核/双Rank/首写下未出现错算、hang或status异常。
- Smoke通过后才启动十场。

## 4. 十场五臂ABBA结果（stock,b1,b2,d4,e1,e1,d4,b2,b1,stock）

每场双Rank各100样本+20热身；按第i样本取`max(rank0[i],rank1[i])`配对统计。全部20个会话Rank JSON bit-exact=true、协议status=0。

### M2048（成对p50，µs）

| 臂 | 场1 | 场2 | 中位 | 场间漂移 |
|---|---|---|---|---|
| A/stock | 1072.8 | 1068.0 | 1070.4 | 0.45% |
| B1/native+MF | 2643.0 | 2664.6 | 2653.8 | 0.82% |
| B2/fullMM+MF | 1619.4 | 1676.3 | 1647.9 | 3.51% |
| C/D4 | 1324.4 | 1399.7 | 1362.0 | **5.69%** |
| E1/optimized | 1256.3 | 1271.0 | 1263.6 | 1.17% |

### M4096（成对p50，µs）

| 臂 | 场1 | 场2 | 中位 | 场间漂移 |
|---|---|---|---|---|
| A/stock | 2016.5 | 2018.5 | 2017.5 | 0.10% |
| B1/native+MF | 5009.6 | 5052.8 | 5031.2 | 0.86% |
| B2/fullMM+MF | 3123.4 | 3061.7 | 3092.5 | 2.02% |
| C/D4 | 2507.0 | 2519.0 | 2513.0 | 0.48% |
| E1/optimized | 2272.7 | 2275.6 | 2274.2 | 0.13% |

### 机器判定（five_arm_assessment.json）

- **总状态：`INCONCLUSIVE_DRIFT`** —— M2048的D4臂场间漂移5.685%>5%门限；按合同不挑场、不复算。M4096全臂稳定（≤2.02%）。
- 漂移根因：stage07 D4 M2048出现p99=2372.3µs周期性长尾（p90=1457.3µs），与R1已知的V2周期性长尾现象一致；非本批环境异常（card watch干净、二进制身份不变）。
- 路由证明有效（4/4）；E1新二进制在全部十场bit-exact（`BIT_EXACT_BUT_DRIFT`因总状态）。

## 5. 结论（严格区分比较对象）

1. **E1 < C（D4）**：M2048 −98.4µs（**+7.79%**）、M4096 −238.8µs（**+10.50%**）。DCCI整核一次性Clean+Invalid + 两级IB Ready树是**真实的设备内关键路径收益**；且M2048下E1的p99（~1327µs）显著优于D4该场长尾（2372µs），尾部更紧。
2. **E1 ≥ A（Stock）**：M2048 **−15.29%**、M4096 **−11.29%**（仍慢于原生NZ-MM+HCCL完整链）。按A1§5：E1保留为下一优化基线，继续压协议/调度；**不声称比Stock快**，L1对Stock未闭合。
3. **B1 ≪ A**（−59.7%/−59.9%）：原生MM+MF AR完整链含强制生产者流同步、两次牺牲stage+一次有效stage、CacheClean与DONE——这是**真实换通信方案的代价**，不能归因为MemFabric固有带宽慢。
4. **B2 > C**（+285.8µs/+579.5µs）：先完整MM后一次逻辑AR**输给**D4的Chunk交叠流水；交叠（overlap=1）在该形状下价值约0.29~0.58ms，结构上确认当前D4流水方向正确，不倾向退回串行。
5. **A1-MM-SHADOW历史对照兑现**：E1每Chunk每核省去8192次逐64B DCCI扫描的假设方向正确（D4−E1为正），但绝对值远小于“139.6µs×2×4”的朴素加法，符合合同中对多点收敛关键路径的预期。

## 6. 决策（按A1§5合同）

- `E1<C 且 E1>=A` → **保留E1为下一优化基线**（工作分支候选，不进正式默认CMake/Serving）；下一E代聚焦：进一步缩减同步消息数/按可见性证明消除冗余清理、消除D4式周期长尾。
- `B2不比C快` → 维持Chunk交叠主线；B2仅作归因基线保留。
- `B1比A慢` → 归因host sync/staging/多包AR成本，不作MemFabric带宽结论。
- M2048的D4漂移>5% → 本批M2048加速比**不签字**（`INCONCLUSIVE_DRIFT`）；M4096数值内部稳定但随总状态降级为参考。
- **L3/L4 Qwen真实业务全部NOT_RUN**；Stock默认不变；正式D4/Host SO未修改（前后SHA一致）。

## 7. 证据索引

| 内容 | 位置 |
|---|---|
| 评估器输出 | `five_arm_assessment.json` |
| 4×路由JSON+原始CANN CSV | `routes/` |
| 24份双Rank原始JSON（20正式+4 Smoke）+12会话日志+10份场后npu快照 | `sessions/` |
| 全程30s卡巡检 | `card_watch_30s.log` |
| 前后二进制SHA | `binaries_before/after.sha256` |
| G0身份与环境 | `g0/g0_identity.json`、`g0/five_arm_launch_env.sh` |
| B2/E1私有构建manifest/日志/符号 | `build/` |
| 进度与控制台日志 | `progress.txt`、`five-arm-campaign-console.log` |
| HEAD/git status | `commit.txt`、`git_status.txt` |

## 8. 恢复与遗留

- 服务器共享系统SO/Python/插件未做任何修改；私有staging目录与隔离worktree保留；卡16384测后健康OK、无残留进程、显存回到基线。
- 无未解决设备故障；D4 M2048周期长尾与R1现象一致，留待下一E代与长尾专项处理。
