# F1-C 实机报告：310P3 单卡双 Die 小 M=10 Stock FP16 HCCL vs T1 MemFabric 严格 ABBA 对照（2026-10-25）

> 状态：**8 场独立进程 ABBA 实机执行完成，全部双 Rank 位精确、T1 协议 status=0、零失败、零 ≥2ms 长尾样本。**
> **裁决：F1C_NO_GO_OR_MORE_EVIDENCE——M=10 下 T1 MemFabric 融合（MM8/Add4）在配对 NPU Event 主指标上比原生 Stock FP16 `F.linear + HCCL AllReduce` 慢 27.5%（两轮 ABBA 窗口分别为 −26.3% 与 −12.9%，方向一致）。F1-1 初筛的 +20%（0.340 vs 0.408ms）被证实为口径偏差产物，不成立。**
> 分支：`perf/310p-f1c-stock-fp16-hccl-vs-memfabric-abba-20261025`（实验运行 HEAD `c23bb1aac`，含 2 项测试代码最小修复，见 §7；计划文档审查基线 `4af36b6fa0fd`）。
> 设备：dev6/7（卡 43520 单卡双 Die 独占全程）。
> 服务器时钟：实验期间主机本地时间为 2026-10-24 16:58–17:20（与文档日期差一天，如实记录）。

## 0. 执行摘要

| 必答问题 | 结论 |
|---|---|
| 8 场 ABBA 全景 | `Stock→T1→T1→Stock` ×2 轮共 8 个独立 torchrun TP2 会话，每场 5 预热 + 50 计时样本 × 双 Rank；全部 `torch.equal` 位精确（vs CPU gloo FP32 规约参考），T1 全部 `protocol_status=0`，无 507014/507899/挂死/超时/端口残留 |
| M=10 谁快（严格口径） | **Stock FP16 快**：Stock 配对 Event p50 中位 **308.2µs** vs T1 **425.2µs**（−27.5%）；两轮 ABBA 窗口 −26.3%/−12.9% 均为负 |
| F1-1 的 68µs 是真的吗 | **不是**。F1-1 Stock 计时窗口含 FP32↔FP16 两次转换（T1 历史口径）且为建池前 host 同步计时；去除后纯 FP16 Stock 比融合路线更快 |
| 长尾 | 全部 8 场 **0 个 ≥2ms 样本**；融合侧 p90/p99（454.5/504.7µs）反而优于 Stock（481.9/618.9µs）——T1 输在中位固定成本，不是抖动 |
| 计时有效性 | Stock HCCL 跨流执行（`hcom_allReduce` COMMUNICATION 任务 Stream=N/A）；3/3 被 Profiler 捕获迭代中 `event_us` ≥ 含 hcom 结束点的全簇跨度（1064≥706、436≥253、410≥261µs）→ **NPU Event 确证覆盖完整 HCCL 通信**（`route_proof/`） |
| 通信隔离 | Stock 进程仅 HCCL 数据面（trace 含 hcom、无建池日志）；T1 进程仅 MemFabric 数据面（trace 零 COMMUNICATION 任务、gloo-only 进程组）——从未混用 |

## 1. 执行身份（manifest 原文归档 `manifest.txt`）

| 项 | 值 |
|---|---|
| repo_head（运行时） | `c23bb1aacabfe11e4eb912b127770d55d9eb1b51` |
| 设备 SO | `2a19390c…`（D4 Tile2048 诊断 ON，受控部署，运行进程经 /proc/maps 实证加载） |
| Python 插件 | `8eb65de9…`（Gate-C 诊断 ON，`nm -D` 实测导出 T1 ABI 五符号） |
| MemFabric Host SO | `1dc46d7d…`（dev6/7 物理号修复版，双安装路径逐字节一致） |
| KVER11 JSON | `25e16b0a…`；KFC helper 双 die 报告 `kernel already present` |
| Stock 路线 | 原生 `F.linear(fp16)` + HCCL FP16 `all_reduce`，计时窗口内 **零** float/half 转换；输出按原生 API 分配 |
| T1 路线 | `mf310p_v2_trial_run_nz_direct_small`，NZ weight（`npu_format_cast(w,29)`）、MM Core=8、Add Core=4，预分配输出 |
| 输入 | 权重种子 4242；输入种子 `1200+rank*17+M+variant*1009`，2 variant 轮换 |
| 门槛变量 | `MF_KFC_BYPASS`/`MF310P_V2_NOWARMUP` 全程未设置；`ASCEND_RT_VISIBLE_DEVICES=6,7` |

## 2. 八场 ABBA 完整结果（配对双 Rank per-index max，Event µs）

| 会话 | 路线 | p50 | p90 | p99 | wall p50 | Host−Event p50 | ≥2ms 样本 |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | stock_fp16 | 303.6 | 350.5 | 541.5 | 394.8 | 96.8 | 0 |
| 02 | memfabric_t1 | 418.4 | 445.5 | 1795.3 | 497.6 | 79.3 | 0 |
| 03 | memfabric_t1 | 418.1 | 454.5 | 479.5 | 495.9 | 81.5 | 0 |
| 04 | stock_fp16 | 312.8 | 341.6 | 618.9 | 392.9 | 86.0 | 0 |
| 05 | stock_fp16 | 296.6 | 329.7 | 534.6 | 391.8 | 95.7 | 0 |
| 06 | memfabric_t1 | 432.0 | 452.1 | 504.7 | 500.4 | 78.1 | 0 |
| 07 | memfabric_t1 | 435.9 | 471.0 | 512.7 | 511.0 | 79.9 | 0 |
| 08 | stock_fp16 | 459.0 | 536.6 | 676.8 | 571.8 | 122.2 | 0 |

聚合（主判据，`f1c_audited.json`）：

- Stock 配对 Event p50 中位（4 场）**308.215µs**；T1（4 场）**425.223µs**；净差 **−27.517%**
- ABBA 窗口：第 1 轮（会话 01-04）Stock 308.2 vs T1 418.2 → **−26.306%**；第 2 轮（05-08）Stock 377.8 vs T1 434.0 → **−12.94%**
- 全样本 p90：Stock 481.93 / T1 454.51；全样本 p99：Stock 618.85 / T1 504.69
- 每场 2 Rank × 50 样本全保留；失败/缺失/脏版本一律 fail-closed（分析器 16 项合同校验通过）

## 3. 分析

### 3.1 已实测事实

1. **两轮 ABBA 方向一致为负**：无论取哪一轮、哪个 Stock 会话（除环境退化的 08），T1 都不占优。最宽松口径（退化会话 08 的 459.0µs 对 T1 最优 418.1µs）也仅 −9.7%，远不到 +5% GO 门槛；严格口径全部为大幅负值。
2. **T1 输在中位固定成本，不是长尾**：T1 的 p90/p99 比 Stock 更低（更稳），但 p50 高 ~117µs。路由取证显示 T1 融合调用是调用者流上的**单 kernel**（`V2NzDirectSmallKernel<16,8>`，Profiler 下时长 288.7–415.2µs），即融合核内部的协议握手/IB 等待/SDMA 编排固定成本本身就超过了 Stock 的"MM 链（~52µs 设备时间）+ HCCL（暖态 ~80–90µs）+ Host 下发空档"总和。
3. **Stock HCCL 跨流且 Event 覆盖完整**：Stock 设备侧为 stream 37 上的 `trans_TransData×3 + MatMul7`（~52µs）+ Stream=N/A 的 `hcom_allReduce`（暖态 78.3/91.6µs）；3/3 被捕获迭代 `event_us` ≥ 含 hcom 结束点的全簇跨度 → 主指标有效，不存在"Event 漏计 HCCL"的系统性低估。
4. **公平性边界事实**（计划 §2 如实保留）：Stock 按原生 API 每次分配输出、T1 预分配输出——该差异若有影响则**有利于 T1**，而 T1 仍然输，故 NO_GO 结论对该边界稳健。
5. **环境事件**（如实披露，不删数据）：会话 02 出现 1 个 1.69ms 单样本长尾（i=18，双 Rank 同期，<2ms 门槛）；会话 08 双 Rank 从首样本起一致变慢（前 25 样本 p50 与后 25 样本一致，434-459µs vs 其他 Stock 会话 285-313µs），为环境性退化（同期其他租户负载/主机抖动），非预热或漂移问题；该会话拉高第 2 轮 Stock 中位，**即使如此第 2 轮仍为 −12.9%**。

### 3.2 与 F1-1 初筛数字的关系（不混用作结论，仅解释偏差来源）

| | F1-1（host 同步，Stock 计时窗含 FP32↔FP16 转换，建池前/后分离） | F1-C（双 Rank 配对 Event，纯 FP16 Stock，独立进程交叉 ABBA） |
|---|---|---|
| Stock M=10 | 0.408ms | **308.2µs**（Event p50 中位） |
| T1 M=10 MM8/Add4 | 0.340ms | **425.2µs** |
| 表观差 | +20.0%（T1 优） | **−27.5%（T1 劣）** |

F1-1 的 68µs "优势"由以下口径偏差共同造成，本次已全部排除：(a) Stock 计时窗口内额外的两次逐元素 dtype 转换及其下发空档；(b) 建池前 Stock / 建池后 T1 的单次时间顺序；(c) host 同步计时混入 launch/sync 抖动且未做双 Rank 配对。**68µs 不是真实 FP16 MM+HCCL 对照收益。**

### 3.3 为什么 F1-C 的 T1 绝对值（425µs）高于 F1-1 的 340µs

两侧协议不同，不可直接相减：F1-C 每样本执行 gloo CPU barrier + 新建 Event 对 + 逐样本 stream 同步 + 输入双 variant 轮换，使每样本的设备侧下发空档（ev_start 与 kernel 启动之间）被完整计入 Event 区间（两侧同口径承担）；F1-1 为连续回环采样（launch 与执行重叠摊薄）。**跨实验绝对值不可比，同实验内配对比较才是本实验的判据**——这正是计划 §2 明文要求的隔离。

## 4. 门槛核对（计划 §5）

| 门槛 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 两轮 ABBA 各自 speedup | ≥ +5% | −26.306% / −12.94% | ✗ |
| T1 配对 p90 | ≤ Stock×1.05 | 454.51 ≤ 506.03 | ✓ |
| T1 配对 p99 | ≤ Stock×1.10 | 504.69 ≤ 680.74 | ✓ |
| ≥2ms 长尾 | 0 | 0 | ✓ |
| 双 Rank 位精确/无 status 异常/无缺失 | 是 | 是 | ✓ |
| 路由与 HCCL Event 审计 | 独立取证为真 | `audit.json` 四项全真（`route_proof/`） | ✓ |

**裁决：F1C_NO_GO_OR_MORE_EVIDENCE（单算子层面，M=10）。** 审计链完整：`f1c_provisional.json`（PROVISIONAL）→ 独立路由/覆盖性取证 → `audit.json` → `f1c_audited.json`（NO_GO）。

## 5. 剩余风险与边界

1. **本结论限定 M=10、K=N=2048、TP=2、FP16、单卡双 Die（dev6/7）**；不自动外推到其他 M、其他卡对或真实模型。按计划，M=10 未取得有意义收益，不扩展 M=1/8/16/32。
2. 会话 08 的环境退化提示同宿主其他租户负载可造成 ~+50% 的会话级漂移——跨会话绝对值比较必须警惕（本实验用同轮配对窗口抑制了该影响）。
3. T1 单 kernel 内部阶段占比（MM/IB/signal/wait/SDMA/Add/DONE）仍未拆分（需 F1-B 协议探针）；本实验只证明"融合整体不优"，不定位核内瓶颈。
4. HCCL Event 覆盖性取证基于 3 个 Profiler 捕获迭代（无 Profiler 的正式会话未做 Trace，按计划隔离）；证据方向一致且与位精确结果自洽。
5. Stock 原生 `F.linear` 在该形状走 `TransData×3+MatMul` NZ 路径（框架自动选择，符合"原生算法"合同），若未来框架版本改变该行为，Stock 基线需重测。

## 6. 下一步建议（按优先级，不改正式 Kernel/协议为前提）

1. **F1-B 协议固定成本探针**（现有 `mf310p_v2_trial_run_ar_only` ABI）：AR-only / signal-wait 循环量化 T1 协议下限，回答"融合核 ~300-420µs 中协议握手占多少"——这是判断 MemFabric 路线在 310P3 双 Die 上是否有任何 M 区间可赢的根本依据。
2. **F1-A MM 隔离**：M=10 纯 MM（NZ）核数 4/8 对比，量化 MM 侧占比，复核 MM=8 选择。
3. 若 F1-B 证实协议下限即 ≥Stock 全程，则 310P3 双 Die 小 M 融合路线应明确归档为 NO_GO（除非协议本身获得独立优化方案，另行评审），Stock 保持默认路由，F2 保留 D4 几何、C3 不合入。
4. 任何后续真实模型三方验证仍固定：Qwen3.6-35B-A3B-w8a8，input 4096 / output 2048 / TP=2 / batch≤10 / Eager，与历史 output 512 数据严格隔离；**在单算子 GO 之前不启动**。

## 7. 环境恢复与测试代码修复记录

### 7.1 恢复执行记录（实验结束后立即执行）

- 设备 SO：`vllm_ascend/libmf310p_v2_trial_device.so` 由 D4 Tile2048 诊断 `2a19390c…` 还原为装机态 **`b8ca96f0387c45ec08e61ec3606f1502da22944a61b139815b62a20695a06811`**（复核一致）。
- MemFabric Host SO：`/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/` 与 site-packages 双路径由补丁版 `1dc46d7d…` 还原为装机原版 **`88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`**（双路径均复核一致）。
- Python 插件 `8eb65de9…`（Gate-C 诊断 ON）保持实验前原状（与 F1-1 收尾口径一致，非本次部署物）。
- 设备侧 dev6/7 的 KVER11 `libmf_sdma_orch_v11.so` 按 D4/F1-1 先例保留（正式部署物）。
- 还原源：`/tmp/opencode/f1c_g0/backup_20261024T165834/`（G0 部署前备份，SHA 见 `g0_backup_sha256sums.txt`）。
- 收尾核验：dev6/7 Health OK、AICore 0%、显存 1614/1338MB 空闲基线、无进程；8581 端口零 LISTEN（`post_restore_npu_smi.txt`、`post_restore_port_probe.txt`）；无残留 torchrun。

### 7.2 本轮测试代码修复（最小、已回归、已提交）

1. `9cc8ebb8e` `fix(310p): F1-C port guard via /proc/net probe for ss-less host, G0 evidence`——宿主无 `ss`/`netstat`/`ip`，初次 G0 用 `ss` 取证实为 command-not-found（无效证据，已如实更正并重新取证）。新增 `test/perf/f1c_port_probe.py`（解析 `/proc/net/tcp{,6}`，对必需空闲端口 fail-closed），`f1c_run_abba.sh` 改用该探针，UT 新增端口守卫合同测试锁定。回归：bash -n、py_compile、`test_f1c_abba_contract.py` 8/8 PASS。
2. `c23bb1aac` `fix(310p): drop unused probe read in F1-C source contract test`——ruff F841 清理。
3. 未修改任何正式 Kernel、Host ABI、通信协议、Stock 默认路由（`git diff 4af36b6fa0fd..HEAD` 仅测试代码与证据）。

## 8. 证据清单（本目录）

| 证据 | 路径 |
|---|---|
| G0 准入报告（含原始证据） | `G0_PRECHECK_REPORT.md` + `g0_*.txt/log` |
| 冒烟会话（Stock/T1 各 30 样本） | `g0_smoke_stock.log`、`g0_smoke_t1.log`、`smoke_*_rank{0,1}.json` |
| 路由/HCCL Event 覆盖性取证 | `route_proof/`（G0 期）与 `abba_route_proof/`（正式目录副本） |
| 8 场会话全量原始数据 | `sessions/01..08_m10_*/`（双 Rank JSON、session.log、前后 npu-smi、端口探针） |
| 环境与身份 manifest | `manifest.txt`（SHA 见内） |
| 初步分析（审计前） | `f1c_provisional.json`（verdict=PROVISIONAL_ONLY…） |
| 独立审计输入 | `audit.json`（四项 true，全部有 Trace 依据） |
| 最终审核结论 | `f1c_audited.json`（verdict=F1C_NO_GO_OR_MORE_EVIDENCE） |
| ABBA 控制台日志 | `abba_console.log` |
| 服务器侧原始数据 | `/tmp/opencode/f1c_m10_001/`、`/tmp/opencode/f1c_route_proof/`、`/tmp/opencode/f1c_smoke_{stock,t1}/`、`/tmp/opencode/f1c_g0/` |

## 9. 最终裁决

- **单算子（M=10，MM+AR，310P3 单卡双 Die TP2）：NO_GO**。T1 MemFabric 融合在严格配对 Event 口径下比原生 Stock FP16 慢 27.5%，两轮 ABBA 方向一致；F1-1 的 +20% 为口径偏差。
- **Stock 保持默认路由；F2 保留 D4；C3 不合入**——与研发原则一致，无变更。
- **真实 Qwen 模型三方验证不启动**（其前置单算子 GO 未达成）。
- 下一步按 §6 优先级推进 F1-B/F1-A 取证，不直接改核或协议。
