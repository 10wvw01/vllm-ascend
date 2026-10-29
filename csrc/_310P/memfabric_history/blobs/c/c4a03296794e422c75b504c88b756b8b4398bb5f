# A1-R1 单批实机闭环报告：G0恢复→M2048 Smoke→L1→L2→L3（2026-10-27）

> **run_id:** `a1-r1-batch-20261027-001`｜**执行:** OpenCode 单批连续授权（[单批合同](../../docs/A1_R1_ONE_BATCH_THROUGH_L3_20261027.md)）
> **工作分支:** `work/310p-mm-ar-large-m-a1-20261026` @ `26d7ea3f515f802d95b1941fcbd3e2a8f99a3b69`（批次内唯一代码修复：事件冲刷，见§8.3）
> **设备:** 310P3 卡`16384`（dev2/dev3 同卡双Die，`ASCEND_RT_VISIBLE_DEVICES=2,3`），全批独占，健康无告警
> **软件身份（全批冻结）:** Host SO `1dc46d7d…`（物理号修复版）｜D4 Device SO `2a19390c…`（Tile2048/Chunk1024）｜插件 `8eb65de9…`（Gate-C诊断ON）｜KVER11 `25e16b0a…`｜CANN 9.1.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 / vllm 0.27.1 editable `6e448d0e` / Python 3.12.13
>
> **结构化裁决：**
> `G0/ISOLATION=PASS`｜`SMOKE=PASS(ALLOW_SCREEN)`｜`L1_SCREEN=PASS(数据完整，正向但基线失真)`｜`L1_CONFIRM=OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS(同失真警示)`｜`L2_TRACE=L2_VERIFIED`｜`L3=L3_INCONCLUSIVE_OR_NO_GAIN`｜`L4=L4_NOT_RUN_BY_DESIGN`

---

## 1. 执行总表

| 阶段 | 判定 | 关键结果 |
|---|---|---|
| S0 环境复原+私有装载+原仓CPU | **PASS** | 私有 PYTHONPATH site + LD_LIBRARY_PATH priv-libs 双隔离装载，零共享文件修改（前后SHA证明不变）；原仓 pytest 108+27 subtests 全绿 |
| S1 M2048双Rank Smoke | **PASS → ALLOW_SCREEN** | Stock/V2 双臂双Rank 位精确全过、V2协议 status=0、首写/建池/IB探针清洁、KFC双Die功能验证、真实装载 maps 闭环（RANK绑定） |
| S2 L1 Screen | **PASS** | M2048 +15.9%、M4096 +22.2%（配对）；但 Stock 基线为行主序 F.linear，后被L2证明不代表真实站点 |
| S3 L1 Confirm 8场ABBA | **PASS（候选+失真警示）** | M2048 +22.5%（窗口25.3/20.4%）、M4096 +21.1%（20.8/21.4%），漂移3.6%/1.2%，尾延迟门槛全过 → OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS；M1024 −11.1% NO_GO（非Serving路由形状） |
| S4 L2 真实双Rank Profiler | **L2_VERIFIED** | 每Rank每步真实40站点（10 o_proj+30 out_proj，M=4096,K=2048,N=2048）；Stock真实链=MatMulV3(NZ) 752µs + hcom_allReduce 1391/1166µs；V2融合kernel 2511/2430µs → **真实站点V2慢~368µs(+17%)** |
| S5 L3 无Profiler四场ABBA | **L3_INCONCLUSIVE_OR_NO_GAIN** | 每步配对p50：V2比Stock慢 **26.1ms(F1对S1) / 27.7ms(F2对S2)**；Stock跨会话漂移仅2.83ms、噪声地板2.83ms；V2另带重尾（p90 2451~2517ms vs Stock 2348ms） |
| S6 L4 正式TTFT | **L4_NOT_RUN_BY_DESIGN** | L3未GO，按合同不运行正式4096/2048/50请求验收 |

## 2. 私有装载与共享环境零扰动（S0核心）

- **方案：** `PYTHONPATH=/tmp/opencode/a1-batch/site`（A1工作树468个`vllm_ascend/*.py` + 主工作树冻结构建产物 + D4替换件）+ `LD_LIBRARY_PATH=/tmp/opencode/a1-batch/priv-libs:…`（`1dc46d7d` Host SO + `2a19390c` D4 SO）。插件为 DT_RUNPATH，`LD_LIBRARY_PATH` 优先于 `$ORIGIN`，故共享插件目录的 baseline `b8ca96f0` 永不会被本批进程装载。CWD 保持中性目录以避免 `sys.path[0]=''` 命中主仓。
- **运行时证明：** U1 双臂双Rank worker（`evidence/smoke_worker_maps`、`confirm_worker_maps`）与 L2/L3 全部8个 serve worker（`s5_l3/worker_maps`）的 `/proc/<pid>/maps` 均实证装载 `priv-libs/libmf310p_v2_trial_device.so`(2a19390c) + `priv-libs/libmf_hybm_core.so`(1dc46d7d) + `site/…vllm_ascend_C…`(8eb65de9)；事件文件名中的 pid 与 maps pid 一一绑定。
- **共享文件零修改：** 两份共享 Host SO（`88d39285`）、插件目录 device SO（`b8ca96f0`）、插件本体（`8eb65de9`）批次前后 SHA 逐字节不变（`s0_environment/` 与 `final_shared_files.sha256`）。**本批未覆盖任何其它任务共享SO。**
- **回滚：** 仅需 unset 环境变量，无系统级部署需要恢复；KVER11 JSON 本体未改动，dev2/3 设备侧 KFC 由建池功能证明（IB/fan探针8核会合+首写位精确+status=0）。

## 3. S1 Smoke（M2048 双Rank，Stock→D4）

`u1_run.sh large smoke`（smoke2/）：Stock rank0/1 与 D4 rank0/1 全部位精确（CPU-gloo FP32参考）；D4 协议 status=0、锚点8/8复验通过；事件 p50：Stock 1633/1635µs，D4 1343/1335µs。首次尝试因我方组装的 PYTHONPATH 覆盖丢失 CANN `tbe` 模块路径而失败（ACL初始化错误500001，无设备触碰），保全后修正重跑通过（见§8.1）。`a1_r1_smoke_gate.py` 验收 **ALLOW_SCREEN**（attestation 9项全真、13份证据文件、含失败尝试的异常记录）。

## 4. L1 Screen/Confirm（完整算子U1，双Die隔离进程）

| M | Stock p50（行主序MM+HCCL） | V2 D4 p50 | 配对加速 | 8场ABBA窗口 |
|---|---|---|---|---|
| 1024（诊断） | 855.5µs | 962.2µs | −11.1% | NO_GO（非Serving路由） |
| 2048（正式） | 1649.0µs | 1345.8µs | **+22.5%** | 25.3% / 20.4% |
| 4096（正式） | 3043.8µs | 2513.7µs | **+21.1%** | 20.8% / 21.4% |

Stock跨会话漂移 3.56%（M2048）/1.19%（M4096），fused p90≤stock_p90×1.05、p99≤×1.10 全过；HCCL跨流完成审计（`route_audit.json`：Stock事件窗含完整AR的载荷线性度证明 264/568/1194µs @4/8/16MB、wall≥event、双Rank对称）后由 `a1_large_m_assess.py` 判 **OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS**。

> **⚠️ 基线失真警示（本批最重要方法论发现）：** U1 Stock 使用 FP16 `F.linear`（行主序B）作MM，M4096 MM-only 1869µs；但 L2 真实模型站点 Stock 走 `aclnnMatmulWeightNz`（NZ权重布局）仅 752µs（快2.5倍）。**U1的“完整算子Stock”不是真实Serving站点的Stock，L1正向结论不可外推到模型。** 这正是L1→L2→L3链存在的意义，且已由L2/L3实测证实（下两节）。

## 5. L2 真实Serving站点取证（Stock/V2 各一场短Profiler）

- **命中：** 每Rank每步恰40个融合目标站点（10×full-attention `o_proj` + 30×GDN `out_proj`，模型 `Qwen3_5MoeForConditionalGeneration` 40层，layer_types 精确对应）；V2 serve 路由日志 80条 `Select MemFabricMmArV2LinearMethod310`（40/worker），首调位精确gate双worker通过（M=4096, aligned=True，无额外contiguous拷贝）。
- **站点设备关键路径（p50，5步×40站点×2Rank）：**
  - **Stock：** `MatMulV3`(NZ) 751.9/752.1µs → 紧邻 `hcom_allReduce`（gap 0.8µs）1390.9/1165.5µs → 链计 **2142.8/1917.6µs**
  - **V2：** `V2TrialKernel<tiling1024,1024>`（含MM+SDMA AR+Add单kernel）**2510.8/2429.7µs**
  - **→ 真实站点 V2 比 Stock 慢约 368µs（+17%）**；V2 kernel p50 与U1微基准D4完全一致（2511 vs 2514µs），慢因完全在Stock侧NZ-MM过快而非V2变慢。
- D4身份：V2臂 `V2TrialKernel` 的SO装载由本批serve worker maps 证明为 priv-libs D4 `2a19390c`（双SO含同名模板符号，名称不可区分，须以maps+时序双证）；kernel时长也与D4微基准吻合、显著异于旧Tile1024（2727µs）。
- `a1_l2_site_audit.py` 判 **L2_VERIFIED**（40站点/Rank/步、20份独立trace切片、真实形状、完成依赖、零拷贝）。

## 6. L3 无Profiler真实Prefill Step ABBA（S1→F1→F2→S2）

每场独立 Serve、6预热+30有效请求、input4096/output1/并发1/TP2/Eager、事件探针（Forward入口→Logits，当前流完成事件，无任何插入同步；仅`query()`读取自然完成）。

| 场 | 臂 | 配对p50 (ms) | p90 | p99 | 组内pstdev |
|---|---|---|---|---|---|
| S1 | Stock | 2346.08 | 2347.85 | 2352.36 | 1.86 |
| F1 | V2 | 2372.20 | 2517.36 | 2557.02 | 56.26 |
| F2 | V2 | 2370.96 | 2450.99 | 2459.81 | 30.74 |
| S2 | Stock | 2343.25 | 2345.42 | 2350.60 | 1.83 |

- **配对增益：** S1−F1 = **−26.12ms**；S2−F2 = **−27.70ms**（负=V2慢）；Stock跨会话漂移2.83ms = 噪声地板；两窗口增益均为负且远超噪声。
- **V2重尾：** p90抬升约100~170ms（单步最坏至2557ms；L2 trace亦见V2 kernel 70.6ms离群），方向与SDMA等待/池行为一致，需O-01归因。
- 客户端时延同向：Stock 2361/2363ms vs V2 2389/2387ms（+25~27ms）。
- `a1_l3_prefill_assess.py` 判 **L3_INCONCLUSIVE_OR_NO_GAIN**；校准审计10项全真+独立L2_VERIFIED依赖+双Rank对齐+四场无Profiler。**L3未GO → L4不运行。**

## 7. 四段传导链（同一D4版本）

| 层 | 结果 | 解读 |
|---|---|---|
| L1 完整算子（U1） | V2 +21~22%（vs行主序Stock） | 算子本身不慢于该基线 |
| L2 真实站点（40/步） | V2 −17%（慢368µs/站点） | 真实Stock用NZ-MM（752µs），比U1基线MM快2.5×；V2单kernel 2511µs > Stock链 2143µs |
| L3 设备Prefill步 | V2 −26.1/−27.7ms（−1.1%） | 40×368µs≈14.7ms 设备差 + ~12ms host/池/发射开销 + V2重尾 |
| L4 客户端TTFT | **NOT_RUN_BY_DESIGN** | L3未GO；L2带Profiler场的+27ms客户端观测仅作诊断参考 |

## 8. 异常、失败与修复记录

1. **[环境组装失败·已修复]** 首次Smoke：我方 `PYTHONPATH` 覆盖丢失 CANN `tbe` 模块路径 → worker ACL初始化错误500001（无设备触碰）。保全 `smoke_attempt1_envfail`，修正后通过。
2. **[工具误杀·已修复]** 二次启动前 `pkill -f` 误匹配自身命令行杀死shell，未产生任何设备影响；改用PID文件方式清理。
3. **[代码修复·已提交本分支]** L3事件冲刷缺陷：每请求恰1次prefill forward（output=1无decode步），最后一个请求的pending NPU事件永不drain → 每场仅35/36事件。修复 `a1_prefill_http_requests.py`：测量后追加一条64-token非4096冲刷请求（其事件被4096过滤排除，不参与样本）。attempt1（35事件）保全于 `s5_l3/sessions_attempt1_35events`；修复后4场全部36/36；原仓 py_compile+108 UT 通过。
4. **[容忍性告警]** HYBM `QueryDeviceInfoSocTolerant` 310P sdid/server-id查询返回207000容错回退（已知310P行为）、可选库 libhcom/extend未装载告警——均为非设备错误，未触发任何STOP条件。
5. **[全批零设备故障]** 无507014/507899/507018；43520 Die7 MCU Alarm为其历史存量（非本批卡）；批次结束卡16384回到基线显存、无进程、hugepages 0、端口全释放。

## 9. 结论与下一优化方向（O-01输入）

**结论：** 当前D4（Tile2048/Chunk1024）在真实Qwen3.6 4096-token单请求Prefill上 **无收益（约−1.1%，且带重尾）**；U1微基准的+21~22%是行主序Stock基线失真所致。Stock默认继续正确。真实站点Stock链（NZ-MM 752µs + HCCL AR 1166~1391µs）是当前必须击败的目标，D4单kernel 2511µs 与之差 368µs/站点。

**下一单变量候选（按证据优先级）：**
1. **融合kernel内MM段NZ化/提效**：D4的MM段当前不能利用 Stock NZ-MM 的752µs效率；若融合kernel的MM段达到NZ-MM同等效率，站点链理论可降至 ~752+SDMA_AR+Add（交叠）< 2143µs。先在kernel内加MM/SDMA分段计时归因，定位2511µs中MM与AR等待占比。
2. **V2重尾归因**：p90抬升100~170ms与L2 trace中70.6ms离群一致，疑SDMA到达等待/池刷新；需AICPU侧或IB探针在长窗下统计。
3. **U1基线修正**：U1增加NZ权重布局的Stock臂（`npu_format_cast(w,29)`+`aclnnMatmulWeightNz`路径）作为第三对照，防止后续L1再被行主序基线误导。
4. **池/发射host开销**：−26ms中约12ms非设备kernel差，归因每次调用的pool bookkeeping/发射路径。

## 10. 证据索引

- S0：`s0_environment/`（环境/四卡/共享SHA/CPU回归/装载预审/运行时probe）
- S1：`s1_smoke/`（u1原始双Rank JSON、worker maps、gate、attestation、失败保全）
- S2/S3：`s2_screen/`、`s3_confirm/`（U1原始场数据、分析、route audit、A1评估）
- S4：`s4_l2/`（manifest、客户端记录、serve log、40站点提取切片`analysis/`、L2_VERIFIED；原始ascend_pt/kernel_details.csv 32件为 SERVER_ONLY，见 `SERVER_ONLY_raw_traces.json`，留存 `/tmp/opencode/a1-batch/l2`）
- S5：`s5_l3/`（四场sessions原始事件jsonl、worker maps、校准审计、L3结果、attempt1保全）
- 本报告所有判定仅限本批版本/卡/单请求稳态负载；**L3未GO，不构成任何模型级或发布级GO。**
