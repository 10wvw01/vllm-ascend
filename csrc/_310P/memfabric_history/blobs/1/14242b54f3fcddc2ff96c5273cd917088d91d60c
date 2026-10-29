# A1-MM-SHADOW 实机报告：D4同构影子MM vs 原生NZ-MM 同构对照、八场U1 Confirm与HCCL审计（2026-10-27）

> **run_id:** `a1-mm-shadow-batch-20261027-001`｜**分支:** `work/310p-mm-ar-large-m-a1-20261026` @ `695d05769`（预检修复 `6d061e2f0`/`48a37f109`，文档 `eb0e1caef`）
> **设备:** 310P3 卡16384 dev2/3（`ASCEND_RT_VISIBLE_DEVICES=2,3`）｜**窗口:** 14:37:24–16:31:36 UTC 连续独占（113分钟，223条30秒只读监视快照，外部进程=0）
> **身份:** Host `1dc46d7d…`、Device D4 `2a19390c…`、插件 `8eb65de9…`、KVER11 `25e16b0a…`；**运行后四重SHA复核不变**；共享SO未覆盖、正式D4未动、KFC首写未跳过、无自动换卡。

## 结构化裁决

`MM影子同构对照=MM_ONLY_COMPARISON_VALID(run2)`｜`run1=INCONCLUSIVE_DRIFT(native臂场间方差9.7%,保留为复份)`｜`U1八场Confirm=INCONCLUSIVE_DRIFT(07场D4-M1024 anchor漂移6.99%,R2已知周期长尾现象)`｜`HCCL完成审计=已附(audit_sha256=0dfc386f…,载荷线性667.8/1236.9µs@8/16MB)`｜`正式D4内部MM精确时长=UNKNOWN(维持)`｜`L1候选=未签`

## 1. 核心问题答复：原生NZ-MM vs V2影子MM 的差距

**run2（2026-10-27T16:10–16:18，单次不中断完整12场，drift全臂≤3.2%）：**

| 指标 | M2048 | M4096 |
|---|---|---|
| 无Profiler双Rank Event p50：原生NZ-MM | 478.13µs | 848.20µs |
| 无Profiler双Rank Event p50：影子ND-MM（D4同构8核/Chunk1024/tiling1024） | 484.72µs | 867.63µs |
| **Δ_MM_only（无Profiler Event）** | **+6.59µs（+1.38%）** | **+19.43µs（+2.29%）** |
| 短Profiler设备Kernel p50（双Rank取max）：原生`aclnnMatmulWeightNz_MatMulV3Common_MatMulV3` | 405.66µs | 777.99µs |
| 短Profiler设备Kernel p50（双Rank取max）：影子`V2MmShadowKernel` | 416.90µs | 814.49µs |
| **Δ（短Profiler纯Kernel）** | **+11.23µs（+2.77%）** | **+36.49µs（+4.69%）** |

**结论1（同构独立MM效率差）：**D4同构影子MM与原生NZ-MM在相同几何（K=N=2048、FP16、NZ权重、M2048/4096、bit-exact）下，无Profiler调用时延差 **+6.6/+19.4µs（+1.4%/+2.3%）**，短Profiler纯Kernel时延差 **+11.2/+36.5µs（+2.8%/+4.7%）**。MM计算实现本身不是D4全链路落后的主要来源。

## 2. 受控内存扰动（run2，同上有效批次）

| 扰动 | M2048 | M4096 | 说明 |
|---|---|---|---|
| B2 sym vs B1 nd（写入真实MemFabric对称arena） | +3.66µs | +7.08µs | 对称GM位置条件开销可忽略（<1%） |
| C1 sym_clean vs B2 sym（每Chunk CleanLines） | **+69.29µs** | **+139.64µs** | **CleanLines是最大受控增量**（占原生MM的14.4%/16.5%） |

**结论2：**每Chunk对输出区逐64B缓存行DCCI清理的成本远大于对称GM写入与MM实现差之和；正式D4流水中的CleanLines是明确的下一优化变量（受控增量证据，不等于正式D4内部缓存阶段精确时长）。

## 3. 八场U1 Confirm（NZ-Stock MM+HCCL vs 正式D4，独立八场ABBA）

- 八场全部bit-exact、协议status=0、双Rank配对；NZ真实路由由本批独立短Profiler证明（`A1_NZ_U1_NATIVE_ROUTE_V1`，rank×M2048/4096共4组`aclnnMatmulWeightNz`+`hcom_allReduce`配对，`u1-stage/nz_route/`）。
- ABBA窗口（stock/fused−1）：M2048 **−21.77%/−19.59%**，M4096 **−19.70%/−20.09%**——正式D4全链路比原生NZ-Stock链慢约20%，与R2 Screen/Confirm趋势一致（M4096: stock链2013µs vs D4 2500–2544µs）。
- **判定INCONCLUSIVE_DRIFT**：07场（D4臂）M1024 anchor早/晚对漂移6.99%>5%门禁；其余7场0.17–3.23%。该现象与R2已固证的`REPRODUCIBLE_PERIODIC_V2_TAIL_CAUSE_UNKNOWN`同源（V2臂M1024采样方差），非外部占用（全程监视0外部进程）、非HCCL污染。按合同不剔除、不换算，**不签L1候选**。
- **HCCL完成审计**（`confirm/route_audit.json`，audit_sha256已附入`u1_analysis_audited.json`）：FP16无转换、事件括号流序、跨流完成在窗内的载荷线性证明——stock链−NZ MM-only = **667.8µs@8MB、1236.9µs@16MB**（比1.85×≈线性，与R1的568/1174µs同量级）；wall≥event；双Rank对称。

**结论3：**D4全链路约20%的落后无法由MM段解释（影子MM仅+1~2%事件差/+3~5%纯Kernel差）；结合§2，缺口主要在CleanLines增量、Chunk间流水与协议等待（R2 diag证据方向一致），**正式D4内部MM精确时长维持UNKNOWN**，不得以影子p50冒充。

## 4. 预检发现并修复的两个确定性缺陷（实机拦截）

1. **NZ物化不流有序（`6d061e2f0`）**：profile脚本首次`npu_format_cast(w,29)`后立即裸指针消费→首个enabled launch全错（out_absmax≈5 vs ref≈229，双Rank 2/2复现；iter1起全对；`aclrtMemcpy`裸读证实拷贝不排队）。修复：cast后显式`stream.synchronize()`+UT源码守卫。
2. **私有Host SO无池API（`48a37f109`）**：`mf310p_*`仅存在于冻结插件内部；arm sym臂AttributeError双Rank复现。修复：独立编译同源V1 adapter导出SO（不触碰任何冻结/共享SO）+`/proc/self/maps`真实映射门禁+campaign私有LD_LIBRARY_PATH。
3. 预检全项通过后才放行campaign：双Rank对称池create/barrier/exchange/layout/destroy全rc=0（rank0 send=0x12c140000000、rank1 recv=0x12c182000000、arena 32MB）；完整sym臂120样本/M×2 bit-exact；CleanLines模式bit-exact；native/nd路由各4份真实CANN CSV。
4. **分批续跑原则落地（`695d05769`）**：按用户指示，外部占用中断不停整批；已完成阶段/会话经同源`read_session`校验跳过，部分目录归档`OUT/aborted/`，source SHA跨批次一致；数值/门禁失败仍硬STOP。本批campaign-run1两次被宿主会话清理击杀后按此语义续跑完成（RESUME_LOG留痕），run2为单次不中断干净批次并作为主结果。

## 5. 证据索引（`csrc/_310P/memfabric_mm_ar_v2/evidence/a1_mm_shadow_batch_20261027/`）

- `campaign-run2/`：主结果完整树——12场双Rank JSON（每场120原始Event+wall样本）、`build/`（影子SO `22b841f4…`+adapter SO `4f04cdd4…`+manifest）、`routes/native|nd/`（8份真实CANN kernel_details.csv+route.json+profiler原始目录）、逐场device_before/after、`mm_shadow_assessment.json`（MM_ONLY_COMPARISON_VALID）。
- `campaign-run1-drift-replicate/`：run1全12场rank JSON+评估（INCONCLUSIVE_DRIFT复份）+RESUME_LOG。
- `u1-stage/`：`nz_route/`（NZ路由证明+CSV）、`confirm/`（八场rank JSON、`route_audit.json`、`u1_analysis.json`、`u1_analysis_audited.json`）。
- `precheck/`：双bug取证脚本与输出、对称池预检、影子/adapter构建。
- `monitoring/card16384_timeline.log`：14:05–16:31全窗口外部进程=0。
- `final_npu_state.txt`：运行后设备状态；四重SO SHA运行前后不变。
- 边界：不外推Qwen E2E/L3/L4；正式D4内部分段UNKNOWN；run1不作为正式对照；U1 Confirm不签L1候选。

## 6. 下一步最小实验建议

1. **CleanLines消减**是首要优化变量：受控增量+69/+140µs远超MM实现差与对称GM扰动之和；候选方向为按Chunk实际脏行范围收敛DCCI遍历、或与Add段合并清理，任何变更须重走G0/正确性/路由/漂移门禁。
2. V2臂M1024周期长尾（本次07场anchor 6.99%）需在Epoch/Gate E体系中根治后方可签L1候选；重测八场Confirm约15分钟，可与Epoch四场同窗口。
3. 影子MM的+11/+36µs纯Kernel差（+2.8%/+4.7%）若要收敛，可对照CANN tiling参数（M4096差大于M2048，提示大M下baseM/流水差异），但收益上限小于CleanLines项。
