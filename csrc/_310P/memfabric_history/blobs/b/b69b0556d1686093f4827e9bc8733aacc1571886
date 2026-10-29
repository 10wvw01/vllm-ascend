# V2 MTE2/Scalar Gate C：4 Chunk 受控隔离诊断实机验证报告（2026-10-22）

> 状态：**实机执行完成（C0–C4）**。C2 正确性 6/7 变体双 Rank 位精确通过，diag1 按纪律裁定 NO-GO 未运行；C3 完成时长 + 管道计数器取证；发现并完整归档一类新的**共模跨 Rank 间歇停顿**现象。
> 分支：`diag/310p-v2-gate-c-controlled-isolation-20261022`（f2ac18ef6，基于 Gate A/B 核验分支 7a1a08930）。
> 设计文档：`V2_MTE2_SCALAR_GATE_C_IMPLEMENTATION_20261022.md`。原始 Trace 与 msprof 数据仅存服务器（SHA 见 `evidence/gate_c_20261022/trace_provenance.json`，36 件 / 43.9MB）。
> 本轮未修改正式 V2 Kernel、Serving 路由、MemFabric 协议与 KVER11 部署；未开展长压测与算子优化。

## 0. 执行摘要

| 核验项 | 结论 |
|---|---|
| C0 环境准入 | 发现 G1-D 残留（tail Host SO + 设备侧 v12），已恢复原装并验证；**共模停顿之外全部 PASS** |
| C1 构建/ABI | 诊断 ON 构建通过，ABI 9→10 符号零丢失；OFF 构建按设计排除全部诊断 TU；CPU UT 22/22 |
| C2 正确性 | formal/loop4 全 4096 行位精确；diag5/0/3/4 首 Chunk TP2 AR 位精确；协议 status 全零 |
| diag1（NoIB） | **NO-GO 未运行**：G1-D P0 挂起根因未明且 9 小时前刚在本设备复现，无法证明安全边界 |
| 4 Chunk 交叠价值 | **有效**：formal 清洁中位 ~2.56/2.61ms vs loop4 ~3.12-3.61/3.02-3.16ms，交叠节省 **~0.5-1.1ms/核（20-30%）** |
| 融合核成本归因 | mac 恒定 459us（=4×diag0 114.6us）；其余 ~2.1ms 全部为等待相 mte2+scalar 活动；与 prefill 轮 serve 级超额（mte2+1375/scalar+686）定量吻合 |
| Rank 不对称 | 单 Chunk 交换中 rank0 比 rank1 多等 **~1.5ms**（diag3: 2143 vs 685us）；formal 交叠吸收后仅剩 ~30-45us |
| 新发现异常 | **共模跨 Rank 间歇停顿**（23-45ms，严格配对，mac 不变，mte2+scalar 吸收）；与 AIC 指标采集强相关；正确性从未受影响 |
| 74ms 异常 | 未在 74ms 幅度复现；43ms 级配对停顿多次出现（见 §5）；Gate E P0 独立维持未关闭 |

## 1. C0 设备准入：G1-D 残留发现与恢复（重要环境事件）

准入检查发现两项 Gate C 手册明确禁止的 **G1-D 诊断环境残留**（时间线完整可溯）：

1. **Host SO 漂移**：双路径 `libmf_hybm_core.so` 为 `bd1760cc…`（G1-D tail 采样诊断构建，`memfabric_hybrid@diag/310p3-g1d-minimal-forensics-20261021`，10-22 03:05 由独立工作树 `vllm-ascend-g1d` 的 g1d3 批次部署），非 Gate A/B 核验的 `88d39285…` 正式恢复态。g1d3 批次 D3R1-R4 正常、**D3R5 于 04:47 故障停止**（serve.log 含 507014×4、aicore timeout×3，证据已保全于该工作树），批次结束后环境未恢复。
2. **设备侧 v12 部署残留**：g1d3 曾向 dev0/1 部署 `libmf_sdma_orch_v12.so`（独立文件名，不删除 v11；无 JSON 指向时不加载）。

**处置**（备份-恢复-验证链完整）：

- Host SO 从 `build-backups/g1d3_backup_20261022T030505Z/` 经原子 rename 恢复为 `88d39285`（双路径、root:root 644、哈希复核）；tail 版本体仍保全于 `build-g1d/`。
- 设备侧 v11 探针（orch_deploy_helper，哈希 849f93c0 与 g1d3 断言一致）：**dev1 rc=3 原装在位**；**dev0 曾缺失**（与 D3R5 故障恢复清除 KFC 自定义核一致），已从 /usr/local 原装源（13960B，Oct 12 安装）重部署并验证（rc=0，符号+尺寸校验）。日志：`evidence/gate_c_20261022/helper_v11_probe.log`。
- KVER11 JSON `25e16b0a`、Gate A/B 原始 Trace 未受影响；shell 无任何 MF_* 残留；pyinject 仅存 /tmp（PYTHONPATH 作用域，惰性）。
- 构建前插件 SO `929b53c2` 已备份。

**环境风险（持续项）**：本机为共享环境，存在周期性短时外部负载（宿主侧进程，容器内不可见 PID），间歇占用 dev0/dev1（曾观测 AICore 100%/87%、内存 +1~12GB，分钟级自退）。已按"每 case 前后核验设备空闲 + 精确内核计数 + 离群单列"纪律管控；formal 两次独立会话（其中一次启动瞬间 dev0 有 78% 尖峰）清洁分布重合（差 <1.5%），证明 Kernel 时长测量未被实质污染。

## 2. C1 构建、ABI 与 CPU UT

- **诊断 ON 构建**（官方路径 `python setup.py build_ext --inplace`，CANN 9.1.0 + MemFabric set_env + SOC_VERSION=ascend310p3 + V1/V2 使能）：通过。插件 `8eb65de9…`、设备库 `b8ca96f0…`。
- **ABI**：新增导出 `mf310p_v2_trial_diag_loop4`（T 符号）；与 Gate A/B 基线 SO 对比 9→10 符号，**零丢失**。
- **诊断 OFF 构建**（预置 CMakeCache `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS:BOOL=OFF` 独立 build-temp）：Host 仅保留 create/destroy/ib_probe/run/check_status 5 个正式符号（`diag_loop4` 及其余诊断 ABI 全部不在，符合 CMakeLists 设计）；设备库无 `V2DiagLoopKernel`、`V2TrialKernel` 完整。ON 制品随后哈希校验还原。
- **CPU UT**：`test/ut/` 22/22 通过。修复一处**预先存在**的失败：`test_v2_project_layout.py` 仍断言 V2 根目录仅 {docs,src,test}，而 `evidence/` 自 5701ec3f9 起为各 Gate 既定提交目录——将 `evidence` 纳入合法集合。
- 过程偏差记录：探测管道指标时误删工作区 `export_only_prof_dir/`（含 Oct 18 既有 profiler 数据），已从 20261020 资产审计档案（`g1_local_archive/…/20261020_310p3_asset_audit`，SHA256SUMS 1122 件 OK）`cp -a` 恢复并保持 `.git/info/exclude` 原状。

## 3. C2 双 Rank 正确性

每变体独立 torchrun 会话；输入为 rank-specific `torch.randn`（FP16）+ NZ 权重，参考值为 FP32 升精度 allreduce 后回落 FP16（HCCL）；`torch.equal` 位精确 + 64B 协议 status=0，**每次采样逐次校验**（含 Profiler 窗口内）。

| case | 语义 | rank0/rank1 结果 |
|---|---|---|
| formal | M4096/4 Chunk/overlap=1 正式核 | 3 预热 + 6 采样全位精确，status=0 |
| loop4 | M4096/4 Chunk 串行（新 `mf310p_v2_trial_diag_loop4`） | 同上 |
| diag5 | 正式核单 Chunk 串行 | 同上（首 Chunk TP2 AR） |
| diag0 | MM-only | 同上（本 Rank MM） |
| diag3 | IB | 同上 |
| diag4 | IBDONE | 同上 |
| diag1 | NoIB | **NO-GO 未运行** |

**diag1 NO-GO 依据**（手册 §5 默认处置）：NoIB 省略核间 IB 会合（消息环时序保护）；G1-D P0（邮件可见性丢失→永久挂起→设备重置）根因未明，且 D3R5 于本验证前 9 小时刚在这对设备复现（507014×4）。无法证明当前条件下的合理安全边界；一旦挂起将再次清除 dev0 v11 并中断 Gate C。不为凑齐矩阵冒险，其隔离价值（IB 增量方向）已由 diag3/diag4 覆盖。

## 4. C3 性能取证：交叠价值与成本归因

方法学边界（手册 §3 禁止项全部遵守）：单 Chunk 变体不可线性外推全形状；loop4 与 formal 仅协议调度形态不同，其差值为交叠收益的方向性度量而非纯收益上界；NoIB/IB 差额不解释为 IBWait 时长；短窗口设备核数据不外推 50×2048 真实 Serving。

### 4.1 干净种群中位时长（us；真实发射；legacy diag 包装器每次调用 2 牺牲 + 1 真实，分析器按每三元组末位提取并强制计数）

| case | rank0 | rank1 | mte2 r0/r1 | scalar r0/r1 | mac |
|---|---:|---:|---:|---:|---:|
| formal（清洁会话 ×2，各 n=6） | 2556/2588 | 2587/2634 | 1623/1647 | 886/922 | 459 |
| loop4（清洁样本，3 会话） | 3118-3609 | 3020-3163 | 1823-2164 | 1792-2477 | 458 |
| diag5（正式核单 Chunk） | 1264 | 935 | 808/294 | 967/270 | 115 |
| diag0（MM-only） | 231 | 360 | 139/128 | 53/53 | 114 |
| diag3（IB） | 2143 | 685 | 1400/432 | 1728/229 | 115 |
| diag4（IBDONE） | 2407 | 872 | 1531/459 | 1891/279 | 115 |

完整分布、p90/max 及污染会话数值见 `evidence/gate_c_20261022/{summary.json,gate_c_duration_census.json}`；原始 Trace 哈希见 `trace_provenance.json`。

### 4.2 交叠有效性（核心问题的回答）

- formal 清洁中位 ~2.56/2.61ms；loop4 清洁中位 ~3.12-3.61（r0）/3.02-3.16（r1）ms。
- **交叠节省：rank0 ~0.93-1.12ms，rank1 ~0.43-0.70ms（约 20-30%）**。
- 交叠同时吸收了 Rank 不对称：formal 中 r1 仅比 r0 慢 ~30-45us，而 loop4 中 r0 反而比 r1 慢 ~0.4-0.6ms（r0 串行等待慢侧 r1 的交付）。
- **结论：4 Chunk 交叠设计有效且量级明确，不是损失来源。**

### 4.3 融合核 ~0.63ms/层超额成本的收窄

1. **计算无辜**：formal mac 恒定 459us ≡ 4×diag0（114.6us），与 Stock MM（prefill p50 795us 同形状，NZ 布局差异另计）同量级；拷贝路径 Gate A/B 已排除。
2. **等待相主导**：formal 其余 ~2.1ms 全部为 mte2（~1.65ms）+ scalar（~0.9ms）活动——即 CleanLines/SDMA 等待/邮件轮询的管道画像。相对 4×MM-only 基线的超额 mte2 +1067us / scalar +674us，与 prefill 轮 serve 级实测超额（mte2 +1375 / scalar +686）**定量吻合**——prefill 的超额即来自融合核自身等待相，非环境或 Profiler 伪影。
3. **Rank 不对称量化**：单 Chunk 交换中 rank0 比 rank1 多付 ~1.5ms（diag3 2143 vs 685；diag4 2407 vs 872）——r1 侧 signal→交付链路显著慢于 r0 侧；r1 本地 MM 亦慢 ~130us（diag0 360 vs 231，且该差额不出现在 mac/mte2/scalar 中，属任务调度层）。此为正式优化的**首要候选机制**：r1 交付路径每提速 1ms，串行等待即等量收缩；交叠模式下亦收敛尾部。
4. 增量链（方向性，禁止当分段耗时）：diag0→diag3（+IB/SDMA/Add 串行）：r0 +1912 / r1 +325；diag3→diag4（+DONE）：r0 +264 / r1 +187。r0 的通信增量近 r1 的 6 倍，再次指向不对称在等待侧。

### 4.4 MTE2/Scalar 行为级画像（替代 Gate B 不可用计时）

- formal 干净样本：mte2/scalar ≈ 1623/886（r0）；loop4 干净样本：≈1823-2164/1792-2477——串行使 scalar 等待轮询显著增加（+0.9-1.6ms），mte2 增加 ~0.2-0.5ms。
- diag5 单 Chunk 正式核：r0 808/967 vs r1 294/270——**同为正式核，r0 的等待相是 r1 的 ~3 倍**，与 4.3 一致。

## 5. 异常记录：共模跨 Rank 间歇停顿（新现象，完整档案）

**现象**：含通信等待的 Kernel 样本间歇性膨胀至 23-45ms，**双 Rank 同一样本位严格配对**（起始差 <1-4ms、时长差 <2%），幅度量子化（观测类：~23ms ×1、~27.5ms ×1、~39-45ms 多数；diag4 两次 42.91/42.86ms，m10 五次 43.4-44.5ms）。部分会话呈**隔样本交替共振**（SLOW-fast-SLOW…，~44ms 周期）。

**签名**：mac 恒定不变（计算零影响）；膨胀由 mte2+scalar 吸收（等待/轮询通道）；个别实例时长膨胀但管道计数正常（混合签名）；**正确性从未受影响**（全部位精确、status=0、无 507014/挂起）。

**普查**（≥6ms 判停顿；详见 census JSON）：

| 会话 | 指标采集 | 停顿 r0/r1 |
|---|---|---|
| loop4（首会话） | AIC | 3/6, 3/6 |
| loop4_metric_rep | AIC | 1/6, 1/6 |
| loop4_m10 | AIC | 6/10, 6/10 |
| loop4_nometric | 无 | 0/6, 0/6 |
| loop4_i10 | 无 | 0/10, 0/10（2 例 4.5/5.3ms 轻度带） |
| formal（6 样本 ×2） | AIC | 0/6, 0/6 |
| formal_m10 | AIC | 2/10, 2/10（+中间带 3.2-5ms ×4-6） |
| diag3 / diag4 / diag5 | AIC | 2/6, 2/6 / 2/6, 2/6 / 1/6, 1/6 |
| diag0（无通信对照） | AIC | **0/6, 0/6**（230us ±1.5%） |

**归因边界（observed）**：

- 与 **AIC 管道指标采集强相关**：全部 ≥23ms 停顿发生于 `--aic-metrics` 会话；无指标会话 16 样本零大停顿（含一次 dev1 外部负载 99% 期间——外部负载排除为主要解释）。
- formal 短会话 0/12 为暴露不足而非免疫：formal_m10 出现 2 例 43ms + 中间带。
- **机制 unknown**。候选（未证实，禁止择一采信）：msprof 设备侧 PMU 采集与 AICPU 编排核/HybmSdmaOrchEpoch 的资源干扰；采集缓冲周期性回写。寄存器/调度器级定位不在本轮。

**与 prefill 轮 74ms 异常的关系**：形态同族（双 Rank 同时、同站点、Wait Time 排除队头、单次大尾延迟）。prefill 的 serve 级 profiler 同样采集管道指标（operator_details.csv 含 mte2/scalar），支持"采集路径效应"候选。**未在 74ms 幅度复现；不能断言该现象在无 Profiler 时存在或不存在**。建议后续用无 Profiler 的主机侧计时普查做判定实验。

## 6. 方法学偏差与工具扩展（本轮新增，均 opt-in/向后兼容）

1. `gate_c_loop4.py` 新增 `--aic-metrics`（须 `--profile`）：经 torch_npu 原生 `ExperimentalConfig(PipeUtilization, Level1)` 采集硬件 PMU 管道计数器（非 Kernel 内埋点，符合 Gate B No-Go 纪律）；`ASCEND_WORK_PATH` 使 msprof 原始数据落入 case 目录。默认路径与已审视版本完全一致。
2. `analyze_gate_c_trace.py`：适配 legacy diag 包装器"每次调用 2 牺牲 + 1 真实发射"结构（真实 = 时间序每三元组末位，计数仍强制 fail-closed）；新增 `collect_pipe()` 从 op_summary CSV 聚合 mte2/scalar/mac/vec（数量不符拒绝结论）。
3. UT 扩展：`test_aic_metrics_optin_and_pipe_parser_fail_closed` 等 3 项断言；UT 22/22。
4. 正式 Kernel、Serving 路由、MemFabric 协议、KVER11 部署：零改动（OFF 构建与源码合同 UT 复核）。

## 7. 结论门禁与下一步正式 V2 优化建议

**~0.63ms/层超额是否已收窄到值得进入正式优化？——是，且已有可测试假设：**

1. **H1（主候选，证据最全）**：融合核超额来自等待相（mte2+scalar 轮询/拷贝活动），其中 **Rank1 侧交付链路慢 ~1.5ms/Chunk 交换**是可定位的结构性不对称。测试：单独 instrument r1 侧 signal→mail 落地路径（设备侧方案需先解决计时源；可先用 r0 等待时长作代理指标做 A/B）；优化方向为 r1 侧 CleanLines/SDMA 提交时序与提前 signal。
2. **H2**：交叠深度可再挖掘——loop4-formal 差 ~0.5-1.1ms 说明当前 overlap=1 已隐藏约一个 Chunk 的等待，但 formal 内仍有 ~2.1ms 等待相活动；候选：signal(k) 提前至 MM(k) 完成前、ack/credit 批量化、DONE 波级合并。
3. **H3（负面排除）**：MM 计算、Host 拷贝、Serving 路由不是来源（mac 恒定 + Gate A/B 拷贝归因 0）。
4. **停顿现象独立立项**：Profiling 会话中 ≥23ms 配对停顿应视为采集交互伪影候选（分析时剔除并单列），需一轮无 Profiler 计时普查判定其在裸路径的发生率；若裸路径也存在，则与 74ms/长尾稳定性同源，升级为正式问题。
5. **74ms**：本轮未复现（43ms 级同族现象见 §5）；Gate E P0（通信邮件可见性/写入缺失）独立维持，不因性能取证关闭。

**不越过门禁的事项**：未修改正式 V2 设计；diag1 NoIB 未运行；Graph/长压/正式调参均未开展。

## 8. 证据索引

| 文件（`evidence/gate_c_20261022/`） | 内容 |
|---|---|
| `env_and_build_record.json` | C0 环境（含 G1-D 残留发现/恢复链、外部瞬态风险）+ C1 构建/ABI/UT 记录 |
| `summary.json` | 分析器 fail-closed 输出（canonical 六例双 Rank p50/p90/max/min + 管道 p50/p90/max） |
| `gate_c_duration_census.json` | 全部 12 个 Profiler 会话的真实发射时长序列、干净中位、停顿普查 |
| `trace_provenance.json` | 服务器侧 36 件原始 Trace/op_summary 的 SHA256/尺寸（43.9MB，不入库） |
| `helper_v11_probe.log` | dev0/dev1 v11 探针与重部署日志 |

服务器原始数据（保留本机，不入库）：`/tmp/opencode/gate_c_c2/`（正确性）、`/tmp/opencode/gate_c_c3/`（canonical Profiler）、`/tmp/opencode/gate_c_c3_supplementary/`（复现/隔离会话）、`/tmp/opencode/gate_c_c0/`（环境核验与备份）。
