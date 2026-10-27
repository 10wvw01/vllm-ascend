# A1-R2 单批实机诊断 STOP 报告：CPU合同修复→NZ路由纠偏→设备占用中断（2026-10-27）

> **run_id:** `a1-r2-batch-20261027-001`｜**执行:** OpenCode 单批连续授权（用户选项2：最小修复后继续R2）
> **工作分支:** `work/310p-mm-ar-large-m-a1-20261026`，本批两笔修复提交 `b25866222`（CPU合同回归）与 `215deb767`（NZ-Stock基线路由）
> **设备（目标）:** 310P3 卡`16384`（dev2/dev3 同卡双Die，`ASCEND_RT_VISIBLE_DEVICES=2,3`）
>
> **结构化裁决：**
> `N0环境/冻结身份=PASS`｜`CPU全量回归=2项FAILED→修复后151全绿`｜`批次attempt1=NZ_ROUTE_NO_GO(路由门正确拒绝假NZ基线)`｜`NZ路由根因=已定位并修复(实机3变体对照)`｜`NZ-Smoke/Screen/8场Confirm=NOT_RUN`｜`D4五Case机制诊断=NOT_RUN`｜`Epoch 15/5/5/15四场=NOT_RUN`｜`最终STOP原因=dev2被外部进程占用(共享环境安全门失败)`
>
> **本轮无任何换卡/重启/驱逐他人进程/共享SO改动/未执行L4。**

---

## 1. 三个P0目标的真实状态（不伪造）

| P0 | 状态 | 说明 |
|---|---|---|
| 1. NZ真实Stock对照收益 | **NOT_RUN（被阻断）** | 路由门在第一步即正确拒绝假NZ基线；基线路由已修复并经实机3变体验证（见§3），但正式Smoke/Screen/Confirm未执行。NZ_L1判定不存在。 |
| 2. D4内部瓶颈可信程度 | **NOT_RUN（被阻断）** | 五Case诊断与短Profiler均未启动。`INNER_PHASE_UNKNOWN` 维持。 |
| 3. 周期长尾根因可信程度 | **NOT_RUN（被阻断）** | 15/5/5/15四场Serving未启动。`EPOCH_CAUSAL_UNKNOWN` 维持。 |

## 2. N0 环境复检（PASS，全部字节级证据）

- 卡16384（dev2/3）进场健康OK、无进程、基线显存、hugepages 0/0；其它租户位于 5664/dev1、43520/dev6/7（互不干扰）。
- 冻结身份核实：Host SO `1dc46d7d`（priv-libs与priv-pylib两份一致）、D4 Device `2a19390c`（priv-libs与site两份一致）、插件 `8eb65de9`、KVER11 `25e16b0a`；共享文件与R1终态逐字节不变（`88d39285`×2、`b8ca96f0`）。
- CANN 9.1.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 / vllm 0.27.1；无毒环境变量；端口 18881/18882/18883 空闲；独立worktree tracked clean；回滚=仅unset环境变量。
- R1私有装载方案复现成功：`PYTHONPATH=site + LD_LIBRARY_PATH=priv-libs:site/vllm_ascend:...` 下插件解析、四SO装载路径与R1 worker maps逐一吻合。

## 3. CPU门失败与修复（`b25866222`）

全量pytest（one_batch与CI同一命令）在最新HEAD `460523b38` 首次执行即 **2 FAILED/149 passed**（该分支CI运行次数=0，从未远程验证）：

1. `test_nz_source_default_is_nonbreaking`：陈旧字面断言 `export U1_STOCK_LAYOUT=`，`8880921f3` 的等价重构使其失配——纯测试侧缺陷。
2. `test_budget_shift_is_timing_correlation_only`：fixture忠实建模5s臂~50%占空比（N4假说与 `2.5≤gap5≤8` 窗口的本意），而评估器MAD离群阈值在占空比≥50%时数学上必然漏检（中位数骑跨双峰→阈值533ms>尖峰120ms）→评估器检测器与fixture不兼容。

修复（用户选项2授权）：#1更新断言至现行等价形态；#2将尖峰检测改为P10基线+半幅差阈值（20ms噪声地板不变，稀疏/密集模式均可检出；裁决语义不变：仍禁止签发因果）。修复后 **151 passed + 27 subtests**，py_compile/bash -n 全过。

## 4. 批次attempt1：路由门正确拒绝假NZ基线（NZ_ROUTE_NO_GO）

`a1_r2_one_batch.sh` 于 CPU_PRECHECK 后进入 NZ_REAL_ROUTE，探针双Rank bit-exact通过并产出4份真实CANN CSV，随后 `a1_nz_route_gate.py` 判 **NOT native CANN kernel details CSV → NZ_ROUTE_NO_GO**，批次按设计自 STOP（`STOP_STAGE.json`）。

**真实发现（本批最重要成果）：** 探针的 `F.linear(x, w_nz)` 在裸torchrun进程实际命中 **通用 `MatMul`(ND输入/NZ输出) + `trans_TransData`（~1830+200µs，与R1行主序MM-only 1869µs一致）**，而非模型真实站点的 `aclnnMatmulWeightNz_MatMulV3Common_MatMulV3`（~750µs，ND输出）。若路由门不存在，NZ-Stock基线将测错2.4倍的慢基线——合同N1“若Profiler显示ND MM，STOP并修路径，不允许报告NZ已完成”的门禁价值被实机证实。

另发现：裸 `torch_npu.profiler.profile`（无 `_ExperimentalConfig`）只产出任务级精简CSV（无aclnn名/形状/格式），vllm serve的完整CANN分析来自 `TorchNPUProfilerWrapper` 的 `_ExperimentalConfig`。

## 5. 路由根因定位与修复（`215deb767`，实机3变体×2配置对照）

隔离实验（每Rank TP2、M4096、bit-exact全程通过、同一探针环境）：

| 变体（无serve启动开关） | 内核 | 时延 |
|---|---|---|
| V1 `F.linear(x,w_nz)` | `MatMul9`(NZ输出)+TransData | 1817–1863µs |
| V2 `torch.ops.vllm.unquantized_gemm` | 同上 | 1809–1863µs |
| V3 `torch.matmul(x,w_nz.t())` | 同上 | 1836–1892µs |

加入 serve 启动开关（`torch.npu.config.allow_internal_format=True`（model_runner_v1.py:229）+ `set_compile_mode(jit_compile=False)`（worker_310p.py:44））后，**全部三种调用形态（含探针原样V1）均命中 `aclnnMatmulWeightNz_MatMulV3Common_MatMulV3`，753–790µs，ND输出，ND;FRACTAL_NZ输入**——与R1 L2真实站点证据（749.8/752.1µs）吻合。

修复内容：`u1_arm.py` 仅对 `stock_fp16` 且 `U1_STOCK_LAYOUT=nz` 的会话设置上述开关（fused/row_major会话保持R1历史环境以保可比）；探针增加同样开关与 serve 式 `_ExperimentalConfig`（去掉 `record_shapes`，等待异步CANN离线分析）；UT合同锁死 serve 镜像要求。修复后151全绿并推送。

**含义（待正式N2验证）：** 修复后的NZ-Stock基线才是真实站点对照。R1 L2已示真实站点 V2(2511µs) 比NZ-Stock链(≈750+1166~1391µs)慢约368µs/站点，因此预期正式NZ Confirm大概率呈负向——判定值不预设，以实机为准。

## 6. 最终STOP：dev2被外部进程占用（共享环境安全门失败）

attempt2 一切就绪（修复HEAD、刷新身份JSON、launcher更新）时，npu-smi时间线显示：

- 03:57（诊断实验前）dev2仍无进程、hugepages 0/0；
- ~04:04 起 dev2 出现**外部宿主进程3554394**（本容器PID命名空间不可见），492MB→30869MB 以~1GB/s模型装载速率增长，hugepages 15392/15392 占满；
- 04:08–04:11 稳定 30870MB、AICore 0%（装载后驻留）、dev3未被触碰；
- 我方诊断trace导出在04:01后无任何新文件（排除孤儿分析进程归属）。

结论：dev2被外部租户/定时任务占用，16384独占性被破坏。按用户规则“共享环境安全门失败立即STOP；不自动换卡、不重启、不覆盖共享SO”执行全批停止。注意凌晨~04:00为定时任务高发窗口，本轮进场时间（03:36）与其冲突。

## 7. 恢复与续跑程序（下一次授权）

1. 重新G0：确认dev2进程3554394释放、显存/hugepages回到基线、四卡健康、无新增占用；
2. 全新OUT目录重启 `a1_r2_one_batch.sh`（launcher与环境快照已备：`/tmp/opencode/a1-r2-launch.sh`，身份JSON随时按新证据刷新）；
3. 顺序不变：NZ真实路由→NZ Smoke/Screen/8场Confirm→D4五Case+短Profiler→Epoch四场→R1离线复算；任一门失败仍按合同STOP。

## 8. 证据索引

- `attempt1_nzroute_nogo/`：批次运行册、STOP_STAGE、原仓CPU回归、探针双Rank JSON与4份**错误路由CSV**（通用MatMul+TransData实证）、路由门拒绝日志。
- `nzroute_fix_diag/`：`run2_without_serve_startup/`（3变体均慢路由+精简CSV schema实证）、`run3_with_serve_startup/`（3变体均`aclnnMatmulWeightNz`、6份完整CSV、bit-exact、findings JSON）、隔离实验脚本与launch日志。
- `cpu_gate_fixes/`：修复前后全量pytest日志、bash -n。
- `device_occupation/`：时间线说明与最终npu-smi快照。
- SERVER_ONLY（不上传）：探针/诊断的原始profiler树（含CANN host数据切片），保留于 `/tmp/opencode/a1-r2-batch-attempt1-nzroute-nogo/` 与 `/tmp/opencode/a1-r2-nzroute-diag/{run2,run3}/`。
- 判定边界：本报告所有结论仅限本轮代码/环境/卡；三项P0正式实机数据 NOT_RUN，无任何伪造或外推。
