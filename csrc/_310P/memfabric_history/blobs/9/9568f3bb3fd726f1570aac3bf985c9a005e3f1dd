# 310P3 V2 同步与连续异步执行性能差异实机报告（2026-10-21）

> 执行：OpenCode @ 310P3（qwen-test），分支 `perf/310p-v2-sync-async-value-20261021`，
> worktree `/home/<user>/310P/vllm-ascend-perf`，基线 SHA `fd59c9767` + 测试驱动兼容性修复
> `4761e0685`。计划文件：`V2_SYNC_ASYNC_PERFORMANCE_PLAN_20261021.md`。
> 全部原始日志与脱敏证据包：`../evidence/v2_sync_async_20261021/`（下称"证据包"）。

## 0. 一句话结论

**在受控算子级微基准中，V2 的相对优势随异步窗口增大单调坍缩并反转**：
严格逐调用同步（W1）时 V2 快 +7%~+22%；连续 4 次异步提交（W4）时 V2 慢 5%~35%。
`Gain(W4)−Gain(W1)` 在 M=4096 为 −24.6/−27.0pp（4 轮 ABBA），M=3072 为 −38.5pp，
M=2048 为 −41.8pp（各 2 轮）。模式判定为计划 §6 表第 1 行：**连续异步排队损害融合优势**。

## 1. 执行环境与 G0 门禁（全部通过）

| 项 | 值（observed） |
|---|---|
| 源码 | 独立 worktree，分支 `perf/310p-v2-sync-async-value-20261021`，HEAD `fd59c9767`（tracked 文件洁净，运行器逐会话校验） |
| 构建 | `SOC_VERSION=ascend310p3`、`VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1`、`VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1`，`setup.py build_ext --inplace` rc=0 |
| ABI | `nm -D` + `ctypes` 双重确认 `mf310p_v2_trial_create/ib_probe/run/check_status/destroy` 自本 worktree `vllm_ascend_C...so` 导出；运行器进程内硬校验扩展 origin（发现并排除了旧 worktree 经 cwd/site-packages 混入的风险，见 §2 注） |
| MemFabric 基线 | 未切换 `wgm-dev-310p`；KVER11 JSON `25e16b0a…`；全局与 Python 包内 `libmf_hybm_core.so` SHA 一致（`6e9dc2d6…`），运行器逐会话校验 |
| 设备 | `ASCEND_RT_VISIBLE_DEVICES=0,1`（310P3×2）；CANN 9.1.0、torch 2.10.0+cpu、torch_npu 2.10.0.post4 |
| CPU 侧 | `py_compile` 通过；`test_sync_async_summary.py` 5/5 通过（含本轮新增回归，见 §2） |

> 注：机器默认 `vllm_ascend` 指向旧 diag worktree 的 editable 安装。加载顺序核验确认
> 以 perf worktree 为 cwd + `PYTHONPATH` 前置后，实际 import 的扩展来自本 worktree；
> `sync_async_tp2.py` 在每 Rank 进程内再做 `commonpath` 硬校验兜底。无需改动任何
> 系统 Python 包。

## 2. 测试驱动兼容性修复（commit `4761e0685`）

**现象**：G1 Smoke 首跑 4 进程中第 4 个（r2_stock）rc=0、双 Rank 实际均输出
`complete PASS`，但运行器判 `completed=[]` 中止。

**根因**：torchrun 两 Rank 共享同一 stdout fd；非缓冲模式下 `print` 的文本与换行
分两次 `write()`，跨进程交错后两条 JSON 记录拼接在同一物理行（该日志共 5 处，
证据包 `parser_bug_differential/interleaved_lines_raw_excerpt.txt`）。行级
`json.loads` 遇到 `…PASS"}{"tag"…` 同时丢弃两条记录。**非设备故障、非数值错误、
非协议状态错误**。

**修复**（仅测试驱动，不动算子/协议/库，符合计划 G0.6）：

1. `sync_async_tp2.py` `emit()` 改为含换行的单次 `os.write`，记录原子落盘；
2. `run_sync_async_abba.py` / `summarize_sync_async.py` 改用 `raw_decode` 流式解析，
   容忍单行多记录；
3. 新增回归 `test_concatenated_rank_records_accepted`（CPU UT 5/5）。

**差分**：修复后解析器对同一 r2_stock 日志恢复 16 条记录、complete=[0,1]、三窗口
全部可汇总（`parser_bug_differential/reparse_after_fix.json`）。Smoke 以修复版完整重跑。

## 3. G1 Smoke（M=4096，steps=8，W1/2/4，repeat=3，ABBA）

**ALL PASS**（`/tmp/v2_sync_async_smoke_20261021_02`，4/4 进程，0 状态错误，0 hang）。
探索性结果即已呈现完整模式：W1 +26.9% → W2 +6.1% → W4 −2.8%（`W4−W1=−29.7pp`），
且两 ABBA 轮同号。据此进入 G2。

## 4. P0 环境事件与处置（G2 首次尝试中止）

10:06 前 G2 首跑（`/tmp/v2_sync_async_m4096_20261021_01`）r1_stock rank0 在构池前
`torch.randn` 阶段设备 OOM（`ret=207001`，"Failed to allocate 5088 bytes device
memory"）。排查（证据包 `p0_device0_occupation/`）：

- device0（card 5664 chip0）被外部进程 **PID 3873049（python3，容器内，非本实验）**
  占用 **42652MB（99% DDR、hugepages 100%）**；dmesg 记录其 09:23:24 `devmm map
  ret=-95` 失败，宿主 `ps` 无此 PID（跨命名空间）；
- 设备 1 空闲（4%），OS host 内存充足（493GB available）；
- 按计划 §7 纪律：**未做任何系统级恢复、未自动重试**，保全 npu-smi/proc-mem/
  dmesg/ps 快照后挂起；
- 经现场负责人确认 device0 为他人作业，部署监控（连续 2 分钟双设备 DDR≤10% 才
  放行）；10:05 外部作业退出、内存自动回收（d0=3%），10:06:33 自动启动 G2 正式轮。

该事件**不污染任何性能数据**：故障进程组未进入计时（OOM 发生在建池前的数据准备
阶段），后续全部正式会话在双设备独占空闲状态下执行。

## 5. G2 正式 A/B 结果（steps=40，W1/2/4，warmup=3，repeat=8，ABBA+45s 冷却）

全部 16 个进程组 PASS：双 Rank `complete PASS`、全窗口 `validated=true`、
0 bit-exact FAIL、0 非零 V2 状态、0 507014/507018。每轮 Gain 取较慢 Rank 会话中位数
（max-rank median 近似）。

### 5.1 M=4096（4 轮 ABBA，满足计划 ≥3 轮统计要求）

| 轮 | Stock W1 | V2 W1 | Gain(W1) | Stock W2 | V2 W2 | Gain(W2) | Stock W4 | V2 W4 | Gain(W4) |
|---|---|---|---|---|---|---|---|---|---|
| r1 | 131.5 | 108.7 | +17.3% | 109.5 | 108.0 | +1.4% | 101.4 | 107.2 | −5.8% |
| r2 | 134.0 | 108.0 | +19.5% | 111.0 | 107.9 | +2.8% | 100.6 | 107.1 | −6.5% |
| r3 | 138.4 | 108.8 | +21.4% | 112.5 | 108.1 | +3.9% | 101.7 | 107.2 | −5.4% |
| r4 | 138.7 | 107.8 | +22.3% | 113.2 | 107.1 | +5.4% | 100.8 | 105.7 | −4.8% |

（单位 ms/40 次调用；P95 见证据包 `analysis/combined_analysis.json`，与中位数偏差 <2%，除 r4 W1 Stock P95=150.3 一处）

- `Gain(W4)−Gain(W1)`：轮 1–2 会话 **−24.56pp**，轮 3–4 会话 **−26.98pp**；4 轮逐轮符号一致。
- W1 增益 +17%~+22% 与 T2 历史同步微基准 +17.9% 同量级：**T2 同步优势可复现，基线未漂移**（排除计划 §6 模式 3）。

### 5.2 M=2048 / M=3072（各 2 轮 ABBA）

| 形状 | Gain(W1) | Gain(W2) | Gain(W4) | Gain(W4)−Gain(W1) |
|---|---|---|---|---|
| M=2048 | +9.1% / +7.1% | −18.5% / −20.3% | −32.5% / −35.0% | **−41.8pp** |
| M=3072 | +7.3% / +13.5% | −19.3% / −7.7% | −33.3% / −23.0% | **−38.5pp** |

### 5.3 enqueue 耗时（max-rank 中位，40 次调用合计）

| 形状 | Stock enqueue | V2 enqueue | 结论 |
|---|---|---|---|
| M=4096 | 4.9–12.9 ms | 1.4–3.1 ms | V2 提交侧便宜 2–4× |
| M=2048 | 5.6–6.3 ms | 1.4–2.3 ms | 同上 |
| M=3072 | 4.9–10.0 ms | 1.5–2.9 ms | 同上 |

## 6. 归因分析（derived）

1. **V2 的 wall 耗时几乎不随窗口变化**（per-call：M4096 2.72→2.68ms，−1.4%；
   M2048 1.39→1.34ms；M3072 2.05→2.01ms）。融合 Kernel 路径没有从"少同步"中获得
   任何流水线收益——40 次调用近似串行执行。
2. **Stock 的 wall 耗时随窗口显著下降**（M4096 3.29→2.53ms，−23%；M2048 −34%；
   M3072 −31%）。`async_op` HCCL 与后续 MM 在窗口内形成重叠，是 Stock 在 W2/W4
   反超的唯一来源（enqueue 本身不足 0.2ms/次，非主因）。
3. **enqueue 数据排除了 Host 提交侧解释**：V2 的 C ABI 提交成本更低，若瓶颈在
   Host，V2 应同样受益；实际 V2 flat，说明瓶颈在设备侧执行链（SDMA 等待/SQ 资源/
   Kernel 串行并发，具体比例本轮未测）。
4. **形状越小反转越深**（M2048 −41.8pp > M3072 −38.5pp > M4096 −24.6/−27.0pp）：
   M 越小 MM 越快、AR 占比越高，Stock 异步流水线相对收益越大；V2 融合路径的固定
   串行成本则被放大。

**模式判定（计划 §6 表第 1 行）**：W1 V2 快、W4 V2 变慢 → 连续异步排队可能损害
融合优势。后续动作：追踪设备 SDMA 等待、SQ 资源与 Kernel 并发（本轮 enqueue 时序
已按计划 G3"host enqueue 时序采集"分支完成，未启动设备 profiler——留待下轮
`计划 §6` 指定的定向取证）。

## 7. 限制声明（不可外推）

- 40 次独立调用是**有界异步提交压力包络**，不是 Qwen 40 层完整依赖图；层间
  attention/GDN/MoE 依赖、真实 Serving 的 Prefill/Decode 交叠未在本轮复现。
- max-rank median 是会话中位数近似，不是逐次配对 max 的中位数；M=2048/3072 仅
  2 轮 ABBA（探索性），M=4096 4 轮符号一致但未做正式显著性检验。
- 无 profiler 计时，本轮不能把延迟差单独归因于 HCCL AICPU Cache 或 SDMA 某一环节。
- 计时期间设备 0 曾被他人在先作业使用（§4），但所有正式会话均在双设备空闲独占下
  执行；温度 67–74°C 正常区间。

## 8. 交付物清单

| 交付物 | 位置 |
|---|---|
| 本报告 | `docs/V2_SYNC_ASYNC_EXECUTION_REPORT.md` |
| 脱敏证据包 + 索引 | `evidence/v2_sync_async_20261021/`（evidence_index.md + 6 会话结构化记录 + 自动分析 JSON） |
| 自动分析 JSON | `evidence/v2_sync_async_20261021/analysis/combined_analysis.json` |
| 测试驱动兼容性修复 | commit `4761e0685`（交错记录解析 + 原子 emit + 回归 UT） |
| 环境未变更证明 | `evidence/v2_sync_async_20261021/env_unchanged_proof/env_unchanged_proof.txt`（Host SO/KVER11 JSON SHA 前后一致；主 worktree 未动；未安装/覆盖/卸载任何库） |
| 服务器本地原始日志 | `/tmp/v2_sync_async_{smoke,m4096,m2048_3072}_20261021_*`（含 run_index SHA256 自校验，不公开） |

## 9. 结论与下一步

本轮达到计划最低成功标准：**明确区分了同步与有界连续异步条件下 Stock/V2 的相对
性能变化（W1 一致正收益 → W4 一致负收益，跨 3 形状 8 轮 ABBA 符号一致），并排除了
资源重建与预热成本的干扰**（建池/进程组一次创建、预热不计时、ABBA+反向窗口+冷却）。

下一步建议（按优先级）：

1. 定向设备侧取证（计划 §6 第 1 行动作）：SDMA 等待、SQ 深度、Kernel 并发度，
   解释 V2 flat 的具体瓶颈环节；
2. 真实 Serving 复现实验：Prefill/Decode 交叠 + 路由覆盖率统计，验证微基准 W4
   反转与 G3 Eager −3.51% 吞吐的对应关系；
3. M%chunk_rows≠0 的 Input Stage 补齐路径（M=2049/4095）单独立项，不混入本轮结论。
