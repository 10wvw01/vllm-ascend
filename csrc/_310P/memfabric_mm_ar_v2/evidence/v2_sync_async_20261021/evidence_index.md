# V2 同步/连续异步 ABBA 实机证据索引（evidence_index.md，2026-10-21）

> 生成：2026-10-21（V2_SYNC_ASYNC 专项验证轮，OpenCode @ 310P3，分支
> `perf/310p-v2-sync-async-value-20261021`）。
> 用途：为本轮全部实机会话建立可核验索引，供独立复核。
> 证据分级：`observed`=日志直录；`derived`=带推导链的算术结论。
> 原始完整日志仅存服务器本地 `/tmp/v2_sync_async_*_20261021_*`（含
> run_index.jsonl 的逐日志 SHA256 自校验）；本目录只含脱敏结构化节选。

## 1. 会话清单与结果

| 会话（服务器本地路径） | 配置 | 结果 | 关键数字（observed） |
|---|---|---|---|
| `/tmp/v2_sync_async_smoke_20261021_01` | G1 Smoke 首跑（steps=8, repeat=3） | **解析器 BUG 中止**（非设备故障） | 3/4 进程 PASS；r2_stock rc=0 双 Rank 实际 complete PASS，但 5 条记录交错拼接导致行级解析丢弃 |
| `/tmp/v2_sync_async_smoke_20261021_02` | G1 Smoke（修复后重跑） | **ALL PASS** | 4/4 进程 rc=0，双 Rank complete PASS；全部窗口 validated=true |
| `/tmp/v2_sync_async_m4096_20261021_01` | G2 正式 M4096 第 1 次 | **P0 环境中止** | r1_stock rank0 设备 OOM：外部进程 PID 3873049 占用 device0 42652MB（99% DDR）；详见 `p0_device0_occupation/` |
| `/tmp/v2_sync_async_m4096_20261021_02` | G2 正式 M4096 轮 1–2 | **ALL PASS** | 4/4 进程；W1/W2/W4 全部 validated |
| `/tmp/v2_sync_async_m4096_20261021_03` | G2 正式 M4096 轮 3–4（统计加固） | **ALL PASS** | 4/4 进程；W1/W2/W4 全部 validated |
| `/tmp/v2_sync_async_m2048_3072_20261021_01` | G2 扩展 M2048+M3072 轮 1–2 | **ALL PASS** | 8/8 进程；全部窗口 validated |

- 全轮次 0 次 bit-exact FAIL、0 次非零 V2 协议状态、0 次 507014/507018、0 次 900s 超时。
- 全部 6 个有效进程组均要求 `torchrun --standalone --nproc-per-node=2`，每进程组
  内 Stock HCCL 进程组与 V2 trial 池各只创建一次，计时循环不建池。

## 2. 目录内容

| 路径 | 内容 | 分级 |
|---|---|---|
| `sessions/*/provenance.json` | 分支 SHA、KVER11 JSON SHA、Host SO SHA、设备映射 | observed |
| `sessions/*/run_index.jsonl` | 每日志 SHA256、退出码、complete Rank 清单 | observed |
| `sessions/*/*.records.jsonl` | 逐日志脱敏 V2_SYNC_ASYNC JSON 记录（含双 Rank 中位/P95/enqueue 原始分位） | observed |
| `sessions/*/per_rank_detail.json` | 每 Rank 每窗口 wall/enqueue 分位数结构化提取 | observed |
| `sessions/*/sync_async_summary.json` | 运行器同款汇总器输出（max-rank median 口径 + limits 声明） | derived |
| `analysis/combined_analysis.json` | 全形状全轮次 Gain(W)、Gain(W4)−Gain(W1) 自动分析 | derived |
| `parser_bug_differential/` | 交错行原始节选 + 修复前后解析差分 | observed+derived |
| `p0_device0_occupation/` | device0 被外部进程占用期间的 npu-smi/proc-mem/dmesg/ps 快照 | observed |
| `env_unchanged_proof/` | 运行前后 Host SO/KVER11 JSON SHA 一致性、主 worktree 未变更 | observed |

## 3. 核心结果（derived，详见 REPORT）

| 形状 | ABBA 轮数 | Gain(W1) | Gain(W2) | Gain(W4) | Gain(W4)−Gain(W1) |
|---|---|---|---|---|---|
| M=4096 | 4 | +17.3/+19.5/+21.4/+22.3% | +1.4/+2.8/+3.9/+5.4% | −5.8/−6.5/−5.4/−4.8% | −24.6pp（轮1–2）/ −27.0pp（轮3–4） |
| M=2048 | 2 | +9.1/+7.1% | −18.5/−20.3% | −32.5/−35.0% | −41.8pp |
| M=3072 | 2 | +7.3/+13.5% | −19.3/−7.7% | −33.3/−23.0% | −38.5pp |

每轮 Gain 定义 `100%×[1−T_V2(W)/T_Stock(W)]`，T 取该轮较慢 Rank 会话中位数
（max-rank median 近似，非逐次配对 max 的中位数）。

## 4. 脱敏说明

- `/home/<user>/...` 统一替换真实家目录；extension 路径替换为
  `<REPO_ROOT>/vllm_ascend/vllm_ascend_C...so`（原始完整路径在服务器本地日志与
  provenance 中保留，运行器在进程内硬校验其必须位于本 perf worktree 下）。
- 本轮计时与校验日志不含 GVA 地址；诊断探针未启用，无内部地址泄露面。
- 原始日志（含 CANN/torch INFO 前缀噪声）不进入仓库，仅保留结构化 JSON 记录。
