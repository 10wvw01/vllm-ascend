# G1-D 实机证据索引（evidence_index.md，2026-10-21）

> 生成：2026-10-21（G1-D 实机执行轮 + 复审批次，OpenCode @ 310P3）。
> 用途：为 G1-D D0–D4 全部实机会话建立可核验索引，供独立复核。
> 证据分级：`observed`=日志直录；`derived`=带推导链的算术结论；`unknown`=无证据。
> 原始完整日志仅存服务器本地（`/home/<user>/310P/g1d_evidence_20261021_R1/`，
> 含 SHA256SUMS.txt 自校验、D1R4 设备 plog 29 件）；本目录只含脱敏节选与确定性结构化产物。

## 1. 会话清单与结果

| 会话 | 性质 | 结果 | 关键数字（observed） |
|---|---|---|---|
| C1OFF | TP2 C1，诊断开关全 OFF | **PASS 51/51 位精确** | 无 507014 |
| C1ON | TP2 C1，全诊断开启 | **PASS 51/51 位精确** | [V2_G1D] 112 同代返回；[MF_G1D] 142 帧 0 错 |
| D2W1 | 异步窗口=1 ×256 + Stock/HCCL | **PASS 256/256** | arrHead 终态 1302 |
| D2W2 | 窗口=2 ×256 | **PASS 256/256** | arrHead 终态 1307 |
| D2W4_ORIG_FAIL | 窗口=4 ×256（原版 harness） | **FAIL @ 第 5 调**：池槽耗尽守卫（两 Rank 同报） | 前 4 调已验证；Epoch 健康 |
| D2W4_PROBEA | 窗口=4 ×4（单窗口） | PASS | 失败锁定在跨窗口边界 |
| D2W4_PROBEB | 窗口=4 ×8（解绑引用+二次 gc 本地探针） | PASS | harness 残留引用定性 |
| D2W4_FIXED_FULL | 窗口=4 ×256（修复 harness，已提交） | **PASS 256/256** | elapsed 103.1s |
| D1R1 | Gate E 620s（fused Eager 原工况） | 未复现：75/75、0 错、0 超时 | gen 3008；arrHead/reqSeq 15027 |
| D1R2 | Gate E 620s | 未复现：75/75、0 错、0 超时 | gen 3008；arrHead 15027 |
| D1R3 | Gate E 1240s（长稳） | 未复现：150/150、0 错、0 超时 | gen 6008；arrHead 30027 |
| **D1R4** | **Gate E 620s（复审批次第 1 轮）** | **复现：双 Rank 同代 1730 停滞 stage50** | rep1 25/25 → rep2 17 → rep3-11 全 0；507014×8（08:34:49.395 / 08:34:58.078）；arrHead 冻结 8632/8634 |
| D2R1–D2R6 | G1-D2 tail 扩展复现批次（620s ×6，Host bd1760cc） | 全部未复现：各 75/75、0 超时 | tail 帧 711–760/会话/Rank，0 错误；终态 t1Stamp==arrHead 全成立 |

- 首轮三会话合计 300 请求 0 失败；V2 代际合计 12024；45081 件 AICPU 消费/设备；
  44 次 1024 边界穿越；`[MF_G1D]` 全程 0 snapshotError。
- D1R4（故障会话）：`[MF_G1D]` 449/459 帧（rank0/rank1）0 snapshotError；
  `[V2_G1D]` 同代返回 369/380；设备 plog 双 Rank aicore timeout 0x7150025（硬件超时 0x25）。
- 复现率合并（derived）：G3 1/2 + G1-C 1/2 + G1-D 1/10 = 3/14 ≈ 21.4% 会话
  （G1-D 时代：D1R1-R3 净、D1R4 故障、D2R1-R6 净）。
- 设备占用阻断记录两段：首轮 04:41–05:00Z（`blockers/`）。
- 复审批次编排：6 轮预算、任一轮复现即停——D1R4 第 1 轮复现，D1R5–R9 未执行
  （`batch2_state.txt`）。

## 2. D1R4 故障窗口判别证据（observed，详见报告 §6.3）

| 证据 | rank0/dev0 | rank1/dev1 |
|---|---|---|
| V2 停滞（8 核一致） | gen 1730，core0 `1730:0:50`，core1-7 stage60 | gen 1730，core0 `1730:2:50`，core1-7 stage60 |
| 被等待邮件 | chunk0，seq 8632（dev0 环最后到达，槽 439） | chunk2，seq 8634（dev1 环最后到达，槽 441） |
| 同代已成功返回 | 无（gen1728 c3 seq8625 为末次） | **c1 seq 8633 status=0**（槽 440，与被等槽 441 相邻） |
| 本端 arrHead（Host D2H） | 8632 冻结 ≥13s（08:34:36 前起 14 帧不变） | 8634（08:34:50.1 自 8632 跳入后冻结） |
| SQ / Epoch / 编排 | hwHead=hwTail=302 空、rc=0；纪 30 运行；phase 8 空闲 | hwHead=hwTail=308 空；纪 31 运行；phase 8 |

判别结论（derived）：
1. stage50 非 52/53 → wait 从未返回，排除"返回后校验失败"分支。
2. 被等邮件均已被 Host D2H 视角计数入环且环绕自洽 → "D2H 可见 vs AICore 不可见"。
3. 被等槽 439/441 环中部 → "仅 1023 边界"假设故障侧直接反证。
4. 剩余 unknown：被等槽（head−1）stamp/mail 内容未采样（动态头采样恰差 1 槽），
   不能区分"写入丢失"vs"AICore 读不可见"。

## 3. 边界取证（boundary_1024_analysis.json，observed）

各会话流量结束后静默帧（D1R4 为故障冻结帧）边界槽实值：

| 会话 | dev0 arrHead | stamp1020..1023 | mail1023Seq | 备注 |
|---|---|---|---|---|
| D2W1 | 1302 | 1021..1024 | 1024 | 健康终态，连续正确 |
| D2W2 | 1307 | 1021..1024 | 1024 | 同 |
| D2W4_FIXED | 1307 | 1021..1024 | 1024 | 同 |
| D1R1 | 15027 | 14333..14336 | 14336 | 健康终态 |
| D1R2 | 15027 | 14333..14336 | 14336 | 健康终态 |
| D1R3 | 30027 | 29693..29696 | 29696 | 健康终态 |
| D1R4 | 8632（冻结） | 8189..8192 | 8192 | **故障冻结帧**：lap-8 边界值自洽；被等槽 439/441 不在固定采样集 |

`[V2_G1D]` 同代邮件返回覆盖：D2 seq 1..1300+；Gate E 首轮全部会话；D1R4 至 gen1730
（含被等前的正常返回 seq 8633/8625）。

## 4. 结构化产物与生成方式

- `session_stats_*.json`：确定性统计（行数、tag 计数、max arrHead/generation、507014
  计数、源日志 SHA256）。生成脚本：G1-D 执行轮证据打包器（离线、确定性）。
- `boundary_1024_analysis.json`：每会话 dev0/dev1 末帧 `[MF_G1D]` 全字段。
- `sanitized/*.log`：关键 tag 首/中/末行（路径脱敏）；`D1R4_FAULT_WINDOW.log` 为故障
  窗口判别证据行专项节选（gen1730 帧、被等邮件返回、arrHead 冻结帧、507014 签名）。
- `raw_manifest_SHA256SUMS_sanitized.txt`：本地原始证据全量 SHA256（路径脱敏）。
- `extract_manifest.json`：源日志路径与哈希。

## 5. unknown（禁止推测补全）

1. 被等待槽（dev0 槽 439 / dev1 槽 441）stamp/mail 内容——区分"写入丢失"vs
   "AICore 轮询读不可见"的唯一直接证据缺口（采样盲区：动态头恰差 1 槽）。
2. 双 Rank 对称丢失是否同因（同窗口对称表现 vs 两件独立丢失）。
3. 发送侧 AICPU 在环末封后停止写入的归属（对端反压致请求环耗尽 vs 写链自身异常）。
4. 是否存在低概率"慢等待自恢复"（三会话一致为"即时完成或永久挂起"，样本量不足）。
5. 外来设备占用与 G1-D 会话无共享设备时段，但主机侧共享资源未隔离测量（derived 边界）。
