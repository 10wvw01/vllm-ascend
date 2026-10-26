# Gate E H1 融合冒烟证据包（2026-10-26，310P3 设备 2,3 / NPU 16384）

> 入库内容为**脱敏节选**；完整原始证据（全量 serve.log、双 Worker plog、15 秒级 npu-smi 采样、
> 全部 S0–S3 过程证据）仅存服务器本地，不入公开仓库。
> 会话：`H1_FUSED_SMOKE_20261026T140451Z`（另有作废事故轮 `…T135319Z`，仅编排事故，无设备数据点）。
> 判读结论与完整分析见 `docs/GATE_E_S0_S3_SMOKE_REPORT_20261026.md`。

## 结果一句话

90 秒融合冒烟在**首调 V2 传输层初始化**即失败：双 Rank `AclrtMemSetAccess` 返回 507899
（`ACL_ERROR_RT_DRV_INTERNAL_ERROR`）→ `hybm_init`/`smem_shm_init` 失败 → 双 Worker 崩溃，0 请求完成。
**不是 Gate E 507014 复现**（负载从未建立）；同一安装库 10-20 在设备 0,1 曾静默成功。
裁定 `INVALID/UNKNOWN (SMOKE_BLOCKED_BY_CARD_INIT_FAULT)`，已按停止契约停止全部后续轮次。

## 文件清单

| 文件 | 内容 |
|---|---|
| `h1_serve_failure_excerpt.log` | serve.log 关键节选：80 选层行计数、健康、双 Rank 507899 失败段、Worker 异常头 |
| `h1_runner.log` | forensic runner 全量（healthy 80 层 → 7 reps 0 完成 → exit 0 弱点示例） |
| `h1_summary.txt` | 脚本自动摘要（初筛 UNKNOWN） |
| `h1_assessment.json` | 离线判读器输出（FAULT_SIGNATURE_OBSERVED[机械字样]，人工改判见报告 §9） |
| `h1_maps_identity.txt` | 运行中 /proc maps：双 Worker SO 装载身份（隔离树 + 生产位，无混树） |
| `h1_s1_build_assertions.txt` | 标准构建产物 SHA256、DT_NEEDED、无诊断后缀、主树不变断言 |
| `h1_recovery_verification.txt` | 环境恢复核验（生产位哈希/进程/端口/设备基线/worktree） |
| `h1_post_failure_device_state.txt` | 故障后设备状态（回基线、无残留进程、Health OK） |
| `h1_device_slog_excerpt.log` | 双 Rank 设备 slog/plog 故障窗口节选（末活动 `Get api RunCpuKernel success`） |
| `manifest.txt` | 本目录 SHA256 清单 |

## 关键版本身份（详见报告）

- vllm-ascend：`diag/310p3-g1d-minimal-forensics-20261021@e7ba8a5b3636f045c19f777ea11dee6856d1466d`（冻结交付 SHA）
- 构建产物：`libmf310p_v2_trial_device.so=b8ca96f0…`（与主树标准件逐字节一致）、
  `libmf310p_device.so=8447a55a…`、`vllm_ascend_C=07c831f0…`（隔离 worktree 标准构建）
- MemFabric 安装：Host 双位置 `libmf_hybm_core.so=88d39285…`（发射前后断言）、v11 so `db7134b8…`、
  v11 JSON `25e16b0a…`（KVER11 锁定）
- 环境：CANN 9.1.0 / 驱动 26.1.1（V100R001C10SPC009B220）/ vllm 0.27.1+empty / torch 2.10.0+cpu /
  torch_npu 2.10.0.post4 / Python 3.12.13
