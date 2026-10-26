# Gate E S3-R 证据包：DeviceID 修复版 Host SO 受控冒烟回归（2026-10-26，设备 2,3）

> 入库内容为**脱敏节选**；完整原始证据仅存服务器本地（会话 `S3R_FUSED_SMOKE_20261026T150457Z`
> 与作废轮 `…T145859Z`），不入公开仓库。
> 完整分析见 `docs/GATE_E_S3R_SMOKE_REPORT_20261026.md`。

## 结果一句话

修复版（`1dc46d7d…`，mf-fix `5802f550`）以 **LD_PRELOAD 进程隔离**装载（生产双位置零改动、
前后哈希断言）：**H1 的 507899 阻断消失**（双 Rank meta GVA 授权在设备 2,3 全通过——修复回归
PASS）；但暴露下一个独立阻断——**卡 16384 上 KFC `system` 内核发射返回 507018
（ACL_ERROR_RT_AICPU_EXCEPTION）**，编排器部署链/`smem_shm_create` 失败，Worker C++ terminate，
0 请求完成，首调 bit-exact 未达成。裁决 `INVALID/UNKNOWN (SMOKE_BLOCKED_BY_KFC_LAUNCH_507018)`。
 guardian 模式表缺 507018 签名未触发（偏差已入案，4 个 0.04 秒空轮无设备接触）。

## 文件清单

| 文件 | 内容 |
|---|---|
| `s3r_a1_version_review.txt` | 新旧 SO 哈希核验、修复提交 diff 范围、构建溯源（含旧 configure 戳说明） |
| `s3r_a2_abi_comparison.txt` | ABI 合同：导出符号 1764=1764 逐名一致、DT_NEEDED/SONAME 一致、+3 getenv/+3 strtoul 修复静态判别 |
| `s3r_b2_isolated_deployment.txt` | 进程隔离验证：私有目录前置失败样例 + LD_PRELOAD 机制 CPU 验证（复刻脚本二次 source 序列） |
| `s3r_incident_launch1_preload.txt` | 第 1 次发射作废事故（脚本二次 source 导致隔离失效，零请求零数据点）与修复 |
| `s3r_serve_failure_excerpt.log` | 正式轮 serve.log 节选：80 选层计数、meta GVA 通过（507899=0）、KFC 507018 全链、Worker terminate |
| `s3r_maps_identity.txt` | 运行中双 Worker maps：修复版 SO 自私有目录装载、C 扩展/设备 SO 自隔离树（无混树） |
| `s3r_runner.log` | forensic runner 全量（healthy 80 层 → 5 reps 0 完成 → exit 0 弱点） |
| `s3r_bench_results.txt` | 5 个 stability_rep 结果（全 0 完成；rep1 28.7s，reps 2–5 各 0.04s） |
| `s3r_assessment.json` | 离线判读器输出（INCONCLUSIVE_EVIDENCE） |
| `s3r_guardian.log` | 监护日志（未触发——模式覆盖偏差记录） |
| `s3r_crosscard_comparison.txt` | 跨卡只读对比（DDR 51% 为 3/4 卡共性，无 16384 特有宿主可见差异） |
| `s3r_recovery_verification.txt` | 恢复核验（生产双位置 88d39285 未变、进程/端口/设备基线/worktree 全过） |
| `manifest.txt` | 本目录 SHA256 清单 |

## 关键版本身份

- vllm-ascend：`diag/310p3-g1d-minimal-forensics-20261021`（基线 `cde4db360`→本轮增补；冻结源
  `e7ba8a5b3`）；V2 设备 SO `b8ca96f0…`/`8447a55a…`、`vllm_ascend_C=07c831f0…`（隔离树标准构建）
- Host SO（本轮实际装载）：修复版 `1dc46d7d…`（LD_PRELOAD 私有目录）；生产位保持 `88d39285…`
- KVER11：JSON `25e16b0a…` 锁定；CANN 9.1.0 / 驱动 26.1.1；设备 NPU 16384 双 Die 2,3
