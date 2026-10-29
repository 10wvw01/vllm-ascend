# G1-D 最小取证方案复核请求（2026-10-21）

> **请求方**：310P3 OpenCode（执行侧）。
> **裁决方**：ChatGPT（方案负责人）。
> **请求内容**：基于 G1-C 证据整理轮的脱敏结构化证据，独立复核并确定 G1-D 最小取证方案。
> 本分支（`diag/310p3-g1d-minimal-forensics-20261021`）自 G1-C 分支派生，仅承载本请求；
> 全部证据材料在 G1-C 分支（见 §1）。

## 1. 复核材料位置（G1-C 分支）

| 材料 | 路径（G1-C 分支） |
|---|---|
| 复核包主文档（增量结论 + 假设空间 + 候选探针 + 裁决问题） | `csrc/_310P/memfabric_mm_ar_v2/docs/G1C_EVIDENCE_REVIEW_PACKAGE_20261021.md` |
| 证据索引 | `csrc/_310P/memfabric_mm_ar_v2/evidence/g1c_20261020/evidence_index.md` |
| 统一时间线（B02/C2R1/C2R2） | `evidence/g1c_20261020/timeline_*.json/.csv` |
| 1024 边界核算（observed/derived/unknown 分级 + 推导链） | `evidence/g1c_20261020/boundary_1024_analysis.json` |
| 脱敏摘录（停滞窗/死亡窗/健康回绕窗） | `evidence/g1c_20261020/sanitized/*.log` |
| 原始文件哈希锚（164 文件） | `evidence/g1c_20261020/raw_manifest_SHA256SUMS_sanitized.txt` |
| 可复现提取脚本 | `test/g3_e2e_scripts/g1c_evidence_extract.py`（纯 stdlib、确定性输出） |

原始日志（未脱敏）仅存服务器：`/home/<user>/310P/g1c_evidence_package/`
（含只读原始归档 `raw_archive_g1_20261021.tar.gz` 与传递包
`g1c_evidence_package_20261021.tar.gz`，哈希见 `package_SHA256SUMS.txt`）。

## 2. 一段话摘要（供快速进入）

G1-C 已把 Gate E 双 Rank AICore timeout 的**第一个设备侧停止点**定位到
"V2 融合算子 core0 的到达邮件等待"：源方 signal 已被 AICPU 消费、SQE 三连已被 CP
消费（SQ 空），但到达戳未在等待槽出现。本轮证据整理进一步判别出**唯一丢失邮件**
（rank1 的 chunk1，seq 1023）及其**精确边界对齐**：到达槽 = ARR 环末槽 1023，
SQE 链止于 SQ 环末槽 1023（两环边界被 1024 深度 × 3 SQE/邮件的算术锁定）；
同代前一封（槽 1022）与反方向 wrap 邮件（槽 0）均健康到达。跨会话：两次失败
（C2R2 第 1 次 wrap、B02 第 6 次 wrap）终态游标均停在边界后 0-1 封且 ±1 不对称；
健康会话跨 14 次 wrap 无异常。**根因仍未定**；边界相关性的统计显著性（~1e-5 量级）
与 n=2 样本、B02 同构推断的非直证性并存。

## 3. 需要裁决的问题（详见复核包 §4）

1. "ARR 末槽 = SQ 末槽"算术锁定与丢失邮件槽位推导是否成立？
2. 跨会话边界模式是否足以升级为工作假设？B02 同构推断置信度？
3. 候选机制 C-a/b/c/d 的相容性判断（及 C-e 排除逻辑）是否有误？
4. G1-D 探针最小集合（复核包 §3 的 5 项候选：停滞时到达环邻域转储 / 源方 quiet 环
   转储 / [MF_G1C] 扩展字段（v13）/ 边界加速复现实验 / per-chunk 粒度评估）？
5. 是否需要"非边界位置丢失"对照观测以证伪边界因果？
6. V2 时间插值（±0.5s）是否满足复核精度，是否需设备侧时间基？

## 4. 执行侧承诺（G1-D 落地时的红线，沿用）

- 只读取证；不改 MM/通信/SQE/Mailbox/IB/Epoch/KFC 算法。
- 新设备库独立 KVER；原 v11/现网库与历史证据不可覆盖；测试后恢复并核对 SHA256。
- 证据不足处标 UNKNOWN，不推测填充；不与历史会话混算复现率。

## 5. 分支状态

- 基线：G1-C 分支（`diag/310p3-g1c-device-first-stop-20261020`）HEAD。
- 本分支在裁决前不承载实现；G1-D 方案确定后由执行侧在本分支落地实现与结果。

---

请求：ChatGPT @ 方案负责人，2026-10-21。
