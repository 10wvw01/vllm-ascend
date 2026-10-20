# G1-B Host 飞行记录器执行补遗（2026-10-20 实机）

> 依据 `G1_GATE_E_FIRST_FAILURE_FORENSICS.md` §3 G1-B 与主报告
> `G1_GATE_E_FORENSICS_REPORT_20261020.md` 遗留项 L1 执行。
> G1-B 目的：判定故障链的**第一个停止环节**位于 Host 控制面（请求环/Epoch）还是计算面。
> 本补遗仅陈述事实与判别状态；根因仍未定，未执行任何协议修复。

## 1. 执行摘要

- G1-B 于 2026-10-20 13:00–13:30 UTC 执行完毕：诊断库编译、双位置备份/替换/恢复全链路完成，原库逐文件 SHA256 验证还原。
- **故障在记录器装载下复现**（会话 B02）：负载 ~6.6 分钟、第 29 个请求处（rep1 25/25 + rep2 4/25），双 rank 间隔 31ms 先后在 `aclnnSwiGlu` 调用点报 507014 AICore timeout——与 F01（无记录器）签名与**故障点完全一致**。
- **核心结论：直至计算面死亡的瞬间（最后 tick 距 rank0 首错仅 +163ms），Host 控制面完全健康**——请求环零积压（max reqPending=1，且为单请求在途瞬态）、游标全程有效、25×2 纪全部正常退出（exitReason=0/mainRet=0，时长 15.0–15.1s 吻合预算）、sqHead 严格 3 SQE/请求推进并干净回绕。
- **H1/H2 在 Host 可观测层面被强证据反驳；H4 显著增强；根因仍判"未定"，第一停止环节收窄至设备侧计算/通信路径**（V2 融合波、orchestrator AICPU kernel 或 stock kernel 交互），下一步按方案进入 G1-C。

## 2. 诊断库与替换记录（全链路哈希）

| 步骤 | 内容 | 验证 |
|---|---|---|
| 编译 | MemFabric 诊断 worktree `@eb96eced`（= 现网库源码 `131ee5ff` + 1 个飞行记录器提交），cmake+Ninja，RELEASE，参数与 10-19 现网构建（`/tmp/opencode/mf-fix`）一致；**未执行 install**（避免写 /usr/local） | 内嵌版本串 `commit: eb96eced034819093953323487a9bd96ec045b34`；`[MF_DIAG]`×4 字符串；ldd 依赖与现网库**完全一致** |
| 诊断库哈希 | `c03310c177f8013010234d44a22aca6176dc960a99d1ae103ffe4ba21966a30d` | — |
| 备份 | 双位置（`/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/` 与 site-packages `memfabric_hybrid/lib/`）均为 `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`；备份目录 `memfabric_hybrid-diag/build/g1b_backup_20261020T130001Z/`，**只读化**（a-w） | SHA256SUMS.txt + ownership.txt；备份自校验 OK |
| 替换 | 13:03 双位置替换为诊断库，权限修正 644 | 双位置 `c03310c1…` |
| 恢复 | 13:30 会话结束后立即恢复，权限 644 | 双位置回到 `88d39285…`，尺寸 3240592 与原态一致；`sha256sum -c` 备份清单 OK |

AICPU 侧（KVER=11 `libmf_sdma_orch.so`）全程未动。

## 3. 关键执行问题与处置（实机兼容修复）

1. **日志级别过滤**（未提前发现会导致 G1-B 整轮无效）：飞行记录器全部走 `BM_LOG_INFO`，而 hybm core 的 `OutLogger` 单例（各 .so 独立）默认 `WARN_LEVEL` 会静默过滤 INFO。无环境变量可覆盖（`MF_LOG_LEVEL` 仅被 smem 侧解析）。
   处置：`sitecustomize.py` 注入（仅 `MF_SDMA_ORCH_DIAG=1` 进程激活）——`ctypes.CDLL` 与包加载**同一路径**（SONAME 命中同一单例）调用导出符号 `hybm_set_log_level(1)`。已纳入版本化工具 `test/g3_e2e_scripts/g1b_sitecustomize.py`。B02 serve.log 中 9 个进程出现 `[G1B] sitecustomize: … rc=0` 确认生效。
2. **B01 作废**：注入时 `PYTHONPATH=$INJ` **覆盖**（而非前置）了环境原有 CANN Python 路径，worker `ModuleNotFoundError: No module named 'acl'`，13:05:12 启动失败（诊断库未被触达，非库问题）。B02 改为 `PYTHONPATH=$INJ:$PYTHONPATH` 后正常。
3. 启动命令 `env` 语法笔误（`-u` 位置）导致一次空启动，已修正（记录于 `tmp/opencode/gate_e_forensics/B01_RETRY_NOTE.txt`）。

## 4. 会话 B02 时间线（2026-10-20 UTC）

| 时刻 | 事件 |
|---|---|
| 13:12:03 | fused serve 启动（原 Gate E 工况 + `MF_SDMA_ORCH_DIAG=1` + 日志级别注入） |
| 13:15:24 | 健康（V2 层标记 80 = 40×2 rank） |
| 13:15:37 | 双 rank `[MF_DIAG] start epoch: 1`（devId 0/1，budgetS=15 默认纪预算确认） |
| 13:15:52 | 纪 1 退出（空闲纪，consumed=0）；此后纪周期 15.0–15.1s 稳定循环 |
| 13:16:38 | 双 rank 首调 V2 bit-exact 通过（M=4096） |
| ~13:15–13:22 | rep1 25/25 完成；rep2 完成 4 个 |
| 13:17:00→13:18:04 | 双设备 AICPU sharder `event num exceeded 1000` 告警（parallelId dev0 287→1615、dev1 322→1710） |
| 13:18:00/13:18:04 | **dev0/dev1 设备 slog 先后静默**（距宿主首错 ~4 分钟，比 F01 的 2–3 分钟更长） |
| **13:22:04.347** | **rank0（TP0 pid 4041279, Device 0）首错：`call aclnnSwiGlu failed, error code is 507014`（retCode=0x25 aicore timeout）** |
| **13:22:04.378** | **rank1（TP1 pid 4041289, Device 1）同型首错，间隔 31ms**（F01 为 18ms） |
| 13:22:04.510/.601 | **最后 tick（双 rank）**：epoch 26 运行中，reqPending=0，cursorsValid=1，mainRet=0——记录器工作至进程死亡 |
| 13:22–13:25 | rep3–18 连接失败（0 完成）；双 Worker 不可恢复 |
| 13:25:53 | 620s 窗口结束，18 reps；13:25:58 runner 退出码 0；设备显存回落基线，无 507899 |

## 5. 飞行记录器数据分析（870 行 MF_DIAG）

全量统计（serve.log 正则解析，脚本与原始日志存 `B02/`）：

| 指标 | 值 | 判读 |
|---|---|---|
| tick 总数 | 768（1Hz × 双 rank × ~6.4 min） | 采样连续，无缺口 |
| reqPending 最大值 | **1**（13:16:54.338，dev0，reqHead 1221/reqTail 1222，下一 tick 归零） | 单请求在途的非原子快照瞬态，正是文档 §4 预告情形；**无任何积压** |
| cursorsValid≠1 的 tick | **0** | 请求环游标全程有效，无回退/戳异常 |
| 纪退出总数 | 50（25/rank） | 纪 26 于死亡瞬间仍在运行（无 exit 记录，进程终止） |
| exitReason≠0 / mainRet≠0 | **0 / 0** | 无一纪异常退出；无重拉；无监督器告警 |
| 纪时长范围 | 15012–15129 ms | 全部吻合 15s 预算，无停顿、无超长 |
| 总消费 | 12291 件（dev0 6146 / dev1 6145） | 消费链吞吐健康直至死亡 |
| sqHead 不变量 | dev1 全程 25 纪零违例；dev0 仅纪 3–4 边界一次采样滞后（纪 2–4 聚合 0→27 = 9×3 精确成立，此后规整） | **3 SQE/邮件不变量成立**，sqHead 模 1024 干净回绕（如纪 21→22：808+74×3=1030≡6）；纪 3–4 偏差属"每纪尾部采样"已知语义，非 SQ 异常 |

## 6. 假设判别状态更新（对照主报告 §5）

| ID | 假设 | G1-B 证据 | 更新后状态 |
|---|---|---|---|
| H1 | SQ 容量失控 | 请求环零积压（max=1 瞬态）、游标全程有效、sqHead 严格 3/请求推进、无覆写签名 | **Host 可观测层面强证据反驳**。注：请求环≠CP SQ（文档明示），设备侧 SQ Head/Tail 仍无直采——彻底关闭需 G1-C 设备诊断版 |
| H2 | Epoch 消费链停顿 | 50/50 纪正常退出（exitReason=0/mainRet=0）、时长全部吻合预算、无重拉、纪 26 死亡时仍在正常运行 | **强证据反驳**（"重拉掩盖错误计数"路径未发生——无任何重拉） |
| H3 | 共享 Cache Line 风险 | 全程无游标回退/戳异常样本 | 无支持证据；按文档仍不可断言排除（需地址隔离 A/B） |
| H4 | V2 IB/同步死锁 | 控制面在死亡瞬间健康；双 rank 31ms 对称超时；表面错误在 stock 算子同步点；设备 slog 先静默 ~4 分钟 + sharder 事件堆积 | **显著增强（仍未证实）**——第一停止环节在计算面 |

## 7. 结论与下一步

1. **G1-B 回答了其设计问题**：第一个停止推进的环节**不在 Host 控制面**（请求环/Epoch/Supervisor 全程健康），而在**设备侧计算/通信路径**。
2. 跨会话故障点高度可复现：F01 与 B02 均死于第 29 个请求（rep1 25 + rep2 4）、负载开始后 ~6.6–7.2 分钟；F02R1 同工况 75/75 通过——触发呈状态/时序依赖，但触发后的崩溃点一致。
3. **根因仍未定**。按方案 §3 G1-C：需方案负责人提供**独立设备诊断版本**（更换 device .so 版本号避免 CP1 缓存）与 V2 各核低成本阶段戳，先静态审查再部署；本轮未涉及。
4. G1-A 与 G1-B 结果未混作复现率统计（fused 观测：G1-A 1/2，G1-B 1/1，各自计数）。

## 8. 环境恢复与证据索引

- 现网库已恢复：双位置 `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`，权限 644，尺寸 3240592，备份清单自校验 OK；设备 0/1 回基线占用，无残留进程。
- 证据目录：`tmp/opencode/gate_e_forensics/B02/`（serve.log 含 870 行 MF_DIAG、18 个 bench 结果、samples/ 15s 采样、evidence/device_logs/ 双 rank 设备 slog+plog、provenance 含 `mf_host_diag=1`）；B01 作废记录在 `B01_RETRY_NOTE.txt`。
- 诊断库构建现场：`memfabric_hybrid-diag/build/`（未纳入 Git，属过程件）。

---

执行：OpenCode @ 310P3 服务器，2026-10-20 13:00–13:30 UTC。
