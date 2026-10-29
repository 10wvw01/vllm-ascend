# G1：Gate E 首故障点取证设计与执行规范

> 方案及代码责任：ChatGPT；310P3 上的 OpenCode 仅按本文执行、修复实机环境/编译问题、采集及汇总事实。  
> 本轮禁止修改 SQ 容量算法、Mailbox 布局、Epoch 重拉算法和 V2 MM/通信同步算法。

## 1. 冻结的事实与版本

- vllm-ascend 原验证基线：`518418f20cf8f09f2c418993400afaf639a101fc`（G3 失败发生在此前版本，最新提交仅文档归档）。
- MemFabric 独立研发仓库：`10wvw01/memfabric_hybrid_310p`，`main=131ee5ffd1a6ae9d4824cbb0cf4d32d99bbfdc5a`。
- 安装运行态是混合产物：KVER=11 AICPU/SMEM 来源于 `2f59dedb` 的安装包；Host `libmf_hybm_core.so` 经 10/19 热替换为 `131ee5ff` 对应代码，Python site-packages 亦替换。**版本元数据本身不能证明实际加载库版本**。
- 既有 G3 Gate E：TP=2、Eager、input 4096、output 256、concurrency 10，首次 ~3 分钟双 Worker AICore timeout，3/775 完成；同工况重跑 fused 75/75 成功，stock 75/75 成功，**根因未定**。
- 最新重构回归只覆盖功能/拆分，**未重新测试 Gate E**。

## 2. 排查假设（明确区分事实和推断）

| ID | 目前可以确认的源码事实 | 如何证伪 / 证实关联 |
|---|---|---|
| H1 SQ 容量失控 | `submitSqe` 的 `sqDepth - used < 1` 对有效环游标始终为假；`sqHead` 仅在 Epoch 起点查询；每封邮件需连续提交 3 个 SQE。 | 需要 SQ Depth/Head/Tail 的可靠设备侧采样和最早异常事件；仅 Host 请求环 backlog 增长 **不能** 证明 SQ 覆写。 |
| H2 Epoch 消费链停顿 | v11 依靠每 100ms Host 心跳、有限纪及 Supervisor 重拉；单纪错误可能被重拉清零错误计数掩盖。 | 取每纪退出、重拉时间、错误码、Host 心跳和请求环游标。 |
| H3 共享 Cache Line 风险 | `ReqTail(0x800)`、`ReqHead(0x808)`、`Quiet(0x810)` 混用同一 64B 行；`ArrMail[1023]` 与 `ArrHead` 同行。 | 若游标回退/戳异常需再做地址隔离 A/B。不可由静态布局直接断言丢写。 |
| H4 V2 IB/同步死锁 | Core0 执行 signal/wait/quiet/DONE，其他核依赖 WaitReady/ReleaseAdd/WaitAddDone。 | 需按 Core0 和其他核进度精确定位；双 Rank timeout 本身不指向 MemFabric。 |

## 3. 阶段划分与采集代码

### G1-A：保留全部现有设备/Host 二进制的无侵入取证

新增脚本：
- `test/g3_e2e_scripts/gate_e_forensic_run.sh`：按实验编号创建独立目录、Hash/Commit 清单、每 15 秒一次 npu-smi/进程快照，调用原 Gate E Eager 脚本；不会清理其他进程或覆盖历史日志。
- `test/g3_e2e_scripts/gate_e_forensic_summary.py`：聚合 JSON 完成数、失败数与日志异常字样，缺失数据输出 UNKNOWN，不把 0/0 误报 PASS。
- `gate_e_eager.sh` / `gate_e_eager_stock.sh`：仅修正输出路径、设备/模型可配置性，并锁定 fused 的 `v11.json`；原始形状、TP、并发、Eager 参数不变。

**执行序列**（必须顺序进行；同一物理两卡不允许并行跑两个 server）：

1. 开始前检查 Git 状态、现场进程/端口、npu-smi、MemFabric Host/Device/Python 库加载证据，保留现有环境的 SHA256、版本信息；非预期版本停下反馈，严禁自动重装。
2. 验证默认 15s Epoch；禁止使用 `MF_SDMA_ORCH_EPOCH_BUDGET_S=4294967295` 常驻诊断模式。
3. 执行 `fused` 两个独立 620s 会话；若第一次故障，先保全双 Rank plog/slog、AICore kernel dump/错误日志，再决定是否继续。之后以同工况 `stock` 做至少一轮对照。
4. 每轮独立输出，轮间确认前进程已退出、meta 区异步释放完成（历史环境需约 75s，复核无 507899 后继续），严禁未经判断的暴力清理。
5. 收集 V2 实际启用证据：40 目标层（每 rank 10 o_proj + 30 out_proj）、M 在 [2048,4096] 命中 V2，decode M<2048 Stock，双 rank 首调 bit-exact 校验。

服务器已存在的实机路径可以保持，调用示例（`cd` 为当前诊断分支中的脚本目录）：

```bash
# 仅设置本轮标识和固定预算，不改变默认两卡、模型和服务参数。
unset MF_SDMA_ORCH_DIAG
unset MF_SDMA_ORCH_EPOCH_BUDGET_S
export GATE_E_DURATION_S=620
bash gate_e_forensic_run.sh fused F01
# 确认回收与 meta 排空后：
bash gate_e_forensic_run.sh fused F02
# 同样按顺序：
bash gate_e_forensic_run.sh stock S01
```

所有输出自动定位到 vllm-ascend 工程内
`csrc/_310P/memfabric_mm_ar_v2/tmp/opencode/gate_e_forensics/<实验编号>/`，必须确保 `tmp/` 被 Git 忽略。

### G1-B：仅在 G1-A 证据不足时启用可关闭的 Host 飞行记录器

MemFabric 独立诊断分支对应源文件：
`src/hybm/csrc/entity/hybm_sdma_orchestrator.cpp`。

启用 `MF_SDMA_ORCH_DIAG=1` 后，Supervisor：
- 每约 1 秒多做一次 16B D2H（请求游标 ReqTail/ReqHead 的非原子快照），记录有效性、积压以及 epoch/hb。
- 每纪退出打印 `exitReason / mainRet / consumed / sqHeadAfter / durationMs`，每次新纪打印 epoch 序号。
- 此记录器**没有**实时硬件 SQ Head/Tail；`sqHeadAfter` 仅是每纪尾部采样，`consumed` 只在退出快照可靠。
- 采样量小于现有 10Hz heartbeat/result 轮询，但增加一次同步 D2H，仍可能改变时序，**G1-A 与 G1-B 不能混作等价复现率统计**。
- 关闭变量时不增加 D2H、不打印飞行日志，不改 AICPU Kernel、wire ABI 或 SQ 提交算法。

实施限制：
1. 在 MemFabric 诊断 **worktree** 中编译 Host core.so，先静态检查和动态依赖检查；必须对已有 `/usr/local` 及 site-packages 的真实加载 SO 分别进行不可覆盖的备份、哈希验证。
2. 只有在确认当前无其他 NPU 任务且具备恢复步骤后，才对本测试进程使用诊断 Host 库；**不能重装旧 .run/wheel 或用 dev-v1.2.0 的 KVER=9 覆盖**。
3. 启动日志应确认 `[MF_DIAG]` 出现；若无，则证明诊断库未被实际加载，结果无效。
4. G1-B 完成后恢复两个位置原始 Host 库，逐文件验证备份 SHA256。

### G1-C：是否需要实时 SQ / V2 Core 进度的决策点

若 G1-A/B 出现明确 timeout，但 `SQ` 及 `V2` 首停止点仍无法确认，停止本轮“猜测式修复”。由方案负责人下一步提供**独立设备诊断版本**（更换 device .so 版本号避免 CP1 缓存）及 V2 各核低成本阶段戳，先静态审查再部署；OpenCode 不得自行变更 Mailbox ABI 或使用 `printf` 风暴埋点。

## 4. 分类规则和交付

- `REQ_TAIL-REQ_HEAD` 仅代表**请求环**积压，不能直接解释成 **CP SQ** 已满；读数来自不同写入者的非原子快照，瞬时无效需结合前后样本。
- `MF_DIAG epoch ERROR` + `mainRet` 是控制面异常强证据；但要用时间线确认发生在 AICore timeout **之前**。
- `AICore wait/IB` 是表象；只有前置阶段标记或设备栈证据，才能判定哪一端先停。
- 只要有一轮 Gate E hang，先保存完整日志，禁止清理现场后再声称“无法复现”。
- 两轮无挂死仅记 **未复现**，不等于根因修复。
- 因 stock 自身存在 greedy 非确定性，不能把所有文本差异直接当作 V2 协议错误；以真实算子 bit-exact、协议状态和服务错误为准。
- 正式报告至少包含：每次会话准确启动/失败/退出时间、各 rank 最早错误、每 epoch 退出/重拉时序、已完成/错误请求数、部署库哈希、设备日志位置、能否排除 H1-H4。根因未定时写明“未定”，不得直接执行协议修复。

## 5. 固定的 OpenCode 权限边界

仅做：备份、编译、环境差异修复、实际执行、日志采集、初步事实汇总；如遇本方案脚本的小型实机兼容 BUG，可修复并记录 Diff。  
不做：重新设计数据协议、决定根因、修改 SQ/Cache Line/Epoch 算法、合并到 `main`、把 `tmp/opencode` 过程文件推到 Public GitHub。

本阶段不生产长稳修复成果；结束后由方案负责人根据现场证据决定下一步最小修复补丁。
