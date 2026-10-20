# G1 Gate E 长稳故障专项诊断报告（2026-10-20 实机执行）

> 执行方：OpenCode（实机执行、环境修复、证据采集与事实汇总）。
> 方案与判读框架依据：`G1_GATE_E_FIRST_FAILURE_FORENSICS.md`（诊断分支 `diag/310p3-gate-e-forensics-20261020`）。
> 本报告仅陈述实机事实与假设判别状态；**根因未定**，未执行任何协议修复。

## 1. 执行摘要

- **G1-A 完整执行**：fused 2 轮（F01、F02R1）+ stock 对照 1 轮（S01），原 Gate E 工况（TP=2、Eager、input 4096/output 256、25 prompts/轮、并发 10、620s/轮）。
- **F01 复现了 G3 Gate E 故障签名**：负载开始后约 5 分钟（rep2 进行中），双 rank Worker 在 18ms 内先后于 stock 算子 `aclnnSwiGlu` 调用点报 507014（AICore execution timeout），双 Worker 不可恢复。故障前 29 个请求全部正常完成。
- **F02R1（75/75）与 S01（75/75）干净**，与既有 G3 画像一致：**间歇性、低概率、高后果**。
- **设备侧证据**：双设备 AICPU slog 在宿主超时检测前 2 分 06 秒（dev0）/ 3 分 06 秒（dev1）先后静默；静默期间 AICPU sharder 事件告警持续累积。宿主侧 Host 库全程无 ERROR 级 epoch 事件。
- **用户追加场景 D01R2**（stock + FULL_DECODE_ONLY 图模式 + output 2048 + 50 prompts，无投机推理）：50/50 完成、0 错误、0 超时。
- **H1–H4 均无法排除**；H4（V2 通信/同步死锁）获得一致性支持但未被证实。**G1-B（Host 飞行记录器）为遗留项**。

## 2. 冻结的执行环境（启动前核验）

| 项 | 值 | 验证方式 |
|---|---|---|
| vllm-ascend 基线 | `518418f20cf8f09f2c418993400afaf639a101fc`（分支 `feat/310p-memfabric-public-api`，editable 安装，tracked 文件无修改） | `git rev-parse` + `status --porcelain` |
| vllm | `0.27.1+empty`（editable，`/vllm-workspace/vllm`） | `pip show` |
| MemFabric Host core.so | SHA256 `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`，`/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/` 与 site-packages `memfabric_hybrid/lib/` **两位置哈希一致**（即 10-19 热替换后的 `131ee5ff` 等价构建） | `sha256sum`，与当日资产审计报告交叉比对 |
| orch_deploy_helper | `849f93c0b4a96877d3e839ae7ceec4b6c479a8de331213f0d7a270c1198bd310` | 同上 |
| AICPU 库 libmf_sdma_orch.so | `db7134b8a2fa56f7250097c36d6a530d38901743d5eb8055bf9168bb52aee401`（10-12 安装包，`2f59dedb` 产物，此后未变） | 同上 |
| KVER=11 启动 JSON | `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json`，SHA256 `25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78` | 同上；fused 脚本锁定 v11 |
| Epoch 预算 | 默认 15s（`HYBM_SDMA_ORCH_EPOCH_TIME_BUDGET_S = 15`）；`MF_SDMA_ORCH_EPOCH_BUDGET_S` 未设置（非常驻诊断模式）；`MF_SDMA_ORCH_DIAG` 未设置（G1-A 无记录器） | 源码核对 + 环境变量核验 |
| 硬件 | Ascend 310P3 ×8，实验固定 dev0/1（NPU 5664），TP=2 | `npu-smi` |
| 模型 | `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`（38GB，在位） | `ls` + `du` |

诊断代码：`vllm-ascend` worktree `@54c367ac`、`memfabric_hybrid` worktree `@eb96eced`（均 detached，原分支未动）。MemFabric 诊断分支本轮**仅静态审查未编译部署**（G1-B 遗留）。

## 3. 执行序列与会话结果总表

| 会话 | 模式 | 启动(UTC) | 健康 | 结束(UTC) | 退出码 | 完成/错误请求 | 判定 |
|---|---|---|---|---|---|---|---|
| F01 | fused 原工况 | 07:17:15 | 07:20:50（V2 层标记 80=40×2 rank） | 07:31:20 | 0 | 29 完成 / 0 协议级错误（rep3-13 为故障后连接失败） | **FAIL：复现 G3 签名** |
| F02 | fused 原工况 | 08:22:38 | 未达（08:23:35 显存预检失败） | 08:23:53 | 1 | 0 / 0 | 无效轮：外部进程瞬时抢占 dev0 约 8GB（非数据点，见 §6） |
| F02R1 | fused 原工况 | 08:28:17 | 08:31:42（80 层） | 08:44:20 | 0 | 75 / 0 | 未复现 |
| S01 | stock 对照 | 08:45:37 | 08:49:07（V2 层=0） | 09:00:43 | 0 | 75 / 0 | 干净对照 |
| D01 | stock+DFlash 投机 | 09:37:10 | 未达 | 09:39:20 | 1 | 0 / 0 | 无效轮：混合滑窗/全注意力模型要求 `VLLM_USE_V2_MODEL_RUNNER=1`（`NotImplementedError`，qwen3_dflash.py:124） |
| D01R1 | stock+DFlash 投机 | 09:45:26 | 未达（配置校验失败） | 09:45:41 | 1 | 0 / 0 | 无效轮：V2 model runner 硬性要求 Triton（vllm/config/vllm.py:2258），本机无 triton-ascend 且镜像源无发行版 |
| D01R2 | stock+FULL_DECODE_ONLY 图模式（投机已按指示移除） | 10:20:11 | 10:23:57（V2 层=0，图模式启动 3m46s） | 10:40:35 | 0 | 50 / 0（output 2048，中位 TTFT 11.8s） | 干净 |

每轮间均核验：前进程退出、端口释放、设备回基线（1352/1594MB）、无 507899、meta 排空窗口 ≥75s。所有会话证据位于 `tmp/opencode/gate_e_forensics/<会话>/`（Git 忽略，仅存服务器本地）。

## 4. F01 故障深描（唯一失败数据点）

### 4.1 宿主侧时间线

| 时刻(UTC) | 事件 | 证据 |
|---|---|---|
| 07:17:15 | fused serve 启动（Eager、TP=2、dev0/1） | runner.log |
| 07:20:50 | 健康检查通过；V2 层启用标记 80 条（每 rank 10 o_proj + 30 out_proj） | serve.log `grep -c` |
| 07:21:14 | 双设备 AICPU 加载 `libmf_sdma_orch_v11.so`（KFC 部署链成功，`Get api HybmSdmaOrchEpoch … success`） | 设备 slog（device-0/1） |
| 07:22:14 | 双 rank 首调 V2（M=4096 ∈ [2048,4096]，chunk=1024）；**双 rank bit-exact 校验通过** | serve.log ×2 |
| ~07:22–07:27 | rep1 25/25 完成；rep2 进行中（完成 4 个后挂死） | stability_rep1/2.json |
| 07:23:01→07:26:06 | 双设备 AICPU sharder `Get event num has exceeded 1000` 告警持续，parallelId 增长：dev0 754→4359，dev1 795→2999 | 设备 slog |
| 07:25:00 | **dev1 设备 slog 静默**（最后写入，AICPU 时钟 15:25:00） | device-1 slog |
| 07:26:06 | **dev0 设备 slog 静默** | device-0 slog |
| **07:28:06.288** | **rank0（TP0, pid 3960587, Device 0）首个错误：`call aclnnSwiGlu failed, error code is 507014`（ERR00100 PTA call acl api failed）** | serve.log L475 |
| **07:28:06.306** | **rank1（TP1, pid 3960682, Device 1）同型错误，间隔 18ms** | serve.log L510 |
| 07:28:06+ | 双 Worker `aclrtSynchronizeEvent` 507014、`rtEventSynchronize … reason=aicore timeout` ×2/rank；EngineCore 异常，服务不可恢复 | serve.log |
| 07:28–07:31 | rep3–13 全部 0 完成（连接失败）；620s 窗口到点 | stability_rep3-13.json |
| 07:31:20 | runner 退出码 0；**设备显存回落至基线，无 507899，无泄漏** | npu-smi 事后快照 |

### 4.2 关键定性事实

1. **首个表面错误点是 stock 算子 `aclnnSwiGlu`**（MoE/MLP 激活，非 V2 融合算子）。AICore timeout 在下一个同步点浮现——真正卡死的 kernel 在 SwiGlu 之前，SwiGlu 是表象，符合文档 §4 分类规则。
2. **双 rank 对称**：首错间隔 18ms，签名一致——与既有 G3 "双 Worker 同时超时" 画像吻合。
3. **设备侧先于宿主侧静默 2–3 分钟**（dev1 −3m06s、dev0 −2m00s 相对 07:28:06）。宿主在设备停摆后约 2 分钟才通过事件同步超时感知。
4. **Host 控制面无 ERROR 级异常**：serve.log 中 `epoch exited with error` / `epoch relaunch failed` / `supervisor exit` 匹配数为 0。注意：G1-A 未开 MF_DIAG，INFO 级纪转换不打印——**"无异常打印"≠"纪生命周期正常"**（单纪错误被重拉掩盖的可能性仍在，见文档 H2）。
5. **故障前后无外部竞争**：F01 全程设备 0/1 专属（15s 采样 npu_smi.log/processes.log 可复核）。

### 4.3 已保全证据（路径）

- `F01/evidence/device_logs/run_device-0/device-3960587_20261020071812639.log`（TP0 设备 slog，止于 07:26:06）
- `F01/evidence/device_logs/run_device-1/device-3960682_20261020071828641.log`（TP1 设备 slog，止于 07:25:00）
- `F01/evidence/device_logs/debug_plog/plog-3960587_*.log`、`plog-3960682_*.log`（双 rank plog，写至挂死）
- `F01/evidence/device_logs/run_device-{0,1}/device-39637{89,87}_*.log`（KFC 部署助手设备日志，07:21）
- `F01/serve.log`（完整宿主日志）、`F01/samples/`（15s npu-smi/进程采样）、`F01/evidence/provenance.txt`（版本/哈希快照）
- 未发现独立 AICore kernel dump 产物（`*aicore*dump*` 目录不存在）；设备错误以 slog/plog 为准。

## 5. 假设判别状态（H1–H4）

| ID | 假设 | 本轮证据 | 判别状态 |
|---|---|---|---|
| H1 | SQ 容量失控（`sqDepth - used < 1` 恒假 → SQE 覆写） | G1-A 无 SQ/请求环游标采样；sharder 告警的 parallelId 增长**不能**区分是正常负载堆积还是 SQ 停摆 | **未证实、未排除**。需 G1-B 请求环游标轨迹 + 文档 G1-C 的设备侧 SQ Head/Tail |
| H2 | Epoch 消费链停顿 | F01 无 ERROR 级 epoch 事件；但无 MF_DIAG 时纪级时序不可见，"重拉清零错误计数" 掩盖路径未排除 | **未证实、未排除**。需 G1-B 每纪 exitReason/mainRet/consumed 时序 |
| H3 | 共享 Cache Line 风险（ReqTail/ReqHead/Quiet 同行） | 无游标回退/戳异常的可观测样本 | **未证实、未排除**。文档规定需游标异常证据后再做地址隔离 A/B |
| H4 | V2 IB/同步死锁 | 一致性支持：双 rank 18ms 对称超时、表面错误在 stock 算子同步点、设备 slog 先静默 2–3 分钟、sharder 事件持续累积后停；且 stock 对照与图模式 stock 场景均干净 | **方向性支持、未证实**（无前置阶段标记/设备栈证据，双 rank timeout 本身不指向 MemFabric——文档明示） |

**根因判定：未定。** 按文档 §4 规则，两轮 fused 中 1 次挂死 + 1 次干净，仅记 "间歇性复现 1 次"，不等于根因修复，亦不构成任何单一假设的证实。

## 6. 环境事件记录（影响执行但不构成故障数据）

共享服务器存在外部租户（另一容器/命名空间，PID 与本容器空间无关），当日多次活动：

| 时间(UTC) | 事件 | 影响 |
|---|---|---|
| 07:11–07:16 | 外部 EngineCor 进程占用 dev0（7419MB，AICore 100%）；同时发现 dev0/1 存在 10-17 时代遗留的已死进程泄漏显存记录（34420+33680MB，驱动随后自行回收） | 延迟 F01 启动至 07:17 |
| 08:23:35 | 外部进程瞬时分配 dev0 约 8GB | **F02 显存预检失败（`Free memory 35.14/43.24 GiB < 38.92`），轮次作废** |
| 09:05–09:36 | 外部 EngineCor 占用 dev0（7419MB） | S01 收尾后等待窗口 31 分钟 |
| 10:15–10:19 | 外部 python 进程短暂占用 dev0（129MB） | D01R2 启动延迟 |

处置原则：仅窗口等待，未杀任何外部进程、未复位设备（容器内 `npu-smi set` 不可用，且未经判断的暴力清理为方案明令禁止）。

## 7. 脚本与配置调整记录（小型实机兼容修复，diff 已存档于 tmp/opencode）

1. **`gate_e_forensic_run.sh` provenance 路径修复**：原脚本探测 `/usr/local/memfabric_hybrid/{version.info,lib64/…}`，与本机实际布局（`latest -> 1.2.0`，库在 `aarch64-linux/lib64/`；vLLM 实际经 site-packages ctypes 加载）不符。已改为 `latest` 路径 + site-packages 双位置哈希采集。
2. **新增 `stock_dflash` 模式**（`gate_e_stock_dflash.sh` + wrapper 三态分支）：承载用户追加场景——stock + `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[10]}'` + output 2048 + 50 prompts。按用户指示**未含投机推理配置**；曾尝试 `--speculative-config dflash(7)`，因模型混合滑窗/全注意力架构强制 V2 model runner → Triton 硬依赖（本机无 triton-ascend，华为云镜像与 pypi.org 均无该包）而移除，连带移除 `VLLM_USE_V2_MODEL_RUNNER=1`，走默认 V1 runner。

原 Gate E 脚本（`gate_e_eager.sh` / `gate_e_eager_stock.sh`）的工况参数**零改动**。

## 8. 结论与遗留项

### 结论（事实）

1. Gate E 间歇性双 rank AICore timeout 在本轮 **1/2 fused 会话复现**（F01），stock 对照 2/2 干净（S01、D01R2），复现率与既有 G3 记录一致，故障签名完全一致。
2. F01 提供了首个**带完整设备侧日志的失败样本**：设备 AICPU 栈在宿主感知前 2–3 分钟已停止推进（dev1 先于 dev0 约 1 分钟），期间 sharder 事件堆积；宿主侧首个错误浮现在 stock 算子 `aclnnSwiGlu` 的同步点。
3. 第一个停止推进的环节**尚未定位到控制面或计算面的具体位置**——这是 G1-A 无侵入手段的天花板，与方案预期一致。

### 遗留项

| # | 项 | 内容 | 前置条件 |
|---|---|---|---|
| L1 | **G1-B：Host 飞行记录器会话**（本轮未执行） | 在 MemFabric 诊断 worktree（`eb96eced`，代码已静态审查：opt-in、只读 16B D2H 采样、不触碰 SQE/Mailbox/Epoch 算法）编译 core.so；双位置备份替换（SHA256 验证）；跑 1–2 轮 fused `MF_SDMA_ORCH_DIAG=1` 会话，采集请求环游标轨迹 + 每纪 exitReason/mainRet/consumed/sqHeadAfter 时序；跑完恢复原库并逐文件验证。G1-A/B 结果不可混作复现率统计 | 用户指示；确认无其他 NPU 任务；约 2–3 小时窗口 |
| L2 | G1-C 决策点 | 若 G1-B 后 SQ/V2 首停点仍无法确认，按文档需方案负责人提供独立设备诊断版本（更换 device .so 版本号避免 CP1 缓存）+ V2 各核低成本阶段戳；本轮未涉及 | L1 完成后 |
| L3 | DFlash 投机场景 | 因 triton-ascend 缺失（镜像源无发行版）无法执行，如需恢复需 Ascend 官方 gitee wheel 或离线包；本轮已按指示移除投机配置，图模式场景（D01R2）已干净通过 | triton-ascend 获取渠道 |
| L4 | 外部租户设备竞争 | 共享服务器 dev0 反复被外部进程占用（4 次/日），建议协调独占窗口后再跑 G1-B，避免窗口等待损耗与启动竞争伪影 | 服务器管理协调 |

### 证据索引（服务器本地，Git 忽略）

```
csrc/_310P/memfabric_mm_ar_v2/tmp/opencode/gate_e_forensics/
├── F01/ … F02R1/ … S01/ … D01R2/          # 各会话完整证据（serve.log、bench、samples、evidence/）
├── env_blocker/
│   ├── devmem_monitor.log                  # 外部竞争监控全程记录（60s 采样）
│   ├── script_fix_provenance_paths.diff    # 脚本修复 diff
│   ├── wrapper_stock_dflash_mode.diff      # wrapper 三态分支 diff
│   └── gate_e_stock_dflash.sh              # 新场景脚本快照
├── F02_RETRY_NOTE.txt                      # F02 作废原因
└── progress_log.txt                        # 30 分钟进展快照（10:20 起）
```

---

执行：OpenCode @ 310P3 服务器，2026-10-20 07:00–12:33 UTC。
