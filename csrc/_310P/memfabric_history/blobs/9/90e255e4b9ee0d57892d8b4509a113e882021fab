# Gate E / 507014 交付 MemFabric 团队：复现版本与现场验证合同（2026-10-26）

> **状态：交付前复验待做（ONDEVICE_NOT_RUN）**。本文件为已有 G1-D 诊断分支增加清晰的接收入口，不声称修复或新实机复现。不得以 GitHub CI/纯 CPU 测试替代双 Rank 设备证据。
>
> **项目主线不是本分支**：当前项目控制台位于 `dev/310p-memfabric-mm-ar`；本分支只承担 507014 故障复现、取证和移交，禁止顺手合入性能优化、DFlash Runner 重建或发布代码。

## 1. 版本锁定与选择理由

- vllm-ascend 诊断原始基线：`10wvw01/vllm-ascend:diag/310p3-g1d-minimal-forensics-20261021@46260392a855922daff742e733d85e1c78bc2cce`。本交付文档和离线判读工具追加提交之后，应以交付验收时的**完整 HEAD SHA** 冻结新版本，不能只引用分支名。
- MemFabric 诊断对照源码：`10wvw01/memfabric_hybrid_310p:diag/310p3-g1d-minimal-forensics-20261021@54d6376d619325fb70831ce16c6ab60305b272ec`（只读对照；实际安装 Host/Device/KVER 独立核验）。
- 这个 vllm 分支**已经在历史实机**通过 G1-D D1R4 复现 `507014`；双 Rank 核心停止点位于 V2 core0 `stage=50`，`smem_shm_sdma_wait_at` 未返回。Host D2H 计数已覆盖被等待 seq，但**不能推断写入已对 AICore 可见**。G1-D2 尾槽诊断扩展 6 场均未复现；根因仍 UNKNOWN。
- 适用边界：此交付目标是原始 **普通 Eager、融合 V2** 故障，不是 DFlash 投机、Graph，也不是普通业务的 output2048 性能基线。不得换负载后仍引用历史复现率。

## 2. 原样复现负载（不可无证据变更）

| 项目 | 值 |
|---|---|
| 设备 | 310P3 同一张物理卡的两 Die，TP=2。历史物理号 0,1 **仅供追溯**，本轮按 G0 重新判定 |
| 目标模型 | `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`（历史现场路径；需核对模型 SHA/config 与真实在位路径） |
| Serving | `--enforce-eager`, `--async-scheduling`, FP16, `--quantization ascend`, `--max-num-seqs 16`, `--max-num-batched-tokens 4096`, `--gpu-memory-utilization 0.90` |
| V2 | `VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1`, `VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE=1`, `VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS=1024`, `VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS=4` |
| Bench | random input 4096 / output 256；25 prompts/轮；最大并发 10；ignore-eos；循环约 620 秒 |
| 核验 | 两 Rank 原始启动日志、40 个目标层/Rank 的真实选择与 Kernel trace、首调位精确、负载完成数、507014/507899、双 Die 设备日志 |
| 阴性对照 | 仅关闭融合，运行 `gate_e_eager_stock.sh`，保持相同业务配置；不能把 DFlash/Graph Stock 作为同工况对照 |

**说明：** 历史报告中“80 条启用日志”意味着两 Rank 合计 40×2，属于选层证据；不自动证明实际运行的每次 Kernel 都是 V2。旧脚本主机接口使用 `--host 0.0.0.0`：现场必须确认网络隔离，否则 NO-GO。

## 3. 可执行源码入口（只复用现有实现）

- `test/g3_e2e_scripts/gate_e_eager.sh`：正式融合 Eager 服务 + 620 秒循环负载。
- `test/g3_e2e_scripts/gate_e_eager_stock.sh`：同口径 Stock 对照。
- `test/g3_e2e_scripts/gate_e_forensic_run.sh`：**优先执行入口**，自动建立唯一实验目录、采样 npu-smi、保存 provenance、runner/serve/bench 与结果文件；注意它不是设备安全授权器。
- `test/g3_e2e_scripts/gate_e_forensic_summary.py`：历史简要摘要；`test/g3_e2e_scripts/gate_e_handoff_assess.py`：交付前新增的离线机器可读评估，独立判断是否有 507014 字样、负载是否有效、融合证据是否不足。
- `test/g3_e2e_scripts/g1d_run.sh`、`g1d_preflight.sh`、`g1d_evidence_summary.py`：只用于**可选** MemFabric+V2 G1-D 深层 trace（双方需要一致的诊断构建/Host 双路径/设备 v12，不能直接当普通 KVER11 最小复现使用）。

本轮**不修改历史复现脚本、正式 Kernel、Host ABI、SDMA SQE、Mailbox 协议或 MemFabric 安装内容**。旧脚本的退出码 0 不能视作“无故障”：它可能在 bench 失败后正常收尾，也可能只采到应用表面错误；必须以后处理报告和双 Rank 原始设备日志为准。

## 4. 移交前第一阶段：只读 G0（允许），不直接开机

由 OpenCode 在 310P3 **现有工作树之外**创建隔离 worktree，原工作树/运行服务/日志/忽略文件先保全，禁止 `reset --hard`、`clean -fd`、覆盖 SO 或强杀其它进程。

1. `git fetch` 后比对远端完整 HEAD 与冻结 SHA，记录 `git remote -v`、每个 worktree 的 HEAD / `git status --porcelain`；校验 `vllm` CLI、真实 `vllm_ascend` import、`PYTHONPATH` 与 CANN 环境。
2. 核验两 Die 是同一卡、没有其它租户/进程/端口占用，现场设备号由 `npu-smi` 与设备拓扑**实测**。不要使用历史默认 0,1 猜测，或触碰其它 Die。
3. 按 `ENGINEERING_GATES.md` 的身份规则记录：CANN/BiSheng/Driver、vLLM 与 PyTorch、V2 Host 扩展/设备 SO、MemFabric Host **两处**加载库、Device SO、AICPU KVER11 JSON，逐项 SHA256；并核 `/proc/<PID>/maps` 中真实 SO 装载路径。**必须确认 MemFabric 采用实际已部署且与已验证协议兼容的基线，不得自动换诊断版。**
4. 检查模型在位、端口 8000 / Store 8581 与安全隔离、`MF_SDMA_ORCH_EPOCH_BUDGET_S` 非 `4294967295`、首调正确性路径和冷池保护未被关闭；复现故障前保留完整原始日志并明确是否可对外分享。
5. **先执行纯 CPU/只读测试**，再人工批准 G0；任何 SO 缺失/哈希不一致、未知 KVER、混树 import、资源占用、模型不一致均 NO-GO。G0 输出只是环境现状，不是 NPU PASS。

## 5. 第二阶段：受控实机复验（逐级授权）

在 G0 通过且已获得用户对本轮测试设备及恢复计划的许可后，由现场人员设置真实值（下例不携带设备号、哈希、模型真实路径的猜测值）：

```bash
# 从本分支隔离 worktree 根目录执行；现场先激活与历史一致且已核验的 CANN/vLLM 环境
export V2_WORKTREE="$(pwd)"
export GATE_E_DEVICES="<G0 确认的两 Die 编号>"
export GATE_E_MODEL="<G0 确认的模型绝对路径>"
export GATE_E_PORT=8000
SCRIPTS=csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts

# 第 1 级：只在隔离设备上受控冒烟，90 秒用于检验模型、首写、选层、双 Rank
GATE_E_DURATION_S=90 bash "$SCRIPTS/gate_e_forensic_run.sh" fused "H1_FUSED_SMOKE_<runid>"

# 保全 smoke 证据并完成审核后，第 2 级：历史 620 秒复现，不自动连续重跑
GATE_E_DURATION_S=620 bash "$SCRIPTS/gate_e_forensic_run.sh" fused "H2_FUSED_620_<runid>"

# 必须先人工确认 fused 会话安全退出/无设备故障，再启动独立 Stock 对照
GATE_E_DURATION_S=620 bash "$SCRIPTS/gate_e_forensic_run.sh" stock "H3_STOCK_620_<runid>"
```

尖括号是**待现场填入**的占位符，不能原样执行。各阶段单独保存结果并审核；**若 fused 复现 507014，应立即停止下一轮/Stock，转为保全日志、设备现场与版本哈希**。不要为凑复现率在设备异常后继续发请求、自动重试或重启设备。

**关键缺陷提醒：** 原始 `gate_e_forensic_run.sh` 并非完整的 fail-stop 监护程序；它只是已有、历史验证过的启动/取证入口。出现硬件超时或服务异常时现场须立即终止新请求、先抓双 Rank slog/plog、进程 maps 与 npu-smi，再按已审批的环境恢复 SOP 有序清退；不能仅依赖 shell 的 `exit 0`。原始日志不要覆盖或删除。

### 每轮离线判读（不会操作 NPU）

```bash
# 在这轮的唯一证据目录中执行；不要覆盖已有评估结果
EVIDENCE_DIR="<该轮 gate_e_forensic_run.sh 打印的证据目录>"
python3 "$SCRIPTS/gate_e_handoff_assess.py" "$EVIDENCE_DIR" \
  --arm fused --out "$EVIDENCE_DIR/gate_e_handoff_assessment.json"

# Stock 轮次使用 --arm stock，必须是其自身独立证据目录
# 评估程序 exit=0 仅说明读取/分析完成，不代表设备无故障。
```

所有测试应运行 `python3 -m unittest discover -s "$SCRIPTS" -p 'test_gate_e_handoff_assess.py' -v` 并保留实际退出码；此前未运行不可写 PASS。失败本身要同时保全最初的 `serve.log` / `runner.log` / 两 Rank slog/plog，不能只交一张评估 JSON。

## 6. 可选第三阶段：G1-D 深取证（只有初级复验需要时）

如果标准 KVER11 仍见 507014，MemFabric 团队可选择历史 G1-D v12 设备取证配方，并使用同名 MemFabric 诊断分支。必须单独完成源码/ABI/设备协议、Host 双位置 SO 备份和 SHA 校验、v12 JSON SHA、设备侧 v12 部署、构建正确性与回滚演练，并由用户再次批准；不能在标准失败现场未经恢复直接替换 `libmf_hybm_core.so`。诊断版复现与标准版结果分别标注，绝不混成同一二进制的复现率。

历史故障最值得 MemFabric 团队继续区分的问题：
- `stage=50` 等待的那封邮件：Host D2H 可见的 `arrHead` 已越过被等待 seq，但到达 stamp/mail 是否真的正确写入对应槽？
- 到达槽写入在位但 AICore 不可见，还是该槽的 SQE/write 原本未落地？`SQ Head == Tail` 本身不能证明每条邮件已正确写入。
- G1-D2 的 `head-1..4` 尾槽采样在健康侧具备 `t1Stamp==arrHead` 不变量，但尚未抓到失败窗口；不能先行认定缓存一致性根因。

## 7. 复验结果的客观裁决

| 结果 | 允许记录 | 不允许记录 |
|---|---|---|
| 双 Rank 设备日志、V2 停滞和 507014 同时出现 | **REPRODUCED**：本环境仍可复现，提交最小原始证据包 | 未取证即宣称是 MemFabric 根因 |
| 只见应用端 `507014` / Kernel 超时 | **FAULT_SIGNATURE_OBSERVED**，继续核实 rank、最先停止核和加载二进制 | 直接宣布“已确认同一根因” |
| 1～N 个有效融合会话没触发 | **NOT_REPRODUCED_IN_N_SESSIONS**，逐轮保留完成数和验证证据 | 宣称已修复、长期稳定 |
| Fused 没命中 / 报告缺 Rank 或 SO 身份 | **INVALID/UNKNOWN**，修复测量合同后才重测 | 把 Stock 对 Stock 算作融合复验 |
| Stock 对照独立完成、无同类故障 | **CONTROL_CLEAN_IN_N_SESSIONS** | 从少量对照推出 MemFabric 必然有错 |

交付包至少应包含：冻结的两仓 SHA、完整启动命令与配置、Host/Device SO 和 KVER JSON 哈希、模块真实 import/maps、双 Rank `serve.log`、bench JSON、失败前后 run/plog/slog、MemFabric/SQ/ARR/epoch/等待阶段截面、完整 `npu-smi`、人工确认的实际融合命中、离线评估 JSON、`SHA256SUMS`、故障后恢复校验及脱敏清单。

**私有模型/DFlash 权重、凭据、设备专属隐私与原始敏感日志不得直接上传公开 GitHub；先在服务器私有目录保全与脱敏审查，公开仓库仅发布适合共享的复现代码与非敏感摘要。**

## 8. 交付裁决

- **代码侧**：已有可运行历史复现脚本，新增本文件和离线证据判读器后可进入现场 G0。
- **设备侧**：新的 G0、模型首写 Smoke、正式 620 秒 fused 复验及对照均 **NOT_RUN**；最终移交前仍需由用户委托 OpenCode 在 310P3 逐级实测。
- **交付方向**：移交 MemFabric 团队时交付此冻结分支 SHA 和版本清单，不将仍在迭代的大 M 项目主线或已废止的 R2 Runner 打包交付。

历史依据：[G1-D 执行报告](G1D_EXECUTION_REPORT_20261021.md) · [G1-C 第一停止点](G1C_DEVICE_FIRST_STOP_REPORT_20261020.md) · [G1-A Gate E 复现报告](G1_GATE_E_FORENSICS_REPORT_20261020.md)。
