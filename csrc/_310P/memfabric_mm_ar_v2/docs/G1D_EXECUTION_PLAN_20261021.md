# G1-D 精准诊断执行方案与验收矩阵（2026-10-21）

> **状态：代码及脚本已静态准备；未在 310P3 上构建或运行，现场执行结果为 UNKNOWN。**
>
> **目标：** 确定 Gate E V2 双 Rank 停滞的第一处可验证异常交接；验证是否需要 1024 边界／异步持续负载／V1-V2 公共协议；为单变量根因修复提供依据。G1-D 不承诺本轮修复完成。
>
> **基线：** vllm-ascend `diag/310p3-g1d-minimal-forensics-20261021`；MemFabric 使用**现有镜像修复线** `10wvw01/memfabric_hybrid_310p@diag/310p3-g1d-minimal-forensics-20261021`（由 G1-C 派生）。**不切换到原仓库 `wgm-dev-310p`**。服务器原 Host v11 与设备 v11 不覆盖，继续使用 G1-C 单独部署的 AICPU v12 取证二进制（不新增设备 ABI、SQE 格式或设备库版本）。V2 编译时使用独立名称 `libmf310p_v2_trial_device_g1d.so`，不会覆盖 G1-C 设备库。

## 1. 最高优先级的证据问题

G1-C C2R2 双 Rank 处于第 208 代；rank0 core0 为 stage50/chunk1；rank1 core0 为 stage50/chunk3，其余核 stage60；Host 请求环与查询时 CP SQ 均空、Epoch 持续运行。**这些不证明 wait 一定没返回，也不证明 Stamp 一定没写入。**

可疑邮件 rank1→rank0 chunk1，`seq=1023 / ARR[1023] / expect=1024` 是基于调用顺序的工作假设，非目标内存直测。B02 无阶段戳，不能证明与 C2R2 同槽。不能拿两次失败计算 `1e-5` 显著性。

**D1 需要给出：**

- G1-D V2 `stage=50`：wait 前；`stage=52`：wait 返回、尚未检查；`stage=53`：邮件校验失败、Trap 前；`stage=51`：正常校验通过。
- `[V2_G1D]` 只在返回邮件的采集代际等于当前代际时提供 `mailSeq/mailDst/mailLen/mailImm/mailStatus`；如果从未返回，这一条应标记“无结果”，不能擅自填零。
- `[MF_G1D]` 从**本端对称段保留区**采集 `ArrHead`、`stamp[1020..1023]`、`stamp[0..2]`、`mail[1023]`、`mail[0]`、`quiet[60..63]`、`quiet[0..1]`，分别采双 Rank；按 1Hz 周期，采集非原子，不代表同一个硬件瞬间。
- 结合 G1-C `[MF_G1C]` 设备 SQ 查询与历史 `[MF_DIAG]` 游标记录判读：**SQ 当前为空不证明每条 SQE 正确完成**；只有 direct GM/AICore 观察支持时才认为具体槽位有异常。
- 关键内存布局：到达戳末 4 槽与 `mail[0]` 共 64B Cache Line（`0x2A00`）；`mail[1023]` 与 `ArrHead` 共 64B（`0xAA00`）；quiet 后 2 槽与 reserved 交接区邻接。这个共享关系本身**不是缓存丢写证明**。

## 2. 阶段顺序与强制 Gate

| Gate | 操作 | 准入/退出要求 |
|---|---|---|
| **D0** 现场基线锁定 | 核对两个工作树、Git SHA、源码分支；记录原装 Host 双路径与设备 KVER11/12/JSON SHA256；备份/恢复演练；原日志保全 | 禁止 dirty worktree 覆盖；基线未知/原库哈希不吻合即 STOP |
| **D1.0** 静态与构建 Gate | 按第 3 节分别构建 V2 G1-D 与 MemFabric Host G1-D；新增诊断全部默认 OFF；静态审视 2 轮 | 两侧编译通过，link/符号/ABI无变化，设备诊断库名称隔离；未通过则不进入实机 |
| **D1.1** 正确性 Gate | 诊断开关 OFF（确认不影响行为）后开 ON，运行现有 TP2 correctness（旧 G1-C Gate C1） | 51/51 等价检查、协议状态、双 Rank 日志完整；不得引入新位精度差异 |
| **D1.2** Gate E 再现 | 当前 V2 + 当前 MF 修复线；两 Rank 都开启 G1C+G1D 取证；会话唯一目录，先现场低频采样、再长稳 | 故障发生必须保全原始日志/设备 plog/slog/版本哈希；若未复现，保留健康证据，绝不宣布已修复 |
| **D2-A** 最小压力 | `g1d_tp2_async_window.py` 从 window=1 到 window=2，再到 window=4（最大输出槽数），每窗口后同步；对照加入 Stock/HCCL | 记录实际请求 seq（包含预热/ DONE），满足跨越多轮真实 ARR/SQ 回绕，正确性始终 PASS；出现 FAIL 停止放大 |
| **D2-B** 边界取证 | 比较真实 `seq mod 1024` 在边界附近的成功/失败消息，以及非边界对照；不能单靠请求数估计精确 seq | 有健康边界样本和失败/无失败记录；如果失败不在边界，及时下调边界假设 |
| **D2-C** V1/V2 共性试验 | **仅当**当前 V1 与 v11/v12 ABI/部署/路由全部兼容后，同一 MF Host/设备二进制与 Eager Gate E 同配置分开跑 | 若 V1 兼容性不能保证则记录 BLOCKED，不通过直接切换 KVER9/KVER10 来凑结果 |
| **D3** 关联性能 | 取窗口级 `[MF_G1D]` 异常、V2 stage、按秒完成率与请求 TTFT/TPOT；未来若有明确慢等待再加轻量统计 | 不在 1Hz 采样下作微秒级计时结论；正式性能必须用完全无取证功能版本重新测 |
| **D4** 根因裁决 | 对候选 C-a/b/c/d、ArrHead 和 mail mismatch 做证据等级表；只有证据支持才提出单变量修复 | 提供直接值、排他条件、可复现步骤、下一试验；无法确认时写 UNKNOWN |

**D1 优先。D2/D3 工具准备可并行，但不能替代 D1 直接证据。不得同时修改通信协议和运行基线。**

## 3. 代码清单与构建要求

| 文件 | 作用 |
|---|---|
| `src/op_kernel/mf310p_v2_g1d_trace.h` | 在 G1-C Core0 私有 64B 槽记录 wait 返回邮件（不读取 ARR） |
| `src/op_kernel/mf310p_v2_kernel.asc` | 两处等待均增加 stage52/53；正式 signal、wait、mail 匹配与 Add 原样保留 |
| `src/op_host/mf310p_v2_trial_adapter.cpp` | 同代邮件日志 `[V2_G1D]`，仅在双开关启用时输出 |
| `src/CMakeLists.txt` 与 `cmake/memfabric_310p_v2.cmake` | 增加 G1D 编译开关并要求 G1C；G1D 设备库独立名称 |
| MemFabric `src/hybm/csrc/entity/hybm_sdma_orchestrator.cpp` | 每秒 5 次短 D2H，从本端保留区读取边界槽，只记录快照，不修改 SQE/doorbell/epoch |
| `test/g3_e2e_scripts/g1d_preflight.sh` | 只读 Git/安装库溯源；输出目录不可覆盖 |
| `test/g3_e2e_scripts/g1d_run.sh` | 逐 Gate 启动，校验分支/KVER12 JSON/设备号，不执行 install 和回退 |
| `test/e2e/g1d_tp2_async_window.py` | 可控连续异步窗口与 Stock/HCCL 交替，按窗口同步、位精确 |
| `test/g3_e2e_scripts/g1d_evidence_summary.py` | 离线结构化摘要，明确 OBSERVED 与 UNKNOWN |

**MemFabric 构建：**

- 基于 G1-C 的既有 Host 构建配置编译 Host 库，**不需要修改 AICPU device kernel**。保留 G1-C 所需的 v12 `-DMF_SDMA_ORCH_KVER=12 -DMF_SDMA_ORCH_G1C_DEVICE_TRACE=ON` 构建和已验证 v12 文件；不要为了 G1-D Host 采样而改 v12 SQE。
- 在独立构建目录完成 `cmake --build`，核对导出、SHA256、链接路径。若只改 Host，必须核对实际 `libmf_hybm_core.so` 已被运行中 Worker 从哪个位置加载。用户态 `MF_SDMA_ORCH_G1D_TRACE=1` 且 `MF_SDMA_ORCH_DIAG=1` 才启动额外取证。
- 服务器安装或临时替换 Host 库涉及两个加载路径；由 OpenCode 在备份及 SHA 验证后实施，必须提供逐字节恢复脚本与实际返回码。**本仓库不附带任何自动覆盖系统库的脚本**。

**V2 构建：**

- 单独 worktree，在原 `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=ON` 与 `VLLM_ASCEND_310P_V2_G1C_TRACE=ON` 基础上，额外指定 `-DVLLM_ASCEND_310P_V2_G1D_TRACE=ON`。CMake/Bisheng/Host 必须同时带宏。
- 必须确认生成的 SO 是 `libmf310p_v2_trial_device_g1d.so`，Host 链接的不是旧 `g1c` 库；不要覆盖正式 `libmf310p_v2_trial_device.so`。
- 单独执行构建/静态验证失败不可自行重新安装正式库。`MF310P_V2_G1D_TRACE=1` 为 Host 输出日志开关；编译宏和运行开关不同。

**诊断开启后的实验环境：**

```bash
export G1D_MF_TREE=/绝对路径/memfabric_hybrid-g1d
export G1D_OUT_DIR=/绝对路径/g1d_evidence_唯一编号
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1d_run.sh preflight

# 需要 OpenCode 先完成 build、SO 双路径备份和部署，并锁定 JSON 实际 SHA。
export MF_SDMA_ORCH_JSON=/绝对路径/libmf_sdma_orch_v12.json
export G1D_PYINJ=/绝对路径/已验证的sitecustomize目录
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1d_run.sh c1

# 较低压力起步；调用数量达到多次真实回绕时再逐级扩大。
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1d_run.sh async --window 1 --calls 256
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1d_run.sh async --window 2 --calls 256
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1d_run.sh async --window 4 --calls 256

# C1 与 D2 均通过、设备/环境恢复正常之后再进行 Gate E。
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1d_run.sh gate-e D1R1

python3 csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1d_evidence_summary.py \
    /绝对路径/本轮证据目录/serve.log -o /绝对路径/本轮证据目录/g1d_summary.json
```

**执行时必须额外核对** `VLLM_WORKER_MULTIPROC_METHOD=spawn`、目标进程是否真正从 G1-D 工作树 import、设备 v12 `kernelSo` 是否在位、模型路径和配置一致。原始 Gate E 会在其 runner 中重新 `source set_env.sh`，验证显式 JSON 不被覆盖。

## 4. STOP/GO 判定、取证和恢复纪律

**D0 停止条件**：当前安装 Host/设备二进制 SHA 与归档不一致、运行库路径混杂、历史 G1-C v12 已丢失、设备号不是 0/1、存在未排干净的 Worker。不要执行长稳，先人工查清版本。

**D1 停止条件**：单测位精确失败、D2H snapshotError、关键日志部分核丢失、Epoch 异常、无法确认 G1-D V2 自己的设备 SO。需要保全日志而不是继续增加并发。

**现场异常后的顺序**：记录最后健康时间 → 捕获 `serve.log`/bench JSON/`[V2_G1C/G1D]`、`[MF_DIAG/G1C/G1D]` → 设备日志与进程快照 → 证据索引和哈希 → 确认故障 Worker/设备状态 → 按 G1-C 已验证的资源清退流程恢复 → 复核原 v11、双位置 Host、设备映射和冷却间隔。**不要未经保全立即清进程或覆盖设备库。**

**禁止**：拉取 `wgm-dev-310p`；在 G1-D 内修 MM、SDMA SQE 配方、mailbox/ARR 布局、Epoch 算法；在实机上制造任意旧游标/强行改 reqSeq；把虚构的“seq=1023 丢写”写成 observed；以一次 Gate E PASS 宣告根因已解决。

## 5. 期待输出与根因级判定

交付物：`G1D_EXECUTION_REPORT.md`、完整原始日志/哈希、两 Rank 时间线、关键 ARR/quiet 实值、各 Gate 执行状态表、G1-D1 候选假设矩阵、D2 跨边界成功与失败对照、V1 对照状态、性能关系初判、恢复/部署 SHA256 证据、下一步最小修复建议。

| 取证发现 | 允许的结论 | 尚需核查 |
|---|---|---|
| Core0 stage50 且本端 ARR 戳不匹配 | 到达/读取交接区间异常 | 需区分 SQE 未写、写错、跨 die 可见性、读到旧缓存 |
| 本端 GM 戳正确但 Core0 stage50 | 与缓存可见性/读游标相关 | D2H 与 AI Core 不同视角，需更低扰动 AICore 原位捕捉 |
| stage52/53 或返回邮件值错误 | wait 已返回或 mail 校验失败 | 错误 seq/target/imm 的源头及 Trap 证据 |
| quiet[63] 落地但 ARR[1023] 错 | 已观察发送侧末 SQE 结果，支持调查到达戳写入链 | 不能仅据此认定 SQE2 未执行，需驱动 FIFO/地址与内存一致性证据 |
| seq 非边界也复现 | 原“只在 1024 边界失败”假设遭反证 | 改查共同并发/缓存/协议条件 |
| G1-D1 完全不复现 | 尚无新增故障证据 | 对照诊断扰动与重复测试，不能宣布修复 |

**G1-D 最低 GOAL：一个真实故障窗口的第一处可证实通信交接异常及仍未知点。最好达到：最小复现+机制单变量 A/B；完整“根因已解决”必须由后续关闭埋点的全矩阵、长稳、Graph/性能等验收来证明。**

## 6. 关联依据

- [G1-C 现场报告（v12、stage50/60、SQ 空、环境恢复）](https://github.com/10wvw01/vllm-ascend/blob/95096ad25d004074e45e3ef65c33de5eb9842644/csrc/_310P/memfabric_mm_ar_v2/docs/G1C_DEVICE_FIRST_STOP_REPORT_20261020.md)
- [G1-C 脱敏证据复核包（1024 线索及 UNKNOWN）](https://github.com/10wvw01/vllm-ascend/blob/86e660edc1c1011a7ca23ac9bb29787fbe079271/csrc/_310P/memfabric_mm_ar_v2/docs/G1C_EVIDENCE_REVIEW_PACKAGE_20261021.md)
- [最初 G1-D 裁决请求](https://github.com/10wvw01/vllm-ascend/blob/adefb2cfc23217bcfc9a356dbcc5ad00ab8f6fc8/csrc/_310P/memfabric_mm_ar_v2/docs/G1D_REVIEW_REQUEST_20261021.md)
- [G3 Gate D/E 原始性能和稳定性](https://github.com/10wvw01/vllm-ascend/blob/518418f20cf8f09f2c418993400afaf639a101fc/csrc/_310P/memfabric_mm_ar_v2/docs/V2_G3_QWEN_E2E_REPORT.md)
- [V1 D-mailorder 与异步挂死历史](https://github.com/10wvw01/vllm-ascend/blob/518418f20cf8f09f2c418993400afaf639a101fc/docs/source/developer_guide/单KernelAR协议重构设计.md)
- [MemFabric 共享保留区布局](https://github.com/10wvw01/memfabric_hybrid_310p/blob/5345cbdd5cf745342c21f7b68b0b0907869c164a/src/smem/include/device/smem_shm_sdma_layout.h)
- [MemFabric 到达原语](https://github.com/10wvw01/memfabric_hybrid_310p/blob/5345cbdd5cf745342c21f7b68b0b0907869c164a/src/smem/include/device/smem_shm_aicore_base_sdma.h)
