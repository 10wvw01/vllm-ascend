# A1 CPU 与 G0 只读取证服务器验收报告（2026-10-26）

> 执行者 OpenCode；仓库 `10wvw01/vllm-ascend`；目标分支 `work/310p-mm-ar-large-m-a1-20261026`（PR#9 Draft）；目标提交 `d89204b0a14af5b1469b2f9f397e32d5c77b6088`。
> 隔离工作树 `/home/w00804146/310P/vllm-ascend-a1`（证据分支 `work/310p-a1-cpu-g0-acceptance-20261026`，基于目标提交创建）。执行合同：[A1_CPU_G0_ACCEPTANCE_20261026.md](A1_CPU_G0_ACCEPTANCE_20261026.md) §3 第一阶段。
>
> **总裁决：A1 CPU 验收 PASS（4/4 项，原仓 checkout 直接执行，补齐交接文档全部"未完成"CPU 项）；G0 静态指纹 INCONSISTENT（3/5 不符，属 F1-C 收尾还原后的预期基线态，非篡改；fail-closed，未签发任何批准）；设备异常 STOP（目标卡 NPU43520 Die7+MCU 新增 Alarm，双 Die devmanager 通道故障窗口）→ M2048 短冒烟前置条件不满足。**

本轮**未执行** torchrun、`u1_run.sh` 任何阶段、任何设备 Kernel、SO 替换、模型推理；**未设置**任何 `U1_*_APPROVED` 标志；**未修改**主仓 Stock 服务树（`perf/310p-f1c-...` 分支与其 untracked 文件原样，NPU5664 上的既有占用未触碰）。

## 1. Git 身份与现场保护

| 项 | 结果 |
|---|---|
| worktree HEAD | `d89204b0a14af5b1469b2f9f397e32d5c77b6088` = 目标提交 ✓（`git rev-parse` 复核） |
| tracked-tree | clean（`git status --porcelain --untracked-files=no` 为空）✓ |
| 目标提交身份 | `test(310p-a1): avoid false positive on torch_npu mention in readonly-probe docstring`，与远程分支 HEAD 一致（fetch 自 origin） |
| 主仓（Stock 服务树） | 未 checkout/reset/clean/commit；会话前后 `git status` 一致（仅既有 untracked `memfabric_v2_callstack.html`） |
| 既有 8 个 worktree | 全部未触碰 |
| 软件栈 | Python 3.12.13；torch 2.10.0+cpu；torch_npu 2.10.0.post4；CANN 9.1.0（`/usr/local/Ascend/cann-9.1.0`）；npu-smi 26.1.1；MemFabric `/usr/local/memfabric_hybrid/1.2.0`（`latest` 指向一致）——与 F1-C/R1 G0 记录完全一致 |

## 2. CPU 测试结果（全部原仓 checkout 源码直接执行，非重建副本）

| # | 命令 | 结果 | 退出码 | 日志 |
|---|---|---|---|---|
| 1 | `python3 -m py_compile test/perf/a1_large_m_assess.py test/perf/a1_g0_readonly.py` | 编译通过 | **0** | `evidence/a1_cpu_g0_20261026/01_py_compile.log` |
| 2 | `python3 -m unittest discover -s test/ut -p 'test_a1_*.py' -v` | **Ran 28 tests ... OK**（test_a1_g0_readonly 10 + test_a1_large_m_assess 18） | **0** | `02_unittest_a1.log` |
| 3 | `python3 -m pytest -q test/ut`（全部 12 个 UT 文件，含既有 U1/F1/F1-C/Gate/C/D1a/V2 合同） | **82 passed, 27 subtests passed**，无失败/跳过错误 | **0** | `03_pytest_ut.log` |
| 4 | `bash -n test/perf/u1_run.sh` | 语法通过 | **0** | `04_bash_n_u1_run.log` |

补齐 [A1_CPU_G0_ACCEPTANCE_20261026.md](A1_CPU_G0_ACCEPTANCE_20261026.md) §1 所列"未完成"项：

- **原仓 `python3 -m unittest` / `pytest` / `py_compile` 独立执行日志**：本报告即为该证据；A1 项目原仓 CPU TEST 由 NOT_RUN → **PASS**（判定算法修复后的 28 项 A1 合同 + 82 项全套在真实 checkout 上通过）。
- **原 U1 `evaluate()` 双 Rank 文件级集成复验**：`test_u1_matrix_contract.py`（11 collected）在 #3 中实际执行并通过，覆盖 `u1_analyze.evaluate()` 的 rank0/rank1 原始文件配对、缺日志/NaN/协议状态/锚点缺失 fail-closed、Screen 绝不 GO、Smoke 不授权性能分析、Confirm 需独立审计等合同。

## 3. G0 只读取证（`a1_g0_readonly.py`，contract `A1_G0_READONLY_V1`）

- 调用：环境变量指向**当前实际部署路径**（非旁路文件），输出全新绝对路径 JSON；`ASCEND_RT_VISIBLE_DEVICES=6,7`。完整口径见 `evidence/a1_cpu_g0_20261026/00_invocation_record.txt`。
- `hardware_executed=false`、`device_kernel_executed=false` ✓（脚本未导入 torch_npu、未触碰设备）。
- Git：HEAD `d89204b0a…` ✓、tracked-tree clean ✓。
- 禁止绕过变量：`MF_KFC_BYPASS` / `MF310P_V2_NOWARMUP` / `MF310P_V2_AR_ONLY_NOWARMUP` 均**未设置**（presence-based 检查为空）✓。
- 工具依赖：python3/torchrun/timeout/sha256sum/nm/npu-smi/readlink 全部在位 ✓；`npu-smi info` 只读快照成功 ✓。
- 指纹核对（实际部署路径 vs `u1_spec.py` 冻结常量）：

| 工件 | 实际路径 | 实测 SHA256 | pinned | 判定 |
|---|---|---|---|---|
| Python 插件扩展 | `/home/w00804146/310P/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` | `8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1` | 同左 | ✓ 一致 |
| KVER11 JSON | `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json` | `25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78` | 同左 | ✓ 一致 |
| D4 设备 SO | `/home/w00804146/310P/vllm-ascend/vllm_ascend/libmf310p_v2_trial_device.so` | `b8ca96f0387c45ec08e61ec3606f1502da22944a61b139815b62a20695a06811` | `2a19390c…c2d2a19` | ✗ 不符（F1-C 收尾还原 pre-F2 态） |
| MemFabric Host SO (lib64) | `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so` | `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1` | `1dc46d7d…f91a30` | ✗ 不符（还原原版） |
| MemFabric Host SO (site-packages) | `/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so` | `88d39285…7bc1`（与 lib64 同源） | `1dc46d7d…f91a30` | ✗ 不符（同上） |

- **工具裁决（如实 fail-closed）**：`static_fingerprint_consistent=false`（3 项 fingerprint 错误）；`runtime_loaded_plugin_verified=false`（设计上不在本工具内证明）；`G0_APPROVED=false`、`SMOKE_APPROVED=false`；6 项 `human_required` 人工取证项原样输出。完整 JSON：`evidence/a1_cpu_g0_20261026/a1_g0_readonly.json`。
- **3 项不符的性质认定**：与 [U1_FULLCHAIN_ATTRIBUTION_PLAN_20261025.md](U1_FULLCHAIN_ATTRIBUTION_PLAN_20261025.md) §"F1-C实验收尾曾恢复旧Host/Device SO"及 [F1_ONDEVICE_REPORT_20261024.md](F1_ONDEVICE_REPORT_20261024.md)（"由 D4 Tile2048 `2a19390c…` 还原为 pre-F2 安装态 `b8ca96f0…`（复核一致）"）完全吻合，属**实验收尾的预期还原态**，不是身份篡改。pinned 测试版工件在服务器在位可查：
  - D4 Tile2048 诊断 ON 设备 SO `2a19390c…` = `/tmp/opencode/d4/build/device_candidate_diag_on.so`（SHA 复核一致，来源提交 `5adbf1810`）；
  - 物理设备号修复版 Host SO `1dc46d7d…` = `/tmp/opencode/mf-fix/build/src/hybm/csrc/libmf_hybm_core.so`（SHA 复核一致，mf 仓 `fix/310p-setaccess-physical-devid-20261023` @ `5802f550`）。
  - 见 `evidence/a1_cpu_g0_20261026/08_backup_and_pinned_artifacts.txt`。

## 4. 独立审查结果

### 4.1 Python 实际插件加载路径

- site-packages 为 editable 安装 `vllm_ascend-0.1.dev4877+gb9a293d95`，映射到 `/home/w00804146/310P/vllm-ascend/vllm_ascend`（主 worktree，非 A1 工作树）。
- `source /usr/local/memfabric_hybrid/set_env.sh` 后按 `u1_run.sh` 同款口径实际导入：`vllm_ascend.vllm_ascend_C` → **`/home/w00804146/310P/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`**，与 `V2_HOST_LIB` 为同一文件，`readlink -f` 等价，SHA `8eb65de9…` = pinned ✓（`06_plugin_load_path.txt`）。
- **注意**：不 source `set_env.sh` 时裸导入该扩展报 `ImportError: libmf_hybm_core.so: cannot open shared object file`——任何 U1/后续实机进程都必须先 source MemFabric 环境（与 F1-C 现场口径一致）。

### 4.2 KVER11 / AICPU 状态

- KVER11 JSON 在位、SHA `25e16b0a…` = pinned ✓；内容登记 `opKernelLib=KFCKernel`、`kernelSo=libmf_sdma_orch_v11.so`、`functionName=HybmSdmaOrchEpoch`。
- `aicpu_kernel/` 目录实际文件为 `libmf_sdma_orch.so`（2026-10-12 起，历史各轮一致）；文件名与 JSON `kernelSo` 字段的解析关系由 KFC 运行时处理，历史上已实证。
- 设备侧 KVER11 AICPU 注册**最后一次实证**为 2026-10-25 F1-C T1 Smoke（deploy helper 双 die 报 `kernel already present`，见 `evidence/f1c_20261025/G0_PRECHECK_REPORT.md`）。本轮按禁令未运行任何 AICPU 探测/部署工具（`orch_deploy_helper` 为 ELF 设备操作程序，只登记不执行）；**当前注册状态 = UNKNOWN，须在设备告警解除后的独立 G0 中重验**。

### 4.3 设备健康与双 Die 同卡独占（本轮新发现，阻断性）

- **同卡映射**：设备 6/7 同属 NPU43520、同 Bus `0000:AB:00.0`、同 310P3 单卡 ✓（`npu_smi_info.txt`）；当前卡上**无进程**、显存近基线（1608/1339 MB），独占性表面成立。
- **健康异常（STOP 项）**：`npu-smi info -t health -i 43520`：Chip0(Die6)=OK，**Chip1(Die7)=Alarm，MCU=Alarm**（`npu_smi_health_43520.txt`）。对照 2026-10-25 R1 G0 快照（双 Die 均 OK），**该告警为一日内新出现**。
- **驱动错误窗口**：dmesg 2026-10-26 03:05:51–04:02:52，dev6 与 dev7 的 `common_msg_devmanager` 通道连续 `Message send finish, no resp. ret=-38`（kworker/npu-smi/python/msprof/asc_dumper 全部受影响，共 158 条；`dmesg_drv_pcie_full.txt`）；04:02 后未再复现，npu-smi 当前可正常读取。板级信息 Chip Fault=0、Compatibility OK（含序列号的板卡文件仅服务器存证，未入库）。
- 其他卡：NPU5664（0/1）npu-smi 列出 PID 895887/3589835 但宿主进程表已无此 PID（跨命名空间/陈旧占用，未触碰）；NPU38144 设备 4 有 ~13GB 残留显存无属主（历史泄漏残留，与 R1 记录一致，未动）；NPU16384（2/3）正常空闲。

### 4.4 备份与恢复方案

- 当前 live 双路径 Host SO（`88d39285…`）与主树设备 SO（`b8ca96f0…`）= F1-C 还原点，与备份 `/tmp/opencode/f1c_g0/backup_20261024T165834/`（`SHA256SUMS` + 三件 SO）及 `/tmp/opencode/sel01_20261024/env/` 逐字节一致（`08_backup_and_pinned_artifacts.txt`）。
- 若后续按 U1 计划受控部署 pinned 三工件，恢复方案为：从上述备份回拷双路径 `libmf_hybm_core.so`（目标 SHA `88d39285…`）与主树 `libmf310p_v2_trial_device.so`（目标 SHA `b8ca96f0…`）并复核 SHA；`u1_run.sh` 本身要求 `U1_RESTORE_PLAN_VERIFIED=YES` 且逐项 SHA 门禁才会放行。
- 507014/507899 停测流程：按 [PROJECT_CONTROL.md](PROJECT_CONTROL.md) §3 与 A1 合同——出现即停、保留双 Rank 原始日志与现场、隔离 Die/KFC、不得重跑"幸存样本"。

### 4.5 门禁健全性（fail-closed 复核）

`u1_run.sh` 在当前环境下的行为：`U1_G0_APPROVED`/`U1_EXCLUSIVE_APPROVED`/`U1_RESTORE_PLAN_VERIFIED` 未设置 → 立即 exit 2；即使补设批准，3 项 SHA 不符仍会 exit 2；危险绕过变量按 presence 拒绝（`0` 也不行）。`a1_g0_readonly.py` 不设置任何 `*_APPROVED`，静态一致也只输出 `false`。**无绕过路径，无需也未曾禁用任何断言。**

## 5. 裁决与 M2048 短冒烟前置条件

| 判定项 | 结果 |
|---|---|
| A1 CPU 验收（4 项命令） | **PASS** |
| 交接文档 §1 未完成 CPU 项 | **已补齐**（原仓 unittest/pytest/py_compile 日志 + U1 `evaluate()` 双 Rank 合同复验） |
| G0 静态指纹 | **INCONSISTENT（3/5）**——预期还原态；插件/KVER 一致，设备 SO 与双路径 Host SO 需受控重部署后才可复核为一致 |
| 设备 | **异常 STOP**：NPU43520 Die7+MCU Alarm（10-25 后新增）+ 双 Die devmanager 故障窗口（10-26 03:05–04:02）；当前无进程、显存近基线 |
| Python 插件运行时身份 | ✓ 与 pinned 一致（需 source `set_env.sh`） |
| KVER11 AICPU 当前注册 | UNKNOWN（最后实证 2026-10-25；待设备恢复后独立 G0 重验） |
| **M2048 双 Rank 短冒烟前置条件** | **不满足** |

M2048 短冒烟解锁前必须依次完成（缺一不可，本轮均未完成、未授权）：

1. NPU43520 Die7/MCU 告警根因定位与恢复，由运维/授权人书面确认设备健康（含 6/7 双 Die 独占复核）；期间 507014/507899 停测流程待命。
2. 按 U1 计划受控部署 pinned 三工件（先备份并记录 SHA/恢复命令）：设备 SO `2a19390c…`、双路径 Host SO `1dc46d7d…`；部署后重跑 `a1_g0_readonly.py` 复核 `static_fingerprint_consistent=true`。
3. 独立 G0 人工取证：真实加载 SO/KVER11 AICPU 注册（设备侧）、双 Die 同卡独占、Stock HCCL 与 MemFabric 通信进程隔离审查、恢复方案可用性。
4. 由授权人显式设置 `U1_G0_APPROVED` / `U1_EXCLUSIVE_APPROVED` / `U1_RESTORE_PLAN_VERIFIED`，并单独批准 M2048 smoke（Screen/Confirm 仍需各自独立批准）。

## 6. 证据清单

入库（本分支 `csrc/_310P/memfabric_mm_ar_v2/evidence/a1_cpu_g0_20261026/`，服务器原件目录 `/home/w00804146/310P/a1_cpu_g0_evidence_20261026/`）：

`00_invocation_record.txt`（调用与环境口径）、`01_py_compile.log`、`02_unittest_a1.log`、`03_pytest_ut.log`、`04_bash_n_u1_run.log`、`05_g0_readonly_run.log`、`06_plugin_load_path.txt`、`07_versions.txt`、`08_backup_and_pinned_artifacts.txt`、`a1_g0_readonly.json`（全新生成）、`npu_smi_info.txt`、`npu_smi_health_43520.txt`、`npu_smi_common_43520.txt`、`dmesg_drv_pcie_full.txt`。

仅服务器存证（含设备序列号 `2106…1155`，按 A1 合同不入库）：`npu_smi_board_43520.txt`。

## 7. 落盘建议（供主线合入时采纳，本分支不修改主线文档）

- `docs/PROJECT_CONTROL.md`：U1 行状态由"CPU/设备实机 NOT_RUN"更新为"CPU 合同已原仓复验 PASS（本报告）；设备实机仍 NOT_RUN 且被 NPU43520 告警阻断"；§3 风险表新增 NPU43520 Die7/MCU 告警条目。
- `docs/DECISIONS.md`：记录"A1 CPU 验收通过 + G0 fail-closed + 设备 STOP"裁决。
- A1 分支 PR#9 描述可引用本报告作为第一阶段服务器回执。

**本轮交付的是 CPU 判定与只读取证，没有任何新的硬件性能数据；V2 完整算子 vs Stock（M2048/4096）仍 UNKNOWN。**
