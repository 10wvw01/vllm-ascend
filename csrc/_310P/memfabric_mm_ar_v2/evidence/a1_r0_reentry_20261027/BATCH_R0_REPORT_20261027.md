# BATCH-R0 报告：310P3 服务器重进场只读审计、A1 CPU 回归与 MemFabric 溯源（2026-10-27）

> run_id `a1-r0-20261027-001`；执行分支 `work/310p-mm-ar-large-m-a1-20261026` @ `235de9f6d13a394ffaaddde1db8843154aeb8813`（= 远程 tip，PR #9 Draft）；隔离工作树 `/tmp/opencode/a1-r0-worktree`（本会话新建，未触碰任何其它任务工作树/进程/已装环境）。
> 依据：主线 `dev/310p-memfabric-mm-ar` 总账与本分支 `A1_REENTRY_BUILD_DEPLOY_AND_BATCH_GATE_20261027.md`。**本批全程零设备计算、零 KFC、零建池、零 torchrun、零 SO 写入、零全局 pip。**

## 最终裁决

| 字段 | 结论 |
|---|---|
| **CPU_PASS** | **PASS**（py_compile 0；unittest 34/34；pytest 88+27 subtests 全绿；bash -n 0；纯CPU DeviceID 边界测试 15/15；u1_run.sh 门禁与五个冻结SHA静态核查原样在位、未被改动） |
| **G0 建议** | **有条件 GO 建议（待用户批准）**：唯一推荐卡 **16384（dev2,3）**，当前健康、空闲、无可见占用、与已知干净基线一致；但正式 G0 批准须由用户在 E2 前复核本报告并另行签发。工具层面 `G0_APPROVED=false` 恒定，未设置任何 U1 批准标记 |
| **MEMFABRIC_PROVENANCE** | **HOST_SOURCE_PROVENANCE_FOUND（路径 A 成立）**：冻结 `1dc46d7d…` Host SO **两份完整副本均在服务器**（mf-fix 历史构建产物 + Gate E S3-R 已验证 staging 副本，SHA 逐一复核一致）；对应源码提交 `5802f550`（分支 `fix/310p-setaccess-physical-devid-20261023`，基于镜像 main `131ee5ff`）仍在本地、patch 已归档；**未推送任何远端**；S3-R/S3-R2 实机验证链完整。**当前生产双路径 Host SO 仍为未修复 `88d39285…`**（与合同预期一致：修复从未永久装机，S3-R/S3-R2 走 LD_PRELOAD） |
| **BUILD_STATUS** | **NOT_ATTEMPTED（Host SO）/ 纯CPU测试构建 PASS**：按合同 §2.2-1，正确历史库**不必无条件重编**；Draft PR #1 候选（`33a2df0d`）按合同仅作兜底，**未构建、未部署、不得宣称与 `1dc46d7d` 字节等价** |
| **DEPLOYMENT_READY** | **NO_GO（待用户审核后另行授权 E2）**：staging 产物与回滚方案已备妥（见 build_manifest.json），但共享目录写入、KFC、建池、双Rank验证均属 E2+，须用户审阅本报告后逐项放行 |

## 1. 环境状态（E0，全部只读）

- **四卡拓扑与健康**（310P3 ×4，每卡双Die，共 8 设备）：
  - `5664`(dev0,1)：OK，被其它任务占用（llama-server ~13GB + VLLM::EngineCore ~42.7GB，进程在其它容器命名空间，不可见且**未驱赶**）→ 排除
  - `16384`(dev2,3)：**双Die OK、零可见进程、内存 ~4.4%/2.3%、hugepages 0/0**，与 S4 报告记载的干净 G0 基线一致；该卡即 S3-R/S3-R2 用修复 SO 验证过的卡 → **首选**
  - `38144`(dev4,5)：健康但 **dev4 残留 ~13GB 孤儿内存 + hugepages 5860/5860 全占**，无可见属主（疑似 10-26 容器崩溃风暴残留）→ **暂不选用**，留待宿主侧查清
  - `43520`(dev6,7)：Die7 Chip+MCU **Alarm** 且被占用 → 按 ADR-014 排除（记录，不处置）
- **dmesg devmm 风暴**：最后一条 2026-10-26 17:19Z（其它容器 vllm 崩溃循环已停），当前无新增
- **有毒环境变量**：`MF_KFC_BYPASS` / `MF310P_V2_NOWARMUP` / `MF310P_V2_AR_ONLY_NOWARMUP` / `MF_SDMA_ORCH_DEPLOY_HELPER` / 常驻 `MF_SDMA_ORCH_EPOCH_BUDGET_S` 在会话、`/etc/profile.d`、`/root/.bashrc`、`/etc/environment` 中**均未出现**
- **设备侧 KFC** `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/`：空（符合"只读阶段不预部署"）
- 工具 `a1_reentry_readonly.py` 正常产出 `environment_inventory.json`；其 `three_setaccess_mapper_static_hint=False` 系因历史树 mapper 定义在 `hybm_define.h`（hint 预期 Draft PR#1 新文件布局 `hybm_physical_device_id.h`），**不影响路径A结论**，人工核验 3 处 SetAccess 全部经物理号换算

## 2. 版本身份（完整 SHA 取证）

| 资产 | 冻结值 | 现场实测 | 一致性 |
|---|---|---|---|
| A1 分支 HEAD | — | `235de9f6d…`（=origin tip） | ✓ |
| 主工作树（其它任务） | — | `perf/310p-f1c-…-20261025 @ bd422330a`，tracked clean（未动） | 记录 |
| CANN / driver / npu-smi | — | 9.1.0 / 26.1.1 / 26.1.1 | =修复验证时 CANN |
| Python / torch / torch_npu | — | 3.12.13 / 2.10.0 / 2.10.0.post4 | 记录 |
| vllm（editable） | — | 0.27.1+empty → `/vllm-workspace/vllm/vllm`（detached 6e448d0，clean） | 记录 |
| 插件 `vllm_ascend_C` | `8eb65de9…` | 主工作树内文件 SHA = `8eb65de9…`；editable 解析指向该目录 | **一致** |
| Device SO（D4 冻结） | `2a19390c…` | `/tmp/opencode/d4/build/device_candidate_diag_on.so` = `2a19390c…`（源 `5adbf1810`）；当前插件目录在装版为 baseline `b8ca96f0…`（E2 才可换） | **归档在位** |
| Host SO 双路径（现装） | 不得为 `88d39285…` | 两路径现装 = `88d39285…`（未修复原版） | **待 E2 修复** |
| Host SO（冻结修复版） | `1dc46d7d…` | mf-fix 构建产物与 gate_e_s3r staging 两份均 = `1dc46d7d…` | **完整在位** |
| KVER11 JSON | `25e16b0a…` | 在装 = `25e16b0a…` | **一致** |
| orch_deploy_helper | 849f93c0…（S3R2 记录） | 生产 lib64 与 mf-fix tools 二进制一致 | **一致** |

## 3. A1 原仓 CPU 回归（隔离工作树）

| 项 | 结果 |
|---|---|
| py_compile 三个 A1 工具 | exit 0 |
| unittest discover test_a1_*.py | **34/34 OK** |
| pytest 全 ut 目录 | **88 passed + 27 subtests**，0 fail |
| bash -n u1_run.sh | exit 0 |
| u1_run.sh 静态门禁 | 三批准变量 + 五冻结 SHA 原样强制，未被绕过 |
| 纯CPU DeviceID 映射测试（复制 `5802f550` 原函数） | **15/15 PASS**（unset/空/identity/2,3/4,5/6,7/空格/畸形/短表/越界） |

## 4. MemFabric 溯源结论（E-02 完成）

- 修复栈（本地分支 `fix/310p-setaccess-physical-devid-20261023`）：`2f59dedb`(KVER v11) → `acd626cc`(KFC helper) → `131ee5ff`(argv 修复，=镜像main) → **`5802f550`(SetAccess 物理号，未推送)**
- 源码核验：3 处 `AclrtMemSetAccess`（meta/own/imported）全部经 `HybmPhysicalDeviceIdOf` 换算；`aclrtMallocPhysical` 保持逻辑号；协议/IB/SDMA/AICPU 核未动——与合同"仅3处换算"完全一致
- 实机验证链：commit 自述（dev6/7 非零起点可见列表 507899 消除、8形状×2Rank bit-exact）→ S3-R（507899 回归 PASS）→ S3-R2（双Rank首调 bit-exact、25/25 请求、helper 模式 KFC 部署、507014/507899 零出现）
- 2026-10-23 备份目录内容为 `.orig=88d39285`（修复前原版），**不是修复版**；修复版在 mf-fix/build 与 gate_e_s3r/fixed_host_lib 两处
- 镜像 7/8 分支与 `gitcode origin` 均无 `5802f550`；Draft PR #1 候选 `work/310p-setaccess-redeploy-20261027@33a2df0d` 仅可作兜底重建来源（字节不等价、未构建）

## 5. 阻塞项与风险

1. **生产 Host SO 未修复**（`88d39285…` 在装）——非零起点可见列表（2,3）下必然 507899，是 E2 的直接部署动机，也是本轮唯一实质环境缺口
2. **38144 卡 dev4 孤儿内存**（13GB + hugepages 全占、无可见属主）——在宿主侧查清前该卡不可用，可用卡暂时只有 16384 一张
3. 插件 editable 指向主工作树，而主工作树当前在其它任务分支上：该分支的 `vllm_ascend/` 下插件 SO 恰与冻结 SHA 一致，**当前可用**；但若其它任务后续重建/切换该目录，运行时插件身份会漂移——E2 前需重新只读复核一次
4. CANN 9.1.0 probe 合同（MallocPhysical 逻辑号 / SetAccess 物理号）仍有效；**任何 CANN 升级都要求重新评审**，不得自动跑设备探针
5. `MF_SDMA_ORCH_DEPLOY_HELPER` 属受控运行变量（S3-R2 用法），**不得成为常驻 Serving 环境变量**

## 6. 建议的恢复部署方案（仅方案，未执行；E2 须逐项批准）

1. 用户审核本报告并签发 E2 授权后，先对 16384(dev2,3) 做一次只读复核（健康/空闲/独占）
2. 时间戳备份现装 `88d39285…` 双路径 → 从归档副本（重新验 SHA 后）将 `1dc46d7d…` 恢复到**两处生产路径**（保持属主/权限）→ 双路径复验 SHA
3. 如目标为 U1 大M实验：同批次将插件目录 `libmf310p_v2_trial_device.so` 从 `b8ca96f0…`（baseline）换为冻结 D4 `2a19390c…`（先备份）；插件 `8eb65de9…` 与 KVER11 `25e16b0a…` 已在位无需动
4. KFC 不预部署：首轮授权 Smoke 建池时由 MemFabric 自动部署 v11 编排核并双 Die 核验 `HybmSdmaOrchEpoch`
5. 回滚：恢复第2步备份即可回到当前态；不使用 `mini_kfc rm -f`
6. 之后按 BATCH-R1 合同执行受控双 Rank Smoke；任何 507014/507899/507018/错算即 STOP 保全

## 7. 证据

机器可读与原始索引见同目录 `evidence_index.json`；敏感主机原始信息（uname/ps/ss/npu-smi 全文）按合同仅存服务器 `/tmp/opencode/a1_r0_evidence_20261027/`（SERVER_ONLY）。

**本报告未设置任何批准旗标：`U1_G0_APPROVED` / `U1_EXCLUSIVE_APPROVED` / `U1_RESTORE_PLAN_VERIFIED` 均未定义；`G0_APPROVED=DEPLOY_APPROVED=SMOKE_APPROVED=false`。等待用户审核。**
