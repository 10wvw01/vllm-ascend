# Gate E S3-R 实测报告：MemFabric DeviceID 修复版 Host SO 受控冒烟回归（2026-10-26，310P3）

> 执行方：OpenCode @ 310P3（<host>），2026-10-26 14:54–15:16 UTC。
> 授权：仅"修复版 Host SO 部署验证 + 单次 90 秒融合冒烟"；S4/S5 未授权，本轮未触碰。
> 证据目录（服务器本地）：`/home/<user>/310P/gate_e_s3r_evidence_20261026/`；会话原始目录在
> 隔离 worktree `tmp/opencode/gate_e_forensics/`（作废轮 `S3R_…T145859Z` + 正式轮 `S3R_…T150457Z`）。
>
> **总裁决（一句话）：DeviceID 修复版对 H1 的 507899 阻断**回归通过**（双 Rank meta GVA 授权在
> 设备 2,3 上全部通过，507899 零出现）；但暴露**下一个独立阻断**——卡 16384 上 KFC 通道
> AICPU 内核发射返回 507018（ACL_ERROR_RT_AICPU_EXCEPTION），编排器部署链失败 →
> `smem_shm_create` 失败 → Worker 崩溃，**0 请求完成，首调 bit-exact 未达成**。
> 修复版以**进程隔离（LD_PRELOAD）装载，生产双位置零改动**（前后哈希断言）。
> 按授权边界停止，无任何额外设备操作。**

## 1. 版本与修复审查（observed）

| 项 | 值 | 判定 |
|---|---|---|
| 旧 Host SO（生产双位置） | `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1` | ✓ 与 H1/基线一致 |
| 修复版（mf-fix 构建） | `1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30` | ✓ 与授权值一致 |
| 修复提交 | `5802f550` "convert logical device id to physical for aclrtMemSetAccess"（mf-fix 树 HEAD） | ✓ |
| 修复范围 | 3 文件 +36/−3：`hybm_define.h`（新增 `HybmPhysicalDeviceIdOf` 内联换算）、`hybm_gva.cpp`（meta 区 1 处）、`hybm_acl_vmm_segment.cpp`（自持/导入段 2 处）——**仅 3 个 `AclrtMemSetAccess` 调用点**；协议/IB/SDMA/DONE/AICPU 内核/其余 VMM 接口零改动 | ✓ 与授权审查要求一致 |
| ABI 合同 | 导出符号集 1764=1764 逐名一致、DT_NEEDED 一致、SONAME 一致（`libmf_hybm_core.so`）——库级即插即用 | ✓ |
| 二进制含修复的独立静态证明 | FIX 比 OLD 恰好多 **3 处 `getenv@plt` + 3 处 `strtoul@plt`**（精确对应 3 个内联换算点；特征串法不可区分，函数级判别可区分） | ✓ |
| 构建溯源 | mf-fix 树（`5802f550`）RELEASE + KVER11 + BUILD_PYTHON=ON；构建产物内嵌 `commit: 2f59dedb`（configure 时旧 HEAD 戳）与 `build time: Oct 23 2026 15:33:16`——**构建早于修复提交（15:50:55），系由未提交工作树构建、戳为旧 configure 头**；10-23 dev6/7 实机验证（修复提交信息载明）+ 本轮静态判别共同佐证 | 记录在案 |

## 2. 安全部署（observed）

- **B1 G0 重检**：设备 2,3（NPU 16384 同卡）无进程/回基线/Health 3×OK；端口空闲；网络隔离结论同 H1（容器 bridge，残余风险用户已批准）。
- **进程隔离方式装载（首选路径）**：
  - 第 1 次尝试（私有目录前置 `LD_LIBRARY_PATH`）**失败作废**：`gate_e_eager.sh` 内部二次 `source set_env.sh` 将生产 lib64 重新前置，Worker maps 显示装载旧库——于模型加载期（未健康/零请求/零 V2 调用）即被 maps 取证发现，干净中止（forensic 走 RC=1 "serve exited early" 设计路径；serve 主进程优雅退出失败后 KILL，已入案）；该会话**不构成设备数据点**。
  - 第 2 次尝试（**LD_PRELOAD 修复版**）成功：CPU-only dlopen 测试（完整复刻脚本二次 source 序列）证明预装库优先于任何 `LD_LIBRARY_PATH` 顺序；正式会话双 Worker maps 确认装载 `/tmp/opencode/gate_e_s3r/fixed_host_lib/libmf_hybm_core.so`（1dc46d7d），**生产副本未映射**。
- **生产零改动**：发射前后双位置 SHA256 断言均 = `88d39285…`（§5 恢复核验）；原装 SO 全量备份与断言式恢复剧本在位（`backup_prod_20261026/`，本轮未动用）。**未影响任何其他租户。**

## 3. S3-R 冒烟（正式轮 `S3R_FUSED_SMOKE_20261026T150457Z`，observed）

运行身份（与 H1 完全一致，除 Host SO 为修复版）：冻结分支隔离 worktree（maps 断言无混树：
`vllm_ascend_C=07c831f0…`、`libmf310p_v2_trial_device.so=b8ca96f0…`、`libmf310p_device.so=8447a55a…`
全部自隔离树；`libmf_smem.so` 自生产位）；模型/KVER11 JSON（`25e16b0a…` 锁定）/Chunk1024/Slots4/
原 Gate E 配置不变；设备 2,3。

时间线（UTC）：

| 时刻 | 事件 |
|---|---|
| 15:05:02 | serve 启动（LD_PRELOAD 修复版；watcher/guardian/launcher 独立会话，规避工具超时误杀） |
| 15:07:08 | maps：双 Worker 装载修复版 Host SO ✓ |
| 15:08:18 | healthy；**选层 80/80** |
| 15:08:1x | bench rep1 开始；首请求 prefill M=4096 → 首调 V2 |
| 15:08:31.7 | 双 Rank 进入 MemFabric 传输层初始化；**meta GVA 四步（Reserve→Malloc→Map→SetAccess）全部通过——H1 的 507899 零出现** |
| **15:08:32.2** | **新阻断：双 Rank `KfcExec` 内核发射失败 ret=507018（ACL_ERROR_RT_AICPU_EXCEPTION）**，cmd=`test -f /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libmf_sdma_orch_v11.so` 与 `rm -f /home/CustAiCpuUser/lib/libmf_sdma_orch.so.*` |
| 15:08:32.2 | `deploy clean residue failed`（devId 0/1 对称）→ `InitTransManager`/`hybm_create_entity`/`smem_shm_create` 全链失败 |
| 15:08:32.7 | 双 Worker C++ `terminate`（HCCL watchdog 异常终止，SIGABRT——非 H1 的 Python 异常路径） |
| 15:08:53–15:09:00 | EngineCore 判定 Worker 意外死亡 → APIServer 关停 |
| 15:08:30–15:09:55 | rep1 实际 0 完成（首请求随 Worker 崩溃失败）；reps 2–5 为 0.04 秒级空轮（连接拒绝，无设备接触）；"load done after 5 reps"（脚本退出码 0 弱点再现） |

**验收结论**：
- ✅ 双 Rank MemFabric meta GVA 初始化成功（修复版生效——本轮核心回归目标达成）
- ❌ KFC 自动部署/加载：**未达成**（发射即 507018，两 Die 均失败，非单 Die 问题）
- ❌ 首调 bit-exact / V2 实跑 / 请求完成：**未达成**（传输层初始化在 KFC 阶段死亡）
- ✅ 80/80 选层行仅为路由证据，未作为 Kernel 命中证明（符合授权要求）
- 离线判读器：`INCONCLUSIVE_EVIDENCE`（completed=0，无 507014/507899 字样）

### 3.1 监护执行情况（诚实记录）

Guardian 未触发：模式表含 507899/507014/WorkerProc 异常等，**未含 507018 与 "died unexpectedly"**；
Worker 经 C++ terminate 死亡亦不产生 "WorkerProc hit an exception" 签名。后果：Worker 崩溃后
reps 2–5 仍被发起（各 0.04 秒、连接拒绝、零设备接触——与 H1 的 7 轮空打同类但无设备侧影响）。
**偏差如实入案；后续轮次模式表必须加入 507018、"died unexpectedly"、"smem_shm_create"、"HCCL watchdog"。**

## 4. 新阻断 507018 的初步归因边界（供裁决，不越权定性）

- 失败点：`AclrtLaunchKernelWithHostArgs`（KFC `system` 内核）同步返回 507018——**发射失败**，
  非设备侧命令退出码；`AclrtBinaryLoadFromFile`（mode=0）与 `GetFunction("K")` 均成功。
- 对称性：双 Rank（设备 2、3）同毫秒级对称失败——卡级而非单 Die。
- 历史/旁证：KFC 链在设备 0,1（历轮）与 6,7（10-23 修复验证，"KFC auto-deploy completes on
  fresh dies"）均工作；**卡 16384 从无 KFC 成功记录**（本轮首次到达该阶段）。
- 跨卡只读对比：DDR 带宽 51% 为 3/4 卡共性（非 16384 特有）；AICPU 占用全 0；`err-count`/
  `aicpu-config`/`custom-op-secverify` 查询本机不支持——**未发现宿主可见的卡级差异**。
- 待排除的干扰项（开放问题）：
  1. **LD_PRELOAD 对 CANN 的潜在干扰**：mf-fix 自带文档警告 `LD_PRELOAD` 曾致驱动版本检查
     异常（ASAN 语境）；10-23 dev6/7 验证的修复版装载方式**无记录**（可能为临时替换而非预装）。
     本轮多数 CANN 调用（VMM 全序列/二进制装载）在预装下正常，仅 KFC 发射失败——干扰可能性低
     但未排除。
  2. 卡 16384 AICPU 子系统状态（需平台侧判读或最小 KFC 探针——属新设备操作，需另行授权）。
  3. serve 上下文 vs 10-23 试验 harness 上下文差异（2,3 上从未跑过 trial harness）。

## 5. 环境恢复核验（observed，15:15:25Z，全部通过）

生产双位置 `libmf_hybm_core.so` = `88d39285…`（未变）；v11 so/JSON 未变；私有修复版副本在位；
进程/端口全清；设备 2,3 回基线（1930/1017MB）且 Health 3×OK；worktree porcelain 3/0/0（主树仅
既有 3 个未跟踪文件）；watcher/guardian/launcher 全部退出。**本轮对系统与共享安装的净影响为零。**

## 6. 裁决（按移交文档 §7）

- **DeviceID 修复回归：PASS**——H1 的 507899（`AclrtMemSetAccess` 逻辑/物理号缺陷）在完全相同
  的工况（设备 2,3、同一 serve 配置、仅 Host SO 换修复版）下**未再出现**，且 meta GVA 初始化
  首次在该非零起点映射下完整通过。该结论可直接移交 MemFabric 团队。
- **S3-R 冒烟整体：INVALID/UNKNOWN（SMOKE_BLOCKED_BY_KFC_LAUNCH_507018）**——修复版前进了一
  整个初始化阶段后被下一个独立阻断拦截；0 请求、首调 bit-exact 未达成；与 Gate E 507014 目标
  故障无关（负载从未建立），不并入复现率统计。
- S4/S5：维持 NOT_RUN。

## 7. 下一裁决的候选路径（供决策，不越权执行）

1. **归因 507018**：由 MemFabric/平台方判读卡 16384 AICPU 对 KFC `system` 内核发射异常的原因
   （对照 0,1/6,7 成功史）；或授权一次最小 KFC 探针/trial-harness 试验（新设备操作，需批准）。
2. **排除 LD_PRELOAD 干扰**：若裁决需要，可经批准后以"断言式临时替换生产库 + 完整备份回滚"
   方式复测（影响其他租户风险需先评估——本机其他租户进程在卡 5664/43520 上活跃，替换期间其
   新进程将装载修复版；修复版 identity 映射行为不变，但属共享库变更，须用户明示批准）。
3. 等待历史卡 5664（0,1）释放后在 identity 映射下复测修复版（同轮可一并验证 KFC 链）。
