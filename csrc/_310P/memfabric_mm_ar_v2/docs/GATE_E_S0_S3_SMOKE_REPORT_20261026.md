# Gate E 移交前第二阶段 S0–S3 实测报告（2026-10-26，310P3）

> 执行方：OpenCode @ 310P3（<host>），2026-10-26 13:29–14:16 UTC。
> 授权依据：用户审阅 G0 报告后批准 CONDITIONAL GO，授权 S0–S2 + **单次** S3 90 秒融合冒烟；S4/S5 未授权。
> 入口：普通 Gate E 复现入口 `gate_e_forensic_run.sh fused` → `gate_e_eager.sh`（未使用强制设备 0,1 的 G1-D 诊断入口）。
> 证据目录（服务器本地）：`/home/<user>/310P/gate_e_s0_s3_evidence_20261026/`（编号文件与本报告对应；
> 会话原始目录在隔离 worktree `tmp/opencode/gate_e_forensics/` 下，均未覆盖）。
>
> **总裁决：S3 冒烟未通过——但失败不是 Gate E 507014 复现，而是 MemFabric 传输层初始化阻断
> （`AclrtMemSetAccess` 507899 = ACL_ERROR_RT_DRV_INTERNAL_ERROR，发生于设备 2,3 / 非零起点可见列表）。
> 按停止契约立即停止全部后续轮次，未做任何自动重试。S4/S5 维持未执行。
> 复审增补（同日归档轮）：507899 的**优先核查方向为已安装旧 Host SO `88d39285` 的逻辑/物理
> DeviceID 已知问题**（详见 §6.1），暂不定性卡 16384 硬件故障。**

## 1. 执行总览

| 阶段 | 内容 | 结果 |
|---|---|---|
| S0 | 设备 2,3（NPU 16384，同卡 0000:41:00.0）独占/健康/端口/进程 + 网络隔离评估 + 备份恢复能力 | **PASS**（含一次外来占用按纪律挂起 7 分钟后回基线） |
| S1 | 隔离 worktree 标准构建（无诊断开关） | **PASS**（12m42s，RC=0；产物与主树标准件逐字节一致） |
| S2 | PYTHONPATH 前置 + 包/C 扩展模块路径断言 | **PASS** |
| S3 第 1 次发射 | 编排事故（本方工具超时杀进程组），serve 未受载、无请求 | 作废（事故记录在案，serve 干净清退） |
| S3 第 2 次发射 | 90 秒融合冒烟 | **FAIL at 首调 V2**：双 Rank `MetaSetDeviceAccess` 507899 → `hybm_init`/`smem_shm_init` 失败 → Worker 崩溃；0 请求完成 |
| 环境恢复核验 | MemFabric/主树哈希、进程/端口/设备基线/worktree | **PASS**（全部回到发射前态） |

## 2. S0：前置门禁（observed）

- **设备**：NPU 16384 双 Die（Chip Logic ID 2,3，Bus 0000:41:00.0 同卡），无进程、AICore/AICPU 0%、
  DDR 1955/993MB（基线）、双 Die + MCU Health 全 OK；卡 5664（0,1）与 43520（6,7）确认被外部租户占用，
  43520 die7 Health=Alarm（均未触碰）。端口 8000/8581 空闲；无残留进程。
- **R5 实战**：13:45:51Z 设备 2 出现外来进程（PID 874846，893MB，命名空间外）——按纪律**挂起发射**并 30 秒级监测，
  13:52:36Z 外来进程释放、回基线后才继续。第二次发射前 14:04:50Z 又见 ~109MB 瞬时探测（1 秒级），
  14:05:42Z 复核已消失；正式会话内存分配期（14:06+）进程表仅本轮 serve（宿主命名空间 PID 936011/936198，
  与设备 slog `tgid[936011]` 互证）。
- **网络隔离**：容器（docker bridge <container-ip>，/.dockerenv 在位）内 `--host 0.0.0.0`；发射前监听 socket 为空；
  容器内无 iptables/nft 可查；对宿主/外网的暴露取决于宿主端口发布配置（容器内不可见）。历史 Gate E/F1-C
  同一绑定方式运行无事故；残余风险由用户批准接受。
- **备份恢复能力**：本轮**零部署**（构建产物仅落隔离 worktree gitignored 路径）；生产位基线发射前后哈希断言；
  应急恢复资产在位（`/tmp/opencode/g1c/kernel_v12_stage`、`g1d` 备份、`build-backups/g1d_backup_*`、
  `restore_g1d.sh` 断言式剧本）。

## 3. S1：标准构建（observed）

- 配方：与 G1-D 构建脚本同构但**不预置任何 G1C/G1D_TRACE/DIAGNOSTICS**；CANN 9.1.0 + memfabric set_env +
  `SOC_VERSION=ascend310p3` + V1/V2 使能 env；`python setup.py build_ext --inplace`，build-temp 在
  `/tmp/opencode/gate_e_20261026/build_v2_std`。12m42s，RC=0。
- 产物（全部在隔离 worktree `vllm_ascend/`，gitignored）：

| 产物 | SHA256 | 与主树标准件比对 |
|---|---|---|
| `libmf310p_device.so` | `8447a55a54d9d1b615353e2197e2057da52882bd6a991d2bbddcc3c574e755df` | **逐字节一致**（源码-二进制溯源独立复证） |
| `libmf310p_v2_trial_device.so` | `b8ca96f0387c45ec08e61ec3606f1502da22944a61b139815b62a20695a06811` | **逐字节一致** |
| `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` | `07c831f007b5eb23241e9be3038e4e4627c2b52317283c2b7f0fc2622f81350d` | 与主树 F1-C 构建不同（冻结分支 host 适配层源码不同，预期内） |

- 断言全过：无 `*_g1d.so`/`*_g1c.so`；DT_NEEDED 仅默认名（libmf310p_device/libmf_smem/libmf310p_v2_trial_device）；
  主树与 g1d 树 SO 构建前后哈希不变；隔离 worktree porcelain 0 行。
- **未覆盖现网 SO**：/usr/local 与 site-packages 生产位全程未写（前后哈希断言）。

## 4. S2：树身分断言（observed）

`source CANN set_env.sh` → `source memfabric set_env.sh` → `PYTHONPATH=<隔离worktree>:$PYTHONPATH` 前置后：
`import vllm_ascend` 解析至隔离 worktree（断言通过）；`find_spec("vllm_ascend.vllm_ascend_C")` 指向隔离树内
新构建 .so（未执行导入，避免 CANN RTS 早期初始化干扰设备选择——真实装载身份由 S3 运行中 maps 取证）。

## 5. S3：90 秒融合冒烟（observed）

### 5.1 第 1 次发射（13:53:19Z，作废）

编排事故：发射命令的进程组被本方工具 120 秒超时终止，forensic 编排脚本随组死亡（samples/.stop 13:55）；
serve 经 `setsid` 独立会话存活且健康，但**负载从未发出**（0 请求、无 V2 调用）。
处置：TERM 清退孤儿 serve（自有残留），设备 100% 回基线；事故全记录（`12_*.txt`）。
**该会话不构成任何设备侧数据点。**

### 5.2 第 2 次发射（14:04:51Z，setsid 脱离，正式冒烟）

**会话**：`H1_FUSED_SMOKE_20261026T140451Z`（证据目录见文首；发射/编排控制台 `13_s3_relaunch_console.log`）。

时间线（UTC，serve.log/runner.log/npu-smi 采样交叉）：

| 时刻 | 事件 |
|---|---|
| 14:04:51 | serve 启动（`ASCEND_RT_VISIBLE_DEVICES=2,3`，TP=2，全历史同款业务配置） |
| 14:06:43 | 双 Worker 设备侧 HCCP/TDT 初始化（slog，tgid 936011/936198） |
| 14:08:02 | maps 末次采样：双 Worker SO 装载身份正确（见 §5.3） |
| 14:08:12 | healthy；**选层 80/80**（40 层 × 2 Rank `Enable 310P3 TP=2 MemFabric unquantized mm_ar`） |
| 14:08:1x | bench rep1 开始，首请求 prefill M=4096 → 首调 V2 路由 |
| **14:08:26.168** | **rank0（TP0）`MetaSetDeviceAccess` 失败：ret 507899，addr 0x17ffbe000000，size 0x2000000，deviceId 0** |
| **14:08:26.264** | **rank1（TP1）同签名失败（deviceId 1，间隔 96ms）** |
| 14:08:26-28 | `hybm_init_hbm_gva`/`hybm_init` 双 Rank 失败 → `smem_shm_init` 失败 → 双 Worker `WorkerProc hit an exception` 崩溃 |
| 14:08:28-14:09:54 | bench rep2–rep7 全部瞬时失败（连接错误），"load done after 7 reps"，**0 请求完成** |
| 14:10:00 | forensic 脚本 exit 0（已知弱点：脚本级退出码不代表无故障），summary 初筛 UNKNOWN |

**离线判读**（`gate_e_handoff_assess.py --arm fused`，exit=0）：`FAULT_SIGNATURE_OBSERVED`（raw_507899_lines=2；
fusion_selection_lines=80；first_call_bitexact_lines=0；completed_requests=0）。判读器注记：
故障字样出现须先保全原始设备 slog/plog/Rank——已执行（见 §7）。

### 5.3 装载身份取证（运行中 /proc maps，第 2 次发射双 Worker，14:08:02 末次采样）

| SO | Worker_TP0 / Worker_TP1 实际装载路径 |
|---|---|
| `vllm_ascend_C.cpython-312-…so` | **隔离 worktree**（07c831f0 新构建） |
| `libmf310p_v2_trial_device.so` | **隔离 worktree**（b8ca96f0 标准构建） |
| `libmf310p_device.so` | **隔离 worktree**（8447a55a） |
| `libmf_hybm_core.so` | `/usr/local/memfabric_hybrid/1.2.0/.../lib64/`（生产位 88d39285；双位哈希一致已核） |
| `libmf_smem.so` | 同上生产位 |

异常 traceback 途经 `/home/<user>/310P/vllm-ascend-g1d-e7ba8a5b/vllm_ascend/worker/worker.py`——冻结分支代码在执行。
**结论：无混树；主进程/EngineCore/双 Worker 的 Python 包、Torch 扩展、V2 设备 SO、MemFabric Host 库装载身份全部符合要求。**

### 5.4 首调 bit-exact 与 V2 运行时命中

**未达成**（首个 V2 调用在传输层初始化即失败，`first fused call verified bit-exact` 0 行、
`V2 MM+MemFabric` 运行时行 0）。此为冒烟验收的**硬失败**，但失败点在 Gate E 历史故障面（运行中 AICore 停滞）
**之前**的初始化层。

## 6. 根因定位（observed + derived，含历史交叉对照）

**失败机制**（MemFabric 源码 `src/hybm/csrc/driver/hybm_gva.cpp`，与安装库 88d39285 行为一致）：
V2 首调 → `smem_shm_init` → `hybm_init` → `Hybm310PInitMetaGva` 四步（ReserveMemAddress →
MallocPhysical(PINNED, P2P_HUGE) → MapMem → **AclrtMemSetAccess 授权 AICore SMMU 读**）。
serve.log 中**首错即第 4 步**（前 3 步无错误日志），固定约定元区地址 `HYBM_DEVICE_META_ADDR=0x17ffbe000000`
（32MB）授权设备侧访问时驱动返回 **507899 = `ACL_ERROR_RT_DRV_INTERNAL_ERROR`**（CANN rt_error_codes.h:148），
双 Rank 对称失败。

**交叉对照**：

| 会话 | 日期 | 设备 | Host 库 | hybm init 结果 |
|---|---|---|---|---|
| G1-A F01/F02R1（Gate E 原工况） | 10-20 | **0,1**（NPU 5664） | **88d39285（与今天完全相同）** | **静默成功**（仅同样两条 WARN，无任何 ERROR；首调 bit-exact 双 Rank 通过） |
| G1-D D1R1–R3（诊断轮） | 10-21 | 0,1 | 6e9dc2d6（诊断版） | 成功（显式 INFO 日志） |
| **本轮 H1 冒烟** | 10-26 | **2,3（NPU 16384）** | **88d39285（发射前后哈希断言）** | **`MetaSetDeviceAccess` 507899 双 Rank 失败** |

（derived）**同一 Host 库、同一驱动（26.1.1，Innerversion V100R001C10SPC009B220，与 10-21 D1R1 slog 记录一致）、同一环境配方**，在设备 0,1 成功、
在设备 2,3 失败于驱动内部错误。**现场报告初版曾将阻断因素归为"卡 16384 的设备侧状态/能力"；
该定性已被 §6.1 复审增补修正——优先方向为已安装旧 SO 的已知逻辑/物理 DeviceID 缺陷，非卡硬件故障。**
**本轮按契约未做任何重试/换卡/换库验证。**

### 6.1 复审增补：507899 优先核查方向 = 旧 Host SO 的逻辑/物理 DeviceID 已知问题（2026-10-26 归档轮）

用户提供并于 10-23 实机验证过的既有事实（本报告据以修正定性）：

- **已知缺陷**：现网安装的旧 MemFabric Host SO `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`
  在 `MetaSetDeviceAccess`（`hybm_gva.cpp`）中把 **`aclrtGetDevice` 返回的逻辑设备号**（经
  `ASCEND_RT_VISIBLE_DEVICES` 重映射后的连续编号）直接填入 `aclrtMemSetAccess` 的
  `acc.location.id`——而该 API 要求**全局物理设备号**。identity 映射（未设置可见列表或 `"0,1"`）下
  逻辑==物理，缺陷从不暴露；**非零起点可见列表（如本轮 `"2,3"`）下以逻辑号授权即 507899**
  （驱动内部错误）。
- **修复版**：memfabric 修复提交 `5802f550`（"convert logical device id to physical for
  aclrtMemSetAccess"，新增 `HybmPhysicalDeviceIdOf` 按可见列表换算，仅 SetAccess 消费、其余 VMM API
  保持逻辑号），修复版构建产物 SHA256
  `1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30`
  （服务器本地 `/tmp/opencode/mf-fix/build/src/hybm/csrc/libmf_hybm_core.so`，已复核哈希一致）。
- **与 H1 现场证据吻合**：本轮失败日志双 Rank 打印 `deviceId: 0` / `deviceId: 1`——恰为逻辑号，
  而物理设备实为 2,3；10-20 G1-A F01 在 `"0,1"` 下同一 SO 静默成功（逻辑==物理）；本轮 `Reserve/
  MallocPhysical/MapMem` 三步（实测要求逻辑号）均无错误、仅 SetAccess 失败——全部指向该已知缺陷，
  **不支持卡 16384 硬件故障定性**。
- **纪律**：按指示，本轮**未替换任何 SO**；修复版是否部署、何时部署由用户/MemFabric 方裁决
  （若部署须按既有断言式备份/恢复纪律，先列变更前后 SHA256 并获批）。

## 7. 已保全证据（路径）

- 会话目录（隔离 worktree `tmp/opencode/gate_e_forensics/`）：`H1_FUSED_SMOKE_20261026T135319Z`（事故作废轮，
  含 13:53–13:55 npu-smi 采样）与 `H1_FUSED_SMOKE_20261026T140451Z`（完整 serve.log/runner.log/bench.log/
  7×stability_rep*.json/summary/assessment/provenance/15 秒级 npu-smi 与进程采样）。
- 设备侧原件拷贝（`device_logs_failure_window/`）：TP0/TP1 slog（device-2/device-3）、双 Worker plog、
  APIServer/EngineCore plog。
- 运行中 maps 取证：`10_s3_proc_maps_capture.log`（41+ 次采样，双 Worker×12 次完整 SO 身份）。
- 全部编号证据 00–20（本目录）。

## 8. 环境恢复核验（observed，14:16:03Z）

| 项 | 结果 |
|---|---|
| MemFabric 生产位（双位置 core.so / v11 so / v11 JSON） | = 发射前基线（88d39285 / db7134b8 / 25e16b0a）✓ |
| 主树 SO（vllm_ascend_C / v2_trial_device） | = 发射前（8eb65de9 / b8ca96f0）✓ |
| 进程 / 端口 8000、8581 | 全部清退/释放 ✓ |
| 设备 2,3 | 回基线 1954/993MB、进程表空、Health 3×OK、无 507899 残留告警 ✓ |
| worktree porcelain | 主树仅 3 个既有未跟踪文件；g1d 与隔离树 0 行 ✓ |
| 诊断环境变量 | 无残留（本轮未设置任何诊断变量）✓ |

## 9. 裁决（按移交文档 §7 客观裁决表）

- **本轮不是 REPRODUCED**：Gate E 目标故障（负载期双 Rank AICore 停滞 + 507014）未发生——负载从未建立。
- **不是 CONTROL/NOT_REPRODUCED**：冒烟未通过验收（首调 bit-exact 未达成）。
- **裁定：INVALID/UNKNOWN（SMOKE_BLOCKED_BY_TRANSPORT_INIT_FAULT）**——融合冒烟在设备 2,3 上被
  MemFabric 传输层初始化失败（`AclrtMemSetAccess` 507899）阻断，先于任何 Gate E 有效测量。
  原始双 Rank 证据已保全，`ondevice_verified=false`。根因归属经 §6.1 复审增补修正：
  **优先方向为已安装旧 SO `88d39285` 在非零起点可见列表下的逻辑/物理 DeviceID 已知缺陷**
  （修复版 `1dc46d7d…` 已在 10-23 验证存在），暂不定性卡 16384 硬件故障；最终定性待 MemFabric 方复核。
- 附注（诚实记录）：判读器机械输出 `FAULT_SIGNATURE_OBSERVED` 系 507899 字样触发；人工判读确认其发生层
  与 Gate E 历史故障（运行期 AICore 超时/停滞）不同类，不应并入历史复现率统计。

## 10. S4 裁决所需的依赖（供决策，不越权执行）

1. **阻断项（优先）**：507899 归因于旧 SO `88d39285` 的逻辑/物理 DeviceID 已知缺陷（§6.1）。
   两条可选路径，均需用户/MemFabric 方批准后执行：
   a. **部署修复版 SO `1dc46d7d…`**（断言式备份/恢复纪律：变更前后双位置 SHA256、备份、回滚剧本），
      在设备 2,3 重跑 S3 冒烟；
   b. **不换库**，改在历史卡 5664（设备 0,1，identity 映射下逻辑==物理）执行 S3 冒烟——但该卡
      当前被外部租户占用（13GB/42.7GB），需等待释放并核验干净。
   注意路径 b 只绕开缺陷、不消除缺陷；移交 MemFabric 团队时仍应如实记录该已知问题。
2. 卡 38144（4,5）die4 有 13GB 无主残留内存，未列入候选。
3. 在上述任一路径就绪前，S4（620 秒复现）与 S5（Stock 对照）维持未执行。

## 11. 移交增补事实清单（供 MemFabric 团队）

- 507899 于本现场= `ACL_ERROR_RT_DRV_INTERNAL_ERROR`；发生点 `AclrtMemSetAccess(HYBM_DEVICE_META_ADDR=0x17ffbe000000, 32MB, ACL_RT_MEM_ACCESS_FLAGS_READWRITE, device)`。
- 同一安装库（88d39285，双位哈希断言）在 10-20 设备 0,1 静默成功、10-26 设备 2,3 双 Rank 对称失败
  （间隔 96ms），失败前的 Reserve/MallocPhysical/MapMem 三步均无错误。
- 双 Rank 各自授权**本端**设备（deviceId 0/1 为 `ASCEND_RT_VISIBLE_DEVICES=2,3` 重映射后的 rank 相对号），
  即非跨卡授权场景；**该逻辑号即 §6.1 已知缺陷的直接现场证据**——`aclrtMemSetAccess` 需要物理号 2/3，
  旧 SO 传入了逻辑号 0/1。
- 设备 slog 在故障时刻（22:08:26 设备本地时钟）最后活动为 `Get api RunCpuKernel from so libpt_kernels.so success`，
  无 AICPU 侧报错堆积。
