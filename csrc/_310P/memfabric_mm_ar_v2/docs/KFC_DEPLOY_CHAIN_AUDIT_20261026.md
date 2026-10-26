# Gate E KFC 部署链只读审计：S3-R 507018 根因判定与 S3-R2 准入（2026-10-26，310P3）

> 执行方：OpenCode @ 310P3（<host>），2026-10-26 16:30–16:50 UTC。**本轮仅 CPU/源码/日志核验，
> 零 NPU 操作、零部署、零环境变更。**
> 依据：用户对 S3-R 报告的复核指示；审计对象为 S3-R 会话
> `S3R_FUSED_SMOKE_20261026T150457Z` 完整 serve.log、MemFabric mf-fix 源码（`5802f550`）与现网安装。
> 证据目录（服务器本地）：`/home/<user>/310P/gate_e_kfc_audit_20261026/`。

## 1. 原因判定（observed + derived，证据链闭合）

**判定：S3-R 的 507018 级联不是卡 16384 硬件/AICPU 故障，而是"LD_PRELOAD 隔离装载使
`FindDeployHelper()` 的 dladdr 同目录查找落空 → 回退进程内 KFC 链 → 首次 `test -f` 探测在
torch_npu 宿主进程内非零退出 → 按源码注释（2026-10-18 实测）永久打死本进程设备上下文 →
后续一切内核发射 507018"。** 完整证据链：

1. **源码查找链**（`hybm_sdma_orchestrator.cpp:358 FindDeployHelper`）：
   `MF_SDMA_ORCH_DEPLOY_HELPER` env（X_OK）→ **dladdr 本 SO 同目录** `orch_deploy_helper` →
   `../bin/orch_deploy_helper` → 均无则返回空 → `RunDeployHelper` 记 INFO
   "deploy helper not found; falling back to in-process KFC deploy"（**默认日志级别 WARN，INFO 被过滤**
   ——`mf_out_logger.h:235 logLevel_ = WARN_LEVEL`，故 serve.log 无此行）→ 调用方回退进程内链。
2. **S3-R 实际模式 = 进程内回退（确证）**：serve.log 中 `kfc launch failed` 由 **Worker 进程**
   （pid 564112/564157，`KfcExec` 为库内函数）打印——助手模式下"本进程零 KFC 命令"（源码注释）；
   15 秒级进程采样中 `orch_deploy_helper` 出现 0 次（无 fork/exec 证据）。
3. **LD_PRELOAD 如何击穿查找**：预装后进程内**唯一**映射的 `libmf_hybm_core.so` 来自私有目录
   `/tmp/opencode/gate_e_s3r/fixed_host_lib/`（S3-R maps + CPU dlopen 双证据）→ dladdr 按语义必返回
   该私有路径 → 候选 `<私有目录>/orch_deploy_helper` 与 `<私有目录>/../bin/…` 均不存在 → 回退。
   生产库原位装载时 dladdr 返回 `/usr/local/…/lib64`，helper 就在同目录 → 助手模式（历轮 0,1 即此形态）。
4. **首探真实执行、上下文被打死（行数算术）**：每 Rank 恰 1× `test -f` 发射失败 + 1× `rm -f` 发射失败。
   若首探即发射失败（上下文从未活过），DeployOrchestratorKernel 锁内复查探针会再打一条 `test -f`
   失败（共 2×）——观测仅 1×，故首探**已在设备上真实执行**（exit 1=文件缺失→同步 507018，INFO
   "kfc probe miss (expected)" 被过滤），随后上下文死亡，复查探针/清残留发射即 507018。
   **推论：卡 16384 的 KFC 通道与 AICPU 工作正常**（命令在设备 OS 真实执行过）；S3-R 报告中
   "卡 16384 AICPU 状态"与"LD_PRELOAD 干扰 CANN"两个开放问题**就此关闭**——LD_PRELOAD 未破坏
   CANN（VMM 全序列+通道装载+首探执行均正常），唯一被破坏的是 helper 发现。
5. **设计文档佐证**：`orch_deploy_helper.cpp` 头注与库内注释（2026-10-18）明示：torch_npu 进程内
   一条非零退出 KFC 命令即永久打死设备上下文，仅新进程可行——这正是 fork+exec 助手存在的原因；
   回退路径注释"纯 ACL 宿主可容忍探测缺失退出码"，但 vllm Worker 是 torch_npu 宿主 → 必死。
6. **历史一致性**：0,1 卡自 10-12 安装起设备侧已有内核（首探命中→`TryLoadKernel` 直接加载，无非零
   退出），故历轮从未暴露；10-23 dev6/7 验证"KFC auto-deploy on fresh dies"成功，应为 helper 模式
   （原位装载或 env 指定——其发射配方无记录，留档说明）。卡 16384 首次走到该阶段且为"新卡"
   （设备侧从未部署）→ 首探必然缺失 → 进程内回退必死。

**S3-R 裁决修正**：`SMOKE_BLOCKED_BY_KFC_LAUNCH_507018` 的归因由"未知（候选：卡 AICPU 状态/
LD_PRELOAD 干扰）"修正为**确定的加载机制问题（helper 发现失败→进程内回退→已知上下文击杀）**；
507899 修复回归 PASS 的结论不变。

## 2. 原装 orch_deploy_helper 身份核验（observed，无伪 helper）

| 项 | 值 |
|---|---|
| 实际路径 | `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/orch_deploy_helper` |
| SHA256 | `849f93c0b4a96877d3e839ae7ceec4b6c479a8de331213f0d7a270c1198bd310`（= G1-A/G1-D 历史记录） |
| 权限 | `-rwxr-xr-x`（X_OK ✓）；aarch64 PIE，not stripped |
| 源码 | mf-fix `tools/aicpu_kernel_deploy/orch_deploy_helper.cpp`；**mf-fix 树内构建产物与现网件逐字节一致**（构建溯源闭合） |
| 全系统副本 | 共 9 份（安装位/源码树/归档），**全部逐字节一致 849f93c0——不存在伪 helper** |
| 接口 | `orch_deploy_helper <devId> <launchJsonPath>`；`aclrtSetDevice(devId)` 用**逻辑号**（语义正确）；退出码 0=已部署/1=用法或 ACL 故障/2=部署校验失败/3=本就在位 |
| KVER11 兼容 | launch JSON 经 argv 传入（脚本已锁 v11 JSON，`dst` 按其文件名派生 `libmf_sdma_orch_v11.so`）；cust 三件套（`libmf_sdma_orch.so=db7134b8…`、`libmf_sdma_orch.json=74ad4668…`、v11 JSON=`25e16b0a…`）在 JSON 同目录在位 |
| 安全面 | 仅执行固定命令集（probe/clean/migrate/cat/verify），非通用 shell；与库内链同款 flock 互斥（`/tmp/.mf_sdma_orch_deploy.lock`） |
| 与 DeviceID 修复关系 | helper 不调用 `AclrtMemSetAccess`（无 meta GVA/池段授权），与修复版（1dc46d7d）兼容，无需修改 |

env 传播链核验：`gate_e_eager.sh` 的 unset 清单不含 `MF_SDMA_ORCH_DEPLOY_HELPER`；spawn 子进程
继承环境 → 双 Worker 均可见。`MF_SDMA_ORCH_JSON` 仍由脚本锁定 v11（G1-C v12 取证逻辑不变）。

## 3. S3-R2 精确部署方案（进程隔离 + 显式助手路径，待批准后执行）

**全部配置与 S3-R 完全一致**（Host 修复版 1dc46d7d 经 LD_PRELOAD 私有目录、V2 设备 SO/隔离
worktree/模型/KVER11/Chunk1024/Slots4/设备 2,3/90 秒不变），**仅新增两个 env**：

```bash
# 在 S3-R 发射脚本基础上新增（其余逐字不变）：
export MF_SDMA_ORCH_DEPLOY_HELPER=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/orch_deploy_helper
export MF_LOG_LEVEL=INFO   # 使 helper 模式/回退路径可见（默认 WARN 过滤 INFO）
```

机制：`FindDeployHelper` **先查 env**（X_OK 即采纳），dladdr 查找被短路——私有目录装载不再影响
helper 发现；Worker 进程零 KFC 命令，探测+迁移+复制+校验全部在 fork+exec 助手子进程（全新 ACL
上下文，非零退出无害）。预期 serve.log 将出现 `[helper] kernel missing, deploying: …` →
`[helper] deployed and verified: … size=…`（或 `kernel already present` rc=3）与库侧
`deploy helper rc: 0|3`；双 Die 共享设备文件系统，后启动一侧预期 rc=3（G1-D 同现象）。

**回滚方案**：本轮**零系统变更**（仅 env 与私有目录副本）——回滚 = 去掉两个 env + LD_PRELOAD；
生产位双位置/设备侧/工作树零触碰（发射前后哈希断言，同 S3-R 恢复核验清单）；应急恢复剧本
`/tmp/opencode/gate_e_s3r/backup_prod_20261026/restore_prod.sh` 仍在位（本轮预期不需动用）。

## 4. fail-stop 监护 v2（语境判别）

**立即停止触发**（新 bench 请求即刻停、先保全双 Rank 原始日志/maps/设备状态再清退）：
- `507014`、`507899`、`0x7150025`、`AICPU.*timeout`、`ERROR.*0x2a`
- Worker 死亡：`died unexpectedly`、`WorkerProc hit an exception`、`HCCL watchdog thread terminated`、
  `terminate called`
- 数值错误：`first fused call verification FAILED`
- **Worker 进程内 KFC 发射失败**：`kfc launch failed`（进程内链特征——S3-R2 助手模式下不应出现，
  出现即配置失败/上下文死亡）
- 助手模式失败：`deploy helper failed`、`deploy helper terminated abnormally`、`deploy helper timeout`、
  `[helper] clean residue failed`、`[helper] cust load failed`、`[helper] copy failed`、
  `[helper] verify failed after retries`、`deploy clean residue failed`
- 传输层失败：`sdma orchestrator acquire failed`、`smem_shm_create.*failed`、`hybm_create_entity failed`
- 模式回退（事后记录性触发）：`falling back to in-process KFC deploy`
- 挂起：healthy 后 serve.log 300 秒零增长；硬上限 1800 秒

**预期不触发（不得脱离上下文误判）**：
- 首次探测缺失：`kfc probe miss (expected on first run)`、`[helper] kfc sync=507018 cmd='test -f…'`
  （**助手子进程内的 507018 同步码 = 文件缺失的正常探测结果**，新进程可容忍——与 Worker 进程内
  `kfc launch failed` 的 507018 性质完全不同）
- `kernel missing, deploying`、`kernel already present`、`deploy helper rc: 3`、
  `cust migrate getfunc ret=… (tolerated, verify later)`

监护实现要点：与 S3-R 相同的 setsid 独立会话（防工具超时误杀）、5 秒轮询、触发即
kill bench → kill bench loop → 保全 → TERM serve 会话组 → 终态核验。

## 5. S3-R2 90 秒冒烟准入清单（全部满足方可发射）

**G0 类（发射前逐项断言）**
- [ ] 设备 2,3：进程表空、DDR 回基线（≈1930/1010MB）、Health 3×OK、无告警；端口 8000/8581 空闲；无残留 vllm/bench/监护进程
- [ ] 生产位不变：`libmf_hybm_core.so=88d39285…`（双位置）、`orch_deploy_helper=849f93c0…`（X_OK）、
      v11 so `db7134b8…`、v11 JSON `25e16b0a…`、cust JSON `74ad4668…`
- [ ] 修复版副本在位：私有目录 `libmf_hybm_core.so=1dc46d7d…`
- [ ] 隔离 worktree：porcelain 0；`vllm_ascend_C=07c831f0…`、`libmf310p_v2_trial_device.so=b8ca96f0…`、
      `libmf310p_device.so=8447a55a…`
- [ ] `MF_SDMA_ORCH_DEPLOY_HELPER` 指向生产 helper 绝对路径且 X_OK；`MF_LOG_LEVEL=INFO`；
      LD_PRELOAD=修复版私有副本（CPU dlopen 预检：私有副本映射、生产副本不映射）
- [ ] 监护 v2 已加载新模式表并演练触发路径

**运行中验收（缺一即本轮 FAIL/STOP）**
- [ ] serve.log 出现**助手模式证据**：`[helper] …` 行或 `deploy helper rc: 0|3`；**不出现**
      `falling back to in-process KFC deploy`、`kfc launch failed`
- [ ] 双 Rank MemFabric 初始化成功（无 507899；`smem_shm_create` 无失败行）
- [ ] 双 Rank `first fused call verified bit-exact`；`V2 MM+MemFabric` 运行时命中（M=4096）
- [ ] bench 请求完成数 > 0；无 507014/507899/Worker 死亡
- [ ] maps：双 Worker 装载修复版（私有目录）+ 隔离树 SO；生产 `libmf_hybm_core.so` 不被映射
- [ ] 结束后恢复核验全过（同 S3-R §5 清单）

**停止边界**：S3-R2 完成即停，不自动进入 S4/S5；任一触发条件出现按 §4 处置。

## 6. 本轮审计产物

- 服务器本地：`gate_e_kfc_audit_20261026/`（00–02 号证据：源码摘录/日志比对/身份核验，封存 SHA256SUMS）
- 本报告（脱敏）提交 `diag/310p3-g1d-minimal-forensics-20261021` 分支并推送
- S3-R 报告的两个开放问题（卡 AICPU 状态、LD_PRELOAD 干扰）随本报告**关闭**，根因单一且可复现
  （加载机制→helper 发现失败→已知上下文击杀），无需卡级硬件排查
