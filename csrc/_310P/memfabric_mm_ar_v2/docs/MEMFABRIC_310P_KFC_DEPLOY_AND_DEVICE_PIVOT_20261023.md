# MemFabric 310P 多卡部署与 KFC 自动部署链实机总结（2026-10-23）

> 场景：5664 卡（die0/1）被外部租户长期占用，D4-4 真实模型回归被迫迁移到同宿主其他
> 310P3 双 die 卡（实测 dev6/7 = 43520 卡）。本文记录：设备号全链路排查结论、
> CANN VMM API 命名空间分裂的实证、最小补丁、KFC 自动部署链在非 0/1 设备上的首次
> 完整实证，以及全部踩坑与恢复清单。
> 状态：**dev6/7 上 trial 正确性矩阵双 Rank 全 PASS（基线核 f8047cbe，8/8 形状位精确）**。

## 1. 设备号从哪里来：没有写死，也没有现成设置入口

全链路唯一的设备号入口是**调用进程的 `aclrtGetDevice()` 返回值（逻辑号）**：

```
vllm-ascend memfabric310p_adapter.cpp:312  aclrtGetDevice(&device_id)   ← 逻辑号
  → smem_shm_init(store, world, rank, device_id, cfg)                   ← config 无 deviceId 字段
    → hybm_init(deviceId) → initedDeviceId
      → SdmaOrchestrator::Acquire(devId) → dieId_ = devId % 2            ← 奇偶推导，非绝对号
      → RunDeployHelper → orch_deploy_helper <devId> <json>（fork+exec 继承环境）
```

逐点核验（均无跨设备硬编码）：

| 检查点 | 结论 |
|---|---|
| `SdmaOrchestrator::InitOnce` | `dieId_ = devId % 2`——只要可见列表保持"偶数在前"（如 `2,3`、`6,7`），逻辑奇偶 = 物理 die 奇偶 |
| GVA 约定地址（`HYBM_DEVICE_META_ADDR` 等，`hybm_define.h`） | 每 die 自身 VA 空间的常量（SVM 区顶部），无跨设备假设 |
| `CheckSdmaReaches` | 仅查对端 slot 是否已建 GVA 映射，无 die/卡假设 |
| 对端发现 | 经 config store 交换 GVA/句柄，无硬编码 |
| AICPU 编排核 `hybm_sdma_orch_kernel.cc:151` | **唯一"写死"点**：`devId == 0 ? 0x7F45 : 0xBF45`（按 die 选 SMMU 流 ID）——奇偶语义，非 0/1 绝对绑定 |
| 环境变量（`mf_env_define.h` 全量盘点） | 无设备选择变量；`env::ASCEND_RT_VISIBLE_DEVICES` 定义了但**从未被消费（死代码）** |

结论：**换卡 = 换 `ASCEND_RT_VISIBLE_DEVICES`，无任何配置可绕过重映射语义**。

## 2. 真正的墙：CANN VMM API 命名空间分裂（探针实证）

在 `ASCEND_RT_VISIBLE_DEVICES=2,3`（当前设备=逻辑0=物理2）下的 2×2 探针矩阵
（`/tmp/opencode/d4/vmm_probe.c`，保留于服务器）：

| API | location.id=0（逻辑） | location.id=2（物理） |
|---|---|---|
| `aclrtMallocPhysical` | **OK** | 失败 107001 |
| `aclrtMemSetAccess` | 失败 **507899**（DRV_INTERNAL_ERROR） | **OK** |

即：**MallocPhysical 收逻辑号、SetAccess 收物理号**。库把同一个逻辑 devId 同时喂给
两者，identity 映射（不设 VISIBLE 或 `=0,1`）下二者恒等从未暴露；非零起点可见列表
（`2,3`/`6,7`）在 meta 区授权处必失败（`hybm_gva.cpp:82`，实测日志证据在
`_failed` 会话目录）。

LD_PRELOAD 拦截不可行：库经 `dlopen(libascend_cl)+dlsym` 直取符号
（`dl_acl_api.cpp:96/161`）。

## 3. 最小补丁（mf-fix 工作树 @ 131ee5ff）

装机 `libmf_hybm_core.so`（`88d39285`）的等价源码即 mf-fix 工作树
（`fix-310p-kfc-deploy-helper` 分支，版本溯源见
`310P3_版本溯源与资产保全报告_20261020.md`）。补丁内容：

1. `src/hybm/csrc/common/hybm_define.h`：新增 `HybmPhysicalDeviceIdOf(logicDevId)`
   ——按 CANN 运行时同款语义解析 `ASCEND_RT_VISIBLE_DEVICES`（逗号列表第 N 项 =
   逻辑 N 的物理号；未设置/解析失败/列表过短一律退回 identity）；
2. 3 个 `AclrtMemSetAccess` 调用点的 `acc.location.id` 改经该函数换算：
   `hybm_gva.cpp:78`（meta 区）、`hybm_acl_vmm_segment.cpp:139`（own 段）、
   `:467`（imported 对端段）；
3. **不动**：`MallocPhysical` 等其余 VMM API（实测要求逻辑号）、
   `hybm_vmm_based_segment.cpp:317`（HAL 层 DRAM 路径，310P 不可用）、
   协议/IB/SDMA/DONE/缓存可见性/KVER11 全部零改动。

构建与部署（增量，与装机版同构建树）：

```
cd /tmp/opencode/mf-fix/build && ninja libmf_hybm_core.so
# 构建树原产物 sha256 = 88d39285…（与装机逐字节同源，证明可复现）
# 新产物 1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30
cp → /usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so
cp → /usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so
# 原件备份：/tmp/opencode/d4/c0/backup_hostso_20261023T153335/libmf_hybm_core.so.orig
```

**验证与恢复**：打补丁后 dev0/1 身份场景不受影响（identity 映射下换算函数原样返回）；
测毕须用备份原件恢复两路径并复核 `88d39285`。

## 4. KFC 自动部署链（首次在非 0/1 设备完整实证）

链路（`hybm_sdma_orchestrator.cpp` + `orch_deploy_helper.cpp`）：

```
LoadOrchestratorKernel → RunDeployHelper（fork+exec 助手，本进程零 KFC 命令）
  → 助手 aclrtSetDevice(devId)（继承 ASCEND_RT_VISIBLE_DEVICES，逻辑号）
  → KFC test -f /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libmf_sdma_orch_v11.so
      └ 缺失 → 退出码 1 → KFC 同步回传 507018（预期，新进程可容忍）
  → MigrateViaCustChannel（aclrtBinaryLoadFromFile CPU 模式 + GetFunction 触发设备写）
  → CopyToSearchPath（cat 分段拷贝）→ verify（grep HybmSdmaOrchEpoch + 尺寸精确匹配）
  → rc=0（部署）/ rc=3（本就在）→ 库内 TryLoadKernel 正常加载
```

关键事实：

- 设备侧路径是**每 die 自身文件系统内**的常量路径（`aicpu_kernels/0/` 非 per-device
  编号），换卡零改动；
- dev6 从零部署成功（日志，`d4_2_trial_dev67/session.log`）：
  `kfc sync=507018`（缺失探测）→ `kernel missing, deploying` →
  `deployed and verified ... size=13960`；dev7 同轮 `already present`；
- **部署是 per-die 的**：换一对卡就要在两个 die 上各走一遍（自动完成，每 die 秒级）。

## 5. 踩坑实录（按时间序）

1. **快捷部署包装器残留 → AICPU 0x2a**：wedge 时期为绕过 helper 退出挂死设过
   `MF_SDMA_ORCH_DEPLOY_HELPER=<即时 rc=3 脚本>`；该包装器"谎报 kernel 在位"仅在
   dev0/1（三重验证过）合法。残留在 dev6/7 上运行 → 库信以为真 → 首次 launch
   `HybmSdmaOrchEpoch` 找不到二进制 → AICPU errorCode=0x2a。
   **教训：任何"假定在位"的快捷路径必须随环境失效而立即移除**；真实 helper 在
   驱动健康时本就秒级完成 probe+部署。
2. **VMM 命名空间分裂**（§2）：visible 重映射下 507899 必现，需 §3 补丁。
3. **宿主租户动态游走**：die0/1（14.5GB 常驻+90% 计算）→ dev2/3（一度全空后被占
   86%）→ dev4/5、dev6/7 空闲。选卡应在每次会话前动态判定（同总线两 die 同卡）。
4. **HCCL 端口 per-卡独立**：各 die 有独立 ROCE IP，16666 冲突只在同卡重复会话间
   发生；换卡后 HCCL 直接可用。

## 6. dev6/7 验证结果

`ASCEND_RT_VISIBLE_DEVICES=6,7`、候选 Host SO（`1dc46d7d`）+ 基线设备核
（`f8047cbe`，当时安装态）：

- `mf310p_v2_trial_create` 建池（含 meta GVA/段映射/协同初始化）全部成功——补丁生效；
- KFC 自动部署 dev6 从零完成、dev7 探测在位；
- trial 正确性矩阵 8 形状（256~4096 × CHUNK 256/512/1024，含 16-chunk 深重叠、
  输出槽轮换、新地址首写）**双 Rank 全部位精确、status=0**
  （`/tmp/opencode/d4/sessions/d4_2_trial_dev67/d4_correctness_rank{0,1}.json`）。

注意：此次验证跑的是**基线核**；候选核（`2a19390c`）在目标卡上另行补验证后再进入
E2E A/B（两臂同卡同 Host SO，内部对比有效）。

## 7. 恢复清单（D4 收尾时执行）

1. `cp /tmp/opencode/d4/c0/backup_hostso_20261023T153335/libmf_hybm_core.so.orig`
   → /usr/local 与 site-packages 两路径，复核 `88d39285…`；
2. mf-fix 工作树补丁已在 memfabric_hybrid 仓提交留档（分支
   `fix/310p-setaccess-physical-devid-20261023`）；
3. vllm_ascend/ 设备 SO 恢复为基线安装态（D4-3 结束时为 `f8047cbe`）或按最终结论
   保持候选；
4. 设备侧各 die 的 `libmf_sdma_orch_v11.so`（dev0/1/6/7）保留——KVER11 正式部署物，
   与原运行态一致（dev6/7 为本轮合法新增）。

## 8. AICPU 编排核运行模式：epoch 有限纪（默认）与常驻哨兵（诊断）

编排核 `HybmSdmaOrchEpoch` 有两种模式，由 `MF_SDMA_ORCH_EPOCH_BUDGET_S` 控制
（`hybm_sdma_orchestrator.cpp` `ResolveEpochBudgetS()`）：

| 模式 | 取值 | 行为 | 适用 |
|---|---|---|---|
| **epoch 有限纪（v11 默认）** | 未设/0/非法 → 15s | 内核按 host 心跳时钟（100ms/跳）到 15s 自退，监督线程轮询结果区（~100ms）后重拉；REQ 环/游标/到达戳环持久于段内保留区，SQ 生产游标每纪活查询（`cp1LiveQuery`：SQ_TAIL 等 4 属性） | **生产/serve 唯一模式** |
| **常驻哨兵** | `4294967295` | 内核不判时限跑到停止标志 | 仅诊断；**serve 下必死** |

**为什么 serve 必须用 epoch 模式**：torch_npu 分配器内部会触发
`AclrtSynchronizeDevice`（全设备、全流、无豁免），常驻 AICPU 任务必阻塞它
（serve 实测死因 507017）。epoch 模式把每次同步的最坏等待压到 ≤1.5×预算。
该修复为 v11（mf 仓 `2f59dedb`，2026-10-12）；**"常驻"是 v11 之前的旧模式**，
选分支时勿回退。装机 `libmf_sdma_orch_v11.so`（KVER11）与 host 默认参数一致，
无需手工设置该变量。

**运维注意**：epoch 换纪有 ~100–300ms 间隙（请求环排队，零丢失设计）；日志默认
WARN 级，换纪正常静默、错误（`epoch exited with error`/`overran`/`relaunch failed`）
才会打到进程 stdout——serve 日志中零此类记录即换纪全程健康。G3 §8 的间歇性 P0
（prefill+decode 交叠工况双 rank aicore timeout）与本模式的关系已被本轮
budget=5s（换纪频率×3）4/4 干净的对照实验削弱——非换纪频率主导，根因仍在
负载相位时序（详见 D4 报告 §5.2）。

## 9. 结论

1. MemFabric 310P 栈**没有写死 die 0/1**：设备号全链来自 `aclrtGetDevice()`，
   die 语义按奇偶推导；换卡只需 `ASCEND_RT_VISIBLE_DEVICES` + §3 补丁。
2. KFC 部署链**设备无关且已闭环**：任何 die 从零到可用秒级完成，无需手工干预。
3. 本补丁是环境适配（CANN VMM 语义分裂修正），非协议变更；ABBA 双臂同 SO，
   性能对比内部有效；对 identity 场景零影响（换算函数原样返回）。
