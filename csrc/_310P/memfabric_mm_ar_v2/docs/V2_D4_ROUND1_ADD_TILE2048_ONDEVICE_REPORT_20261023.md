# V2 D4 第一轮：Add Tile 1024→2048 正式 Kernel 实机验证报告（2026-10-23）

> 状态：**D4-0/1/2/3 实机执行完成；D4-4 真实模型回归被外部租户占用阻塞（6 小时看护窗口 0/4 会话，详见 §5），最终建议：单算子层面有条件 GO，真实模型判据 PENDING。**
> 分支：`perf/310p-v2-d3-add-tile2048-20261023`；候选源码 `5adbf1810`（= D3 `25fb1951d` + 本轮修复）；
> 对照基线 `edebac40d`（D1-A 末态，Tile1024）。
> 单变量声明：基线与候选之间**仅** `src/op_kernel/mf310p_v2_kernel.asc` 的 Vector Add 批量
> 1024→2048 FP16 与 UB 预算差异（外加本轮修复 D3 自身编译 BUG 的等价断言迁移）；
> 诊断 Kernel、`mf310p_v2_kernel_common.h`、Host/Serving ABI、MemFabric 协议、IB/SDMA/DONE、
> 缓存可见性全部零改动。插件 SO 全程使用 Gate C 构建 `8eb65de9…`，仅按会话切换设备 SO。

## 0. 执行摘要

| 必答问题 | 结论 |
|---|---|
| **Tile2048 能否通过 BiSheng 编译与 UB 资源门？** | **能，但 D3 源码存在一个编译破坏 BUG（已修复，`5adbf1810`）**：命名空间作用域的 `static_assert(TOTAL_UB_SIZE…)` 在 CANN 9.1 头文件体系下不可见（该常量仅在 `__aicore__` 设备上下文可见；基线只在核函数体内使用故不受影响）。修复=把同一断言迁移到核函数体内其守护的 `InitBuffer` 旁，语义零变化。Matmul `SetLocalWorkspace` 充分性以**编译期证明**关闭：需求 `transLength×2 = 131072B`（最坏 tiling1024，`enableUBReuse=true` 系数）≪ 候选可用 `249824B`（裕量 118752B）。 |
| **正确性是否保持位精确？** | **是。** Trial 直调 8 形状×双 Rank×(2 预热+4 采样+新地址首写) 全部位精确、status=0；Serving 混合路由 60 调用×双 Rank（36 次 V2，含 2049/3073/4095 tail、输出槽复用、stock 交错）全部位精确、status=0。 |
| **单算子收益是否达到 ≥5% 首轮门槛？** | **达到且超出。** ABBA 8 会话×30 样本：配对 TP2 event p50 基线 2726.9us → 候选 2556.3us，**+6.26%（-171us）**；四组同期 ABBA 对照 +5.50%~+7.16% 全部 ≥5%；跨会话离散 ≤1.2%（≪3.3% 噪声包络）；≥6ms 停顿 0；基线与 D1-A 锚定吻合。 |
| **真实模型（Qwen3.6 TP2 Eager）表现？** | **执行中**（环境延迟：服务器异常重启 + 外部租户占用 die0/1，见 §7；单算子收益已达标准入条件）。 |
| **首轮建议** | **有条件 GO**（单算子层面）：正确性零回归 + 可复现 ≥5% 收益 + 编译/资源门全过。真实模型数据落地后最终确认；未达标或 E2E 无窗口则维持"单算子 GO、E2E PENDING"。 |

## 1. D4-0 环境准入

- **Host SO** 双路径 `88d39285…`（Gate A/B/C/D1-A 身份）；**KVER11 JSON** `25e16b0a…`；**插件 SO** `8eb65de9…`（Gate C 诊断 ON 构建，全程未重编）；备份于 `/tmp/opencode/d4/c0/backup_20261022T211729/`。
- **设备侧 v11**：双 die `kernel already present`（探针 `849f93c0…` 与 Gate C/D1-A 同一二进制）。
- **die0/1 健康探测**：单 die 与双 die 组合的分配/计算/同步全部 PASS；dmesg 全量扫描 dev0/1 零错误。
- **环境偏差（重要，全程记录）**：会话开始时宿主 dev2/dev3 devmanager 通道被外部租户（llama-server 崩溃）wedge，npu-smi 任何形式挂起。处置：
  1. 替代空闲门 = `/dev/davinci0|1` fd 归属扫描（D/Z 态退出清理视为无害残留）+ dmesg dev0/1 增量 = 0 + 双 die 功能探针；
  2. 所有设备进程一律 shell 层导出 `ASCEND_RT_VISIBLE_DEVICES=0,1`（未导出的进程会在运行库加载期枚举全部 8 卡而挂死）；
  3. 会话全部 setsid 后台 + 日志轮询（工具超时组杀曾误杀一次会话，见 §7）。
- **准入结论**：PASS with deviation（目标 die 独占且健康；污染在他人设备上）。

## 2. D4-1 编译与资源审查

### 2.1 构建矩阵（BiSheng dav-2002，逐字复刻 cmake 命令）

| 变体 | SHA256 | 字节 | 结果 |
|---|---|---|---|
| 基线 diag-ON（edebac40d worktree） | `f8047cbe…` | 2185568 | PASS |
| 基线 diag-OFF | `629aa220…` | 657472 | PASS |
| 候选 diag-ON（5adbf1810） | `2a19390c…` | 2185952 | PASS |
| 候选 diag-OFF | `805b1a47…` | 657856 | PASS |

### 2.2 BUG 修复（本轮唯一源码改动，`5adbf1810`）

- 现象：候选 `mf310p_v2_kernel.asc` 命名空间作用域 `static_assert(TOTAL_UB_SIZE > …)` 编译失败 `use of undeclared identifier`；同一命令下基线 rc=0。
- 根因：CANN 9.1 `kernel_utils_constants.h` 中 `TOTAL_UB_SIZE`（=256KB，`__NPU_ARCH__==2002` 分支）仅在 `__aicore__` 设备编译上下文可见；三个最小探针实证（kernel_operator.h 单独引入可用；acl_rt.h 先引/经 common.h 链不可用）。
- 修复：断言原文迁移至 `V2TrialKernel` 体内、紧邻被守护的 `pipe.InitBuffer(mmUb, TOTAL_UB_SIZE - kProductionUBForVector - kUBForIB)`；断言语义与 CPU 合同（`test_v2_d3_production_add_tile_source.py` 仍 31/31 PASS）零变化。诊断 Kernel/common.h 未动。

### 2.3 Matmul 临时 UB 充分性（编译期证明，关闭 D2/D3 遗留 OPEN 项）

依据 CANN 9.1 `MatmulLocalWorkspace::SetWorkspace` 的运行时 ASSERT 合同
（`localBuffer.GetSize() >= max(transLength × ENABLE_TRANS_COEFF(=2), biasLen)`，
`enableUBReuse=true` 时系数 2），以包含**真实 common.h tiling 实例**的探针在 BiSheng 编译期断言：

| 项 | 值 |
|---|---|
| transLength（CHUNK 256/512/1024） | 16384 / 32768 / 65536 B |
| 需求（最坏 tiling1024 ×2；biasLen=1024B 不影响 max） | **131072 B** |
| 候选可用（262144 − 12288 − 32） | **249824 B**（裕量 118752B） |
| 基线可用（262144 − 6144 − 32） | 255968 B（裕量 124896B） |
| cfg 前提 | enableUBReuse=true、enVecND2NZ=true、Norm 模板（均已编译期断言） |

管道总分配仍恰好占满 256KB UB（12288+32+249824），与基线同性质。**结论：SetLocalWorkspace 充分性 PASS。**

## 3. D4-2 正确性

### 3.1 Trial 直调矩阵（`test/perf/d4_trial_correctness.py`，候选 `2a19390c`）

| 形状（M×CHUNK） | 覆盖 | 结果 |
|---|---|---|
| 4096×1024 | D1-A 基线形状、4 Chunk overlap | 双 Rank 位精确、status=0 |
| 256×256 / 512×512 / 1024×1024 | 三个 tiling 实例各自最小形状 | 同上 |
| 2048×512 / 2048×1024 | 多 Chunk + overlap | 同上 |
| 4096×256 / 4096×512 | 16/8 Chunk 深重叠 | 同上 |

每形状：独立 handle、ib_probe 先行、2 次预热（含新符号首写保护+新输出地址设备级同步）、4 次采样（双输出缓冲交替=输出槽复用）、1 次全新地址首写；全部逐样本 `torch.equal` + status=0。**8/8 形状 × 双 Rank 全 PASS。**

### 3.2 Serving 混合路由（`test/perf/d4_serving_mixed_route_gloo.py`，候选）

60 调用 × 双 Rank（3 reps × 计划表）：V2 36 次（含 M=2048/2049/3000/3073/4095/4096 非整 Chunk tail 经 input stage 补齐、仅返回真实行），stock 回退 24 次（M=1/10/2047/4097/5000），建池后 stock/v2 交错、逐次 del+gc 输出槽复用、逐次 status=0。**全 PASS。**
（偏差：本会话用 gloo 生成参考与 stock 路径——TP2 两操作数 FP32 加法与顺序无关，位精确等价；"HCCL-after-pool 共存"未在本轮复测，但其不受单变量内核改动影响，前几轮已覆盖。）

## 4. D4-3 受控性能 A/B（ABBA，external 无 Profiler）

口径：与 D1-A 完全一致的 NPU Event 外部计时（`event_us`=双 Event 夹 Kernel 的设备流区间，主验收）；wall_us 辅助。
配对：每个双 Rank 请求取 `max(rank0, rank1)`。计划：`base,cand,cand,base` × 2 = 8 会话，每会话 3 预热 + 30 有效样本（双 Rank 共 240 样本/条件）。设备 SO 按会话切换并 SHA 校验（fail-closed 批调度）。

| 会话 | r0 ev p50 | r1 ev p50 | 配对 ev p50 | 配对 ev p90 | 停顿 |
|---|---:|---:|---:|---:|---:|
| base_r1 | 2672 | 2739 | 2743 | 2819 | 0 |
| base_r2 | 2684 | 2683 | 2714 | 2802 | 0 |
| base_r3 | 2667 | 2724 | 2727 | 2791 | 0 |
| base_r4 | 2684 | 2709 | 2716 | 2790 | 0 |
| cand_r1 | 2516 | 2509 | 2547 | 2591 | 0 |
| cand_r2 | 2547 | 2508 | 2553 | 2612 | 0 |
| cand_r3 | 2565 | 2548 | 2577 | 2624 | 0 |
| cand_r4 | 2553 | 2533 | 2556 | 2641 | 0 |

- **聚合（主判据）**：基线配对 event p50 **2726.9us** → 候选 **2556.3us**，**Δ=170.6us（+6.26%）**；wall p50 2816→2647（-169us）一致。
- **同期 ABBA 对照**：r1 +7.16%、r2 +5.93%、r3 +5.50%、r4 +5.89%——**四组全部 ≥5%**，无一低于门槛。
- **可复现性**：基线组内会话 p50 离散 1.1%，候选 1.2%（D1-A 噪声包络 3.3%）；基线 r0 p50 2667–2684 落于 D1-A 锚定区间 2656–2731，r1 2683–2739 与 2708–2796 基本重合（1 个会话低于下界 25us，属噪声）。
- **异常样本**：≥6ms 配对停顿 **0**；无 507014、无协议非零、无位精确失败。
- （偏差：进程组/参考 all-reduce 由 HCCL 改 gloo，计时窗口内不含任何集合通信，口径不受影响。）

## 5. D4-4 真实模型回归（Qwen3.6-35B-A3B-w8a8 TP2 Eager）

**状态：BLOCKED（外部环境，非候选问题）。** 单算子正确性+≥5% 收益的入场条件已满足；
执行准备全部就绪（fused-eager、input=4096/output=512、20 prompts、并发 10、
ABBA（base,cand,cand,base）、仅设备 SO 单变量、每会话空闲门+SHA 校验+失败即停重试）。

阻塞事实链（证据 `/tmp/opencode/d4/e2e_status.txt`）：
1. 服务器重启清除设备侧 v11 → 看护器在首个空闲窗口用**真实 helper 完成重部署并验证**（02:10）；
2. 重启后外部租户（另一容器，进程不可见）在 die0/1 周期性占用：26–34GB 内存突发、约 4–5 分钟周期，
   3 分钟持续空闲校验多次通过但 bench 启动即碰撞（02:10、08:01、08:05、08:10、08:14 五次
   preflight 拒绝，均未污染任何数据）；
3. 6 小时看护窗口（02:14–08:14）耗尽，0/4 会话；
4. 窗口后 die0 仍被常驻 14.5GB 外部分配占用（14:02 实测，无计算负载）——模型加载所需的
   独占显存物理上不可得。

**未运行任何 Qwen3.6 会话，故对真实模型 Prefill/TTFT/吞吐不作任何宣称。**
复跑入口就绪：设备释放后执行 `bash /tmp/opencode/d4/d4_e2e_runner.sh`（自动等待窗口、
v11 复验、ABBA 四会话、失败即停）。

## 6. 版本与证据

| 项 | 值 |
|---|---|
| 候选提交 | `5adbf1810`（`perf/310p-v2-d3-add-tile2048-20261023`） |
| 基线提交 | `edebac40d226a8aeb08460b5a33a3283a842bba5` |
| 插件 SO（全程不变） | `8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1` |
| 设备 SO 基线/候选 | `f8047cbeb48ca25dc5f3369bf702d494f5c52e25772f5224d24c952a6519a658` / `2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19` |
| Host SO / KVER11 JSON | `88d39285…` / `25e16b0a…`（双路径/重启后复验一致） |
| 入库脱敏证据 | `evidence/d4_20261023/`（环境/构建记录、正确性矩阵、ABBA 16 份 rank JSON + 分析） |
| 驱动脚本（入库） | `test/perf/d4_trial_correctness.py`、`d4_serving_mixed_route_gloo.py`、`d4_abba_perf.py`、`d4_abba_analyze.py` |
| 服务器侧原始数据 | `/tmp/opencode/d4/`（含全部失败/隔离会话、dmesg 快照、探针、构建日志） |

## 7. 环境事件与处置时间线（完整留痕）

1. **dev2/3 wedge**（外部租户 llama-server 崩溃，D4 开始前）：npu-smi 挂起、未限可见设备的进程加载期挂死、每个 ACL 进程退出残留 D 态线程（5–18 分钟自清）。处置见 §1。
2. **工具超时组杀误杀一次会话**（21:35）：setsid 修复启动方式；被杀 rank0/rank1 18 分钟内自清，证据保全于 `_killed_*`。
3. **HCCL 16666 设备侧泄露**（被杀会话遗产，约 40 分钟自愈）：期间正确性/ABBA 会话改 gloo（位精确等价论证见 §3/§4），HCCL 恢复后探针 PASS。
4. **MemFabric 库 deploy helper 180s 超时**（wedge 环境下 helper 退出挂起 >180s + 库内阻塞 waitpid）：以 `MF_SDMA_ORCH_DEPLOY_HELPER` 指向即时返回 rc=3 的包装器（kernel 在位已三次实测；协议/库零改动；缺失时数据面显式报错 fail-loud）。会话 store 端口轮换规避 fd 继承泄露。**服务器重启、驱动恢复后已弃用该包装器，改回真实 helper 并完成 v11 重部署。**
5. **服务器异常重启**（约 01:4x，用户侧中断）：全部 D4-0~D4-3 证据/构建/提交完好；文件哈希复验一致；设备侧 v11 kernel 被清除，已由真实 helper 重新部署并验证。
6. **重启后外部租户间歇占用 die0/1**（26GB/die，波动重启）：E2E 看护器等待持续空闲窗口（≥3 分钟 AICore=0 且 mem<4GB）后按会话重试执行；一次启动碰撞已 fail-closed 退出未污染。

## 8. 结论与建议

1. **单算子层面：有条件 GO。** Add Tile 1024→2048 在正式 V2 Kernel 上以零正确性回归、零停顿、四组 ABBA 全部 ≥5.5% 的可复现方式获得 **配对 p50 +6.26%**，超过首轮 ≥5% 目标；BiSheng 编译与 Matmul UB 资源门全部关闭。
2. **真实模型待定。** E2E 数据落地前不得宣称 Qwen3.6 TTFT/吞吐改善；单层 +6.26% 亦不足以外推整体打平 Stock（参照 D2 决策约束）。
3. **遗留 OPEN**（与本轮单变量无因果，维持原状）：Gate E 507014/P0；23–45ms 配对停顿机制；formal 2.6↔3.4ms 双态触发器。
4. **不合并**：本分支按纪律不合入 `feat/310p-memfabric-public-api`；E2E 判据 PENDING，
   在其落地并确认真实模型收益前不接受合入（单算子 GO 不足以放行生产）。

> 附：本轮所有环境偏差均未触碰 MemFabric 协议、IB/SDMA/DONE、缓存可见性、KVER11 协议与正式优化点边界。
