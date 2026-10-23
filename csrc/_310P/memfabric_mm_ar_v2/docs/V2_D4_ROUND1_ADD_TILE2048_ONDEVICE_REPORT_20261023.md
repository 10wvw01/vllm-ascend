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
| **真实模型（Qwen3.6 TP2 Eager）表现？** | **已完成（die0/1 被外部租户阻塞，经 Host SO 物理设备号补丁迁移至 dev6/7 执行，§5）**：性能中性偏正——TPOT p50 **-1.98%**、输出吞吐 **+1.86%**、TTFT 噪声级（干净会话 base n=2 / cand n=3）；幅度与单算子 +6.26% 的整模型稀释算术自洽。稳定性：默认 epoch 预算下 7 会话 2 次挂死（**两臂各一**），经双 die AICore 100%×60s 监视画像、dev4/5 新卡对对照（2/2 干净）、budget=5s 换纪频率放大（4/4 干净）与分支溯源（装机线含 v11 epoch 修复，未错装），定性为 **G3 §8 已知间歇性 P0 的同型复发，与 Tile2048 无关**（§5.2）。 |
| **首轮建议** | **维持单算子 GO；E2E 性能 PASS；稳定性维持既有 P0 风险披露**。Tile2048 本身零回归、收益可复现；V2 栈整体生产启用仍被 G3 §8 P0 阻断（先于本轮存在，与单变量无关）。 |

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

**状态：COMPLETED（dev6/7 卡对执行，经环境适配迁移）。** die0/1 因外部租户常驻
14.5GB 物理不可得（原 BLOCKED 记录见 §7-6）；经设备号全链路排查确认 MemFabric 无
die 0/1 写死后，以 CANN VMM 命名空间分裂修复（`aclrtMemSetAccess` 需物理设备号，
仅 setup 期 3 个调用点；mf 仓分支 `fix/310p-setaccess-physical-devid-20261023`，
Host SO `88d39285→1dc46d7d`，**测毕已还原并复核 `88d39285`**）迁移至 dev6/7。
KFC 自动部署在 dev6 从零完成；基线与候选设备核在 dev6/7 各补跑 trial 正确性矩阵
（8/8×双 Rank 位精确）后进入 E2E。迁移经验独立成文：
`MEMFABRIC_310P_KFC_DEPLOY_AND_DEVICE_PIVOT_20261023.md`。

配置：fused eager、input=4096/output=512、20 prompts、并发 10、ABBA+复跑、
仅设备 SO 单变量、每会话空闲门+SHA 校验、bench 结果 completed/failed 校验。

### 5.1 性能（干净会话：base n=2，cand n=3；dev6/7，默认 epoch 预算）

| 指标 | base 均值 | cand 均值 | Δ |
|---|---|---|---|
| TTFT p50 | 12.110s | 12.084s | -0.21%（噪声级） |
| TTFT p95 | 27.839s | 27.866s | +0.10%（噪声级） |
| TPOT p50 | 0.241s | 0.236s | **-1.98%** |
| 输出吞吐 | 37.96 tok/s | 38.67 tok/s | **+1.86%** |

判读：方向性正向，幅度与稀释算术一致——V2 仅覆盖 M∈[2048,4096] 的
o_proj/out_proj 融合 AR（decode M=10 走原生路径），单算子 -171us 在整模型
步长时间中占比小。样本 2v3 属小样本，作方向性结论而非精确幅度宣称。

### 5.2 稳定性：G3 §8 已知 P0 的同型复发（与 Tile2048 无关）

默认 epoch 预算（15s）下 dev6/7 共 **7 会话、2 次挂死（cand_r2、base_r3，两臂各一）**：

- 签名：双 rank 同时 `aicore execution times out`（60s 看门狗，`retCode=0x25`），
  20/20 请求全失败，会话不可恢复；
- **侧车监视画像（本轮新证据，G3 时期不具备）**：挂死期间双 die AICore **双双钉死
  100% 约 60 秒**直至看门狗击毙——双侧自旋互等（跨 die 数据等待死锁）实锤；
- 挂死相位：prefill+decode 持续交叠段（bench 开始后 ~85s，吞吐 409→819→0 衰减链），
  与 G3 §8 Gate E 的失败条件分析完全一致；
- **归因排除矩阵**：
  - 非卡特异：同型挂死 G3 期间已发生于 die0/1（2026-10-19，772/775 失败、间歇性、
    重跑干净）；本轮 dev4/5 新卡对 2/2 干净；
  - 非候选特异：基线臂同样挂（base_r3）；G3 挂死时 Tile2048 尚不存在；
  - 非 epoch 换纪频率：`MF_SDMA_ORCH_EPOCH_BUDGET_S=5`（换纪频率×3）4/4 全干净
    （环境变量传入 vllm 进程已用 /proc/environ 实证）——换纪边界竞态假说被削弱，
    指向负载相位时序竞态；
  - 非分支错装：装机线 = `fix-310p-kfc-deploy-helper`（含 v11 epoch 修复
    `2f59dedb`，2026-10-12，serve 全设备同步死因修复）；"常驻"为 v11 前模式
    （serve 必死 507017，现仅为诊断哨兵）；AICPU 二进制 `libmf_sdma_orch_v11.so`
    与 host `epochBudgetS=15` 一致，六份服务器日志零换纪错误/超限记录；
- 结论：**G3 §8 已知"低概率高后果"P0 的同型复发**，根因未定（候选方向：到达戳环
  覆写路径、换纪间隙 SQ 积压交互——需 mf 仓侧 plog 级深挖，沿 G3 §12 排查路线）。
  **V2 栈整体生产可用性维持 G3 判定：不具备，直至该 P0 根因修复。**
- 数据有效性：挂死会话无有效性能指标，不污染 A/B；干净会话两臂同环境同暴露。

### 5.3 D4-4 判定

- 性能：真实模型 E2E **PASS**（中性偏正，与单算子收益自洽）；
- 稳定性：**既有 P0 维持**（非本轮引入、非本轮恶化，证据链完整）；
- E2E 判据 PENDING → 关闭。

## 6. 版本与证据

| 项 | 值 |
|---|---|
| 候选提交 | `5adbf1810`（`perf/310p-v2-d3-add-tile2048-20261023`） |
| 基线提交 | `edebac40d226a8aeb08460b5a33a3283a842bba5` |
| 插件 SO（全程不变） | `8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1` |
| 设备 SO 基线/候选 | `f8047cbeb48ca25dc5f3369bf702d494f5c52e25772f5224d24c952a6519a658` / `2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19` |
| Host SO / KVER11 JSON | `88d39285…` / `25e16b0a…`（D4-4 期间临时替换为补丁版 `1dc46d7d…` 用于 dev6/7 迁移，**测毕已还原 `88d39285` 并复核**；补丁留档 mf 仓分支 `fix/310p-setaccess-physical-devid-20261023` @ `5802f550`） |
| 入库脱敏证据 | `evidence/d4_20261023/`（环境/构建记录、正确性矩阵、ABBA 16 份 rank JSON + 分析、**E2E 13 会话结果 JSON+监视日志+挂死服务器日志摘录+双实验对照**、`d4_e2e_final_analysis.json`） |
| 驱动脚本（入库） | `test/perf/d4_trial_correctness.py`、`d4_serving_mixed_route_gloo.py`、`d4_abba_perf.py`、`d4_abba_analyze.py`；E2E 运行器 `evidence/d4_20261023/d4_e2e_pair_one.sh`（bench 主脚本 `tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh`） |
| 服务器侧原始数据 | `/tmp/opencode/d4/`（含全部失败/隔离会话、dmesg 快照、探针、构建日志） |

## 7. 环境事件与处置时间线（完整留痕）

1. **dev2/3 wedge**（外部租户 llama-server 崩溃，D4 开始前）：npu-smi 挂起、未限可见设备的进程加载期挂死、每个 ACL 进程退出残留 D 态线程（5–18 分钟自清）。处置见 §1。
2. **工具超时组杀误杀一次会话**（21:35）：setsid 修复启动方式；被杀 rank0/rank1 18 分钟内自清，证据保全于 `_killed_*`。
3. **HCCL 16666 设备侧泄露**（被杀会话遗产，约 40 分钟自愈）：期间正确性/ABBA 会话改 gloo（位精确等价论证见 §3/§4），HCCL 恢复后探针 PASS。
4. **MemFabric 库 deploy helper 180s 超时**（wedge 环境下 helper 退出挂起 >180s + 库内阻塞 waitpid）：以 `MF_SDMA_ORCH_DEPLOY_HELPER` 指向即时返回 rc=3 的包装器（kernel 在位已三次实测；协议/库零改动；缺失时数据面显式报错 fail-loud）。会话 store 端口轮换规避 fd 继承泄露。**服务器重启、驱动恢复后已弃用该包装器，改回真实 helper 并完成 v11 重部署。**
5. **服务器异常重启**（约 01:4x，用户侧中断）：全部 D4-0~D4-3 证据/构建/提交完好；文件哈希复验一致；设备侧 v11 kernel 被清除，已由真实 helper 重新部署并验证。
6. **重启后外部租户间歇占用 die0/1**（26GB/die，波动重启）：E2E 看护器等待持续空闲窗口（≥3 分钟 AICore=0 且 mem<4GB）后按会话重试执行；一次启动碰撞已 fail-closed 退出未污染。6 小时窗口耗尽后 die0 常驻 14.5GB（14:02 实测）——独占显存物理不可得，die0/1 路线放弃。**自我披露**：看护器首轮"无窗口"的部分原因为本人 `idle_now` 门禁返回语义反转 BUG（ok=1 被当失败码），已修复并如实计入。
7. **卡对迁移（15:33）**：dev2/3 探针通过但建池 `507899` → vmm_probe 实证 CANN VMM 命名空间分裂 → Host SO 三调用点补丁（`1dc46d7d`）→ dev6/7 建池/正确性全 PASS。期间两次自伤事故并披露：① 残留的 fast_deploy_helper 包装器在 dev6/7"谎报 kernel 在位"→ AICPU `0x2a`（移除后真实 helper 秒级完成自动部署）；② dev2 随后被外部租户占用（86–96%），dev2/3 路线放弃。
8. **E2E 与挂死调查（16:03–18:54）**：ABBA 4 会话中 cand_r2 于 ~85s 挂死（首版 runner 仅查文件存在误判 OK，分析脚本以 completed/failed 校验揭穿，runner 已修复）；复跑 cand_r3 干净、**base_r3 同型挂死**（基线臂，排除候选特异）；侧车监视捕捉双 die 100%×60s 画像；定位为 G3 §8 已知 P0 同型复发（§5.2）。对照实验：dev4/5 2/2 干净、budget=5s 4/4 干净。
9. **还原（19:0x）**：Host SO 双路径还原 `88d39285` 并复核；设备 SO 还原 pre-D4 `b8ca96f0`；vllm 进程零残留，dev6/7 显存回落基线。

## 8. 结论与建议

1. **单算子层面：有条件 GO。** Add Tile 1024→2048 在正式 V2 Kernel 上以零正确性回归、零停顿、四组 ABBA 全部 ≥5.5% 的可复现方式获得 **配对 p50 +6.26%**，超过首轮 ≥5% 目标；BiSheng 编译与 Matmul UB 资源门全部关闭。
2. **真实模型：E2E PASS（性能）。** TPOT -1.98%/吞吐 +1.86%/TTFT 噪声级，与单算子收益的稀释算术自洽（§5.1）；单层 +6.26% 亦不足以外推整体打平 Stock（参照 D2 决策约束）。
3. **稳定性：G3 §8 P0 同型复发，维持"V2 不具备生产可用性"判定。** 本轮新增证据（双 die 100%×60s 画像、基线臂同挂、dev4/5 与 budget=5s 对照）确认其与卡、与 Add tile、与换纪频率无关，为负载相位时序竞态；修复需 mf 仓侧根因深挖（沿 G3 §12：挂死时刻 plog/aicore dump、压力放大复现、orchestrator 等待路径日志补强）。**建议将到达戳环覆写路径与换纪间隙 SQ 积压交互列为首要排查方向**（本轮静态分析定位的两条候选路径，§5.2）。
4. **不合并**：本分支按纪律不合入 `feat/310p-memfabric-public-api`；虽 E2E 性能判据已关闭，但 P0 稳定性风险未解除前不接受生产合入（单算子 GO + E2E 性能 PASS 不足以放行）。

> 附：本轮所有环境偏差均未触碰 MemFabric 协议、IB/SDMA/DONE、缓存可见性、KVER11 协议与正式优化点边界。
