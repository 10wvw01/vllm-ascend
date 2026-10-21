# G1-D 精准诊断实机执行报告（2026-10-21，310P3）

> 方案依据：`docs/G1D_EXECUTION_PLAN_20261021.md`（下称"执行计划"）。
> 执行：OpenCode @ 310P3 服务器，2026-10-21 04:00–06:15 UTC（首轮 D0–D4），
> 复审追加复现批次 08:22–08:50 UTC（D1R4）。
> 证据分级约定：**observed**（实机直接观测）、**derived**（由观测推出）、**unknown**（无证据，禁止推测补全）。
> 原始完整证据（含 SHA256 自校验清单）仅存服务器本地
> `/home/w00804146/310P/g1d_evidence_20261021_R1/`；入库脱敏包见
> `csrc/_310P/memfabric_mm_ar_v2/evidence/g1d_20261021/`。

## 1. 执行总览与 Gate 状态表

| Gate | 内容 | 状态 | 关键结果 |
|---|---|---|---|
| D0 现场基线锁定 | 双仓库 SHA、原装库哈希、KVER/JSON、加载路径、备份 | **PASS** | 原装基线与 G1-C 恢复态逐字节一致（§2） |
| D1.0 静态与构建 | V2 G1-D + MF Host G1-D 构建、符号/ABI/宏核验 | **PASS** | v12 kernel 重构建与保全产物逐字节一致（§3） |
| D1.0 CPU UT | 双仓库离线解析器/源码合同测试 | **PASS** | V2 侧 32/32，MF 侧 3/3（含 1 处测试层修复） |
| D1.1 正确性 Gate | TP2 C1（诊断 OFF→ON） | **PASS** | OFF 51/51、ON 51/51 位精确（§4） |
| D2-A 最小压力 | 异步窗口 1/2/4 × 256 + Stock/HCCL 对照 | **PASS** | 三窗口全部位精确（w4 需 harness 修复，§5.2） |
| D2-B 边界取证 | 真实 seq mod 1024 边界/非边界对照 | **完成（健康侧）** | 边界槽/邮件实测正确，768 调用零失败（§5.3） |
| D2-C V1/V2 共性 | 同二进制 V1 对照 | **BLOCKED** | V1 兼容性无证据，按纪律不切 KVER9/10（§5.4） |
| D1.2/D3 Gate E 再现 | 首轮 3 会话全净；复审批次 D1R4 **复现** | **复现（D1R4）** | G1-D 判别证据落袋：stage50 未返回 + 被等邮件=环末封（§6） |
| D4 报告与恢复 | 结构化证据、SHA256、恢复证明 | 本文件 + §9 | 首轮环境已恢复断言校验；复现批次部署态见 §6.4 |

**总结论（observed）**：正确性与受控压力全部通过；Gate E 首轮三会话未复现后，
复审复现批次第 1 轮（D1R4）即复现 G1-C C2R2 型双 Rank 停滞，并首次取得判别性
设备侧证据：双 Rank 同代际（1730）core0 停在 **stage 50（wait 从未返回，非 52/53）**，
被等待的邮件均为各自到达环**最后到达的一封**（Host D2H 视角已计入且环绕自洽），
槽位 439/441 位于环中部（**非 1023 边界**）。根因裁决状态与剩余 unknown 见 §8。

## 2. D0：现场基线锁定（observed）

### 2.1 分支与工作树

| 项 | 值 |
|---|---|
| vllm-ascend | `diag/310p3-g1d-minimal-forensics-20261021`，起点 `70734f715`，执行中追加诊断兼容提交（§7） |
| MemFabric | `10wvw01/memfabric_hybrid_310p` 同名分支，起点 `1f87d0f4a`，追加 1 提交（§7） |
| 工作树 | 主树直接检出 G1-D 分支；`*.so`/`build*`/`tmp` gitignore，porcelain 全程洁净；预存未跟踪资产经 `.git/info/exclude` 原地保留 |
| 隔离保证 | 未拉取 `wgm-dev-310p`；`feat/310p-memfabric-public-api` 分支未动（其原构建 .so 三件备份于 `/tmp/opencode/g1d/backup_main_tree_so_20261021T040553Z/` + SHA256SUMS） |

### 2.2 原装环境哈希（与 G1-C 恢复态一致）

| 产物 | SHA256 / 探针 | 判定 |
|---|---|---|
| Host `libmf_hybm_core.so`（/usr/local + site-packages 双位置） | `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1` | = G1-C 恢复值 |
| AICPU v11 `libmf_sdma_orch.so`（安装侧）/ launch JSON | `db7134b8…` / `25e16b0a…` | 一致 |
| 设备侧 v11（`aicpu_kernels_device/libmf_sdma_orch_v11.so`） | mini_kfc 探针 dev0/dev1 sync=0 | 在位，全程未触碰 |
| 设备侧 v12 | 探针 507018 | 已清除（符合 G1-C 结束态） |
| 保全 v12 三件套（`/tmp/opencode/g1c/kernel_v12_stage/`） | `.so=02290083…` `v12.json=90172726…` `cust.json=74ad4668…` | = G1-C 记录 |
| 实际加载路径 | site-packages 包内 `lib/libmf_hybm_core.so`（ctypes CDLL）+ /usr/local 同步 | 双位置同哈希 |
| KVER | 安装 11；诊断线 12（显式 JSON） | - |

备份与恢复演练：G1-C 备份 `sha256sum -c` 复核 OK；G1-D 新建
`build-g1d/g1d_backup_20261021T041522Z/`（只读 + 自校验 + ownership）；恢复脚本为
G1-C 已验证模式的断言式改写。设备占用准入见 §6.1（外来占用按纪律挂起记录）。

## 3. D1.0：构建、静态核验与部署（observed）

### 3.1 MemFabric Host G1-D

- 配方与 G1-C 同构：Ninja RELEASE、`-DMF_SDMA_ORCH_KVER=12 -DMF_SDMA_ORCH_G1C_DEVICE_TRACE=ON`、BUILD_PYTHON=ON，独立 `build-g1d/`。
- 产物 `libmf_hybm_core.so` = `6e9dc2d699aa7b718a83f93397fbc2d0f18a799e384ffda9031d62f25195e55a`；内嵌 `commit: 1f87d0f4…`；`[MF_G1D]`/`[MF_G1C]` 字符串在位；ldd 依赖集合一致；导出 T 符号 1052 = 原装。
- **v12 可复现性（derived）**：G1-D 分支重构建的 KVER=12 `libmf_sdma_orch.so` 与保全产物 SHA256 逐字节一致（`02290083…`）——设备侧源码自 G1-C 起零改动。

### 3.2 V2 G1-D（vllm-ascend）

- `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=ON + G1C_TRACE=ON + G1D_TRACE=ON`（CMakeCache 预置）。
- bisheng 与 Host CXX 命令均含 `-DMF310P_V2_G1C_TRACE=1 -DMF310P_V2_G1D_TRACE=1`（build.make/flags.make 复核）。
- 设备库独立名 `libmf310p_v2_trial_device_g1d.so`（`d9a6de66…`）；`vllm_ascend_C…so`（`bcbb4f67…`）DT_NEEDED → g1d 库；正式默认名库未被覆盖；导出符号 21/21 与默认构建一致。

### 3.3 部署（断言式）

原装双位置备份 → G1-D Host 替换（逐位置哈希断言）→ v12 经**独立进程** `orch_deploy_helper`
（G1-D 树构建，`849f93c0…` 与现网版一致）部署设备侧：dev0 rc=0 "deployed and verified
size=14072"，dev1 rc=3 已存在（双 die 共享设备文件系统）。校验级别 = 符号 + 精确字节数
（与 G1-C 同级；设备 OS 无哈希工具）。

### 3.4 CPU UT

V2 侧 `test_g1d_evidence_summary.py`/`test_g1c_trace_summary.py`/`test_gate_e_forensic_summary.py`/`test/ut/` 全量 **32/32**；MF 侧 `test_g1c_device_trace_source.py` **3/3**。

## 4. D1.1：TP2 C1 正确性 Gate（observed）

| 运行 | 配置 | 结果 |
|---|---|---|
| C1-OFF | G1-D 全套二进制，诊断环境变量全关 | **51/51 位精确 PASS**；无 507014 |
| C1-ON | 同二进制，全开关开启（g1d_run.sh c1 + pyinject 提级） | **51/51 位精确 PASS** |

C1-ON 取证链：`[G1B]` 7 进程；`[MF_DIAG]` 160 行（15s 纪干净轮转，exitReason/mainRet=0）；
`[MF_G1C]` 142；`[MF_G1D]` 142 帧 **0 snapshotError**；`[V2_G1C]` 138 帧；`[V2_G1D]` 112 条
同代邮件返回（如 `generation=38 captured=1 mailSeq=154 waitChunk=3 mailStatus=0`）。

**derived**：诊断开关不影响正确性行为；同代捕获机制工作正常。

## 5. D2：受控异步窗口与边界取证（observed）

### 5.1 D2-A 窗口压力（M=4096、chunk=1024、256 调用、含 Stock/HCCL 对照）

| 窗口 | 结果 | 耗时 |
|---|---|---|
| 1 | **PASS 256/256** | 139.3s |
| 2 | **PASS 256/256** | 114.1s |
| 4（修复 harness 后） | **PASS 256/256** | 103.1s |

### 5.2 窗口=4 失败定位（observed → derived）

- 现象：原版 harness 两 Rank 第 5 调抛 `V2 pooled outputs exhausted … Refusing unsafe
  alias/reuse`；前 4 调已验证；Epoch 健康；无 timeout；无 snapshotError。
- 探针 A（单窗口 ×4）：PASS → 失败锁定在**跨窗口边界**。
- 探针 B（解绑验证循环变量 + 二次 gc）：×8 PASS；修复后全量 256 PASS。
- **derived**：测试脚本侧残留 Python 引用（`result` 循环变量 + 单次 gc 不足以使池视图
  StorageImpl 引用计数回落）触发**保护性**槽耗尽；V2 池"拒绝不安全别名/复用"守卫行为正确。
  非通信/池缺陷。修复已提交（§7 #2）。

### 5.3 D2-B 边界取证（observed）

三窗口流量结束静默帧（dev0）：

| 字段 | 实测值 |
|---|---|
| arrHead | 1302 / 1307 / 1307 |
| stamp1020..1023 | **1021, 1022, 1023, 1024** |
| stamp0..2 | **1025, 1026, 1027** |
| mail1023Seq / mail0Seq | **1024 / 1025** |
| quiet60..63, quiet0..1 | 1277..1282 连续 |
| headSlot/headStamp | 278→279、283→284（与 arrHead 语义自洽） |

`[V2_G1D]` 返回 seq 覆盖 1..1300；`[MF_G1C]` reqSeq 至 1302；hwHeadLast 推进跨 1024。
768 次调用零失败，无失败侧对照样本（须由 Gate E 提供）。

**derived**：受控窗口工况下，"仅 1024 边界丢邮件/丢写"假设未获支持；不得外推到
prefill/decode 交叠工况。

### 5.4 D2-C：BLOCKED

V1 与 v11/v12 的 ABI/部署/路由兼容性无证据；按计划不切 KVER9/10 凑结果。本轮无 V1 侧结论。

## 6. D3：Gate E 再现（observed）

### 6.1 设备占用阻断（准入纪律执行）

04:41–05:00Z 设备 0 被外来命名空间作业间歇占用（AICore 0→87%、内存 2→9.7GB 周期，
npu-smi PID 宿主不可见）。按 D0 停止条件挂起并记录（`blockers/`），05:00:55 确认
双设备回基线（1352/2025MB）后才启动 D1R1。

### 6.2 三次会话结果（fused Eager 原工况：input 4096 / output 256 / 25 prompts / 并发 10）

| 会话 | 窗口 | 请求 | V2 末代际 | arrHead/reqSeq 终态 | 507014 | snapshotError |
|---|---|---|---|---|---|---|
| D1R1（05:01–05:20Z） | 620s | **75/75** | 3008 | 15027 | 0 | 0 |
| D1R2（05:22–05:38Z） | 620s | **75/75** | 3008 | 15027 | 0 | 0 |
| D1R3（05:41–06:09Z） | 1240s（长稳） | **150/150** | 6008 | 30027 | 0 | 0 |

- 三会话合计 300 请求 0 失败 0 超时；`[V2_G1C]` 全部会话末帧 8 核 stage=120（干净退出，
  g1c_trace_summary.json）；`[V2_G1D]` 同代邮件返回 1694/1390/2816 条/会话；
  `[MF_G1D]` 每会话 0 snapshotError。
- 边界静默帧（observed）：D1R1/D1R2 `stamp1020..1023=14333..14336、mail1023Seq=14336、
  stamp0..2=14337..14339`；D1R3 `…29693..29696、mail1023Seq=29696、mail0Seq=29697`
  ——第 14/29 圈边界连续正确。
- **未复现 G1-C C2R2 型停滞**（C2R2 于 rep1 第 3 请求即双 Rank gen208 卡 stage50/60；
  本轮三会话无任何中途停滞帧，V2 代际全程推进）。
- bench 窗口（D3 关联性能，BENCH_WINDOW_ONLY）：D1R3 六窗 output_tok/s
  22.9→30.3→28.5→30.0→30.0→29.9，TTFT p50 ~12s（并发排队），TPOT p50 ~250ms；
  与 G1-C C2R1 健康基线同量级。1Hz 采样下不作微秒级计时结论；正式性能须无取证版本另测。
- 判定（按计划）：**保留健康证据，不宣布修复**。三会话 ≈ 45k 件 AICPU 消费、44 次
  1024 边界穿越、~12k V2 代际零异常——构成对 H3"低概率单件丢失"假设的**压力反驳证据**
  （derived；非证明，见 §8）。

### 6.3 复审批次 D1R4：**故障复现，判别证据落袋**（08:23–08:38Z，observed）

背景：首轮三会话未复现后，经裁决追加 6 轮复现批次（620s 标准工况，任一轮复现即停）。
**第 1 轮（D1R4）即复现**，编排器按纪律停止放大（D1R5–R9 未执行）。运行时二进制与
首轮完全相同（Host `6e9dc2d6…` 双位置、`libmf310p_v2_trial_device_g1d.so d9a6de66…`、
v12 `02290083…`；会话 provenance 已记录，分支源码-二进制溯源复核见 §6.4）。

**会话时间线（UTC，双 Rank）**

| 时刻 | 事件 |
|---|---|
| 08:23:12 | fused serve 启动（全诊断环境，G1-D 全取证） |
| 08:26:52 | healthy（80 层） |
| 08:28:17 | 双 Rank 首调 V2（M=4096 路由） |
| rep1 | 25/25 全部完成（健康） |
| ~08:34:36 前 | dev0 到达环冻结于 arrHead=8632（最后推进 ≤08:34:35，此后 14 个 1Hz 采样不变） |
| 08:34:49.395 | rank0（Worker_TP0）`aclrtSynchronizeEvent` 507014 AICore timeout（plog 0x7150025，硬件超时 0x25） |
| ~08:34:50.1 | dev1 到达环 8632→8634（rank0 侧 AICPU 编排继续排空请求环，向 dev1 环补发 8633/8634 后冻结） |
| 08:34:58.078 | rank1（Worker_TP1）507014（同签名；间隔 8.7s，rank1 先完成 gen1730 的 c1 返回再卡 c2） |
| rep2–rep11 | rep2 完成 17 请求后停滞；rep3–11 全 0（连接失败） |

**停滞瞬间双 Rank 关键状态（最后有效帧，observed）**

| 证据 | rank0 / dev0 | rank1 / dev1 |
|---|---|---|
| V2 8 核代际 | 1730（8/8 一致） | 1730（8/8 一致） |
| V2 core0 | `1730:0:50` = 等 **chunk0** 邮件，**未达 52/53** | `1730:2:50` = 等 **chunk2** 邮件，**未达 52/53** |
| V2 core1–7 | `1730:0:60`（IB 汇合等待，下游） | `1730:2:60`（同） |
| `[V2_G1D]` 同代返回 | 末次为 gen1728 c3（seq 8625）；gen1730 零返回 | gen1730 c1=seq 8633 **status=0 正常返回**后卡 c2 |
| 本端到达环 arrHead | **8632**（冻结 ≥13s） | **8634**（08:34:50 冻结） |
| 被等待邮件 | seq 8632（= dev0 环**最后到达**的一封，槽 439） | seq 8634（= dev1 环**最后到达**的一封，槽 441） |
| 动态头槽（head=下一写槽） | 440，headStamp=7609（上圈值，自洽） | 442，headStamp=7611（同） |
| 边界槽 stamp1020..1023 / mail1023 | 8189..8192 / 8192（lap-8 值，自洽） | 同左 |
| `[MF_G1C]` SQ | hwHead=hwTail=302（空，rc=0），reqSeq 8634 全消费，phase 8 空闲 | hwHead=hwTail=308（空），reqSeq 8636，phase 8 |
| Epoch | 纪 30 运行中，此前全 exitReason=0 | 纪 31 运行中 |
| `[MF_G1D]` snapshotError | 0（全程 449/459 帧双 Rank） | 0 |

**G1-D 判别性发现（相对 G1-C 的增量，observed）**

1. **stage 50 ≠ 52/53**：双 Rank 的 `smem_shm_sdma_wait_at` 从未返回——直接排除
   "wait 返回但邮件校验失败"（计划 §5 判定表 stage52/53 分支关闭）。
2. **被等待邮件 = 各环最后到达的一封**：两侧 arrHead（Host D2H 读共享保留区计数）
   均已计入被等待的 seq（8632/8634），且环绕内容完全自洽（lap-9 槽 0..439 连续、
   边界槽 lap-8 值正确、head 槽持上圈值）——**Host D2H 视角"已到达" vs AICore 等待
   "永不观察"**，命中计划 §5 判定表"本端当前 headStamp 正确但 Core0 stage50"行。
3. **相邻槽不对称（dev1）**：seq 8633（槽 440）正常返回后 ≤0.2s，seq 8634（槽 441，
   同 lap 相邻）落环但等待永不观察——支持单件丢失而非连续损坏。
4. **槽位 439/441 为环中部**（lap 9 第 439/441 槽），距 1023 边界约 584 槽——
   "仅 1023 边界丢写"假设获**故障侧设备证据直接反证**（与健康侧 D2-B 反证互补）。
5. 控制面签名与 C2R2 完全一致：SQ 排空、epoch 干净轮转、编排空闲、请求环全消费
   ——首停点保持在"CP SQE 链执行完成"与"AICore 观察到对端到达邮件"之间。

**剩余 unknown（本轮无法判定，已精确命名）**：被等待槽（head−1：dev0 槽 439、
dev1 槽 441）的 stamp/mail **内容**未被采样——G1-D 采样器读动态头（head=下一写槽
440/442），恰差 1 槽。因此不能区分：
(a) 该槽 stamp/mail 写入丢失或未落地（发送侧 AICPU 写链异常）；
(b) 写入在但 AICore 轮询读不可见（缓存可见域/读游标问题，H3 收窄形态）。

**与历史故障率合并（derived）**：G3 1/2 + G1-C 1/2 + G1-D 1/4 → 3/8 ≈ 37.5% 会话级
复现率；G1-D 埋点下可复现（掩蔽假说被证伪——首轮"三净"系样本量所致）。

### 6.4 复审批次溯源与分支核对（08:50Z 复核，observed）

- vllm-ascend `118339651`：构建基线 `70734f715`→HEAD 的 `src/`、`cmake/`、
  `csrc/**/src/` **零改动**（仅 test/docs/evidence 层）；树内 `vllm_ascend_C`（`bcbb4f67…`）
  DT_NEEDED→`libmf310p_v2_trial_device_g1d.so`（`d9a6de66…`）。
- memfabric `b4b45878`：构建基线 `1f87d0f4`→HEAD 仅 `tools/.gitignore`；部署 Host
  双位置 `6e9dc2d6…`（内嵌 `commit: 1f87d0f4`），与分支源码等价。
- 设备 v12 在位（探针 sync=0）；D1R4 会话 provenance 记录的 Host/v12 JSON 哈希与
  上述一致。
- CPU UT 复核：V2 32/32、MF 3/3（本地 `tmp/` 会话目录移入本地存档后）。

## 7. 诊断兼容修复与错误记录（均已提交各自 G1-D 分支）

| # | 仓库 | 提交 | 内容 | 定性 |
|---|---|---|---|---|
| 1 | vllm-ascend | `bc50a079d` | V2 目录合同测试未含 G1-C 证据打包提交入库的 `evidence/`，分支上从未可能通过 | 测试层 |
| 2 | vllm-ascend | `5118ac82d` | 异步窗口 harness 跨窗口残留引用释放（§5.2） | 测试层 |
| 3 | memfabric | `b4b45878` | tools/.gitignore 先于 orch_deploy_helper 存在，构建后污染洁净门禁 | 工具卫生 |

执行期环境问题（未构成代码修复，已按约定规避）：`g1d_run.sh` 不自带 CANN 环境；
调用方 source 后必须以**前置追加**方式设 PYTHONPATH（覆盖式 export 丢失 set_env.sh 的
CANN Python 路径 → `No module named 'tbe'` → ACL 500001 双 Rank 启动失败）。

## 8. 根因假设矩阵与 unknown（含 D1R4 复现后更新）

| ID | 假设 | 证据 | 状态（D1R4 后） |
|---|---|---|---|
| H3 缓存可见性/单件丢失（G1-C 主嫌疑） | D1R4：双 Rank 同代 stage50 永不返回；被等邮件=环末封，Host D2H 计数已计入且环绕自洽；相邻槽 440 成功/441 失败 | **显著收窄为两分支**：(a) 发送侧该槽 stamp/mail 写入丢失；(b) 写入在但 AICore 轮询读不可见（缓存可见域/读游标） |
| "仅 1023 边界丢写" | 健康侧（D2-B + 首轮三会话静默帧）边界槽全对；**故障侧**：被等槽 439/441 环中部 | **双向反证，假设关闭**（故障可在任意槽位发生） |
| wait 返回后校验失败（stage52/53 分支） | D1R4 双 Rank 停在 50，未出现 52/53 | **关闭**（本轮故障形态下排除） |
| H1 SQ 容量失控 | D1R4 停滞瞬间 SQ 空、rc=0；epoch 轮转正常 | 维持反驳（再确认） |
| H2 Epoch 停顿 | 同上，exitReason 全 0 | 维持反驳 |
| G1-D 埋点掩蔽故障 | D1R4 在全 G1-D 埋点下复现 | **证伪**（首轮三净系样本量） |
| 槽耗尽（首轮新增） | harness 残留引用，守卫正确 | 关闭（非通信类） |

**unknown（D1R4 后仍不能回答，禁止推测补全）**：
1. 被等待槽（dev0 槽 439 / dev1 槽 441）的 stamp/mail **内容**：采样盲区（动态头采样
   读 head=下一写槽，恰差 1 槽）。区分 (a) 写入丢失 vs (b) AICore 读不可见的唯一直接
   证据缺口。
2. 双 Rank 对称丢失是否同因（两侧被等邮件均为各自环末封——可能是同一 wall-clock
   窗口的对称表现，也可能是两件独立丢失；C2R2 亦然）。
3. 发送侧 AICPU 写链在 seq 8633(dev1 环)/8632(dev0 环) 之后停止的原因归属：对端
   V2 停滞导致的反压（derived 倾向：请求环已排空，属正常耗尽）vs 写链自身异常——
   现有证据不能排除后者，但 (a) 分支下两者本为一体。
4. 无"近失/慢等待"样本：等待要么即时完成要么永久挂起（C2R1/C2R2/D1R4 一致），
   不支持"慢速型"，但样本量不足以排除低概率自恢复。

**下一步最小单变量（建议，须复核方批准）**：扩展 Host 采样器读 **head−1..head−N**
（含 stamp+mail 全字段，仅 Host 侧 D2H 读取集扩展；不动 SQE/Mailbox/Epoch/布局/
设备核），重跑 Gate E 复现轮。复现时即可直接读出被等槽实值，一步区分 (a)/(b)。

## 9. 环境恢复证明（observed）

**首轮恢复（06:11–06:13Z）**：

| 步骤 | 结果 |
|---|---|
| 残留进程 | `pgrep vllm serve/torchrun/EngineCore/Worker_TP` 为空 |
| Host core.so 双位置 | 恢复 `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`（断言通过），权限 644 |
| 设备 v12 | `rm -f` 后探针 507018（不存在，符合预期） |
| 设备 v11 | dev0/dev1 探针 sync=0（在位，未触碰） |
| 备份重封闭 | `g1d_backup_20261021T041522Z/SHA256SUMS.txt` 校验 OK |
| 诊断环境变量 | 新 shell 无残留 |
| 设备占用 | dev0 1353MB / dev1 1594MB（≈G1-C 基线 1661/1593） |
| 主工作树 | G1-D 分支 porcelain 洁净；分支切换回 `feat/310p-memfabric-public-api` 后如需原 .so 可自 `/tmp/opencode/g1d/backup_main_tree_so_20261021T040553Z/` 恢复（或重新构建） |

**复现批次当前态（08:50Z，D1R4 取证完成后）**：G1-D 诊断部署仍在位（Host 双位置
`6e9dc2d6…`、设备 v12 在位）——复现轮证据已全部保全，等待裁决是否实施 head−N 采样
扩展后继续复现轮，或恢复原装。若恢复：执行 `/tmp/opencode/g1d/restore_g1d.sh`
（断言式，同首轮；D1R4 后新备份 `g1d_backup_20261021T082*` 亦在位）。设备 0/1 无
残留 Worker，内存 1673/1701MB。

## 10. 交付物与下一步建议

交付物：本报告；脱敏证据包 `evidence/g1d_20261021/`（索引、会话统计、边界 JSON、
脱敏节选、原始清单 SHA256；含 D1R4 故障会话）；本地完整原始证据
`g1d_evidence_20261021_R1/`（不入库，含 D1R4 设备 plog 29 件）；双分支诊断兼容提交（§7）。

下一步（不越权执行）：
1. **（最高优先级，最小单变量）** Host 采样器扩展读 head−1..head−N（仅 Host 侧，
   不动协议/布局/设备核），重建部署后重跑 Gate E 复现轮——一步区分"写入丢失"vs
   "AICore 读不可见"（§8 unknown #1）。
2. 复现率统计：如需收敛会话级概率，可在扩展版就绪前继续 G1-D 标准轮（当前 3/8）。
3. D2-C V1 对照需先完成 V1 ABI/部署兼容性核验（单独评审）。
4. 性能终验与"根因已解决"结论必须由关闭埋点的全矩阵长稳给出（计划 D4 纪律）。
