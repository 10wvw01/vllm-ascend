# 310P3 MemFabric 单波 AR 引擎设计（阈值 2048 + q=4 固化）

状态: M1 已实施，Phase A（算子级快速验证）五 gate 全部通过（2026-10-14 实机） | 日期: 2026-10-14 | 作者: 10wvw01
关联: [小M计算优化设计.md](./小M计算优化设计.md) · [310p_memfabric_mm_ar.md](./310p_memfabric_mm_ar.md) · [VALIDATION_REPORT](../../../csrc/_310P/memfabric_mm_ar/VALIDATION_REPORT.md) · 后继（协议固定开销重构）: [单KernelAR协议重构设计.md](./单KernelAR协议重构设计.md)

---

## 1. 背景与目标

### 1.1 L1 实测现状（2026-10-12/13，v9/v11 内核两轮一致，q 为 BATCH_BASEM_COUNT）

| M | stock (ms) | fused q=1 | fused q=2 | fused q=4 | 结论 |
|---:|---:|---:|---:|---:|---|
| 16 | 0.30 | 0.318 | 0.318 | 0.318 | N-split 持平略亏 |
| 256 | 0.38~0.42 | **0.475~0.480** | 0.627~0.632 | 0.786~0.791 | 补零越多越亏 |
| 512 | 0.47~0.51 | 0.632~0.635 | **0.627~0.628** | 0.789~0.790 | 单批恰好最优 |
| 1024 | 0.78 | 1.05~1.10 | 0.863~0.872 | **0.804~0.863** | 同上 |
| 2048 | 1.46~1.48 | 1.84~1.89 | 1.48~1.50 | **1.28~1.29** (+12.5%) | 流水开始盈利 |
| 4096 | 3.26~3.33 | 3.39~3.44 | 2.73~2.76 | **2.30~2.34** (+29%) | 收益峰值 |

规律：**恰好装满一个通信批（单批 = 单波单 signal）永远是该 M 的最优 fused 配置**；
亏损全部来自"补零凑批"（q=4 M=256 亏 91%）与"批太小协议摊不平"（q=1 全线亏损）。

### 1.2 目标（用户决策，2026-10-14）

1. 路由简化为两段：**M < 2048 = 小 M，MM 全量算完后一次性 AR（单波串行链）**；
   M ≥ 2048 = q=4（batch_m=1024）批流水。
2. 把"一次性 AR"沉淀为**基于 MemFabric 的独立通用 allreduce 算子**（不绑定 MM），
   mm_ar 融合路径复用同一引擎。
3. N-split（M ≤ 255）保留不动——它已经是"MM 算完 → 单波单 signal"结构。
4. 实施顺序：**先算子级快速验证**（Phase A），数据说话后再引擎化与 serve 验证。

**单波的性能定位（与 stock 结构对齐）**：单波 = MM 串行 AR，结构与
stock（matmul 串行 HCCL allreduce）一致；且 SDMA 单向传输量为 HCCL allreduce
的一半，协议开销（~0.2ms/波）被传输节省基本抵消。预期：**模板命中直读时
单波 ≈ stock（±5%）**；批流水的价值是在 M ≥ 2048 区间把 SDMA 藏进下一批
MM 再拿 +12~29%，属锦上添花而非保底。2048 以下不追求赢，只要求不输。

## 2. 已定设计决策

| # | 决策 | 内容与理由 |
|---|---|---|
| D1 | 单波区间与阈值 | `M < SINGLE_WAVE_MAX_M`（默认 **2048**，env 可调）走单波；≥ 阈值走 q=4 批流水。阈值 1024<M<2048 区间的单波 vs 2 批流水优劣由 Phase A4 实测定夺终值（见 §8 预期与 O3） |
| D2 | q 固化 | `BATCH_BASEM_COUNT` 默认 1 → **4**；批流水仅服务 M ≥ 阈值。合法值仍 {1,2,4} 供 bench A/B |
| D3 | 单波 MM 计算 | M-split cooperative（现有 `Mf310pDirectProducerKernel` 直接复用）。模板阶梯为 **256 粒度铺满 {256, 512, 768, 1024, 1280, 1536, 1792, 2048}**（8 套，per-core 均为 32 的倍数）：`P = ceil(M/256)×256` 必命中模板值，**所有 M%256==0 的点直读 x 零 staging**——这是"单波 ≈ stock"成立的前提（2 幂阶梯会把 M=1536 强制跳到 2048 模板，多付 8MB staging + 512 行白算，已否决）。不扩展 N-split 到 >255：带宽账在 M≈2048 对称反转 + UB 约束（见小M计算优化设计 §6 与本文 §11-O1） |
| D4 | 新增模板 tiling | 新增 5 套 {768, 1280, 1536, 1792, 2048}：`singleCoreM = P/8 ∈ {96,160,192,224,256}`，`basicM = min(singleCoreM, 128)`（即 {96,128,128,128,128}），basicN=256、basicK=64、CONFIG_NORM 沿用。per-core ≤128 的模板 UB 峰值与现役一致；2048 模板 basicM=128 控制 UB 与 1024 模板持平 |
| D5 | 变长 signal，kernel 零改动 | producer 的 signal 长度本就是运行时入参 `batchBytes`。单波以 `batch_index=0`、`signal_bytes = P×4096` 复用同一 kernel，仅新增 launch 入口（变长校验）。wait 复用现有变长机制（同 small path）；add 复用 `mf310pAddKernel`（连续 elems） |
| D6 | staging 策略（M1 简单版） | `M % 256 == 0` 必命中模板 → 直读 x 零拷贝；`M % 256 ≠ 0` 时 copy M 行到 P 行 once-zeroed scratch（≤2048 行），从 scratch 算。与现有尾批同构。段式 producer + 尾块局部 staging（仅 copy M%256 行）列为优化项 O2 |
| D7 | 通用 AR 算子 | 新 torch op `memfabric_allreduce(x, tp_rank) -> y`：copy x → send_arena → signal region → wait → add → quiet → ack。首版限 rows ≤ 2048（单波语义），超出 TORCH_CHECK 拒绝由调用方分段。诚实定位：其价值是能力复用，D2D copy 开销使其纯 AR 场景未必快于 HCCL |
| D8 | 两段实施 | **M1（算子级快速验证）**：mm_ar 内部单波路由 + 256 粒度模板阶梯 + q=4 默认 + bench 对比，验证 §8 性能假设；**M2（引擎化）**：抽通用 AR op、UT/文档/envs 全套。M1 不达标（中 M 改善 < 预期一半）则回评审 |
| D9 | 两侧一致性 | P/阈值/q 均由 M 与 env 派生；env 两侧不一致 → MAIL_MISMATCH trap（BUG#2 前科）。UT 守卫 + 使用文档显式要求全 rank 一致 |

## 3. 分区图（目标状态）

```text
M: 0 ──[ N-split 单波（现状不动） ]──≤ 255
     ──[ M-split 单波: P=256 粒度模板阶梯覆盖 M,
          MM 全量 → 单 signal(P 行) → wait → add(M 行) → quiet → ack ]──< 2048
     ──[ q=4 批流水: batch_m=1024, lookahead=1（现状不动） ]──→
```

单波串行链（与现有 N-split 波编排同构，仅 producer 不同）：

```text
prepare_wave(gen) → gate → [staging copy M 行, 仅 M%256≠0]
→ wave_producer(P 模板, 变长 signal_bytes)
→ wait(len=P×4096, imm=0) → add(out, elems=M×2048)
→ quiet → ack
```

## 4. AR 引擎分层（M2 目标架构）

```text
torch.ops._C_ascend.memfabric_allreduce(x, tp_rank)     ← 通用 op（新）
torch.ops._C_ascend.memfabric_mm_ar_allreduce(...)      ← 融合 op（现有）
        │ mm_ar 单波路径: MM 写 send_arena 零拷贝          │ 通用路径: D2D copy
        └──────────────┬─────────────────────────────────┘
                       v
        runtime: 单波编排（gate/signal/wait/add/quiet/ack + generation/warmup）
                       v
        ABI v8: mf310p_wave_producer_async / mf310p_signal_region_async
                / wait_wave(变长) / add(连续) / gate / quiet / ack
                       v
        MemFabric public: smem_shm_sdma_signal/wait/quiet（黑盒边界不变）
```

## 5. ABI 变更（v7 → v8）

`VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION` 7 → 8（内部定制 ABI，不维持二进制兼容，
Python/C++/device 同 commit 升级）。

| 项 | 类型 | 内容 |
|---|---|---|
| `mf310p_wave_producer_async` | 新增 | 变长单波 M-split producer：入参 `x, w, padded_rows(∈{256,512,768,1024,1280,1536,1792,2048}), generation, stream`；slot=send_arena 基址、signal `len=padded_rows×4096`、`imm=0`。内部按 padded_rows 选模板 launch |
| `mf310p_signal_region_async` | 新增（M2） | 通用 AR：单 block signal `send_arena[0, len)` → `peer_recv_arena`，`imm=0`；源数据由调用方先行 copy 入槽（同 stream 排序保证可见性） |
| `mf310p_wait_wave_async` | 新增（薄封装） | 变长 wait：expected `dst=recv_arena 基址, len=padded_rows×4096, imm=0`。device 侧复用 `mf310pWaitBatchKernel`（len 本就是入参） |
| `mf310p_add_wave_async` | 新增（薄封装） | 连续 reduce：`elems=valid_rows×2048`，复用 `mf310pAddKernel`；源恒为 send/recv arena 基址（batch 0） |
| `mf310p_warmup_wave_producer_async` | 新增 | 覆盖全部 8 个模板符号（含新增 5 套）+ signal region kernel 的首启 no-op |
| `mf310p_layout_t` | 不变 | 单波 ≤2048 行 ≤ arena_rows(8192)，槽位/credit 协议零改动 |
| 既有 API | 不变 | 批流水/N-split/协议族全部保留 |

## 6. 内核改动清单（`memfabric310p_device.asc`）

| 改动 | 说明 |
|---|---|
| `MF310P_PRODUCER_TILING` ×8 | 阶梯 {256, 512, 768, 1024, 1280, 1536, 1792, 2048}；新增 5 套按 D4 tiling，CONFIG_NORM。UB 峰值：per-core ≤128 与现役持平，2048 模板 basicM=128 亦持平 |
| `mf310p_device_launch_wave_producer_async` | 新 launch 入口：`switch(padded_rows)` 八 case；校验 `signal_bytes == padded_rows×2048×2`、`batch_index==0`、`generation!=0`。`Mf310pDirectProducerKernel` 本体零改动（slot 偏移 = 0×len，signal len = 入参） |
| `mf310pSignalRegionKernel`（M2） | 单 block：`smem_shm_sdma_signal(gva, send_arena, peer_recv_arena, len, 0)` + fail-stop。不做 MM、不碰 ready cells |
| warmup case ×8 | 新增 5 套模板 + signal region 的 generation/len=0 no-op 形态 |
| wait/add/gate/quiet/ack | **零改动** |

## 7. Runtime / Python / 环境变量改动

`memfabric_mm_ar_runtime.cpp`：

- 常量 `kSmallPathMaxM = 256` 保持；新增 `kSingleWaveMaxM`（env
  `VLLM_ASCEND_310P_MEMFABRIC_MM_AR_SINGLE_WAVE_MAX_M`，默认 2048，合法值
  {1024, 2048}，M1 后由 A4 定终值）。
- 路由：`small_path = M < min(batch_m, 256)`（不变）；
  `single_wave = 256 ≤ M < kSingleWaveMaxM`；`M ≥ kSingleWaveMaxM` 走批流水（q=4）。
- producer scratch 扩容：`max(batch_m, 2048)` 行（q=4 下 16 MiB，+8 MiB 显存），
  分配时清零一次契约不变。
- 波编排：单波路径 = §3 串行链；`M % 256 == 0`（必命中模板）直读 x，否则
  staging copy M 行。
- warmup：`warm_wave_locked` 覆盖 8 模板 + signal region（M2）；graph 捕获前
  eager 预热契约不变。
- 通用 AR impl（M2）：同 RuntimeState/context 复用；`rows > 2048` TORCH_CHECK。

`memfabric_mm_ar_binding.cpp`：注册 `memfabric_allreduce`（M2）。
`vllm_ascend/_310p/ops/memfabric_allreduce.py`（M2 新文件）：torch op 封装。
`vllm_ascend/_310p/ops/memfabric_mm_ar.py`：Plan 增加 `single_wave_max_m`；
`BATCH_BASEM_COUNT` 默认改 4。
`vllm_ascend/envs.py`：新增 `VLLM_ASCEND_310P_MEMFABRIC_MM_AR_SINGLE_WAVE_MAX_M`
（默认 2048）文档化；`BATCH_BASEM_COUNT` 注释更新为"批流水专用"。
`tests/ut/_310p/test_memfabric_mm_ar_source.py`：守卫三段路由、模板阶梯、
staging 契约、两侧一致性校验、feature-off 零依赖。

## 8. 性能预期与实测（Phase A 验收对照，2026-10-14 实机 v11 内核）

fused 相对 stock（负=亏）。"现状 q=4"取 2026-10-12/13 两轮 v9/v11 实测。

| M | 现状 q=4 | 预期（单波） | **实测（单波）** | 备注 |
|---:|---:|---:|---:|---|
| 16 | 0.318 | 不变 | **0.321（+2.1%）** | N-split 路径不动 ✓ |
| 256 | 0.786~-91% | -12~-27% | **0.477（-24.8%）** | ✓ 区间内；较现状快 39% |
| 300 | ~0.79 | 外推亏 | **0.629（-54.4%）** | 较现状快 20%；stock 0.407 快于锚点拉大百分比 |
| 512 | 0.789~-66% | -15~-34% | **0.633（-32.6%）** | ✓ 区间内；较现状快 20% |
| 768 | ~0.80 | -15~-25% | **0.700（-26.7%）** | 略超上限；较现状快 12% |
| 1000 | 0.80~0.86 | — | **0.893（-27.2%）** | 与现状同构（补零单批），轮次噪声 |
| 1024 | 0.804~-3% | 不变 | **0.792（+0.9%）** | ✓ 持平 |
| 1536 | 流水 1.270 | ≈stock（-5~+3%） | **1.234（-12.5%）** | 单波较流水略优 2.8%（A4） |
| 1792 | 流水 1.304 | — | **1.352（-27.1%）** | 流水略优 3.7%（A4）；本行 stock 1.064 异常偏快 |
| 2047 | 流水 1.327 | -8~0% | **1.578（-16.0%）** | 流水优 15.9%（A4）；M%256≠0 最坏点 |
| 2048 | 1.28 +12.5% | 不变（走流水） | **1.292（+11.7%）** | ✓ |
| 4096 | 2.30 +29% | 不变（走流水） | **2.319（+29.4%）** | ✓ |
| 8192 | — | 不变（走流水） | **4.432（+34.5%）** | 流水大 M 收益扩大 |

**staging 定标（R2 解除）**：M=1000 vs 1024（同为 1024 模板）差 101µs/4MB、
M=511 vs 512 差 41µs/2MB → D2D copy 实测 **~40-50GB/s**，非引擎慢速路径
（4.9GB/s 前科未复现）；M%256≠0 的 staging 代价约为 25µs/MB。

**A4 阈值结论**：1536 单波与流水互有胜负（±4% 内）、2047 流水优 16%——
**默认阈值 2048 保留**（结构简单 + 直读点不劣），M%256≠0 且接近 2048 的
点（如 2047）是已知小亏损，O2（尾块局部 staging）可消除。

数据文件：`bench_results/single_wave_m1/{op_ab_q4.csv, op_ab_q4_sw1024.csv}`。

## 9. Phase A：算子级快速验证（M1 完成定义）

改动范围刻意最小（runtime 路由 + 新增 5 套模板 + launch 入口 + bench 扩行，
不动 ABI 版本号的通用 AR 部分），目标是用最少代价证实/证伪 §8。

| Gate | 内容 | 通过标准 | 结果 |
|---|---|---|---|
| A1 build | feature-on bisheng dav-2002 编译（8 模板+wave launch）；feature-off 构建+导入无回归 | 双构建通过 | **PASS**（2026-10-14；feature-on/off 双向重编验证） |
| A2 bit-exact | eager TP=2，q=4：M ∈ {255,256,257,511,512,513,767,1000,1023,1024,1025,1280,1536,1792,2047,2048,2049,3072,4096,6144,8192}，atol=rtol=0，两 rank 双向 | 全 PASS | **PASS**（21/21，覆盖 N-split/单波直读/单波 staging/批流水） |
| A3 性能 | 上表 M 集 vs stock vs 现状（同进程交替、3 轮取中位、含 staging 定标行） | 中 M（256~1023）改善 ≥ 预期区间下限；>15% 偏差逐行归因 | **PASS**（256/512 落预期区间；300/768 偏差源于 stock 基线快于锚点，绝对值好于外推；staging 定标见 §8） |
| A4 阈值专项 | 1024<M<2048：SINGLE_WAVE_MAX_M=2048（单波）vs =1024（流水）同 M 对比（如 M=1536/1792） | 预期单波 ≈ stock、流水再优 ~10%；数据定阈值终值并回写 D1/D2 与本表 | **PASS**（1536 单波优 2.8% / 1792 流水优 3.7% / 2047 流水优 15.9%；阈值 2048 保留，见 §8） |
| A5 graph | 最小 capture/replay（M=512/1536/2047 单波 bucket ×32 replay）+ replay 后 eager 连续性 | 双 rank bit-exact | **PASS**（3 bucket 全绿） |

A1~A5 全绿后才进入 M2（通用 AR op + UT/文档/envs）与 Phase B。

## 10. Phase B：完整验证矩阵（M2 后）

- **正确性**：A2 全集 × eager/graph 交叉；diverge-va（rank1 预占 4GB）；
  大小 M 交错（含先单波后流水的 arena 复用）；通用 AR op bit-exact
  （rows ∈ {1, 255, 256, 1024, 2048, 2049→拒绝}）。
- **稳定性**：repeat≥1000 @M=1024 单波；mixed-M 交错；60s soak；进程
  create/destroy；HCCL 共存微基准（建池前后无退化）。
- **协议健康**：debug_snapshot 全程 status=0；两侧 env 不一致场景显式
  MAIL_MISMATCH trap（负向测试）。
- **serve**：四组合（stock/fused × eager/graph）+ TTFT 专项（serve 级误差棒：
  serve 重启重复，方法论教训沿用 10-14 报告）；eager 融合长负载 >10min 禁令
  不变（上游 D3/R6，graph 模式为准）。

## 11. 不变量、风险与开放问题

**不变量（全部沿用）**：MemFabric 黑盒（仅 public signal/wait/quiet）、
core0 唯一 data signal owner、mail 严格校验 status/dst/len/imm、fail-stop +
失败 context 不复用、arena 按预算定容、eager 单流、graph 前 eager warmup、
feature-off 零 MemFabric 依赖。

**风险**：

- R1: 新增 5 套模板（per-core 96~256）编译/数值——A1/A2 首查；失败则逐套
  剔除降级为 2 幂阶梯（接受对应 M%256==0 点的 staging 开销），方案不受阻。
- R2: ~~staging copy 走引擎慢速路径（4.9GB/s 前科）~~ **已解除**（Phase A 定标
  实测 ~40-50GB/s，M%256≠0 代价约 25µs/MB）。
- R3: 1024<M<2048 单波较流水少 ~10%（SDMA 掩盖差）——预期内权衡：单波换
  路由简单性与"≈stock 保底"，A4 用数据确认后定阈值终值。
- R4: 两侧 P/env 不一致 → trap——负向 UT + 文档（BUG#2 前科）。
- R5: 通用 AR 的 copy→signal 可见性依赖同 stream 排序——A/B 首版若数值错，
  在 signal region kernel 前加显式 cache clean 再判。
- R6: 上游 D3/R6 wedge 与本设计正交，eager 长负载约束照旧。
- R7: scratch +8 MiB 显存（16 MiB 常驻）。

**开放问题**：

- O1: 中 M（256~1024）改用 N-split 计算（模板扩 T=512/1024，blocked slot +
  blocked add）：每核流量账更优（A 重复读 vs B 重复读在 M<2048 区间 N-split
  占优），但 L2 聚合带宽行为未定标。列为 A3 后的可选实验，不阻塞主线。
- O2: 段式 producer（满模板段直读 + 尾 256 行段 scratch）消灭全量 staging，
  使 M%256≠0 仅 copy 尾块（≤255 行，~60µs）：复杂度高、收益集中在
  M%256≠0 的非整行点（当前最坏 M=2047 多付 ~400µs），M1 后按 R2 定标
  结果决定。
- O3: 1024<M<2048 终态（单波 vs 2 批流水）——A4 输出。当前判断：单波
  ≈stock（结构对齐、单向传输减半抵消协议开销），流水再优 ~10%；若 A4
  证实且该区间 serve 占比可观，可将阈值降至 1024，默认 2048 换路由简单性。
