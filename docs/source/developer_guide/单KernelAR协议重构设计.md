# 310P3 MemFabric 单 Kernel AR 协议重构设计

状态: P0/P1/P2 已收官——V1（两-kernel）为最终形态；P2 V2（单 kernel）实测负收益已回退；N-split m=241 mail 混淆竞态已根因修复（MF310P_WAVE_TAG） | 日期: 2026-10-15 | 作者: 10wvw01
关联: [单波AR引擎设计.md](./单波AR引擎设计.md) · [小M计算优化设计.md](./小M计算优化设计.md) · [VALIDATION_REPORT](../../../csrc/_310P/memfabric_mm_ar/VALIDATION_REPORT.md)

---

## 1. 背景：固定开销是单波路径亏损的唯一来源

### 1.1 算子级证据链

单波（mm 串行 ar）与 stock（matmul 串行 HCCL allreduce）同构，MM 与传输
本体均已对齐（MM ~273µs vs ~270µs；SDMA 2MB ~100µs vs HCCL ~100µs，
多批流水时 wait 降至 1.5-2.2µs 可证），但 M=512 单波仍亏 27~33%。
亏损全部来自 AR 协议栈的固定开销。

### 1.2 A6 实验（2026-10-14）：graph 模式不降反升，锁定根因

| M | eager fused | graph replay | 差 |
|---|---:|---:|---|
| 256 | 0.477 | **0.633** | +33% |
| 512 | 0.630 | **0.647** | +3% |
| 1536 | 1.258 | **1.278** | +2% |

graph 消除了 launch 间隙（~140µs），但捕获路径把 **32 KiB ready 区清零
（GVA 池上引擎慢速路径 ~260µs，与大小无关的任务调度开销）烘进了图**，
每次 replay 都付。净效果变慢。

### 1.3 固定开销账本（逐项归因与消灭手段）

| 开销项 | eager | graph | 根因 | 消灭手段 |
|---|---:|---:|---|---|
| 6 个 AICore launch 的 host 间隙 | ~140µs | 0 | gate/producer/wait/add/quiet/ack 每协议步一个 kernel | 合并 kernel |
| 64B error-channel memset | ~56µs | ~56µs | host enqueue 的引擎任务在池内存上走慢速调度 | kernel 内 AICore 清（ns 级） |
| 32 KiB ready 清零（graph 路径） | 0 | **~260µs** | 同上，graph 需要 wave-invariant generation | kernel 内每核清自己的 cell |
| 协议内核本体（gate/wait/quiet/ack） | ~10µs | ~10µs | 语义必需 | 保留 |
| **合计** | **~206µs** | **~326µs** | | → 目标 **~30-50µs** |

AICore 内直接 store+clean 一个 64B cell 是纳秒级；慢的是"host enqueue 引擎
任务 + 池内存"组合的固定调度开销，与清零大小无关。**把所有清零和协议步
挪进一个 kernel 是唯一同时消灭前三项的手段。**

## 2. 目标

1. 单波路径（256 ≤ M < 2048）协议固定开销 **~0.2/0.3ms → ≤50µs**；
   eager 与 graph 两种模式开销一致（graph 不再需要烘焙引擎 memset）。
2. 协议语义（mail FIFO、固定 credit、arena 复用安全、fail-stop）**不变**，
   只改执行位置：6 kernel + 2 引擎任务 → 1~2 个 AICore kernel。
3. 预期算子级：M=512 单波 0.63 → **~0.46ms（追平/反超 stock 0.477~0.495）**；
   M=256 0.477 → **~0.34（反超 stock 0.358~0.382）**；
   M=1536 1.26 → **~1.1（追平 stock 1.10~1.11）**。
4. N-split（M≤255）与批流水（M≥2048）路径本期不动（前者已持平、后者
   盈利且开销被摊薄），但 probe 结论为它们未来同构改造铺路。

## 3. 已定设计决策

| # | 决策 | 内容与理由 |
|---|---|---|
| K1 | 形态：V1 两 kernel → V2 单 kernel | V1 = `[clear+gate+MM+signal]` + `[wait+add+quiet+ack]`：消灭全部引擎 memset 与 4 个 launch，绕开 TPipe 复用风险，先兑现 ~90% 收益；V2 = 全合一，解决 MM 的 `Matmul/TPipe` 与 add 的 `TQue` 同 kernel 共存后落地 |
| K2 | ready/error 清零进 kernel | kernel 开头每核 store 0 + `DataCacheCleanAndInvalid` 自己的 64B cell；core0 清 error channel。eager 单调 generation 值域技巧退役，两种模式统一为小值域 generation（graph 语义）。跨波安全由同 stream kernel 串行保证（上一 kernel 完全结束下一 kernel 才启动，graph replay 同理） |
| K3 | gate 位置：kernel 开头、core0 执行 | core0 在自己 MM 前轮询 credit mail，等待与其他 7 核的 MM 并行；lockstep 下 ≈0 暴露，非 lockstep（serve 节奏漂移）下等待与 MM 重叠优于现状（现状 gate 独立 kernel 阻塞全波） |
| K4 | mail FIFO 顺序保持 | 产出顺序（signal data → quiet → ack）与消费顺序（gate credit → wait data）合并前后完全一致（见 §4.3 论证），MemFabric 侧无需任何变更 |
| K5 | wait 广播机制 | core0 `wait` 到 data mail 后写一个 "mail-ok" flag（复用 ready cell 区第 9 个 cell 或独立 64B），其他核轮询后进入 add——与现有 8-cell ready 汇合同构反向 |
| K6 | staging copy 保持前置引擎任务 | M%256≠0 时的 D2D staging 实测 ~40-50GB/s（25µs/MB），非慢速路径，不值得纳入 kernel |
| K7 | 批流水与 N-split 不动 | 批流水多批 signal/wait 的编排与单 kernel 形态不兼容（跨核状态机复杂度爆炸），且 M≥2048 已盈利；N-split 可作为后继同构改造（O3） |
| K8 | ABI v8 → v9 | 内部 ABI，同 commit 升级；新增 `mf310p_wave_fused_a/b_async`（V1 两入口）与 warmup；旧 6 入口保留给批流水/N-split 路径 |

## 4. 协议设计

### 4.1 现状（每波 6 AICore kernel + 2 引擎任务）

```text
host:  memset64B(error)            ← 引擎慢速 ~56µs
k1:    gate        (core0 等 credit mail)
host:  staging copy                ← M%256≠0，引擎快速
k2:    producer    (8核 MM + ready 汇合 + core0 signal data)
k3:    wait        (core0 等 data mail + 校验)
k4:    add         (8核 向量加)
k5:    quiet       (core0 等本地出站 SDMA)
k6:    ack         (core0 signal credit)
       graph 路径另有 32KiB ready memset 烘图 ~260µs
```

### 4.2 目标 V2（单 kernel 状态机；V1 为其按 `||` 切两半）

```text
k_fused (blockDim=8):
  [每核]  清自己的 ready cell;  [core0] 清 error channel      ← ns 级
  [core0] gate: 轮询 credit mail（与其他核 MM 并行）
  [每核]  MM(M-split 模板) → clean 自己的 C 区 → 写 ready
  [core0] 等全部 8 ready → signal 整个 padded 槽
  [core0] wait data mail（校验 status/dst/len/imm）           ← 唯一暴露段
  [core0] 写 mail-ok flag
  [每核]  轮询 mail-ok → add 自己的 valid/8 行分片
  [core0] quiet（等本地出站 SDMA）→ signal credit
  V1 切分: k_a = 清+gate+MM+signal ‖ k_b = wait+flag+add+quiet+ack
```

### 4.3 mail FIFO 正确性论证（合并不改语义）

MemFabric 每端一个 FIFO mail 队列；正确性 = 产出顺序与消费顺序各自保持。

- **对端产出**（对端 wave N，stream 顺序）：`signal data_N → quiet → signal
  credit_N`；wave N+1 的 `data_{N+1}` 在其后。FIFO 到达序：
  `… data_N, credit_N, data_{N+1}, credit_{N+1} …`
- **本端消费**（wave N+1）：`gate 消费 credit_N`（k_a/k_fused 开头）→
  `wait 消费 data_{N+1}`。与到达序一致 ✓
- 合并前后两个顺序逐字不变——kernel 合并只把六段顺序代码放进一个（或
  两个）launch，未移动任何 signal/wait 的相对次序。
- **credit 语义不变**：本端 wave N+1 的 gate 通过 ⇔ 对端 wave N 的
  wait+add 已完成（credit_N 已到）⇔ 本端 send arena 槽可安全复用。
  gate 与 MM 并行不破坏此语义：signal data_{N+1} 发生在 gate 通过之后
  （core0 在 ready 汇合后、且 gate 已过才 signal）。

### 4.4 清零的跨波安全

- eager：同 stream kernel 串行，wave N+1 的 k_a 清 cell 时 wave N 的全部
  kernel（含 core0 读 cell 的汇合与 k_b）已结束。
- graph：同一张图顺序 replay，replay N+1 的 k_a 在 replay N 的 k_b 后
  （stream 顺序）；图内清零随 replay 执行，替代烘焙的引擎 memset。
- generation 统一小值域（`wave % 2^16` 或每图固定值）：因 cell 已在 kernel
  内清零，不再依赖"陈旧值 ≠ 当前值"的单调性技巧。

## 5. ABI v9 变更

| 项 | 类型 | 内容 |
|---|---|---|
| `mf310p_wave_fused_a_async` | 新增 | V1 kernel A：`x, w, padded_rows, generation, stream`；内含清零 + gate + MM + ready 汇合 + signal |
| `mf310p_wave_fused_b_async` | 新增 | V1 kernel B：`out, valid_rows, padded_rows, stream`；内含 wait + flag 广播 + add 分片 + quiet + ack |
| `mf310p_warmup_wave_fused_a/b_async` | 新增 | 8 模板符号 + 两 kernel 的 no-op 首启 |
| `mf310p_probe_api_async`（P0） | 新增（临时） | probe：同核连续 `signal→quiet→wait→signal→quiet` 混用序列验证（见 §7） |
| 旧入口（gate/quiet/ack/wait_wave/…） | 保留 | 批流水与 N-split 路径继续使用 |
| `prepare_wave_async` | 单波路径退役 | 不再为单波清任何区（清零进 kernel）；批流水/N-split 保留现状用法 |

## 6. 内核改动清单（`memfabric310p_device.asc`）

| 改动 | 说明 |
|---|---|
| `Mf310pClearCell` | 每核清自身 ready cell / core0 清 error channel（store + clean，ns 级） |
| `Mf310pWaitCreditCore0` | core0 轮询 credit mail（复用 gate 校验逻辑：status/dst/len=CREDIT_TAG） |
| `Mf310pWaveFusedAKernel<ST, P>` | 清零 + core0 gate + 8 核 MM + ready 汇合 + signal（模板 8 套复用现有 tiling） |
| `Mf310pWaveFusedBKernel` | core0 wait + mail-ok flag + 8 核行分片 add + core0 quiet + ack |
| mail-ok flag | ready cell 区追加 1 个 64B cell（layout 偏移 +1 cell，producer_control 区域扩 64B） |
| launch/warmup 入口 | `mf310p_device_launch_wave_fused_a/b_async`、warmup ×16（8 模板 × 2 kernel） |

runtime 侧：单波编排缩为 `staging(可选) → fused_a → fused_b`；wave_seq/
prepare_wave 单波路径退役；warmup 更新；trace 保留。

## 7. P0：MemFabric API 同核混用 probe（已通过，2026-10-14 实机）

当前每个 public API（signal/wait/quiet）各占一个 kernel，**同一 AICore
连续混用从未验证过**。probe（双 rank、真实 pool/geometry、专用 kernel
`mf310pApiProbeKernel`，torch op `memfabric_ar_probe`）按序执行：
`wait(credit) → N × [AICore 清 status → signal(8B) → quiet → wait →
mail{dst,len,imm} 校验 → payload 往返校验]`。

| 模式 | 结果 |
|---|---|
| per-launch：1000 个 launch，每 launch 一轮完整混用序列 | **PASS**（V1 生产形态） |
| single：单 kernel 内 1000 轮循环 | 初次 **PAYLOAD_MISMATCH**（见下），读前显式 `DataCacheCleanAndInvalid` 后 **PASS** |
| single 压力：5000 轮 | **PASS** |

**产出结论**：

1. **public API 同核任意混用安全**——signal/quiet/wait 连续序列在同一
   AICore 上 5000 轮全对，V1/V2 前提成立。
2. **L1 一致性规则（V2 硬约束）**：launch 边界保证 AICore L1 与 SDMA
   写入一致（per-launch 免 invalidate 全过）；**同一 launch 内重复读
   SDMA 写入的地址必须先 clean+invalid**（single 模式第 2 轮读回第 1
   轮陈旧值实锤；SDMA 写不被 AICore L1 侦听）。V1/V2 的 add 段是每
   launch 对各自分片的首读，天然安全；flag 轮询沿用 ready cell 的
   clean+poll 模式。
3. K2 清零模式（AICore store+clean status）在 kernel 内循环工作正常。
4. 排查中确认 probe 曾报 status=8 系测试自身期望值错误（用本端 tag
   校验对端 payload），非 MemFabric 缺陷；mail 校验全过即已证明序列
   正确。

probe 脚本/日志：`/tmp/opencode/p0_probe.py`、`/tmp/opencode/p0_*.log`。

## 8. 验证矩阵与 P1 实测（2026-10-14 实机，v11 内核）

### 8.1 P0 probe

（见 §7——per-launch 1000 轮 / single 1000+5000 轮全 PASS，含 L1 重复读
规则实锤。）

### 8.2 P1 V1（两-kernel）核心验证

| 项 | 结果 |
|---|---|
| bit-exact（atol=rtol=0） | **PASS**：21 个 M 点（255~8192，覆盖 N-split / 单波直读 / 单波 staging / 批流水），eager TP=2 |
| graph（A5） | **PASS**：M=512/1536/2047 单波 bucket × 32 replay + replay 后 eager 连续性，双 rank bit-exact |
| 性能 vs stock（同进程交替，A3 方法论） | 下表 |

fused 相对 stock（负=亏），对照 M1（六-launch 协议链）：

| M | M1 单波 | **V1 fused** | 变化 |
|---:|---:|---:|---|
| 256 | 0.477（-24.8%） | 0.471（-22.4%）¹ | 见脚注¹ |
| 512 | 0.633（-32.6%） | **0.475（+8.3%）** | 反超 |
| 768 | 0.700（-26.7%） | 0.642（-16.5%） | 减亏 |
| 1000 | 0.893（-27.2%） | 0.785（-13.5%） | 减亏 |
| 1024 | 0.792（+0.9%） | **0.783（+1.5%）** | 反超 |
| 1536 | 1.234（-12.5%） | **1.109（+4.3%）** | 反超 |
| 1792 | 1.352（-27.1%） | 1.268（-17.1%） | 减亏 |
| 2047 | 1.578（-16.0%） | 1.456（-7.1%） | 减亏 |
| 2048+ | 流水不变 | 流水不变（2048 +10.1% / 4096 +27.5% / 8192 +34.8%） | — |

¹ **2026-10-15 更正**：初版记录的 256=0.347（+17.6%）经复核是 **N-split
路径**（T=256 模板，当时 `kSmallPathMaxM` 路由边界使 M=256 走 N-split）
的成绩，并非单波 fused。单波 fused 的 M=256（pad 256 直读）真实水平
就是 ~0.47——比 N-split 的 0.347 慢 34%。当前路由（M=256 走单波）
下该点亏损；N-split 在 M=256 的收益待 O1（N-split fused 化）收割。

**§2 设计预期逐项核对**：M=512 → 预期 ~0.46 / 实测 0.475 ✓；M=1536 →
预期 ~1.10 / 实测 1.109 ✓。固定开销削减量：M=512 eager 0.633→0.475 =
**-158µs**，与账本（memset 56 + launch 间隙 ~100）吻合。

**graph 模式（A6）**：M=256 replay 0.379（vs stock-eager -1.8%，追平）；
M=512 0.474（+4.0%）；**M=1536 replay 0.995（+10.6%）**——引擎 memset
烘图问题消除（M1 时 graph 反而慢 33%），eager 与 graph 开销一致。

**已知残留**：768/1000/1792/2047 亏 7~17%，集中在 staging 点与非 2 幂
per-core 模板（768 的 per-core 96 行 tiling 效率待 profile）。

### 8.3 P1 稳定性（2026-10-15 实机收尾）

| 项 | 结果 |
|---|---|
| repeat 压力 | M=512 repeat1000 PASS；单波 band 90s/88000 次零错 |
| 混合 soak（修复前） | BAND=all 随机序列 **79680 次 7 次 mismatch（全部 m=241）**，mismatch 后挂死 |
| 竞态根因 | **mail 混淆**：N-split T=256 的 data mail（dst=recv arena、len=1MB、imm=0）与单波 256 的 data mail（dst/len/imm 全同）**不可区分**。N-split 波比单波快 → 对端领先一波时，其 N-split data mail 被本端单波 kernel-B 的 wait 误消费（blocked 布局按行主序读 → mismatch），mail 错位后 credit 错配 → 双向死等挂死 |
| 修复 | `MF310P_WAVE_TAG`：单波 kernel A 的 signal imm 由 0 改为专属 tag，kernel B 校验同改（`memfabric310p_device.asc`）。三条路径的 data mail 全部可区分（单波=WAVE_TAG / N-split=batchIndex / 批流水=len+batchIndex） |
| 修复后验证 | 定向 256↔241 交替 5000 次 PASS；soak 序列重放 6000 次 PASS；纯净 band：单波 88000 / N-split **195560** / 批流水 33640 全零错；bit-exact 21 点复跑 ALL PASS |
| graph 竞态（A5 期间发现并修复） | mail-ok flag 残留（graph 固定 generation=1）：kernel A core0 顺带清 mail-ok cell，变输入 replay ×32 PASS |
| 异步深 enqueue 死锁 | in-flight wave 深度≥2 永久挂死（kernel 内 signal 满环自旋占 AICore + 后续 wait 无法执行）。`kMaxInFlightWaves=1`（每波 sync）规避；对 bench/serve 现有用例无性能影响（每步天然 sync 点）。根因待 MemFabric 侧排查 |

### 8.4 P2 V2（单 kernel）实测与回退（2026-10-15）

V2（`Mf310pWaveFusedKernel`，MM 与 add 同 kernel，TPipe 两段作用域——
Matmul pipe 结束后 add 段用第二个 TPipe 从 UB 头重新分配）**编译与
bit-exact 全过**（21 点 + A5 变输入 replay ×3 档 + graph/时序行为正常），
**TPipe 共存风险（R2）就此关闭**。但性能为**负收益**：

| M | V1（两-kernel） | V2（单 kernel） |
|---:|---:|---:|
| 256 | 0.347（N-split¹） | 0.473 |
| 512 | 0.475 | 0.502 |
| 1024 | 0.783 | 0.783 |
| 1536 | 1.109 | 1.101 |
| 2047 | 1.456 | 1.514 |

¹ 见 §8.2 脚注（该点 V1 时代走 N-split）。

小 M 劣化 27~126µs、M=2047 劣化 58µs，大 M 持平。**机理**：V1 的两个
launch 之间的 host 派发间隙并非纯开销——它恰好作为**流水缓冲**，掩盖
了对端 data mail 的到达延迟；V2 合一后 core0 的 wait 与 blocks 1-7 的
mail-ok 轮询在 kernel 内直接自旋暴露该延迟。**决策：V1（两-kernel）
为最终形态，V2 回退**；"V2 增量仅剩一个 launch 间隙"的预期被实测
否定（该间隙是免费流水）。

### 8.5 待办（后续）

- ~~P1 剩余~~：repeat/soak/竞态修复已完成；HCCL 共存未跑（脚本 import
  旧模块名，待修后补）。
- ~~P2~~：已完成（负收益结论 + 回退）。
- P3：serve 四组合 + TTFT 专项（serve 级误差棒）。
- O1：N-split fused 化（收割 M=256 的 34% 收益 + 协议面收敛）。

### 8.6 P3 serve 快速验证（2026-10-15，Qwen3.6-35B-A3B-w8a8，TP=2，50 prompts × 4096in/256out，并发 10，q=4）

| 组合 | ttft_med (ms) | tpot_med (ms) | tpot_p99 (ms) | itl_med (ms) | out_thr (tok/s) |
|---|---:|---:|---:|---:|---:|
| stock eager | 11789 | 257.8 | 290.8 | 190.4 | 33.3 |
| stock graph | 11842 | 148.5 | 182.4 | 79.1 | 51.8 |
| fused eager | 12009 | 292.7 | **4143.9** | 216.4 | **14.9** |
| **fused graph** | 11855 | 148.9 | 182.8 | 79.4 | 51.7 |

结论：

- **fused graph ≈ stock graph（-0.2%，噪声级）**：融合的算子级收益
  （单波段 +1.5~8.3%、流水段 +10~35%）在 serve 端到端**无法兑现**——
  decode 主路径是 M≤并发数（≤10）的 N-split 小 M 段（**未融合**），
  prefill 走流水段但 TTFT 占比小。**O1（N-split fused 化）才是 serve
  收益的钥匙**。
- **fused eager 生产不可用**：吞吐 -71%（vs stock graph），tpot_p99 高达
  4143ms，engine 吞吐时间线全程周期性崩塌（46 → 2-3 tok/s 的 10-20 秒
  级 stall 反复出现）——**D-mailorder 在真实 serve 混合序列下高频软性
  发作**（错位→kernel 内双向死等→被上层超时/重试解开→再错位）。
  stock eager 同条件下无此现象。
- **fused graph 稳定**：graph 整图原子重放锁定了波节奏，错位窗口消失，
  90 次请求零 stall。**当前生产推荐组合：fused + FULL_DECODE_ONLY
  graph**（融合无害且算子级更优，graph 是 tpot -42% 的真正来源，两者
  正交）。
- 与 v11 基线（2026-10-13）一致方向：stock eager tpot 218.7（本次
  257.8，机器底噪差异 ~15%），同批次内 A/B 对比可信。

## 9. 风险与开放问题

- R1: **probe 失败**（API 同核混用不安全）→ §7 降级方案，收益从
  ~180µs 降至 ~150µs（仍显著）。（已关闭：P0 通过）
- R2: TPipe 共存 → 已关闭（V2 实测编译+bit-exact 通过，见 §8.4）。
- R3: core0 gate 与其他核 MM 并行后，非 lockstep 场景（serve）credit
  等待与 MM 重叠——不会更差（现状 gate 阻塞全波），L2 的 92% 等待
  问题属两芯片节奏对齐（上游/调度层），不在本设计范围。
- R4: mail-ok flag 轮询可见性——沿用 ready cell 已验证的 clean+poll
  模式，P1 覆盖（含 graph 竞态修复）。
- D-mailorder: **混合序列跨路径 hang/软性 stall（开放，P3 已实测其
  serve 影响）**——WAVE_TAG 修复后 mismatch 消失，但 BAND=all 混合
  soak 仍偶发挂死（~1/12730，双向 kernel 内死等，无数据错；三条纯净
  band 各 3.3 万~19.5 万次零错）。**P3 serve 实测（§8.6）：fused
  eager 下高频软性发作（周期 10-20 秒级 stall，吞吐 -71%），fused
  graph 下零发作**——graph 的整图重放锁定波节奏是当前规避方案。
  错位第一步未定位（所有已知 mail 混淆面已被 dst/len/imm 严格校验
  拦截），疑点在 MemFabric mail 排序/消费语义（orchestrator 侧），
  需 MemFabric 团队配合。
- O1: N-split（M≤255）同构 kernel 化（其协议栈与单波同构，probe 结论
  直接适用；顺带收割 M=256 的 34% 收益）——后继。
- O2: 批流水路径的波编排 kernel 化（复杂度高、当前已盈利）——暂缓。
- O3: eager 单调 generation 技巧退役后，N-split/批流水是否统一收敛到
  kernel 内清零——随 O1/O2 处理。
