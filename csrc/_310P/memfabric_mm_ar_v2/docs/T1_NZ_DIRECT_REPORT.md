# T1 NZ-Direct 小 M 融合算子实机验证与修复报告（2026-10-17/18）

> 分支 `feat/310p-memfabric-public-api`，对象 `csrc/_310P/memfabric_mm_ar_v2/`（V2 T1 小 M：原始 NZ 权重 N-split + 直接 ND 输出）。
> 结论先行：**T1 全部正确性关口 PASS**（单卡 1028/1028、双卡 160 组合矩阵、随机 M/动态指针/4 权重/千次复用、T0 G0/G1/G2 回归、V1 回归）；修复 1 处 kernel 级缺陷（basicM=256 UB 工作区越界）与 1 处 host 侧守护缺陷（首启/新 y 的 MTE3 丢弃治愈策略），另修复 2 处测试程序缺陷。性能上 M≤255 仍以 stock（FP16 MM + HCCL AR）最快，M=256 起 V2 开始占优，M≥2048 V2 T0 overlap 全面最优。

## 1. 环境与对象

| 项 | 值 |
|---|---|
| 机器 | dav-2002（310P3）×8，测试用 dev0/dev1（bus 18:00.0），ASCEND_RT_VISIBLE_DEVICES=0,1 |
| CANN | 9.1.0（/usr/local/Ascend/cann-9.1.0，bisheng） |
| MemFabric | memfabric_hybrid 1.2.0（AICPU 内核 KVER=11；KFC 通道仍损坏，沿用 T0 的 assume pre-deployed 绕过） |
| 框架 | torch 2.10.0 + torch_npu 2.10.0.post4，vllm_ascend editable |
| 被测 | `mf310p_v2_trial_device.asc`（V2NzDirectSmallKernel 20 模板 + 新增 V2DiagNzDirectMmKernel）、`mf310p_v2_trial_adapter.cpp` |
| 参照 | V1（冻结未改动）、stock（F.linear + HCCL all_reduce, FP16） |
| 正确性判据 | 与 T0 一致：参考 = `F.linear(x,w).float()` → HCCL all_reduce → `.to(fp16)`，融合路径**位精确**（逐位相等）且协议 status=0；HCCL 参考全部在建池之前计算 |

## 2. 编译记录（PLAN 步骤 1）

- `bisheng --npu-arch=dav-2002 -O2 -std=c++17` 编译 20 个 `T×MM_CORES`（T∈{16,32,64,128,256} × MM∈{1,2,4,8}）模板 + T0 全部内核 + 新增单卡诊断内核：**全部通过**，产出 `libmf310p_v2_trial_device.so`（~2.1MB）。
- 官方构建路径 `python setup.py build_ext --inplace`（`VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1`、`..._V2=1`，cmake/memfabric_310p_v2.cmake）重编全量 `vllm_ascend_C`：**通过**，ctypes 可见 19 个 `mf310p_v2_*` C ABI 符号。
- `SetOrgShape(m,2048,2048,2048,2048)` + `SetSingleShape(m,NC,2048)` + 原始 NZ 子视图（B 偏移 `colOffset×16` FP16）+ ND 非紧凑行步长（orgKc=2048）：**dav-2002 编译与运行均支持**，无 API 合同回退。

## 3. 正确性结果

### 3.1 单卡 MM 隔离（PLAN 步骤 2；新增 `V2DiagNzDirectMmKernel`）

不建池、无通信，仅 N-split MM 直写 ND y，与 `F.linear` 位精确对比：

- 修复 basicM 越界（见 §5 BUG-1）后：**m=1..256 全量 × mm∈{1,2,4,8} = 1028/1028 PASS**。
- 无预同步时约 25%~30% 组合首启 MTE3 写被丢弃（y 全零，见 §5 BUG-2）；`DIAG_PRESYNC=1`（launch 前 stream 同步）后全部治愈——该现象与池无关（单卡亦复现），是对 T0 BUG #3 定性的一次修正。

### 3.2 双卡融合矩阵（PLAN 步骤 3/4）

`mf310p_v2_trial_run_nz_direct_small`，M×MM×Add 全组合，每次调用后 status=0 断言 + 位精确校验：

| 关口 | 范围 | 结果 |
|---|---|---|
| quick | M={1,16,17,255,256} × MM{1,2,4,8} × Add{1,8}，repeat 3 | 40/40 PASS（含修复前崩溃的 M=256） |
| full | M={1,2,4,8,16,32,64,128,255,256} × MM{1,2,4,8} × Add{1,2,4,8}，repeat 30 | **160/160 PASS**（两轮复测一致） |
| 官方路径复测 | repo `bench_nz_direct_small.py`（新编 vllm_ascend_C）quick 矩阵 | 40/40 PASS |

列边界专项：M=1、16、17、255、256 的 C(ND) 步长与 NZ 取列在全部核数下位精确；相邻 N 片 512B 行边界无串扰（mm=8 时每核 512B 列片恰好 cache line 对齐不互踩）。

### 3.3 压力与复用（PLAN 步骤 5）

| 项 | 结果 |
|---|---|
| 随机 30 个 M（seed 20261017，两端同列表）× MM{1,2,4,8} × Add{1,8}，4 权重种子轮换，输出指针 4 缓冲轮换 | 160/160 PASS |
| 1000 次连续调用（M/mm/add/seed 循环，逐次位精确） | 1000/1000 PASS |
| 1 核 MM + 8 核 Add | 无需额外 MM 参与，PASS |
| 同权重/不同权重指针、动态输出指针 | PASS |

### 3.4 T0 回归（大 M 串行/交叠不退化，PLAN 步骤 7）

| 关口 | 结果 | 与 T0 基线对照 |
|---|---|---|
| G0 IB probe（chunk 256/512/1024 × reps 1000/10000） | ALL PASS | 7.15–9.06 µs/rep（T0：7.1–7.7） |
| G1 串行（3 chunk × 3 rows × 1000 次复用逐次校验） | ALL PASS | med 一致（±5%） |
| G2 串行+交叠（3 chunk × {2048,4096,8192} × 2 模式 × 100 次） | 18/18 PASS | overlap c1024 M=8192：5.247ms（T0：5.198ms） |
| V1 层基线（rows 1..8192） | 全部 PASS 位精确 | V1 代码未改动 |

G1 以默认模式（建池后新分配 y）与 PREALLOC 模式各复测一轮，均 PASS。

## 4. 性能对比（中位数 ms，同机同环境；均为算子级 host 计时：enqueue + stream sync）

### 4.1 小 M（T1 范围）

stock=FP16 MM+HCCL AR（专用进程实测，FP16 all_reduce）；V2 T1=full 矩阵最优分核（两轮中位，会话间方差约 ±0.1ms）：

| M | stock | V1 | V2 T1 最优 (mm,add) | 最优方 |
|---|---|---|---|---|
| 1 | **0.307** | 0.346 | 0.330–0.357 (4–8,1) | stock |
| 2 | **0.297** | 0.351 | 0.330 (4,2) | stock |
| 4 | **0.296** | 0.337 | 0.327 (8,8) | stock |
| 8 | **0.292** | 0.394 | 0.324 (8,8) | stock |
| 16 | **0.257** | 0.350 | 0.322 (8,1) | stock |
| 32 | **0.291** | 0.342 | 0.325 (4,8) | stock |
| 64 | **0.288** | 0.361 | 0.321–0.469 (8,4) | stock |
| 128 | **0.296** | 0.461 | 0.322–0.454 (8,8) | stock |
| 255 | **0.427** | 0.542 | 0.462–0.495 (8,8) | stock |
| 256 | 0.519 | **0.473** | 0.451–0.496 (8,4~8) | V1/V2 并列优于 stock |

V2 T1 相对 V1：M=64–255 普遍更快（如 M=128：0.32–0.45 vs 0.46；M=255：0.46–0.50 vs 0.54），T1 的设计目标（超过 V1 小 M N-split）达成；但对 stock 的 HCCL FP16 AR 路线，M≤255 仍差 0.03–0.10ms，单 Kernel 固定协议开销（signal/wait/quiet/DONE 一来一回 + IB 扇入/放行）在小数据量下不回本，与 T0 结论一致。

### 4.2 大 M（T0 范围，含 V1/stock 对照）

| M | stock | V1 | V2 T0 serial (c1024) | V2 T0 overlap (c1024) | 最优方 |
|---|---|---|---|---|---|
| 512 | **0.557** | 0.594 | 0.765 (c512) | — | stock |
| 1024 | 0.891 | **0.838** | 0.974 | — | V1 |
| 2048 | 1.691 | 1.550 | 1.733 | **1.420** | V2 T0 |
| 4096 | 3.128 | 2.799 | 3.265 | **2.771** | V2 T0 |
| 8192 | 6.936 | 5.470 | 6.366 | **5.247** | V2 T0 |

### 4.3 分核配置结论（M 范围 → 推荐配置）

| M 范围 | MM 核 | Add 核 | 依据 |
|---|---|---|---|
| 1–63 | 4–8 | 1（任意） | mm=1 因单核串行 8 个 N 块劣化 +0.13–0.16ms；数据量小，Add 并行无收益 |
| 64–254 | ≥4 | 1–2 | mm≤2 劣化 0.12–0.16ms；Add 仍无收益 |
| 255–256 | 8 | 4–8 | add=1 劣化 +0.15ms（0.5–1MB 的 Add 工作量开始需要并行） |
| ≥2048（T0） | 8 (M-split) | 8，overlap + c1024 | 交叠相对串行提速 17–32%，全部超过 V1 与 stock |

首调用开销（T1 路径，符号预热后）：0.39–1.1ms，其中新 y 的 stream-sync 守护 <1ms；不再有 T0 时期的 15s 级首调用（见 §5 BUG-2/3）。

## 5. BUG 修复记录

### BUG-1（kernel，致命）：DirectNsplit basicM=256 触发 MTE on-chip read OOB

- 现象：M=256（=模板 T 满块）任何 MM_CORES 下首次发射即 `aicore exception`，错误串 "The address for the MTE instruction to read on-chip buffer is out of bounds"；M=255 同模板完全正常。单卡诊断内核同现象，排除通信/协议因素。
- 根因（CANN 头文件定性 + 实测）：ND C @ GM 且 `enableUBReuse` 时，matmul NZ2ND 输出暂存位于本地工作区偏移 `transLength = basicM×basicN×sizeof(fp16)`，共需 `2×transLength`（`matmul_local_workspace.h`：`GetNZ2NDWorkspace` 取 `[transLength]`，`SetWorkspace` 断言 `≥2×transLength`）。`basicM=256×basicN=256` 需 256KB，而本 kernel 的 256KB UB 还要容纳 6KB Add/IB 队列；release 构建把 `ASCENDC_ASSERT` 编译掉，满 basic 块发射越界读 UB。M=255 走尾块路径暂存量较小故幸免。
- 修复：`DirectNsplit::BASIC_M = min(T, 128)`（T0 已证实的 basicM 上限），T=256 拆两个 128 行 M 块。UB 预算复核：`2×128×256×2B = 128KB ≤ 250KB` 可用。
- 回归：单卡 1028/1028、双卡 160 组合 + M=256 全组合 PASS；M≤128 模板 tiling 不变，原结果不受影响。

### BUG-2（adapter，正确性守护修正）：首启/新 y 的 MTE3 丢弃治愈策略

T0 原实现：`run()` 对新 y 执行 `aclrtSynchronizeDevice()`。本轮三项实测修正：

1. **T0 kernel 首个真实发射也会丢 MTE3**（不只新 y）：G2 chunk=1024 首调用在已被 200+ 次调用写热的 y 上仍全零。enable=0 预热不激活 MTE 路径。修复：`run()` 在「首个真实发射或新 y」时 device-sync（`trial_symbol_launched` 状态位）。
2. **launch-stream 同步对 T0 kernel 无效**（G1 首调用回归复现：y 全零、status=0）；对 T1 kernel 则实测全部治愈（102 单卡 + 60 新指针样本 + 160 矩阵 + 1160 压力）。修复：`run_nz_direct_small()` 改用 `aclrtSynchronizeStream`（<1ms），`run()` 保留 device-sync。两者均非 Graph-safe，调用方逐次校验的合同不变。
3. **带池 `aclrtSynchronizeDevice` 随机 0–15s**：实测池存活期间 device-sync 等待 AICPU SDMA orchestrator epoch，单次 0/0.6ms–15s 不等（sync probe：12.3s、0、0、15.0s、9.2s、0、0、15.0s…）。这解释了本轮最初每组合 ~15s 的首调用；也暴露一个**偶发风险**：两 rank 的 device-sync 时长不对称时，先发射一方的 kernel 在核内邮件等待中被硬件看门狗超时（一次 1000-rep G1 复现， refine 后未再复现，机制分析见 §7）。

### BUG-3（测试程序）：跨 rank 种子/列表不一致

- stress 权重种子 `Random(777+rank)` 与随机 M 列表 `Random(20261017+rank)` 均按 rank 派生 → 两端选用不同权重/不同 M，kernel 协议（单邮件定长）被破坏，表现为 trap 41/42/超时。修复：种子去掉 rank 项。此为测试程序缺陷，kernel 无辜。

### 设备环境（非代码缺陷）

dev0/1 KFC system() 通道仍损坏（沿用 T0 绕过）；本轮多次 aicore 崩溃后设备无需复位，新进程数据面即恢复，但崩溃后首个 device-sync 会拖长。

## 6. 静态代码审视（两轮）

**第一轮（修复后 diff 审视）**：`DirectNsplit` BASIC_M 域分析（T∈{16..256}→BASIC_M∈{16..128}）；新增诊断内核与生产 MM 路径逐行镜像（含相同 UB 分区）；两个 launcher 参数校验与既有小 M launcher 一致；adapter 状态机（warmed/launched/warmed_y）在 NOWARMUP=1 与失败路径下的行为核查。结论：无新问题。

**第二轮（最终态全文件审视）**：`V2NzDirectSmallKernel` 的核参与矩阵（mm=1/add=8 与 mm=8/add=1 两极端）逐步推演无死锁；IB 事件复用（G0 万次探针 + 千次复用背书）；CleanLines 纪律（ownC 写后清、ownPeerC 读前清、y 走 MTE3 无需清）与 T0 一致；`SetOrgShape/SetSingleShape ≤ 静态 tiling` 合同满足；文档三处状态更新与代码一致。结论：无阻塞项；遗留观察记入 §7。

## 7. 遗留问题与风险

1. **MTE3 首写丢弃的硅级机制未明**：本轮新增证据——与池无关（单卡无池复现）；stream 同步对 T1 kernel 治愈、对 T0 kernel 无效，提示「时间窗」而非「同步语义」在起作用（50ms 纯延迟亦治愈）。T0 kernel 的 device-sync 守护在带池时随机 0–15s，不可用于生产；Graph 支持前必须以「输出缓冲预注册/预写」或驱动侧修复替代。
2. **偶发跨 rank 超时**：一次 1000-rep G1 中 aicore timeout（RTS_HWTS 看门狗杀掉核内 100M 次轮询的 mail wait）。触发模型：一 rank device-sync 慢（≤15s）而对方已发射。守护修正后未复现，但只要 run() 依赖 device-sync 该窗口就存在；根治需消除 device-sync（同 1）。
3. **profiler 分解未做**：本轮以 160 组合矩阵的配置敏感度替代 MM/IB/SDMA/Add/DONE 的 profiler 时间线分解（时间预算所限），G3 阶段补。
4. **小 M 生产路由价值有限**：M≤255 stock 仍最快，T1 接入模型路由前需评估端到端（Kernel launch 数、Graph 密度）收益；M=256 恰好跨过 HCCL 的算法分界（stock 0.43→0.52），V2/V1 在该点反超。
5. KFC 通道损坏依赖绕过；M>8192 多波、动态尾 M、Graph replay、同池多 stream 均未支持（沿 T0）。

## 8. 复现命令

```bash
# 编译（standalone）
bash /tmp/opencode/v2/build_host.sh          # 设备库 + host 库
# 官方路径
VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1 \
VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1 python setup.py build_ext --inplace

# 单卡 MM 隔离（1028 用例）
python csrc/_310P/memfabric_mm_ar_v2/t1_diag_single.py   # DIAG_PRESYNC=1

# 双卡矩阵 / 压力 / y-guard / stock（torchrun 2 proc，环境见脚本头注释）
torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/t1_nz_direct_trial.py --mode full
torchrun ... --mode stress
torchrun ... --mode yguard            # MF310P_V2_NOWARMUP=1
torchrun ... --mode stock

# T0 回归（/tmp/opencode/v2）
bash run_v2.sh g0
bash run_v2.sh g1 --chunks 256 512 1024 --reps 1000
bash run_v2.sh g2 --chunks 256 512 1024 --reps 100
# V1 基线
bash run_v1_bench.sh 1 2 4 8 16 32 64 128 255 256 1024 2048 4096 8192
```

原始日志：`/tmp/opencode/v2/{t1_quick_try*.log,t1_full*.log,t1_stress*.log,g0_final.log,g1_default.log,g1_final.log,g2_final2.log,v1_bench.log,repo_bench_quick.log,yguard2.log}`。

## 9. 后续代码审视更正（2026-10-18，待补跑）

本次报告中的 **quick/full 与单卡 1028/1028** 正确性结果不受以下问题影响；原始执行记录不作改写。

- 原 `t1_nz_direct_trial.py --mode stress` 在列表推导式内反复构造 `random.Random(20261017)`，导致所谓「随机 30 个 M」实际上只产生一个随机数，加上边界 `1,255,256` 共 **4 个 M**。原记录的 `160/160` 压力调用是这 4 个 M 的有效结果，但**不是** 30 个 M 的覆盖。
- 原连续 1000 次调用将 `mm/add/weight_seed` 分别绑定同一 `i % 4` 或 `i % 2` 周期，组合覆盖不足，不能据此声称已经充分覆盖所有分核和权重排列。
- 已更正测试脚本：确定性生成 **30 个不同 M**（包含 `1,255,256`）、独立抽取 `M/mm/add/seed`，并强制检查全部 M 与所有 `mm×add×4 seed` 组合实际出现。静态复核固定 seed 下 1000 次可覆盖全部组合；**新脚本尚待 310P3 实机复测，不能把原 PASS 自动转给更正后的测试**。
- G3 可靠性与 profiler 分解的下一步验收见 [G3_HARDENING_PLAN.md](G3_HARDENING_PLAN.md)。
