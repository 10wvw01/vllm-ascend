# T2 原生 MM + MemFabric AllReduce（不融合）实机验证、BUG 修复与性能对比报告

> 分支 `feat/310p-memfabric-public-api`，对象 `csrc/_310P/memfabric_mm_ar_v2/`（T2：原生 `F.linear` + 独立 AR-only 传输内核，与 V1/V2 融合路径并存；V1 与正式模型路由未改动）。
> 结论先行：**T2 正确性关口全部 PASS**（3 chunk 配置 × 16 M × 4 Add 核 = 192 组矩阵位精确 + status=0；变内容专项 2000+ 次调用位精确；压力 1000+300+300 次跨 3 chunk 全 PASS）。但**性能上 B（仅替换通信、不融合）在全部 M 档位稳定慢于 Stock（FP16 HCCL AR）**：正确实现 B 所必需的平台一致性代价（生产者→搬运同步 + 搬运内核）使其固定开销达 ~0.3 ms、大 M 时总耗时为 Stock 的 1.8–2.2 倍。**C（V2 融合）在 M≥2048 维持对 Stock 的优势（c1024 overlap：+3.4%/+17.9%/+22.8%）**。T2 的科学结论：MemFabric 传输的价值只能通过**算子融合**兑现，不能作为 HCCL AllReduce 的等价替换。
> 本轮共定位并修复 **7 项缺陷**（4 项为本路径新缺陷、1 项测试程序缺陷、2 项守护策略缺陷），全部有可复现实测证据；修复后 T0（G0/G1/G2）、T1（修正版 30-M stress）、V1（15/15）回归全部 PASS。

## 1. 环境与对象

| 项 | 值 |
|---|---|
| 机器 | dav-2002（310P3）×8，测试用 dev0/dev1（bus 18:00.0），ASCEND_RT_VISIBLE_DEVICES=0,1，TP=2 |
| CANN | 9.1.0（/usr/local/Ascend/cann-9.1.0，bisheng） |
| MemFabric | memfabric_hybrid 1.2.0 run 包（AICPU 内核 KVER=11，libmf_sdma_orch_v11.json）；dev0/1 KFC system() 通道仍损坏，建池走 assume pre-deployed 分支（与 T0/T1 相同；数据面由 G0 IB probe 与位精确结果背书） |
| 框架 | torch 2.10.0 + torch_npu 2.10.0.post4，vllm_ascend editable 指向本仓库 |
| 构建 | 官方路径 `VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1 VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1 python setup.py build_ext --inplace`，dav-2002 全量通过；AR-only + 新增 Stage 内核均编入 `vllm_ascend_C` |
| 被测 | `mf310p_v2_trial_device.asc`（新增 `V2MemfabricArOnlyKernel` + `V2ArStageKernel`）、`mf310p_v2_trial_adapter.cpp`（`mf310p_v2_trial_run_ar_only`） |
| 参照 | A=原生 `F.linear` + FP16 HCCL all_reduce（唯一收益基线）；C=V2 T1（M≤256）/ T0（chunk 倍数，overlap）已验证融合路径 |
| 正确性判据 | 参考 = `F.linear(x,w).float()` → HCCL all_reduce → `.to(fp16)`，B 路径**位精确**且协议 status=0；HCCL 参考全部在建池之前计算 |

**B 的最终数据流（较 T2_AR_ONLY_TRIAL.md 原稿有一处实现变更，见 BUG-3/BUG-6）：**

```text
原生 torch_npu F.linear(x, W_NZ) -> partial[M,2048] 普通 NPU Tensor
        [host 侧 stream synchronize —— 平台一致性必需代价，计入 B]
                     |
        V2ArStageKernel（AICore 8 核，MTE1 装载 + 向量 Adds 拷贝 + MTE3 存储
                        + MTE3_S fence + 逐行 CleanLines；牺牲发射 ×2 + 有效发射 ×1）
                     v
              symmetric ownC[M,2048] (ND)
                     |
        V2MemfabricArOnlyKernel（AR-only，addCores 核）
           core0: signal -> wait -> 放行 Add 核
           1/2/4/8 核: FP16 Add（逐行 CleanLines）
           core0: quiet -> peer DONE -> wait DONE
                     v
                y[M,2048] (ND)
```

原稿中的 `aclrtMemcpyAsync` D2D 搬运**被实测否定**（BUG-3：拷贝引擎读与 AICore 写不透明），替换为 AICore Stage 内核；D2D/stage 搬运仍是必须计时的真实代价，口径不变。

## 2. 本轮修复的缺陷（全部实测定位）

### BUG-1（adapter，守护策略）：AR-only 首启 MTE3 丢弃，stream-sync 治愈无效
- 现象：首启真实发射后 y 保留预填值（-777），status=0、协议完整——T0 BUG#3 同类。
- 实测：launch-stream synchronize **不能**治愈本内核（与 T1 内核不同）；`aclrtSynchronizeDevice` 治愈（STEP1 即位精确）。
- 修复：首启/新 y/新 partial 源地址统一 device-sync 守护。带池代价：随机 0–15 s/次（AICPU epoch），仅新地址触发。

### BUG-2（adapter，守护策略）：单地址 warm 缓存 × y 轮换 = 每调用 15 s
- 现象：压力脚本按 `i%4` 轮换 y 槽后，每调用 enqueue 耗时 15.1 s。
- 根因：守护仅记忆**单个**上次地址，任何轮换都被当作"新地址"触发 device-sync。
- 修复：y 与 partial 源各用 64 深度循环集合；稳态（已知地址）零同步。实测修后 30 个 stage 地址 + 4 个 y 槽全部命中集合，逐调用 <0.5 ms。

### BUG-3（设计缺陷，关键）：拷贝引擎（aclrtMemcpyAsync D2D）读取与同流 AICore 写不透明
- 现象：内容逐调用变化时（固定 x 的矩阵全部"通过"是**盲区**——旧读返回同一内容不可分辨），D2D staging 读到旧字节：F.linear 源 64/64 调用错、预写 copy_ 源 4/64 错（y 恰为上一调用参考值），status=0。
- 根因：310P3 本栈上拷贝引擎读取路径观测不到 AICore 仍驻留缓存的写（matmul/向量写）。对照实验：同流紧随 F.linear 的 torch `clone()`（向量读）320/320 位精确——向量/AICore 间经 L2 一致，拷贝引擎例外。
- 修复：废弃 D2D，新增 `V2ArStageKernel` 以 AICore 完成 staging（读侧 AICore 一致；写侧走官方 MemFabric 示例的 store+clean 纪律）。

### BUG-4（stage kernel，关键）：MTE1 装载张量直喂 MTE3 存储，缺 V→MTE3 交接
- 现象：stage 内核改为 MTE1 装载 + DataCopy 直存后，发送区 DDR 视图出现 **512B 粒度旧数据块**（OWNC_HOLE 探针，新增 `mf310p_v2_trial_read_send_arena` 同步 D2H 读回发送区证实），两 rank 位置完全相同，SDMA 如实搬运后 y 出现对称 512B 错区。
- 根因：`TQue<VECIN>` 的 `DeQue` 只发布 MTE1→V 就绪；无向量算子介入则无 V→MTE3 交接，MTE3 存储可在装载落地前读 UB，持久化上一 tile 的陈旧内容。
- 修复：以 `Adds(o, got, 0, n)` 向量拷贝经 `TQue<VECOUT>` 交接（与已验证的 AR-only Add 完全同构）。修复后 512B 错区消失。

### BUG-5（平台缺陷对策）：空闲窗后的首批 MTE3 存储发射被丢弃
- 现象：守护 device-sync（0–15 s）或其他 host 侧同步间隙后，紧随的 stage 单发射可整体不落地——两 rank 同时命中时 y **恰等于上一调用的参考值**（对称旧数据求和，两 rank 完全一致，是定位此缺陷的关键指纹）；M=1 时双发射亦有 1/128 概率双双丢弃。
- 修复：V1 的 launch-twice 纪律加固为**无条件 2 次牺牲发射 + 1 次有效发射**（同流覆盖，语义不变）。修后 8/8 轮 × 128 次 M=1 调用零失败。

### BUG-6（平台一致性代价，计入口径）：MTE1 装载可读到生产者内核仍驻缓存的旧值
- 现象：背靠背变内容调用中，M=1 专项 ~1/128 出现 `own=当前+peer=上一调用` 混合版本（两 rank 互补混合）；M≥257 未复现（工作集大，缓存行自然淘汰）。
- 根因：matmul 输出仍驻 L1/L2 时，MTE1 装载读到 DDR 旧值；向量读（clone 探针）无此问题。
- 修复：**B 路径在生产者（F.linear）与 staging 之间强制 stream synchronize**（平台一致性必需代价，计入 B 计时；融合路径 C 无此需求——其 MM 读的是早已稳定的输入）。8/8 轮 × 128 次 M=1 全部位精确；M=64 × 512 次零失败。

### BUG-7（守护必要性证明）：fresh 页 partial 源 staging 读到垃圾
- 现象：关闭 partial 源守护（`MF310P_V2_AR_ONLY_PARTIAL_GUARD=0`）后，21 种奇尺寸 fresh 分配 **42/42 调用全部数据错**（status=0）；开启守护 42/42 位精确。
- 结论：partial 源地址守护是**承重**的，保留。

### 设备环境问题（非代码缺陷，记录备查）
- **跨 rank 偏斜 × 28 s aicore 看门狗**：c256 首跑在 M=8192 发生 aicore timeout（507014）——一 rank 连续多个 0–15 s 守护同步期间，对端内核核内邮件等待超 28 s 被击杀，池报废。缓解：bench 改为每 M 单一 y（守护同步次数 ÷4）后复跑通过。残余风险：守护 device-sync 的随机时长仍是已知风险（沿 T1 报告遗留问题 2），生产化前必须消除 device-sync 依赖。
- KFC 控制面损坏沿 T0/T1 绕过（assume pre-deployed）；数据面（IB probe、位精确 192+ 组）健康。

## 3. 正确性结果

### 3.1 全矩阵（最终代码，每配置首调用含守护同步、逐配置前后位精确校验）

| chunk | 覆盖 | 结果 |
|---|---|---|
| 1024 | M={1,2,4,8,16,32,64,128,255,256,257,512,1024,2048,4096,8192} × Add={1,2,4,8}（repeat 30）+ fused 参照 | **78/78 PASS，0 FAIL** |
| 512 | 同上 | **79/79 PASS，0 FAIL** |
| 256 | 同上（首跑因 BUG-环境 aicore timeout 中断，缓解后复跑） | **79/79 PASS，0 FAIL** |

多 chunk（M=2048/4096/8192 × c256/512）、尾块边界（M=1/255/256/257）、M=2 等 HCCL 敏感形状全覆盖，两 rank 输出与 FP32 参考位精确、协议 status=0。

### 3.2 变内容专项（固定 x 矩阵的盲区补测，全部逐调用位精确）

| 专项 | 结果 |
|---|---|
| M=1, add=1, 32 变体 × 4 轮 × 8 独立进程（含 BUG-6 修复后） | **1024/1024 PASS** |
| M=64, add=8, 64 变体 × 8 轮 | **512/512 PASS** |
| M=257 / 1024 / 8192 变体轮换 | 96/96、48/48、24/24 PASS |
| fresh 源扫掠（21 奇尺寸，守护开启） | 42/42 PASS（对照关闭守护 0/42，见 BUG-7） |
| 纯 torch 对照（F.linear 紧随 clone） | 320/320 位精确（定位 BUG-3/6 的对照轴） |

### 3.3 压力（t2_ar_stress.py：30 个不同 M × 4 x-seed 轮换 × Add 轮换 × 4 y 槽复用 + 每 97 次 fresh y + 有界 rank 偏斜 0–100 ms；逐调用 status=0 + 位精确）

| chunk | 次数 | 结果 |
|---|---|---|
| 1024 | 1000 | **PASS**（unique_m=30，adds={1,2,4,8}，fresh_y=10，602.4 s） |
| 512 | 300 | **PASS**（unique_m=30，fresh_y=3，496.1 s） |
| 256 | 300 | **PASS**（unique_m=30，fresh_y=3，496.7 s） |

覆盖断言全部满足：30 个 M 全部出现、Add 全档出现、fresh-y 调用实际执行、无死锁、无 aicore 超时、无 status≠0。

## 4. 性能对比（三会话，max-rank median / P95，ms）

统一口径：A/B/C 同 stream enqueue→sync host 计时；B 含 BUG-6 强制同步 + 全部 staging 代价；首调用（含 0–15 s 守护同步）单列不混入稳态；两 rank 取较慢者。

### 4.1 主表（chunk=1024 会话；B 为 4 种 Add 核最优）

| M | A: Stock | B: MM+MemFabric(stage) | B 最优 Add | (A−B)/A | C: V2 融合 | (A−C)/A | 最优方 |
|---|---|---|---|---|---|---|---|
| 1 | **0.3254** | 0.6249 | 8 | −92.0% | 0.3179 | +2.3% | C（微弱）/Stock |
| 2 | **0.3238** | 0.6243 | 8 | −92.8% | 0.3186 | +1.6% | Stock≈C |
| 4 | **0.3285** | 0.6252 | 8 | −90.3% | 0.3200 | +2.6% | Stock≈C |
| 8 | **0.3288** | 0.6247 | 8 | −90.0% | 0.3182 | +3.2% | C（微弱） |
| 16 | **0.3403** | 0.6246 | 8 | −83.5% | 0.3191 | +6.2% | C |
| 32 | **0.3362** | 0.6209 | 1 | −86.0% | 0.3181 | +5.4% | C |
| 64 | **0.3326** | 0.6228 | 8 | −87.3% | 0.3176 | +4.5% | C |
| 128 | **0.3475** | 0.6251 | 8 | −79.9% | 0.4640 | −33.5% | Stock |
| 255 | **0.3720** | 0.7790 | 4 | −109.4% | 0.4724 | −27.0% | Stock |
| 256 | **0.3925** | 0.7793 | 8 | −98.6% | 0.4717 | −20.2% | Stock |
| 257 | **0.3876** | 0.7786 | 8 | −100.9% | — | — | Stock |
| 512 | **0.6059** | 1.0866 | 8 | −79.3% | 0.6336 (c512) | −4.6% | Stock |
| 1024 | **0.7828** | 1.7090 | 8 | −118.3% | 0.9451 (c1024 serial) | −20.7% | Stock |
| 2048 | 1.4778 | 2.9671 | 8 | −100.8% | **1.4269** | **+3.4%** | **C** |
| 4096 | 3.2735 | 6.2442 | 8 | −90.8% | **2.6875** | **+17.9%** | **C** |
| 8192 | 6.7529 | 12.4765 | 8 | −84.8% | **5.2126** | **+22.8%** | **C** |

（B 的 P95 见附录原始矩阵；Stock P95 ~median+5%；B P95 ~median+1%——B 波动小但基线高。）

### 4.2 chunk 敏感性（B 与 C 的最优 chunk 均为 1024）

| M=8192 | c1024 | c512 | c256 |
|---|---|---|---|
| B (add8 median) | **12.48** | 13.55 | 14.76 |
| C (T0 overlap) | **5.21** | 6.02 | 6.92 |

### 4.3 三会话 Stock 稳定性

M=1: 0.325/0.371/0.34（±0.05）；M=8192: 6.75/6.73/6.75。早前会话中 M=2（0.55–0.70）与 M=257（0.62，1/3 会话）的 Stock 慢化为 HCCL FP16 AR 形状敏感的偶发（最终三轮会话未复现 0.6+ 的 M=257；M=2 在最终会话 0.324）。**B 在任何会话、任何 M 均未稳定低于 Stock**；C 在 M≥2048 的三会话均稳定低于 Stock（+2.4%~+22.8%）。

### 4.4 结论

1. **B（仅替换通信）无性能收益**：正确实现所需的 host 同步（~0.1–0.3 ms）+ AICore staging 内核（含 fence/clean）使 B 固定开销 ~0.3 ms，小 M 即被 Stock 反超 2 倍；大 M 时 staging 搬运 + 串行 chunk 协议使总时延为 Stock 的 1.8–2.2 倍。此前无守护/无同步版本的 B 数字（0.47 ms @M=1、8.75 ms @M=8192）**不可作为正确实现的收益依据**。
2. **C（融合）的收益真实且稳定**，但仅限 M≥2048（c1024 overlap）；M≤1024 Stock 最快。与 T0/T1 报告一致。
3. T2 的核心问题（把 MemFabric 传输价值与算子融合价值拆开）答案明确：**传输本身不构成收益来源，融合（消除 D2D/搬运与 host 同步、SDMA-Add 交叠）才是**。

## 5. 静态代码审视（两轮）

**第一轮（AR-only 内核 vs T0 协议逐行对照）**：chunk 循环 done-wait 全序（core0 不会超前慢 worker）；行交错每行 4096B/64 cache line 单核独占；与 T0/T1 tag（MF2AR vs MF2V）不混扰；DONE+quiet 跨调用保护对称区；邮件收支逐调用平衡。结论：内核协议无逻辑缺陷（后续全部实测问题均在守护/平台一致性层）。

**第二轮（最终态全文件审视，修复后）**：Stage 内核 UB 预算（4×2KB 队列）远低于 256KB；切片 512B 对齐、尾块 256 元素粒度满足向量算子约束；`Adds(x,0)` 的 −0.0→+0.0 符号位翻转经 `torch.equal`（IEEE ±0 相等）判定不影响验收判据；MTE3_S fence 用 `FetchEventID`（避免与队列事件 ID 冲突）；launcher 参数校验与既有内核一致；adapter 死代码（guard_fired/ar_stage_launched）已清除。结论：无阻塞项。

## 6. profiler 分解（torch_npu profiler chrome trace，设备侧 async 事件，rank0）

### M=8192（chunk=1024，add=8，B 每 callers 调用 = 3 次 stage + 1 次 AR 内核）

| 设备侧事件 | 次数/调用 | median | 小计/调用 | 占 B 总时延 |
|---|---|---|---|---|
| MatMul7（F.linear 主内核） | 1 | 3.90 ms | 3.90 ms | 31% |
| V2ArStageKernel（staging，2 牺牲 + 1 有效） | 3 | 1.31 ms | **3.92 ms** | 31% |
| V2MemfabricArOnlyKernel（SDMA+Add+协议） | 1 | 3.99 ms | 3.99 ms | 32% |
| trans_TransData_2（MM 内 NZ 转置）+ memset 等 | — | — | ~0.47 ms | 4% |
| **B 设备侧合计** | | | **≈12.3 ms** | 与 host 实测 12.48 ms 吻合 |

对照 Stock（A）M=8192：MM ~3.9 ms + HCCL allReduce 内核 ~2.8 ms ≈ 6.7 ms。**B 相对 Stock 的两项净亏**：staging 搬运 +3.92 ms（HCCL 原地读取 MM 输出，零搬运）+ AR 内核比 HCCL 慢 ~1.2 ms（串行 chunk 协议、无 overlap）。staging 中 ~2.6 ms（2/3）是平台缺陷对策的牺牲发射——即便免费，B 仍慢 ~1.3 ms。

### M=64（chunk=1024，add=8）

| 路径 | 设备侧构成 | host 实测 |
|---|---|---|
| A: Stock | MM 18 us + HCCL AR 115 us | 0.33 ms |
| B | MM 18 us + stage 3×11 us + memset 31 us + AR 内核 217 us；其余为强制同步与发射开销 | 0.62 ms |

AR-only 内核 217 us 中绝大部分为协议固定开销（1 chunk signal/wait + quiet/DONE 两轮 AICPU 往返，单次邮件往返 ~50–100 us），与 T1 报告的协议固定成本结论一致；M=64 时 B 的固定开销（~0.3 ms）已超过 Stock 整条 AR 链路。

## 7. 回归（修复后，2026-10-18 上午实机）

| 套件 | 内容 | 结果 |
|---|---|---|
| T0 G0 | store/arena 基础 ABI | **ALL PASS** |
| T0 G1 | chunk 数据面（256/512/1024 × 1000 reps） | **PASS**（7 组） |
| T0 G2 | T0 融合端到端（256/512/1024） | **PASS**（19 组） |
| T1 stress（修正版） | random_m 1200 + seq_1000（**30 M 全覆盖**，mm×add×seed 32 组合） | **PASS**（同时关闭 T1 报告 §9 遗留项） |
| V1 层级 bench | rows 1–8192 全档位 | **PASS 15/15**（位精确；V1 路径不受本轮改动影响） |

V1 首跑曾因 `LD_LIBRARY_PATH` 中旧版独立构建的 `libmf310p_v2_trial_device.so` 遮蔽（RUNPATH 次序）报 undefined symbol，清理环境后 15/15 通过——非代码缺陷，记录为环境注意事项。

## 8. 遗留风险与后续

1. **平台一致性三缺口**（BUG-3/4/6）为实测定性，未获官方硅级解释：拷贝引擎读不透明、MTE1 读可滞后、空闲窗首批 MTE3 存储丢弃。任何后续使用普通 Tensor 作 SDMA 源或 D2D 中转的设计必须先行验证；上游修复前 B 不可生产化。
2. **守护 device-sync 依赖**：0–15 s 随机 + 跨 rank 偏斜可击穿 28 s 看门狗（c256 首跑实测一次）。与 T1 遗留问题同源，Graph/生产化前必须以输出缓冲预注册/预写替代（G3 计划 B.1）。
3. B 的 host 同步使其退化为半同步管线；若未来平台修复一致性缺口，可重测无同步 B。
4. KFC 控制面损坏沿绕过（assume pre-deployed）；健康设备复验仍待 G3-2。
5. C 的收益窗口（M≥2048, c1024 overlap）与 T0/T1 结论一致；真实模型路由按 M 阈值混合路由的评估属 G3+。

## 9. 复现命令

```bash
# 构建
VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1 \
VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1 python setup.py build_ext --inplace

# 环境
source /usr/local/memfabric_hybrid/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/latest/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581

# 全矩阵（×3 chunk）
torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/bench_ar_only.py \
  --rows 1 2 4 8 16 32 64 128 255 256 257 512 1024 2048 4096 8192 \
  --chunk-rows 1024 --add-cores 1 2 4 8 --repeat 30 --fused

# 压力（30 M × 4 seed × 动态 y × 偏斜）
torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/t2_ar_stress.py --chunk-rows 1024 --repeat 1000

# 变内容专项（M=1/64/257/1024/8192，源新鲜度决定性实验）
torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/t2_freshdata.py --m 1 --calls 1024

# profiler 分解（chrome trace，设备侧事件）
torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/t2_prof.py --m 8192 --add 8 --chunk-rows 1024 \
  --iters 20 --out /tmp/prof8192

# 一次性平台探针（源新鲜度扫描 / 隔离，未入库）：/tmp/opencode/t2/
#   t2_diag2.py --mode srcsweep  t2_isolate.py
```

原始日志：`/tmp/opencode/t2/`（t2_matrix_c{1024,512,256}.log/.json、t2_stress_c*.log、freshdata/final、isolate/srcsweep 系列、reg_*.log、prof_*）。
