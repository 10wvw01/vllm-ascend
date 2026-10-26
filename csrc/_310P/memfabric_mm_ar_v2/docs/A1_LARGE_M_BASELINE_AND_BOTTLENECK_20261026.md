# 主线 A｜V2 大 M 完整算子 Stock 基线重建及高价值瓶颈归因 A1（2026-10-26）

> **开发工作树：**`work/310p-mm-ar-large-m-a1-20261026`，起点为唯一主线 `dev/310p-memfabric-mm-ar@7dc8ad465ce1dc6400ace3f11df7335b97411172`。  
> **范围：**只对 Ascend 310P3 单卡双Die TP2 的 FP16 MM + MemFabric SDMA AllReduce + Add 与原生 FP16 MM + HCCL AllReduce 的完整调用做测量、离线归因、下一轮算子优化决策。**不改 Kernel、MemFabric 协议、LinearMethod、Stock 默认、不建新 Runner 或 DFlash 兼容栈。**  
> **截至撰写：**本轮尚未在310P3执行新测量。历史数仅用于提出诊断假设，**新一轮 M2048/4096 严格完整 Stock vs 最新 V2 的性能差距仍 UNKNOWN**。

## 1. 为什么先重建完整 Stock 基线，而不是继续改 Add/Chunk

现有数据来自三种不同条件，**不得直接拼成当前差距**：

| 取证批次 | 对比口径 | 可复述的实测值 | 用途与限制 |
|---|---|---|---|
| 2026-10-21 真实 Prefill profiler | 同一历史 Serving 工况下 Stock MM≈795μs，HCCL AR≈1149μs；V2 kernel≈2573μs；新增输入 contiguous≈66μs/站点 | 当时同工况可见 Stock 两段总约1944μs，V2 更慢；40站点/步累计劣势 | 反映真实业务站点，但受Profiler、Tensor布局与跨流时序影响；**不是今天Tile2048的新实测** |
| D4 Add ABBA (2026-10-23) | V2内部Tile1024 vs Tile2048，M4096/Chunk1024/Overlap1 | 双Rank Event p50 **2726.9→2556.3μs**，内部降幅6.26% | 只证明同条件V2优化；**D4未用原生Stock做同期完整ABBA** |
| SEL-01 模型 A/B | 输入4096/输出2048/TP2/并发10 Eager | Stock约45.98、V2约46.49 tok/s（+1.12%，不稳定） | 端到端方向性，不是单算子相位分解，不能用于精确倒推出硬件关键路径 |

**A1要回答**：在**同卡对、同热态、同M/K/N/FP16权重、无Profiler**时，最新D4正式V2完整算子，相对于Stock FP16 F.linear+HCCL AR究竟差/快多少μs？差距是否来自 MM效率、SDMA等待/IB串行、布局拷贝，还是基线测量遗漏HCCL异步尾部？

使用U1既有工具，拒绝另写或绕过首写/KFC/可见性屏障的“快测”Kernel。

### 历史硬件计数器：优先级不是凭空猜测

2026-10-21 的**同一真实Prefill Profiler（Rank0 p50）**提供了进一步、但仅属于旧版本的归因证据：

| 硬件计数器 | V2TrialKernel | Stock MatMulV3 | 对新一轮A1的启示 |
|---|---:|---:|---|
| Cube mac_time | **458.9 μs** | **456.6 μs** | 纯数学时间基本持平，**MM并非首要怀疑对象** |
| MTE2 time | **1711.4 μs** | **335.9 μs** | 优先追查SDMA数据到达等待、缓存失效后搬入和MM重叠不足 |
| Scalar time | **872.8 μs** | **186.5 μs** | 优先追查Core0 Mail/Signal轮询、IB会合、跨Chunk控制开销 |
| MTE3 time | **191.1 μs** | **91.5 μs** | 待核查输出规约写回与必要的缓存操作 |
| Vector time | **127.4 μs** | **194.9 μs** | 单纯增加Add算力很难解决更大的通信等待问题 |

**硬约束：**这些是可能重叠的设备计数器，不能相加/相减形成一份精确的Kernel关键路径分解；也不能把旧Profiler的MTE2 1711 μs直接写成最新Tile2048的“SDMA传输时间”。它们用于给后续诊断排序，**现版完整算子Stock ABBA结果仍为UNKNOWN**。旧报告同时显示每站点约66 μs input.contiguous开销，属于模型薄接入层的次级优化机会，不是Kernel内部的设备MTE2阶段。

## 2. A1的实际新增代码

- `test/perf/a1_large_m_assess.py`：**纯stdlib离线验收与决策工具**。只重新读取原始U1屏幕/Confirm目录，由现有 `u1_analyze.evaluate()` 逐文件校验 manifest、Rank0/1原始样本、时序、二进制SHA、双Rank配对与路由。拒绝仅凭手抄汇总表宣布GO。检查屏幕和确认的Git HEAD、物理卡号及四类SHA身份一致。
- 输出 `A1_LARGE_M_FULL_OPERATOR_V1` JSON：按 M1024/2048/4096给出完整Stock与V2配对p50、V2额外延迟μs、相对于Stock的延迟比率与等效速率差、来自Screen还是Confirm、独立HCCL/路由审计是否已经齐备；输出下一步诊断队列。**不生成任何新增设备性能数据**。
- `test/ut/test_a1_large_m_assess.py`：合成数据的纯CPU数学/非法证据/门禁回归；合成耗时仅在测试局部使用、不得写入性能账本。
- 这不是对U1的小M+大M整组Campaign的简单重复：A1**只运行大M**，优先重建M2048与4096完整Stock基线；M1024只用于探针/实验形状，正式Serving阈值`2048≤M≤4096`，严禁将它计为命中收益。

### 离线命令（没有设备也可执行）

```bash
# 需先在工作分支检出A1全部文件；无需torch、torch_npu或CANN
python3 -m unittest discover -s csrc/_310P/memfabric_mm_ar_v2/test/ut -p 'test_a1_large_m_assess.py' -v
python3 -m py_compile csrc/_310P/memfabric_mm_ar_v2/test/perf/a1_large_m_assess.py
# 拿到已经执行并归档的真实U1大M screen后：
python3 csrc/_310P/memfabric_mm_ar_v2/test/perf/a1_large_m_assess.py \
  --screen-root /绝对路径/large_screen --out /全新绝对路径/a1_screen_review.json
# 如独立完成双Rank ABBA Confirm且已经有HCCL跨流/路由审计：
python3 csrc/_310P/memfabric_mm_ar_v2/test/perf/a1_large_m_assess.py \
  --screen-root /绝对路径/large_screen \
  --confirm-root /绝对路径/large_confirm \
  --route-audit /绝对路径/large_confirm/independent_route_audit.json \
  --out /全新绝对路径/a1_confirm_review.json
```

若仅有screen，JSON **必须**标记`SCREEN_ONLY_NOT_GO`；若Confirm没有audit也只能`CONFIRM_REVIEWABLE_NOT_MODEL_GO`且每形状需要独立路由审计；即便完整L1认证也**不是**真实模型GO。工具仅处理隔离完整算子，无法量化Serving `x.contiguous()`和Graph capture/replay。

## 3. 实机设计：复用原U1；不自动进入设备

### G0（**先独立查验，未GO不得继续**）

1. 当前服务器`git HEAD`、分支及clean tree；确认现场只从此次A1代码工作分支获取代码，**严禁从废止的R2 Runner分支Cherry-pick**。
2. 真实物理双Die映射与独占、KFC/SMEM启动和协议；准确认出Python已加载插件SO、MemFabric Host双路径SO、D4设备SO与设备侧AICPU/KVER11。旧U1脚本对SO/JSON有**精确SHA约束**，发现本机合法升级但SHA不符时需先重新评审版本并更新合同，不得为了让测试跑过直接跳过或伪造SHA。
3. 先备份可恢复的Host/Device SO、KVER身份，记录原位置和SHA，确认恢复命令以及异常后停测路径。严禁`MF_KFC_BYPASS`、`MF310P_V2_NOWARMUP`、`MF310P_V2_AR_ONLY_NOWARMUP`。
4. 310P服务器上实际执行 `bash -n`、`python3 -m py_compile`、`unittest/pytest`，且同时确认`U1`已有测试/版本门禁能通过。
5. G0证据未满足不执行`torchrun`。任何507014/507899、位精确失败、丢样本、错Rank、SO身份漂移立即停止并保全日志；不得自动重试产生“幸存者偏差”。

### D0：最省现场时间的安全冒烟与快速基线

通过G0且取得用户许可后，严格**使用已存在**的 `test/perf/u1_run.sh`：

```bash
# 由OpenCode根据现场实机G0证据设置设备/库路径以及批准标记；
# 下列命令是执行次序，不是直接授权或自动运行指令
PERF=csrc/_310P/memfabric_mm_ar_v2/test/perf
bash "$PERF/u1_run.sh" /tmp/opencode/a1_large_smoke_01 large smoke
# 人工检查Stock/D4各一场M2048双Rank位精确、status与前后设备健康，
# **两个Smoke成功并审核后**：
bash "$PERF/u1_run.sh" /tmp/opencode/a1_large_screen_01 large screen
# 产生Screen真实样本后执行纯CPU A1工具；Screen不能推出性能GO
```

U1一次`large screen`分别以原生FP16 Stock（独立HCCL进程）与正式D4 Kernel（MemFabric进程）测试`M=1024,2048,4096`。Stock对照完整链是本Rank `F.linear(fp16)` + `all_reduce(fp16)`，Fusion是正式`mf310p_v2_trial_run`，K=N=2048、TP2、Chunk1024/Overlap1。真实时序配对时每样本先取`max(rank0[i],rank1[i])`再聚合，不允许相减两Rank的p50或把“Stock MM-only”误称为HCCL耗时。

**D0结果决策**：正向、负向、无明显收益都继续大M研发；仅用它**选择高价值问题**，不会因为ROI小便结束主线。必须保留每M两个Rank日志和SO/manifest。

### D1/D2：只在具备证据时升级

- **D1诊断**：M4096正式4Chunk overlap与4Chunk loop4串行对照；所有诊断独立进程、每Case先位精确，再看延迟方向，**不允许用两条不同内核的p50直接相减去声称重叠节省了Xμs**。
- **D2正式**：独立路径 `u1_run.sh <新目录> large confirm`，8会话Stock/D4 ABBA ×2，每会话每M5预热+40双Rank样本；除5%每轮收益、p90/p99和热态漂移门槛外，还需单独证明Stock HCCL跨流Event真的覆盖所有通信与MemFabric路由真实性。**不自动启动、单独向用户申请实机许可**。
- **D3 Profiler**：只有D1定位具有价值时开启**独立短窗口**，取设备端TimeLine区分MM、跨Die SDMA发起/等待、IB同步、Add、final DONE；不能用Profiler污染主ABBA。若采样不支持精确阶段拆解，写`PHASE_TIMING_NOT_OBSERVABLE`，严禁凭假设臆造分段μs。
- **D4 Serving模型回传**：现成Stock+DFlash路径仅以融合LinearMethod薄接入，锁定输入4096/输出2048/TP2/batch10/50请求/K7/Eager。需要真实模型侧每目标层M分布、融合命中、input contiguous copy调用次数及计时证据，才能用既有`u1_e2e_budget.py`估算乐观上界。上界不是实测吞吐，绝不复活ModelState/Runner重建任务。

## 4. 高价值瓶颈诊断队列

以下排名是**按既有代码路径和历史证据设置的优先验证顺序，不是已测各阶段耗时**。

| 顺序 | 优化疑点 | 静态依据 | 如何实证和止损 |
|---|---|---|---|
| 测量前置GATE | Stock HCCL设备计时/跨流Event覆盖 | 早期W1/W4单算子基线与真实Serving的Stock表现差别大；旧U1也将其列为硬门槛 | 用独立HCCL Stream Timeline核实Event真实完整包围 MM→AR→完成；未闭环前不给“V2快于Stock”结论 |
| **优化P0** | 4Chunk MM 与SDMA/接收等待交叠不足 | 正式核每Chunk MM之后需要Core0 signal、前一Chunk wait、IB Released/Done，可能有不可重叠段；低占空的SDMA不能等于无需等待 | 经独立正确性验证的formal/loop4对照与Profiler；特别看Core0 recv wait期间其他Core是否已空转 |
| P2 | MM计算效率与权重NZ格式 | 正式V2用8核、Chunk1024，较原生F.linear的tiling行为不同 | 计算与通信分开取证，同输入布局/热态对照MM-only诊断，仅作方向判断，不对不等价时间差分 |
| P1 | `input.contiguous()`与输出Tail阶段 | 旧Prefill报告每站点约66μs附加copy，实际Serving目标层可能不是连续输入 | 真实Serving per-layer事件/拷贝与融合次数，考虑通过薄接入避免可证明冗余的布局转换；绝不直接改DFlash模型本体 |
| **优化P0/P1** | 协议Scalar/IB控制与Final DONE；Add/DCCI待确认 | Add Tile2048内部优化已实现；仍有串行写回/控制成本 | 按需添加**隔离、已审**精确埋点，不引入未认证多核/小粒度写；不再盲扫256/512 Chunk |

真实量化后再决定改：如果P0主要来自Core0 wait，应优先在不违反SDMA/缓存可见性协议下改善交叠；如果MM主导，则调整Cube tiling与布局；如果主要来自Serving拷贝，则限于LinearMethod薄接入。**任何改动都要单变量、Stock完整链同卡ABBA、双Rank首写/IB/超时风险回归、可回滚。**

## 5. 两轮静态审查记录（不冒充CI执行）

**审查R1：原始测量与统计方法。** 与主线 `u1_spec.py/u1_arm.py/u1_analyze.py/u1_run.sh` 对照，确认Stock计时实际执行FP16 `F.linear + HCCL AllReduce`、V2执行正式D4算子；两个数据面在不同Worker会话；U1按样本双Rank`max`配对，Screen和Confirm隔离，Confirm要求ABBA×2/独立路由审计。发现Stock HCCL任务可能使用内部Stream，因此**现有设备Event覆盖仍待现场证明**，代码不会擅自给操作级GO。新增A1 `M1024`只作诊断，不允许计入Serving阈值收益。

**审查R2：源码改动隔离、证据不可混算和错误门禁。** GitHub变更清单仅含A1纯CPU分析/UT与项目文档；不修改`src/`、`vllm_ascend/_310p/ops/`、Stock默认、DFlash Runner或MemFabric KVER协议。A1读真实原始U1目录时先调用既有`evaluate`复验Rank日志与SHA，Screen只能产生`SCREEN_ONLY_NOT_GO`；Confirm独立audit缺失时`L1_complete_op_confirmed=false`；所用SO身份、Git SHA和物理卡必须一致；历史Profiler计数器不填入当前M的耗时字段；阶段诊断`phase_duration_us`恒为`null`，等真实设备Profiler取证。

**未完成的独立动态证明**：截至本次静态审查，GitHub新PR Actions与本机Python `unittest`/语法检查尚未取得成功记录，**不能给A1 CPU-TEST-PASS或STATIC-GO**。正式设备G0/Smoke/Screen/Confirm/Profiler均NOT_RUN。下一轮只需补真实CPU合同PASS和服务器只读G0身份便可提出分阶段设备准入；绝不能直接跳到M4096 ABBA。

## 6. 当前阶段结果、准入与交付

| 检查 | 2026-10-26状态 |
|---|---|
| 废止R2 Runner重建 | **PASS**：以ADR-013为准，此任务没有Runner/ModelState相关改动 |
| 已有U1设备测试框架复用 | **静态检查完成**，没有重造Stock/Fused设备计时 |
| A1大M离线工具、纯stdlib合同测试 | **源码已提交到A1工作分支；CI/本地执行结果以实际运行记录为准** |
| D4正式大M Kernel代码 | **零改动** |
| 当前M2048/4096真实Stock/V2完整算子时延 | **NOT_RUN / UNKNOWN** |
| U1跨流HCCL Event route独立取证 | **NOT_RUN** |
| 310P G0、Smoke、Screen、Confirm、Profiler | **NOT_RUN** |
| Stage P0 既有507014/静默首写/双Die死锁 | **生产阻塞仍在；可受控隔离研究，不得Production-GO** |

**本次允许的下一动作：**完成A1纯CPU测试、两轮静态审查、独立G0清单审核；在明确发出`STATIC-GO for G0/CPU only`之前，用户不用委托OpenCode做性能测试。后续会根据G0身份、安全、恢复证据分段向用户请求实机操作，禁止一次性压测跳过冒烟。
