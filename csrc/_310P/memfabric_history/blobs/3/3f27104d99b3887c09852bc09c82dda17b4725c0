# F2｜310P3 单卡双 Die：大 M MM+MemFabric SDMA+Add 重叠优化首轮

> 2026-10-24 · **F2 已启动：独立 GitHub 实验分支、可执行TP2性能扫描、成对统计与CPU合同测试已提交；尚未在310P3实机执行。**  
> 分支：`perf/310p-f2-mm-sdma-overlap-sweep-20261024`。基线：`perf/310p-v2-d3-add-tile2048-20261023@33e3f2cfc81e`，即已验证的D4 Add Tile2048。  
> **Stock（MM+HCCL）仍是默认产品路线；不触碰正式Serving分发，不引入C3 CleanLines NO-GO改动，不与F1混合二进制。**


> **2026-10-24 实机准入更新（必须先读）**：完整 G0 前置检查、D4设备SO身份、dev6/7物理设备号修复、诊断ABI、CPU测试、超时与失败处理规则以同分支 [F1/F2强制准入与执行规程](./F1_F2_ONDEVICE_PREFLIGHT_GATES_20261024.md) 为准。旧示例卡号、SO路径**不能直接照抄**。本分支截至此提交仅完成源码审视/检查规则审计，GitHub CI 仍排队，**并无310P3现场G0或pytest PASS证据；状态是PRECHECK-GO，性能采样待G0全部PASS才允许开始。**

## 一、目标和边界

研究在 **310P3单卡双 Die/TP2** 上，如何进一步减少单个 `MM [M,2048]×[2048,2048] + AllReduce` 融合操作的真实完成时长。核心不仅仅是MM、SDMA或Add各自的速度，而是**两 Rank 的最后一个完成点**：哪部分计算、通信等待或规约站在关键路径上。

当前证据：D4 Tile2048 的单算子提升6.26%，C3 CleanLines循环展开只得+1.70%且不稳定，C3正式NO-GO；T0/T2在M>=2048曾见融合优于Stock，但不同构建/卡对和真实Serving里有明显回归。故不能拿历史T0/T2算子数字直接给F2新候选判GO。

**本轮F2的第一步是低风险参数探索而不是猜测性协议改造。** 正式D4 Kernel已支持 `chunk_rows∈{256,512,1024}` 和 `overlap∈{0,1}`，所以无需改动任何MM、signal、wait、IB、DCCI、DONE代码，就可分辨“更多小块”是否给通信隐藏带来大于额外握手开销的收益。

## 二、F2-0 源码依赖审查：现有同步边界

```text
8核 MM(chunk) 完成
       ↓
每核 ownC CleanLines（结果供对端SDMA读取）
       ↓
WaitReady（8→1，Core0只在全核就绪后发信号）
       ↓
Core0 signal_at → 对端异步SDMA
       ↓
如果 overlap=1：下一块的MM可能与上一块SDMA并行
       ↓
Core0 wait_at；校验邮件 dst、len、imm
       ↓
ReleaseAdd（1→7）；每核失效已被对端SDMA写入的 ownPeerC
       ↓
Vector Add→输出
       ↓
所有块完成→WaitAddDone→Core0 quiet→双方 DONE 交接
```

**不能将 CleanLines、WaitReady、wait_at、quiet、DONE随意移到错误的依赖一侧。** 原协议曾有507014偶发挂死，本轮只比较已经存在且经过D4正确性验证的 overlap=0/1 分支，任何新增挂死均作为异常立即保全、禁止自动重跑混数。

## 三、F2-1 实机参数扫描（优先执行）

| M | Chunk候选 | Overlap | 比较基准 |
|---|---|---|---|
| 2048 | 256 / 512 / 1024 | 0 / 1 | 同一M的D4 C1024/O1 |
| 4096 | 256 / 512 / 1024 | 0 / 1 | 同一M的D4 C1024/O1 |

单卡两Die固定N=K=2048、FP16、NZ权重，8 AICore全部做MM且Core0参与MM并负责signal；Add Tile固定2048；运行时KVER11/Host SO/Device SO、CANN版本和卡对锁定。

F2仅扫描参数，绝不修改`mf310p_v2_kernel.asc`。测试目录：

- `test/perf/f2_chunk_overlap_perf.py`：从D4正式验证驱动派生；可指定M、Chunk、Overlap；每次**双Rank逐样本位精确**、status0、3次预热、30次NPU Event/Wall采样，错误即停。
- `test/perf/f2_run_sweep.sh`：先检查设备独占、KVER11、SO指纹及不允许绕过首写保护；在同一个设备对上串行执行18场，每M穿插多次C1024/O1参考锚。
- `test/perf/f2_sweep_analyze.py`：样本索引匹配Rank0/1后取`max(event_r0,event_r1)`，不得用`max(p50_r0,p50_r1)`代替；异常样本不删除；按M与Chunk/Overlap输出排名。

**执行前置：** OpenCode先读报告、确认选定卡对确实是同卡双Die，设置并校验以下环境；不得直接照抄卡号/Host SO路径：

```bash
git switch perf/310p-f2-mm-sdma-overlap-sweep-20261024
python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_f2_sweep_contract.py
# source CANN/MemFabric set_env.sh，并在310P3确认实际设备和部署：
export ASCEND_RT_VISIBLE_DEVICES=6,7  # 示例：按实机独占情况替换
export MF_SDMA_ORCH_JSON=/path/to/libmf_sdma_orch_v11.json
export F2_DEVICE_SO_PATH=/path/to/actually_loaded/libmf310p_v2_trial_device.so
export F2_EXCLUSIVE_APPROVED=YES       # 仅经OpenCode检查通过后设置
bash csrc/_310P/memfabric_mm_ar_v2/test/perf/f2_run_sweep.sh /tmp/opencode/f2_sweep_001
```

先进行**独立形状正确性**再跑18场计时；必须记录实际加载的设备SO与`sha256sum`对应。dev6/7如需物理设备号修复的Host SO，需按D4 precedent先统一部署、核验后测试结束恢复，严禁一场切换Host SO导致可比性失效。不能通过更换插件SO隐藏错误。

## 四、F2-2 以源码已知依赖为起点，做有目标的设备阶段归因

网格数据只说明`Chunk×Overlap`最终耗时的关系，**不能证明具体在等哪封信**。只有以下条件满足才实施下一项诊断（独立测试代码，不能给正式核添加常驻Profiler）：

1. 对相同M和同一SO比较C1024/O1 vs 排名前列配置的双Rank时间线；核对`IterateAll`后清缓存、Core0 signal、对端邮件、Add、quiet/DONE 的设备证据，给出关键路径**真实等待空档**而非凭代码段计数臆测。
2. 若C256/512相对C1024明显更慢：优先检查**Chunk数量增加导致signal/wait与IB固定开销增长**，不马上修改协议。
3. 若overlap1明显优于0但仍差Stock：优先定位**MM计算路径本身是否较原生高效MM慢**；不能把融合下降全归罪于MemFabric。
4. 若会话差异与设备时间线对不上：检查测量漂移、卡对负载、Host版本、异步调度；标记`INCONCLUSIVE`。

阶段诊断不等于可以移除某个缓存同步。未来任何单变量源码优化必须先证明合法的producer/consumer生命周期与数据可见性合同，然后走编译、数值、ABBA。

## 五、验收、止损与同场 Stock 对照

- F2首轮网格的唯一基线是**D4 Tile2048 C1024/O1**；若候选与它相比单算子收益不稳定或不足约3%，暂不进入正式优化。
- 如果某参数组合稳定改善≥3%、至少两轮交叉方向一致且无长尾退化，才申请独立新Kernel优化分支；**这只是下一轮候选，不是战胜Stock**。
- 被选参数还必须对相同形状的 **Stock FP16 NZ MM+HCCL AllReduce** 做同卡、同口径比较；不能借用T0/T2的旧卡数据。所有成功与失败样本均记录。
- 最终真实模型三方指标严格：**Qwen3.6、输入4096、输出2048、TP2、batch上限10、Eager**，实际步内batch另外留证。Stock默认不变；稳定性仍是最终发布门禁，但仅在故障影响性能测量时先做最小定向排查。
- 出现507014、位精确失败、SO错版或卡间状态异常：终止当场并保全日志，不为了取得更好样本自动筛掉故障。
- **本文件是研发起步方案，不是实机报告。** 不能填虚构时延和收益。

## 六、两轮静态审视

- **R1：数据/通信合同**：以Git diff和CPU UT核对正式核、公共头文件、Host ABI、MemFabric协议均未改；测试驱动仅将D4固定的M4096/C1024/O1改为三参数，并保留`torch.equal`、status、IB Probe、Event与设备SO指纹。
- **R2：性能统计与归因边界**：每个双Rank样本按索引max再统计p50/p90；30次样本禁止缺失、跨组SO不能改变；保持同M的C1024/O1参考锚。软件只给`SCREENING_ONLY`，不自动给Stock/V1/V2最终选型或正式部署GO。

**优先下一步**：先跑F2-1，确认是否存在无需改协议即可取得稳定正收益的Chunk/Overlap配置，再决定F2-2的真实关键路径专项。
