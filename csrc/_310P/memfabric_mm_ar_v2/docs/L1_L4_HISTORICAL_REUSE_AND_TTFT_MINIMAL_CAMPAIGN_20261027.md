# A1-TTFT｜历史 L1～L4 证据复用审计与最小闭环实验（2026-10-27）

> **主线权威实验合同（2026-10-27）：**正式目标/验收/历史取证归档于 `dev/310p-memfabric-mm-ar`；具体 A1 测试实现和未验证工具留在 `work/310p-mm-ar-large-m-a1-20261026`（PR #9 Draft）。本文是**静态历史资料核对及待授权实验设计**，不表示新310P3 G0/Smoke/Profiler或模型实机已运行；D4 完整L1和新TTFT仍 UNKNOWN/NOT_RUN。
>
> 唯一目标：**真实 Qwen3.6 Prefill 的客户端 TTFT 改善**。L1 对照完整原生 FP16 Stock `F.linear + HCCL AllReduce`，聚焦每Rank 40 个 `o_proj/out_proj` 真实目标站点（实际命中按Trace计）；不将旧V2局部加速、Decode吞吐或 DFlash 推测当成 TTFT 收益。

## 1. 复用的历史取证与覆盖结论

| 历史实验、证据 | L1 独立完整Stock对照 | L2 实际40站点 | L3 真实Prefill关键路径 | L4客户端TTFT | 复用边界 |
|---|---|---|---|---|---|
| [2026-10-21 Prefill Attribution](V2_PREFILL_ATTRIBUTION_REPORT_20261021.md)、[证据索引](../evidence/v2_prefill_attribution_20261021/evidence_index.md) | **历史Serving站点的MM+AR设备Trace可用**；严格独立Stock/V2 ABBA未做；不是D4 Tile2048 | **可用**：Eager、M4096，双Rank各40站点/step，共200站点/Rank，Stock和V2双臂；输入contiguous复制已记录 | **可用但带边界**：单请求input4096/output1双Rank步周期，有一例74ms长尾；跨Rank时钟未校准 | **部分可用**：同一次诊断单请求时延；另有无Profiler正式4096/2048/50请求/并发10的E2E回归 | 旧版V2、旧Host/Device二进制，绝不当作当前D4实测 |
| [2026-10-22离线CSV独立复算](../evidence/v2_prefill_attribution_20261021/analysis/gate_a_offline_audit_20261022.json) | 不增加新Stock ABBA | **可用**：已按真实逐站点CSV重算40站点合计，不能将200次Kernel时长误作关键路径 | **可用**：双Rank各5步锚、4个周期，明确剔除完整异常Step | 不提供独立TTFT队列分解 | 重用已有计算，不重复新增相同的离线工具 |
| [G3 Qwen Eager](V2_G3_QWEN_E2E_REPORT.md) | 无新D4完整L1 | 路由和使能已取证；Graph实为Stock-vs-Stock，**不可复用** | 缺与当期L1同步时序 | **可用（历史）**：4096/2048/50/10，三会话TTFT全部恶化约2.1～2.5%；该轮旧V2 | 不可把G3 Graph或Gate E故障当作Prefill加速证明 |
| [D4正式Kernel优化](V2_D4_ROUND1_ADD_TILE2048_ONDEVICE_REPORT_20261023.md) | **只有L0**：旧V2 Tile1024→D4 Tile2048，M4096 event p50 2726.9→2556.3µs，+6.26%，**不是Stock基线** | 有Serving正确性/命中旁证；缺当前完整逐层开销 | 无同期全Prefill关键路径对照 | 有输出512/并发10小规模E2E；**不可当4096/2048正式TTFT** | 新D4 L1～L4完整链不成立 |
| [SEL-01 2026-10-24](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-sel01-three-arm-eager-c2-audit-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/SEL01_STOCK_V1_V2_EAGER_4096_2048_TP2_BATCH10_REPORT_20261024.md) | 无严格完整算子ABBA | D4路由审计，真实M4096已见证 | 缺逐请求/逐层同时序 | **可用（方向）**：Stock/V2四场有效、TTFT p50大致11.8s vs 11.9–12.0s；方向与吞吐不构成稳定正收益 | 不能把+1.12%输出tok/s冒充TTFT加速；场间Stock热态漂移明显 |

### 1.1 旧版历史闭环能给出的四层回答（只限2026-10-21当时工况）

- **历史L1站点测量**：Rank0 Stock MatMulV3 p50 795µs + HCCL AR p50 1149µs；旧V2 Kernel p50 2573µs。两段独立p50之和1944µs是**描述性近似**，不是逐站点配对完整L1 p50或现版D4严格ABBA。此近似对应旧V2比Stock更慢约629µs/站点；不把不同P50之和当精确计算结果。
- **历史L2**：[离线复算](../evidence/v2_prefill_attribution_20261021/analysis/gate_a_offline_audit_20261022.json)取完整4个无异常Step，Rank0 Stock每Step目标站点设备时长均值77.8525ms，旧V2 Kernel均值105.9149ms，差 **+28.0624ms**（未计独立输入Copy）；实际Serving另有约3.6ms/Step的`x.contiguous()`设备拷贝，原报告按自身统计口径给出整体 **约+30.9ms**。两个口径存在选择/计数差异，**不要合并成一个伪精确数字**；Rank1旧复算无异常步目标站点差为 +16.7805ms（亦不含Copy）。两臂40个有效目标站点均有设备级证据。
- **历史L3**：原归因报告Rank0完整Prefill锚定步周期 Stock 2364.4ms → 旧V2稳态2395.7ms，**+31.3ms**。Rank1各自锚定周期同方向（由于跨Rank时钟未校准，不做Rank0与Rank1绝对开始时间差分）；一次双Rank同步74ms异常需保留原样并单独统计，不能删后假称没有长尾。
- **历史L4（独立诊断负载）**：固定4096/1、并发1、Profiler双臂，剔除首调建池与V2的一次74ms异常后，稳态客户端单请求时延约+34ms/+1.4%；这不是4096/2048业务负载。
- **历史L4（正式负载，另一会话）**：无Profiler、input4096/output2048/TP2/并发10/50请求/Eager，TTFT mean Stock11341.9→旧V2 11578.6ms（+2.09%）；p50 11811.2→12091.7ms（+2.38%）、p90 +2.28%、p99 +2.33%。**不能**用+34ms与这组TTFT差额直接解出`queue_ms`，因为不同负载/会话，且服务器未提供逐请求排队、调度阶段时间。
- **历史解释**：W1逐调用同步使Stock约3.29ms，而真实Serving Stock MM+AR约1.94ms；W4连续通信有背靠背竞争。旧V2约2.6～2.7ms波动较小；真实Serving中目标AR相隔约60ms。故历史“微基准正收益未传导”的首要原因是**Stock微基准口径不代表真实Prefill，V2在Serving下原本就慢**，随后才是40站点占比、额外布局Copy和并发调度。此结论只对旧二进制有效。

### 1.2 必须显式标红的版本与原始证据限制

1. 历史报告 §2写Host `libmf_hybm_core.so` SHA前缀 **6e9dc2d6**，但同轮已提交的 [provenance.json](../evidence/v2_prefill_attribution_20261021/provenance.json) 的 `host_core_so_sha256` 为 **88d392850733…**。两者**不一致，原因尚未查清**；原始Trace也只在现场`/tmp/opencode/v2_prefill_captures/`（可能已清理），不能在本次静态核查中宣称同一SO身份已核实。该问题影响跨实验W1/W4与Prefill的版本可比性，不抹杀当时已归档的双臂观察，但必须记录为`PROVENANCE_CONFLICT`。
2. 2026-10-22的离线审计已明确`cross-rank wall clock not calibrated`、`input copy net delta not independently validated`；历史报告的成本合计是一个可信的同向归因，不是精确跨设备追踪的因果恒等式。
3. SEL-01的逐请求输出与当期Perf trace没有同一时间戳关联；当前D4 Tile2048的L1严格Stock对照、L2/L3同场Trace、L4正式TTFT全链**仍UNKNOWN**。不得跨日期拼旧版Stock与新D4 kernel时延。

## 2. 唯一新增实验组：A1-TTFT（G0→L1/L2→L3性能门槛→达标后L4）

**主线管方向、工作分支管实施**：本文为主线有效方案；实施只复用现存A1工作分支`work/310p-mm-ar-large-m-a1-20261026`（PR#9），测试资产待静审与实机核准，不修改正式Kernel/Host/MemFabric协议/Serving默认/投机Runner。不抢跑设备，**每阶段都由用户单独放行**。选择健康闲置的单张310P3物理卡双Die；旧43520 Die7/MCU告警未清除前禁止选用；ABBA全程固定同卡、版本、频率/温度记录与SO指纹。安全标准依[ADR-014四卡/KFC规则](310P3_四卡动态选卡与KFC自动部署准入规范_20261026.md)。

### G0（当前唯一可先行的设备相关阶段；CPU与四卡只读）

- 先在隔离worktree执行原仓`unittest/pytest/py_compile/bash -n`，确认当前A1 HEAD且无未提交修改。
- 读四卡每Die健康、43520 Die7及MCU告警、温度/频率与租户进程；确认同卡双Die、逻辑/物理号映射、独占。
- 哈希核对D4 Device SO（冻结`2a19390c…`）、Python插件（`8eb65de9…`）、MemFabric Host系统+Python双路径（`1dc46d7d…`）、KVER11（`25e16b0a…`），真实加载路径、现存部署版本与可恢复备份；合法更版需**评审修改合同**，不得绕过SHA。
- 仅运行`test/perf/a1_g0_readonly.py`等只读探针；不可导入/启动MemFabric池、执行KFC部署、`torchrun`或自动设置U1准入旗标。得到人工`G0_GO`才可申请Smoke；任何阻塞报`G0_NO_GO`。当前G0 **NOT_RUN**。

### A. L1完整版Stock基线：复用U1，不再重写Kernel微基准

按独立批准依次运行现有`test/perf/u1_run.sh <NEW_ABSOLUTE_DIR> large smoke` → `large screen` → 完整审核后才考虑`large confirm`。每个阶段只跑已经批准的部分，不允许一次脚本串起三段：

- **Smoke**：M2048 Stock、D4各1场独立双Rank，位精确/status/首写/KFC正常，故障立即STOP；不做性能GO。
- **Screen**：复用既有两场大M Stock、D4完整测量（M1024仅探针，正式路由只有2048/4096），各M p50/p90/p99、双Rank按样本先max配对；同时测Stock原生MM-only供方向性诊断，不能与完整MM+AR时延简单相减。
- **硬门**：单独审查HCCL跨Stream提交/完成的Event覆盖、实际加载Stock MatMul与HCCL内核、V2 D4 kernel/Chunk1024/Overlap1、输入权重stride/NZ与物理M；不许只根据`torch.equal`和脚本Success宣称完整链公平。
- **Confirm**：仅在Screen与真实HCCL/路由独立审核合格、用户再次批准后，复用已有双轮`Stock–V2–V2–Stock`×2，共8场，每M 5 warmup+40 samples。按原U1完整性/漂移/尾部合同判定`L1_CONFIRMED`或`L1_INCONCLUSIVE/NO_GO`，不可擅自修改既定5%筛选规则来伪造正向GO。**不论L1正负，持续大M优化方向不终止**；是否进入后续短Trace由用户基于本轮设备安全决定。
- 为解决历史W1/W4误导：短Trace校验真实Serving目标站点完成时间与独立L1实际差异，**禁止**拿逐调用Device Sync税或HCCL连续拥塞时延宣传真实模型收益。若需要新的受控“站点间隔约60ms”微基准，应先提单独源码方案/CPU审查，**不是本轮必做项**。

### B. L2与L3时序校准：必要时只补一对短Profiler（不是L3性能GO）

在L1通过真实性/安全审核且另获许可后，借用已有`tests/e2e/_310p/v2_prefill_capture.sh`采集链（在当前部署参数上静态审计并更新独立取证目录，**不可照搬旧版本的二进制和卡号**）：

- **两次独立Eager serve（Stock、D4各1次）**，同样模型/代码/版本/物理双Die，input4096/output1、并发1，至少6次热态预热、后续5个短窗口请求；Profiler只用于L2布局/核路由/L3设备时序*诊断与测量口径校准*，**既不把带Profiler的TTFT当正式TTFT，也不只凭这5个Profiler请求签发L3性能GO**。历史归因机制已清楚的内核版本可减少重复Profiler频度，只有kernel/ABI/布局/归因假设有变化时重新采。
- 为每Rank逐层采`site_index / layer_name / MM开始结束 / HCCL AR开始结束+真实完成 / V2 kernel开始结束 / input contiguous copy / output可被下一消费Kernel读取 / M / stride / physical M / staged / stream`，双Rank各40站点，每个完整Prefill请求检查目标数与实际命中（不把选层日志当Kernel命中）。
- 从整个Prefill首目标站点到最后目标站点之后的实际依赖终点画**各Rank独立的关键路径**；标注每Step开始/结束、两Rank到达和等待、完整Step时长，记录Profiler持久AICPU核、初始化/首调数值门与停顿，不得把重叠Kernel时长简单相加。跨Rank时钟不校准时只比较**各Rank自身时差**及安全屏障/服务端步时，不能任意相减绝对时间戳。
- 原历史分析`evidence/v2_prefill_attribution_20261021/scripts/`与`analysis/gate_a_offline_audit_20261022.json`可作为计算与异常剔除规则参考；无需重做已归档旧CSV。
- 若Profiler路径引起507014/507899/IB/KFC/bit-exact故障则立即STOP、封存rank0/1完整原始Trace和SOSHA，不自动换卡重跑。

### C. L3先决性能门槛：只测真实模型Prefill步，不跑正式L4

**正式裁决：**L3必须先对比真实Qwen3.6 Eager模型的单请求Prefill**执行关键路径**，只有可重复的正向收益才进入L4。L3失败并不是停止大M研发的条件，而是**当前候选继续回到L1/L2迭代、禁止启动正式L4**的条件。

- **唯一主要负载**：input4096、output1、并发1、TP2、热态Eager、Stock/V2同模型/同CANN/SO/KVER/同卡双Die，实际Forward M及每Rank 40目标站点真实命中必须与B相符。无Profiler的同一服务模型路径上取真实Prefill step的设备完成跨度：从该步进入模型计算的可审计起点，到本步末尾结果/首Token logits确实对后续消费可见的终点；同时留每Rank持续时间、执行重叠/同步证据和Step标识，且二者不能用未经校准的跨Rank绝对时钟相减。**这不等于客户端4096/1 TTFT**：后者包含发送/排队/采样/返回，只作为旁证。
- **测量实现硬门**：现有`v2_prefill_capture.sh`与U1并没有被证实能在**不启Profiler且不插入额外Device Sync**的情况下计量完整真实Prefill step；因此必须先基于当前vLLM/torch_npu真实源码审查是否存在可靠的内置Prefill step执行指标，若缺失，只允许做**低侵入、仅测试路径启用**的事件/时间戳采集并完成CPU静态审查和Profiler校准。不得重建Runner、修改模型数流、逐层`torch.npu.synchronize()`来造计时；拿不到有设备完成语义的独立指标必须标`L3_MEASUREMENT_NOT_READY`，不能用TTFT或40×Kernel p50填表。
- **轻量Screen**：真实可审计计时就绪、全部安全门禁通过并取得**单独许可**后，仅Stock→V2两个无Profiler独立Serve，各热态若干请求（建议先至少10成功样本），冻结随机输入token、真实M、Kernel签名、双Rank映射、下游完成定义；采完整请求/Step原始时间戳与队列空闲旁证。没有方向性正收益先归因L1/L2、优化Kernel，不耗费正式长输出业务。
- **L3正式Confirm**：仅Screen信号正向且另获许可后，做Stock→V2→V2→Stock四次无Profiler独立Serve，每场冻结同一单请求Prefill工作负载，预热后建议≥30个成功样本；按会话级ABBA检验（请求样本不是独立会话），报告均值/中位/尾部、双Rank各自完整Step、Stock跨场漂移与设备计时分辨率。正式判断的**预先冻结阈值**应来自当前Stock热态重复性、可观测的双Rank实际设备完成时间和统计不确定性：要求两组配对方向一致、收益超过测量波动上界且不以丢弃异常样本制造增益；**不规定L3必须≥5%**，因为旧测仅40站点约占完整Prefill 3.3%，5%本身可能超出本项目可改变部分的物理范围。
- **分流裁决**：`L3_GO`（同版本可重复、显著超出噪声、数值与设备安全完好）→后续经用户另行许可启动D的正式L4；`L3_NO_GAIN`（可靠观测且零收益/负收益）或`L3_INCONCLUSIVE`（漂移/波动淹没弱收益）→暂不运行L4，先回L1完整Stock差距和L2拷贝/布局/SDMA等待持续优化，下一候选仅重测变化所影响的阶段；`L3_MEASUREMENT_NOT_READY`→先补计时证据，不可把测不到误判为算子负收益。L3结果只作为**Prefill单请求性能门槛**，并不担保真实并发10混合调度下TTFT必然改善，L4仍须独立验收。
- **阶段成本说明**：若候选L3不达标，省去原方案4次4096/2048长输出模型会话；L3达标时才追加4次L4会话。L2短Profiler只在归因不明或内核/布局/关键路径变化时补采；**不承诺每个Kernel微调都重新跑2次Profiler+4次正式Serve**。所有设备行为仍由人工逐阶段批准，故障立即STOP保全。

### D. L4 TTFT 无Profiler正式ABBA：仅L3-GO后启动的4次Serve

**严格以当前版本L3-GO作为性能前置门槛**，且独立完成设备安全、数值和人工授权后，才允许启动以下四场正式TTFT负载；没有L3-GO时 **L4_NOT_RUN_BY_DESIGN**，不得为了输出吞吐去提前运行。此处是后期正式验收，不是每轮大M优化的例行测试。

| 会话 | 臂 | 正式工作负载 |
|---|---|---|
| S1 | Stock | input4096/output2048/TP2/并发10/50请求/Eager |
| F1 | 经L3确认的正式V2候选 | 完全相同 |
| F2 | 经L3确认的正式V2候选 | 完全相同，独立Serve与MemFabric池 |
| S2 | Stock | 完全相同 |

- **L4仅需4次无Profiler服务启动**，不重复L3的正式单请求Prefill筛选，避免在全部失败的Kernel候选上反复执行2048-token长输出业务。正式每场独立热态预热、所有前段请求及设备队列排空且两臂同等热态；若热态漂移超过合同门槛则标注无效、保存原始数据，不勉强宣称正收益。
- 正式负载沿用项目强制合同：模型Qwen3.6-35B-A3B-w8a8、`--enforce-eager`、TP2、4096 input/2048 output、50请求/并发10、`--ignore-eos`、chunked-prefill/async-scheduling保持相同、相同随机种子/实际token列表、服务版本与设置一致。优先复用`tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh`的启动、`vllm bench serve --save-detailed`及路由审计，必要的**组合启动器须先纯CPU审查**、不能直接绕开现有安全门禁。
- 每请求必须保存`request_id / 发送时刻 / first_token时刻 / TTFT / 输入输出实际Token数 / 服务会话 / 设备卡+SO SHA`；服务端能被当前实际vLLM接口可靠观测到的`queue_enter/queue_leave / prefill_step_start/end / first_token_ready / response_flush`才写有效数值，否则写`UNKNOWN`。**禁止用客户端TTFT减去单请求Profiler步周期臆造排队时延。**
- 锁定随机请求序列，保存实际发送/到达节奏；不能仅依赖`--max-concurrency 10`就假定四场队列状态完全相同。保存逐请求raw TTFT、每会话p50/p90/p99、会话级ABBA paired delta与Stock漂移；不得把50条相关请求冒充50次独立ABBA实验。若收益落在实际会话漂移包络内则`TTFT_INCONCLUSIVE`，后续增加会话由用户另行批准；不通过循环直到出现正收益制造幸存者偏差。
- L3已独立归档的4096/1单请求样本不得复制成新的L4观测；正式50请求的p99仍需带不确定性说明。任何错误、跨Rank漏执行或环境漂移则保留失败场、停止本阶段并判无效。

### E. 四层统一出账（严禁混合日期/版本的L1-L4数字）

| 结果字段 | 来源 | 无证据时 |
|---|---|---|
| `L1.stock_complete_us / fused_complete_us / delta_us` | U1 Confirm 8场完整MM+HCCL/正式D4、真实HCCL跨流审计 | UNKNOWN |
| `L2.site_hits_per_rank / physical_M / copy_cost / downstream_visible` | B的双Rank短Trace；C的轻量服务采样校验 | UNKNOWN |
| `L3.prefill_step_ms / delta_ms / dual_rank_critical_path / gate_status` | B的Profiler仅负责设备依赖归因/计时校准；C须提供**不启Profiler**且已验证覆盖真实双Rank Prefill设备完成边界的计时+ABBA数据 | UNKNOWN / NOT_MEASURABLE |
| `L4.formal_ttft_p50_p90_p99` | D的4场正式4096/2048/50/10 ABBA客户端逐请求观测；**只有L3-GO才执行** | L4_NOT_RUN_BY_DESIGN |
| `transmission_explanation` | 同期数据判断Stock基线、额外拷贝、执行重叠、双Rank与Queue；缺时间戳不得强行解释 | UNRESOLVED |
| `comparison_identity` | 同一设备、Kernel/Host/插件/KVER、模型Eager、负载种子、真实40站点记录 | FAIL-CLOSED |

**GO与继续研发分开**：L1即便优于Stock，也只给算子候选；**L3是大M日常性能筛选门槛：L3_NO_GAIN/INCONCLUSIVE/NOT_MEASURABLE时回到L1/L2持续调优、L4不开跑**；只有L3-GO才申请L4。L4必须在正式业务负载中展示稳定、可信的**TTFT下降**，才可以声称大M对Prefill终端指标带来收益。性能正向不解除Gate E/507014和冷首写安全发布阻塞。L1为负也继续极致优化，不因本轮收益小停大M线。

## 3. 实施分工、执行许可、归档

- **ChatGPT（本仓研发）**：该历史复用与实验合同、严格分析器/采集工具的静态实现和两轮审查、读回GitHub实机证据后给L1～L4结论；今后正式代码优化只限Kernel/LinearMethod薄接入，不引入DFlash Runner重建。
- **用户/OpenCode（310P3现场）**：仅按逐阶段明确许可执行CPU+四卡只读G0 → 经单独批准L1 Smoke/Screen/Confirm → 经单独批准L2短Profiler/轻量L3 Screen与Confirm → **仅L3-GO后经单独批准**L4正式模型ABBA；每阶段上传脱敏manifest、rank0/1日志、原始样本、实际安装+装载二进制指纹、异常和恢复记录。
- **停止/换卡原则**：G0阶段可排除不健康/被占用的整卡并在其它卡重新G0；实验中507014/507899/IB/KFC/静默首写/MCU异常**立即STOP，禁止自动换卡、继续ABBA或清除现场**。Stock与融合必须在同一物理卡上比较。
- **旧文件可复用**：`test/perf/u1_{run.sh,arm.py,analyze.py}`、`test/perf/a1_large_m_assess.py`、`test/perf/u1_e2e_budget.py`、`tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh`、`tests/e2e/_310p/v2_prefill_capture.sh`、旧归因离线脚本与per-site CSV。新组合入口若现有脚本不能满足，则**另行编码静态验收**，不能伪称现有U1已经具备端到端ABBA自动化。
- 本次主线仅纳入目标、归因与实验设计文档，没有生产代码修改、没有新模型启动、没有性能结果；A1 实施 PR #9 继续 Draft、未合入。**眼下首先需要的是原仓CPU合同+四卡只读G0的现场报告，不是对新设计的自动实机执行许可。**
