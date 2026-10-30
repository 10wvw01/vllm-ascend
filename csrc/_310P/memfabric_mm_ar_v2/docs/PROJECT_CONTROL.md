# 310P3 MM+MemFabric AR｜项目控制台（唯一权威当前状态）

> **维护于** `10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`；2026-10-25。**新对话优先读本文件**。这是活动状态与交接合同，不是覆盖历史实验报告。所有未在当前主线更新的旧 README、PLAN、实验总结都只作为历史证据。
>
> **主线创建基底：** `perf/310p-u1-fullchain-attribution-20261025` @ `d0d57b94df06…`。基底已包含 D4 Tile2048 正式 Kernel、F1-C 实验及 U1 测试工具；U1 仅完成源码/静态审查，**没有新的 U1 实机证据**；主线建立本身没有修改正式设备 Kernel、Host ABI、通信协议或 Serving 开关。

## 0. 项目目标、边界与验收

> **2026-10-27大M实验总账已设立：**[LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md) 是本项目**大M Prefill→TTFT实验任务、执行状态、结果证据、阻塞和更新日志的唯一维护入口**；本 PROJECT_CONTROL 继续负责全项目事实与方向。本轮仅新增总账/链接，没有A1原仓CPU、四卡G0、L1～L4新设备结果，禁止将计划当性能PASS或上机许可。


> **2026-10-27 项目主线正式方向（ADR-015 + ADR-016）：**大M最终以**真实Qwen3.6 Prefill TTFT**为唯一最终性能主指标；L1唯一有效算子级基线为**完整原生FP16 Stock MM+HCCL AR**，聚焦每Rank实际命中的40个`o_proj/out_proj`目标站点在双Rank上的关键路径。性能研发采取 **L1/L2归因优化 → L3真实Prefill设备Step无Profiler可重复正向收益门槛 → L4正式4096/2048/TP2/并发10/50请求TTFT验收**；L3无稳定正收益则继续大M优化，不提前运行长输出L4。4096/1客户端时延不等于L3设备Step；当前无Profiler设备Step计时能力尚未完成静审/校准。**主线管方向，A1工作分支只管实施**；本次只纳入文档，不代表已有新的设备测试结果或提供实机许可。详见 [主线正式实验合同](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)、[ADR-015/016](DECISIONS.md)。

**唯一项目：** 在 Ascend **310P3 单卡双 Die、TP=2** 上，用 MemFabric 实现**单 Kernel MM+AllReduce（SDMA+FP16 Add）**，覆盖 Qwen3.6-35B-A3B-w8a8 中已确认未量化的 attention `o_proj` / GDN `out_proj`；模块选择以运行时 shape、dtype、TP、真实模型前缀为准，禁止为了扩大覆盖误改量化 MoE 路由。

**不可动摇：**
1. 大 M **持续性能攻坚到可证实的硬件/协议瓶颈**，不再为每个局部参数计算投入价值，工作重点是正式 Kernel 的 MM 效率、SDMA 真实交叠、额外拷贝/布局、同步关键路径。须报告 **完整融合算子 vs 原生 Stock MM+HCCL AR**，不能拿“D4相对旧V2”冒充。
2. 小 M 是**MM+MemFabric AR 同 Kernel**带来的固定开销优化。真正目标是原生热态 FP16 Stock（完整 MM+HCCL），不是旧 T1 诊断核；普通 Decode M=1/8/10/16 和投机验证实际 M 分布都要覆盖，不假设 speculative M 恒为10。不可删除必要 cache clean、同步、credit 来造收益。
3. **同步兼容 DFlash 投机推理、Graph 和异步调度**，避免算子优化完以后另做一套集成。外部代码只能只读参照，不修改、不推送、不向 `HaonanWu1/vllm-ascend` 创建 PR。
4. **Stock 默认**，功能/数值/冷首写/双 Rank 状态/已验证 Stream 与 Graph 合同未满足时 fail-closed，不能静默回退后报告融合收益。
5. 原先强制普通模型验收固定为 **input4096 / output2048 / TP2 / batch上限10 / Eager / 50请求 / ignore-eos**，保留真实调度 M 和 batch trace；投机推理单列第二业务场景，只在配置和真实路由被核对后与同一投机模式 Stock 对比；不可混算两种模式的收益。

## 1. 当前技术形态（已知源代码事实）

- `src/op_kernel/mf310p_v2_kernel.asc`：正式大 M T0 / Tile2048 Kernel，调用公用 MemFabric SDMA，并在单 Kernel 内完成 Add；`src/op_kernel/mf310p_v2_diagnostics.asc`：T1 小 M、诊断探针；二者不能把“诊断通过”当生产支持。
- `src/op_host/mf310p_v2_trial_adapter.cpp`、`src/torch_adapter/mf310p_v2_serve.cpp`：Host/Serving 生命周期、启动、输出池、首写与桥接。KVER11和物理设备号修复为实机关键依赖。
- `vllm_ascend/_310p/ops/memfabric_mm_ar_v2_serve.py`：**当前正式 V2 Serving 仅 M∈[2048,4096] 走融合**，其余走原生 Stock；自带首次命中正确性 gate；大 M Graph 并未获得正式验证，不能混用原有 Graph A/B 假阳性。
- `vllm_ascend/_310p/quantization/modelslim_config.py` 通过定制 LinearMethod 选择性导流；`vllm_ascend/_310p/ops/memfabric_mm_ar_runner_hooks.py` 包装 Runner dummy 和 sync 生命周期。除补丁化适配外，不在自己的代码里复制维护 DFlash runner。
- 现存 DFlash 代码：与指定外部开发分支的 `dflash_full_decode_only.py`、`dflash_piecewise.py`、`dflash_full_and_piecewise.py` 比对内容一致；自身 `model_runner_310p.py` 另有小范围上游依赖兼容导入，ModelSlim 文件多出定制 MemFabric 路由。**文件相同不等于 Graph+MemFabric 联合行为已实机通过。**
- HCCL 与 MemFabric 必须审查生命周期共存合同；F1-C 为公平测量隔离两种数据面于不同进程；G3 有特定混合路由短测证据，但不能外推到 DFlash Graph/长稳。

## 2. 已知结论（严格区分比较对象）

| ID | 验证 | 证据 | 状态 / 对未来决策的约束 |
|---|---|---|---|
| D4 | 正式 V2 Add Tile 1024→2048，8场ABBA | [D4 实机报告](V2_D4_ROUND1_ADD_TILE2048_ONDEVICE_REPORT_20261023.md) | **相对旧V2算子 +6.26%**；不是完整 V2 vs Stock；保留 D4 Tile2048，不默认量产 |
| P1 | 老版本 Prefill 站点归因：V2约2573µs，Stock MM+HCCL约1944µs，额外 input contiguous约66µs | [Prefill 归因](V2_PREFILL_ATTRIBUTION_REPORT_20261021.md) | 老版本真实Serving路径存在净慢约32%；**不同日期/版本，不当成D4最新完整对照**；查 MM效率和拷贝 |
| SEL-01 | input4096/output2048/TP2/batch10/Eager Stock/V1/V2 | [三方报告：SEL-01](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-sel01-three-arm-eager-c2-audit-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/SEL01_STOCK_V1_V2_EAGER_4096_2048_TP2_BATCH10_REPORT_20261024.md) | Stock 45.98 vs V2 46.49 tok/s（V2 +1.12%，跨轮方向不一致），V1 0/3 完整有效；**不够声明模型稳定正收益** |
| F2 | 大 M、Chunk256/512/1024、Overlap0/1 同 D4 SO | [F2 实机报告](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-f2-mm-sdma-overlap-sweep-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/F2_ONDEVICE_REPORT_20261024.md) | Chunk1024/Overlap1保留，缩小Chunk得负收益；仅停止这条参数网格，**不停止大M极致优化** |
| F1-C | M10 T1诊断 vs原生FP16 Stock独立进程两轮ABBA | [F1-C实机报告](../evidence/f1c_20261025/F1C_ONDEVICE_REPORT.md) | Stock约308.2µs，T1约425.2µs；T1等效速率 −27.5%；历史F1初筛“融合+20%”被严格对照推翻；旧T1 NO-GO，不是所有小M单Kernel方案NO-GO |
| C3 | CleanLines 4x循环展开 | [C3报告](https://github.com/10wvw01/vllm-ascend/blob/perf/310p-v2-c3-cleanlines-unroll4-20261024/csrc/_310P/memfabric_mm_ar_v2/docs/V2_C3_CLEANLINES_UNROLL4_ONDEVICE_REPORT_20261024.md) | 单算子约 +1.70%，有噪声，无实模型验证；局部方案NO-GO，不合入 |
| U1 | 多形状统一测试脚本与分析工具 | [U1 静态审查](U1_STATIC_REVIEW_AND_GO_GATE_20261025.md) | **源码静态合同通过，CPU/设备实机 NOT_RUN**；只作为后续复用工具，不把U1自立为主研发方向 |

**切忌**：Tile2048 相对旧V2的收益和新V2 vs Stock的收益是两个指标；历史D4输出512的端到端数据不能混进强制输出2048的三方评测。上述单算子 Event 指标含可能的Host下发空档，不一定是纯设备Kernel耗时。

## 3. 真实风险与未闭环

- **Gate E / 507014**：长输出Eager的双Rank/SDMA等待曾出现偶发挂死；根因是否为“已投递但不可见”或“未投递”尚未证实。**部署/发布阻塞，性能研发中如果异常出现则必须保留完整现场并停测**。
- **MTE3首写与冷池**：曾出现 status=0但静默错误，须保留持久预分配、首调用位精确、多Chunk深度预热等已验证缓解，不得因追求性能删除。
- **Graph伪命中**：G3 发现首次 dummy 编译固化 Stock 路由，导致名义Graph V2 A/B其实Stock-vs-Stock。需有融合 Kernel 命中、Graph模式/尺寸、输出正确性和重放证据。
- **DFlash形状改变**：投机验证M≈有效请求数×(草拟token数+1)仅为逻辑例子，实际physical M包括Graph padding；不能硬编码M=10或只按token总数裁决。
- **装机二进制并非自动与Git一致**：历史dev6/7依赖MemFabric物理号补丁、两路径Host SO及设备SO/KVER11匹配；现场必须独立G0核对；此控制台不声称当前服务器运行于本HEAD。

## 4. 当前优先级与任务边界（动态更新此节）

| 顺序 | 工作包 | 责任 | 完成标准 / 证据 | 当前状态 |
|---|---|---|---|---|
| P0-1 | **建立并维护唯一主线/知识落盘** | ChatGPT研发侧 | 本分支入口、状态、分支规则、决策记录入库；后续任务持续同步 | **初始建制已完成；后续每轮必须持续更新** |
| P0-2 | **DFlash同步基线与融合薄适配契约** | ChatGPT编码审查，OpenCode现场验证 | 固定外部只读SHA；普通/投机/Graph真实M、选层、路由和Graph capture/replay审计；仅改我方仓库 | **设计中，无新实机证据** |
| P0-3 | **大M极致优化（持续）** | ChatGPT方案/源码，OpenCode硬件测 | 每轮含最新D4完整V2 vs Stock同场、MM/SDMA/Add关键路径、逐步优化、正确性，解释到Qwen TTFT/TPOT/吞吐 | **持续主目标，尚未建立最新严格完整Stock对照** |
| P1 | **小M单Kernel固定开销专用路线** | ChatGPT方案/源码，OpenCode硬件测 | 先以DFlash真M分布确定形状，设计MM+SDMA+Add单Kernel，原生FP16 Stock/HCCL同口径胜负 | **新方案待设计，不复活旧T1** |
| P0-4 | **统一模型收益传导** | 双方 | 普通4096/2048/TP2/batch10/Eager不变；投机Stock vs投机融合独立比较、接受率/Forward数/真实融合调用计数；图/数据/失败样本保全 | **与P0-2/3并行，尚待新轮次** |
| 阻塞 | **Gate E稳定性及Graph生产准入** | ChatGPT归因，OpenCode取证 | 只有正式部署前必须关闭；测试失败/静默破坏当轮fail-stop | **尚未关闭** |

**一律不要**再用是否达到小于3%的局部ROI决定停止大M主线；也**不要**把全模型MoE/GDN/Attention一般优化当成这个项目的任务。U1作为通用测量工具，必要时使用，但不重置研发方向。

## 5. 每次交接、报告和代码提交的强制格式

每轮在GitHub本主线的状态更新至少包含：
1. **HEAD/基线版本**：主线SHA、实验SHA、外部DFlash固定SHA、CANN/Host/Device/KVER11 SHA及本次改动文件。
2. **当前结论**：只写有证据的PASS/FAIL/UNKNOWN；明确是D4内部、完整算子Stock对照还是Qwen真实业务。
3. **测试合同**：TP2/单卡双Die、M/K/N/dtype/weight布局、Overlap/Chunk、Graph模式、普通/投机、实机设备号及样本数。
4. **问题与下一动作**：一个清晰可执行的主线任务、阻塞与下一次实机准入条件；明确谁在何处分工。
5. **证据与回退**：可访问的GitHub报告/JSON/双Rank日志或注明只有现场文件、测试命令、回退的二进制指纹。
6. **决策**：新方向、停止/保留/替代需同步更新 [DECISIONS.md](DECISIONS.md)，不改写旧证据。

必须先核对 `git fetch` 后当前远程 `dev/310p-memfabric-mm-ar` HEAD，不能借用旧聊天消息、旧分支、未注明M/数据类型的数字当新进展。

## 6. 源码/仓库关系与入口

- **自有 GitHub唯一项目主线**：`10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`；工作包从本分支派生并评审回主线（详见 [BRANCH_GOVERNANCE](BRANCH_GOVERNANCE.md)）。
- **我们的 DFlash 基础代码**：已经位于自有仓库；`HaonanWu1/vllm-ascend:dev_v24/feat-310p-dflash` 只读对照（2026-10-25所检查HEAD `1989b04c04e6…`，其提交日期是9月29日，不代表其所有分支的最新状态）。
- **MemFabric独立源码/安装**：历史工程另有GitCode/自有镜像与服务器多套源码、build及SO；**不要假定GitHub代码SHA=安装SO版本**；必须以现场Git远程、部署清单及实际SO SHA为准。此前资产取证见 [310P3版本溯源报告](310P3_版本溯源与资产保全报告_20261020.md)。
- **现存分支完整盘点**见 [BRANCH_GOVERNANCE.md](BRANCH_GOVERNANCE.md)；历史性能/稳定性总账在`docs/310p-v1-v2-perf-stability-ledger-20261023`，不再作为今日项目控制台。


## 7. 2026-10-25 工程主线锐评与治理闭环（G-DOC）

- **确认**：本项目唯一研发主线固定 `dev/310p-memfabric-mm-ar`，`main` 仅为通用仓库默认分支。相对于D4的正式Kernel/Serving文件未发现主线建立导致的变更；已存在的U1资产仍是 **CPU/310P实测 NOT_RUN**。
- **审查缺陷**：此前只有文字“必须更新控制台”，缺少自动检查；版本溯源与服务器实际装载 SO/KVER易被混用；旧G3 README/PLAN虽已标注历史，但若脱离项目入口阅读仍有误导风险。
- **本轮处置**：新增 [ENGINEERING_GATES.md](ENGINEERING_GATES.md)、纯标准库项目检查器、6个CPU单元测试和主线 GitHub Actions 工作流。静态检查要求涉及正式源码/测试/证据的提交同步更新本文件；新证据/报告同步更新性能账本。**检查器本地单元测试已通过；CI须等待GitHub实际运行确认，不能写成CI PASS。**
- **严正限制**：本次没有修改正式Kernel、Host、Runner、Serving默认开关或 MemFabric；**未执行CANN编译、既有V2 pytest、310P3实测或Graph验证**。没有由此次治理产生的L0/L1/L2新性能结论。
- **遗留人工门禁**：GitHub仓库保护/强制PR与状态检查尚未在仓库设置中确认；所有者须另行配置和核验。若改变了生产Kernel/Host/Graph合同，仍必须人工完成两轮源码评审及实机前G0，不可因这组治理检查通过而开机。
- **后续研发优先级不变**：DFlash运行时真实物理M与Graph路由调查、大M严格完整算子Stock对照、小M专用同Kernel方案；待提交对应可执行方案和环境/版本准入清单，明确GO后再请用户委托OpenCode上机。


### CI首次扩展回归暴露并修复的工程漂移（2026-10-25）
- [GitHub Actions 40126853357](https://github.com/10wvw01/vllm-ascend/actions/runs/40126853357)：6/6 治理单测 PASS，既有V2源码合同测试 **53 PASS / 1 FAIL**。
- 唯一 FAIL：`test_v2_project_layout.py` 的旧根目录白名单只允许 docs/src/test/evidence，未接纳已正式入库的项目 `AGENTS.md`；属于**测试目录合同滞后于合法项目治理文件**，不是设备Kernel的数值/同步失败。
- 处置：严格保留目录集合验证，明确添加 `AGENTS.md`，不放宽成任意新增文件；静态回归实际结果须按本次后续 GitHub Actions run 确认。**U1设备实验仍NOT_RUN**。


### GitHub Actions扩展回归复验：已通过（2026-10-25）
- [Run 40126962427](https://github.com/10wvw01/vllm-ascend/actions/runs/40126962427)，验证提交 `2f5079163500e254b335a5b3540263691b903552`：`project-control` SUCCESS。
- 实际日志：治理检查器6/6 `unittest` PASS、既有V2 `test/ut` **54/54 pytest PASS**，项目入口静态检查和本轮Git变更台账检查均 PASS。
- 这是**GitHub Ubuntu/Python 3.11 的纯CPU源码及合成数据合同测试**，未使用310P、CANN、MemFabric设备库；不能推导出 U1 设备运行、DFlash Eager/Graph真实命中、数值正确性或任何L1/L2性能收益。**U1 设备侧仍 NOT_RUN**。
- 本项目任务的后续合入必须沿用这一CPU门禁；分支规则保护尚待GitHub仓库所有者确认启用。

### 下一批需要解决的源码级风险（静态审查发现，不等于实机证实）

| 风险 | 源码事实 | 具体下一步验证门槛 |
|---|---|---|
| R1 静默丢失融合资格 | `vllm_ascend/_310p/quantization/modelslim_config.py` 将 `should_enable_memfabric_mm_ar` 异常统一捕获后置 `eligible=False` | 先在隔离纯CPU测试中区分“合法不命中”和“检测器故障”，设计可审计日志/错误策略；不得在未验证模型加载路径前直接删除回退 |
| R2 输入布局拷贝 | `memfabric_mm_ar_v2_serve.py` 正式路径调用 `x.contiguous()`，旧Prefill归因存在约66µs额外搬运 | 用真实输入Stride/权重NZ布局证明无需额外转换或保留所需转换并全量计入L1 |
| R3 生命周期薄适配 | `memfabric_mm_ar_runner_hooks.py` 包装 `_dummy_run` / `_sync_device`，池启动后改用当前流同步 | 对外部DFlash Runner接口变化、Async Scheduler、Graph capture/replay、混合HCCL/MF及Pool销毁分别取证 |
| R4 Decode/投机路线缺位 | 现有正式 `v2_threshold_route` 仅 `2048≤M≤4096` 融合，其余Stock | 运行时记录物理M、真实目标层路由及投机verify分布；新小M正式核未开发/未准入，不得把旧T1诊断核冒充新V2 |
| R5 发布安全性 | 旧Gate E 507014、MTE3首写、Graph假命中尚未闭环 | 发布前需闭环；当轮出现数据错误/死锁立即停测留双Rank证据，不因CI通过开启融合默认 |

**当前是否需要310P3实机：** 本轮G-DOC治理与CPU验证不需要；下一阶段**真实D4完整V2 vs原生Stock L1、真实DFlash物理M/Graph路径**需要设备。上机前先依据 [ENGINEERING_GATES.md](ENGINEERING_GATES.md) 完成脚本/环境/版本准入审查，再明确提交给用户委托OpenCode；本轮无上机授权或执行。


## 8. Stock+DFlash 未成功启动：上游源码与环境依赖审计（2026-10-25）

- **用户新增历史事实**：此前Stock+DFlash投机推理尝试失败、疑似缺依赖；V2 Graph历史从未完成成功验收，曾有Graph AB实则Stock-vs-Stock。两者为独立问题，不可混报。
- 本轮静态追溯见 [Stock+DFlash上游与依赖审计](STOCK_DFLASH_UPSTREAM_DEPENDENCY_AUDIT_20261025.md)：指定上游核心DFlash、patch、需求声明、Docker/验收工具与自有主线对应文件逐SHA相同；我方Runner多一处ImportError兼容，ModelSlim增加MemFabric路由；Git历史并非完全追平、差异需按文件审。
- 上游明确依赖`vllm v0.24.0`，其验收脚本使用硬编码开发机路径并默认35B投机K=15，**不符合本期服务器与K=7**。实际历史报错堆栈及服务器Python包/导入路径目前缺失，所以根因只能标`UNKNOWN`，不能宣称已找到缺包或合并错误。
- 新增 [Stock+DFlash只读G0诊断工具](../../../../tools/310p_mm_ar/dflash_stock_preflight.py)，支持查看 Python版本/模块路径、pin兼容、vLLM所需API、模型config、Stock禁用融合标志及`pip check`，可选子进程导入；不做包安装或实际NPU推理。配套纯CPU合成单元测试，需由GitHub CI实际确认PASS。
- **研发门槛调整**：DFlash集成先修复并证明**Stock+DFlash Eager(K=7)**最小单请求与batch10运行，再分别核对Stock+DFlash Graph、V2融合Eager、V2融合Graph；任何一次Stock成功都不可替代 V2 Graph真实Kernel/Replay取证。大M正式算子L1优化可与此环境问题并行推进。
- **此提交无实机操作**，无正式Kernel/Host/Serving修改、无L0/L1/L2新性能结果；下一轮如需现场先取`stock_dflash_preflight.json`和实际ImportError/模块路径，再决定最小修复。


### 上游依赖审计工具的 GitHub CI 结果（2026-10-25）
- [CI run 40128444031](https://github.com/10wvw01/vllm-ascend/actions/runs/40128444031)，提交 `4f97de5e00451f3cc089c175e1105efa78c04b78`：治理/诊断单测 **12/12 PASS**，既有V2源码合同 **54/54 PASS**，元数据/改动闭环检查PASS。
- 仅证明检查工具在GitHub无NPU环境下可执行。历史Stock+DFlash故障根因与服务器实际vLLM/torch/transformers依赖均仍 **UNKNOWN**；Graph/V2不因该CI改变验收状态。


## 9. R1：Stock+DFlash 和 V2+DFlash 正式研发与实机准入（2026-10-25）

- 用户配置固定：DFlash模型 /home/models/Qwen3.6-35B-A3B-DFlash，K7；目标 /home/models/Qwen3.6-35B-A3B-w8a8；TP2单卡双Die。上游相同TP2投机测试会把draft TP显式配置为2，本轮沿用。
- 新增 [R1四象限实机合同](DFLASH_STOCK_V2_R1_GO_GATE_20261025.md)、tools/310p_mm_ar/dflash_v2_trial.py及纯CPU合同UT；第一层准入是只读依赖G0，实际设备身份和SO/KVER/Gate E情况仍需OpenCode现场核验。
- 顺序：G0通过后Stock+DFlash Eager smoke→V2+DFlash Eager smoke→两个独立Eager完整4096/2048/TP2/batch10/50请求；Graph待两组Eager经过审阅后另行做Stock、V2两组短烟测。V2 Graph历史没有有效PASS，不能将V2 Prefill Eager和Graph Replay混算。
- V2正式Serving仅2048～4096融合；真实DFlash Verify/Decode的physical M及实际融合调用数仍UNKNOWN。Eager V2必须取得正式Kernel双Ranktrace和首调数值证据，Stock不能被V2选层污染。
- 修复只读环境探针中过严的pip check合同：上游310P Docker故意卸载Triton/Triton-Ascend，仅这两类明确未安装错误带标识放行；其余Python分发、API和安装冲突照常拦截。
- **源码工具已经写入本主线，最后CI应以实测run成功记录为准；服务器NPU和模型实验本轮全部NOT_RUN。准入裁决为PRECHECK-GO，仅进入310P3现场G0，正式上机冒烟仍以现场全部门禁PASS为前提。**

## 10. R1服务端/客户端启动合同对齐用户原始参数（2026-10-25）

- 之前的实验脚本与用户完整启动命令不完全一致，现正式修正：Target由`/home/models/Qwen3.6-35B-A3B-w8a8`改为`/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`；Host改`0.0.0.0`（实机前需网络隔离确认）；max-num-seqs 10→16、gpu-memory-utilization 0.85→0.90。移除额外`--safetensors-load-strategy eager`、Bench额外`--num-warmups`，DFlash JSON不再显式增加draft TP，按vLLM v0.24.0运行时自然解析并由现场验证。
- 模型Bench正式负载仍是4096输入/2048输出/50请求/并发10/ignore-eos，与用户命令一致；Bench明细及分位数据、结果文件隔离目录是诊断附加，不是不同业务场景。
- 唯一刻意保留核心行为差异：**Eager组用cudagraph_mode=NONE**以排除Graph；用户原始FULL_DECODE_ONLY `capture_sizes=[10]`在K7/Verify宽度8的描述符整除规则下不合法，因此Graph研究组用候选`[8,16,32,64,80]`并进行实机鉴别。Stock和V2各自同模式严格一致，Graph不能继承Eager结论。
- 说明与操作完整见 [R1 DFlash K7 准入合同](DFLASH_STOCK_V2_R1_GO_GATE_20261025.md)第7节；本轮仅更新测试资产与文档，**未在310P实机运行，现场准入仍从G0开始**。

## 11. R1-B主线选择性整合与历史准入记录（2026-10-25；执行计划已由§13取代）

- 原始现场G0：分支 `work/310p-mm-ar-r1-g0-nogo-20261025` @ `45056500a8254787733ffb9d05e0c8efbbcb05d4`，隔离工作树从旧 `07586f2` 派生；**原始 frozen G0=NO-GO，Stock/V2+DFlash Eager/full/Graph均NOT_RUN**，保留 [现场报告](R1_G0_ONDEVICE_NOGO_REPORT_20261025.md) 和 [原始证据](../evidence/r1_g0_20261025/)。修复后 preflight 的单独静态GO不能改写历史裁决；本次工具整合也没有新的实机结果。
- 审查确认：现场真实 Python3.12.13、官方vLLM editable v0.27.1@6e448d0、torch-npu 2.10.0.post4、transformers 5.17.0，原G0使用的v0.24/旧pin及冷导入探针不符合已知环境。按真实Serve的 `NPUPlatform.pre_register_and_update` 初始化后导入通过；历史Stock+DFlash确切失败根因仍UNKNOWN。
- 以当前唯一主线 `592e708` 的 **用户原始服务参数** 为基底选择性整合诊断修复；绝不直接合入旧R1任务分支的试验脚本，以免重新引入host=127.0.0.1/max-seqs10/memory0.85/K7额外draft TP/额外加载策略和bench预热。
- 当前Trial冻结Target `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`；现场实测Draft `/home/models/Qwen/Qwen3.6-35B-A3B-DFlash`（比用户原始草稿路径多一级`Qwen/`），不通过猜测/符号链接回退。Stock/V2两臂JSON完全相同，K=7；`max-num-seqs=16`/0.90/host0.0.0.0，input4096/output2048/50请求/concurrency10，Eager NONE vs Graph待诊断；端口对外暴露前须核验网络隔离。
- 新G0：修复pin并同时**收窄**pip check豁免：仅允许指定声明者+指定缺包组合、精确已证实版本冲突，任何不在快照中的新依赖、冲突、空输出均STOP；强制真实Serve顺序导入探针 `exit_code=0` 才准进入模型，保持完整pip stdout/stderr。严格绑定python可执行路径、目标/草稿模型路径和本任务工作树；禁止editable finder静默指向主工作树。
- V2之前必须在本隔离工作树内核验 `vllm_ascend_C*.so`、`libmf310p_v2_trial_device.so`、`libmf310p_device.so` 存在，并人工核对真实Host/Device SO+KVER11 SHA与已核准现场部署；**脚本不自动复制、不自动重编/重装**。
- 历史R1-B原定的下一次310P实机按 [R1-B准入与执行合同](DFLASH_STOCK_V2_R1_GO_GATE_20261025.md#8-r1-b-整合后-310p3-实机准入与止损合同) 的条件推进。当时计划先Stock+DFlash Eager smoke，再V2 Eager smoke及两个同口径full；**这只是历史安排，2026-10-26废止Runner重建后不得照此推进通用V2迁移**。现只允许在已经跑通的Stock+DFlash同一框架中做融合薄接入；Graph仍需独立人工许可。拒绝伪命中、失败后继续跑和拿日志代替双Rank Kernel trace。
- **截至本次合入：没有运行310P模型服务，K7实际推理PASS/Graph PASS/V2实际Kernel命中均UNKNOWN；Gate E与大M Stock L1仍未闭环。** GitHub CI只是无设备CPU合同测试，不是设备准入。

### R1-B第二轮静态审查补强（2026-10-25）

- 审查除G0诊断本身以外的启动边界：旧报告可能来自另一安装、另一Python或同名editable代码。Trial新增 `validate_runtime_identity` 校验预检报告Python执行器、真实vllm CLI脚本shebang、vllm/vllm_ascend真实根目录、四个关键分发版本与当前状态完全一致；CI用合成目录/假包元数据测试不一致时FAIL。不得据此声称实际服务已加载。
- 审查确认：对V2的build artifact校验仅检查当前隔离worktree里文件是否存在，**不验证设备已经正确装载、Host/Device ABI及SO哈希**；这些仍是OpenCode现场G0人工门禁，不允许用“存在文件”替代签名/运行期映射证据。

## 13. 2026-10-26｜重大纠偏：停止并剔除重建 V2 Runner / DFlash 兼容体系（当前最高优先级）

> **新决策覆盖旧 ADR-011/012 和所有 R2-00～R2-21 / R3 Runner迁移工作。** 旧文本与实验日志仅作为历史，不再是研发执行要求。正式裁决见 [ADR-013](DECISIONS.md)。

- **明确取消**：定制310P V2 Worker/ModelRunner、通用 Triton 替换31项、46项Runner准入矩阵、Hybrid/Mamba/GDN ModelState重写、K7 rejection/sampling、multi-KV/draft 状态缓存重建、Device Copy Consumer、ModelState facade、API对齐/单独runner CI。禁止再以“Runner不能运行”为由扩展这些模块。
- **隔离并废止**：`work/310p-v2-r2-implementation-20261025`的所有Runner重建资产不得合入主线，关联 PR #7 关闭为**不采纳**；旧 `V2_RUNNER_TRITON_FREE_MASTER_PLAN_20261025.md`从主线删除，旧测试/文档不再作为当前CI或项目入口。R2分支仍存在时只可作为只读历史；分支删除需工具支持，不通过危险的强制移动引用代替。
- **回滚误入主线的试验门禁**：310P3 V2 Triton验证豁免恢复为历史 Stock/V1 原有env-only选择逻辑；保留`NPUWorker310.init_device`对未实现V2的提前拒绝，防止静默把V1冒充V2。无设备编译/执行，Stock默认与已验证的V2融合Host/Kernel/LinearMethod完全不变。
- **仅保留融合相关代码**：`vllm_ascend/_310p/ops/memfabric_mm_ar.py`、`memfabric_mm_ar_v2_serve.py`、`memfabric_mm_ar_runner_hooks.py`、`_310p/quantization/modelslim_config.py`与正式`csrc/_310P/memfabric_mm_ar_v2/src`。在用户指定能够真正运行Stock+DFlash的同一Runner/模型/业务负载上做开启/关闭融合A/B；只允许必要的算子ABI、LinearMethod、生命周期和Graph共存薄修复，禁止动DFlash模型本体。
- **实测口径**：TP2/单卡双Die，input4096/output2048/concurrency10/50请求、K7 DFlash与相同真实Runner对照；L1完整 Stock MM+HCCL vs V2 MM+MemFabric，L2 Stock+DFlash vs 融合+DFlash，双Rank首调正确性、真实调用数/层、physical M/Graph是否命中、性能传导。大M继续极限调优，小M专攻同Kernel固定开销。Stock默认。
- **实证边界**：R1-B过去在特定Stock+Draft+vLLM0.27.1旧Runner组合上构造失败（已归档），不得据此否定其他Stock成功配置；本次只是GitHub代码/项目治理，**没有新310P3推理、Kernel、Graph性能或现成Stock成功实测**。确切可工作Stock环境由后续只读取证锁定，不推断。
- **准入**：只保留融合算子自身的数值/首写/SO/KVER/G0/Gate E安全门禁；**46项通用V2 Runner准入矩阵失效，不得用于阻塞本项目核心算子研发或DFlash薄接入。**


## 14. 2026-10-26｜Gate E / 507014 移交 MemFabric 团队的独立复现基线（新测试 NOT_RUN）

- **复用已存在的诊断分支，不新建分支**：`diag/310p3-g1d-minimal-forensics-20261021` @ `e7ba8a5b3636f045c19f777ea11dee6856d1466d`。新增 [复现交付说明、G0 与安全执行合同](https://github.com/10wvw01/vllm-ascend/blob/e7ba8a5b3636f045c19f777ea11dee6856d1466d/csrc/_310P/memfabric_mm_ar_v2/docs/GATE_E_MEMFABRIC_HANDOFF_20261026.md)、只读离线证据判读工具、纯CPU单元测试与诊断分支专用CI工作流；**没有修改历史复现脚本、正式Kernel、Host/Serving或MemFabric协议**。
- MemFabric 诊断源码固定参照 `10wvw01/memfabric_hybrid_310p:diag/310p3-g1d-minimal-forensics-20261021@54d6376d619325fb70831ce16c6ab60305b272ec`；历史 G1-D D1R4 曾在设备复现双 Rank 507014、core0 stage50 邮件等待未返回；D2R1–D2R6 六轮没复现，写入丢失与AICore不可见尚不能区分。**这些均为历史证据，不是本轮测试结论**。
- **交付前门禁**：OpenCode 先只读 G0，核对两仓 SHA、真实 Python import、TP2 同卡双 Die、设备独占、Host双路径/Device/KVER11/实际装载SO SHA、网络隔离及恢复备份；用户授权后仅执行原 fused Eager **input4096/output256、25请求/轮、并发10、620s**，先短smoke后长稳，独立Stock同口径对照。出现 507014/507899/静默数值错误立刻止损保全，避免其它对照覆盖第一故障现场。可选G1D v12 诊断库需另行人工许可。
- **截至本次归档状态**：交付文档/CPU测试源码已提交；CPU单测**未在原仓 CI 实际执行**，现场 G0 / NPU Smoke / 620s fused / Stock 均 **NOT_RUN**。未授权直接上机。只有新的双Rank日志、真实融合Kernel trace、哈希与恢复证据完整并经审核后才交付 MemFabric 团队，不得将 0 ExitCode 或选层日志写成修复证明。

## 15. 2026-10-26｜四卡动态选卡和KFC自动部署标准（ADR-014，即日起所有实机生效）

- 310P3服务器共有**4张物理卡**；单卡双Die TP2实验不绑定原dev6/7。每次只读遍历四卡，动态选择双Die及MCU健康、同卡拓扑明确、无租户且空闲的一张；旧43520 dev6/7告警未清除前排除。G0卡级健康/占用失败可以换卡再G0；**运行中**507014/507899/KFC/IB异常必须先STOP保全证据，不准脚本自动轮测四卡。
- 选卡之外必须重新核查§ASCEND_RT_VISIBLE_DEVICES§的逻辑/物理DeviceID映射、非0/1设备下§aclrtMemSetAccess§物理号修复的MemFabric**双路径Host SO**和KVER11；目标Die的AICPU SDMA编排核在**授权首次建池/Smoke时自动KFC部署**，只读G0不提前执行KFC，禁用未经授权的伪helper与常驻模式。Stock/V2同一次ABBA必须固定同卡、相同版本，换卡单独run_id重G0不混算。
- 完整换卡与KFC实证指南：[四卡动态选卡/KFC准入规范](310P3_四卡动态选卡与KFC自动部署准入规范_20261026.md)，源自已有[2026-10-23 dev6/7 KFC部署取证](MEMFABRIC_310P_KFC_DEPLOY_AND_DEVICE_PIVOT_20261023.md)。**此提交仅文档治理，不代表新卡已验证或KFC已部署。**

## 16. 2026-10-27｜主线权威方向已从A1实验分支选择性归档（ADR-015/016）

- **分工已明确**：本`dev/310p-memfabric-mm-ar`主线承载真实研发目标、L1～L4历史证据归因、L3-first决策与阶段门槛；A1工作分支`work/310p-mm-ar-large-m-a1-20261026`的[PR #9](https://github.com/10wvw01/vllm-ascend/pull/9)保留**Draft实施身份**，代码/纯CPU测试/310P待验脚本不因文档归档而自动准许合入。不创建长期第二主线，禁止从A1整分支合入后反向覆盖本控制台。
- **唯一生效性能实验合同**：[历史L1～L4数据复用 + L3-first门槛 + L4延后执行](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)。历史2026-10-21旧V2的Rank0站点约78→106ms/步（不含独立输入Copy）、Prefill步骤约2364.4→2395.7ms、孤立4096/1单请求约+34ms，**仅可解释旧版负收益**；D4目前只有旧V2内部 +6.26%，其严格完整Stock L1与新L3、L4结果**UNKNOWN/NOT_RUN**。旧报告Host SO SHA前缀与provenance记录不一致，标`PROVENANCE_CONFLICT`；不得跨时间拼接成现版加速。
- **最新执行状态**：A1原仓CPU合同/310P四卡只读G0仍未得到新证据；Smoke、Screen、Confirm、L2时序、无Profiler L3设备Step采集、正式L4 TTFT均NOT_RUN。仅可按已存在G0安全合同推进，设备阶段各自获许可后才执行，旧43520 Die7告警与Gate E/507014风险不被文档合入解除。Stock默认、KVER11/Host双路径、Runtime身份、任何异常STOP保全要求不变。
- **主线交接规则**：今后新的经审查性能事实与决定同步更新`PROJECT_CONTROL.md`和`PERFORMANCE_BASELINES.md`，如变更方案追加ADR；A1 PR #9在实际合入前需针对新主线重新审阅和去重其旧项目控制文档，避免把历史分支误当主线。此处只确认**治理与实验合同同步**，不表示A1实机验收通过。

## 17. 2026-10-27｜建立大M Prefill TTFT唯一任务与进度总账

- 新增 [大M实验总账](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)：固定ADR-015的最终TTFT目标、L1完整Stock、L2真实40目标站点、**L3无Profiler真实Prefill设备Step先决门槛**、L3_GO后独立L4，并以唯一任务ID清单管理做什么/解决什么/责任/验收/当前状态/实绩/下一动作；附历史证据、风险、实机run_id记录表和增量进度日志。
- **首次状态**：M-01/M-02已正式归档；A1离线验收和只读G0工具只在 [PR#9 Draft](https://github.com/10wvw01/vllm-ascend/pull/9)，隔离副本CPU20项通过不等于仓库CPU回归通过；**当前原仓CPU/现场四卡只读G0、L1 Smoke/Screen/Confirm、L2短Profiler、L3无ProfilerStep Screen/Confirm与L4正式TTFT都没有新实机或正式原仓PASS记录**。
- **近期两项并行准备**：研发侧先审查L3无Profiler设备Step可靠测量能力（现有U1/旧capture链尚未证明此能力）；用户许可的现场第一阶段仅可进行A1原仓CPU回归和四卡只读G0，审查通过后才分阶段申请后续Smoke与ABBA。当前仍保持Stock产品默认，Gate E与首写/双Rank协议风险不能越级。
- **归档约束**：以后每次现场进度必须更新总账任务状态、真实结果证据、批次与日期日志；项目级新事实同步此控制台，已审核性能数字同步[性能基线](PERFORMANCE_BASELINES.md)，新决策再追加[ADR](DECISIONS.md)；禁止另建工作分支的“替代总账”。

## 18. 2026-10-27｜服务器切换其它分支：A1重新进入实机前的MemFabric源码/部署取证

- **环境已漂移**：用户明确310P3服务器切换其它分支运行其它任务，旧G0/安装SO/真实Python import/卡占用与权限全部作废；任何编译或部署首先只读枚举四卡、git worktrees及Host双路径/设备SO/KVER/当前CANN，且不能驱赶正在使用设备/共享库的其它用户。当前服务器新G0 **NOT_RUN**。
- **重大可复现性缺口**：2026-10-23历史已实证`ASCEND_RT_VISIBLE_DEVICES=2,3/6,7`中`MallocPhysical`需逻辑ID、`aclrtMemSetAccess`需物理ID，经过验证Host SO完整SHA `1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30`。2026-10-27复核`10wvw01/memfabric_hybrid_310p`当前7个分支的三个授权点**尚未发现物理ID修复**；历史修复可能只存在服务器`/tmp/opencode/mf-fix`未推Git源码/patch及SO备份，须先寻找，不能从镜像main盲编译覆盖装机。为保底创建[MemFabric Draft PR #1](https://github.com/10wvw01/memfabric_hybrid/pull/1)，**源码候选未完成原仓CPU/CANN/NPU验证、不等于历史实机合格SO**，禁止直接部署或篡改U1冻结SHA。
- **首批现场可委托范围 BATCH-R0**：按 [A1重进场与编译部署批量准入合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_REENTRY_BUILD_DEPLOY_AND_BATCH_GATE_20261027.md) 在**隔离工作树**一次完成四卡/环境只读快照、原仓A1 CPU和bash/python静审、旧修复源码/完整SO指纹查找、可行的隔离staging编译与备份/回滚计划。**同批不准**设备初始化/建池/KFC预部署、全局`pip install`/双Host SO覆盖、Smoke/Screen/Confirm；结果回传审核后再逐阶段放行。
- **任务状态入账**：[实验总账](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)新增E-01/02/03及B-07记录环境漂移和修复溯源；A1只读工具与CPU测试、实施文档在Draft PR#9。没有任何新服务器原仓CPU/设备PASS或当前D4 L1/L3/L4正收益数据，Stock默认与Gate E STOP纪律不变。

## 19. 2026-10-27｜A1 BATCH-R0原仓CPU/四卡取证通过，历史MemFabric修复与SO成功找回

- **正式证据审查**：[R0独立验收与R1阻塞清单](A1_BATCH_R0_ACCEPTANCE_20261027.md)；现场[原始CPU、修复Patch、环境快照与结果索引](https://github.com/10wvw01/vllm-ascend/tree/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r0_reentry_20261027)，run_id`a1-r0-20261027-001`。A1现场执行SHA`235de9f6...`、归档HEAD`73114fd...`，隔离工作树未修改其它任务。
- **原仓CPU PASS**：`unittest`34/34、`pytest`88项+27 subtests、`py_compile`、`bash -n`及U1冻结SHA/审批闸静审通过；纯CPU物理ID映射15/15。此项是原仓CPU结果，**不是NPU Smoke或L1性能结果**。
- **MemFabric非零设备映射的历史源码终于找回**：服务器本地未推送commit`5802f55026c16e153e783d8ef26a29f813d63dd9`（三处`aclrtMemSetAccess`换物理号、MallocPhysical仍逻辑号），两份完整历史修复SO SHA`1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30`均在本地归档。旧源码Patch已纳入A1证据。**不必按兜底Draft PR#1另造新版Host SO；后续应将实际旧commit补推到正确MemFabric仓库**。
- **服务器四卡只读取证**：16384(dev2,3)在采样时双Die OK、无可见进程且hugepages0/0，推荐作为唯一候选；5664被其它任务占用；38144 dev4有约13GB残留/hugepages全占；43520 dev7 MCU/Chip Alarm且被占用。**授权和启动前仍需复核16384的实际独占、卡健康及环境映射**；不处理其它租户进程。
- **部署仍未准备就绪**：当前共享Host系统/Python双路径SO为`88d39285...`（未修复），当前插件目录Device SO为`b8ca96f0...`（非D4），唯D4归档SHA`2a19390c...`、插件`8eb65de9...`、KVER11`25e16b0a...`已就绪。不能把候选SO路径存在当成**实际装载通过**，不能直接修改他人工作的共享安装环境。优先审查历史S3-R/S3-R2的进程级`LD_PRELOAD`/隔离Device SO方案；若不能隔离必须独立维护窗口和快照可回滚、用户批准。
- **门禁最新状态**：`R0_EVIDENCE_ACCEPTED / CPU_PASS / PROVENANCE_FOUND / G0_READONLY_OBSERVED / G0_APPROVAL_PENDING / DEPLOY_NO_GO / SMOKE_NOT_RUN`；U1实际G0、M2048 Smoke、L1 Screen/Confirm、L2设备归因、L3设备Step、L4正式TTFT**均无新运行结果**。Gate E/507014、首写与KFC首调安全阻塞仍保留；Stage E2/R1须独立授权，不因历史S3-R2 25/25短场正确就认定当前正式发布安全。
- **同步唯一实验总账**：[大M Prefill TTFT实验总账](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)已更新E-01/02/03、G-01、B-01/04/07、R0原始取证批次和进度日志。主线负责目标/真相，A1 PR#9保持Draft只负责实现。

## 20. 2026-10-27｜R1：历史MemFabric原补丁已落正确GitHub仓库；隔离装载先于设备Smoke

- **长期源码源头已修复**：将R0找回的`5802f550`原始Patch恢复到`10wvw01/memfabric_hybrid_310p`，3个被修改源文件分别与历史目标Git Blob SHA完全一致；[PR#1](https://github.com/10wvw01/memfabric_hybrid_310p/pull/1)已合入镜像主线`main@0670f3c7db7cfb090e8a312b9347fa039ada7da2`。这不是同SHA的Git提交（GitHub重新提交），也不代表本轮已重编或新NPU验证；**旧经过S3-R2实证的Host SO SHA`1dc46d7d...`仍为唯一复用二进制**。另一仓语义不同的备用PR现已关闭，不作为装机来源。
- **R1最关键的新审查**：U1现有SHA门禁只证明`U1_DEVICE_SO_PATH`候选文件有效、`U1_CORE_SO_*`路径字节有效，却**没有证明运行中真正注册的是D4 Device SO**；同时A1 Python插件editable导入依赖另一任务工作树，生产系统/Python Host SO仍旧`88d39285...`、Device SO实际在装旧`b8ca96f0...`。未经私有化Loader与Device实际身份审查，禁止直接运行M2048 Smoke或修改共享安装。
- **实施已准备，尚无新测试**：A1实施PR#9新增`test/perf/a1_r1_loader_readonly.py`及CPU拒绝性UT、[R1A静态/只读与R1B受控Smoke分段合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R1_ISOLATED_LOADER_PREFLIGHT_AND_SMOKE_GATE_20261027.md)。只读工具严格不导入torch、不dlopen、不触NPU，显式输出实际SO装载未被证明，**原仓CPU及R1现场运行仍NOT_RUN**。
- **当前允许的下一次委托**：单批**BATCH-R1A**（新隔离工作树→四卡/租户重扫→7个候选/在装文件SHA与ELF静态依赖→私有Host/D4 staging copy→Python插件/Device注册路径追溯→真实原仓CPU回归→可行性、维护窗口/回滚计划）。全部只读/CPU，不安装SO、不建池/KFC、不执行Kernel；回传证据后才可能独立许可**R1B**（真实装载证明、16384独占复查和最小M2048 Smoke）。R1B Smoke仅正确性准入，不会自动进入L1 Screen/Confirm/L3/L4。
- **唯一进度台账**：[大M Prefill TTFT实验总账](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)新增E-04与E-05，E-02源归档完成、B-07运行时装载阻塞未解。主线继续Stock默认；Gate E首写/507014、Die7 Alarm/共享租户和全部阶段STOP约束不放宽。

## 21. 2026-10-27｜用户最新授权一次进场：环境→M2048 Smoke→L1→L2→L3，按真实门禁连续推进

- **最新授权覆盖旧组织流程**：无需再反复回来申请Smoke/Screen/Confirm/L2/L3各个正常步骤；只要R1真实隔离环境/同卡双Die/源码SO装载/回滚和CPU均通过，现场同一OpenCode委托可以运行M2048 Stock/V2两Rank Smoke；**Smoke双Rank位精确、协议、首次KFC/首写均正确，并通过`a1_r1_smoke_gate.py`现场证据核验后**，继续完整Stock L1 Screen→HCCL跨流完成审计→Confirm（ABBA×2）→真实Qwen3.6双Rank 40层L2短Profiler→无Profiler L3 Prefill Event四Serve ABBA。若L3_GO、设备安全与正式业务脚本都符合冻结合同，后续正式L4仍不得跳过L3直接开跑。新用户许可**不包含**共享SO/别人的editable代码写入、设备异常后换卡重试或跳过硬门。
- **已准备的A1工具**：Smoke纯CPU严格验收、L2逐层实际Kernel/HCCL/Copy和每Rank40站点证据校验、测试环境开关的`forward→logits` Device Event（无任何额外sync）、L3双Rank四场Stock–V2–V2–Stock原始事件对照、短Profiler/无Profiler模型请求器和隔离Serve封装、对应CPU测试。详见[一次批量实验实施合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R1_ONE_BATCH_THROUGH_L3_20261027.md)。**目前只是代码提交及静态检查准备，原仓CPU/实机事件API与完整Prefill依赖计时覆盖仍需现场验证**，不能提前记为L2/L3 PASS。
- **为什么要真实L2校准L3**：Event只取真实模型Forward入口→Logits产生，若没有与设备Trace/跨Stream依赖校验，它**尚不能冒充整个Step关键路径**。实际目标Kernel必须按10个o_proj+30个out_proj核实，不能凭Profiler Kernel名称或40次Kernel时长相加签L3；若无可审计完成边界，输出`L3_MEASUREMENT_NOT_READY`，保留本批可用L1/L2结果，不执行正式L4。性能负增益本身不取消L2诊断或大M继续优化。
- **主线与实施分工**：主线[实验总账](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)增加E-06任务/本次最新授权；A1 Draft PR#9只承载新测试代码与运行报告。未取得新310P实测数据、Stock默认、Gate E/507014和首写安全发布阻塞不变。

## 22. 2026-10-27｜A1-R1单批实机归档：工程安全通过、真实模型L3无正向收益

- **执行与当前版本**：A1工作分支`ff885ff559efea6523f8026e24704a0797c2ef8b`完成`a1-r1-batch-20261027-001`；[实机原报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027/A1_R1_BATCH_REPORT_20261027.md)、[主线独立复盘](A1_R1_实机复盘与优化优先级_20261027.md)、[唯一实验总账](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)。本次复核是远端原始证据与20份逐站点JSON的静态验收，未重新访问服务器原始整包profiler CSV。
- **安全/正确性准入已通过**：16384(dev2,3)同卡独占；私有PYTHONPATH+LD_LIBRARY_PATH成功装载实际修复Host`1dc46d7d`、D4`2a19390c`、插件`8eb65de9`、KVER11`25e16b0a`；`/proc/maps`按PID取证且原共享双Host/Device SO未改；CPU108 passed+27subtests；M2048 Stock/V2双Rank位精确/KFC/首写status=0，`ALLOW_SCREEN`。首次PYTHONPATH遗漏tbe错误500001、清理命令误杀自身和L3末条事件35/36已修复且留原证据。**这不代表历史发布Gate E/507014全关闭**。
- **核心性能事实**：U1 row-major Stock基线8场ABBA：M2048 V2+22.5%，M4096 V2+21.1%，不能以此宣称实际模型提升；真实Qwen `aclnnMatmulWeightNz` 约752µs、Stock HCCL AR约1391/1166µs，融合D4约2511/2430µs，V2每Rank每目标站点分别慢约368/512µs（近似），双Rank各40站点已取证且无额外contiguous copy。L3 input4096/output1、TP2、并发1、无Profiler四ServeABBA，V2慢26.12/27.70ms，Stock跨会话漂移2.83ms，V2有周期性长尾。**L3未GO、正式4096/2048/并发10/50请求L4_NOT_RUN_BY_DESIGN，Stock默认继续保留。**
- **方向P0**：①在原A1工作分支把U1 `Stock_NZ MM+HCCL`改成Serving同源主基线，row-major仅诊断；②真实短窗剖面/计时分解D4 MM/SDMA/IB/Core0/Add及设备依赖，**尚不能判定仅靠NZ化就足够**；③调查V2约6～7请求周期重尾；④新Kernel单变量优化后同批按正确基线跑L1/L2/L3，有真正L3_GO才正式L4。性能负结论不终止大M研发，但当前D4不得转正式默认/发布。数据见[现版性能基线](PERFORMANCE_BASELINES.md)。

## 24. 2026-10-27｜R2单批工具就绪（静态）、首轮原仓CPU和实机尚未运行

- **实施分支**：`work/310p-mm-ar-large-m-a1-20261026`的[单批执行合同](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/docs/A1_R2_NZ_STOCK_25MS_周期长尾联合诊断合同_20261027.md)已更新，提供可执行入口`test/perf/a1_r2_one_batch.sh`：CPU/真实NZ路由→Stock_NZ/正式D4双RankSmoke及8场ABBA→五种D4机制Case（可选短Profiler）→15s/5s/5s/15s四Serve长尾对照→原R1证据复算。**正式Kernel/D4 Device SO、生产Stock默认、MemFabric冻结Host及KVER11不改**。
- **严格身份与失败防护**：NZ原生`torch_npu.profiler.tensorboard_trace_handler`必须真实导出四份CANN解析CSV且每Rank/M的MatMulWeightNz及HCCL唯一配对通过；所有子进程继承同一完整冻结Host/D4/Plugin/KVER SHA。循环的独立进程池、双Rank错误/前后卡状态、`STOP_STAGE.json`及私有运行目录必须保留。没有G0/独占/回滚或共享SO无法隔离则NO_GO，不清KFC、不自动换卡。
- **机制归因真实性**：现有`formal/loop4/diag0/diag4/diag5`有明确模式差异，正式D4的2.5ms**不能**用模式时间差精确拆成MM/SDMA/IB/Add，独立内分段设备指标未出现时`INNER_PHASE_UNKNOWN`。Profiler短窗用于定位Whole Kernel及可用CANN硬件指标，不与正式无Profiler L1样本混合。
- **Epoch风险**：KVER11正常换纪监督循环不逐次打印完成日志，R2脚本同时检查`MF_SDMA_ORCH_EPOCH_BUDGET_S`显式预算和每次Serve两Die`sdma orch configured ... epochBudgetS`运行配置；日志不可见标`EPOCH_RUNTIME_BUDGET_UNVERIFIED`，即便峰频率随15s/5s改变，也只能支持假说，不签因果已证明。需要实际同源时间戳或经授权的设备/Host进度探针才能定责。
- **测试状态**：已提交新增CPU合成拒绝性UT、原仓CPU/静态验证命令和独立A1工作分支专用Actions工作流`.github/workflows/310p_a1_r2_cpu_only.yml`；截止此次远端审核**未观察到Actions执行记录**，也未进入310P3，因此`R2_CPU_NOT_RUN / R2_G0_NOT_RUN / R2_NZ_NOT_RUN / R2_PHASE_NOT_RUN / R2_EPOCH_NOT_RUN`。本轮主线仅记录实施准备，不提前宣称P0根因或加速。
- **下一步**：用户在310P3委托OpenCode按当前A1 HEAD开展**一次**完整R2诊断；先原仓CPU及G0，真实私有加载成功即按机器门连续推进，出错原地STOP；回传后由主线审原始CSV、Rank JSON、Profile和epoch日志，再确定实际要改哪个Kernel/通信变量。

## 25. 2026-10-27｜A1-R2三次现场尝试复盘：真NZ修复成功，Confirm/长尾因独占失败而未闭环

- **现场证据与代码身份**：[R2正式报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_FINAL_REPORT_20261027.md)、[N0→STOP历史](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/A1_R2_BATCH_STOP_REPORT_20261027.md)、[总账](LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)、[性能基线](PERFORMANCE_BASELINES.md)。工作分支当前归档HEAD`e1c3b3d5`，实机attempt3运行HEAD `2718f41dc`；Host`1dc46d7d`、D4`2a19390c`、插件`8eb65de9`、KVER11`25e16b0a`与R1一致，私有加载不覆盖他人SO，卡16384/dev2,3起始健康。CPU两项合同/评估器测试缺陷修复后**151 passed+27子测试通过**，不以最初2 failed否定修复后整批。
- **P0-1真实NZ基线的工程问题得到关键突破**：Attempt1裸torchrun只有通用`MatMul+TransData`（~1.8ms），即使`npu_format_cast(29)`也不等价于Serving；修复只对U1 NZ Stock会话开启`allow_internal_format=True`和`jit_compile=False`，后Rank0/1×M2048/4096各6组真实`aclnnMatmulWeightNz`+相邻HCCL证明通过（由[原生CANN CSV路由门](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_route/nz_route_verified.json)确认）。M2048双RankSmoke PASS；[有效Screen JSON](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_screen/u1_analysis.json)显示真NZ Stock vs D4 M2048=1057.474 vs1367.878µs（V2−22.692%），M4096=2006.133 vs2539.831µs（V2−21.013%）。R2正式报告正文的Screen粗写672/1067/2023µs与机器源有不完全一致，主线以源JSON为准。
- **八场Confirm必须作废，不允许挑场重算**：[原始Confirm机器判定](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r2_batch_20261027/attempt3_completed_stages/nz_confirm/u1_analysis.json)明确`INCONCLUSIVE_DRIFT`，M2048 Stock场间漂移79.675%、M4096 663.527%，M1024 2080.483%，且`audit_sha256=null`。外部进程轮换与污染时段高度相关，但其中04/05性能异常缺同一请求时基的外部PID证据，具体因果不能越权确认。**真实NZ性能方向强烈指向D4落后，但L1正式加速比仍`NOT_SIGNED`**。
- **P0-2机制诊断虽完成但没有内部分段**：Formal M4096 Event 2635.925/2617.149µs、Loop4 3234.206/3213.112µs、Diag0单Chunk 6378.946/412.448µs、Diag4 ~27ms、Diag5 ~1.02ms；短Profiler证明Formal单个D4设备Kernel p50 ~2343/2355µs。诊断模式非等价，不能用Mode差值把MM/SDMA/IB/Add的实际时间拆出，**`INNER_PHASE_UNKNOWN`**；后续先核CANN9.1在310P3可用的真实设备内阶段计时/低扰动探针，再单变量调优。
- **P0-3 Epoch与业务请求因显存抢占而NOT_RUN**：E15A首Serve实际空闲27.33GiB<90%显存预算要求38.92GiB，Engine初始化按防护拒绝，`STOP_STAGE=EPOCH_ABBA`；后续5/5/15均未启动。R1的双Rank周期尖峰及15秒预算相关假说不变，但新因果状态`EPOCH_CAUSAL_UNKNOWN`。
- **P0前置安全约束升级为外部调度独占**：Attempt2曾在初始G0转为空闲后两分钟再次被别的宿主进程占用，随后本批进程树异常消失，其机制/操作者未知。Attempt3亦遇到dev2/dev3反复短暂占用。需要管理员协调约80min**连续双Die独占租约**与全程宿主级只读进程/显存/hugepages监视，不能仅在脚本开头拍一张G0快照就当整轮独占。若无租约，优先停止重复跑Confirm/Serving；任何未知抢占不自动换卡、重启、杀他人进程或覆盖共享SO。
- **下一步工作划分**：现场有稳定窗口时，尽量只补最新真NZ Stock八场ABBA Confirm+原生HCCL完成独立审计、Epoch 15→5→5→15无Profiler四场，原先有效路由/Smoke/机制证据不用重复耗时跑；研发侧并行设计**单独实验编译产物**的D4正式路径设备阶段探针，并验证CANN/310P3可用性。新L3收益仍无正向，**Stock默认与L4 Gate E历史阻塞不变**。


## 26. 2026-10-28｜E3已测证据回归主线、E4分层流水正式开工（ADR-020）

**主线权威裁决**：[ADR-020](DECISIONS.md)、[E3机器证据审核与E4准入合同](E3_实机证据审核与E4分层流水准入_20261028.md)、[L1基线](PERFORMANCE_BASELINES.md)。E3实验源码及全量原始Evidence仍由[PR #20](https://github.com/10wvw01/vllm-ascend/pull/20)工作分支承载（`run_id=e3-chunk2048-20261028-001`）；本次将**经审定事实、决策和下一阶段唯一目标选择性写入主线**，不将PR #20全部代码与过期治理文档合入主线，不动Serving/正式Kernel/Stock/MemFabric。

- **实机状态：E3完成**。物理卡16384 dev2/3、真实NZ路由4/4、修复后E3-G/T双RankSmoke和八场`stock,e1,e3g,e3t,e3t,e3g,e1,stock`全部完成，16份正式RankJSON bit-exact/status0，正式冻结SHA前后一致。E3-T原L0C满载首个真实Kernel触发507015→`basicM256→128`修复重新构建PASS；E2-P此前507014根因仍未知。M4096配对p50 Stock2022.553/E1 2271.836/E3-G2119.506/E3-T2267.090µs；**E3-G相对E1+7.187%，仍慢Stock96.953µs**；M2048全批E1跨场漂移6.192%>5%，整批`INCONCLUSIVE_DRIFT`，E3-G慢E1方向稳健。任何版本不称E3“超过Stock”。
- **精确收益归因待取证**：E3-G相对E1保持四次1024-MM和四条4MiB传输；producer/consumer整核DCCI、ReadyTree、AddRelease从4→2次，其余调度/内存所有权改变。152.330µs只能作为整体净回收，不能拆分宣称DCCI或IB独占。M2049～4095**全部未实测**，现有Serving仅支持1024/512/256的Padding，E3私有Host只接受M2048/4096。
- **E4开工**：新增**唯一**E4隔离分支`work/310p-mm-ar-e4-layered-pipeline-20261028`，基于已测E3代码，E4-0 AddTile单变量，E4-A包级按Core0～3/4～7分两批Add并保持唯一mail消费者，E4-B连续M只先建立可验证的混合Chunk/尾块规划合同。新的设备端行为必须经过两轮静态深审、UT和独立CANN编译，现均`NOT_RUN`，不得直接让OpenCode运行正式八场或改生产Serving默认。
- **下一轮阶段口径**：先E4-0单变量+E4-A协议审查及最小可运行候选→双Rank安全Smoke→短Screen（Stock/E3-G锚点M2048/4096）→连续M边界专项测试→胜出者正式确认。优先调查E3-G vs Stock M4096最后96.953µs，随后解决M2048单2048组失去交叠的问题与非对齐Padding。下一层真实40站点/L3/L4在E3/E4下仍NOT_RUN。
- **证据补全P1**：PR #20报告引用的`bugfix1/smoke_e3t.log`、AICore摘录、`build/`manifest以及`card_watch_30s.log`当前仓库证据目录未见，要求现场核实原始文件并只读补档或说明SERVER_ONLY；本次主线事实不得伪称其已完整落盘。


### E3证据已脱离PR#20独立保全（2026-10-28）

在本主线`evidence/e3_chunk2048_20261028/`现已**逐文件原样保存52份**E3关键原始文本证据：机器评估、16份正式Rank JSON、4份Smoke、8份Stock NZ路由JSON/CSV、G0、前后二进制SHA和逐场设备记录、原实机中文报告等；所有归档文件与原E3实验分支的同名文件**Git blob SHA完全一致**。详情：[主线归档Manifest](../evidence/e3_chunk2048_20261028/MAINLINE_ARCHIVE_MANIFEST.md)。
**更正上述“E3原始Evidence仍由PR#20承载”的历史说法**：目前PR#20仅为额外的大型Profiler和实验Kernel代码来源，不再是主要L1机测结果的唯一保全位置。构建日志、507015原始PLog、30s巡检因源分支未上传而保留待补状态。


## 27. 2026-10-28｜E4 Attempt1 G0/CPU/编译/NZ/E3G Smoke PASS，外部占卡 STOP（ADR-021）

- **正式状态**：E4 Draft [PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)、执行源码`1e20d3522cc9dd8040b257fae8d887e7d12f3b5e`、归档`5b727e9f9153e8292c6022a14b63e974dfb0db03`、run_id `e4-layered-20261028-001`；报告、G0、真实NZ路径、E3G Smoke及宿主占用证据已**按原文与源Git blob相同**选择性归档本主线23份：[E4 Attempt1归档清单](../evidence/e4_layered_20261028/MAINLINE_ATTEMPT1_ARCHIVE.md)。E4的CPU合同、2049M纯离线规划、AddTile2048/4096/8192与PacketAdd私有BiSheng编译全部PASS；真实NZ路由Rank0/1×M2048/4096四组PASS；E3-G双RankSmoke bit-exact/status0 PASS。
- **STOP及性能真实性**：G0时16384 dev2/3空闲、KVER11和四冻结SO身份一致，运行中UTC06:29:39首份card_watch发现宿主PID506263已占四卡且持续增长；E3-G Smoke被同一时间的并发资源污染，M4096耗时约2.3/6.5～6.8ms交替，不能签任何正式新性能。E4-T2048因执行工具SIGTERM在Kernel前终止，非E4错算；E4-T4096/8192/PacketAdd全未运行；**E4四新候选均无设备正确性和速度结论，七臂Screen NOT_RUN，十四场Confirm从未授权**。相关[中文原始报告](../evidence/e4_layered_20261028/A1_E4_第一轮实机验证报告_20261028.md)和[STOP取证](../evidence/e4_layered_20261028/stop_incident/STOP_INCIDENT_ANALYSIS.md)可在主线独立打开。
- **下一步强制修正**：停止无效重跑；由我方研发在同一E4 PR中补G0后持续独占**主动STOP**检测，不可仅做30s日志、后验发现外部PID；执行器需要真正独立session与明确进度/结束状态轮询，避免工具超时SIGTERM。审查、负面CPU合同通过后，取得目标物理卡连续双Die排他租约，重新G0并核源码HEAD+候选SO SHA，执行E3G/E4-T2/T4/T8/P双RankSmoke，再短Screen；任异常STOP。**未见新性能前不启十四场Confirm，不擅自打开Serving或变长M路由。**
- **路线保持**：E3-G仍为M4096研究基底，旧E3轮完整L1 M4096 E3-G2119.506µs、StockNZ2022.553µs（差96.953µs），但不跨批推导E4收益。E4-B目前仍只覆盖2049个整数M的CPU规划，不支持真实设备尾块。Qwen L2/L3/L4在E4下仍NOT_RUN。


## 28. 2026-10-28｜E4 Attempt1工程P0最小测试保护已经编码，复测仍待实机

- **实现位置**：原有唯一E4 [Draft PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)，`work/310p-mm-ar-e4-layered-pipeline-20261028`。新增`test/perf/e4_stage_guard.py`使`e4_campaign.sh`全部四类torchrun（真实NZ/Smoke/Screen/Confirm）被阶段级只读npu-smi监控包裹：每次开跑前目标物理卡双Die健康/无人占用检查、运行中2秒监视跨卡同PID/超额进程/接口异常STOP、结束后补采，异常保存`lease_guard.jsonl`、`lease_guard_stop.npu.txt`且**仅终止自己创建的torchrun进程组**；原30秒卡巡检保留。
- 新增`test/perf/e4_detached.py`两层独立Session、`launch.json`/`result.json`明确退出码、`start/status`查询，防止工具超时误杀；`E4_CONFIRM_APPROVED=YES`的未经批准Confirm不允许经此入口启动；全部Stage之前需核`E4_PHYSICAL_CARD_ID`及新Git HEAD与Build manifest匹配。附纯CPU负面测试`test/ut/test_e4_test_protection.py`与GitHub Actions语法/单测入口。
- **状态不能混淆**：此项为工具源码实现和静态检查，**新HEAD CPU pytest/CI、BiSheng重编、G0、E4各候选Smoke/Screen均未执行**。E4 Attempt1旧轮CPU/Build PASS只对旧HEAD有效。E4旧基准Stock/E3-G数值不更新，生产Serving Stock默认不变，十四场Confirm仍需用户独立批准。
- **仍需真实独占**：仅凭容器内npu-smi，无法证明“目标卡单独进入”的某个宿主PID是否属于本次torchrun，也无法检测2秒间隙内的极短抢占；因此**宿主/调度器连续双Die排他租约是永久P0约束**。方案和最简上机命令在[E4保护操作说明](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-e4-layered-pipeline-20261028/csrc/_310P/memfabric_mm_ar_v2/docs/A1_E4_最小实机测试保护与复跑操作_20261028.md)。这次只做测试工具，不合入实验Kernel。


## 29. 2026-10-28｜E4 Attempt2完成全部允许阶段 + Attempt3保护机制验收（设备Smoke/Screen因外部分时占用BLOCKED）

- **attempt2（执行HEAD `5b727e9f9`，早于守护代码合入，结果不因守护代码后合入而失效）**：16384独占窗口完成**五臂双Rank Smoke全部PASS**（e3g/e4t2/e4t4/e4t8/e4p，bit-exact、status=0、首写成功；E4四臂为首次设备执行，e4t8无507015类资源异常，e4p的Core0唯一邮件消费者/分核释放协议实机成立）与**七臂Screen**（30样本/臂，`SCREEN_ONLY_NO_SIGNED_PERFORMANCE`）：M4096配对p50 Stock 2037.7 / E3-G 2133.9 / **e4t8 2076.2µs**——e4t8相对E3-G **−2.71%（−57.7µs）**、对Stock差距从96.2µs（复现E3 Confirm的96.953µs）缩至38.5µs；e4t2与E3-G差0.10%（同构重编排除重建漂移）；e4p −1.34%；**M2048四臂与E3-G持平、仍落后Stock约32-34%**（单组交叠结构性损失，消费者侧无解）。全程零外部进程、结束后卡回基线、前后二进制SHA一致。Confirm未授权未运行。
- **attempt3（保护机制验收，`05d679341`修复→`1bfacb251`证据，最终见PR #21分支log）**：CPU（原仓UT 93+9、保护专项13、E4合同9+9、静态门）与无NPU真进程故障注入46/46全PASS；**修复真实BUG**——`e4_detached.py`的SIGTERM handler内在持有`Popen._waitpid_lock`的`child.wait()`被打断上下文内再wait同一child导致**永久futex死锁**（worker挂死/result.json不落盘/进程泄漏，100%复现），修复为handler只raise、清理移到except块，回归UT双盲验证。**守护器实机四场景全部实证**：受守护双Rank NZ路由全生命周期无误杀（真实NZ路由4/4）、preflight拦截真实单PID（2433615）与双PID（2620627/2621272）外部入侵、**运行中拦截第三方进程（2640164）并只清理自己**——attempt1的506263事故模式真实闭环。隔离重编候选SO与attempt1/2逐字节一致（`08f73f2c…`），四冻结SHA复核OK。
- **attempt3的Smoke/Screen为BLOCKED（非NOT_RUN_BY_DESIGN）**：16384被宿主级外部负载持续分时占用（阵发40-90秒任务、最高34GB、占空比~50%、最长安静窗口约8分钟，与§25 dev2/dev3模式一致；五臂Smoke需~12分钟、完整campaign需35-45分钟）。两次有界campaign尝试均被守护器正确STOP保全；未杀外部进程、未换卡、未清KFC。
- **当前裁决**：E4设备正确性已由attempt2 Smoke证实；e4t8为Confirm最优先候选（次优先e4t4、e4p暂缓、M2048需新结构候选）——但**进入正式Confirm的工程前置（保护验收/构建/G0/NZ路由）已满足、环境前置（连续双Die独占租约）未满足，管理员协调租约仍是P0**；任何后续实机前须在新HEAD重新隔离构建（manifest强绑定）。attempt2 Screen方向不因attempt3无复测而失效（候选SO逐字节一致）。[attempt3验收报告（E4分支）](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-e4-layered-pipeline-20261028/csrc/_310P/memfabric_mm_ar_v2/evidence/e4_layered_20261028/A1_E4_attempt3_保护机制验收报告_20261028.md)、[主线归档33文件](../evidence/e4_layered_20261028/attempt2_3_mainline_archive/)与[归档说明](../evidence/e4_layered_20261028/MAINLINE_ATTEMPT2_3_ARCHIVE.md)。

## 30. 2026-10-28｜E4 Attempt3b用户授权换卡38144：五臂双Rank Smoke完成、守护器无扰动实证、Screen 4/7场（待Confirm裁决）

- **换卡依据与安全**：用户明确授权换卡（解除"不自动换卡"约束）后按ADR-014选**38144**（双Die+MCU健康、全天零进程；5664 die1长驻租户、43520 die1 Alarm排除）。受守护映射探针实证`ASCEND_RT_VISIBLE_DEVICES=4,5`仅落38144双Die；已知风险die4 hugepages 5860/5860无主满载**实测不阻塞**（matmul/HCCL/MemFabric建池全部成功）。新HEAD `e818334de`隔离重编候选SO第三次逐字节一致（`08f73f2c…`）。
- **五臂双Rank Smoke全部PASS**（e3g/e4t2/e4t4/e4t8/e4p，bit-exact/status=0/首写/池创建）：**E4四臂设备正确性在第二张卡复证attempt2结论，委托核心设备回归完成**；真实NZ路由4/4。
- **受守护Screen 4/7场后STOP**：stock/e1（campaign内）、e3g/e4t2（手动续跑，同一守护调用）完成；screen_05_e4t4运行中外部进程2932134入侵→守护器3>2检测→STOP清理我方进程组（第5次实机保护实证）；e4t8/e4p按no-retry合同未启动。守护审计10阶段零跨卡/零超额PID。
- **守护器无性能扰动实证**：38144受守护 vs attempt2的16384无守护——stock锚2025.8 vs 2037.7µs（+0.58%）、e3g锚2142.3 vs 2133.9（+0.40%）、**p99近乎逐值复现**（e3g 2227.2 vs 2228.2、e1 2426.4 vs 2408.8）；同构对照e4t2-vs-e3g本轮−1.55%（attempt2 −0.10%）标定单场Screen噪声带±1.5%→**e4t8的−2.71%仅约1.8倍噪声带，Screen只筛方向、正式裁决必须十四场Confirm**。
- **当前状态**：e4t8/e4t4/e4p三场Screen因外部调度扩散至38144未跑成，e4t8方向维持attempt2原判（对Stock差距38.5µs）待Confirm；四卡均无稳定独占窗口，**管理员连续双Die独占租约仍是进入Confirm的P0前置**。attempt3b零Kernel/Serving/Stock/共享SO改动、未杀租户、未清KFC。[attempt3b报告（E4分支）](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-e4-layered-pipeline-20261028/csrc/_310P/memfabric_mm_ar_v2/evidence/e4_layered_20261028/A1_E4_attempt3b_换卡38144报告_20261028.md)、[主线归档新增12文件](../evidence/e4_layered_20261028/attempt2_3_mainline_archive/)。


## 31. 2026-10-29｜V3 SDMA ADD代码开发与一次快速算子验证合同

用户已批准完整V3方案，并进一步指定不重跑V2/真实NZ Stock性能臂，只测V3并复用历史数据。
从本主线`54d5518d`派生`work/310p-mm-ar-v3-sdma-add-20261029`，新增[独立V3工程](../../memfabric_mm_ar_v3/README.md)：
公共signal_at ADD_FP16执行分片Reduce，COPY All-Gather恢复完整结果，MTE输出搬运；不含本地Vector Add。
MM保持已有1024子调用/basicM，支持256/512/1024/2048组、串行/一组前瞻、显式尾块padding及Torch C ABI薄适配。
私有构建不修改现有V2/Serving/共享安装，依赖冻结MemFabric `beaf490b`/KVER28，公共头与全部SO需重核身份。
快测在单次有界会话逐关推进IB→FP16 ADD→协议/回绕→融合/尾块→V3 M2048/4096双组长两短轮。
对照由归档JSON/性能表生成，保留历史漂移与Screen标签；不同卡/SDK/批次不能签为正式因果收益。
设备选择使用未重映射的直接DeviceID并经CANN验证；守护仅停止自有进程，独占窗口由现场协调。
**CPU合同与两轮静审已完成；CANN编译、V3实机正确性及性能均NOT_RUN。**
Opencode可先私有编译，成功且G0满足后按脚本执行短测；失败STOP，无自动重试或旧性能臂补测。
不更改E4历史裁决；不启用生产默认或DFlash/Graph，不触发L3/L4。


## 32. 2026-10-29｜上游MemFabric源码树同步与V3执行单固定

用户明确授权推送我方V3工作分支，并要求研发侧完成全部仓库/分支/编译部署规划，OpenCode只执行。
GitCode `dev-v1.2.0`最新`ba3e7f42`相对旧镜像增加SetAccess物理号修复；
GitHub镜像同步为`b9d46f97`，完整tree均`59f1fb6f`。使用插件快照提交保留GitHub历史，提交号不相同但源码树相同。
V3依赖改锁镜像`b9d46f97`，ADD/公共头/KVER28不变；上游占位自拷贝差异仍未改变。
执行规划见[V3 OpenCode执行单](../../memfabric_mm_ar_v3/OPENCODE_EXECUTION.md)，来源证据见[同步JSON](../../memfabric_mm_ar_v3/UPSTREAM_SYNC.json)。
新增prepare_sdk.py：新隔离克隆调用上游wrapper，关闭Python/HCOM/测试包，显式收集Host/头/AICPU三件套；
避免上游run包漏复制AICPU目录导致现场自行拼包。之后只编私有V3直调库，不替换共享SDK/Serving。
默认固定38144 dev4/5，在真实独占/G0成立后一次V3-only有界快测；失败STOP回传，由研发修复源码。
本轮没有服务器CANN编译或NPU结果；不能把镜像同步或CPU检查写为实机PASS。

## 33. 2026-10-29｜V3 R3全门禁通过但性能失败；chunk512/1024/2048重排（ADR-024）

- 证据归档在V3分支`6144af16`的[三轮记录](../../memfabric_mm_ar_v3/experiments/quick-20261029-r1r2r3/README.md)。
  执行`f2c3eaf0`、MemFabric`db38d75e`/KVER28：修复fork+exec助手缺失和守护器助手窗口误杀，
  CPU87、CANN、IB/ADD/协议/融合/尾块双Rank全部PASS，4配置×2轮×30样本完成。
- 逐样本max双Rank复算8场p50与原JSON一致。最佳group2048的M2048/4096为
  1709.941/2883.197微秒，较历史E4/E3-G慢17.29%/35.11%，较历史真实NZ Stock慢57.11%/41.49%；
  轮间漂移0.28%～1.37%。接受用户要求：当前V3性能目标失败，重新设计，不能以功能PASS替代收益。
- [V3.1复核与设计](../../memfabric_mm_ar_v3/PERFORMANCE_REVIEW_20261029.md)：
  chunk仅512/1024/2048，保留V2 MM tiling；将MM/Reduce/Gather分段推进，
  不再Finish上一组后才提交当前Reduce；验证1/2/4MiB包与双UB MTE，
  单Core0邮件分发器处理合法跨tile消息，保持DCCI/首写/精确一次ADD/最终quiet+DONE。
- 历史跨批对照仍不签正式性能GO；不重跑V2/Stock、不启模型或TTFT。
  本次只做证据审查和设计，V3.1实现/编译/设备均NOT_RUN。
- 职责更正：OpenCode负责执行及执行BUG修复、汇总采样/证据并提交远程；
  研发负责方案/协议、实现审查、结果归因和后续推进，覆盖§32旧“只执行不修复”的交接限制。

## 34. V3.1实现交付与编译准入

实现三段推进、合法交错邮件分发、双UB copy与4MiB包，chunk固定512/1024/2048。
CPU88项PASS（含实际C++调度器480场异步模拟及负例），设备CANN编译/NPU均NOT_RUN。
没有设备编译证据前不宣称实机准入已通过；服务器先编译再按G0和门禁执行。
完整条件与交接见[准入单](../../memfabric_mm_ar_v3/V31_ADMISSION.md)。Stock默认不变，不重跑历史V2/Stock性能臂。

## 35. 2026-10-29｜V3.1实机：编译/门禁/短测完成，性能未超V2

- 证据归档在V3分支的[quick-20261029-v31实验记录](../../memfabric_mm_ar_v3/experiments/quick-20261029-v31/README.md)。
  前缀`632370a`（用户交接锁定）+执行中守护器修复`2a5f3b37e`（设备so逐字节不变），
  MemFabric`db38d75e`/KVER28。CPU 88→89项PASS（新增守护窗口回归）。
- 私有CANN编译PASS（dav-2002）：双UB、NZ2ND临时区、MTE2_MTE3/MTE3_MTE2双事件对、
  Progress数组全部通过。38144双Die四会话：ib/add/protocol/fused/tail双Rank全PASS。
- 4MiB默认六配置（M2048/4096×chunk512/1024/2048）两会话各两轮完成，
  跨会话复现性≤1.77%、轮内漂移≤3.8%（多数≤1.5%）。1/2MiB包长诊断（4预热+10样本×2轮，
  独立目录）差异≤~1.5%，包长在[1,4]MiB对本组形状无实质影响。
- 执行BUG（已修复留证）：1MiB诊断轮守护器STOP(97)。根因：torch_npu在worker首个设备算子处
  fork异步编译基础设施子进程（~8s存活，继承~103MB设备映射，宿主PID无法映射回容器），
  在部署窗口内浮现在驱动进程表，超过守护器窗口容忍上限。修复对齐V3.0 R2→R3守护器
  演进先例：worker窗口标记提前到首个设备算子前；窗口容忍2+2→2+2+2=6；
  60s持续界限与全部fail-closed检查不变。无外部租户/设备故障/协议错误。
- **性能结论：V3.1未超过历史V2。** 最佳配置M2048/chunk1024=1656.0µs（V2 D4 1367.9，
  慢21.1%）；M4096/chunk2048=2516.1µs（V2 D4 2539.8，−0.9%在漂移带内，不可判胜）。
  距真实NZ Stock（E3）：M2048慢53.6%、M4096慢24.4%。相对V3.0内部收益明确
  （g1024 M2048 −12.7%、g1024 M4096 −23.9%、g2048 M4096 −12.7%）。
- 跨批对照与KVER28占位自拷贝限制不变，不签长稳/发布GO；Stock默认不变。

## 36. 2026-10-29｜V3.2单Y输出与融合排布推导完成，尚未编码

用户否决追加独立归因设备轮，要求按inlineReduce重构。
[完整推导](../../memfabric_mm_ar_v3/V32_LAYOUT_SCHEDULE.md)：单份池内Y承担MM、ADD/Gather和最终输出，
Gather覆盖旧源严格晚于ADD读完，TP2不需要此前提出的额外半份S。
核心改动为输出Storage生命周期和去掉最终copy、按64/128行全宽带独立发布而非逐chunk会合。
保持chunk512/1024/2048和八核MM；官方约束不支持推理AI Core异步WaitIterateAll/GetTensorC，
不以该能力为前提，Core0在子MM边界pump；MemFabric需补公共非阻塞try_wait。
600种CPU随机所有权交错通过，只证明抽象依赖/和值，不代表CANN/NPU或性能。
新Kernel/API/输出适配未实现，不改Serving默认，不追加旧V2/Stock重测或独立设备归因轮。

**2026-10-29接口约束追加**：MemFabric由其他团队维护。V3.2主方案A仅使用现有signal/wait/quiet，
采用固定R/D/G发送与阻塞接收顺序，保留单Y输出及独立producer；1024场CPU协议交错排空通过。
公共try_wait及逐片段自驱改为备选B，等待负责团队提供，不作为本轮交付依赖。
详见[V3.2排布推导](../../memfabric_mm_ar_v3/V32_LAYOUT_SCHEDULE.md)，两套协议不允许会话内混用，设备性能尚UNKNOWN。

## 37. 2026-10-29｜V3.2方案A编码，设备准入待服务器编译

用户授权实现A，并索取供MemFabric团队评估的公共非阻塞收件需求。
已实现固定FIFO OrderedProgress、八核独立producer、单Y直接输出、双槽集体选择及Storage持有；
协议31/32建池协商，epoch/slot generation与每代一次run，拒绝活跃输出覆盖/关闭。
A不新增MemFabric接口、不修改其仓库；B仍为备选。
实际C++调度/Transport 480场异步FIFO和Host mock槽管理已通过；全套CPU/静态结果见
[V3.2准入单](../../memfabric_mm_ar_v3/V32_ADMISSION.md)。
新增`quick --plan-a`和Storage/单边释放/下游消费门禁；完整计时包含acquire的流同步与allgather。
**CANN/TorchNPU编译、真实Storage/缓存/数值/性能NOT_RUN，不签设备GO**。
由用户委托OpenCode后先私有编译，再G0/设备门禁，门禁通过才短测；Stock默认不变。
[MemFabric需求原稿](../../memfabric_mm_ar_v3/MEMFABRIC_TRY_WAIT_REQUEST.md)已整理，尚未代发。

## 38. 方案A复核修复，B非阻塞收件目的澄清

A2新增输出包装后的双端prepare；任一端失败均封禁且不得启动数据波。
collective close按双方租约决定busy，避免单边关闭；同步异常封禁，旧波完成后回收输入引用。
控制信封统一校验操作/epoch/token，Host协议3202拒绝旧版本混用。
Host双Rank rendezvous及Python故障注入覆盖上述反例，设备门禁增加真正的单边持有close。
完整计时包含新增prepare，性能仍UNKNOWN；CANN/Storage/设备门禁仍NOT_RUN。
B的try_wait仅允许无邮件时继续调度，最终完整性仍依靠每片段MM/READY/ADD/Gather状态、
最终quiet+DONE及输出可见性；EMPTY绝不代表完成。见[V32准入单](../../memfabric_mm_ar_v3/V32_ADMISSION.md)。

## 39. 2026-10-29｜A2服务器准入+六门禁+短测完成：正确性全过，性能仍不过

OpenCode在38144双Die独占窗口按[执行单](../../memfabric_mm_ar_v3/OPENCODE_EXECUTION.md)一次性完成A2私有编译与验证，
[实验记录](../../memfabric_mm_ar_v3/experiments/quick-20261029-v32a2-plan-a/README.md)：
起点`213cfcedb`，修复两个编译BUG后HEAD `0f638d511`（CANN ACL头路径缺失；锁定版
libtorch_npu不导出`StorageDescHelper::SetDesc`，改经torch 2.10 `c10::make_storage_impl`
注册的PrivateUse1工厂创建NPUStorageImpl并按SetDesc语义填`npu_desc_`，未退回普通from_blob）。
MemFabric锁定db38d75e未改动，helper指纹与生产验证版一致。
G0/双Die独占成立，六门禁双Rank全PASS（ib/add/输出生命周期含单边持有busy close/协议含单chunk
排空与环回绕/融合含下游ND消费/尾块）。六配置两轮短测漂移1.17%~2.69%，
最佳M2048/chunk1024=2018.7µs、M4096/chunk1024=2727.1µs（双Rank max、完整接口窗口）。
**相对历史V3.1最佳慢21.9%/8.4%，相对V2 E3-G慢39.1%/28.7%，相对真实NZ Stock慢87.2%/34.8%，
方向性判定A2未达性能目标**；逐波acquire同步+双Host协商+Storage包装的接口成本为主因结构
（未做独立归因轮，不签同批正式收益）。KVER28自拷贝差异仍在，不签长稳/发布GO；Serving默认不变。

## 40. 2026-10-29｜A2的L0+L1插桩归因：Host双allgather~430µs与传输固定延迟是主成本，MM与IB无罪

用户授权后在同一独占流程下完成归因轮（工具提交`b0bdb2cba`，修复部署窗口顺序`f8b5b30a9`，
首轮STOP(97)为守护器对无窗口超额进程的正确拦截）。四臂同会话差分+逐波Host相位：
[实验记录§L0+L1](../../memfabric_mm_ar_v3/experiments/quick-20261029-v32a2-plan-a/README.md)。
要点：①每波acquire(~250µs)+prepare(~180µs)两次控制allgather是Host接口主成本，
基线合计~550µs/波，loadavg≈13宿主抖动可抬至~890µs；②Δ(fused−partial)显示8核MM
与MTE整块拷入同量级（g2048下近零边际），MM不是瓶颈；③裸ADD传输固定成本~500µs
（控制往返+quiet）+边际~103µs/MiB，定量证实减包减同步方向；④IB~4.8µs/次可忽略。
M4096 g1024分解：Host 21% + 设备79%。优化去向：砍每波双allgather（最大单项）、
减控制往返；MM/IB不动。归因结论不构成性能GO；L2内核内分段未做，需另行授权。

## 41. A3设计、审核与编码：完整S固定席位 + 独立Y + NZ MM / Direct ADD

用户已定稿并授权设计/审核/编码/实机准入检查。基于最新1aa95969归因证据推进，
不重新启动独立归因轮，不据fused-partial差分声称单独MM已测最优。
[A3设计](../../memfabric_mm_ar_v3/A3_DESIGN.md)：会话级完整S不在单波复用，Y独立租约；
compute512/1024与comm512/1024/2048解耦，八核连续ND生产，显式seed后双向Direct ADD，无Gather。
Host3301固定槽、冷warm、设备START一致性；正常reserve/run无Host集合或流同步，输入事件回收。
忙槽/包装失败为终止性START拒绝，禁止覆盖/重试；旧A2可恢复接口仅独立保留，不混用。
两轮设计源码审查、实际START/调度器FIFO与Host热路径回归详见[A3准入](../../memfabric_mm_ar_v3/A3_ADMISSION.md)。
新CANN/TorchNPU/设备门禁及性能NOT_RUN；CPU成功不签设备GO，Stock默认不变。

## 42. 2026-10-29｜接口优化Phase1实机A/B：合并acquire+prepare单allgather，事件时间-6%~-12%，360波零回退

依据§40去向①实施（提交`6fb54e09d`，协议3202→3203）：`wave_acquire`无协商投机占槽
（最低空闲确定性规则，双方命中即一致），`prepare`成为每波唯一控制allgather并对账
槽位，分歧返回可恢复-29回退经典交集路径；generation降级为本地租约令牌不再跨Rank比较
（投机不对称烧代次会活锁）。先打tag备份（`backup/v32a2-pre-ifaceopt-20261029`→`1aa95969c`，
另存git bundle于`/home/w00804146/310P/backups/`，SHA256 `e58e4df6098a0b4730532d9cc1b57bec8c24d2f63d1bcf4f9f01383e83e632b2`）。
r6私有构建后同会话五臂A/B（新增fused_merged臂，`merged_wave`默认False作为opt-in）：
**双Rank全PASS（含逐配置FP16位等值校验），fused_merged 180波×2Rank零回退**。
设备事件时间：M2048 -10.2%~-12.3%（最佳g1024 1993→1790µs），M4096 -6.0%~-7.5%
（g1024 2782→2573µs）。Host相位：acquire 253-277µs→52µs（省~200µs/波），prepare不变
（仍单allgather），enqueue合计-177~-249µs/波。首次STOP为manifest漏登tensor适配器
（KeyError），补跑build_tensor.py登记后PASS。回退路径已由Host打桩UT+Python注入UT
（`test_v32_wave.py`5项+`test_v32_host.py`1项）覆盖；CPU套件57项全过。不构成性能GO；
merged默认翻转见后续提交；L2内核分段仍未做，需另行授权。

## 43. A3 profile3302：首READY合并帧校验，取消非必要启动固定开销

用户要求减少非必要固定开销。独立START已删除，正常计算不等待跨Rank启动确认；
第0范围的64B READY合并profile/epoch/slot/形状/粒度/mode/admit，双方校验后才可ADD。
取消计算前的本地fan-out，状态清零移入Kernel，不再每波提交ACL memset任务。
失败端不做MM/seed，发送ready=0并在错误路径quiet后Trap；正常对端可以已计算首单元，
但双方均不允许数据ADD；该故障仍为终止性，不在原会话重试。
profile3302拒绝与3301混用，源码/设计/准入单及实际协议故障回归同步更新。
保留现有数据可见性、READY、精确完成、最终quiet/DONE及租约/事件；没有借减开销删除数据依赖。
新CANN/设备门禁及性能NOT_RUN；不改A2 merged-wave实现或历史实验证据。


## 42. A3-2专职通信与A3-tail末尾席位候选（设备NOT_RUN）

以A3首READY版970a0c414e9786b9106f7032e6c5da6ffd0ea978为基底，原3302保持对照；
新增3303七核MM+Core0专职通信、3304 Core0计算末尾席位。S/Y ND连续行、NZ只描述权重。
详见[A3席位设计/同批执行](../../memfabric_mm_ar_v3/A3_SEAT_SCHEDULE.md)。
同批包含两种MM粒度下各A3前/七核/末尾/A3后，共八会话；沿用六门禁及失败STOP。
不逐chunk等源回收，不增加正常Host同步/集合/清零任务；每座位发布缓存行状态有真实开销。
MemFabric仍锁db38d75e，主线核对54d5518d；工作仅落自有V3短分支，不自动合主线或启用Serving。
本地CPU/源码验证后交服务器私有编译和门禁，CANN/NPU/新性能NOT_RUN。无新模型/TTFT/Graph测试。

## 44. 2026-10-29｜A3同批三候选服务器私有编译完成，设备阶段因38144被外部租户占用STOP

固定起点81780f342（tree 6b94a0b3）+MemFabric db38d75e，新隔离worktree/克隆，无 tracked 改动。
CPU全过（V3+V2 UT 135、治理38、静态门、V3范围Ruff、git diff --check）；私有SDK（137步，
helper与生产验证版逐字节一致）与V3 Host/Device（BiSheng dav-2002，device_compile PASS）、
TorchNPU Storage适配（冻结进manifest）全部完成；三候选3302/3303/3304的模板实例化、
Matmul注册/UB NZ2ND预算/双MTE缓冲/IB第二半64B座位标志/Core0独立分支等按席位合同逐项核对通过。
设备G0预检时38144 dev4被外部租户python进程持续占用（19.8GB、51% DDR带宽，观察18分钟），
dev5间歇探测——按本轮用户指令（卡忙即STOP、不换卡、不杀租户）与执行单§5停止，
compare_a3八会话批次NOT_RUN、未触发任何设备Kernel。
[实验记录](../../memfabric_mm_ar_v3/experiments/a3-compare-20261029-r1-device-stop/README.md)。
恢复条件：协调38144双Die覆盖整批（约1小时）真实独占窗口后直接用已冻结manifest启动批次；
KVER28占位自拷贝差异仍在，不签长稳。无修复提交，分支HEAD仍为81780f342。

## 45. 2026-10-29｜A3同批六会话PASS后遭外部租户干扰STOP：c512组完整对照成立，原A3占优

用户授权换卡16384/dev2,3。批次HEAD 4d5ce62e8（docs提交后按工具合同重建；Device SO与
起点逐字节一致）。11:42–12:06独占窗口内六会话全部PASS_QUICK_SCREEN_ONLY（六门禁双Rank、
24计时文件、全样本correct）：a3-before/a3-2/a3-tail/a3-after-c512 + a3-before/a3-2-c1024。
首次在dev2/3部署v28（部署窗口3.6秒自愈）。**同批c512组（完整夹逼，六配置全稳定，
漂移0.12%–3.06%）**：原A3六配置全快于A3-2（2.5%–7.3%）、5/6快于A3-tail（13.5%–25.3%），
唯一例外M2048/comm2048 A3-tail +1.70%但夹逼漂移2.19%，不判稳定收益。
c1024组2/4会话方向不一致，不签。12:06:42外部跨卡进程（同占5664+16384）触发守护器STOP，
第7会话死于池创建（无门禁/ADD执行）；重试批次preflight遇vLLM EngineCore占用再STOP；
终态全服务器无独占窗口（16384 dev2被vLLM serving 20+分钟、38144原租户未走、43520 dev7 Alarm）。
未杀租户、失败现场全保留、未重试ADD；不签整批GO。
[实验记录](../../memfabric_mm_ar_v3/experiments/a3-compare-20261029-r2-card16384/README.md)。
恢复：manifest已冻结，约20分钟独占窗口即可补完整批次（compare_a3.py start --card 16384 --devices 2,3）。


## 统一完整I/O计时与A3阶段归因（代码完成，设备NOT_RUN）

本轮在591a61fa实机归档之上补齐Stock/V2 D4/V3/三A3统一Tensor输入到ND输出ready边界。
新增独立进程对照入口、Stock NZ/HCCL路由门、V2 D4/KVER11冻结、11会话前后对照；
三A3有无打点前/Host/Device/无打点后，设备每核与座位独占cacheline，默认关闭。
详见[统一计时合同与执行单](../../memfabric_mm_ar_v3/UNIFIED_TIMING.md)。
每核占比与通信暴露等待分别统计，禁止重叠时长相加、Wall−Event当Host或fused−partial当MM。
本地无CANN/TorchNPU/NPU，新设备编译、资源、数值、路由与归因NOT_RUN，Stock默认不变。


## 2026-10-30｜V1/V2/V3主线资产整合（用户授权，研发暂停）

以主线54d5518d与最新V3 d59fec7d为固定起点，整合54条项目相关分支。
V1/V2正式实现及Serving/build默认保留；新增E1/E2/E3/E4、G1取证、MM shadow等独立候选，
NZ-aware U1工具、测试、原始实验与数据；冲突及废止实现按原blob完整保全，不覆盖当前模块。
[整合说明](../../memfabric_history/README.md)、[分支HEAD/快照清单](../../memfabric_history/manifest.json)。
历史源码特定的五个测试文件与废止R2计划归入legacy，匹配源分支可完整恢复；未删除原件。
407 CPU测试及36 subtest通过；无CANN/设备新实验、无生产启用、无外部仓库或SDK变更。
所有源分支保留；本轮不删除分支、不强推、不以归档恢复算法推进。


## 2026-10-30｜V3新Profiler证据增量归档

按用户要求，将work/310p-mm-ar-v3-sdma-add-20261029的新提交
b34901ba6d031e46386cbe5f1e7b07e813ac87ba合入主线（上次已归档d59fec7d）。
新增A3/A3-2/A3-tail硬件Profiler的6份CSV、采集脚本、索引和摘要；原始文件不重编码。
[原报告](../../memfabric_mm_ar_v3/experiments/a3-unified-shared-card-20261029-r1/README.md)、
[增量来源和校验](../../memfabric_history/INCREMENTAL_ARCHIVES.json)。
主线原始整合历史与旧快照保留。仅归档，不改变算子/协议/Serving；
用户已暂停的统一L1框架和版本文档工作继续暂停，未混入其未提交草稿。

## 2026-10-30｜统一L1独立Stock、设计版本库与统一采集框架

按用户“继续，不需要查格式”恢复此前三项工具与文档任务；以主线8e8c4def为基底，
短分支work/310p-l1-framework-20261030交付，不推进MM/协议新优化、不改Stock默认。
[统一框架合同和操作](../../../../csrc/_310P/tools/L1_test/README.md)：独立Stock无需选择融合版本，
ready/chain包含完整Tensor调用及下游真实残差ADD；W/N/R一致，Stock/候选ABBA独立TP2进程，
先按样本取双Rank max再算术均值，保留全部N。独占与接受漂移的共享模式明确分开。
公共Host/Profiler采集有无打点前后控制，按形状/场景隔离CSV；版本适配仅负责本版ABI/状态/生命周期。
[V1](../../memfabric_mm_ar/docs/VERSION_INDEX.md)、[V2](VERSION_INDEX.md)、
[V3](../../memfabric_mm_ar_v3/docs/VERSION_INDEX.md)共35份VN-M文档：使用、输入输出、整体图、逐步说明及证据缺口。
C2/D2/D3/D4统一到相同AddTile2048设计；Graph/诊断不混入完整Eager收益。
[性能证据](L1_PERFORMANCE_EVIDENCE.md)整理36行历史Host阶段范围，保留p50口径，
明确AICore时钟不可用、管线并行不作互斥占比、SDMA占用未知、6份CSV与18次采集索引的差别。
新框架仅CPU验证，私有编译/双Rank设备Smoke/路由与正式性能NOT_RUN。
历史external适配须恢复匹配源码/工件及冻结工厂，不能用同名新实现替换；注册不表示版本已可上机。
本轮按用户要求不执行格式检查；验证及两轮复核见[L1框架复核](L1_FRAMEWORK_REVIEW.md)。

## 2026-10-30｜用户要求PR #23直接合入，后续实机统一使用主线

用户明确要求“直接合入主线，后续都直接在主线进行实机实验”。
统一L1框架与35份VN-M文档（来源1415d479、PR #23）已按此指令纳入主线，
CPU 428项及36个子测试通过的事实保留；本次只追加执行规则，无新增设备测量。
后续开发、实机BUG修复、脚本、原始数据及分析直接回`dev/310p-memfabric-mm-ar`，不再默认使用任务分支或等待PR合入。
服务器SSH获取主线，每轮固定完整SHA/tree与工件/SDK/KVER/物理映射；隔离构建及结果目录，修复后重冻结身份。
[分支治理I节](BRANCH_GOVERNANCE.md)、[OpenCode执行单](../../memfabric_mm_ar_v3/OPENCODE_EXECUTION.md)、
[统一L1入口](../../../../csrc/_310P/tools/L1_test/README.md)已同步。
现有G0、CPU、私有编译、双Rank短Smoke、正式对照、故障STOP及恢复SO SHA要求继续有效。
本轮未运行NPU，主线合入不等于新框架已通过实机准入；Stock默认不变。按用户要求继续跳过格式检查。

## 2026-10-30｜统一L1代码迁入310P工程目录

按用户代码归属要求，将统一L1框架、CLI、版本登记、冻结模板、文档/证据生成器与CPU测试整体迁至
`csrc/_310P/tools/L1_test`。新入口为`python csrc/_310P/tools/L1_test/l1_bench.py`。
同步修正仓库根目录解析、包导入、后台CLI路径、torchrun的`L1_test.worker`模块、子进程PYTHONPATH及框架SHA清单。
35份版本说明、索引、项目入口、操作示例与CI测试入口已更新；历史快照和原始实验文件保留原文。
新增跨工作目录启动/worker模块发现与freeze路径回归；迁移后运行manifest按新主线SHA重新冻结。
本轮仅目录及引用迁移，完整计时、统计、采样和协议语义不变，无新增实机数据；按用户要求不查格式。
迁移后全量CPU回归430项及36个子测试通过，其中L1框架23项；项目静态与变更台账门禁通过。

## 2026-10-30｜V1/V2全版本规整与封版证据审核

用户要求从全部V1/V2选最优正式版，优先复用历史结果，缺证据由OpenCode查服务器而非重跑。
[审核包](../../releases/v1-v2-20261030/README.md)锁定37个源码/构建输入；V1六核心文件与fa1c4217逐blob一致，V2 D4/E1/E3/E4的SHA与已有E4构建manifest全部匹配。
five-arm和E2机器评估已从历史快照对象读出，E3/E4实体原件已复核，不能再把它们误报为缺失。
[V1选型](../../memfabric_mm_ar/docs/RELEASE_SELECTION.md)：ABI7多能力不是互斥版本；默认q/单波/小M组合需找匹配原始对照，SEL-01 0/3自然场完成的失败事实保留。
[V2选型](RELEASE_SELECTION.md)：当前D4仍是构建参考，E1偏向M2048，E3-G在M4096有稳定改善，E4-T8仅单场Screen最快；不能一刀切成全形状winner。
登记表新增implementation_group/release_role用于归类；最终winner明确为空，源算子及构建/ABI/Serving零改动，不创建虚假final标签。
待用户委托OpenCode按[只读查找单](../../releases/v1-v2-20261030/OPENCODE_HISTORY_LOOKUP.md)回传V1身份/配置/后续稳定性与E4后续复核原件，再完成最终版本选定和封版。
本轮无CANN/NPU新执行，不因为新L1框架尚未实机而要求重测V1/V2；不查格式。


## 2026-10-30｜大M性能优先封版选择：V1 q4，V2 E1/T8

用户确认M=2048～4096，并明确“谁性能好，选谁”。按ADR-038复用成功记录，不重跑V1/V2。
V1选历史成功q4/batch_m1024，仅余准确成功源码/二进制/配置映射；无需全M/Decode/模型长稳或SEL-01后续修复。
失败实验不参与排名，原件及适用模式限制保留。V2 M2048选E1、M4096选E4-T8；不因D4正式身份或E3-G复核较多而选更慢版本。
已抽出14份E4 attempt2原始Rank JSON：每序号先Rank max再算完整Wall均值，
E1/M2048=1394.627µs，T8/M4096=2175.690µs；历史Event p50分别1279.596/2076.211µs。
T8/T4 M4096 Wall均值仅差0.128µs，近乎持平；保留Screen标签，不宣称稳定显著领先。
[复算/工件身份/源码封存](../../releases/v1-v2-20261030/README.md)与[只查V1的OpenCode任务](../../releases/v1-v2-20261030/OPENCODE_HISTORY_LOOKUP.md)已更新。
V2实测点选择及源码封存完成，2049～4095最优未知，不拼未经验证的新运行时，不改Serving默认。
两轮静态复核、14份原件Git blob/源码SHA校验、31项CPU回归及项目元数据门禁通过；仅离线复算，无NPU新执行，不查格式。


## 2026-10-30｜V2所选公共接口与构建代码梳理完成；V1延期

按用户指令先完成V2，V1历史身份查找留到下一次实机。所选仍为M2048/E1、M4096/E4-T8，
新增selected_api.h/selected_adapter.cpp固定M公共C API；每句柄只绑定一种配置/Stream，不混合路由。
新selected.cmake独立构建E1/E4 Device并将公共Host编入扩展，摆脱test/perf私有Host加载；
旧D4/Serving、诊断与历史候选保留。设备E1/E4/common、基础Host及内部结构原SHA不变。
句柄持有预热/输出地址/poison状态，避免旧私有裸地址表跨销毁复用；保留首写保护、status清零及协议排空。
统一L1显式api=v2_selected_v1接入，固定M、实际Device装载身份、错配置和历史私有ABI混入均检查。
[代码图谱](CODE_MAP.md)、[接口使用](SELECTED_API.md)、[两轮复核](CODE_REVIEW_20261030.md)、
[源码包身份](../../releases/v1-v2-20261030/V2_CODE_PACKAGE.json)为当前代码入口。
全量CPU452项与36子测试PASS；含Host C++故障注入和CMake四开关组合。新入口CANN/NPU NOT_RUN，
不将旧SO成绩写为新构建实测；没有安排V1/V2重跑，无Serving默认切换，不查格式。

装载复核补充：统一L1识别libmf_及libmf310p_两类库，新增未冻结基础Device和Stock误载V2 Device负面回归，全部452项与36子测试通过。


## 2026-10-30｜四家族L1简易适配与Stock/V2/V3固定M一轮计划

用户要求给Stock/V1/V2/V3各实验接统一L1并准备一轮M2048/4096，当前重点V3迭代。
适配器拆为L1_test/adapter_plugins四家族，仍共用worker计时/正确性/统计/采集；V1适配可用但本轮不查历史、不测。
V2修复诊断OFF仍强制绑定AR-only的问题并提前分配输出；所选公共接口算法与ABI未改。
V3八种迭代显式配置，旧0/2/4按冻结Python+SO加载，包命名空间隔离不偷用当前代码。
默认完整43组/170独立TP2会话，Stock+V2所选+全部V3参数，W20/N100/R1、ready+chain；
每配置/每M/每场景/每阶段Stock与候选各200配对样本，逐样本Rank max再算均值，不跨配置合并Stock。
active显式缩减为30组/118场，另外选择V2历史则64组/254场，打点三段完整510场。
所有工件/历史身份/门禁前置预检，不齐整批BLOCKED；首错STOP不继续后组，无自动ADD重试。
[接入与矩阵](../../tools/L1_test/ROUND_FIXED_M.md)、[OpenCode执行单](../../tools/L1_test/OPENCODE_ROUND.md)、
[两轮复核](../../tools/L1_test/REVIEW_FIXED_M.md)。475项CPU及36子测试PASS，未做新CANN/NPU测验。
本轮依用户新要求允许新测Stock/V2同批对照，覆盖此前仅复用历史的范围限制；Stock生产默认不变。

## 2026-10-30｜快速shared与正式exclusive分开，复用服务器两个原仓

用户明确快速验证用非独占。本次先shared、W2/N10/R1、collection off，完整43组/170会话，
每臂每M每场景20个配对样本；允许外部PID及漂移，保留健康/物理映射/SDK/数值/协议门禁。
结果OBSERVATIONAL_SHARED，不因目标卡忙而拒绝，不自动追加正式独占轮或打点轮。
后续正式性能另生成exclusive、W20/N100/R1计划，分别freeze/归档，不混合两类结果。
框架已支持两mode，本次只明确执行策略和文档，不改算子、适配器、计时/统计或CLI全局默认。

目录沿用现有vllm-ascend与memfabric_hybrid，干净且无运行依赖时原地切已有分支；
vllm主线仍dev/310p-memfabric-mm-ar，V3 MemFabric锁fix/310p-v3-orch-deploy-helper-20261029@db38d75e。
不新增clone/worktree；构建/必要快照集中原vllm仓build/310p_l1/<run_id>，
计划/证据/结果集中csrc/_310P/tools/L1_test/experiments/<run_id>，不自动删除旧目录。
优先复用匹配SDK；旧fresh源码构建要求不能解释成允许清除原MemFabric的build/output或另起顶层克隆。
[执行单与简短提示词](../../tools/L1_test/OPENCODE_ROUND.md)、[模式与矩阵](../../tools/L1_test/ROUND_FIXED_M.md)、ADR-041同步。
本次验证两mode计划生成和公共守护接线、现有源码包身份及项目元数据；无NPU新执行，无新性能数值，不查格式。

## 2026-10-30｜MemFabric两算子共用源码分支，澄清历史SDK与公共部署修复

用户指出V2/V3可用同一MemFabric分支。现行执行单统一源码依赖为
fix/310p-v3-orch-deploy-helper-20261029@db38d75e8485d31451302219eb7a1bfeb1dd107c，
不为算子版本另切MemFabric分支；已有SDK_V11/SDK_V28模板路径仅为历史工件示例。
逐项核该提交相对b9d46f97的6个文件：公共Host编排器改为必须fork+exec部署助手，
探测/CUST迁移/复制/校验在子进程完成，宿主零KFC；缺助手/失败明确拒绝。
针对torch_npu进程内KFC非零退出导致上下文持续507018，以及V3首次建池部署失败；
ADD指令、KVER28设备核/合同、SetAccess物理号修复未被该提交改动。
本修复作用于公共部署路径，并非每次MM或通信chunk启动助手。
统一源码分支与统一SDK/旧SO兼容性分别确认；V2用新SDK须重建匹配Host/Device并过双Rank短测，
不直接换JSON，不将历史v11成绩标作新SDK成绩。当前新组合设备验证仍NOT_RUN。
[最新执行单](../../tools/L1_test/OPENCODE_ROUND.md)、ADR-042同步；本次仅文档澄清，不改算子或模板默认工件。

## 2026-10-30｜下游只验证不计时，L1性能仅完整MM+AR

用户明确“下游消费验证不需要计时”。基于df6256ff升级L1_TENSOR_IO_V3 / campaign schema V2：
唯一ready计时，Wall/Event覆盖完整输入调用至输出/流完成；两个计时Event，不再有chain、
operator_event_us或consumer_event_us。每次必需Host准备/包装/拷贝/seed/同步仍在run。
下游ADD在采样前后、每M及双地址独立运行：op返回立即同Stream消费，最后才同步；
只写PASS/失败证据，不读取框架时钟/计时Event，不在热态循环或Profiler性能范围内。
聚合要求双Rank完整before/after证据且timed=false，缺失/重复/错算或旧指标字段均失败。
逐样本完整输出与协议检查保留；均值仍每序号双Rank max后聚合，不丢样本。

完整43组/170会话及shared快速W2/N10/R1、正式exclusive W20/N100/R1继续有效，
删除的是第二个计时场景，会话数不变。新计划须重新生成并freeze，旧原件不重标/不混入。
框架仍为单次完整调用基准与独立下游验证，尚未实现连续异步推理吞吐；不据此签模型TTFT。
[执行单](../../tools/L1_test/OPENCODE_ROUND.md)、[复核](../../tools/L1_test/REVIEW_FIXED_M.md)、
35份版本文档/默认计划/所选代码包身份同步；CPU482项及36子测试PASS，无新CANN/NPU或性能数字，不查格式。
