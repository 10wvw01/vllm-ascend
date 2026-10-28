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

## 26. 2026-10-28｜A1五层MM+AR与E1性能优先候选（工作分支，**未有NPU结果**）

- **用户确认目标**：五层完整链对照A=原生NZ-MM+HCCL、B1=原生NZ-MM+MemFabric AR、B2=自研完整MM结束后再启动逻辑一次MF AR、C=原D4多Chunk交叠、E1=新缓存/同步关键路径优化融合Kernel。B2不是D4`overlap=0`；必须证明**所有M的MM完成后才发出第一条SDMA消息**。
- **研发实施位置**：从**本唯一主线**派生短期`work/310p-mm-ar-five-arm-dcci-20261028`；对应[五层实施/实机合同](A1_五层MM_MemFabric_AR与E1关键路径优化_20261028.md)及[ADR-017](DECISIONS.md)。工作分支只放置独立`mf310p_v2_b2_full_mm_ar.asc`、`mf310p_v2_dcci_entire_candidate.asc`、`five_arm_private_host.cpp`、私有构建、十场/route/smoke/分析器和UT。不存在独立长期状态主线，候选B2/E1不混入正式默认CMake。
- **E1核心设计**：D4中生产/消费两侧按64B每次循环DCCI的3处静态调用，改为各核每次一次`ENTIRE_DATA_CACHE`；再把每Chunk Core0逐次等待另外7核改成树形Ready：Core0只等1/2/4/6四个代表核，Core2/4/6先等待各自邻核。减少DCCI扫描和Core0串行会合，共同争取提早SDMA signal/减少通信Wait暴露。IB事件完成依赖、signal/wait、DONE/quiet仍保留；新拓扑须实机证明正确并测性能。CANN9.1官方文档警告整缓存刷新可能昂贵，性能是否正向**UNKNOWN**。这是第一版实质性优化而不是仅旧逻辑分析；后续不能因为E1负收益停止大M研发。
- **历史正确限定**：A1-MM-SHADOW valid run2 M4096原生NZ-MM Event约848.2µs、影子ND MM约867.6µs，发送端CleanLines受控增量约139.6µs；A1 U1新的StockNZ完整链对照仍受M1024 anchor场间漂移6.99%阻断，L1仍不可签。R2`loop4`与`formal`不是单变量交叠对照，不能把约0.6ms差值当严格交叠收益；早期T2原生MM+MF中转和首写修复成本重，旧Stock还非当前NZ直接路由，所以只能历史参考。
- **新五层测试事实**：当前仅完成工作分支源码/脚本/CPU合成合同提交，**尚未有新的本地CPU执行证据、CANN BiSheng编译、310P3真机G0、双Rank候选Smoke、十场原始样本、模型L2/L3/L4**。精确状态：`G-KERNEL_IMPLEMENTED_PENDING_REVIEW_AND_G0`；不得把实验脚本的`PASS`字面量或文档当硬件PASS。
- **门禁与交接**：完成两轮静态深审和CPU合同，构建私有B2/E1独立Device+Host SO（无生产SO覆盖），现场核主线HEAD/工作HEAD/两Host/D4/KVER SHA/真实逻辑物理Die/连续独占、首写及真实NZ route；B2/E1短Smoke正确后再走10场平衡五臂ABBA并存原始证据。507014/507899/507018/错算/首写/外部占用则保留现场STOP。工作分支实机证据审核后，择优回写本控制台与[性能基线](PERFORMANCE_BASELINES.md)、审PR并挑选合入正式E候选；Stock默认不改。

## 27. 2026-10-28｜A1五层+ E1单批实机完成：E1显著快于D4但仍慢于Stock，M2048 D4漂移致总状态INCONCLUSIVE_DRIFT

- **执行与安全**：run_id `a1-five-arm-20261028-001`，工作分支`work/310p-mm-ar-five-arm-dcci-20261028` @ `b6ac7330b`隔离worktree。卡16384（dev2,3）同卡双Die+MCU健康、全程30s只读巡检独占、无Alarm/507014/507899/507018、测后显存回基线。私有装载沿用R1/R2验证方案（Host`1dc46d7d`/D4`2a19390c`/插件`8eb65de9`/KVER11`25e16b0a`，私有PYTHONPATH+LD_LIBRARY_PATH，共享系统未改，前后SHA一致）。CPU62项PASS、两轮深审PASS、B2/E1 dav-2002隔离编译（候选`72b868c7…`，无正式ABI导出）；KFC在授权Smoke首次建池自动部署。报告与原始证据：[实机报告](A1_FIVE_ARM_ONDEVICE_REPORT_20261028.md)、[evidence/a1_five_arm_20261028/](../evidence/a1_five_arm_20261028/)。
- **路由与正确性**：真实NZ路由证明4/4（每Rank×M2048/4096原生CANN CSV：`aclnnMatmulWeightNz`≥6、`hcom_allReduce`≥6、TransData=0）；B2/E1双Rank Smoke与十场共24份Rank JSON全部bit-exact、协议status=0。
- **十场五臂配对p50（max(rank0,rank1)）**：M2048：Stock 1070.4µs、B1 2653.8µs、B2 1647.9µs、D4 1362.0µs、E1 1263.6µs；M4096：Stock 2017.5µs、B1 5031.2µs、B2 3092.5µs、D4 2513.0µs、E1 2274.2µs。**E1 vs D4：+7.79%/+10.50%（−98.4/−238.8µs）**且M2048尾部更紧（E1 p99≈1327µs vs D4该场2372µs）；**E1 vs Stock：−15.29%/−11.29%**（仍慢，L1对Stock不签）；B1慢约60%（staging/流同步/多launch真实代价，非MF固有带宽）；B2比D4慢285.8/579.5µs（串行结构输给Chunk交叠）。
- **机器判定**：总状态**`INCONCLUSIVE_DRIFT`**——M2048的D4臂场间漂移5.685%>5%门限（stage07出现p99=2372µs周期长尾，与R1已知V2周期长尾一致），按合同不挑场不复算；M4096全臂稳定（≤2.02%）但随总状态降级为参考。
- **决策（按A1§5）**：`E1<C且E1>=A`→**E1保留为下一优化基线**（仅工作分支候选，不进正式默认CMake/Serving），下一E代聚焦同步消息缩减/冗余清理消除/D4式周期长尾；维持Chunk交叠主线（B2负向）；B1归因staging成本。正式D4/Host SO零改动，Stock默认不变，L3/L4 NOT_RUN。工作分支证据审核后择优回写主线并审PR。

## 30. 2026-10-28｜E3 实施：Chunk2048计算组 / 双4MiB通信分包｜阶段性源码提交

- **为何E3新增2048**：2026-10-28 E2正式报告证明R/P/512负收益、E1仍最佳（M4096 E1约2285.5µs、Stock约2022.6µs），用户要求继续MM生产/producer缓存关键路径并补充真实Chunk2048实验。**8MiB逻辑组不能作为8MiB单条MemFabric包投递**；E3分离ComputeGroup2048与TransportPacket1024行（2×4MiB）。
- **两种实际 Kernel**：G模式每个2048组每核两次128行MM(`tiling1024`)；T模式每组每核一次256行MM(`e3Tiling2048`)。同一ReadyTree、整核producer/receiver缓存维护、两封独立tag的SDMA mail、Vector Add2048、quiet与DONE；与E1同批对比MM分组/tiling效果，不把多因素差额误归因于某个等待。
- **分支关系**：自E1实机分支`work/310p-mm-ar-five-arm-dcci-20261028`派生临时`work/310p-mm-ar-e3-chunk2048-20261028`，不从已证负收益且存在未闭环507014的E2-P分支继承设备路径。主线`dev/310p-memfabric-mm-ar`仍唯一项目权威；正式D4/E1默认Serving零改动。
- **工程实施与合同**：`src/op_kernel/mf310p_v2_e3_group2048.asc`、`test/perf/e3_private_host.cpp`、`e3_build.sh`、`e3_arm.py`、`e3_assess.py`、`e3_campaign.sh`、`test/ut/test_e3_chunk2048_contract.py`，详见[正式实验设计](A1_E3_CHUNK2048_设计与实机验证_20261028.md)。Host仍以已审1024创建池，避免污染生产ABI；只将逻辑Chunk2048传给独立设备候选，数据发送最大4MiB。
- **实机执行**：项目G0与四卡只读选卡→原仓CPU和新合同→隔离BiSheng编译（新tiling可能编译不支持）→NZ路由4/4→G/T逐项双Rank bit-exact Smoke→`stock,e1,e3g,e3t,e3t,e3g,e1,stock`八场TP2完整链、5%漂移判定；保存双RankJSON、设备报警、4MiB双包、SO身份、源码HEAD。507014等异常STOP保全，不能自动清他人资源。
- **当前状态**：E3源码与测试合同已提交，但CPU测试运行、CANN编译、G0和设备数据均`NOT_RUN`。绝不继承E2有效性能作E3硬件成绩。E3若确证正收益，才考虑Qwen真实40站点/L3/正式input4096/output2048/TP2/batch10。

## 31. 2026-10-28｜E3 Chunk2048单批实机完成：M4096 E3-G真实超越E1，单组场景退化，T被L0C约束击败

- **执行与安全**：run_id `e3-chunk2048-20261028-001`，分支`work/310p-mm-ar-e3-chunk2048-20261028`基准`65837dc4`；隔离worktree`/home/w00804146/310P/e3wt`。四卡只读G0后16384（dev2,3）同卡双Die+MCU健康独占，全程30s巡检无507014/507899/507018/IB/KFC异常、无外部抢占，测后卡回基线；私有装载沿用R1/R2/five-arm方案（Host`1dc46d7d`/D4`2a19390c`/插件`8eb65de9`/KVER11`25e16b0a`，前后SHA一致）。原仓UT 71项、py_compile/bash_n、项目门禁全PASS。
- **BUG修复1（已提交`3e58ae38`）**：E3-T首版`e3Shape2048`baseM=256×baseN=256的FP32累加器=256KB恰好100%占满dav-2002 L0C（编译期未拦截），首次Smoke首个真实kernel即双Rank对称AICore异常（retCode=0x26, failure_mode=1）；E3-G同批全过证明异常隔离在T tiling。修复为basicM=128（singleCoreM=256不变，T仍是每组每核一次256行MM调用），UT锁定断言+复跑+重建+双Smoke转PASS。[修复记录](../evidence/e3_chunk2048_20261028/bugfix1_l0c_overflow/README.md)。
- **正确性**：真实NZ路由4/4（`aclnnMatmulWeightNz`≥6、`hcom_allReduce`≥6、TransData=0）；E3-G/E3-T双Rank Smoke与8场16份Rank JSON全部bit-exact、协议status=0；8MiB逻辑组2×4MiB分包、双包唯一tag/len/dst校验、两包齐后Add均由内核断言+UT+运行期status证明。
- **八场性能（配对p50）**：M4096：Stock 2022.6、E1 2271.8、**E3-G 2119.5（vs E1 +7.19%/−152.3µs，vs Stock −4.57%/−97.0µs）**、E3-T 2267.1（vs E1 +0.21%，vs E3-G −6.51%）；M2048：Stock 1078.4、E1 1222.6、E3-G 1451.0（vs E1 −15.74%）、E3-T 1483.9。M4096全臂漂移≤2.02%稳定；M2048 E1臂6.19%>5%使总状态`INCONCLUSIVE_DRIFT`（E3-G劣于两场E1各自p50，方向稳健）。[实机报告](A1_E3_CHUNK2048_实机报告_20261028.md)、[性能台账](PERFORMANCE_BASELINES.md)。
- **结论**：①Chunk2048只在组数≥2（M4096）赢：固定同步开销减半>交叠损失；单组（M2048）零交叠净亏228µs。②真2048 MM在310P3不可实现更大基础tile（L0C硬限制），E3-T的−6.51%是单次大调用相对两次调度的真实代价，非通信差异。③E1→E3-G把对Stock差距从−10.84%缩到−4.57%（−97µs）。下一轮：单组组内"前1024行MM完成即发首包"混合粒度流水、producer DCCI分段、Core0串行链、周期长尾。
- **边界**：全部为L1离线算子；L2/L3/L4 NOT_RUN；正式D4/E1源码、Serving默认、共享SO零改动；E3候选SO仅隔离目录未安装；Stock默认不变。E3-G是否升为下一优化基线待主线裁决。

## 31. 2026-10-28｜E4隔离开发现场状态（本分支不是项目权威主线）

> 正式项目事实/ADR/E3审签结果以 `dev/310p-memfabric-mm-ar` 的 [PROJECT_CONTROL.md](https://github.com/10wvw01/vllm-ascend/blob/dev/310p-memfabric-mm-ar/csrc/_310P/memfabric_mm_ar_v2/docs/PROJECT_CONTROL.md) 和 [ADR-020](https://github.com/10wvw01/vllm-ascend/blob/dev/310p-memfabric-mm-ar/csrc/_310P/memfabric_mm_ar_v2/docs/DECISIONS.md) 为唯一权威。该章节仅解释E4代码/测试变更，不覆盖旧状态，也不宣称上机成绩。

- **新实现**：私有`E4Group2048Kernel<ADD_TILE,PACKET_ADD>`，MM生产仍冻结E3-G的两次1024调用与2048组、两包4MiB、完整DCCI/ReadyTree和quiet/DONE；AddTile2048/4096/8192三档+Packet到达半组Add（Core0唯一mail消费者、Core1～3/4～7独立release），其余默认不变。
- **离线自适应M**：`test/perf/e4_shape_plan.py`全覆盖M2048～4096、精确合法组组合/≤4096容量/≤4MiB每包，默认路由Stock；**只具备CPU规划，不具备真实变长尾块设备Kernel**。
- **私有构建/运行**：`e4_private_host.cpp/e4_build.sh/e4_arm.py/e4_assess.py/e4_campaign.sh`；E4只接受M2048/4096，先双RankSmoke→七臂短Screen→明确授权才十四臂Confirm；原生NZ路由与冻结SO/KVER SHA均为门禁。
- **当前真实性**：源码已落分支；CPU pytest、BiSheng编译、G0和全部E4设备性能均`NOT_RUN`。E3历史实机已完成，但不能继承E3的PASS给E4；私有候选必须经过两轮静审后独立实机。生产Stock/Serving/正式D4/E1/MemFabric无改动。详见[实现与门禁](A1_E4_分层流水与连续M_实施及门禁_20261028.md)。

### E4第一轮实机（2026-10-28，run_id=e4-layered-20261028-001）

- **已完成并PASS**：G0（16384/dev2,3独占、四冻结SHA与E3同批次逐字节一致）、原仓UT 80项+E4合同UT、2049-M穷尽规划审计、py_compile/bash -n/治理静态门、E1/E3-G/E4四实例BiSheng dav-2002隔离编译一次通过（**AddTile8192无BUILD_UNSUPPORTED**；正式D4符号未导出）、真实NZ路由4/4（MatmulWeightNz+HCCL、0 TransData，CANN CSV归档）、E3-G双Rank Smoke（M2048/4096、bit-exact、status=0、首写成功）。
- **STOP**：E4四候选Smoke与七臂Screen因宿主级外部进程PID 506263占用全部四卡（16384双Die持续增长、hugepages满）触发"设备被占用→立即STOP"合同止损，均`NOT_RUN`；十四场Confirm未授权未运行。执行侧另有e4t2 torchrun被工具超时SIGTERM误杀的工艺失误（无设备执行，已如实归档）。**E4设备正确性与全部性能结论维持NOT_RUN**；M4096约97µs Stock差距、AddTile4096/8192、PacketAdd、M2048四个问题本轮均无数据。
- 证据与报告：[第一轮实机验证报告](../evidence/e4_layered_20261028/A1_E4_第一轮实机验证报告_20261028.md)、[STOP事件取证](../evidence/e4_layered_20261028/stop_incident/STOP_INCIDENT_ANALYSIS.md)。下一轮前置：协调四卡无外部租户的连续独占窗口后重G0续跑；campaign以setsid全脱管启动并（经审查后）为巡检增加外部占用自动STOP门禁。


### E4 Attempt1后最小测试保护已编码（2026-10-28，尚未进行新实机）

- **只改测试代码**：`test/perf/e4_stage_guard.py`加入目标卡G0后每阶段2秒巡检、跨卡宿主同PID/超额Rank进程/设备健康/坏npu-smi格式的主动STOP，单独终止自身创建的`torchrun`进程组；`e4_campaign.sh`中4处NZ/Smoke/Screen/Confirm的torchruns已受统一守护且新增`E4_PHYSICAL_CARD_ID`与当前HEAD绑定构建manifest强制检查。原30秒卡巡检日志继续保留。
- **独立启动收尾**：`test/perf/e4_detached.py`以两层独立Session启动/监督测试，输出`launch.json`、`campaign.log`、`result.json`、可查询明确退出码，只允许Screen，不会绕过Confirm授权。测试代码/模版+CPU负面合同`test/ut/test_e4_test_protection.py`及CI扩展。
- **诚实边界**：监控无法单凭容器npu-smi识别所有目标卡独占PID的所属方，也不能捕获短于采样间隔的外部抢占；**外部独占租约仍为P0**。该改动仅属于源码/结构门禁，新的CPU pytest、CANN编译、G0、E4-T2/T4/T8/P Smoke及Screen全部**NOT_RUN**；已完成E4 Attempt1报告不被改写。详见[最小测试保护与复跑操作](A1_E4_最小实机测试保护与复跑操作_20261028.md)。

