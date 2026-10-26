# 310P3 MM+MemFabric AR｜项目控制台（唯一权威当前状态）

> **维护于** `10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`；2026-10-25。**新对话优先读本文件**。这是活动状态与交接合同，不是覆盖历史实验报告。所有未在当前主线更新的旧 README、PLAN、实验总结都只作为历史证据。
>
> **主线创建基底：** `perf/310p-u1-fullchain-attribution-20261025` @ `d0d57b94df06…`。基底已包含 D4 Tile2048 正式 Kernel、F1-C 实验及 U1 测试工具；U1 仅完成源码/静态审查，**没有新的 U1 实机证据**；主线建立本身没有修改正式设备 Kernel、Host ABI、通信协议或 Serving 开关。

## 0. 项目目标、边界与验收

> **2026-10-27 最新正式裁决：ADR-015。** 大 M 最终性能指标从笼统的“融合算子/端到端吞吐更快”收敛为 **真实 Qwen3.6 Prefill 阶段的客户端 TTFT 下降**；L1只以 **完整Stock FP16 MM+HCCL AR** 作同卡双Die公平基线；优化重点是40个实际融合 `o_proj/out_proj` 站点的**双Rank设备关键路径**。报告必须闭环「完整算子净时差→含布局/拷贝的Serving站点→完整Prefill步→TTFT（含排队）」，明确TTFT未改善的关键因素；输出tok/s/TPOT只作附加观测。正常业务验收input4096/output2048/50请求/并发10/TP2/Eager/ignore-eos不变；input4096/output1/并发1只作独立Prefill诊断，不可替代正式业务或与其混算。此决策不等于任何新实测GO，A1仍只准CPU+四卡只读G0，后续Smoke/Screen/Confirm与E2E均分别审批。

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
| P0-3 | **大M极致优化 → Prefill TTFT（ADR-015）** | ChatGPT方案/源码，OpenCode硬件测 | L1完整Stock MM+HCCL vs正式V2，同卡双Die；40站点真实Serving及双Rank Prefill关键路径；最终以正式Qwen3.6 Eager **TTFT p50/p90/p99**为主验收，解释四级传导；TPOT/吞吐只作辅助 | **目标正式锁定；最新D4 L1和新TTFT回归仍UNKNOWN/NOT_RUN** |
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

## 14. 2026-10-26｜主线A启动：大M完整Stock基线重建与瓶颈判定（A1工作分支）

- 工作分支`work/310p-mm-ar-large-m-a1-20261026`从干净主线`7dc8ad465ce1`派生，**只新增纯CPU归因工具/测试及工程合同文档**；不改V2正式Kernel/Host/Serving/Stock默认，也不涉及ADR-013废止的V2 Runner、Hybrid ModelState、DFlash重建。
- [A1设计、历史证据间的不可直接比较性、两阶段实机合同与瓶颈队列](A1_LARGE_M_BASELINE_AND_BOTTLENECK_20261026.md)。正式`M=2048,4096`完整算子Stock FP16 F.linear+HCCL AR与D4 V2 MM+MemFabric AR必须严格双Rank配对，M1024仅隔离诊断。现有U1`u1_run.sh`/分析器直接复用，不新建危险诊断核、不自动跑设备。
- 新增`test/perf/a1_large_m_assess.py`对真实U1原始manifest、设备SO身份、两Rank日志和样本调用原`u1_analyze.evaluate()`复算；输出完整Stock差距、Screen/Confirm路由审计与优先诊断队列。`test/ut/test_a1_large_m_assess.py`覆盖合成数学与伪GO拒绝。**没有新310P3结果，本轮设备、G0/Smoke、Confirm、Profiler均NOT_RUN；CPU CI以实际Actions为准。**
- 下一动作：审核独立Draft PR的静态/CPU合同（两轮审查），确认G0/备份/恢复/编译/设备独占后才提示用户委托OpenCode执行**大M G0和短Smoke**；Screen/Confirm/Profiler不能在G0未经准入时自动连跑。已有507014/首写/双Die挂死仍为Production P0阻塞。

- **2026-10-26 A1第一轮执行回报**：改进`a1_large_m_assess.py`严格区分路由证据与M2048/4096性能候选，拒绝其中任一形状NO_GO但整组误放行；隔离本地CPU副本20项合同测试PASS（**不是GitHub原仓CPU或U1原始取证PASS**）。新增只读`a1_g0_readonly.py`/UT和[A1 CPU/G0交接](A1_CPU_G0_ACCEPTANCE_20261026.md)。PR#9保持Draft；原仓CI无运行记录；310P3 G0/Smoke/Screen/Confirm全部NOT_RUN。下一动作只允许用户委托OpenCode执行**真实原仓CPU+只读G0**并回传门禁，不准直接投放`U1_*_APPROVED`绕过。

### 2026-10-26｜A1 只读 G0 默认仓库根目录修复（未实机）

- A1工作分支修复 `test/perf/a1_g0_readonly.py` 的ROOT计算：旧 `parents[4]` 指向仓库内 `csrc/`，现改 `parents[5]` 指向仓库根；增加根目录UT并更新A1 CPU/G0交接记录。
- 本项仅影响只读取证的Git HEAD/工作树判断；未改变正式Kernel/Serving/Stock，也未获得原仓CPU/CI执行成功及现场G0确认。所有310P3 Smoke/Screen/Confirm/Profiler仍NOT_RUN；PR#9保持Draft。

## 2026-10-27｜ADR-015落地：历史L1～L4证据复用与A1-TTFT唯一补测设计

- [历史证据复用审计及最小补测方案](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md) 已在活动A1分支入库，**没有新设备测量**。2026-10-21旧版Prefill已经有历史L1(Serving MM+AR设备Trace近似)、L2(双Rank40站点)、L3(两Rank各自Prefill步周期)、L4(4096/1单请求时延+另场4096/2048/50/10 TTFT)的方向性链，**不再重复旧CSV归因**。
- 旧版实测：Rank0目标站点 Stock约77.9ms/步，V2约105.9ms/步（不含另约3.6ms/步输入Copy）；Rank0完整Prefill步周期2364.4→2395.7ms（旧V2退化约31.3ms）；单请求稳态延迟约+34ms。并发10历史TTFT mean11341.9→11578.6ms，p50 11811.2→12091.7ms。**不同负载数据绝不拆成伪精确排队税；这些仅旧V2，并非当前D4**。
- **历史版本指纹未决**：2026-10-21归因报告Host SO SHA前缀`6e9dc2d6`与同批`provenance.json`记录`88d39285`不一致，标`PROVENANCE_CONFLICT`，不以旧W1/W4与Serving跨环境数值证明今天D4的完整L1。
- **2026-10-27 L3-first性能门槛修订（ADR-015执行补充）**：大M每个候选先做现有A1/U1完整Stock L1与真实站点L2归因；只有真实Qwen3.6的单请求Prefill**设备Step关键路径**在不启Profiler的独立有效测量下出现可重复、超过Stock基线漂移的正向收益（`L3_GO`），**才申请正式4096/2048/50/10的L4 TTFT ABBA**。L3无收益/不确定/未具备真实Step计时（`L3_NO_GAIN/INCONCLUSIVE/MEASUREMENT_NOT_READY`）不启动L4，继续Kernel/布局/SDMA优化或补测量能力。旧计划2次短Profiler只负责L2与L3设备归因校准（按需、不充L3最终无Profiler性能GO）；L3轻量Screen后可申请4场4096/1无Profiler ABBA Confirm；L4若进入再单独4场4096/2048无ProfilerABBA。**4096/1客户端TTFT不可冒充L3 Prefill Step设备时长，也不规定L3必须≥5%**（旧占比仅约3.3%）。详见[更新后的A1-TTFT方案](L1_L4_HISTORICAL_REUSE_AND_TTFT_MINIMAL_CAMPAIGN_20261027.md)。
- **准入不变**：用户尚未反馈原仓CPU及四卡只读G0结果；CPU/G0/Smoke/Screen/Confirm/Profiler/E2E的新轮次全是NOT_RUN。Stock默认、43520 Die7排除、同卡双Die、异常STOP留证、分阶段独立审批及ADR-013禁重建Runner全部继续有效。工作分支PR#9保持Draft。

