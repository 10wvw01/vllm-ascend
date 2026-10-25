# 310P3 MM+MemFabric AR｜项目控制台（唯一权威当前状态）

> **维护于** `10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`；2026-10-25。**新对话优先读本文件**。这是活动状态与交接合同，不是覆盖历史实验报告。所有未在当前主线更新的旧 README、PLAN、实验总结都只作为历史证据。
>
> **主线创建基底：** `perf/310p-u1-fullchain-attribution-20261025` @ `d0d57b94df06…`。基底已包含 D4 Tile2048 正式 Kernel、F1-C 实验及 U1 测试工具；U1 仅完成源码/静态审查，**没有新的 U1 实机证据**；主线建立本身没有修改正式设备 Kernel、Host ABI、通信协议或 Serving 开关。

## 0. 项目目标、边界与验收

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

## 11. R1-B主线选择性整合、下一次310P3准入（2026-10-25）

- 原始现场G0：分支 `work/310p-mm-ar-r1-g0-nogo-20261025` @ `45056500a8254787733ffb9d05e0c8efbbcb05d4`，隔离工作树从旧 `07586f2` 派生；**原始 frozen G0=NO-GO，Stock/V2+DFlash Eager/full/Graph均NOT_RUN**，保留 [现场报告](R1_G0_ONDEVICE_NOGO_REPORT_20261025.md) 和 [原始证据](../evidence/r1_g0_20261025/)。修复后 preflight 的单独静态GO不能改写历史裁决；本次工具整合也没有新的实机结果。
- 审查确认：现场真实 Python3.12.13、官方vLLM editable v0.27.1@6e448d0、torch-npu 2.10.0.post4、transformers 5.17.0，原G0使用的v0.24/旧pin及冷导入探针不符合已知环境。按真实Serve的 `NPUPlatform.pre_register_and_update` 初始化后导入通过；历史Stock+DFlash确切失败根因仍UNKNOWN。
- 以当前唯一主线 `592e708` 的 **用户原始服务参数** 为基底选择性整合诊断修复；绝不直接合入旧R1任务分支的试验脚本，以免重新引入host=127.0.0.1/max-seqs10/memory0.85/K7额外draft TP/额外加载策略和bench预热。
- 当前Trial冻结Target `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`；现场实测Draft `/home/models/Qwen/Qwen3.6-35B-A3B-DFlash`（比用户原始草稿路径多一级`Qwen/`），不通过猜测/符号链接回退。Stock/V2两臂JSON完全相同，K=7；`max-num-seqs=16`/0.90/host0.0.0.0，input4096/output2048/50请求/concurrency10，Eager NONE vs Graph待诊断；端口对外暴露前须核验网络隔离。
- 新G0：修复pin并同时**收窄**pip check豁免：仅允许指定声明者+指定缺包组合、精确已证实版本冲突，任何不在快照中的新依赖、冲突、空输出均STOP；强制真实Serve顺序导入探针 `exit_code=0` 才准进入模型，保持完整pip stdout/stderr。严格绑定python可执行路径、目标/草稿模型路径和本任务工作树；禁止editable finder静默指向主工作树。
- V2之前必须在本隔离工作树内核验 `vllm_ascend_C*.so`、`libmf310p_v2_trial_device.so`、`libmf310p_device.so` 存在，并人工核对真实Host/Device SO+KVER11 SHA与已核准现场部署；**脚本不自动复制、不自动重编/重装**。
- 下一次310P实机按 [R1-B准入与执行合同](DFLASH_STOCK_V2_R1_GO_GATE_20261025.md#8-r1-b-整合后-310p3-实机准入与止损合同) 的条件推进。优先Stock+DFlash Eager smoke，之后V2 Eager smoke及两个同口径full；Graph仍需Eager双组实机结果和独立人工许可。拒绝伪命中、失败后继续跑和拿日志代替双Rank Kernel trace。
- **截至本次合入：没有运行310P模型服务，K7实际推理PASS/Graph PASS/V2实际Kernel命中均UNKNOWN；Gate E与大M Stock L1仍未闭环。** GitHub CI只是无设备CPU合同测试，不是设备准入。

### R1-B第二轮静态审查补强（2026-10-25）

- 审查除G0诊断本身以外的启动边界：旧报告可能来自另一安装、另一Python或同名editable代码。Trial新增 `validate_runtime_identity` 校验预检报告Python执行器、真实vllm CLI脚本shebang、vllm/vllm_ascend真实根目录、四个关键分发版本与当前状态完全一致；CI用合成目录/假包元数据测试不一致时FAIL。不得据此声称实际服务已加载。
- 审查确认：对V2的build artifact校验仅检查当前隔离worktree里文件是否存在，**不验证设备已经正确装载、Host/Device ABI及SO哈希**；这些仍是OpenCode现场G0人工门禁，不允许用“存在文件”替代签名/运行期映射证据。

## 13. 310P3 V2 Model Runner 的 Triton 验证豁免（2026-10-25，研发阶段）

- 来源：[R1-B现场分支](https://github.com/10wvw01/vllm-ascend/tree/work/310p-r1b-stock-eager-smoke-nogo-20261025)：真实G0 PASS，Stock+DFlash加载Target后被混合sliding/full Draft的V1多KV组门禁阻断；显式启V2又被上游 `Model Runner V2 requires Triton` 阻断。此证据尚未通过本次代码分支替代/抹除，Stock+DFlash smoke仍FAIL；未有V2模型/Kernel性能数据。
- 用户决策：**优先310P3 V2 Runner（不再改造V1多KV组）**。官方不支持Triton设备的事实只支持采用非Triton后端的架构方向，不代表上游V2 GPU Runner可直接运行。
- 本次落地**仅一项条件豁免**：平台patch在`VLLM_ASCEND_310P_EXPERIMENTAL_V2_NO_TRITON=1`、`VLLM_USE_V2_MODEL_RUNNER=1`、真实310P3、vLLM 0.27.1、TP2、Eager(cudagraph_mode=NONE)、无投机或DFlash时，跳过上游 `_validate_v2_model_runner` 中的Triton存在性检查，保留其它上游unsupported_features/reasoning检查。没有设置开关时，完整调用原上游验证。绝不写`HAS_TRITON=True`，不替换依赖、不启用Graph。
- **安全关键**：`NPUWorker310.init_device`对V2标志增加显式`NOT_IMPLEMENTED`拦截，保证验证豁免后不会被旧310P Worker静默路由到V1，也不会误将通用GPU V2 Triton Kernel上机运行。故本次是**CONFIG-GATE-ONLY**，**不是**Stock+DFlash服务GO或310P3实机GO。
- 下一阶段另起短期工作分支：实现并审阅310P V2专用Worker/ModelRunner，完成真实vLLM 0.27.1的DFlash Speculator set_attn ABI、input_batch/状态更新Triton替代、多KV Group/Attention、W8A8+GDN+TP2契约，以及两轮静态审查与CPU测试后，才可申请受控NPU Eager。

### V2豁免二轮复审补充
- 单独设置`VLLM_ASCEND_310P_EXPERIMENTAL_V2_NO_TRITON=1`而未强制`VLLM_USE_V2_MODEL_RUNNER=1`，此前会由env-only属性继续选V1，存在“绕过成功但实为V1”的污染可能；已调整属性为明确抛错（仅310P并仅豁免开关打开时），并对flag-off的旧V1默认/显式False做回归。

## 14. R2 V2 Runner全量任务与验收门槛锁定（2026-10-25）

- 所有者要求提高效率，**先一次性锁定下轮全部必经工作**，不再“修一个启动报错→跑一次设备→再问下一步”；权威任务表为 [R2 V2 Triton-free Model Runner总执行计划](V2_RUNNER_TRITON_FREE_MASTER_PLAN_20261025.md)，立项基线为已合入的 `4b23e89`。
- R2 顺序：A 逐接口ABI/全调用链Triton盘点→310P专属V2 Worker+Runtime、纯NPU非Triton输入/状态、DFlash多KV、Qwen W8A8/GDN和采样、CPU合同/两轮静审→B 非投机V2 Target Stock Eager→C V2 Stock+DFlash K7→D 同V2 Runner接MemFabric融合 L1/L2；Graph/507014长稳列为R3独立验收。
- 关键已知缺口：`NPUWorker310`现仍V1派生并对V2显式拒绝；现有Ascend V2 DFlash `set_attn` API与vLLM 0.27.1不一致、输入和状态更新包含Triton；多KV GID/不同BlockSize/Sliding+Full、rejection与GDN/Mamba未获310P Eager证据。**任务列表不是实现完成声明。**
- 工作规则：R2-A由ChatGPT负责GitHub源码/UT/两轮静态审查/PR及CI，在达到静态GO且现场G0得到许可后，才交由用户委托OpenCode按分级Stop/GO上机。Stock默认保留，不改V1、不改上游、不默认启V2，失真Graph不计PASS。R1-B现场分支报告与ADR-010须先选择性归档，主线性能基线标记本轮 NO-NEW-PERF。

## 15. R2-00 R1-B原始实机证据选择性归档（2026-10-25）

- 已从历史 `work/310p-r1b-stock-eager-smoke-nogo-20261025@f51aa689` **只导入报告和 evidence/r1b_20261025 下原始证据 blob（含原始压缩完整日志）**，没有合并旧任务代码/项目状态；保留其元数据及原始 SHA，便于回溯。
- R1-B新事实：G0 **PASS**；Target双Rank完成1324条权重加载；Stock+DFlash K7 **FAIL**：原草稿5滑窗+1全注意力在V1 Runner校验失败；未做V2实际运行。后续full bench与Graph **NOT_RUN**，新吞吐/融合收益UNKNOWN，不能用该失败判断 310P V2 能否推理。R2目标是修复V2 Runtime与无Triton调用路径，不改V1或草稿配置。
- 证据：[R1-B现场报告](R1B_STOCK_EAGER_SMOKE_NOGO_REPORT_20261025.md) 与 [原始目录](../evidence/r1b_20261025/)；原始现场只读身份、SO、G0、Rank回溯与恢复状态详见报告。注意R2的当前分支/代码和历史R1-B对应不同提交及不同执行时间，不得把历史G0直接用于下一次上机。

### R2-01 / R2-06首批真实ABI源代码修订（仍不能上机）

- 与固定上游 `vLLM@6e448d0` 比对已确认：Ascend V2 DFlash `set_attn` 少 `target_input_buffers,target_attn_groups`；旧 `build_draft_attn_metadatas` 在0.27.1调用链不存在且少必填参数；Ascend V2 ModelRunner 的 `postprocess` 调用上游已删除的方法；`torch.cuda.current_stream()` 不适合310P；组合 speculative logits 应显式传 ModelState bonus-token 个数。
- 上述五项修订均已纳入本 R2 开发分支，并新增 `tools/310p_mm_ar/test_v2_runner_abi_0271.py` 纯CPU AST合同测试；仅证明这些**局部接口**更接近冻结上游，不证明 V2 Worker、Triton-free算子、DFlash KV/ModelState/TP2可执行。已知必经Triton仍阻断，不允许修改worker的NOT_IMPLEMENTED防护。

### R2 第2轮静态审查：通用 Ascend V2 防回归

- 调整首批ABI修补：`AscendDFlashSpeculator.set_attn`改为按实际继承的上游签名兼容新5参数或旧3参数，不为310P3修复而破坏其它设备旧ABI；`NPUModelRunner.postprocess`保留旧版受保护回调，vLLM0.27.1使用`postprocess_sampled`。真正上游0.27.1及310P无Triton执行仍需进一步验证，不能宣称合格。

### R2-01/02 固定ABI与Triton全路径静态审计（第一次覆盖，非闭环）

- 已创建 [vLLM0.27.1↔Ascend V2 API/ABI矩阵](V2_API_ABI_MATRIX.md) 和 [310P Eager必经Triton替代矩阵](V2_TRITON_REPLACEMENT_MATRIX.md)，识别GPU BlockTables多组gather/slot mapping、GPU采样rejection flatten、上游post_update和DFlash多组prepare_dflash_inputs等前次漏列入口。当前**10项BLOCKED、1项UNWIRED、1项UNVERIFIED、1项UNRESOLVED**。
- `tools/310p_mm_ar/test_v2_runtime_audit.py` 增加CPU源码合同：检查上游GPU input helper仍在本地V2 import、Worker fail-closed及固定SHA文档；它不替代上游真实路径解析、CPU golden与NPU adapter。**R2-02 OPEN / R2-A NOT-GO**。

### R2-07/08 多Group/DFlash K7参考语义合同（第一批）

- 新增`vllm_ascend/_310p/v2/reference_ops.py`：不依赖torch/npu、可验证的多KV组分配/slot编号及K7 bonus + 7 mask、rejected/context/query/sample indices参考实现，附`test_v2_dflash_k7_reference.py` CPU gold和负例。
- **仅Gold，不是NPU执行代码**：需要再将V2真正的 Triton prepare、group block tables、Attention metadata 和sampling映射接到受控310P非Triton后端，且必须验证内存布局/dtype、溢出、事件顺序。不能移除NPUWorker310拒绝标志。

### R2 CI修复记录（非NPU）

- 第一次加入R2审计表的CPU检测对TRITON矩阵错误要求必须包含字面`NO`，而表中阻塞统一采用`BLOCKED`；已收窄修复为接受`BLOCKED`或`NO-GO`，不更改实际Blocked判定与算子执行逻辑。

### R2文档级CPU断言清理

- 审计表使用两种状态列：ABI表为`| NO |`，Triton表为`BLOCKED`。首次两个CI循环均因测试对状态字面值过度假设而FAIL，现按真实表格列值校准测试；不变更代码路径或实机状态。

### R2 审查发现新增P0：GPU V2 RequestState/UVA及KV staging

- 对上游0.27.1 `gpu/block_table.py`、`gpu/buffer_utils.py`、`gpu/states.py`补充审查：除了已列多组gather/slot映射，KV分配时`apply_staged_writes`依赖 **Triton `_apply_write_kernel`**；`RequestState`构造又通过`UvaBackedTensor`/`StagedWriteTensor`依赖UVA。这些是先于DFlash推理的实际数据所有权路径，并非“装上Triton”可解。
- 310P3的UVA是否可用**尚未由现场确认**；需设计保留slot/状态契约的NPU安全替代或用真实相容性探针证明可用。已更新R2-01/02矩阵新增 T14/T15；**仍明确R2-A=NO-GO**，不得因ABI局部修补/CPU gold通过解除Worker锁。

### R2 实际实施状态总账（对话/跨分支唯一恢复入口）

- 详见 [R2执行状态动态账](R2_EXECUTION_STATUS_20261025.md)。截至R2分支`6163344`：R2-00 GitHub证据归档PASS（PR#8已并入主线）；R2-01/02 ABI/Triton静态审计仅PARTIAL，R2-03～10尚无可运行310P V2后端；R2-11/12 CPU CI 61项治理/ABI/语义UT +54项既有V2 pytest PASS；R2-13两轮运行时审查未完成。**当前STATIC NO-GO；不需要用户上机**。


## 16. R2隔离数据所有权候选实现（2026-10-25，仍STATIC NO-GO）

- R2分支新增`_310p/v2/staged_buffers.py`、`request_state.py`、`block_tables.py`和CPU合同测试：在不改上游/不装Triton的前提下，分别实现Eager 32B padded staged write、1D状态dirty scatter、无UVA RequestState候选、多组不同blocksize的gather/slot候选。所有模块**尚未接入**真实Runner；无310P实机测试，CPU Torch合同可能在轻量CI中Skip。
- 审查新增P0：原GPUModelRunner父构造先触发UVA，后替换req_states不安全；原GPU BlockTables也在KV初始化触发Triton；sampling GPU状态与CPU staging所有权存在陈旧镜像覆盖风险。候选只局部消减风险，不能将T05/T06/T14/T15/T16标READY；其它T01~09/T11和DFlash/GDN/Mamba仍未实现。
- 保留`NPUWorker310`的`NOT_IMPLEMENTED`、PR#7 DRAFT，**R2-A STATIC NO-GO / R2-B/C/D ONDEVICE NOT_RUN**。禁止因CI绿启动设备；下一步继续冻结构造链、非Triton输入/采样和真实DFlash KV协议。


## 2026-10-25 R2-A GitHub源码续研：CPU Tensor合同实跑，静态准入仍NO-GO

- 唯一主线`dev/310p-memfabric-mm-ar@a37c0212f4222317552e56c357a77930f85696f2`保持不动；全部新增代码/测试/CI/状态仅在`work/310p-v2-r2-implementation-20261025`，PR#7 **DRAFT、未合并**。未触碰外部仓库/310P服务器、SO、模型配置或Triton真实性。
- 修复隔离的310P V2候选：T06 DFlash直接读取的`block_tables.slot_mappings`之前未被slot计算结果写回（严重P0），现保持固定buffer地址、填充padded PAD并返回别名；T14阶段写在H2D异常前过早CPU commit，现失败poisoned并要求Runner销毁受影响状态；T15/T16请求复用先清理旧token尾部/prefill lookahead，禁止pending写尚未落地时释放。新增T02/T04 Eager Torch非Triton候选与纯Python gold，位置先以int32算再cast int64；**全部候选尚未接线**。
- **已核实CI**：PR#7 SHA`dfb4a0faa91e937dd83122e860697913b676beaa` 的GitHub Actions Run `40164930360`：轻量治理`79 unittest OK（3 skipped，无torch）+54 pytest PASS`，独立`Python3.12/torch2.10.0+cpu`动态合同`59 unittest OK`；**不是310P3设备验证**。新增`r2-torch-cpu-contracts`作业以避免Torch测试被轻量CI Skip掩盖。
- **STATIC NO-GO / ONDEVICE NOT_RUN**：上游GPUModelRunner构造期UVA RequestState与KV初始化GPU BlockTables尚未替换，T01/T03/T07~09/T11/T13未闭环，T02/T04/T05/T06/T14/T15/T16未接生产路由/设备算子核验，GDN/Mamba/W8A8/TP2/DFlash多Group及stream所有权未完成。Worker的`NOT_IMPLEMENTED`防护保持，Stock默认不变；不得委托上机或把CPU绿称为STATIC-GO。下一步优先构造链、完整input/sampling状态、DFlash多Group与双轮深审。


### R2-A 追加P0复核：状态容器整体不可恢复保护（未设备准入）

- 隔离候选`NPUBlockTables`与`NPURequestState`现增加`_poisoned` fail-closed：任一多组KV写、计数上传、请求新增/状态回写异常，不允许继续gather/slot/请求复用；补故障注入Torch CPU合同，详见[ADR-013](DECISIONS.md)。这不是跨组原子事务；未来Runner仍须实现异常abort、in-flight同步和销毁，当前未接线。
- PR#7仍DRAFT，主线与Stock默认不变，310P3 **STATIC NO-GO / ONDEVICE NOT_RUN**。新增代码后的CI以最新GitHub SHA为准，旧`dfb4a0fa`绿不可外推。


### R2-A 门禁工程化：31项ABI/Triton必须逐条给证据

- 研发分支新增`tools/310p_mm_ar/r2_static_admission.py --check`和`R2_STATIC_ADMISSION_EVIDENCE.json`，16条Triton/UVA/ownership + 15条ABI明确检查；当前账本空、`STATIC-NO-GO`，缺失review或接线不能借CI绿绕过。CI调用`--report`只展示NO-GO，静态GO需真实两轮审查与独立`--check`、用户再批准实机。
- 所有代码仍仅在PR#7工作分支；未改外部上游、现场SO、Draft config或NPUWorker310阻断。后续需完成真正Runner父类构造替代和T01~T16实际调用链，不能用现有Torch候选充当设备路径。


### R2-A GitHub CI复核：31项静态门禁保持NO-GO（2026-10-25）

- 真实SHA `f9a60ff5f64e3d6e1595f89c5a004490557eae0d`，GitHub Actions Run `40165197698` 两个job均**SUCCESS**：轻量治理 **82 unittest OK（3 skipped：无torch）+54 pytest PASS**；独立Python3.12/torch2.10.0+cpu **61 unittest OK**（未报告skip）。
- `r2_static_admission.py --report`在同次CI明确输出 **`R2-A STATIC-NO-GO: 0/31 mandatory nodes`**，因两轮审查尚未PASS、31项仍未真实接线/给足证据；这是按真实任务口径的正确阻断，不因CI成功改变。R2-B/C/D **ONDEVICE NOT_RUN**；Worker仍NOT_IMPLEMENTED。


### 2026-10-25 R2-A T07/T08采样合同研发记录（工作分支，不代表主线GO）

- 工作支路`work/310p-v2-r2-implementation-20261025`继续仅新增隔离采样候选与gold/CPU测试；主线`dev/310p-memfabric-mm-ar@a37c0212`不变，PR#7继续DRAFT。固定上游vLLM`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`只读。
- T07/T08候选不替换上游模块级Triton绑定，不解除`NPUWorker310`的NOT_IMPLEMENTED；所有Triton/UVA/多KV/真实采样与ModelState未审查闭环前仍**R2-A STATIC NO-GO / ONDEVICE NOT_RUN**。本次无设备、SO、模型配置和Graph操作；CPU单测仅作合同证据，GitHub CI须按本次新SHA确认。


### 2026-10-25 R2-A T03 CPU候选研发（工作分支）

- `work/310p-v2-r2-implementation-20261025`新增T03 sampled/draft输入组合Eager候选、纯Python gold和Torch CPU测试；T07/T08上轮`3467c8f4`的真实GitHub CI已成功（85治理UT含4 Skip、54 pytest、68 Torch CPU UT），不等于本次T03 CI或310P设备GO。仍保留上游Triton绑定、NPUWorker310 NOT_IMPLEMENTED，PR#7 DRAFT，R2-A STATIC NO-GO / ONDEVICE NOT_RUN。


### 2026-10-25 R2-A T03/T07/T08 GitHub CI核验（工作分支）

- PR#7代码`2ba84b8fa1daa142b679a359b1db6d525dcb0423`的[GitHub Actions Run 40169623722](https://github.com/10wvw01/vllm-ascend/actions/runs/40169623722) **SUCCESS**：87治理unittest（5 skipped：无torch）+54 pytest，独立torch2.10 CPU 73 unittest OK；`r2_static_admission --report`仍 **0/31 STATIC NO-GO**。新增代码是T03/T07/T08隔离候选，不是可执行Runner或NPU GO；主线与服务器均未修改，PR#7保持DRAFT。


### 2026-10-25 R2-A T01源码续研（非静态准入）

- 仅自有R2任务分支新增冻结vLLM0.27.1 `prepare_prefill_inputs`的Torch Eager候选与stdlib CPU oracle、随机/负例Torch合同，覆盖prefill、lookahead、混合decode及请求slot隔离；参见`R2_EXECUTION_STATUS_20261025.md`和`V2_TRITON_REPLACEMENT_MATRIX.md`。T01仍未接生产Runner，所有GPU V2 UVA/BlockTables、T09/T11及采样多KV调用链仍阻断。
- **R2-A STATIC NO-GO / R2-B/C/D ONDEVICE NOT_RUN**；Worker fail-closed与PR#7 DRAFT不变，CI绿不能替代310P设备算子、stream/所有权审查及两轮静态深审。新代码CI必须按新SHA取证，不复用旧计数。

### R2-A T09尾部状态正确性P0修订（2026-10-25；工作分支）

- 310P专属V2候选的`post_update_ops.py`在row仅余1个槽、sampler宽度大于1时，原`scatter_`对重复clamp列的masked写可能覆盖真实accepted Token；改为只对accepted位置累计差值，并将空penalty vocab和int64接受token→int32范围校验提前到首次持久写之前。附Torch CPU边界回归；无NPU或性能结果。
- 当前仍**R2-A STATIC NO-GO 0/44**；T09/ABI-04依旧BLOCKED/NO，第二轮静审未开始，NPU Worker仍NOT_IMPLEMENTED，PR#7继续Draft，Stock默认与主线未变。此修订CPU CI尚待按新SHA核验，设备写原子性/流/错误传播仍是实机准入阻塞。
