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
