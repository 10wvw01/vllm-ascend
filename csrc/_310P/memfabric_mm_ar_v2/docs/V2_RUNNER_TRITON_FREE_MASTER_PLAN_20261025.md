# R2｜310P3 Triton-free V2 Model Runner + Qwen3.6 DFlash：全量执行与验收锁定表

> 决策日：2026-10-25。状态：**任务与门槛已锁定；尚未完成 V2 Runtime 适配；NPU NOT_RUN**。唯一长期主线 `dev/310p-memfabric-mm-ar`；立项基线 `4b23e89a62b330e33c3ffddc29300192e8d1dbdc`。此表是**今后各轮实现、审查和实机任务的任务总账**，不是静态审查/实测 PASS 的证明。主控状态以 [PROJECT_CONTROL.md](PROJECT_CONTROL.md) 为准。
>
> 适用范围：**我们自己的** `10wvw01/vllm-ascend`；官方 vLLM `/vllm-workspace/vllm` @ `6e448d0ea9bf…` 只读且实际版本 `0.27.1+empty`；HaonanWu1 的 310P DFlash 分支只读；MemFabric 官方/安装库不因本 R2 修改。发现新的必经调用路径时须**追加任务、风险和验收**，不得用「已全部覆盖」掩盖未审计路径。

> [R2当前执行状态及阻断账](R2_EXECUTION_STATUS_20261025.md)：持续更新已完成/未完成及下一步；不得将PR中的局部CI结果写成设备GO。

## 1. 目标、交付边界和执行原则

本项目的 V2 有两个完全不同的对象：

- **V2 Model Runner**：推理框架的第二代 GPU/NPU ModelRunner。本 R2 先要在**无功能性 Triton 的 310P3** 实现 Target Qwen3.6、混合注意力 DFlash Draft、K7、TP2 **Eager 正确推理**；现阶段只完成了 `HAS_TRITON` **配置校验的条件豁免**，`NPUWorker310` 故意 `NOT_IMPLEMENTED` 阻断 V2 执行。
- **MM+MemFabric AR V2 Kernel**：我们定制的矩阵乘法、SDMA AllReduce、本地 Add 融合 Kernel。其工程不因 Runner 迁移而推倒重来；只在同 Runner 的 **Stock+DFlash Eager** 正确后再接入，完整 Stock MM+HCCL AR 是融合效率对照。

**整体次序锁定**：R2-A CPU/静态/代码实现 → R2-B V2 Target Stock Eager → R2-C V2 Stock+DFlash K7 Eager → R2-D V2+融合 Eager 和 L1/L2 性能 → R3 Graph/长稳发布门槛。只按依赖关系推进；不得因为 R2-A 代码编译通过就直接执行 R2-D 或 Graph。

**不做**：扩展 V1 多 KV Group、安装/伪造 Triton、全局 `HAS_TRITON=True`、跳过不支持的 V2 特性检查、换模型/重写 config.json、把 V1/Graph Stock fallback 说成 V2 命中、自动替换 SO、向外部仓库推送、顺手改 MoE/GDN/Attention 性能、将 Stock 改为默认融合。

## 2. 不可变环境与测试合同

| 项目 | 冻结口径 |
|---|---|
| 设备 | 310P3 **单卡双 Die TP=2**；先前安全设备 6/7、同卡 Bus `0000:AB:00.0`，**每次重做 G0 以现场为准**；不碰其他人的作业/设备4遗留显存 |
| 运行时 | Python 3.12.13，vLLM 0.27.1+empty / `6e448d0ea9bf…`，torch 2.10.0、torch-npu 2.10.0.post4、transformers 5.17.0；CANN 9.1.0 / Driver 26.1.1，按现场 G0 重新核 SHA |
| Target | `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`；W8A8 ModelSlim/Ascend 量化；含 GDN/Mamba/MoE |
| Draft | `/home/models/Qwen/Qwen3.6-35B-A3B-DFlash`；`layer_types` = 5×sliding + 1×full，`sliding_window=4096`；K=7，不替换权重 |
| V2开关 | `VLLM_USE_V2_MODEL_RUNNER=1` + `VLLM_ASCEND_310P_EXPERIMENTAL_V2_NO_TRITON=1`；**仅310P3/TP2/上游0.27.1/Eager**，其他原有校验保留 |
| 原服务参数 | `--host 0.0.0.0`（现场确认隔离网络）、`--dtype float16`、`--tensor-parallel-size 2`、`--quantization ascend`、`--gpu-memory-utilization 0.90`、`--max-num-seqs 16`、`--max-num-batched-tokens 4096`、`--max-model-len 8192`、`--mamba-ssm-cache-dtype float16`、`--mamba-cache-mode align`；保留原本明确的其它模型/调度选项 |
| Eager | 强制 `cudagraph_mode=NONE`；`enable_flashcomm1=false`、`enable_flashcomm2_parallel_size=0`、`enable_prefill_mc2=false`、`enable_fused_mc2=0`，禁止额外启用会改变基线的加速特性 |
| 业务压测 | 输入 4096、输出 2048、TP2、并发10、请求50、`ignore-eos`；记录 TTFT/TPOT/ITL、throughput、acceptance、physical M；冒烟另用短输出，不得混用 |
| Stock 对照 | S1：**V2 Runner + 原生 MM+HCCL AR**，融合两项开关=0；S2：**同一 V2 Runner + V2 MM+MemFabric AR**。V1 Runner Stock 为历史/迁移成本参照 S0，**不可拿 S0 vs S2 冒充 Kernel 融合收益** |
| 历史证据 | [R1-B现场NO-GO](R1B_STOCK_EAGER_SMOKE_NOGO_REPORT_20261025.md) 当前尚在 `work/310p-r1b-stock-eager-smoke-nogo-20261025`；G0 PASS/Target加载成功，但 Draft 在 V1 混合注意力门禁失败，**V2 Runtime和Graph NOT_RUN**。若主线尚无该报告，以上相对路径不得视作已落盘：以[证据任务分支](https://github.com/10wvw01/vllm-ascend/tree/work/310p-r1b-stock-eager-smoke-nogo-20261025)为准，R2-00负责归档整合 |

## 3. 全量工作分解与依赖（唯一任务编号）

**阶段 R2-A：完成后一次性交付静审+CPU门禁，未完成不得向用户申请实机。** 工作可由代码/测试/文档分模块并行，但公共 API/类型与关键路径先约定再实现；一个短期集成分支集中复核，避免每修一个报错又跑一轮设备。

| ID | 必须完成的事情（包含已知关键路径，不限于枚举的函数） | 输出证据 / 验收 | 依赖 |
|---|---|---|---|
| R2-00 | 将 R1-B 原始证据、报告和 ADR-010 **选择性审阅**后合入主线；不得将旧实验代码整体 merge；更新主控和基线 NO-NEW-PERF | 原始 log/G0/commit/source 指向同一次任务，不伪称 V2 PASS；主线可读 | 无 |
| R2-01 | 冻结上游 `vllm@6e448d0` API/源码、服务器依赖/真实编译产物；对照 `vllm_ascend/worker/v2/**`、310P原实现；编写**入口→Kernel**调用路径矩阵和 `inspect.signature`/AST 漂移测试 | `V2_API_ABI_MATRIX.md`，每个重写函数的上游签名、返回值、调用者、版本与测试 | R2-00 |
| R2-02 | 逐节点梳理 V2 Eager **必经 Triton**：`prepare_prefill_inputs`、`prepare_pos_seq_lens`、`combine_sampled_and_draft_tokens`、`get_num_sampled_and_rejected`、`post_update`、`post_update_num_computed_tokens`、`expand_idx_mapping`、上游 `prepare_dflash_inputs`、Ascend V2 `post_update` 及拒绝采样等调用；区别只定义未调用的 `@triton.jit` | `V2_TRITON_REPLACEMENT_MATRIX.md`：每条的 Eager/Graph 可达性、输入输出、替换方法、UT；未分类的必经节点=NO-GO | R2-01 |
| R2-03 | 310P Worker/Runner 路由：`platform.check_and_update_config`→`worker_cls`→`NPUWorker310.init_device`→真正的 **NPU V2 ModelRunner**，安全继承原310P设备/RC内存、分布式初始化/TP/HCCL、workspace、生命周期；V1开关关时原行为不变 | Runner 实际 class/import root 双Rank可取证；真正 V2 完成后才移除现有 V2 fail-closed，异常不自动回 V1 | R2-01 |
| R2-04 | 310P V2 核心 ABI：`GPUModelRunner`→Ascend V2的初始化、模型加载、`prepare_inputs`、`prepare_attn`、`execute_model`、`postprocess_sampled`、`sample_tokens`、`shutdown`；核对 vLLM 0.27.1 的 `BatchDescriptor`/`RequestState`/ModelState 返回合同、静态/动态图差异；Eager-only | 函数表无未对齐覆盖，禁止以通过构造函数冒充服务完成 | R2-01,R2-03 |
| R2-05 | 非 Triton 输入/状态更新：用 **310P已验证的 AscendC/CANN/torch_npu/PyTorch** 实现 R2-02 的全部实际调用；处理 prefill、decode、spec token、padding、index/positions、sampled/rejected、post-update、异步可见性；防 310P int64 Add/unaligned MTE/首写风险 | 逐算子 CPU 参考比较 + NPU前静态别名/越界检查；没有 `HAS_TRITON=True`、placeholder `kernel[grid]`、偷偷同步替代行为 | R2-02,R2-04 |
| R2-06 | DFlash Speculator 与 vLLM 0.27.1 完整 ABI：当前 `set_attn` 少 `target_input_buffers,target_attn_groups`；`build_draft_attn_metadatas` 命名/签名与上游 `_build_draft_attn_metadata` 不一致；`propose` 参数、`_group_causal` 属性和完整上游调用链都核对 | 精确 ABI 测试捕获参数/返回值，禁止只修一个 TypeError；可构建 speculator，不碰上游 vllm | R2-01,R2-04 |
| R2-07 | DFlash 输入构造移植：V2 上游 `prepare_dflash_inputs` 必经 Triton，310P已有 `npu_copy_and_expand_dflash_inputs` 仅在旧Proposer路径；新 V2 需同语义覆盖 N=7 + bonus=8、chunk prefill、rejected、sampling seeds/temperature、PAD slots | token/position/slot/context/seed 对照 CPU gold；绝不让旧Proposer当作已兼容V2 | R2-05,R2-06 |
| R2-08 | 多 KV Group：保留 Draft 5滑窗+1全注意力、per-group causal、每层 GID 映射、KV Page/Block Size、BlockTables、Slot Mapping、预写Context KV、mask/窗口裁剪与请求释放/重用；核对 310P block size 64/128/128对齐及非因果 Attention 后端 | 独立多组/不同物理BlockSize/尾部窗口/回收重复执行的参考UT；错KV/错层即NO-GO | R2-04,R2-06,R2-07 |
| R2-09 | Qwen3.6 Target 兼容：W8A8/ModelSlim，GDN/Mamba与Attn混合缓存、MoE、FP16非量化 o_proj、模型权重重命名/实际加载、TP2、HCCL和上下文状态；确认 V2 model states 的 Mamba/Attention 组合 | 可核验的纯CPU结构/权重scheme合同 + 实机分层 smoke 阶段；禁将 Target 权重加载成功等于生成正确 | R2-04,R2-08 |
| R2-10 | 采样/输出：DFlash draft_logits、rejection sampler、接受率/撤销、随机性/种子一致性、流/事件等待、async scheduler、不同请求长度、EOS/abort；覆盖各组KV的请求生命周期和显存释放 | 每轮 token及KV状态参考、全拒绝/全接受/部分接受/混合Batch测试；无静默丢失 | R2-05,R2-08,R2-09 |
| R2-11 | 安全治理与真路由：两级feature flag、V2 Runtime class assert、Stock融合开关=0/1、Unsupported mode拒绝、Graph拒绝、模型路径/版本/上游SHA/已装SO指纹验证、失败诊断不静默fallback；保留现有未移植V2 fail-closed直到替代足够完整 | `R2_STATIC_GO.json` 可机器解析，列出 every mandatory gate，不能用启动日志代替调用trace | R2-03~R2-10 |
| R2-12 | **同一批次实现全面测试**：CPU UT + 契约/无设备 import mock + schema/ABI + 负例/强制STOP + mock双Rank + 无Triton import/dispatch；检视 310P GDN/Mamba/量化/多组KV/随机采样边界 | CI现有项目门禁 + 既有全部 CPU/V2 UT + 本批新UT PASS；列数值，不预先伪定数量；未覆盖列表显式NO-GO | R2-01~R2-11 |
| R2-13 | 两轮静态深审：第1轮独立审模型/Kernel ABI、调度数据流、内存安全；修复→第2轮复核异常路径、设备约束、部署回退；G0/runbook与trace字段定版 | `R2_STATIC_REVIEW.md` 含问题→修复SHA→复审结果、准入 GO/NO-GO；CI绿只证明CPU | R2-12 |
| R2-14 | 工程交付：单一 `work/310p-v2-runner-r2-integration-*` 短命分支+PR，所有正式源码/诊断/单测/本文任务状态/PROJECT_CONTROL/ADR/perf同步；仅合审过的改动；自有repo | 全部必需审查项 PASS后通知用户去安排一次 **R2-B/C/D分段实机活动**；未经用户同意不主动改服务器SO/工作树 | R2-13 |

**阶段 R2-B/C/D：同一套取证工具与冻结基线，现场由 OpenCode 在用户授权后执行；先取得真实信息，再用独立 GO 决策逐段放行。**

| ID | 现场执行或后续开发任务 | 完成标准与阻断 |
|---|---|---|
| R2-15 | **现场 G0**：隔离 worktree/产物/进程/端口、设备6/7与同卡 Bus、Driver/CANN、Python CLI/import root、HCCL、Stock禁融合标志、Host+Device SO/KVER两处SHA、实际 /proc/maps；所有旧工作树不得checkout/clean；部署变更须备份+明确审批 | G0 全PASS才允许真正 NPU；任何未知包异常、被占用设备、SHA不一致 STOP |
| R2-16 | **R2-B：V2 Runner + 无DFlash Target Stock Eager**：先双Rank模型加载/最小正确性、再 1→10 并发、短出输出；逐层确认W8A8、GDN/Mamba、KV与真正 V2 class | 真实Token生成与正确性，双Rank无数据错误；单次服务健康/模型加载不算 PASS |
| R2-17 | **R2-C：V2 Runner + 当前Draft DFlash K7 Stock Eager**：保留模型、K7 TP2，small smoke→多batch混合→请求复用与接受率；真实观察多个draft KV Group、masked/full attention、上下文KV写入、Spec proposer、接收/拒绝及实际物理/padded M | 真实 DFlash token + reference / token validity、双Rank一致性、非零且合理的 acceptance、无奇异回退；若 R2-B 未过禁止执行 |
| R2-18 | **R2-D集成适配**：仅在 R2-C通过之后接入已有 `vllm_ascend/_310p/ops/memfabric_mm_ar*`/正式V2 Kernel/LinearMethod；目标层未量化FP16 o_proj、Threshold 2048~4096、双Rank route、layout/首写与同步，审图/真实 kernel trace | Stock同 V2 Runner 默认不变；V2融合实际调用和返回数值逐层可见；缺SO必须安全构建/核对不能从老工作树暗拷 |
| R2-19 | **L1**：相同FP16 M/N/K/布局的完整原生 MM+HCCL AR vs MM+MemFabric AR，含 input.contiguous、SDMA/规约/同步/拷贝，ABBA双Rank p50/p90/p99，冷首写与长尾 | 真实完整算子差距；不以D4内部+6.26%当融合Stock收益 |
| R2-20 | **L2**：严格 S1=V2 Runner+Stock 与 S2=V2 Runner+融合，输入4096/输出2048/并发10/50/TP2/ignore-eos；3次以上成对重复与随机/ABBA顺序，性能波动、输出速率/TTFT/TPOT/ITL/接受率/物理M、fusion命中覆盖率 | 解释 L1→L2 传导不足（目标层占比、真实eligible M、Decode/Verify Stock fallback、其他Kernel、等待/overlap）；S0旧V1只单列迁移差异 |
| R2-21 | **现场恢复与证据入主线**：失败即保存 Rank0/1 trace、配置/SHA/实际装载地址，终止服务并核验设备恢复、端口清空、已备份SO一致；每个阶段 PASS/FAIL/NOT_RUN 独立归档；新的错误修复后只重跑受影响阶段，不虚构前序结果 | 完整原始证据、Go gate裁决、PR与 PROJECT_CONTROL/DECISIONS/PERFORMANCE_BASELINES 同步 |

**阶段 R3（已锁定但不作为 R2 Eager 上机阻塞）：**

| ID | 后续事项 | 独立准入 |
|---|---|---|
| R3-01 | 310P3 V2 ACL Graph capture/replay 与 Triton-free 编译/状态更新；明确形状/padding/K7 descriptor、Stock和融合两臂真实 graph/captured kernel | R2-C Eager通过且单独出Graph方案、补丁、CPU和NPU审查，不复用Eager通过结论 |
| R3-02 | V2+MemFabric Graph 真融合、Full/Piecewise路由与Async scheduler/draft KV缓存生命周期；必要时禁用不支持模式并显式告知 | capture/replay双Rank真实自定义Kernel调用；出现Graph Stock伪命中一票否决 |
| R3-03 | Gate E：507014/507899、首写静默损坏、pool init/destroy、混HCCL与MemFabric、长稳与恢复 | 未全部关闭不得发布/默认启融合；性能研究仍可以隔离开展 |
| R3-04 | 大M正式Kernel继续冲极限、小M独立单Kernel fixed-overhead路线，仍坚持完整Stock对照；与R2 runner适配**可并行**，互不改写历史基线 | 此为原项目核心性能线，不因为投机运行时研发而冻结，不在R2 Worker任务中顺手改正式Kernel |

## 4. 并行规则：一次开发交付、一次分段实机，不反复试错

- **并行轨**：A=R2-01/02 API+Triton矩阵；B=R2-03/04 Worker+Runner基础；C=R2-06/07/08 DFlash+多KV；D=R2-05/09/10 非Triton执行+模型/采样，待共享接口定稿后可并行实现。R2-11/12与各轨同步写自动测试，R2-13/14统一收敛。代码负责人（ChatGPT）负责可审源码/测试/文档/PR/CI；用户只需在达到实机GO时委托OpenCode实机、部署或异常取证。
- **硬依赖**：不能在 R2-04 未通过时假设 DFlash 能用；R2-08 混合KV数值未闭环禁止把草稿模型config改为全Full Attention；R2-16 Stock Target失败禁止运行 DFlash；R2-17 DFlash Stock失败禁止 V2融合性能；Eager失败禁止Graph。
- **省轮次**：上机提示词一次包含所有已获准阶段的脚本、短冒烟和故障留证；让OpenCode通过每级可机器判定的结果报告继续下一组，不要求每个小错误重复执行旧G0，**但每次独立上机仍须确认设备/安装身份**。
- **永久止损**：507014/507899、首写/MTE/缓存可见性不满足、数值错、Rank漂移、unknown pip、不明SO、设备占用/外部进程、Graph假命中，立即 STOP 并保全；绝不能通过删除cache clean、signal、credit/wait、降低精度或伪造Triton规避。
- **未审核的新发现**：必须记录到当前计划的 `GAP-<ID>`，标明根因/owner/预计需要修改哪些接口/UT/实机门槛；只有据实更新本计划、ADR（若变更范围）、控制台才能新增或撤销，不用新对话口头决定。

## 5. 在进入 R2-B 设备前必须交付的五份可审文件

1. `V2_API_ABI_MATRIX.md`：上游 vLLM 当前 SHA ↔ 当前 Ascend V2 API/返回值/调用者逐接口表；尤其 `set_attn`、`_build_draft_attn_metadata`、`propose`、Runner 的 `prepare_inputs/postprocess_sampled`。
2. `V2_TRITON_REPLACEMENT_MATRIX.md`：每个必经 Triton Kernel 的引用图与310P非Triton实现、输入输出 golden、UT、fallback/Graph适用状态。
3. `R2_CPU_CONTRACT_REPORT.md`：新增与现有全部UT、模拟无Triton运行、双Rank/多KV/混合长度/token acceptance负例，实测命令+计数+SHA。
4. `R2_STATIC_REVIEW.md`：两轮静态审查、修复回归、风险与NPU GO/NO-GO，至少一轮独立于代码作者的人工核查逻辑。
5. `R2_ONDEVICE_RUNBOOK.md`：冻结启动参数、真实Worker身份、分层设备脚本/结果判定、G0/SO指纹、止损/恢复、R2-B/C/D独立报告和证据树。

上述是**要完成的交付件**，不是当前已存在的文件；首次 R2-A 开发PR完成上述文档和代码才能提交实机。R1-B原始现场数据须先安全合入/引用，保持历史NO-GO，严禁更改结论。

## 6. 本次锁定裁决

- **下一项实际工作**：从本计划执行 R2-00 / R2-01 / R2-02 与对应CPU API测试，随后不间断推进 R2-03～R2-14；不要停在每个补丁后问一次“是否可以继续”。唯一等待用户介入的正常节点是**静审/CI/可回退方案全部完成后，确认310P3实机运行与可能的SO部署**。
- **本计划锁定的是事项、依赖、验收标准，不预言硬件适配一定成功。** 若发现公共Scheduler/Request State必须大规模替换，应给出单独成本/范围证据，并在用户裁决前不得作为“兼容小补丁”静默扩大。
- 进入设备前的状态必须写作 `STATIC-GO / ONDEVICE-NOT_RUN`；配置豁免PR#5当前只证明CPU级特定条件跳过Triton校验，不等于V2 Runtime GO。

**基线参考**：[310P V2校验豁免PR#5](https://github.com/10wvw01/vllm-ascend/pull/5)；[R1-B现场证据分支](https://github.com/10wvw01/vllm-ascend/tree/work/310p-r1b-stock-eager-smoke-nogo-20261025)；[工程准入合同](ENGINEERING_GATES.md)。


### R2-A 静态准入可执行账本（2026-10-25）

- 新增`tools/310p_mm_ar/r2_static_admission.py --check`：严格读取冻结0.27.1的ABI-01~15和T01~16表格行，缺失、BLOCKED、UNVERIFIED、UNWIRED、NO均返回`STATIC-NO-GO`（exit 2）；只有显式READY/PASS且每节点具备源码/CPU测试/实际路由/设备算子静审/两轮审查证据，才可输出`STATIC-CHECKS-COMPLETE`。不可把`STATIC-CHECKS-COMPLETE`当作自动上机授权。
- `R2_STATIC_ADMISSION_EVIDENCE.json`当前刻意留空并标记`NO-GO/ONDEVICE_NOT_RUN`；禁止把CPU Torch绿或候选文件存在自动填为READY。CI仅调用`--report`打印阻断账，保证研发分支可以在NO-GO状态继续提交和跑CPU回归；人工准入必须调用`--check`并完成两轮静审。
