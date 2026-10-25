# R2-02｜310P3 Triton-free V2 Eager必经路径与替代矩阵

> **纠偏（2026-10-25）**：本表盘点的是通用GPU→310P候选的**静态潜在调用路径**，并非指定模型已现场执行29/31条Triton。唯一Target锁定`Eco-Tech/Qwen3.6-35B-A3B-w8a8`、310P3/TP2、4096/2048/batch10+DFlash K7；原310P V1 DFlash输入已经AscendC替换，冻结GPU V2 DFlash仍含Triton。当前V2 Worker尚未启动。详见 [具体模型路由证据](R2_TARGET_MODEL_DFLASH_ROUTE_20261025.md)，实现任务必须先有动态分支条件证据。\n>\n> 本文来源：锁定上游 `vllm@6e448d0`、我方Ascend V2及310P模块静态复核。**保守估计，不等于动态覆盖证明**。CPU测试和调用图可静态发现入口；必须在实际纯NPU适配完成并验证接口后才能把某条改为`READY`。
>
> 运行期：310P3无功能性Triton，`HAS_TRITON=False`。vLLM自带`TritonPlaceholder.jit`只是导入用装饰器，不实现`kernel[grid]`。**配置跳过Triton校验≠允许真正调用Triton内核**。

| ID | 调用源→Triton函数 | 现场可达性（Eager, Target / DFlash K7） | 替代来源/设计 | 状态 |
|---|---|---|---|---|
| T01 | Ascend V2 `prepare_inputs` →上游 `input_batch.prepare_prefill_inputs` | Prefill **必经** | 已有作用域Eager绑定；32B按请求对齐lookahead物理行及整行写候选（63b93afa），CPU合同通过；其它T01写入、as_strided/index_copy_设备/流审查未通过 | BLOCKED |
| T02 | Ascend V2 `prepare_inputs`→ `prepare_pos_seq_lens` | 每批 **必经** | 310P int32 staged position/seq_len，不走不对齐 int64 Add | BLOCKED |
| T03 | Ascend V2 `prepare_inputs`→ `combine_sampled_and_draft_tokens` | 每批 **必经** | 分块prompt尾/last_sampled/draft tokens；ModelState bonus count0/1、logits_indices | BLOCKED |
| T04 | Ascend V2 `prepare_inputs`→`expand_idx_mapping` | 仅含draft token批次 **必经** | index expansion、mask、padding，对照上游gold | BLOCKED |
| T05 | GPU V2 `BlockTables.gather_block_tables`→`_gather_block_tables_kernel` | KV Attn有效请求 **必经** | NPU安全多Group BlockTables；不能直接复用旧V1派生input_batch对象 | BLOCKED |
| T06 | GPU V2 `BlockTables.compute_slot_mappings`→`_compute_slot_mappings_kernel` | KV Attn **必经** | 多Group Eager候选已接，0e819f23补32B物理行与整缓冲copy；需设备验证非连续逻辑view stride、算子/流、5 Sliding+1 Full异构blocksize与mask | BLOCKED |
| T07 | 上游 `Sampler`或`RejectionSampler`→`get_num_sampled_and_rejected` | Decode/Spec sampling **必经** | NPU sampled/rejected及prefill-mask，避免输出数错 | BLOCKED |
| T08 | 上游`RejectionSampler`→`_flatten_sampled_kernel` | K7 Verify **必经** | NPU rejection、sample positions与随机数接口，FP16重复测试 | BLOCKED |
| T09 | `GPUModelRunner.postprocess_sampled`→上游`post_update` | 有生成token **必经** | 310P专属候选已override并调用`post_update_ops.py`，补masked尾部重复列覆写、Token int32溢出/旧状态容量与空penalty预检CPU负例；310P3`_assert_async/scatter_add_/index_copy_`、多Buffer原子性/异常/Stream未审，通用Ascend仍保留Triton | BLOCKED |
| T10 | `GPUModelRunner.postprocess_num_computed_tokens`→`post_update_num_computed_tokens` | PP>1/其它分支；**当前TP2/PP1可能不可达，需Trace证实** | 保留禁止默认/显式阻断PP；如实际调用必须替代 | UNVERIFIED |
| T11 | `DFlashSpeculator.propose`→上游`prepare_dflash_inputs` | DFlash K7每Draft group **必经** | 310P原`npu_copy_and_expand_dflash_inputs`可复用**语义**，需另写V2 ABI适配；不能直接调用V1 Proposer | BLOCKED |
| T12 | 我方 `worker/v2/input_batch.py:post_update`的`_post_update_kernel` | 本地函数存在，**未证实当前GPU ModelRunner自动引用** | 避免假替换；只有调整实际`GPUModelRunner.postprocess_sampled`调用才算覆盖 | UNWIRED |
| T13 | GPU V2 `ModelState`/KV初始化、采样辅助模块/Graph Manager等隐式Triton或CUDA调用 | 尚未穷尽；**需要逐路径import/AST静态和真实设备Trace** | 对每个继承方法审查`torch.cuda`、`triton`、`torch._inductor`及单次运行需的CANN替代 | UNRESOLVED |
| T14 | `gpu/block_table.py:BlockTables.apply_staged_writes`→`buffer_utils.StagedWriteTensor.apply_write` 或 `FusedStagedWriter.apply`→`_apply_write_kernel` | **Prefill/Decode每次新block**，不止Gather/Slot，TP2多Group时尤其必经 | 需要替换整条`stage_write→device state`链，禁止沿用默认GPU Buffer类的`_apply_write_kernel` | BLOCKED |
| T15 | `GPUModelRunner.__init__`→`gpu.states.RequestState`→`StagedWriteTensor(uva_instead_of_gpu=True)`、`UvaBackedTensor`→`UvaBufferPool` | **Runner构造期**；当前Ascend`AscendRequestState`继承此父类（需检查310P是否有实际UVA映射支持），与T14共同形成先于Kernel的P0门槛 | 310P3可用性需严审`is_uva_available`和`get_accelerator_view_from_cpu_tensor`；若不支持需独立NPU-safe RequestState/BufferPool实现 | UNVERIFIED/P0 |
| T16 | 310P staged writes ↔ GPU post_update 设备所有权与in-flight写序 | 新请求/请求复用/采样每批可能触发 | 32B设备向量RMW、pending三buffer回收检查及继承ModelState释放前预检候选（e7158aaf）；仍需stream/event barrier、活跃行回写、异常销毁全链 | BLOCKED |

| T17 | `worker/v2/sample/gumbel.py:gumbel_sample` → `_gumbel_sample_kernel` | Sampler随机采样/温度>0 **可达**，patch_v2显式替换上游全局 | 310P无Triton的确定性/随机数等价Gumbel实现，冻结seed/position分布合同 | BLOCKED |
| T18 | `worker/v2/sample/gumbel.py:apply_temperature` → `_temperature_kernel` | 采样温度缩放 **可达** | 设备Torch/AscendC温度变换、0/1特殊值和FP32语义 | BLOCKED |
| T19 | `worker/v2/sample/penalties.py:apply_penalties/bincount` → 两个Triton kernel | 频率/重复惩罚、prompt统计 **按采样参数可达** | 无Triton bincount/penalties及CPU gold；确保请求重用不污染 | BLOCKED |
| T20 | `worker/v2/sample/min_p.py:apply_min_p` → `_min_p_kernel` | min_p采样 **条件可达** | Torch Eager masked logits与稳定归一化合同，未接线 | BLOCKED |
| T21 | `worker/v2/sample/bad_words.py:apply_bad_words` → `_bad_words_kernel` | 禁词约束 **条件可达** | NPU安全禁词mask、prefix匹配与请求生命周期 | BLOCKED |
| T22 | `worker/v2/sample/logprob.py:compute_token_logprobs/compute_topk_logprobs` → `_topk_log_softmax_kernel/_ranks_kernel` | logprobs输出/采样日志 **条件可达** | FP32 logsumexp、top-k/rank、输出形状的无Triton合同 | BLOCKED |
| T23 | `worker/v2/spec_decode/rejection_sampler_utils.py:rejection_sample` → `_probabilistic_rejection_kernel/_resample_kernel` 等 | DFlash K7 rejection verify **必经** | **P0：Ascend现有实现温度>0分支令u=0恒接受，不能冒充正确拒绝采样**；需真正概率采样/Greedy合同及无Triton实现 | BLOCKED |
| T24 | `worker/v2/structured_outputs.py:_apply_grammar_bitmask_kernel` | 结构化输出条件可达；本期是否禁用须可执行证明 | 若不可达提供双审证明，否则非Triton grammar mask实现 | UNVERIFIED |
| T25 | 冻结`GPUModelRunner.sample_tokens`对`req_states.draft_tokens[idx_mapping] = draft_tokens`的K7短宽写 | DFlash K7每次提案 **必经**，非Triton但310P写粒度风险 | 32B padded 8-token整行更新/无跨请求覆盖及in-flight ownership；禁止直接7×int64窄视图scatter | BLOCKED |

| T26 | 冻结`Sampler.__init__`→五个`SamplingStates/PenaltiesState/LogitBiasState/BadWordsState/LogprobTokenIdsState`嵌套UVA/StagedWrite | **load_model每次必经**；替换RequestState不能覆盖 | 专属`NPUSampler310V2`局部克隆全部五类状态构造器，注入NPU-backed/StagedWrite；设备状态和生命周期待双审 | BLOCKED |
| T27 | 冻结`GPUModelRunner.load_model`→`StructuredOutputsWorker.__init__`→`torch.cuda.Stream()` | **load_model必经**，即使无grammar | 专属`NPUStructuredOutputsWorker310V2`不分配CUDA，grammar显式拒绝；运行时路径/设备待双审 | BLOCKED |
| T28 | 冻结`GPUModelRunner.__init__`→`DraftTokensHandler.__init__`→`torch.cuda.Stream/Event` | **Runner构造必经**；私有`torch`代理不影响该类自身模块globals | 专属`NPUDraftTokensHandler310V2`，保留无grammar全-1 DraftTokenIds回退；设备/异步边界待双审 | BLOCKED |
| T29 | 冻结`GPUModelRunner.sample_tokens`→`AsyncOutput.__init__`→`torch.cuda.Event/set_stream` | **每次生成token必经**；即使main_stream已改NPU | 专属`NPUAsyncOutput310V2`同步D2H快照、K7按count裁剪；无性能重叠承诺，需设备/调度/stream审查 | BLOCKED |

| T30 | `GPUModelRunner.execute_model` → hybrid `model_state.preprocess_state` → `preprocess_mamba_align_fused_kernel[grid]` → `MambaSpecDecodeGPUContext.run_fused_precopy` | **条件可达，尚非Target实测必经**：需实际`is_hybrid`→`MambaHybridModelState` + `mamba_cache_mode=align` + 正常Forward；R2计划的确配置align | 两条冻结上游Triton launch；需非Triton状态块迁移与CPU/设备/流正确性，候选在模型load前拒绝，不能无条件NO-OP | BLOCKED |
| T31 | `GPUModelRunner.sample_tokens` → 310P `postprocess_sampled` → hybrid `model_state.postprocess_state` → `_scatter_num_accepted_kernel[grid]` 与 `run_fused_postprocess_align` | **条件可达，现场未测**：具体Target须选中上游MambaHybridModelState且确有采样回写；align附加postcopy，需确认真实采样/请求重用分支 | 非Triton接受数、mask/reqslot、align后状态保存与K7拒绝语义；需310P跨流/短写/负例/异常证明 | BLOCKED |


## 实施准则（严禁投机捷径）

1. 优先**消灭T01～T09、T11、T14的真实调用并证明T15可用或替代**再讨论worker解除阻断。替代实现必须有相同Tensor shape/dtype/device、非因果/滑窗 semantics、负例/随机种子/请求混批和输出gold。
2. 只能在自己的310P任务分支接线，并限制影响作用域；单独负责其它Ascend设备的原有Triton路径。不允许全局覆盖 `vllm.triton_utils.HAS_TRITON`，也不允许临时覆盖全局 `torch.cuda` 而不证明真实路径。
3. 若 `T13` 仍UNRESOLVED，**不能**用“前11项都已改好”声称完成G0或静态GO。每发现一个路径追加ID/UT/实机止损点。
4. Graph路径仍独立`R3`；本表仅Eager，不因关闭Graph省掉DFlash输入Triton。
5. 可复用的310P历史代码：`_310p/block_table.py`、`_310p/spec_decode/dflash_proposer_310.py`、`_310p/attention/metadata_builder.py`、`_310p/ops/*`；这些来自旧V1 Runner使用环境，须通过V2 ABI/物理KV mapping逐项证据对齐。
6. CPU-only合同PASS≠设备可用；旧GateE/首写/507014与KVER11/SO装载身份另验。

## 追踪判定

当前 `BLOCKED`：11项（T01～T09、T11、T14）；`UNWIRED`：T12；`UNVERIFIED/P0`：T15；`UNVERIFIED`：T10；`UNRESOLVED`：T13。**R2-02仍OPEN，不得在机器上运行或解除V2 Runner fail-closed。**


## R2新增候选实现（2026-10-25；状态仍为BLOCKED/UNVERIFIED）

- T05/T06：`vllm_ascend/_310p/v2/block_tables.py:NPUBlockTables`提供独立非Triton Eager `gather_block_tables/compute_slot_mappings`，按组使用`kernel_block_sizes`，padding返回PAD=-1；`out/out_ptrs` 尚显式拒绝，`torch.searchsorted`在310P3设备上尚无支持证据。**UNWIRED，不能改READY**。
- T14：`staged_buffers.py`的`NPUStagedWriteTensor/NPUFusedStagedWriter`提供单/多组独立设备写候选；1D device-mutated state使用dirty-index `index_copy_` 而非整向量陈旧CPU镜像，矩阵采用32B对齐整行阻塞H2D。**未接入GPU父类BlockTables，NPU op可用性/异常恢复待审**。
- T15：`request_state.py:NPURequestState`避免UVA构造并保持0.27.1的属性/方法外观；但`GPUModelRunner.__init__`当前仍会先构造UVA父类，不能靠事后重置对象绕过。**P0仍在，不能解除Worker阻断**。
- 新审计项 **T16（P0）**：采样post_update设备修改`num_computed_tokens/total_len/all_token_ids`与CPU staging之间的所有权和可见性。候选1D dirty scatter避免覆盖其它请求；矩阵active-row重复写、请求重用、in-flight流/事件、NPU `index_copy_`/短写安全尚无闭环CPU+设备合同。**BLOCKED**。
- 纯CPU计划与Torch CPU合同见`tools/310p_mm_ar/test_v2_staged_buffers.py`、`test_v2_no_uva_adapters.py`；轻量CI不安装torch时Torch CPU测试跳过，必须另有明确的全量CPU Torch合同报告，不得把skip写PASS。


### R2-A 静态复审补丁：DFlash persistent slot / 事务失败止损（2026-10-25）

- **T06 新发现并修正候选 P0**：原`NPUBlockTables.compute_slot_mappings`返回`torch.stack`新对象，却没有写回`self.slot_mappings`。固定上游`DFlashSpeculator.propose`直接读取`block_tables.slot_mappings[gid]`，因此旧实现即使返回slot正确，DFlash仍读取全PAD。现改为**固定存储原位更新并返回同一data_ptr的切片**，支持positions仅含实际token、padded尾部=-1；CPU Torch测试覆盖别名、padding、空batch。设备Eager `searchsorted/index_copy_/copy_`支持及stream顺序**未验证**，T06仍BLOCKED。
- **T14 新发现并修正候选 P0**：原`NPUStagedWriteTensor.apply_write`在H2D前提交并清空CPU写计划；若中途失败会丢失写记录并留下半提交KV。现使用暂存提案→设备复制→CPU commit，任何设备异常标记poisoned并禁止继续；多组Fused串行任一组失败将全部组poisoned，要求Runner销毁受影响状态。**并非跨组原子提交**，R2-A仍NO-GO。
- **T15/T16 RequestState复用修复候选**：新增请求时先stage全行零化再stage新token前缀；对重用slot清除`next_prefill_tokens`；有未提交token行时拒绝释放。仍缺in-flight请求所有权、stream/event barrier、GPU设备采样写和CPU mirror协议；设备侧`index_fill_`待支持性核验，不能标READY。
- **CPU验证升级**：新增独立`r2-torch-cpu-contracts` CI job（PR专用，Python3.12/torch2.10.0 CPU），与不安装torch的轻量治理job区分。运行结果必须按实际head SHA报告；Torch测试若Skip/失败不能宣称完成CPU门槛。以上全部**未接入真实NPU Runner**。


### T02/T04 非Triton Eager候选（2026-10-25；UNWIRED，非READY）

- 新增`_310p/v2/input_ops.py`，提供与固定上游0.27.1同名同参数的`prepare_pos_seq_lens`（T02）和`expand_idx_mapping`（T04）CPU Torch可运行候选；采用设备端`searchsorted/index_select/where`，位置先以int32计算再cast int64，避免直接依赖310P int64 Add；不用`.item()/tolist()`强制host同步，不调用Triton。
- 新增`input_gold.py`和`test_v2_input_ops.py`，覆盖混合Prefill/Decode、K7 8 logits、padding、空batch、错误累计长度/容量和dtype；不安装torch的轻量CI仍执行gold，独立CPU Torch CI执行实际Tensor函数。
- **不能升级T02/T04状态**：现有`worker/v2/model_runner.py`仍直接从上游`gpu.input_batch`导入Triton函数，候选尚未路由；310P3`searchsorted/index_select`设备算子与非阻塞stream可见性未验证；`cu_num_logits/query_start_loc`动态GPU内容一致性尚未实现不引发host同步的运行时强校验。T01/T03/T07～09/T11/T13及真实Runner仍阻断。


### T07/T08采样候选与模块级绑定审查（2026-10-25，仍BLOCKED）

- `_310p/v2/sampling_ops.py`实现T07与上游0.27.1同签名、原地更新`num_sampled`；T08实现无Triton的`flatten_sampled`（`num_logits`为host已知形状，不从device提取scalar），并以`sampling_gold.py`和`test_v2_sampling_ops.py`的K7、chunked prefill、noncontiguous stride、随机gold覆盖。仅独立CPU Torch合同，非设备PASS。
- 上游`gpu/spec_decode/rejection_sampler.py`直接`from gpu.input_batch import get_num_sampled_and_rejected`，并在`_get_logprobs_tensors`直接启动`_flatten_sampled_kernel`。**不能**通过在310P包新增同名函数实现替换，必须在310P专属RejectionSampler/Runner实际调用路径绑定，并检查`rejection_sampler_utils.rejection_sample`是否还有其它Triton/设备限制。
- 上游T08仅在需要logprobs时可达，但不得据此在K7目标下删除；T07每次SpecVerify可达。输入累计长度、req-id与sampled<=logits必须先在CPU scheduler侧预校验，禁止为校验增加设备`.item()`隐式同步。310P Eager`searchsorted/index_select`支持与流可见性仍未静态审查完成；状态继续BLOCKED。


### T03 sampled/draft组合输入Eager候选（2026-10-25，仍BLOCKED）

- `_310p/v2/token_ops.py`保持冻结0.27.1 `combine_sampled_and_draft_tokens` ABI（bonus=0/1、原位`input_ids`、返回int64`logits_indices`），`token_gold.py`及`test_v2_token_ops.py`覆盖K7、prompt尾保护、chunked prefill、padding、混合request、随机CPU Torch差分。该候选使用整条Eager D2D写回，不是已核实的310P3小写安全方案。
- 实际`worker/v2/model_runner.py`仍从`gpu.input_batch`直接绑定Triton函数，尚未有310P专属Runner；需要静审设备`searchsorted/index_select/where/copy_`和stream/event、scheduler累计长度/req-id合法性与多group DFlash配合，T03维持BLOCKED。


### T01 prefill/lookahead no-Triton候选（2026-10-25，UNWIRED，仍BLOCKED）

- 固定上游0.27.1 `gpu.input_batch.prepare_prefill_inputs`七参数：仅 `num_computed_tokens < prefill_len` 的请求读取 `all_token_ids` 并更新 `input_ids`，`next_prefill_tokens` 按 **[lookahead, req_state]** 布局写入，超出prefill的lookahead填0；混合decode行及非活跃slot保持原值。
- 新增 `_310p/v2/input_ops.py:prepare_prefill_inputs` Torch Eager候选、`prefill_gold.py`纯stdlib oracle、`test_v2_prefill_ops.py`混合prefill/decode、越界负例、随机差分和Torch CPU合同；不使用Triton、UVA、设备标量`.item()`。其`index_copy_`可能更新值不变的非prefill列，仍需审查310P3写粒度、并发所有权和stream顺序。
- **不可标READY**：原Ascend V2仍绑定上游同名Triton函数；上游UVA构造链与BlockTables、采样、DFlash仍未替换。CPU scheduler须预校验req-state唯一、累计长度和`computed+query_len`容量；尚无310P3算子支持/动态stream证据，Worker NOT_IMPLEMENTED继续保留。

### R2-A 第二轮T13路径发现（2026-10-25；矩阵由31项扩展至40项）

- 通过`patch/worker/patch_v2/patch_triton.py`逐个追踪真实全局patch到`worker/v2/sample/{gumbel,penalties,min_p,bad_words,logprob}.py`、`structured_outputs.py`、`spec_decode/rejection_sampler_utils.py`，确认这些文件包含**实际执行的Triton launch**，而非仅定义未调用kernel。T17～T24明确登记并维持BLOCKED/UNVERIFIED；T25为冻结GPU V2 sample_tokens的K7窄写P0，不能被T11/T09 CPU合同掩盖。
- 特别记录：Ascend`_probabilistic_rejection_kernel`在温度>0时设置`u=0`并始终接受Draft，这不等于原始概率拒绝采样。需确定目标采样策略、给出分布正确性或禁用的静态证据；不能以“能输出token”作为正确性验收。
- `r2_static_admission.py`强制纳入T17～T25；未知额外矩阵ID将触发NO-GO而不被静默忽略。原31项历史CI结果仍是当时有效的**旧版本口径**，后续按40项新门禁统计。T13仍保持UNRESOLVED，直到完整可达路径审查闭环。
