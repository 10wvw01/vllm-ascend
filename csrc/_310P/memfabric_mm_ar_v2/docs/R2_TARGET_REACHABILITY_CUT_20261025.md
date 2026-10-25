# R2｜唯一Target模型 + 自研DFlash K7：冻结源码调用链裁剪（2026-10-25）

> **裁剪对象，不是放行判定**：`Eco-Tech/Qwen3.6-35B-A3B-w8a8`，310P3单卡双Die，TP=2、PP=1、input=4096、output=2048、并发10、DFlash K=7、Eager优先、50请求；冻结`vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`（tag v0.27.1）与自有`10wvw01/vllm-ascend` R2分支。自研Draft受保密条款保护：**不读取、不重建、不推断其私有逐层结构，不索取权重或私有源码**。证据基线为[已脱敏的R2实机只读静态探针](R2_TARGET_ROUTE_PROBE_RESULT.md)，不是runtime trace。
>
> **本次没有修改任何Kernel、Runner/Worker/采样实现，也没有运行设备。** 该裁剪只界定该业务应当考虑哪些分支，不把CPU合同、文本命中、配置推断冒充设备PASS。V2 Worker当前仍在init_device时直接`NOT_IMPLEMENTED`：就**当前可运行代码**而言，下表所有V2推理调用均不可在310P设备上到达；下表描述的是**若未来保持现有310P专属V2候选控制流并接线**时的静态可达性。

## 1. 真正与本Target关联的路径及先后顺序

```text
本地Target config.json ── [READ + Qwen3.5 MoE公开架构匹配；身份SHA UNVERIFIED]
                        └─ 冻结 Qwen3_5MoeForConditionalGeneration
                           └─ Qwen3_5ForConditionalGeneration / IsHybrid
                              └─ ModelConfig.is_hybrid:
                                 _model_info.is_hybrid + hf_config.layer_types 判定
310P NPUWorker310.init_device
  ├─ V1 → NPUModelRunner310（旧DFlash的输入准备已替换为AscendC）
  │          └─ 旧Runner无法支持历史Draft所需多KV group（R1-B NOGO）
  └─ V2 → 直接 NOT_IMPLEMENTED（当前绝对阻断，未运行）
       [以下为未注册的310P V2候选静态设计，而非已运行路径]
       NPUModelRunner310V2Candidate
        ├─ __init__ → AscendNPUModelRunner → 私有GPU父构造克隆
        │                ├─ NPURequestState/NPU DraftTokensHandler，不执行旧UVA构造
        │                └─ 本地 DFlash K7 NPUDFlashSpeculator310V2
        ├─ load_model
        │    ├─ **若 is_hybrid 且 mamba_cache_mode=align：直接拒绝**
        │    └─ 否则 GPUModelRunner.load_model（私有globals）
        │         └─ init_model_state 优先 get_model_state_cls
        │             → cross-/encoder-only → is_hybrid? MambaHybrid : Default
        │             → NPU Sampler + RejectionSampler + StructuredWorker构造
        ├─ initialize_kv_cache
        │    └─ 多group NPUBlockTables候选 + DFlash.set_attn ABI
        ├─ execute_model（非dummy、非空批）
        │    ├─ RequestState/BlockTables staged writes (T14-16)
        │    ├─ prepare_inputs / prepare_attn (T01-06)
        │    ├─ ModelState.preprocess_state（hybrid+align时T30）
        │    └─ Target GDN/Full Attention + W8A8/MoE Forward（设备未验证）
        └─ sample_tokens（PP=1走last-rank）
             ├─ normal sampler / K7 rejection verify (T07/T17/T23)
             ├─ postprocess_sampled → ModelState.postprocess_state (T09/T31)
             ├─ DFlashSpeculator.propose（按draft KV group输入准备T11）
             │     └─ scoped non-Triton prepare_dflash_inputs
             ├─ 安全保存 K7 draft tokens（T25）
             └─ AsyncOutput/NPU handler（T29）
```

**严格证据分级**：
- **已由服务器探针确认：** Target原始config的**公开架构字段**与`qwen3_5_moe / qwen3_5_moe_text / Qwen3_5MoeForConditionalGeneration`一致；`vllm v0.27.1`公开源码指纹存在；Draft未读。本地Target权重/量化工件身份**未核验**。
- **已由冻结源码确认：** `qwen3_5.py`中MoE类继承`Qwen3_5ForConditionalGeneration → IsHybrid`；`ModelConfig.is_hybrid`本身还通过注册元信息和layer_types计算；`init_model_state`优先自定义state工厂、再分流Hybrid。类声明**不是最终对象类型证明**。
- **必须区分**：本期Serve计划给`mamba_cache_mode=align`，但本次没有观察到Serve进程 flags；`Candidate.load_model`已对`is_hybrid && align` fail-closed。**不能为了让路由跑通擅自取消该保护，亦不能未经验证改为`none`规避状态搬运。**

## 2. T01～T31 重新分类：只针对冻结Target/候选，不扩展通用能力

具体逐项条件与证据字段落盘在机器可读的 [R2_TARGET_REACHABILITY_CUT_20261025.json](R2_TARGET_REACHABILITY_CUT_20261025.json)；下面按**候选路径**分组。表中的“核心”表示此阶段要保留业务合同，并不意味着原GPU Triton函数还会被调用，或310P候选已设备READY。

| 分类 | 节点 | 该Target静态裁决 |
|---|---|---|
| **19项：候选核心链，按阶段必需** | T01 T02 T03 T04 T05 T06 T07 T09 T11 T14 T15 T16 T17 T23 T25 T26 T27 T28 T29 | 保留；其中prefill、存在draft、采样或写KV各有自身阶段条件；已有私有310P替代候选**不等于已完成** |
| **7项：仅候选greedy禁用功能可绕开** | T08 T18 T19 T20 T21 T22 T24 | **有执行前显式拒绝前提**才暂不移植；原始用户需求未冻结temperature/top_p/logprobs等采样参数，**不得当成长期不需要** |
| **1项：由PP=1静态裁掉的分支** | T10 | 已定位的`postprocess_num_computed_tokens`唯一非最后PP rank入口在PP=1不走；新增其它调用必须重新判 |
| **1项：候选中被方法覆盖的旧定义** | T12 | 私有`postprocess_sampled`直接走`_310p/v2/post_update_ops.py`，旧Ascend `input_batch.post_update`不能再作为本候选必经独立开发项 |
| **1项：继承路径总审计未闭环** | T13 | 不可按“只剩一个kernel”计量；覆盖Target模型state、GDN、MM输入、缓存、Stream、W8A8/TP2的隐藏调用 |
| **2项：与目标Hybrid/align强相关的条件阻断** | T30 T31 | 不能排除；当前候选首先被load_model守卫挡住；静态确认上游GPU ModelState含真实Triton launches，尚无310P专属state实例/设备证据 |

**注意：** T01的prefill阶段一定出现于该输入场景，但可能分片；T04/T11/T23/T25仅在已经进入draft提案/验证的批次必经，不能写成“所有请求每一步都是K7”。T27/T26/T28即便禁掉grammar、penalty、随机采样仍需构造状态对象，**不能因功能关闭连构造UVA/CUDA风险一起删掉**。T29即使Eager亦会创建异步输出对象，已有310P同步快照候选但没有device审查。

### T08真实降级理由

冻结`gpu/spec_decode/rejection_sampler.py:94-112`的`_get_logprobs_tensors`在`max_num_logprobs==NO_LOGPROBS`**直接返回**，不会启动`_flatten_sampled_kernel`。我方`_310p/v2/sampler.py:NPURejectionSampler310V2._get_logprobs_tensors`也在这个分支返回，否则拒绝。T08**不是SpecVerify每轮必经**；其风险与T22 logprobs一并变成条件功能。T23拒绝验证本体则**仍必经**，不能混淆。

### T17/T18/T23拆分，禁止“禁随机就无需采样”

冻结`GPU Sampler.sample`即便greedy也通过`gumbel_sample`入口，310P候选`gumbel_ops.gumbel_sample`以`temperature==0`断言后采用argmax，因此**T17采样调用入口仍必须有正确替代**。T18温度缩放可以在已验证greedy条件下不走。T23 DFlash Verify仍需对K7正确做接受/拒绝，`rejection_ops.rejection_sample`现仅支持greedy且拒绝温度>0；绝不允许把已有Ascend非greedy `u=0`恒接受当成概率正确实现。

### T30/T31 不应按“此Target已运行Triton”宣传

- 冻结`gpu/model_states/__init__.py:init_model_state`先查`model.get_model_state_cls()`，随后cross-attention/encoder-only，才是`model_config.is_hybrid`路由。静态Hybrid声明高度关联，但**没有实际对象身份**。
- 冻结`gpu/model_states/mamba_hybrid.py`：`preprocess_state`仅在align且非空真实batch调用`preprocess_mamba_align_fused_kernel[grid]`与`ctx.run_fused_precopy`；`postprocess_state`在tensor `num_sampled`的非空请求使用`_scatter_num_accepted_kernel[grid]`，align条件额外触发`run_fused_postprocess_align`。
- 当前`_310p/v2/model_runner.py:load_model`先判断`is_hybrid && align`并抛`NotImplementedError`；而当前Worker在设备初始化前就阻断V2。因此应判：**确定的源代码条件风险 + 确定的当前阻断，不是观察到Triton launch**。
- 修复若必要，最小边界是 **TP2 K7 下GDN缓存读写、接受数更新、block边界迁移、请求释放/复用、Stream可见性及CPU gold**；先复用310P旧GDN语义，不能偷换`align`/禁用计数算子，不开发全模型混合状态框架。

## 3. 经过裁剪的最小剩余实施面，而不是“重新移植46项”

| 优先 | 只做哪些事 | 静态代码证据/工作结果 | 什么不能当成完成 |
|---|---|---|---|
| **P0-A** | 查清`ModelConfig.is_hybrid`、注册模型工厂/自定义`get_model_state_cls`优先级、Target GDN cache contract；固定候选`align`条件 | 冻结Qwen类 + `gpu/model_states/__init__.py` + 当前`load_model`守卫 | 类上`IsHybrid`就宣称实际`MambaHybridModelState`已实例化 |
| **P0-B** | 仅针对Target的`preprocess_state/postprocess_state` T30/T31列状态读写与非Triton替代语义，不着急编码 | 接受数、GDN块迁移、前后步同步、非空batch/K7 | 用`none`或直接删除pre/post绕过；CPU单点测试替代完整KV合同 |
| **P1** | 已有NPU候选的构造/状态/KV链最小接线审计：T15→T14/05/06→T11，保持多KV group支持 | `_310p/v2/{model_runner,block_tables,staged_buffers,dflash_speculator}.py` | T11局部覆盖存在就声称Draft已跑；碰自研源码、读取私有配置 |
| **P2** | 仅已锁定的输入/采样/Async链：T01~04/T07/T09/T17/T23/T25~29 与T16 RequestState | frozen GPU callsites + 310P私有局部绑定 | 因greedy-only候选通过就隐式变更原业务的采样行为 |
| **P3** | 以现有请求配置固定是否需要T08/T18~22/T24；不能确认时保持guard与条件账 | `sampling_scope.py:validate_greedy_310p_sampling_params` | 默认用户同意greedy/无logprobs |
| **另行归档** | Graph、DP/EP、LoRA、MM多模态输入、PP>1、任意其它模型 | `R3`或未来单独scope | 偷渡成本期运行时准入 |

**额外Target专属风险**：`Qwen3_5MoeForConditionalGeneration`是注册多模态处理器的类，**“本期压测只有文本”不等于构造链上所有MM/EncoderCache自动消失**；但也不等于会实际跑视觉编码器。冻结`GPUModelRunner.__init__`对`supports_mm_inputs`有条件创建`EncoderCache`；须列入T13后续ABI审阅。历史Target双Rank加载不证明当前V2量化kernel、GDN backend和TP2 Token/AllReduce已设备PASS。

## 4. 决策门禁与本轮交付

1. **冻结硬约束**：310P3 TP2 PP1、同一Target/Draft、DFlash K7、input4096/output2048/batch10、Eager、Stock对照；50请求用于性能阶段；不读取私有Draft数据。本轮没有冻结用户尚未给出的温度/采样参数。
2. **真实性**：当前310P V2设备路径全因Worker阻断而**NOT_RUN**；当前greedy候选仅为可能选用的严格子集；`T30/T31`源代码不能从“潜在”升级到“设备必经”。
3. **静态风险账不删**：原T01～T31 + ABI01～15合计46项照旧保留，`R2-A STATIC-NO-GO:0/46`，本文的**分类不等于更改准入分母或宣称19项已完成**。
4. **先裁剪再实施**：当前不新增任何通用Triton实现、不修改NPU Worker保护、不合入PR#7、不做设备推理、不改`mamba_cache_mode`。未来仅在证实该模型实际需要且替代合同成熟时开始最小适配编码。

### 经核对的源码（两种证据应分别引用）

- **已实际在310P执行的脱敏只读核查**：[R2_TARGET_ROUTE_PROBE_RESULT.md](R2_TARGET_ROUTE_PROBE_RESULT.md)；现场`target_config=READ`但`effective_runtime_runner=NOT_OBSERVED`。
- **冻结vLLM公开源码**：`vllm/model_executor/models/qwen3_5.py:286-430,660-670`；`vllm/config/model.py:1764-1772`；`vllm/v1/worker/gpu/model_states/__init__.py:11-44`；`mamba_hybrid.py:160-202,289-333`；`vllm/v1/worker/gpu/model_runner.py:300-400,1209-1345,1457-1605`；`gpu/spec_decode/dflash/speculator.py:303-414`；`gpu/spec_decode/rejection_sampler.py:94-113`。
- **自有310P候选**：`vllm_ascend/_310p/v2/model_runner.py:71-155,207-300,327-521`；`vllm_ascend/_310p/v2/sampler.py`；`sampling_scope.py`；`rejection_ops.py`；`dflash_speculator.py`；`block_tables.py`；`worker_310p.py:32-50`。
- **验证约束**：本轮交付`tools/310p_mm_ar/test_r2_target_reachability.py`执行机器裁剪账的完备性/一致性检查，非运行时动态trace。

**最终：完成唯一Target的源码级可达性裁剪；需保留的19项候选阶段核心节点 + 2项Hybrid条件硬风险 + 1项继承审计。只在显式greedy子集下可暂绕开7项功能节点，另有T10一条PP1静态排除与T12一条候选覆盖。无新增算子、无设备运行、无正式准入。**

## R2-P0 GDN/Hybrid专属审计补充（2026-10-25）

[Hybrid ModelState四段状态语义与两轮静态审查](R2_P0_TARGET_HYBRID_MODELSTATE_DESIGN_20261025.md)已经完成，明确对应旧模型专属分类T30/T31：`preprocess index+precopy`、`accepted count scatter+postalign copy`，其中conv SD/DS与temporal不能混用复制策略，`src==dst && bias>0`必须复制。本项只生成设计、零NPU数学参考及测试；**未编写真实NPU实现，不能将原分类T30/T31升为设备READY/STATIC-GO**。用户自研Draft数据仍未读取。
