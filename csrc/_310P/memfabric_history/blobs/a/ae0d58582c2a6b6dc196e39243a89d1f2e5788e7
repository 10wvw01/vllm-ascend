# R2-P0｜唯一 Target 的 Hybrid ModelState 完整静态设计与两轮审查（2026-10-25）

> **阶段裁决：DESIGN/STATIC-AUDIT COMPLETE，RUNTIME/DEVICE STATIC-NO-GO。** 目标仅为`Eco-Tech/Qwen3.6-35B-A3B-w8a8`、310P3单卡双Die/TP2/PP1、prompt4096/output2048、batch10、DFlash K7、Eager，性能期50请求；`mamba_cache_mode=align`为**R2启动计划值，非现场进程已观测值**。固定`vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` (v0.27.1)。Draft为保密自研模型，**NDA：未读取、未请求其配置、权重、训练或源码**。本轮**不新增运行时Kernel、不解除V2 Worker保护、不更改Stock、不实机、不合PR**。
>
> 前置证据：[已脱敏Target真实config的结构分类报告](R2_TARGET_ROUTE_PROBE_RESULT.md)、[31节点模型专属裁剪](R2_TARGET_REACHABILITY_CUT_20261025.md)。本稿是**冻结公开源码的语义/ABI设计**，并附带标准库纯CPU决策Oracle；**不是**“目标真实创建了MambaHybridModelState/发射了Triton”的设备取证。

## 一、调用与状态选择：确定事实 / 条件 / 绝对阻塞

```text
Qwen3_5MoeForConditionalGeneration (Target config 中 architectures)
  ├─ Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid)
  │    └─ get_mamba_state_{shape,dtype,copy_func}_from_config() [GDN]
  └─ ModelConfig.is_hybrid = _model_info.is_hybrid && layer_types != 全attention
       (冻结源码中该模型类和Qwen3VL基类均未定义get_model_state_cls；
        其它mixin/实际实例可能有动态属性，仍不能宣称100%无custom override)

[当前本仓NPUWorker310.init_device]: use_v2_model_runner? → NOT_IMPLEMENTED
     └─ 下述V2只存在于未注册的310P候选设计，并未执行
[NPUModelRunner310V2Candidate.load_model]:
  if is_hybrid && cache_mode==align → NotImplementedError (BEFORE model weights)
  else clone GPUModelRunner.load_model
           → model_loader.load_model → init_model_state
                1 custom model.get_model_state_cls()?
                2 CrossAttention modules?
                3 EncoderOnlyAttention modules?
                4 config.is_hybrid? MambaHybridModelState : DefaultModelState
           → load/create sampler, rejection sampler, structured output state
[NPUModelRunner310V2Candidate.execute_model, real nonempty batch]
  prepare_inputs → prepare_attn(BlockTables) → model_state.preprocess_state
  → model_state.prepare_attn(accepted-count, -1 no-spec sentinel)
  → Target GDN/FullAttention forward
[NPUModelRunner310V2Candidate.sample_tokens]
  K7 greedy/rejection → post_update(num_computed/sampled/rejected)
  → model_state.postprocess_state → DFlash.propose(multi-group) → K7 draft tokens
```

**源码定位**：`vllm/model_executor/models/qwen3_5.py:286-406,444-608,665-690`；`vllm/config/model.py:1764-1772`；`vllm/v1/worker/gpu/model_states/__init__.py:11-44`；`vllm/v1/worker/gpu/model_runner.py:300-367,1209-1336,1457-1600`；我方`_310p/v2/model_runner.py:282-309`与`_310p/worker_310p.py:32-48`。

**R2-P0只允许“Target-specific ModelState/metadata + 拷贝决策”**，**不允许**为了消除风险就把`is_hybrid`强行改False、把`align`偷偷改成`none`、忽略`postprocess_state`、使用UVA/TritonPlaceholder冒充NPU Kernel，也不能重写通用vLLM-GPU ModelState影响其它设备。先在310P专属受限候选中建立语义等价契约，等P1 ABI和CPU gold/静审完成才讨论设备批准。

## 二、Target GDN的缓存事实与数据结构

`Qwen3_5ForConditionalGeneration`有三个继承于其公开模型层的专属接口：
1. `get_mamba_state_dtype_from_config` → `gated_delta_net_state_dtype(model_dtype,mamba_cache_dtype,mamba_ssm_cache_dtype)`：conv dtype取cache/model规则，temporal dtype可被`mamba_ssm_cache_dtype`覆盖；计划声明float16不能替代真实启动验证。
2. `get_mamba_state_shape_from_config` → `gated_delta_net_state_shape(TP2,num_k_heads,num_v_heads,key_dim,value_dim,conv_kernel_dim,num_spec=7)`。conv每TP shard的特征维度为`(2*key_dim*num_k_heads + value_dim*num_v_heads)/2`；卷积历史长度`conv_kernel_dim - 1 + 7`；`VLLM_SSM_CONV_STATE_LAYOUT`决定`SD:(state_len,dim)`还是`DS:(dim,state_len)`。temporal state是`(value_heads/2,value_dim,key_dim)`。**这是源码公式，不是本地私有尺寸/显存分配实测**。
3. `get_mamba_state_copy_func` → `(get_conv_copy_spec,get_temporal_copy_spec)`。每层每状态类型读取`forward_context[attn_layer].kv_cache`、实际`stride(0)`、`element_size`、SD/DS轴、所属`MambaSpec` KV Group、请求对应的`int32`持久BlockTable。必须处理Mamba组与FullAttention组共存，不能用单KV BlockTable/假定物理blocksize相同。

**运行期所需持久状态（不是新增实现的宣称）**：
- `num_accepted_tokens[req_slot]:int32`：默认1，K7正常Verify回写1～8；不是“连续接收的draft数”而是当前步**生成/接受的token总数**（含首个不一致目标token或bonus）。
- `running_mamba_col[req_slot]:int32`：初始化`(num_computed_tokens-1)//mamba_block_size`；fresh req computed=0时是-1（未有可拷贝前态）。先从当前已存在的`mamba_spec.block_size`获得Block语义，不能随意拿Attention的blocksize混算。
- `precopy_src_col[req_slot]:int32`：旧运行块列；`precopy_bias[req_slot]:int32`：`max(previous_accepted - 1,0)`。新请求/reuse必须复位这几个状态；请求index映射是`batch_idx→持久req_slot`，**block-table行则按active batch次序**。
- `Mamba group KV`的每层 conv/temporal物理状态缓存：基址、group_id、block_table、stride、dtype、conv layout和物理块的生存期；Src+Dst地址不能根据逻辑列臆造，必须经过各自KV group物理block_id且保证范围与不别名。

## 三、T30：preprocess align 两段都必须保留

冻结`mamba_hybrid.py:160-207`调用**两次Triton**：`preprocess_mamba_align_fused_kernel`决策+更新，以及`MambaSpecDecodeGPUContext.run_fused_precopy`发`precopy_mamba_align_fused_kernel`。**它们不是一个`index_copy_`的简单替换。**

对真实非空批次的每个有效`batch_idx → req_slot`，在原版Kernel允许的条件下：
```text
old_col  = running_mamba_col[req_slot]
accepted = num_accepted_tokens[req_slot]                 (上一轮)
src_col  = old_col
bias     = max(accepted - 1, 0)
computed_after = num_computed_tokens[req_slot] + query_len[batch_idx]
new_col  = ceil(computed_after / B) - 1
crossed  = old_col >= 0 and old_col != new_col
running_mamba_col[req_slot] = new_col
if crossed: num_accepted_tokens[req_slot] = 1
if crossed: for every mamba KV group/state type, precopy(old_col, new_col, bias)
```

- 按冻结`mamba_utils.py:283-391`，实际precopy kernel对`src_col < 0`或`src_col == dst_col`跳过；新请求`old_col=-1`不应读取BT[-1]。**本稿的纯Python Oracle取更严格的“发生跨块才计划Copy”安全子集**，未来实现须独立校验此条件与模型状态初始化/异步调度，不可把Oracle当上线内核。
- **Conv SD**：`src[conv_token_offset=bias:] → dst[:conv_width-bias]`，每个历史位置覆盖完整dim。**Conv DS**：同样沿state_len轴切片，但按dim逐行非连续跨stride拷贝；不能用SD的整段D2D copy伪装等价。
- **Temporal**：`src physical_block = block_table[req,bias+src_col]`（不是src block内部字节偏移），复制全部**自然state数据长度**至`block_table[req,dst_col]`；物理页stride可包含padding，不得直接照搬页长字节。Src/Dst按具体Mamba KV Group解析。
- **顺序**：输入/BT staged writes → T30索引更新 → T30物理状态precopy完成且可见 → Target GDN forward。不能只看Eager就假定跨Stream写入自动可见；310P NPU Copy与MM/算子并行前需正确event/stream依赖。
- `preprocess`即使每个请求未跨块，原GPU仍每批调度该launch做fast-exit：将其“直接NO-OP”会漏掉未来跨块/接受偏移；是否在310P可用CPU已知值安全跳过，须以真实数据依赖而非乐观CPU镜像证明。

## 四、T31：采样接受数及postalign 保存也都不能删

`GPUModelRunner.postprocess_sampled`先让`post_update`更新真实`num_computed_tokens`，然后`model_state.postprocess_state`。冻结`MambaHybridModelState.postprocess_state:289-328`：
- `num_sampled`是Tensor时，`_scatter_num_accepted_kernel`按`idx_mapping`分散到持久ReqSlot，跳过`req_slot=-1`，写`max(count,1)`；**greedy K7也必须进行这个更新**。chunked prefill传入Python整数0时写neutral 1。
- 仅当align、`num_computed_tokens`不为空且Ctx已初始化时，再进入`run_fused_postprocess_align`，调用`postprocess_mamba_fused_kernel`（Triton）。若Ctx尚未创建不执行额外postcopy；但这不意味着整个ModelState可以不维护接受数。

冻结`mamba_utils.py:203-282`在`PRECOMPUTED_NEW_COMPUTED=True`下对每请求计算：
```text
a         = max(num_sampled, 1)
C         = num_computed_tokens[req_slot]   (已经执行 post_update)
running   = C - a + 1
aligned   = floor(C / B) * B
if aligned < running:  没有postcopy
else:
    bias       = aligned - running
    dst_col    = aligned / B - 1
    src_col    = running_mamba_col[req_slot]
    if src_col == dst_col:
        num_accepted_tokens[req_slot] = 1    # 每请求仅做一次
    if src_col != dst_col OR bias != 0:
        for every mamba KV group/state type, postcopy(src_col,dst_col,bias)
```

- **核心语义**：`postcopy`即使源/目标列相同，`bias>0`仍需要按GDN窗口移位，不能因`src==dst`一律跳过；当`src==dst && bias==0`才是拷贝NO-OP。与T30的跨块判断不等价。**在`src==dst && bias>0`场景，卷积状态窗口左移的源/目的区域可能重叠；310P实现必须使用memmove式安全语义或经过审计的临时缓冲，不能假定普通copy_/index_copy_具有重叠拷贝保障。**
- 后处理的`num_accepted_tokens`复位取决于`src==dst`，不同块时仍保留接受数；提前统一填1会破坏下一轮precopy的token偏移。
- 多个状态类型与层共享一个请求接受数：复制过程中每层Kernel可能看到同样的元数据，NPU候选应**一次/请求计算copy计划，再执行每组状态复制，最后按事务提交可见值**，不能在处理第一层后改变其它层读取的决策。
- Src/Dst为非法列、`bias<0`、`bias>=conv_width`、越过block table或请求slot复用与Copy in-flight相撞，都必须fail-closed/poison请求状态；不能用`clamp` silently产生错误token。

## 五、最小非Triton 310P候选架构（实施设计，不在本轮编写）

```text
NPUHybridModelState310Candidate(DefaultModelState语义受限继承/独立组合)
  ├─ state_factory: 仅在Target公开family+resolved hybrid+align，显式opt-in
  ├─ persistent request state: accepted:int32, running_col:int32, pre_src/pre_bias
  ├─ init_gdn_metadata: model.get_mamba_state_copy_func(), KV MambaSpec,
  │    forward_context state tensors, per-group BlockTables, frozen shape/dtype
  ├─ preprocess_state: 310P安全int32 per-request decision
  │    → GDN conv SD/DS + temporal batched *per-group* state precopy
  ├─ prepare_attn: 继承Hybrid GDN-specific accepted/draft metadata语义
  ├─ postprocess_state: 310P sampled-count scatter
  │    → GDN alignment decision → per-group state postcopy
  └─ add_request / remove_request / abort / preempt / reuse:
       slot reset + outstanding writes/Copy/Event ownership + error poison
```

**范围严控**：实现阶段优先用独立`_310p/v2/hybrid_state.py`或等价310P局部适配，不修改上游`vllm`与其它Ascend型号；只支持本Target TP2/PP1/K7/Eager。T30的`preprocess+precopy`、T31的`scatter+postcopy`是**四个最低等价语义块**，不需要为了实现它们先搬运通用Triton全部功能。

**版本与安全合同**：
1. 不改原`NPUWorker310.init_device`的`NOT_IMPLEMENTED`；进入设备前必须另签STATIC-GO。当前`Candidate.load_model`的早拒绝守卫保留。将来的状态工厂接线必须有新的本地备选与逐项覆盖证明，**不得直接删除保护**。
2. NPU Tensor contract：`int32 idx_mapping/num_computed/num_accepted`、slot/BT/device/type/stride、K7接受计数1..8、每组Mamba KV独立物理映射；避免GPU CUDA stream/Event/UVA和非对齐小写/32B写粒度。
3. 正确性合同：复制字节数用`state.numel/stride/element_size`推导并涵盖as_strided padding；SD/DS与temporal独立CPU gold，需对齐上游/旧V1已有状态复制语义；多Rank结果、request生命周期/abort/reuse、prefill→decode→spec混批与边界位置均要验证。
4. 不承诺性能：T30/T31只是让Target可能拥有正确V2状态通路的必要条件，离MM+MemFabric算子和整模型收益仍有多个独立工程门槛。

## 六、纯CPU决策Oracle与覆盖（非设备等价证明）

新增`tools/310p_mm_ar/r2_hybrid_align_reference.py`（零torch、零NPU）和`test_r2_hybrid_align_reference.py`：检查fresh`state_idx=-1`、边界跨越、same-block no-copy、K7接受偏移、`postalign`同块`bias>0`需要拷贝、`bias=0`不拷贝、跨块保持接受计数、SD/DS conv移位长度、temporal`src_col+bias`、非法源列/shape fail-closed。仅计划Copy语义而**不分配/复制任何真实GDN state tensor**，因此不能给真实Torch-NPU ABI/Stream标PASS。

机器合同：[R2_P0_HYBRID_STATE_CONTRACT_20261025.json](R2_P0_HYBRID_STATE_CONTRACT_20261025.json)。

## 七、两轮静态代码审查（本轮进行，保留所有阻塞）

### 第一轮｜冻结源码与数学语义审查（S1）

| 审查点 | 冻结源码依据 | 裁决 |
|---|---|---|
| ModelState真实选择 | `qwen3_5.py`、`config/model.py`、`gpu/model_states/__init__.py` | **CONDITIONAL**。公开模型家族与IsHybrid确定，本地模型对象未运行，无法给实例类型PASS |
| GDN状态形状/dtype/拷贝接口 | `qwen3_5.py`、`model_executor/layers/mamba/mamba_utils.py` | **SOURCE-DEFINED**。TP2和K7公式可直接复核，实际Local dtype/stride/groups未观测 |
| T30 pre/state advance & copy | `mamba_hybrid.py:160-207`、`mamba_utils.py:283-391` | **CONTRACT-DEFINED**。按旧状态/旧接受数决定copy，更新后Forward；禁用NO-OP |
| T31 accept scatter & postalign copy | `mamba_hybrid.py:289-328`、`mamba_utils.py:203-282` | **CONTRACT-DEFINED**。辨别同块bias>0与dest同块接受数复位 |
| GDN metadata | `mamba_hybrid.py:209-284` | **CONTRACT-DEFINED**。`num_decode_draft_tokens_cpu=-1`非spec sentinel；普通Prefill不能误作为Spec |
| 源码等价与异构布局 | `_copy_mamba_state_block`SD/DS/temporal，`MambaSpecDecodeGPUContext.initialize_from_forward_context` | **BLOCKED for device**。纯数学匹配，不证明310P支持物理拷贝/流 |

第一轮**静态设计通过**，但不构成runtime实现/STATIC-GO。若模型运行时自定义ModelState或Meta后端不同，这份设计须重新定位，不可强制打补丁适配。

### 第二轮｜现有310P候选整合与安全性审查（S2）

| 审查点 | 自有源码事实 | 结论 |
|---|---|---|
| V2绝对阻断 | `_310p/worker_310p.py:NPUWorker310.init_device` | **保持**，无设备执行 |
| Hybrid提前fail-closed | `_310p/v2/model_runner.py:load_model`有`is_hybrid && align`提前`NotImplementedError` | **保持**，先设计后替代 |
| 输入/KV组绑定 | 已有`NPUBlockTables`/StagedWrite与DFlash局部non-Triton输入准备 | **PARTIAL**，不适合直接拿旧V1 Proposer代替V2 ABI |
| K7核验/状态回写 | 当前候选greedy K7，`postprocess_sampled`调用ModelState更新 | **PARTIAL**，用户尚未冻结greedy；概率拒绝不可冒充 |
| 缓存/流/异常与回退 | NPU state index/pointer/stream/CANN实际ABI | **OPEN/P0**，未实现ModelState四块复制、无真实实验 |
| NDA | 全部仅公开类、抽象维度公式与固定枚举 | **PASS**，不需要Draft私有数据 |
| 范围隔离 | 本轮只新增`docs`与`tools`里的stdlib参考，不改`src`/Worker/Kernel | **PASS** |

第二轮结论：**设计具备“限定Target继续最小实施”的完整语义依据；真实state+NPU接口、生命周期与异步所有权未实现，因此R2-A继续STATIC-NO-GO。**

## 八、下一次真正要做的最小任务

- **工程 P0-E1**：对照冻结`MambaHybridModelState.prepare_attn`和310P现有GDN Attention metadata backend，固定每个公开可见`MambaSpec` KV group的ABI/int32 BlockTable/请求索引（与冻结源码一致），不读取自研Draft。
- **工程 P0-E2**：将四块T30/T31逻辑在**独立未注册候选**中实现安全NPU Eager路径，必须为conv SD/DS、temporal、任意group、block边界、K7计数写CPU Tensor gold / 故障注入 / request重用测试。实现前先核实所用`torch_npu`算子的支持和内存写粒度；不以CPU无异常冒充硬件可以跑。
- **工程 P0-E3**：独立审查**状态所有权与跨Stream**，双Rank TP2/模型量化/FORWARD/多KV group真实路由；完成独立两轮针对可执行实现的代码静态审查。仍须保留Worker阻断直至完整R2总门禁通过。
- **用户无需提供**自研模型细节。唯一可能影响P3采样工作的是正式请求的采样参数是否greedy，这与本轮Target ModelState静态合同无关。

**最终裁决：R2-P0 Target专属Hybrid ModelState的两轮静态设计/风险审计已完成；4段状态语义 + request/缓存生命周期被明确；真实Runtime/真实NPU适配仍未完成。R2-A 0/46 STATIC-NO-GO，R2设备/Graph NOT_RUN，PR#7 DRAFT，Stock默认。**
