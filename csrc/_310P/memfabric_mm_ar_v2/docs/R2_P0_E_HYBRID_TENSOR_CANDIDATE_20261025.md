# R2-P0-E｜唯一Target的GDN Hybrid align最小状态搬运候选（2026-10-25）

> **状态：CPU Tensor候选已编码，NPU/Runtime尚未实现/未准入。** 这不是GPU/Triton替代算子已经跑在310P3上的声明；它只是从[R2-P0静态设计](R2_P0_TARGET_HYBRID_MODELSTATE_DESIGN_20261025.md)前进到“能够实际在Torch CPU Tensor上搬运多组GDN状态”的**未注册独立执行组件**。目标唯一：公开Target Qwen3.6-35B-A3B-w8a8 / 冻结vLLM 0.27.1 / 310P3 TP2 PP1 / DFlash K7 / Eager。自研Draft NDA：不读私有代码或配置。Worker的`NOT_IMPLEMENTED`与候选Runner的Hybrid/align提前拒绝**完全保持**；Stock与算子不动，PR#7继续Draft。

## 1. 本轮实际编码文件和职责

- `vllm_ascend/_310p/v2/hybrid_state.py`：**未注册**`HybridAlignTensor310Candidate`（CPU Torch Tensor候选组件，**不是ModelState直接替代类**）。实现专有GDN组的物理BlockTable寻址、持久req_slot/num_accepted/running_col、前处理跨块预复制、K7回写与postalign复制、SD/DS Conv滑窗和Temporal跨列复制、临时快照处理同块重叠、全组先计划再写、异常poison、同步释放/复用。数据不依赖任何Draft私有结构。
- `tools/310p_mm_ar/test_v2_hybrid_tensor_state.py`：Torch CPU真实Tensor gold，覆盖2组Mamba、batch↔持久req_slot、K7接受数、prefill neutral=1、卷积SD/DS、temporal源列bias、同块重叠、故障注入、物理块冲突、非法BlockTable、reused req slot及NPU fail-closed。依赖Torch，`project-control`轻量无Torch CI可跳过，专门CPU Torch job不得跳过。
- 继续引用`tools/310p_mm_ar/r2_hybrid_align_reference.py`数学oracle（保留，不改）。本轮新增的组件才会真实拷贝CPU Tensor，不可以拿Oracle替代它。

## 2. 数据ABI与四段执行

```text
binding: tuple[GDNStateGroup310] (每Mamba KV Group独立group_id)
  group.block_table int32[active_batch, logical_columns] // 按batch row，不是req_slot
  group.layers[]:
    conv TorchTensor[physical_blocks,state_len,dim] (SD)
                      或 [physical_blocks,dim,state_len] (DS)
    temporal TorchTensor[physical_blocks, ...]       // 复制自然数据，不复制页padding
persistent: req_slot -> {running_col=(computed-1)//B, accepted=1}
preprocess_state(slots, num_computed_before, query_lens):
  preflight all slots+BlockTables → old_state_idx & old_accepted
  crossed? src=old_col, dest=ceil((computed+query_len)/B)-1, bias=old_accepted-1
  cross: snapshot copies for EVERY group/layer/conv+temporal
  all copies commit → store new_col; crossed resets accepted=1
postprocess_state(slots, sampled_counts, computed_after_post_update):
  accepted=max(num_sampled,1) (0仅chunk-prefill)
  running=computed_after-accepted+1; aligned=floor(computed_after/B)*B
  if aligned >= running:
    dest=aligned/B-1; src=running_col; bias=aligned-running
    unless(src==dest && bias==0): snapshot/commit per-group copies
    if src==dest: accepted=1
  commit all copies → scatter accepted into persistent req_slot
```

**物理寻址**：Conv直接取`BT[row,src_col]`及`BT[row,dst_col]`对应的物理块；当`bias>0`，SD沿state_len第一维截取`src[bias:]`写`dst[:width-bias]`，DS沿最后一维截取；Temporal取`BT[row,src_col+bias]`的整个实际状态元素，而不是`BT[row,src_col]`上的字节偏移。所有源片段先`.clone()`快照，才执行写回，确保`src==dst&&bias>0`的同块重叠拷贝正确，避免后续拷贝受到先写入污染。缓存dtype允许Conv与Temporal不同，不偷换ModelState的dtype约束。

**错误与所有权**：所有逻辑/物理BlockTable检查和所有源快照在任何写入前完成；两个活跃batch写到同一物理目的块时拒绝；写入中途异常必须poison，后续全部fail-closed不重试。CPU同步`remove_request`支持复用，但这不包含NPU Stream in-flight和异步buffer回收证明。

## 3. 静态审查一：冻结接口和公式逐项对照

| 检查 | 冻结公开源 | 本地组件状态 |
|---|---|---|
| `num_accepted`和slot索引 | `gpu/model_states/mamba_hybrid.py:63-105,289-341` | 持久req_slot，K7计数及chunked neutral已实现 |
| T30 advance/跨块复刻 | `gpu/mamba_hybrid.py:160-207`; `worker/mamba_utils.py:283-406` | src old, dest ceil()-1, bias accepted-1，已实现CPU Tensor |
| T31 postalign | `mamba_hybrid.py:289-329`; `mamba_utils.py:202-281` | running/align/dest/bias及same-block reset，已实现CPU Tensor |
| 双Conv布局 | `_copy_mamba_state_block`的`CONV_STATE_DIM_FIRST`分支 | SD/DS沿不同滑窗轴切片，clone避免同块覆盖 |
| Temporal与多Group | `_copy_mamba_state_block`按group_id和`BT[src_col+bias]` | 每组使用本组block_table，每层分别拷贝 |

**审查发现**：`src==dst&&bias>0`下Conv需memmove语义；禁止只有dest/ src不同才搬运；Temporal偏移是**block-table列偏移**，而不是字节偏移。当前Torch CPU版本用源快照严格处理。上游每group缓存元数据源自`forward_context`，本候选必须由经过审查的绑定方显式传入组/层张量（未接入Runner）。

## 4. 静态审查二：安全、性能、接口缺口

- **PASS，仅CPU候选**：不导入`vllm`、GPU/Triton/CUDA/UVA或私有Draft；不修改进程范围绑定；不会运行任何CANN/NPU；冻结TP2/K7并拒绝未审物理块访问；带poison的部分写入不会被原地重试。
- **BLOCKER，暂不准入**：`bind_groups`要求Torch CPU Tensor及int32 CPU BlockTable，显式拒绝NPU/其他设备；`postprocess_state/preprocess_state`接收Python整数序列和同步Tensor切片，与GPU device-owned状态、异步调度和`ModelState`的冻结vLLM ABI**不等价**。这不是一个可以直接在310P3调用的ModelState替代，不能为了试机器去删除CPU限制。
- **BLOCKER，设备基础**：310P3 NPU的小写粒度/32B安全、真实cache物理页stride与别名、`torch_npu` Conv/Temporal拷贝能力、多group状态地址和Stream依赖、完成事件、Rank一致性、abort/preempt/reuse、mixed-attn/GDN metadata和Draft proposal仍未验证。
- **性能**：Python per-request/per-layer快照是**正确性原型**，不应被当成生产性能实现；真实NPU版本应将动态偏移/slot/memcpy留设备端，避免逐req`.item()`同步与随机小写，并允许按已审条件合批或使用AscendC。但未获硬件ABI证据，不预先选择未验证算子。
- **工程门禁**：不解除`_310p/worker_310p.py`保护，不修改`_310p/v2/model_runner.py:load_model`的Hybrid早拒绝、不更改Serving默认，不合PR#7。原`0/46 R2-A STATIC-NO-GO`继续保留。

## 5. 下一次最小工作和准入要求

1. **R2-P0-E/NPU-ABI**：不接触自研Draft，从已有正式Host/NPU开发接口调查310P3支持的device int32 masked gather/scatter、跨页完整物理状态拷贝、Conv SD/DS不同stride、原地重叠安全copy以及32B/small-write限制。形成`allowlist`与流时序合同。
2. **R2-P0-E/NPU-Kernel**：把这个CPU Tensor gold作为**真正的逐项字节比对基准**，为T30/T31开发310P专属未注册设备候选，避免逐请求D2H；独立测试模型前向前后状态与K7接受数的正确性。验证异常不会部分写污染其他活跃请求。
3. **R2-P0-E/ModelState绑定**：核对上游`init_model_state`自定义工厂优先级、`prepare_attn`和request lifecycle；待模型专属NPU State ABI完整、CPU gold/两轮静审通过后方可针对候选更新早拒绝条件；**Worker总门禁仍需完整46条才可动**。
4. **公开披露**：不要求提供自研DFlash配置、逐层布局或私有实现；已知唯一Target+K7足以推进。

**本轮最终判定**：实际可执行的CPU Tensor最小组件与故障注入测试已编写，真实NPU四段搬运+ModelState绑定仍**NOT_IMPLEMENTED / NOT_RUN**，不得报告R2-P0-E“设备实现完成”。此文为设计到原型的工程证据，不是实机GO。
