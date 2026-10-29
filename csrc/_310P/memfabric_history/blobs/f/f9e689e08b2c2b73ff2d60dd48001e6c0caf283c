# R2-P0-E｜Device-Native GDN Copy Consumer 与 ModelState 调用链关键连接（2026-10-26）

> **裁决：CPU Tensor 完整链路候选实现完成；310P3真实NPU COPY与vLLM注册仍NO-GO。** 锁定Target `Eco-Tech/Qwen3.6-35B-A3B-w8a8`、310P3单卡双Die TP2/PP1、prompt4096/output2048/batch10、DFlash K7、Eager，以及 `vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`。全部基于公开Target模型/冻结上游接口，**没有读取或重建自研Draft保密数据**。未触碰MM+MemFabric正式Kernel、310P Worker和运行时Runner。PR#7仍Draft。

## 一、关键连接这次真正实现了什么

以前的两端彼此不兼容：

```text
[已有] GDNDeviceMetadata310Candidate
  T30-A/T31-A: src_col[N], dst_col[N], bias[N], needs_copy[N], validity[N]
  均为torch.Tensor(device)，拒绝丢失实际device-owned采样状态

[旧复制器] GDNWholeBlockCopy310Candidate
  execute(list[GDNMove310]) // 每个字段都是CPU int，需要Host镜像
```

现在**另外新增**未注册的 `vllm_ascend/_310p/v2/hybrid_state_device_copy.py`：

```text
device Tensor决策 (int32/bool, batch N)
      ↓ 直接使用Torch Tensor，绝不逐req .item()转Host
GDNIndexedGroup310:
  group_id
  block_table: int32 Tensor(device)[active_batch,logical_cols]
  layers: {conv,temporal,layout SD|DS}×真实Mamba KV Group
      ↓
tensor BT[batch_row,src_col], BT[batch_row,dst_col],
       BT[batch_row,src_col+bias]  (temporal)
      ↓
整批物理索引预检/重复目标块检查 + 所有group所有layer的源快照
      ↓
Conv: gather移动窗口，保留目标尾部，按物理块整行index_copy_
Temporal: source physical block整行index_copy_
      ↓
T30-B/T31-B复制完成后，才允许后续GDN读/ModelState状态推进
```

CPU Torch模式已经能用**设备形态的Tensor决策+设备形态的int32 BlockTable**走到Conv/Temporal物理缓存，而不用之前Host `GDNMove310`转换。SD与DS按照各自状态窗口轴复制，Temporal的源列明确是`src_col+bias`。跨KV Group原始状态快照在**任何写入之前**完成，避免同块重叠或group写入顺序污染。全部请求先验证block-table范围和目的物理块唯一性，预检失败不能产生部分写；中途写失败则poison整个Consumer，禁止重试。

**重要设备边界：这条连接在NPU分支尚未准入。** 在真实NPU上，动态block id越界、设备侧mask压缩、并发目的块冲突、Tensor `index_copy_`的32B写粒度和完成Event没有证据。Consumer对 `device.type=='npu'` 执行请求**在任何物理写之前抛NotImplementedError**，而不是偷偷 `.cpu()/.item()`做同步。如果认为“所有路径都已打通”，这会是错误的：当前可执行的完整闭环仍然仅是**CPU模拟物理GDN状态缓存**。

## 二、未注册 ModelState 顺序适配：与真实ModelState的关系

新增 `vllm_ascend/_310p/v2/hybrid_state_modelstate_link.py`，内部依赖注入两个上述候选，不导入`vllm`；**仅允许CPU，拒绝将其冒充可注册NPU ModelState。**

接口顺序按冻结 `MambaHybridModelState` 与 `GPUModelRunner` 签名对齐（但`prepare_attn`**仅返回GDN部分payload，不替换真实`build_attn_metadata`**）：

| 函数 | 已落地的CPU Tensor顺序 | 明确保留的真实接口缺口 |
|---|---|---|
| `add_request(req_index,new_req_data)` | seed请求槽、初始化accept=1、running列 | 真实请求所有权、stream inflight |
| `preprocess_state(input_batch,block_tables,kv_cache_config,num_computed_tokens)` | T30-A device Tensor决策 → DeviceIndexed Copy T30-B → CPU ack | NPU validity gate/全部KV group真实地址/完成event |
| `prepare_attn(input_batch,cudagraph_mode,block_tables,slot_mappings,attn_groups,kv_cache_config,...)` | 验证copy后GDN接受数及非投机行-1哨兵 | 真实Attention backend build_attn_metadata |
| `forward_completed_for_cpu_gold()` | 显式标记前向已结束，不触模型 | 正式Forward/GDN kernel和stream事件 |
| `postprocess_state(idx_mapping,num_sampled,num_computed_tokens)` | T31-A：接受数+computed-after-post-update → T31-B DeviceIndexed Copy → CPU ack | 真实采样T09依赖、device validity/copy/Event |
| `remove_request(req_index)` | 空闲时请求槽回收/复位 | 设备在途copy与async owner/abort/preempt/reuse |

冻结真实Runner的调用顺序为：`prepare_inputs → prepare_attn(物理KV) → model_state.preprocess_state → model_state.prepare_attn → forward`；T09在采样后先`post_update`再`model_state.postprocess_state`。本期适配器只保持可审计的接口顺序，**未改Runner、不初始化/加载实际Target模型、不构造真实Attention backend**。

## 三、CPU Gold与负例

测试 `tools/310p_mm_ar/test_v2_hybrid_device_copy_link.py`：

- 完整两轮执行：普通decode回写接受数4；下一轮跨块，T30-B使用上一轮bias=3搬运2个KV Group；然后T31-B按后验accepted计数执行postalign，并检查Conv SD/Temporal的完整真实Tensor内容；
- SD/DS两种Conv窗口，双batch行且请求slot与batch row不相同；独立设备int32 BlockTable真实定位；
- fresh请求old_col=-1无需拷贝，chunked prefill num_sampled=0映射neutral1；
- 两请求物理目标块冲突 → **所有物理缓存保持未写**、poison；第2 KV Group块表错误 → 第1组无写；中途Copy故障 → 全Owner fail-closed；
- BlockTable必须为ModelState与Consumer共有的权威Tensor（不会在Runner偷偷接另一份Host镜像）；前向未结束、Copy未完成、slot释放/复用错误均会拒绝；
- 锁定Worker `NOT_IMPLEMENTED`、Hybrid `load_model()`提前拒绝、未注册类与CPU/NPU门禁，防止后续把CPU Gold充当NPU实机证据。

本轮新增的是一个真实CPU-Torch **端到端状态复制合同**，不仅是看文件是否包含字符串。数学Oracle和历史纯CPU状态搬运测试继续保留；与正式DFlash/模型推理的端到端性能没有关系。

## 四、两轮静态审查与未解决阻断

### S1：数据读写与冻结语义
1. 源列/目标列/bias直接由Device Tensor进入device BT索引，**没有 per-request D2H；**复用冻结 T30/T31的`ceil/floor`边界和post-update接受数公式。
2. Conv SD/DS的状态窗口左移保留目标尾部；Temporal按照`src_col+bias`取源物理块；不同Mamba KV group独立绑定状态、各自block-table。
3. 所有待写的物理源状态提前快照，之后才写目的整块；动态错误/重复目标物理块必须在首次写之前被拒绝。
4. `num_sampled`可以是实际Device Tensor；但本期CPU模拟使用合成counts，**不能宣称真实RejectionSampler已接通**。

### S2：接口/安全/性能/实机准入
1. NPU Consumer **还没有**经310P3审核的device validity/collision error gate，或支持mask的定长物理块复制ABI。因此`execute`在NPU分支**无条件拒绝**，不可删除该拒绝只为测通。
2. Torch CPU使用`torch.nonzero`动态active compaction、`bool(torch.all())`与`torch.unique`做数值错误门禁，这些逻辑**只适用于CPU Gold**；不能简单把device切到NPU后运行（否则可能发生host sync或不确定行为）。下一阶段应开发device-side fail-closed +定长mask/物理copy实现，优先考虑AscendC受限专有kernel或有明确310P3证据的Torch-NPU算子。
3. 32B物理state行在Python形状层校验；真正的NPU小写粒度、GDN缓存Stride、Mamba group物理block大小、`index_copy_`底层原子/数据流、重叠写正确性没有设备证据。
4. `TargetGDNModelStateSequence310Candidate`不是`DefaultModelState`子类，也不提供真实`build_attn_metadata`；只做CPU序列核验。上线还需目标专属`ModelState`注册路径、当前设备`prepare_attn`与RequestState连接，完整46项准入证据。
5. 未改Runtime/Worker/Host/AscendC kernel、Stock默认、Graph、私有Draft，PR #7继续Draft。报告的“已连接”只限未注册CPU Gold。

### 后续唯一高优先级
**R2-P0-E/NPU-VALIDITY-COPY**：在同一目标下，将device tensor有效性、按组block id、目的冲突检测和带mask整块copy封装成能在310P3上fail-closed的设备端操作；必要时以独立AscendC算子实现，然后审查Stream/Event/ReqSlot生命周期。形成独立STATIC-GO后再讨论调度器/真实ModelState接线，绝不先删除双保护。

**本轮结论：关键连接的CPU Tensor真实拷贝与未注册ModelState顺序合同已实现；正式310P3 NPU Copy和真实ModelState注册仍NO-GO / NOT_RUN，R2-A STATIC-NO-GO 0/46。**

## 五、第二轮审查追加回归与机器门禁

- 新增`tools/310p_mm_ar/test_r2_p0_e_device_copy_static.py`的**零Torch静态保护**：冻结当前NPU execute拒绝位于任何Copy操作之前；Runner/Worker不得意外导入未注册Consumer/Facade；旧Hybrid align早拒绝仍在；完全排除逐请求Host转标量。
- 追加CPU Torch实测合同：同一物理KV组两个batch分别从物理块0/2交换写入，必须对**全部源块先快照**后提交，避免跨请求读取已覆盖的旧数据；device目标列越界和validity=false必须在全组首次写入前拒绝并中毒。
- 这些机器门禁只提高**CPU金标与静态审计**强度，**不改变310P3 NPU拒绝执行策略、真实ModelState未注册及R2-A 0/46**。
