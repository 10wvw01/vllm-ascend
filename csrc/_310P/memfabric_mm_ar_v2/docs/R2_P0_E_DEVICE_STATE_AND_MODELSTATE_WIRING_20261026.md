# R2-P0-E｜Target GDN 的 Device-Owned T30-A/T31-A 与 ModelState 接线审计（2026-10-26）

> **范围与裁决**：冻结上游 `vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，唯一公开Target `Eco-Tech/Qwen3.6-35B-A3B-w8a8`，310P3双Die、TP=2、PP=1、DFlash K=7、Eager。自研Draft私有资料未读。**新增device-owned Tensor计算候选与静态ABI审计；没有完整 ModelState、无真实310P3执行、没有解除Worker/Hybrid早拒绝；R2-A STATIC-NO-GO 0/46。**

## 1. 冻结调用顺序（必须保持原语义）

```text
310P V2 Worker            NOT_IMPLEMENTED (当前运行时绝对阻断)
NPUModelRunner310V2Candidate.load_model
  is_hybrid && align       NotImplementedError (第二道阻断)
若将来有独立准入实现:
  GPUModelRunner.load_model（被局部clone）
    model_loader.load_model -> init_model_state()
      (1)get_model_state_cls? (2)cross (3)encoder-only (4)is_hybrid?
  execute_model(仅真实非空批):
    prepare_inputs → self.prepare_attn: 得到每KV Group block_tables/slot_mappings
      → model_state.preprocess_state(input_batch,block_tables,kv_config,
                                   req_states.num_computed_tokens.gpu)
         T30-A(device state_idx/accepted)→T30-B(跨block GDN复制)
      → model_state.prepare_attn(...): GDN accepted/num_decode_draft_tokens元信息
      → Target forward
  postprocess_sampled:
    post_update（含num_computed_tokens.gpu的实际更新）
      → model_state.postprocess_state(idx_mapping,num_sampled,
                                     req_states.num_computed_tokens.gpu)
         T31-A(接受数scatter)→T31-B(postalign状态保存)
```

**实际代码位置**：冻结上游 `vllm/v1/worker/gpu/model_runner.py:300-365,1177-1208,1230-1328`；`gpu/model_states/__init__.py:init_model_state`；`gpu/model_states/mamba_hybrid.py:63-105,160-282,289-343`；本仓 `vllm_ascend/_310p/v2/model_runner.py:153-180,248-305,470-493`。当前静态Target公开IsHybrid与计划`mamba_cache_mode=align`不等于已经观测真实ModelState实例或设备Kernel。

## 2. 本轮实际实现：设备端决策，而非Host镜像假决策

新增 `vllm_ascend/_310p/v2/hybrid_state_device_meta.py`（**未注册**）：

- `GDNDeviceMetadata310Candidate` 仅TP2/K7、明确device和NPU诊断Stream；持久 `int32[ceil(max_reqs/8),8,4]`，字段 `[running_col,accepted,pre_src,pre_bias]`。**一行128B全写**，避免单个4B槽位窄写；同一物理行多个req_slot的更新先合并再一次`index_copy_`。
- `seed_requests/release_requests`：用权威CPU调度元数据初始化/回收；必须IDLE、没有未完成NPU元数据操作；新的req_slot neutral `[-1,1,-1,0]`。
- `preprocess_device(slots,idx_mapping_gpu,num_computed_gpu,query_start_loc_gpu)`：**动态数值只读设备Tensor**，按Batch实际Query长度算`ceil((computed+query_len)/B)-1`、old accepted bias、cross-boundary需Copy标志；同时把新running和接受数复位值提交整128B行。返回`GDNDeviceDecision310`（包含源/目标列、偏移、copy mask、动态valid mask），不调用`.item()`、不把设备计数搬到CPU。
- `prepare_attn_payload`：严格在precopy完成后才允许，接受数从设备持久元数据gather；仅对Scheduler**权威CPU**的`num_scheduled_tokens/num_draft_tokens_per_req`生成公开Frozen GDN规范的`-1`非投机行标记（仅当d>0且scheduled=d+1才写d）。
- `postprocess_device(slots,idx_mapping_gpu,num_sampled_gpu,num_computed_after_post_update_gpu)`：接收**真实device-owned sampled与post_update后computed**；计算`running=C-max(sampled,1)+1`、`aligned=floor(C/B)*B`、`dst=aligned/B-1`、`bias=aligned-running`、copy mask与同块重置接受数。整行提交，不用CPU同步取每请求标量。
- **显式阶段门禁**：`PRECOPY_DECISION_PENDING`不得直接执行`prepare_attn`；`POSTCOPY_DECISION_PENDING`不得释放/复用请求。**只有CPU Gold有专用ack接口**；NPU缺少设备端搬运消费者及完成Event时永远不可越过这些门禁。NPU enqueue后需明确`confirm_completed`同步，**同步仅确认本模块metadata写完成，不代表GDN物理状态copy完成**。
- `validity` 是设备Tensor。`idx_mapping`错位、负query、非法接受数和无效跨块参数不能仅凭Python shape证明有效。CPU合同执行torch.all数值校验；**NPU侧不允许隐藏D2H，而必须后续新增Device异常标志/失败门禁的消费路径**。在该路径具备前仍不能放行下一步model forward。

这一实现**没有声称**使用NPU kernel直接替代冻结的`preprocess_mamba_align_fused_kernel`，仅有可审计的Torch张量算子实现候选；需要真实Torch-NPU 310P3支持证据。

## 3. ModelState ABI对照与接线裁决

| 必经入口/所有者 | 冻结ABI / 实际数据 | 本轮状态 | 未完成原因 |
|---|---|---|---|
| `init_model_state` | 模型自定义工厂优先，随后cross/encoder/is_hybrid | **静态已审** | 没有注册Target专属`ModelState`，不可绕过早拒绝 |
| `add_request` | 请求slot + computed seed、接受数1 | **独立候选CPU合同** | 实际Runner请求映射/动态生命周期未绑定 |
| `preprocess_state` T30-A | device idx_mapping、computed、query_start_loc | **新设备计算代码已写** | Runtime语义与Stream、动态异常门禁未验证 |
| `preprocess_state` T30-B | 对Mamba组Conv/Temporal跨块复制 | **已有独立Torch-NPU整块复制候选** | 现有`GDNWholeBlockCopy310Candidate.execute`仍接收CPU `GDNMove310`；不能把设备col/bias强制D2H |
| `prepare_attn` | GDN num_accepted Tensor + 非Spec行-1 + build_attn_metadata | **仅完成独立payload计算** | 未提供真实 `MambaHybridAttnMetadata` / GDN backend绑定 |
| `postprocess_state` T31-A | device num_sampled + post_update后device computed | **新设备计算/元信息写已实现** | 与RejectionSampler真实权威事件顺序未验证 |
| `postprocess_state` T31-B | device决策、Mamba组物理BT与重叠状态复制 | **返回设备决策、物理copy未接通** | 需要device-native物理寻址、去重/poison、完成事件 |
| `remove_request` | free/abort/preempt/reuse | **静态门禁** | Device copy pending +跨stream/多rank所有权缺失 |

**绝不采用的捷径**：去掉`load_model()`早拒绝、把align改none、隐式`.item()/tolist()/cpu()`把设备状态转成Host `GDNMove`、或在`prepare_attn`之前伪造“copy完成”。

## 4. 两轮静态审查

### 第一轮：Frozen调用点和数值语义
- Upstream执行次序为preprocess后收集attention元数据；请求数存在`idx_mapping`(device)与`idx_mapping_np`(CPU)，必须要求两者对应，否则请求错位。
- Frozen T30会携带old accepted的偏移，跨边界reset=1；T31使用`post_update`后的computed，same-block与bias非零的copy不能省；`num_decode_draft_tokens_cpu=-1`标志不是0。
- 同批量的`num_sampled`是Tensor，**不可将Greedy K7简化为固定接受数，也不能用Host乐观镜像覆盖设备真实结果**。

### 第二轮：Torch-NPU ABI、异步与失败安全
- 元信息整行物理更新是128B，CPU Torch可算，但Torch-NPU`torch.div/where/stack/index_select/index_copy_`在310P3的真实支持、小写入与吞吐**未验证**。
- 设备校验`validity`目前未由任何真实NPU消费者执行失败中止，因此生产链必须停在前处理决策阶段，绝不算STATIC-GO。
- T30/T31的设备决策不能交给已有**Host标量型** GDNMove copy：这里是新的真实ABI缺口，必须开发device native group-BT gather/copy+错误标记与Event/fence，或在独立诊断中明确测量受限D2H成本而不能伪装无同步生产方案。
- 双Rank TP2的GDN物理group/block配置、模型实际ModelState实例、真实GDN backend/quant数据路径仍需现场验证；CPU通过不等于硬件等价。
- `NPUWorker310.init_device`和`Candidate.load_model`的双重保护未改；没有设备或模型操作，没有读取保密Draft数据。

## 5. 可验证交付与下一次任务

- 未注册设备Tensor计算：`vllm_ascend/_310p/v2/hybrid_state_device_meta.py`；纯CPU Torch回归`tools/310p_mm_ar/test_v2_hybrid_device_meta.py`，包含K7接受数、跨块、混batch/多req-slot、same-block偏移、非Spec -1、错位拒绝、请求释放重用与阻断点。
- **准入等级**：设备状态**计算代码已写/CPU待CI确认**；生产设备状态copy+ModelState真实绑定仍未实现。保留R2-A 0/46、Draft PR #7、Stock默认。
- **下个严格范围**：在不接入Worker的条件下实现device-native的per-group Mamba物理BlockTable寻址/状态复制消费接口，处理跨请求物理冲突与device-side失败标志，再提供与ModelState签名一致的未注册adapter及CPU模拟调用序列；其后分别审查NPU ABI和Stream，在用户单独批准后才开始最小设备验证。

本稿不存在目标工件身份/运行时实际类或Kernel命中的新取证，仍为冻结公开源码 + 隔离候选。
