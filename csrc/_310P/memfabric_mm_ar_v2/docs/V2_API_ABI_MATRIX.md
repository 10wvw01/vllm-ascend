# R2-01｜V2 Model Runner / DFlash vLLM 0.27.1 ABI 源码核对表

> 版本冻结：上游 `vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`（R1-B实机 `vllm 0.27.1+empty`）；我方 `10wvw01/vllm-ascend@dev/310p-memfabric-mm-ar`；R2工作支路。**静态源码事实；没有新的310P3推理PASS。**
>
> 本文比对的上游基底是 `vllm/v1/worker/gpu/` 的**V2 Model Runner 实现**。虽然位于 `v1/worker/gpu` 路径，类 `GPUModelRunner` 是上游新版Model Runner；不能因为目录名包含v1误判为旧版 `GPUModelRunnerV1`。

| ID | 上游函数或结构（固定SHA） | 我方对应或调用处 | 现状 / 当前处理 | 能否设备GO |
|---|---|---|---|---|
| ABI-01 | `GPUModelRunner.__init__(vllm_config,device)` + GPU `torch.cuda.Stream` | `worker/v2/model_runner.py:NPUModelRunner.__init__` 用 `torch_cuda_wrapper` | 构造初始化在NPU侧的全部CUDA使用/实例替换仍需核实，不能仅以包装上下文判断全部安全 | NO |
| ABI-02 | `GPUModelRunner.prepare_inputs(scheduler_output,batch_desc)` + `InputBatch` | Ascend同名override | T01～T04已在310P候选Runner局部绑定非Triton实现，T01 lookahead物理整行候选已补；输入/seq_lens设备短写、完整ModelState和310P3算子/流合同尚未通过。 | NO |
| ABI-03 | `GPUModelRunner.prepare_attn` / `BlockTables.compute_slot_mappings` | 310P专属Runner局部clone注入`NPUBlockTables`候选 | 已有非Triton多组Gather/Slot与32B物理行写候选；Attention/backend是否接受非连续逻辑view、NPU算子/流、KV多组生命周期和真实5 Sliding+1 Full仍未通过 | NO |
| ABI-04 | `GPUModelRunner.postprocess_sampled(idx_mapping,sampled_tokens,num_sampled,num_rejected,query_start_loc=None)` | Ascend V2 override | 当前用上游函数，内部直接调用上游GPU `input_batch.post_update`（Triton）；`worker/v2/input_batch.post_update` 并未自动替换该符号，不能以本地函数存在为PASS。当前Sync已经改为`torch.npu.current_stream`，但状态更新仍未适配 | NO |
| ABI-05 | 上游0.27.1 **没有** `GPUModelRunner.postprocess(...)` | Ascend V2旧版遗留hook | 保留旧版本可调用时的受保护兼容，0.27.1实际走 `postprocess_sampled` | 仅源码修复 |
| ABI-06 | `DFlashSpeculator.set_attn(model_state,kv_cache_config,block_tables,target_input_buffers,target_attn_groups)` | Ascend V2原先仅3入参 | 已修复为条件签名兼容：0.27.1需5参，早期Ascend旧ABI仍可以3参；初始化draft模型与attn metadata需继续检查 | 仅源码修复 |
| ABI-07 | `DFlashSpeculator._build_draft_attn_metadata(num_reqs,num_reqs_padded,num_tokens_padded,seq_lens_cpu_upper_bound,step, ...,causal)` | Ascend V2旧`build_draft_attn_metadatas(num_reqs_padded)` | 已删除无上游入口且必填参数不足的过时包装；上游运行时继承正确的新方法，并在`propose`期间把构建函数指向Ascend backend；需检验`_group_causal`、per group metadata | NO |
| ABI-08 | `DFlashSpeculator.propose`、`prepare_dflash_inputs` | `worker/v2/spec_decode/dflash/speculator.py` | 参数表当前外观接近0.27.1，但上游global `prepare_dflash_inputs`仍必经Triton，多组GID循环尚未适配；`_context_slot_mappings`的int32布局须确认后端 | NO |
| ABI-09 | `GPUModelRunner.sample_tokens` + `sample.Sampler` + `spec_decode.RejectionSampler` | Ascend V2 runner继承 | Sample`get_num_sampled_and_rejected`和RejectionSampler`_flatten_sampled_kernel`、`post_update`等CUDA/Triton符号尚无310P V2全套替代 | NO |
| ABI-10 | `GPUModelRunner.initialize_kv_cache` / `init_kv_cache` /`ModelState` | Ascend V2 `initialize_kv_cache`、`attn_utils` | 需要核对W8A8与Mamba/GDN、multi KV Group、每层block_size/真实kernel block size、不同窗口mask及请求复用；没有310P3 V2实机证据 | NO |
| ABI-11 | `GPUModelRunner.execute_model` / `shutdown` / `torch.cuda.*` | AscendV2继承 | inherited执行并非全部经`torch_cuda_wrapper`，检查流、device内存、profiling、Graph manager/Context、Eager的CG manager=NULL路径，不能盲目解除NOT_IMPLEMENTED | NO |
| ABI-12 | 上游 `VllmConfig._validate_v2_model_runner` 与 `use_v2_model_runner` | 平台patch已完成 | 310P3双开关、vLLM0.27.1、TP2 Eager只豁免Triton，剩余feature检查保留；Worker因未移植V2仍主动报NOT_IMPLEMENTED | 配置准入已PASS，Runtime NO |
| ABI-13 | `GPUModelRunner.__init__` 直接构造UVA `RequestState` | 专属Runner局部clone注入 `NPURequestState` | 已避免GPU UVA RequestState首次构造，但Sampler嵌套状态/ModelState/生命周期、310P3设备算子和双轮静审仍未完成；未注册Worker | NO |
| ABI-14 | `GPUModelRunner.initialize_kv_cache` 构造GPU `BlockTables` | 专属Runner局部clone注入 `NPUBlockTables` | KV构造候选已接，但多KV组/Attention/ModelState真实设备合同、物理blocksize及DFlash mask尚未静审闭环 | NO |
| ABI-15 | `post_update`修改设备端Token/length状态 | 310P Eager post_update/32B staged RMW与释放预检候选 | P0，pending写和继承释放顺序CPU合同已补，但in-flight所有权/异步失败、设备流与异常销毁未审计闭环 | NO |

## 0.27.1主要签名与调用源

- [上游GPUModelRunner](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu/model_runner.py) `postprocess_sampled` / `prepare_inputs` / `initialize_kv_cache` / `sample_tokens`。
- [上游DFlashSpeculator](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py) `set_attn`、`propose`及每组`prepare_dflash_inputs`。
- [上游DraftModelSpeculator](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu/spec_decode/speculator.py) `_build_draft_attn_metadata`需CPU upper bound + step。
- [我方Ascend V2](../../../../vllm_ascend/worker/v2/model_runner.py)实际源码请从仓库根目录进入，`_310p/model_runner_310p.py`则是旧V1派生Runner，**不可直接把名字当成V2**。

## 审查结果与变更纪律

首轮已确认并修复部分静态ABI，但 **DFlash V2 Runtime、310P非Triton数据路径、W8A8/GDN、双Rank数据正确性均尚未完成**；剩余12个合约点不得因本地合同测试绿而自动闭环。下一步必须将R2-02的每个必经Triton函数映射到真实310P路径，单测完备并复核上游ABI，再决定移除Worker防护；不得在R2-A期间宣布实机GO。

## 第二轮继承数据结构审视（新增P0）

- 上游 `gpu/block_table.BlockTables.apply_staged_writes` 单/多组均调用 `gpu/buffer_utils._apply_write_kernel`（Triton）。此前仅发现`gather_block_tables/compute_slot_mappings`；如果只替换后两者仍会在新增KV块时失败。
- 上游 `gpu.states.RequestState` 构造依赖 `StagedWriteTensor(uva_instead_of_gpu=True)` 与`UvaBackedTensor`，两者初始化需 `UvaBufferPool`，后者要求 `is_uva_available()` 且 `get_accelerator_view_from_cpu_tensor` 可在设备上成功映射。**未有310P3实机证据证明可用，也不得猜测一律不支持**；G0静态契约需为其建立替代或特征级守卫。
- 由此，安全准入不能仅解决DFlash多KV输入：`RequestState/UVA→StagedWrite→BlockTables→Sampling/PostUpdate` 是一个连贯数据所有权协议，局部patch某个Triton函数会产生空KV/错误指针风险。


## R2构造路径补充（2026-10-25）

- ABI-13 **P0**：固定上游`GPUModelRunner.__init__`直接实例化`RequestState`（其构造通过`StagedWriteTensor(uva_instead_of_gpu=True)`及`UvaBackedTensor`创建UVA），现有Ascend `NPUModelRunner.__init__`在`super().__init__`之后才`del self.req_states`并重新构造`AscendRequestState`，且后者仍继承上游UVA父构造。**仅新增NPURequestState并不能阻断上游UVA构造，必须重构310P Runner初始化实际路径**。
- ABI-14 **P0**：`GPUModelRunner.initialize_kv_cache`会构造原`gpu.block_table.BlockTables`；新`_310p/v2/block_tables.py:NPUBlockTables`必须通过专属Runner显式接线，并保证`append_block_ids/apply_staged_writes/gather_block_tables/compute_slot_mappings`和DFlash外部out指针/缓存的完整ABI。现在未接线，设备GO=NO。
- ABI-15 **P0**：`RequestState.total_len/num_computed_tokens`由采样后处理在设备上修改；不能每次从CPU镜像整向量回写。新增staging候选采用dirty-index scatter，尚需冻结GPU`post_update`写序、NPU算子支持及in-flight可见性。非因果多KV、W8A8/GDN、采样全路径仍NO-GO。
