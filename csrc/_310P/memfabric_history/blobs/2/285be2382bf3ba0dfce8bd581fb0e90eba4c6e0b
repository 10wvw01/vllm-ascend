# R2-01/02：T13 V2 完整继承调用链静态审计（2026-10-25）

> **2026-10-25 范围纠偏（优先阅读）**：本报告最初把“冻结GPU V2 + Hybrid Align条件满足时的Triton入口”表述为目标模型**已经确定每步必经**，缺少模型config→动态ModelState选择→310P实际Runner的可达证据。**撤回未经验证的真实必经断言**；T30/T31保持源码风险项而非设备实证。冻结vLLM的`mamba_cache_mode`默认`none`，但本工程R2启动方案明确传`--mamba-cache-mode align`；仍应审，不得当成已运行。V1 310P DFlash早有AscendC输入替代，上游GPU V2才有`_prepare_dflash_inputs_kernel`。目标模型限定为`Eco-Tech/Qwen3.6-35B-A3B-w8a8`，详见 [特定模型路由核查](R2_TARGET_MODEL_DFLASH_ROUTE_20261025.md)。历史正文保留原审查时间点，优先以上述修正口径解读。
>
> 冻结上游：`vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`；我方基线：`10wvw01/vllm-ascend:work/310p-v2-r2-implementation-20261025@9c50342369e5`（本轮起点）。目标：310P3单卡双Die、TP2、PP1、Eager、Qwen3.6 Target Stock与DFlash K7。审查方式：冻结源码逐入口/类方法/委托的跨模块调用图复核；不含真实NPU执行。**判定：源代码调用链审查取得新缺口，T13仍UNRESOLVED；静态准入NO-GO。**

## 1. 继承合同、为何局部替代不等于整条路径安全

实际继承：`NPUModelRunner310V2Candidate → vllm_ascend.worker.v2.model_runner.NPUModelRunner → vllm.v1.worker.gpu.model_runner.GPUModelRunner`。冻结`GPUModelRunner`本身有约30个生命周期方法（见下表），只有调用`self.method()`时才动态派发到310P的override；**函数正文里直接引用的module-global `InputBatch`、`init_model_state`、`AsyncOutput`、`init_attn_backend`等不会随继承自动更换**。而被`torch.inference_mode`装饰过的方法需先`inspect.unwrap`再执行私有globals clone；其他模块方法、ModelState类及构造器不在该作用域内。

## 2. 入口到设备调用的审计矩阵

| 阶段和冻结GPUModelRunner入口 | 动态继承与委托 | 310P候选/状态 | 静态裁决 |
|---|---|---|---|
| Worker `init_device` | `NPUWorker310`原V1派生 | V2开关显式`NOT_IMPLEMENTED`且在设备初始化前拒绝 | Fail-closed PASS；R2-03 NO-GO |
| `__init__` | Ascend父构造→310P `_310p_v2_parent_constructor` clone | 私有替换RequestState、DraftTokensHandler、torch代理；InputBuffers后来被Ascend重建 | UVA/Stream/Event其它间接源及构造失败所有权仍P0 |
| `load_model` | `get_model_loader`→`init_model_state` / sampler / grammar | 局部克隆注入Sampler、Rejection、Structured Worker；新增hybrid align早期NO-GO | T30/T31阻塞Target Stock；不等于ModelState适配 |
| `get_kv_cache_spec,initialize_kv_cache` | `init_attn_backend`、`BlockTables`、`init_kv_cache`、speculator.set_attn | 独立NPUBlockTables注入；Graph NONE管理器 | 5 Sliding+1 Full与Mamba/GDN group、allocator、Stride及流未核 |
| `profile_run,_dummy_run` | GPU `_dummy_run→self.execute_model(dummy_run=True)` | normal Eager调用310P `prepare_inputs`，但dummy由GPU module-global `InputBatch.make_dummy`构造 | 310P `execute_model`现局部绑定`AscendInputBatch`，设备profile UNKNOWN |
| `_dummy_sampler_run` | GPU函数直接用module-global `InputBatch.make_dummy` | 310P新增局部`AscendInputBatch`绑定 | 后端采样/shape依旧待证明 |
| `finish_requests,add_requests,update_requests` | GPU请求复用、ModelState与BlockTables staged状态写 | 310P `_remove_request`先预检pending | in-flight多Stream、KV复用及ModelState清理未事务化T16 |
| `execute_model`正常Eager | `self.prepare_inputs`,`self.prepare_attn`,`model_state.preprocess_state` | 前两段走310P/Ascend候选 | **后者未替换**；真实hybrid Align进入T30 |
| `model_state.prepare_attn/prepare_inputs` | ModelState factory选hybrid、后端attention建metadata | CPU seq_lens与NPU group backend | Slot stride、mask、KV metadata与GDN输入未实际证明 |
| `model(**model_inputs)` | Target W8A8/Mamba/GDN/MoE真实模块 | 没有做一般模型算子重写 | Model执行链GPU/Ascend Kernel闭包未获设备证据 |
| `sample_tokens,postprocess_sampled` | 310P复制sample_tokens、T09 override；仍`self.model_state.postprocess_state` | K7对齐行、Greedy候选、AsyncOutput替代 | **后者进入T31**；拒绝采样概率、状态回写和device写仍未闭环 |
| `take_draft_token_ids,pool,postprocess_num_computed_tokens` | 按DFlash/PPl/Pooling配置条件可达 | TP2/PP1非Pooling主路径不应默许其他模式 | 需要负例阻断和实际调用证明 |
| `capture_model,profile_cudagraph_memory`、Graph manager | FULL/PIECEWISE分支 | 本期配置仅Eager | R3独立，不能外推为Graph支持 |
| `shutdown`、KVConnector/AsyncOutput | GPU `torch.accelerator.synchronize`、异步输出、池清理 | 无正式模型生命周期证据 | MF长生命周期/失败恢复/跨流等待UNKNOWN |

## 3. 新增必经 Triton 路径与冻结源码证据

**T30，Target Stock就触发的 Hybrid Align preprocess。** `vllm/v1/worker/gpu/model_states/__init__.py:init_model_state`在`model_config.is_hybrid`时创建`MambaHybridModelState`。冻结`GPUModelRunner.execute_model`于真实batch调用`self.model_state.preprocess_state`；`mamba_hybrid.py:160-207`在`mamba_cache_mode=align`且有活动请求时启动`preprocess_mamba_align_fused_kernel[grid]`，又经`MambaSpecDecodeGPUContext.run_fused_precopy`启动`precopy_mamba_align_fused_kernel[grid]`（`vllm/v1/worker/mamba_utils.py:803-829`）。**不依赖于开启DFlash；关闭投机推理仍不能绕开。**

**T31，采样回写的 Hybrid Mamba状态。** 310P`postprocess_sampled`虽已替换T09，但仍调用`self.model_state.postprocess_state`。冻结`mamba_hybrid.py:289-342`对Tensor计数直接启动`_scatter_num_accepted_kernel[(n,)]`，在align且上下文已初始化时通过`run_fused_postprocess_align`再次启动`postprocess_mamba_fused_kernel[grid]`（`mamba_utils.py:847-879`）。这里还涉及请求索引-1过滤、accepted计数归一、请求重用及可能的4B状态写，不能用无条件argmax/no-op替代。

**安全处置**：310P候选`load_model`在`model_config.is_hybrid`且`mamba_cache_mode=="align"`时**早于模型加载直接NotImplementedError**，不容许以后误注册Worker时“装载成功但第一步误入Triton”。该防护不是310P Mamba实现；R2-B Target Eager仍不可开展。

## 4. dummy/profile输入分叉修复与剩余风险

正常`execute_model`调用`self.prepare_inputs`会进入310P override；但冻结`execute_model` dummy分支直接`InputBatch.make_dummy`、`_dummy_sampler_run`也直接引用其模块global。因此只写普通input适配仍会在profile/dummy得到不带Ascend seq_lens/attn_state合同的对象。本轮为**仅310P专属候选**scope clone `inspect.unwrap(GPUModelRunner.execute_model/_dummy_sampler_run)`，只覆盖`InputBatch: AscendInputBatch`；无进程全局patch、无修改上游GPU或其他NPU。

尚需检查`AscendInputBatch.make_dummy`的`asdict`拷贝/输出别名、`update_cos_sin`及NPU流、`KVConnector.set_disabled`发生异常时是否恢复（冻结`_dummy_run`没有finally），不能因CPU AST通过宣称profile/首写或多Group状态安全。

## 5. R2-01/02 后续硬阻断清单

- **T30**：实现真实无Triton的Align预处理/状态前拷贝，对照冻结mamba_utils正确性。覆盖普通Decode、K7、跨block、不同blocksize、5 Sliding+1 Full、首调/复用/异常，设备最小写粒度与跨stream状态可见性。
- **T31**：实现无Triton Accepted Token散写和Align后处理，覆盖accepted=0/1/8、masked idx=-1、全拒绝与部分接受、KV与SSM状态同步；禁止不完整Greedy误充概率语义。
- **T13剩余**：分支级静态穷尽`get_model_state_cls`定制模型、Attention backend/allocator、Mamba/rotary helpers、KVConnector/AsyncOutput/Shutdown及NPU算子动态派发，特别`MambaHybridModelState.add_request`单个int32 state `fill_`窄写和异常后KVConnector禁用位复位。
- **准入变化**：发现两条真实必经Triton，不重复泛称T13，准入从先前29T+15ABI=44，扩为**31T+15ABI=46**，历史44仍保留溯源。新增T30/T31的源代码/UT/实际绑定/设备算子审查/两轮深审仍均缺完整证据。**T13维持UNRESOLVED，R2-A STATIC-NO-GO，Worker NOT_IMPLEMENTED，R2-B/C/D ONDEVICE NOT_RUN。**

本报告是**第一轮静态深审补遗**，不构成第二轮独立静审PASS，也不授权OpenCode上机。
