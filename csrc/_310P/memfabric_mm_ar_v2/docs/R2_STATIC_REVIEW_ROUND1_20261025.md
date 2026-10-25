# R2 310P3 Triton-free V2 Runtime｜第一轮静态深审（2026-10-25）

> 冻结上游：`vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`。审查目标：`work/310p-v2-r2-implementation-20261025` 当前候选。结论 **ROUND-1 NO-GO**。本文是针对候选实际源码的缺陷/合同审视，不能替代第二轮静审和NPU设备算子核验。

## 审查覆盖的实际源代码

- `_310p/v2/model_runner.py`：scope-cloned upstream GPUModelRunner.__init__/load_model/initialize_kv_cache/sample_tokens、T01~T04局部绑定、T09 postprocess、T25 K7 padded写。
- `_310p/v2/request_state.py`、`block_tables.py`、`staged_buffers.py`：不经过GPU UVA构造的候选状态、分组BlockTable、KV写、slot alias、poison恢复。
- `_310p/v2/sampler.py`、`rejection_ops.py`、`gumbel_ops.py`、`dflash_input_ops.py`：DFlash K7准备、greedy采样和概率拒绝边界。
- `_310p/worker_310p.py`：真实运行入口仍主动`NOT_IMPLEMENTED`，没有误接未合格Runner。旧V1保持不动。

## 严重度P0的剩余缺陷/阻断

| ID | 源码路径/调用 | 结论 | 修复和复核出门条件 |
|---|---|---|---|
| P0-01 | `NPUWorker310.init_device`→`NPUModelRunner310V2Candidate` | **未注册**；继续fail-closed为正确行为，整个V2服务不可启动 | 所有P0闭环才受控接线；不得先删保护 |
| P0-02 | `GPUModelRunner.__init__`冻结代码local globals clone | 解决了父构造中UVA RequestState，但整体还可能含`torch.cuda`、get-model/ModelState上下文与stream/CPU→NPU异步语义；非“全Runner GPU-free” | 静态枚举全部继承方法与实际被patch后的CUDA/Triton符号；NPU设备算子审核，错误路径保证clean abort |
| P0-03 | `initialize_kv_cache`局部clone注入`NPUBlockTables` | 多KV group已经候选接线，但`init_attn_backend/init_kv_cache`、W8A8/GDN/Mamba/TP2 device allocator/状态/模型注册无法仅靠CPU模拟证实 | 针对5 sliding+1 full逐group验证layer映射、block size、slot映射/causal和release，保留设备G0与Rank验证 |
| P0-04 | `DFlashSpeculator.propose` + `prepare_dflash_inputs` | Torch Eager T11候选与多组slot已做，`torch._assert_async`、`index_select/gather/searchsorted/copy_`、小粒度写及NPU流均**没有可验证的310P3支持证据** | 逐算子明确310P3 dtype/shape/同步合同、错误注入、上下文流、异步可见性；不凭CPU推断 |
| P0-05 | `NPUSampler310V2` + `NPURejectionSampler310V2` | greedy T17/T23候选不是完整T17～T24；处理温度、penalties、top-p/min-p/grammar/bad words、logprob仍含Triton或未覆写；温度>0不能伪造为argmax或u=0 | 逐实际调用边追踪并用独立Torch/CPU gold和无Triton代码补齐；若业务需概率采样必须实现正确的随机拒绝与种子语义 |
| P0-06 | `sample_tokens`复制冻结上游 + T25 padded K7 | 64B整行修复是必要候选；`index_copy_`、padding、replay buffer、请求复用与跨流生命周期仍无310P3证据 | 请求mixed prefill/decode、KV preemption、abort/reuse、双Rank重复1000次并查first-write 507014/507899 |
| P0-07 | `_310p/v2/post_update_ops.py` | T09 Eager逻辑已接候选，但持久数据多Buffer写不能事务式回滚，`_assert_async`未确认NPU异常/多流传播 | host/Device slot权属一次一owner，校验在写前生效且失败立即poison并中断推理 |
| P0-08 | `R2_STATIC_ADMISSION_EVIDENCE.json` | 40个强制节点证据全为空，缺第二轮审查，仍为 `STATIC-NO-GO: 0/40` | 每条提交源码真实连线、test、device-op静态审视及双轮独立审查证据并独立`--check`，不得凭绿CI补造 |

## 本轮发现的关键原则

**最危险的错误是“各候选模块都实现了，认为完整Runner已经可运行”。** 上游GPUModelRunner方法有独立模块全局名、装饰器、ModelState/KV ownership、stream和sampling；只有在真正的`_310p/v2/model_runner.py`每处已显式连线并替换后才可能消除必经Triton，缺一个仍NO-GO。

目前最好继续保留`NPUWorker310` fail-closed并保持PR#7 DRAFT，直到真闭环；CPU动态合同是必要且有价值的证据，但不是设备支持证明。

## 最后两步实施顺序

1. 编码者逐条消除P0-02~07，特别是`GPUModelRunner` inherited路径、`Sampler.apply_sampling_params`概率/惩罚分支，固定K7的真实DFlash KV Group/ModelState；完善T17~T24代码与随机/异常gold。
2. 在所有路径完整后，完成**第二轮独立静审**与真实`r2_static_admission.py --check`，再出具STATIC-GO并请用户委托OpenCode G0。此时仍不得宣称实机PASS或改动服务器SO。

**ROUND-1 verdict：NO-GO。**

## 第一轮静审补遗：新增四条构造/采样必经路径（2026-10-25）

复读冻结上游GPUModelRunner`__init__`、`load_model`、`sample_tokens`发现此前8项P0报告遗漏了**T26～T29**四条独立CUDA/UVA路径：Sampler内五种嵌套状态、StructuredOutputsWorker CUDA Stream、DraftTokensHandler CUDA Stream/Event、AsyncOutput CUDA Event/set_stream。它们已在R2候选代码中用私有构造clone或专属类替换，且有CPU合同；但未通过310P3设备算子/流验证及第二轮独立静审。R2-A仍NO-GO，必须将四项单独纳入正式门禁，不能因T13概括性条目而漏审。

## T01/T16 静审补遗：两项已修复源码候选，设备准入仍NO-GO

- **P0-12：T01短列写。** `NPURequestState.add_request`原来按`[lookahead, req_state]`列`index_fill_`，`prepare_prefill_inputs`按列`index_copy_`，均可能发出4B粒度设备写。已改按请求分配32B对齐物理行、逻辑转置视图和整行提交（`63b93afa`）；未验证310P3`as_strided/index_copy_`、其它T01写入和实际Stream，仍BLOCKED。
- **P0-13：T16释放顺序。** 冻结上游`GPUModelRunner._remove_request`先释放ModelState，再调用带待写检查的RequestState，拒绝时可导致状态不一致。已新增非变更`assert_can_release_request`，并在专属Runner继承清理前预检（`e7158aaf`）；未覆盖in-flight设备所有权/异步错误，仍BLOCKED。
- 双job CI 40188687250 已核实139轻量unittest（13 skip）+54 pytest+162 Torch CPU PASS，但仍0/44 STATIC NO-GO、第二轮独立静审未PASS、ONDEVICE NOT_RUN。新增审查发现应在后续P0复核中继续追踪，不得提前注册Worker。

- **P0-14：T06 KV Slot Mapping短行写。** 原`slot_mappings[group,max_tokens]`的物理stride及`[:, :active_tokens].copy_`未确保32B行完整提交；现用物理padding及全行构造/整缓冲写（`0e819f23`），两组异构blocksize及非8倍数token CPU测试通过。仍需真实310P3 Attention接受非连续逻辑view的stride/alias合同、NPU算子/流验证，T06继续BLOCKED，不能视作STATIC-GO。

- **P0-15：T09接近Token行尾时 masked 列与accepted列冲突**。原`rows.scatter_`以clamp后的列写入，sample宽度大于剩余容量时无效列可擦掉合法已接受Token。新增accepted-only delta `scatter_add_`和空penalty词表、int64→int32越界写前检查及Torch CPU负例。**设备`scatter_add_`、同流顺序与多Buffer原子性仍未核验，静态审查不通过，第二轮独立深审NOT_RUN**。
