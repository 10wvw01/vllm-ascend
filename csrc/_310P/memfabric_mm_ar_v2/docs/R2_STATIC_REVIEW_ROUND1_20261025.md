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
