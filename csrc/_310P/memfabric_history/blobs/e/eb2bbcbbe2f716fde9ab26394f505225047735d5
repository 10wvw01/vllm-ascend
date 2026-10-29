# R2-P0-E/NPU｜Target GDN Hybrid 状态设备候选与 ABI 审查（2026-10-25）

> **审查范围**：唯一Target `Eco-Tech/Qwen3.6-35B-A3B-w8a8`，310P3 单卡双Die、TP2、PP1、DFlash K7、Eager，冻结 `vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`。不读取自研Draft的配置、源码或权重，外部仓库只读。**仅 unregistered candidate 和 CPU-Torch 复验；没有服务器上的 CANN 编译、torch_npu ABI 测试、设备启动或模型推理。R2-A 仍 STATIC-NO-GO 0/46，PR#7 保持 Draft，Worker 的 NOT_IMPLEMENTED 与 Target Hybrid load_model 早拒绝保留。**

## 1. 本期代码与 NPU 静态执行路径

新增 `vllm_ascend/_310p/v2/hybrid_state_npu.py`：

- **`GDNWholeBlockCopy310Candidate`**：在目标设备（CPU gold 或**显式 opt-in** 的 NPU 诊断）通过真正的 `index_select → cat → index_copy_` 操作，针对已知物理BlockTable，将整块Conv/Temporal状态写回；不是仅生成地址或数学公式。按batch行+独立Mamba Group/层寻址、SD/DS两种滑窗轴、Temporal的源列`src_col+bias`；所有组/层的源块快照先完成，再进行目的块覆盖；消除同块`src==dst,bias>0`的源数据被自己覆盖的风险。
- **`PaddedInt32State310Candidate`**：在真实Torch Tensor（CPU或同样需要opt-in的NPU）上，以8个int32组成32B物理行，将一个或多个持久req_slot接受数合并成整行更新。同一行多个槽位只提交一次；0规范化为1（chunked prefill）。这是一条**Host权威数据 / 同步Eager**的诊断路径，当前不从NPU采样结果直接读取动态标量。
- **`GDNMove310`**：严格CPU预检过的逻辑移动意图`(batch_row,src_col,dst_col,bias)`，不得凭Draft私有结构推造。原`hybrid_state.py` CPU状态迁移组件和`r2_hybrid_align_reference.py`决策gold继续作为独立对照，设备候选**尚未注册ModelState**。

最低安全条件：
1. 只接受TP2/K7已审范围下的GDN物理状态。Conv、Temporal必须是完整连续、每行数据至少32B且32B整除、物理行stride与基址32B对齐；不支持未审`as_strided` padding/别名、任意dtype、未审视的物理页粒度。
2. 每个Mamba KV Group与每个状态Tensor单独维护物理块；BlockTable必须是**同步且权威**的CPU int32快照。既不是上游GPU的UVA，也不能拿async调度的**乐观镜像**充作权威。
3. 所有目的写是整物理行`index_copy_`，窗口移位只操作临时完整`cat`块，而不是对持久缓存直接写`tensor[off:]`等窄地址。但**Torch-NPU底层index_copy是否确实产生安全32B写，仍没有310P3实机证据**，不能由形状推导设备PASS。
4. 全批次跨多Group的所有索引、块ID/容量、目的地址去重及源数据快照在任何写回前完成。设备提交失败立即poison，禁止在可能部分提交的状态上重试；同一物理目的块在本批被两个请求竞争写时提前拒绝。
5. NPU模式需显式`allow_npu_diagnostic=True`且绑定当前`torch.npu.current_stream`。设备发射完置`device_pending`；**不允许复用状态**直至由调用方明确`confirm_completed()`执行NPU stream同步且成功。CPU模式为同步写；NPU模型的生产异步Event所有权仍属未完成项。
6. 流中具体的生产者到消费者依赖（RequestState post_update → T31、T30 precopy → GDN forward）、失败后丢弃物理块、请求 abort/reuse、双Rank TP状态一致性尚未打通；绝不因为存在`confirm_completed`就宣称完成生命周期证明。

## 2. 四段状态处理：已完成与未完成的真实边界

| 契约 | 2026-10-25可核查实物 | 阻塞 |
|---|---|---|
| T30-A 运行块索引决策 | `r2_hybrid_align_reference` / `hybrid_state.py` CPU语义 | 真正device-resident `ceil((computed+query_len)/B)-1`、旧accepted状态、batch→slot读取及全行设备写入未接入 |
| T30-B GDN precopy | `GDNWholeBlockCopy310Candidate.execute` 可执行Torch整块Conv/Temporal搬运 | NPU index_select/index_copy_支持、stream/物理cache布局未验证 |
| T31-A K7 accepted scatter | `PaddedInt32State310Candidate` 真实Torch整32B行合并写 | Host当前是权威同步列表，不是实际rejection sampler设备输出；不能替换生产Triton scatter |
| T31-B align postcopy | 同一WholeBlock组件实现`src==dst,bias>0`安全搬运 | NPU post_update实时computed/accepted→物理块意图转换尚未设备resident，ModelState后处理未接线 |
| ModelState工厂与GDN metadata | 冻结上游静态源码分析/前轮CPU gold | 未实现ModelState类、真实`prepare_attn`、真实分组绑定/生命周期、Runner入口仍blocked |

**因此本轮应称作「NPU Tensor op候选 + 设备ABI前置约束 + CPU gold」，不能称作「T30/T31已完成310P3 NPU Kernel/ModelState替换」。**

## 3. 两轮代码审查

### S1：冻结语义 / 数据路径审查
- `mamba_hybrid.py:160-207` + `worker/mamba_utils.py:283-407`：T30决定是否跨块、使用上一轮接受数的`bias`，`precopy`跨Mamba group和2种状态类型；**单组OneTensor拷贝不足**。
- `mamba_hybrid.py:289-329` + `mamba_utils.py:202-281`：T31接受数scatter以`idx_mapping`映射持久req_slot，`postalign`可在`src==dst,bias>0`时发生同块移位，temporal偏移是BlockTable列。
- 上游物理状态可能带padding/不同group stride，本候选对未证实布局严格拒绝；不能偷换原始KV blocksize及FP16 dtype。
- **S1结论：** 最小搬运算法与公开源码公式对应，实机输入/Memory ABI未证实。

### S2：设备写粒度、同步和失败安全审查
- AscendC公共搬运约束通常以32B DataBlock为基本粒度，UB要求32B对齐；部分产品不支持DataCopyPad，重叠目的写需要barrier。出处：
  - https://asc.gitcode.com/api/SIMD-API/basic_api/data_move_guide/overview/overall_constraints.html
  - https://www.hiascend.com/developer/techArticles/20250627-1
- Torch `index_select/index_copy_`不是经310P3二进制审核的CANN Copy ABI。完整32B行的**源代码形状**不等于底层物理32B安全；严格不注册，不解除Worker或Hybrid加载守卫。
- 同batch多个请求的完整目标地址冲突必须在写前拒绝；跨Group第2组预检失败绝不能污染第1组；失败后poison，不允许半提交重试。NPU异步报错可能在发射后到同步时才出现，故设置`device_pending`与显式`confirm_completed`。
- **S2结论：** 可提交为隔离的设备实验候选；无CANN/310P3算子ABI与Stream证据，不能进入真实Serving，更不能由其认定静态GO。

## 4. CPU验证 / 下一工作包

测试 `tools/310p_mm_ar/test_v2_hybrid_npu_candidate.py`：
- 在CPU运行真正的`index_select/cat/index_copy_`（而非仅文字校验）；
- Conv SD/DS、Temporal、2组Mamba、跨块、同块重叠、两请求独立物理块、源快照、冲突拒绝、非法block前置拒绝、失败poison、8槽×4B整行接受数合并、未安装torch_npu的fail-closed；
- CI的纯标准库project-control可因无Torch跳过；单独CPU Torch必须全部实际执行。

接下来的**独立设备可行性门禁**应先用公开/普通合成cache完成，不需要Draft数据：
1. **S0：现场版本/来源只读核验**：torch、torch_npu、CANN、目标310P3、加载`.so`/分支/Git SHA、CPU物理块表布局，以及目标正式ModelState的可观察范围；G0未完成禁止任何设备测试。
2. **S1：最小NPU ABI验证（仅已批准时由用户委托OpenCode）**：Torch NPU `index_select`、`cat`、`index_copy_` 的int32索引/FP16或GDN dtype、完整32B行、同块源快照及设备Stream实际命中，尤其确认是否退化CPU或触发隐式同步；出现未解决507014/异常即停测。
3. **S2：真实NPU状态方案**：如果上述现成Torch op在310P3上不成立或性能不可接受，则在未注册的310P专属AscendC工程中开发真正的全行/memmove状态copy与int32 padded scatter；必须在确认该硬件DataCopyPad/不对齐/多核写冲突约束后选择接口，不能凭其他Ascend芯片文档照搬。
4. **S3：最后才允许**按Target自己的ModelState工厂、`prepare_attn`、请求生命周期对接Runner，经过两轮独立代码审查、CPU Tensor gold和全46条实际静态证据再考虑解除V2 Worker保护。

本轮未改变融合MM+MemFabric正式Kernel、Stock默认、Qwen性能基线或DFlash自研代码。**最终准入判定仍是 R2-A STATIC-NO-GO 0/46；设备/Graph NOT_RUN。**
