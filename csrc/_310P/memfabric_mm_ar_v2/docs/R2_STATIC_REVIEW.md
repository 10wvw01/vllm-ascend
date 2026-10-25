# R2-A 静态深审账｜310P3 Triton-free V2（2026-10-25）

> **审查结论：ROUND-1 AUTHOR AUDIT = NO-GO；ROUND-2 INDEPENDENT REVIEW = NOT_RUN。**
> 本文件不是独立第二轮审查，更不是设备实测或静态准入批准。
> 自有唯一主线 `dev/310p-memfabric-mm-ar`，R2 PR #7 始终 DRAFT。
> 冻结上游 `vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`（0.27.1）。

## 本轮实际审查范围和证据

| 调用段 | 冻结真实入口 | 310P候选接线/代码 | 结论 |
|---|---|---|---|
| V2构造 | `GPUModelRunner.__init__` -> `RequestState` -> UVA | `_310p/v2/model_runner.py`私有globals clone注入`NPURequestState`；Ascend公共Runner仅增加子类可选工厂 | **部分**：不再先构造UVA；但模型/流/采样全生命周期未审 |
| KV | `GPUModelRunner.initialize_kv_cache` -> `BlockTables` | 私有clone注入`NPUBlockTables`，分组物理/Kernel blocksize独立 | **NO-GO**：`uint64`指针元数据、NPU gather/index_copy和ModelState KV ownership未核实 |
| T01～04 | `Ascend NPUModelRunner.prepare_inputs` -> GPU Triton globals | 私有clone注入Eager函数 | **部分**：CPU Torch合同；设备算子/输入时序未证实 |
| T09 | `GPUModelRunner.postprocess_sampled` -> Triton `post_update` | 专属override + `post_update_ops`；动态req-id/count/vocab/capacity先校验 | **NO-GO**：`torch._assert_async` NPU语义和多缓冲写失败原子性未知 |
| T11 | `DFlashSpeculator.propose` -> Triton `prepare_dflash_inputs` | `NPUDFlashSpeculator310V2` 使用 `inspect.unwrap` + scoped clone；每KV组Eager构造 | **部分**：5 sliding+1 full不同物理blocksize CPU gold；非因果mask/backend未审 |
| T17～24 | `patch_v2/patch_triton.py` -> Ascend sampler、rejection、grammar、logprob Triton | **没有310P Eager全链** | **P0 NO-GO** |
| T25 | `GPUModelRunner.sample_tokens` -> `draft_tokens[idx_mapping]=draft_tokens` | K7长度7写入padding宽度8行 | **P0 NO-GO**：可能违反310P写粒度/复用隔离 |

## Round-1 作者静态审查：确认的P0与待分类项

1. **采样/拒绝算法 P0**：通用Ascend V2的 `gumbel_sample`、`apply_temperature`、`penalties/bincount`、`min_p`、`bad_words`、`logprob`、`rejection_sample`和grammar patch仍执行Triton；仅T07/T08/T09并不能让DFlash K7正确采样。现有Ascend `_probabilistic_rejection_kernel` 的非贪心分支把 `u`固定为0（恒接受），不是概率正确的验收方案。按T17～T24登记并阻断。
2. **K7短宽写 P0**：冻结上游 `GPUModelRunner.sample_tokens` 对 `draft_tokens[idx_mapping]`直接赋值，`NPURequestState.draft_tokens`是8宽存储的7宽view；可能产生短写/相邻行污染。必须设计整行32B/64B安全更新，并保持读取K7视图的ABI。
3. **请求所有权 P0**：CPU staged request/KV状态与device端sample/post_update并行修改；错误中途只能poison，不能声称跨组原子事务。必须对in-flight请求、复用、异常销毁、流event排序作完整Runner协议。
4. **设备算子支持 UNKNOWN**：`torch._assert_async`、`index_select`、`gather`、`index_copy_`、`searchsorted`、`torch.uint64`指针张量、全行D2D/H2D在310P3 torch-npu/CANN组合的可用性及无隐式host sync未获证据。**不能凭CPU Torch成功批准。**
5. **ModelState/Attention P0**：Target Qwen3.6 W8A8、GDN/Mamba、TP2、5 sliding/1 full Draft多KV attention及非因果mask的实际`init_attn_backend/ModelState`、KV物理分配、DFlash proposer构建尚未完成静态调用链审计；T13保持UNRESOLVED。
6. **安全入口未启用**：`NPUWorker310`继续 `NOT_IMPLEMENTED`，候选Runner未注册，正确。当前任何尝试取消fail-closed均应拒绝。

## CPU/CI核对（严格限定）

- `82e5cbe2a4015d8e8f058f5fc59c77953d300071` GitHub Actions `40174130432`：治理 115 unittest（轻量环境Torch用例skip）、54 pytest；独立Torch CPU 115 unittest，两个job SUCCESS。仅证明该SHA的CPU合同。
- 后续T17～T25登记及T09修补应按**最新HEAD对应CI**重新核对；不能借用上述旧绿。
- 门禁由T01～16+ABI-01～15（31项）扩至T01～25+ABI-01～15（40项）。`R2_STATIC_ADMISSION_EVIDENCE.json`仍空；`r2_static_admission.py --check`应拒绝准入。

## Round-2 独立静审（待完成）

**NOT_RUN，不能代签。** 需要与代码作者不同的审查者按冻结上游方法入口逐项复核：UVA/stream/所有真实Triton launch、采样随机数分布、5+1 group/物理blocksize、KV零化与复用、错误路径、双Rank/TP2生命周期、静态证据。审查者记录逐项SHA、失败样本、复验CI、签字/时间，不能用“脚本全绿”代替逻辑审查。

**最终判定：R2-A STATIC NO-GO / R2-B/C/D ONDEVICE NOT_RUN。**
