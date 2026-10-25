# R2执行状态｜310P3无Triton V2 Model Runner（2026-10-25）

> 本文件是 [R2总任务表](V2_RUNNER_TRITON_FREE_MASTER_PLAN_20261025.md) 的**动态执行记录**。`PASS`仅指明确列出的代码/CI/取证，不得外推。自有仓库唯一主线`dev/310p-memfabric-mm-ar`；进行中的 [R2代码PR#7](https://github.com/10wvw01/vllm-ascend/pull/7) **DRAFT / 不合入**。

## 当前明确结论

- **R2-A总体 NO-GO；R2-B/C/D ONDEVICE NOT_RUN。** PR#5 的校验豁免只在CONFIG阶段生效；`NPUWorker310`仍主动`NOT_IMPLEMENTED`并阻断V2实机启动。严禁把禁令删掉后先上机试错。
- R1-B现场报告/23份原始证据已单独通过 PR#8 归档唯一主线（合入SHA `a37c0212f4222317552e56c357a77930f85696f2`）。G0 PASS、原Stock+DFlash Eager失败、V2 NOT_RUN；该现场G0不能跨代码工作树当作下一次GO。
- R2-A代码分支截至`61633443e50692a02725633b0a0cdd6018f51cb1`：局部 DFlash `set_attn` 0.27.1 ABI、旧Ascend回退、GPU V2旧postprocess处理、NPU current_stream、bonus logits 修补；`v2/reference_ops.py`多KV K7 gold。**仅参考语义和局部API修补，不是NPU V2 Runtime**。
- [vLLM ABI矩阵](V2_API_ABI_MATRIX.md)和[Triton替代矩阵](V2_TRITON_REPLACEMENT_MATRIX.md)确认至少 T01～09、T11、T14 BLOCKED；T15（RequestState/UVA）未核实；T13继承路径仍未穷尽。调试时`HAS_TRITON=False`必须保持真值。

## 工作项状态与下一批目标

| 范围 | 状态 | 已有实物 | 剩余硬门槛 |
|---|---|---|---|
| R2-00 | **PASS（GitHub归档）** | PR#8，R1-B原始报告与完整压缩日志、ADR-010 | 无需重复归档；下次设备G0必须重做 |
| R2-01 | **PARTIAL** | `V2_API_ABI_MATRIX.md`、6个ABI CPU静态UT | 尚未枚举完`GPUModelRunner`继承实际方法和ModelState/310P旧适配差异 |
| R2-02 | **PARTIAL/阻断** | `V2_TRITON_REPLACEMENT_MATRIX.md`（15项） | T01~T09、T11、T14实际替代未做；T13/T15未闭环 |
| R2-03～05 | **NOT_READY** | 独立 V2 Runtime开发方案；原Worker fail-closed | NPU安全Worker/Runner、RequestState/UVA、BlockTables/staged-write及真正Eager Torch/AscendC输入与状态更新 |
| R2-06 | **PARTIAL** | DFlash `set_attn`签名兼容/遗留metadata修订 | `_group_causal`及模型与多KV group/attention的实际构造、全接口审计 |
| R2-07～08 | **CPU-GOLD-ONLY** | `reference_ops.py`、K7/multi blocksize 8项gold/负例 | NPU Tensor adapter、真实attention metadata、sliding/full语义/SlotMap全路径接线 |
| R2-09～10 | **NOT_READY** | R1-B证明旧V1可加载Target，不等于V2 | V2 W8A8+GDN/Mamba、TP2 MoE、RejectionSampler、request abort/reuse、Rank一致性 |
| R2-11～12 | **PARTIAL** | 原V2模式fail-closed、双开关单测；最新CPU CI 61项unittest+54项pytest通过 | 真正新runtime完整无Triton合约、错误注入、多KV/采样/生命周期覆盖 |
| R2-13～14 | **OPEN** | 初轮审查和PR#7草稿 | 完成**两轮**针对可执行运行时的独立静态审查、CPU报告、无未分类依赖，才申请准入 |
| R2-15～21 | **NOT_RUN** | 已有冻结实机Runbook合同 | 未取得静态GO不得通知用户委托OpenCode进行NPU验证 |
| R3-01～04 | **后续** | 规划已锁定 | Graph/GateE/长稳和大M小M算子研究与R2分账 |

## 下一组实际编码顺序（不按异常逐轮上机）

1. **先修V2数据所有权基础**：替换/验证 `RequestState`中的UVA、`StagedWriteTensor`/ `FusedStagedWriter`以及多组`BlockTables`，消灭实际`_apply_write_kernel`/Gather/Slot三类Triton路径；记录CPU gold和FP16/多page真实性。
2. **再修无Triton执行输入+采样+状态**：T01～T04、T07～T09，含`Sampler`与`RejectionSampler`导入的上游模块级函数绑定；默认上游`gpu/input_batch.post_update`不能被只新建Ascend同名函数替代。
3. **最后接入DFlash多组与Target**：T11 K7 input builder、5 Sliding+1 Full group causal、物理blocksize、GDN/Mamba/W8A8/TP2模型加载和状态生命周期。必须在冻结vLLM 0.27.1版本、对方通用Ascend V2不回归的约束下实现。
4. **统一CPU和两轮静审**：R2`--check`脚本需明确记录每条mandatory ABI/KERNEL的READY/NO-GO，真实CI绿与CPU gold不充当NPU正确性；才允许出具**STATIC-GO**和实机委托提示词。

## 强制安全线

- 仅向自有`10wvw01/vllm-ascend`仓库提交短期分支和PR；上游 `vllm-project/vllm` 与 `HaonanWu1/vllm-ascend`只读。
- 不编辑DFlash config/layer_types、不安装Triton、不关闭`HAS_TRITON`真实性、不消除不当CUDA兼容性校验、不自动启用Graph。
- 直到完整静态门槛关闭，**PR#7不得合入且Worker fail-closed必须保持**；现场由用户批准并交OpenCode，不允许本侧宣称已实机PASS。
