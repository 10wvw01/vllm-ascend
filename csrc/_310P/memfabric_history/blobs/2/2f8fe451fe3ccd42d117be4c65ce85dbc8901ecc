# R2｜唯一Target+DFlash“静态实路径”证据闭环与最小披露合同（2026-10-25）

> **唯一业务口径**：`10wvw01/vllm-ascend` + **310P3 单卡双 Die/TP2** + **`Eco-Tech/Qwen3.6-35B-A3B-w8a8`（W8A8）** + input4096/output2048/batch/并发10 + DFlash K7；性能阶段50请求。先Eager + 同Runner Stock对照。Draft为独立**自研模型，受保密条款保护**，本项目不索取权重、训练资料、原始配置、私有架构或私有源文件。
>
> 本轮目标是**确认上述唯一配置的源码静态选路**，而不是继续补通用V2兼容代码。静态源码证明只用于减少待确认集合，**不能替代运行时对象类型、函数调用和实机Kernel命中**。

## 一、三种证据等级（不得混淆）

| 等级 | 所需证据 | 已知结果 |
|---|---|---|
| **S：冻结源码事实** | 指定版本文件、patch绑定、函数源码入口 | `310P旧DFlash AscendC`、`上游GPU V2 Triton`、`310P V2候选局部非Triton`均可静态定位 |
| **C：该模型的条件选路** | Target/Draft的允许披露字段（仅枚举）、serve参数、动态ModelState factory分支条件 | Target公开架构为Qwen3.5 MoE系；R2 Serve声明`mamba-cache-mode=align`，本地config版本/实际ModelState对象尚未核实 |
| **R：运行时真实命中** | 可控只读trace / 实机取证，在已授权设备上观察Runner class、ModelState class、Speculator class、Kernel | **NOT_RUN**。310P V2 Worker保护仍为NOT_IMPLEMENTED；旧Stock历史在自研Draft多KV Group门禁先失败 |

因此旧R2 T01～T31是**潜在静态库存**。只有C与R支持某项**当前Target实际必需**时才允许把它转成开发工作；不能因GPU通用层存在Triton就推导“要实现全部31项”。

## 二、已经从公开/自有代码确认的必要最小链

1. **Target / W8A8**：唯一公开ModelScope模型卡标注`qwen3_5_moe`和W8A8，且说与Qwen3.5-35B-A3B架构一致。该事实只能验证公开模型家族，不能证明服务器上Target目录下每个文件的具体版本。
2. **旧310P Runner / Draft**：`patch_idex_310.py`显式绑定`AscendDflashProposer.set_inputs_first_pass=AscendDflashProposer310.set_inputs_first_pass`，其实现使用`torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs`替代Triton DFlash输入。旧Runner的Stock+DFlash失败是Draft混合组的上游门禁，非Triton执行失败。
3. **通用GPU V2 / Draft**：冻结`vllm@6e448d0`的`DFlashSpeculator.propose`调用`prepare_dflash_inputs`，后者发`_prepare_dflash_inputs_kernel[grid]`（Triton）。这是**GPU路径存在**，不是310P现场命中。
4. **310P专属V2候选 / Draft**：`_310p/v2/dflash_speculator.py`用`inspect.unwrap`和独立函数globals把上述`prepare_dflash_inputs`**仅在当前实例**替换为`_310p/v2/dflash_input_ops.py`，因此T11已有静态替代候选；Worker仍未接线/上机。
5. **Target的ModelState**：`init_model_state`选择`MambaHybridModelState`依赖运行时`model_config.is_hybrid`和是否有模型自定义`get_model_state_cls`。冻结源码`mamba_cache_mode`默认none，但R2既定Serve参数明确align。T30/T31保留**条件风险**，待核实Target实际State及其内部具体调用后才决定是否需新增算子。
6. **当前运行边界**：历史Target双Rank加载PASS；Draft旧Runner多KV组失败；310P V2 Worker`NOT_IMPLEMENTED`仍然阻断。尚无公开证据可证明Target+DFlash K7实际进入了某个Triton Kernel，更不能据此评估Kernel性能收益。

## 三、最小披露的只读证据收集工具

新增`tools/310p_mm_ar/r2_target_route_probe.py`：

- **只读**两个指定`config.json`（可只填Target；Draft为可选）；严格1MiB上限、文件名限定，不扫描模型目录，**绝不读取权重、Safetensors、量化权重、私有Python代码、tokenizer或其他文件**。
- 检查本仓已知**公开源码**的旧310P Proposer、V2候选和Worker安全门禁；可选`--vllm-root`只读冻结vLLM的3个公开Python源码文件。
- **不import**`torch/torch_npu/vllm/vllm_ascend`，不装依赖，不连接NPU，不启动Serve，不load模型，不改任何配置/文件，不探测进程/端口。
- 标准输出只有固定**枚举/布尔式结论**，不会输出目标或私有Draft的原始config内容、路径、错误异常文本、权重或层详情。Draft只会输出如`MIXED_SLIDING_FULL`、`ALL_FULL`、`UNVERIFIED`等**注意力布局类别**，不输出逐层列表、隐藏尺寸、层数、专家参数。
- 输出的`effective_runtime_runner/ModelState`恒为`NOT_OBSERVED`，`actual_device_kernel`为`NOT_RUN`；**即使所有源码特征都出现，也没有STATIC-GO或上机许可**。

对310P服务器本地的零设备运行方式（先`git pull`到与审计分支一致的独立只读工作树，**不要修改现有在线工作树和服务**）：

```bash
python3 tools/310p_mm_ar/r2_target_route_probe.py \
  --target-config /home/models/Qwen/Qwen3.6-35B-A3B-w8a8/config.json \
  --draft-config /home/models/Qwen/Qwen3.6-35B-A3B-DFlash/config.json \
  --vllm-root /vllm-workspace/vllm \
  --mamba-cache-mode align
```

注意：以上路径是**历史R1-B证据中的建议路径**，并非本轮重查后的确认路径；若实机安装目录不同，应由服务器维护者**在本地**选择确切的`config.json`；**不要上传文件**。如Draft配置读取也不符合保密政策，省略`--draft-config`即可，工具会报告`NOT_PROVIDED`；后续只请负责人私下确认布局是否属于“混合Sliding+Full”这一类别。

**仅请用户回传stdout里的固定枚举JSON**；输出前仍可再自行检查是否满足保密条款。切勿附原始config、训练资料、权重、私有目录树、日志全文、traceback或完整内部Layer清单。

## 四、剩余闭环需要的最少信息（全部不涉及私有实现）

| 问题 | 期望的允许披露答案 | 原因 |
|---|---|---|
| **Q1：本地Target身份** | 探针返回的公开`architecture_compatibility`枚举即可 | 确定与唯一公开Target体系一致，不要求config文件或任何SHA |
| **Q2：自研Draft多KV形态** | 仅`MIXED_SLIDING_FULL`等布局**类别**，如已能由历史日志核对也无需重复索取 | 判断V1多Group禁用是否仍成立、V2必须解决什么 |
| **Q3：实际Runner/ModelState** | 仅枚举类名的**公开基类**、是否V1/V2、是否使用`MambaHybridModelState`及是否有自定义State（是/否） | 决定T30/T31在该Target是否实际可达。当前Worker保护下不能借CPU探针假称已观察 |
| **Q4：采样策略** | `greedy`或`non-greedy`（温度>0与否）即可 | 决定是否真的需要概率拒绝采样T23；不需要任何私有采样设计 |

Q3和Q4目前都尚无完整证据。**先由公开代码/只读配置尽可能推断，再请求最低限度的枚举答复；不要要求用户披露任何DFlash算法/内部接口细节。** 若某项受保密条款限制，就标`CONFIDENTIAL_NOT_DISCLOSED`，继续其他已证实的工作，不得猜造。

## 五、本轮STOP/GO界限

- 本阶段只允许：公开源码审计、只读配置的脱敏分类器、CPU测试、更新工作分支审计总账。
- 本阶段**不允许**：启动设备、配置/权重修改、DFlash私有源码读取或上传、追加Triton兼容算子、解除Worker保护、修改模型实现、GPU/NPU动态import或微基准。
- 未来要得到**R等级真实路由**，必须另经静态安全审阅，指定不修改设备的运行时类型/调用点取证，获授权再由OpenCode在310P服务器执行；本轮CPU信息不足以声称真正完整闭环或正式运行GO。

**最终当前状态：源码S部分确定，模型私有信息未收集，C部分待脱敏核对，R=NOT_RUN，R2-A仍STATIC-NO-GO，PR#7 Draft、Stock默认、融合正式Kernel不动。**

### Qwen35-MoE真实模型类的静态证据补充

冻结上游`vllm/model_executor/models/qwen3_5.py`定义`Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts)`，基类`Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid)`；`vllm/config/model.py:1764`的`ModelConfig.is_hybrid`仍需结合模型registry的`_model_info.is_hybrid`及`layer_types`动态判定。新增探针枚举`qwen35_moe_hybrid_declaration=ISHYBRID_PUBLIC_MODEL_CLASS_PRESENT`只代表**冻结源码模型类声明**，不是本地模型最终选到该类、或`MambaHybridModelState`设备已真实调用。用户无需披露自研Draft的类型或任何非公开实现。
