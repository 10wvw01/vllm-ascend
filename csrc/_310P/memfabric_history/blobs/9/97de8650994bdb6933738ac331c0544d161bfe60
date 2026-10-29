# R2｜唯一Target模型与DFlash是否调用Triton的真实路由核查（2026-10-25）

> **纠偏裁决：具体模型/具体Runner/具体DFlash调用优先于“通用源码中有Triton”的推断。** 本文是冻结源码+历史已归档现场证据检查，**没有在本轮运行310P3推理**。不得把静态“存在”误作硬件“已命中”。权威研发源`10wvw01/vllm-ascend`，外部`HaonanWu1/vllm-ascend`仅只读。

## 1. 用户唯一场景：禁止拓展成泛模型V2兼容项目

| 字段 | 本项目约束 |
|---|---|
| 软件 | 自有`vllm-ascend`，依赖冻结`vllm@6e448d0... (0.27.1)` |
| 硬件 | **310P3 单卡双Die、TP=2、PP=1** |
| Target唯一公开身份 | **`Eco-Tech/Qwen3.6-35B-A3B-w8a8`**，ModelScope `https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8` |
| 模型信息 | ModelScope模型卡为`qwen3_5_moe`、W8A8、与Qwen3.5-35B-A3B相同混合架构（GDN/Full Attention）。**尚未取得ModelScope原始config.json的SHA及完整值，不冒充已锁定版本** |
| 现场Target路径 | R1-B记录`/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`；需读取本地config/量化文件及身份校验，不能仅凭目录名取代用户指定ID |
| Draft | R1-B实证独立的`/home/models/Qwen/Qwen3.6-35B-A3B-DFlash`，5 Sliding+1 Full，`sliding_window=4096`；此模型**不是Target** |
| 压测不可变负载 | **input=4096、output=2048、batch/并发=10、TP=2、DFlash启用**；沿用 **K=7** 与完整 **50 requests**、Eager先验，之后可独立研究Graph |
| 对照 | 在同一真实Runner、同一Target/Draft、同负载下Stock MM+HCCL AR 与 V2 MM+MemFabric AR；严禁拿不同Runner或不同DFlash状态的数据算融合收益 |
| 运行额外配置 | R2计划明确`--mamba-cache-mode align`、`--mamba-ssm-cache-dtype float16`、`--quantization ascend`、`--dtype float16`、`cudagraph_mode=NONE`；必须与实际启动命令逐项比对 |

## 2. “310P DFlash 会调用 Triton 吗？”分路径回答

**A．现有310P专属旧Runner / 旧DFlash Proposer：关键DFlash输入构造路径已不走Triton。**

- 我方`vllm_ascend/patch/worker/patch_idex_310.py`约71-95行将`AscendDflashProposer.set_inputs_first_pass`绑定到`AscendDflashProposer310`；后者`vllm_ascend/_310p/spec_decode/dflash_proposer_310.py`的`set_inputs_first_pass`调用`torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs`，明确是AscendC替代通用Triton input builder。
- 同一个310P patch约108-124行将几个采样/Spec helper模块的`HAS_TRITON`强制为`False`走已有的PyTorch后备实现；并有310P GDN相关自定义路径。**这不能证明整个旧Runner的所有潜在CUDA/Triton均不可达**，但足以否定“310P3的DFlash输入天生必须Triton”。
- 真实R1-B在Target两个Rank装载完成后，**在DFlash Draft混合Sliding/Full的旧V1 Runner架构门禁失败**。上游`vllm/model_executor/models/qwen3_dflash.py:115-127`明确禁止旧Runner混合多KV组；并非观察到DFlash Triton launch失败。由此不能宣称旧路径DFlash推理实际PASS。

**B．冻结 vLLM 0.27.1 通用GPU V2 DFlash：源码中确实直接发Triton。**

- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`第303行`DFlashSpeculator.propose`在第376行调用同模块的`prepare_dflash_inputs`；约472行`@triton.jit`定义`_prepare_dflash_inputs_kernel`，约674行`kernel[(num_reqs,num_blocks)]`启动。
- 当前310P专属V2候选`vllm_ascend/_310p/v2/dflash_speculator.py`已局部绑定`dflash_input_ops.py`的非Triton候选，不是通用GPU输入核；但其**Worker仍明确NOT_IMPLEMENTED**，没有该Target+DFlash的真实V2 Runtime/Device追踪。故静态结论只是：*若未正确接入310P替代而调用冻结GPU V2路径，则可能落入Triton；实际V2部署尚未发生*。

**C．目标 Qwen3.6 的 Hybrid/Mamba 模型状态：不应跳过，但要先证明运行时分支。**

- ModelScope模型卡说明此Target采用Qwen3.5式GDN+Full的混合架构。冻结`init_model_state`对`model_config.is_hybrid`选择`MambaHybridModelState`，其中`preprocess_state`的Triton在`mamba_cache_mode=align`且有活跃请求时可达；`postprocess_state`采样计数路径也含Triton。**上一轮T30/T31审计针对的是冻结GPU V2继承路径，不是已证实的这个310P Target的设备trace**。
- vLLM0.27.1的`CacheConfig.mamba_cache_mode`源码默认`none`，**但本项目V2冻结Serve参数明确为`--mamba-cache-mode align`**。因此不可误删T30；下一步要确定该Target选用的`ModelState`类、是否重载、在310P适配后的真正调用点及参数，才能决定T30/T31非Triton替代的实际设计范围。
- 现存 **46项（31 T + 15 ABI）为保守静态风险清单**，不是“46条均由该Target真机命中”。任何特定分支能判`NOT_APPLICABLE`都必须附固定Config/源码可达性、双轮复核，不能凭推测放行。

## 3. 当前证据等级、未完成的确认与实施止损

| 命题 | 结论 | 证据等级 |
|---|---|---|
| Target是`Eco-Tech/Qwen3.6-35B-A3B-w8a8`且W8A8 | 是；唯一用户锁定模型，ModelScope卡可佐证 | 用户合同 + 公开模型卡；本地版本SHA未核 |
| 310P旧DFlash输入替代Triton | 是，已连接`npu_copy_and_expand_dflash_inputs`源码 | 我方定制源码事实 |
| GPU V2 DFlash仍定义/启动Triton | 是，冻结`DFlashSpeculator.propose`源码直接引用 | 冻结上游事实 |
| 此Target在310P3的DFlash已经跑到了Triton并失败 | **没有此证据**；R1-B在Draft混合KV门禁先失败 | 历史实机日志 |
| T30/T31为本Target每步已观察到的必经Triton | **没有此证据**；R2确设align但ModelState动态选型/Worker运行未过 | 条件可达静态风险 |
| DFlash K7 end-to-end正确及MemFabric收益 | **NOT_RUN / UNKNOWN** | 尚无现场证据 |

下一步严格先做**模型-运行时选路证据表**，依次确认：Target和Draft本地原始config字段/SHA→现有310P patch注册实际身份→旧Runner失败门禁与V2必要性→具体`init_model_state`/GDN Attention backend→DFlash Speculator/Proposer实际函数全局符号绑定→Eager是否还存在Triton/CUDA/UVA**真实可达**入口；再根据实际需要安排最小源码适配、CPU gold及两轮静审。**现在停止继续泛化兼容开发，不改变现有V2 Worker的NOT_IMPLEMENTED安全保护，不部署/重启/改动用户310P3机器。**

### 来源
- 公共ModelScope Target：`https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8`
- 冻结vLLM源码：`https://github.com/vllm-project/vllm/tree/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- 自有310P V1 Patch与DFlash Proposer：`vllm_ascend/patch/worker/patch_idex_310.py`、`vllm_ascend/_310p/spec_decode/dflash_proposer_310.py`
- 历史实机证据：`R1B_STOCK_EAGER_SMOKE_NOGO_REPORT_20261025.md`、`DFLASH_STOCK_V2_R1_GO_GATE_20261025.md`

## 4. 自研Draft NDA：最小披露闭环（2026-10-25）

用户确认Draft为**自研保密模型**。既有R1-B现场报告已公开的“混合Sliding+Full导致旧Runner不兼容”是当前已存在的项目证据，不据此要求用户再提供私有模型信息，也不读取任何私有权重、源文件或训练资料。仅当具体配置核实必不可少时，允许负责人自行在服务器本地用[只读脱敏工具](../../../../tools/310p_mm_ar/r2_target_route_probe.py)分类配置，**只回传固定枚举，不上传原始文件、路径、日志、函数实现、层参数**。Draft config为可选；未提供则保持`UNVERIFIED`，不能猜造。

[保密版静态/运行时三级证据合同](R2_TARGET_ROUTE_MIN_DISCLOSURE_20261025.md)明确：V1定制AscendC已接入、冻结GPU V2含Triton、候选V2已有scope替代，均属S等级**静态事实**；模型分支C等级与运行时R等级未闭环，`NPUWorker310`仍`NOT_IMPLEMENTED`。接下来的工作是按真实模型条件排除不相关矩阵项，不再为所有泛模型先编码。

## 5. 冻结Target模型类源码事实（比模型卡更进一步）

冻结`vllm@6e448d0`的`vllm/model_executor/models/qwen3_5.py`有`Qwen3_5MoeForConditionalGeneration → Qwen3_5ForConditionalGeneration → IsHybrid`继承；`ModelConfig.is_hybrid`依赖内部ModelInfo和layer_types，`init_model_state`优先自定义get_model_state_cls，然后才按模型/模块判断是否走`MambaHybridModelState`。因此Target公开架构与Hybrid候选路径具有**更强静态关联**，但在没有本地模型config/真实Runner动态状态前**不升级T30/T31为现场必经**。新增脱敏源码字段`qwen35_moe_hybrid_declaration`，只读取冻结公开Python文件，不涉及保密Draft资料。
