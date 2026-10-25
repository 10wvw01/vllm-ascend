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


## 2026-10-25 R2-A T05/T06/T14/T15 隔离候选实现（继续 NO-GO）

- 本轮实际源码新增 `vllm_ascend/_310p/v2/staged_buffers.py`（CPU事务写计划、32B padded矩阵H2D、1D状态只更新dirty index的device `index_copy_`，避免新请求覆盖其它活跃请求的GPU采样状态）、`request_state.py`（不调用上游UVA父构造的独立RequestState候选）、`block_tables.py`（每组独立物理/Kernel blocksize、append/overwrite、padded gather、slot mapping候选）。新增 `test_v2_staged_buffers.py` 和 `test_v2_no_uva_adapters.py`；只在本R2任务分支，未改变生产入口。
- **T14/T15未闭环**：上游 `GPUModelRunner.__init__` 在Ascend替换RequestState前已构造UVA `RequestState`，后续 `initialize_kv_cache` 还构造原GPU `BlockTables`；必须以310P专属Runner完整消除这些父类构造调用，不能用构造后 `del self.req_states` 冒充替换。新增类目前未接入，故仍NO-GO。
- **T05/T06仍阻断**：Torch `index_select/searchsorted/index_copy_` 等只是Eager候选，尚未核实310P3 torch-npu支持及同步/流行为；DFlash `out/out_ptrs` 外部缓存接口被明确 `NotImplementedError` 拒绝。多组不同物理blocksize仅有CPU合同，不代表5 Sliding+1 Full后端接通。
- **发现的P0数据所有权缺口**：上游GPU采样 `post_update` 会修改设备端 `total_len/num_computed_tokens/all_token_ids`；若从陈旧CPU镜像整向量回写，会覆盖其它活跃请求。本轮修复候选1D为dirty-index scatter并补回归；矩阵只允许CPU权威的新增/替换行全行回写，活跃GPU改写行的再staging、异步in-flight可见性、异常半提交仍待端到端审查。`next_prefill_tokens`重用后的清理和310P小写入安全尚未核实。
- **CI口径**：原基线`9022f7dc`的61/61 unittest、54/54 pytest PASS；本轮新增GitHub CI运行结果须按最新SHA单独记录，CPU Torch tensor测试在无torch的轻量CI环境可能Skip，不能把Skip写成PASS；尚未执行独立310P3硬件测试或两轮运行时深审。NPUWorker310的NOT_IMPLEMENTED必须保留。
- **下一步**：冻结0.27.1的GPUModelRunner构造与所有 `RequestState/BlockTables` 访问点，设计真正不经父类UVA构造的专属Runner；对 `searchsorted/index_copy_`、请求重用、并发写顺序和DFlash out缓冲补无设备可执行gold；再逐一替代T01～09/T11、采样和ModelState。任何无法保证设备语义的操作保持fail-closed。

### T05/T06 CPU gold加强（同轮后续提交）

- 新增`_310p/v2/block_table_gold.py`和`test_v2_block_table_gold.py`：无torch必跑，覆盖两组64→32与128→128物理块展开、不同kernel blocksize的slot、请求重用overwrite、padded gather、尾部/未分配PAD、空请求、容量失败的跨组原子性及负例。Eager候选复用同一`expand_physical_block_ids`函数，减少gold与实现漂移。
- **测试层级必须区分**：以上是纯Python语义gold；`test_v2_no_uva_adapters.py`和`test_v2_staged_buffers.py`中的Torch CPU动态tensor合同在轻量GitHub CI不安装torch时会Skip；310P NPU真实算子支持和UVA构造链未解决。R2-A仍NO-GO。


### T05静态复核修复：request reuse后KV旧尾部泄漏

- 审查发现：原候选`gather_block_tables`将整个旧block row复制到batch缓存；`overwrite=True`只重置`num_blocks`和有效前缀，旧尾部仍可能被attention误读。已修改为按每组设备`num_blocks`对选中的**整条物理padded行**做`torch.where`零化，补Torch CPU reuse回归；纯Python gold此前已经要求无效尾部为0。此处仍待NPU Eager op支持核实，不可宣布T05 READY。


## 2026-10-25 R2-A 持续源码研发：T06/T14/T15/T16 P0 修正（STATIC NO-GO）

- 本轮直接修改`_310p/v2/block_tables.py`、`staged_buffers.py`、`request_state.py`及其CPU gold/torch测试，新增独立CPU Torch CI工作；非状态摘要。修复DFlash读`block_tables.slot_mappings`持久存储却未更新的错误（固定上游`DFlashSpeculator.propose`使用该属性），保持slot映射地址稳定并对padding尾部填PAD。纯Python gold允许`positions`只含实际token，测试覆盖错误长度。
- T14写失败安全性：修复旧版`plan.commit()`早于设备H2D的半提交/丢写问题；新方案只在设备写全部成功后提交CPU镜像，单组/多组异常都进入不可恢复`poisoned`状态并拒绝继续，等待Runner统一abort/销毁。多组Eager串行写**仍不是原子事务**，不能用此代替stream同步或失败回滚。
- T15/T16复用安全：新请求stage整条token行清零后覆盖前缀；清理复用slot的prefill lookahead；未提交token行拒绝释放，防止旧pending写污染新request。NPU`index_fill_`和stream/event依赖未获设备证据，`next_prefill_tokens`及活跃采样行数据所有权未闭环。
- **验证口径**：此前PR#7 HEAD`126db642` project-control CI PASS；本轮代码HEAD以实时GitHub PR为准。轻量CI通过只表示无torch合同；新增`r2-torch-cpu-contracts`在PR上执行torch2.10.0/py3.12动态tensor测试，必须实际读job结果，不能将待运行或Skip当PASS。
- **准入判定仍为R2-A STATIC NO-GO / ONDEVICE NOT_RUN**。`GPUModelRunner.__init__`原UVA构造、`initialize_kv_cache`原GPU BlockTables、T01~04/T07~09/T11采样与DFlash prepare、T13继承、GDN/Mamba/W8A8/TP2、真实stream和物理blocksize仍未完成；Worker NOT_IMPLEMENTED不变，PR#7继续DRAFT。下一批必须优先完整构造链替换，不得把候选模块接线当作完成真实Runner。


### R2-A T02/T04 输入侧源码候选（仍NO-GO）

- 本轮新增`vllm_ascend/_310p/v2/input_ops.py`，冻结上游`prepare_pos_seq_lens/expand_idx_mapping`参数；配套无torch`input_gold.py`与Torch CPU测试`test_v2_input_ops.py`。T02通过int32计算position后cast到int64，T04通过device searchsorted展开K7/混合batch；二者不执行Triton，但**尚未接入实际Ascend V2 Runner**。
- 已确认上游0.27.1`GPUModelRunner.prepare_attn`先调用BlockTables的gather/slot；`DFlashSpeculator.propose`直接读取持久`slot_mappings`，不是只用prepare_attn的返回值；当前修补是必要但非充分条件。
- **CPU CI已实际完成的旧SHA证据**：`284c7167d065d8df0076aa7826317d8244dafe2c` 的project-control **76 unittest OK（2 skipped: Torch无依赖）+54 pytest PASS**；独立Python3.12/torch2.10.0+cpu job **53 unittest OK**（Run 40164802716）。新增T02/T04后的SHA须另读最新CI，不能继承旧绿。Worker仍fail-closed，PR#7 DRAFT，设备NOT_RUN。


### R2-A 异常路径再审：BlockTables/RequestState整体poisoned

- 新发现：仅`NPUStagedWriteTensor`poisoned还不够，`NPUBlockTables`可能在某组写成功/另一组失败后仍被上游调用gather/slot；`NPURequestState.add_request`设备重置失败时CPU req-id/空闲slot已部分改变。已新增容器级`_poisoned`保护，任何多组KV/num_blocks H2D异常或请求新增/状态回写异常都阻断后续调用，补Torch CPU注入失败回归。
- 此保护要求未来Runner具备**异常后停止调度并销毁状态**的全链路实现；目前没有接线，也不等于事务原子/设备已验证。参考[ADR-013](DECISIONS.md)；**STATIC NO-GO**不变。


### R2-A 静态准入可执行门禁（持续NO-GO）

- 增加`r2_static_admission.py --check/--report`及`R2_STATIC_ADMISSION_EVIDENCE.json`空证据账，明确追踪16个Triton/UVA/ownership节点和15个ABI节点。所有必经项均需真实接线、CPU合同、设备算子静审和两轮独立review证据才可升级；`--check`在当前状态应以exit 2拒绝准入，`--report`在GitHub CI只输出真实NO-GO状态。
- T16和ABI-13~15已提升为表格显式行，不再藏在正文注释里；`test_r2_static_admission.py`覆盖缺证据、重复ID、合成完整性（合成测试不代表真实review）。当前绝不移除Worker NOT_IMPLEMENTED，PR#7仍DRAFT。


### R2-A GitHub CI复核：31项静态门禁保持NO-GO（2026-10-25）

- 真实SHA `f9a60ff5f64e3d6e1595f89c5a004490557eae0d`，GitHub Actions Run `40165197698` 两个job均**SUCCESS**：轻量治理 **82 unittest OK（3 skipped：无torch）+54 pytest PASS**；独立Python3.12/torch2.10.0+cpu **61 unittest OK**（未报告skip）。
- `r2_static_admission.py --report`在同次CI明确输出 **`R2-A STATIC-NO-GO: 0/31 mandatory nodes`**，因两轮审查尚未PASS、31项仍未真实接线/给足证据；这是按真实任务口径的正确阻断，不因CI成功改变。R2-B/C/D **ONDEVICE NOT_RUN**；Worker仍NOT_IMPLEMENTED。


### R2-A T07/T08 采样侧非Triton候选（本轮源码交付，仍STATIC NO-GO）

- 固定上游vLLM 0.27.1 @ `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`：`gpu.input_batch.get_num_sampled_and_rejected`原地清零chunked prefill的sampled/rejected；`gpu.spec_decode.rejection_sampler._flatten_sampled_kernel`只填已接受的token前缀，余量为0。
- 新增`vllm_ascend/_310p/v2/sampling_ops.py`与`sampling_gold.py`，以及`test_v2_sampling_ops.py`：Torch Eager的T07原地写回与T08无host-sync的固定长度flatten，覆盖K7、非连续row stride、混合prefill、零样本、随机差分及负例；本侧CPU Torch2.10.0 7/7 UT PASS（仅本新增测试，不是全仓CI）。
- **重要未接线事实**：上游`RejectionSampler`直接引用其模块级Triton函数和从`input_batch`导入的T07符号；新增同名候选不会自动替换。必须实现310P专属Sampler/RejectionSampler绑定，并审核`rejection_sample`自身的实际CUDA/Triton依赖及K7采样流。
- **剩余阻断**：T01/T03/T09/T11、T13及UVA父构造和多组KV真实接线，T07/T08设备`searchsorted/index_select`可用性与无短写同步证据，CPU预校验的动态count不变量和端到端ModelState所有权；T07/T08在矩阵仍BLOCKED，31项静态门禁不能晋级。Worker NOT_IMPLEMENTED、PR#7 DRAFT、ONDEVICE NOT_RUN不变。最新新增提交的GitHub CI须按新SHA复核，不继承旧成功。


### R2-A T03输入组合候选（2026-10-25；未接Runner）

- 固定上游0.27.1 `input_batch._combine_sampled_and_draft_tokens_kernel`：bonus-token 0/1、K7 draft末尾、`seq_len<=prefill_len`不覆写、prompt尾部不能被last_sampled覆盖、`logits_indices`按每请求query尾部生成。新增`_310p/v2/token_ops.py`、`token_gold.py`、`test_v2_token_ops.py`，本侧Torch2.10 CPU 5/5 UT PASS（只指新增测试）；随机混合请求/负例覆盖。
- T03使用整条`input_ids` Eager D2D copy，避免小尺寸scatter候选，但设备`searchsorted/index_select/where/copy_`及流次序、输入padding、CPU scheduler的累计长度/req-id前置校验均未审查完成；实际Ascend V2 Runner仍从上游导入Triton同名函数，**不能标READY**。
- 上一提交`3467c8f4d23120eaf4c6e073713cfdae4f43d8e4`的真实GitHub Actions Run `40169496168` **SUCCESS**：轻量治理85项unittest OK（4 skipped：无Torch）+54 pytest PASS；独立Python3.12/Torch2.10.0 CPU 68项unittest OK；门禁`0/31 STATIC-NO-GO`。T03新增提交需要单独读取CI，不能沿用该SHA结果。Worker继续NOT_IMPLEMENTED、PR#7 DRAFT、ONDEVICE NOT_RUN。


### R2-A T03/T07/T08 新增代码CI核实（2026-10-25）

- 新增T03代码提交`2ba84b8fa1daa142b679a359b1db6d525dcb0423`，GitHub Actions [Run 40169623722](https://github.com/10wvw01/vllm-ascend/actions/runs/40169623722) 两个job **SUCCESS**：轻量`87 unittest OK（5 skipped：无Torch）+54 pytest PASS`；独立Python3.12/`torch 2.10.0+cpu` **73 unittest OK**。静态准入脚本同Run输出 **`R2-A STATIC-NO-GO: 0/31 mandatory nodes`**。此成功只覆盖CPU代码合同，不是NPU、Triton调用链替代或两轮Runtime静审PASS。
- 本轮实际新增6份源码/测试：T07/T08采样、T03组合输入的Eager候选+stdlib gold+Torch CPU合同；**仍未接入实际310P V2 Runner**。PR#7继续DRAFT，Worker NOT_IMPLEMENTED不变；R2-B/C/D ONDEVICE NOT_RUN，下一步优先UVA构造链和真实Sampler/DFlash/BlockTables绑定。


### R2-A T01 prefill/lookahead Eager候选源码与合同（2026-10-25；STATIC NO-GO）

- 本轮实际新增T01非Triton Torch候选 `vllm_ascend/_310p/v2/input_ops.py:prepare_prefill_inputs`，独立`prefill_gold.py`及`test_v2_prefill_ops.py`，冻结上游0.27.1七参数和`next_prefill_tokens[lookahead, req_state]`布局。覆盖混合prefill/decode、lookahead超prefill零填充、非活跃slot不变、容量负例和随机差分；**尚未接入实际Runner**。
- 当前分支新源码提交依次为`f367bf2c`、`53bc65ef`、`3067ab12`；本条文档提交后须按**最新HEAD**读取GitHub Actions双job的实测计数。旧`2ba84b8f`的87/54/73 CPU PASS不得直接继承到新增T01代码。
- T01仍`BLOCKED`：真实Ascend V2入口继续import上游Triton；`index_copy_`/device stream/短写与UVA构造、KV和采样未闭环，CPU侧输入元数据强校验尚未生产接线。31项强制门禁仍没有真实READY证据，PR#7保持DRAFT，`NPUWorker310`的`NOT_IMPLEMENTED`不能移除，R2-B/C/D ONDEVICE NOT_RUN。

### 本次交互式续研：T09 frozen 0.27.1 PostUpdate 状态gold（仍 STATIC NO-GO）

- 实际新增 `_310p/v2/post_update_gold.py` 与 `test_v2_post_update_gold.py`：对照上游 `gpu/input_batch.py:_post_update_kernel`，覆盖有效/负数遮蔽state idx、K7部分接受/拒绝、bin_counts重复token、最后采样token、prefill/query长度增量、容量与词表越界、拒绝半提交及80组随机守恒关系；只用标准库、完全不依赖安装Triton/Torch。
- **T09状态仍BLOCKED**：CPU reference 不是NPU kernel，也未把真正`GPUModelRunner.postprocess_sampled`的上游Triton绑定换成310P3实现，仍需要真实device-safe 32B写、row ownership/stream验证与统一Runner生命周期。当前 Worker `NOT_IMPLEMENTED` 保留、实机不得准入。

### R2-03 第一阶段：安全的真正V2父构造/RequestState工厂入口（仍NO-GO）

- 在`worker/v2/model_runner.py`增加**仅当310P子类主动声明**时才调用的构造/RequestState工厂钩子，其他Ascend默认行为逐字保留。新增`_310p/v2/model_runner.py:NPUModelRunner310V2Candidate`，通过仅复制冻结上游GPUModelRunner.__init__函数的globals隔离注入`NPURequestState`、310P NPU Stream代理和无GPU draft临时初始化；**不修改共享模块全局符号、不在GPU父构造先分配UVA再删除**。
- 该入口还强制核对vLLM0.27.1、NPU TP2、PP1、Eager、V2明确打开。此候选**没有注册到Worker**；`initialize_kv_cache`仍显式拒绝，避免误进上游Triton BlockTables。后续必须将冻结`GPUModelRunner.initialize_kv_cache`的Attention/KV/Sampler实际调用全部切换到310P实现，经两轮静审才能移除故意fail-closed。

### R2-10 T09 Eager Torch状态回写候选与真实310P子类接线（仍NO-GO）

- 新增`_310p/v2/post_update_ops.py`实现冻结上游0.27.1参数+310P必要的CPU scheduler state-id / padded last-sampled存储；非Triton地更新sampled token prefix、num_computed、last_sampled、total_len、output_bin_counts（含重复token），对pad -1 行通过CPU active-row列表排除，依赖Device tensor运算，未用`.item()`。相应`test_v2_post_update_ops.py`与前一轮gold做Torch CPU差分/负例。
- 310P的`NPUModelRunner310V2Candidate.postprocess_sampled`已明确调用本地T09实现、调用ModelState postprocess并使用NPU stream更新CPU长度；不会走上游GPU Triton post_update。其他Ascend Runner保持原有逻辑不变。
- **T09仍BLOCKED**：这是针对NPU未来使用的CPU Torch Eager候选；310P3全行index_copy_、输出bin_counts /单元素状态向量、事件时序与动态sample count GPU边界仍需静审并与设备真实支持核对。且KV init仍fail-closed、Worker未接、非Triton sampler与DFlash完整链未接。CPU PASS/源码真实接线≠STATIC-GO。

### R2-04 310P专属V2输入T01～T04作用域接线（候选调用链）

- `NPUModelRunner310V2Candidate.prepare_inputs`现在真实调用冻结Ascend V2 `prepare_inputs`函数体，却用本地`_clone_upstream_function`仅在此Runner的方法全局字典中注入T01 `prepare_prefill_inputs`、T02 `prepare_pos_seq_lens`、T03 `combine_sampled_and_draft_tokens`、T04 `expand_idx_mapping`；不改变进程级Ascend/GPU模块。CPU AST合同核对四个原始Triton符号均被专属绑定。
- `BLOCKED`判定不变：GPU父构造已隔离、T01～04已在**候选Runner**作用域绑定，但该Runner没有Worker注册且KV初始化主动fail-closed；其余T05～16/采样/DFlash/设备Eager算子与流未闭环，不得从“绑定成功”推出STATIC-GO。

### R2-05 KV Cache多组对象/slot dtype候选接线（仍NO-GO）

- `NPUModelRunner310V2Candidate.initialize_kv_cache`改为冻结GPUModelRunner.initialize_kv_cache的**局部作用域clone**：动态真实KV group宽度/ Attention/backend/cudagraph_mode-none生命周期来自上游原始逻辑，但仅对该候选注入`NPUBlockTables`和`ModelAclGraphManager`，不改全局模块；Graph mode不为NONE即阻断，KV失败poison而不重试。
- `_310p/v2/block_tables.py.slot_mappings`修正为**int32**（Ascend reshape_and_cache实际ABI），同步Torch CPU和静态合同；上游GPU默认int64并不适用到Ascend。后续需要核对`_make_ptr_tensor`的uint64支持和真实NPU BlockTables schema。
- **非准入声明**：KV allocator、ModelState W8A8/GDN/Mamba、双Rank和多group draft attention尚未验证；本专属Runner仍未注册，DFlash out buffers限制仍触发显式NotImplemented；T05/T06/T14/T15依然未满足31项静态准入证据，不得更改READY/GO。

### R2-07 T11 DFlash K7六KV组 Eager候选及专属propose绑定（2026-10-25；STATIC NO-GO）

- 冻结vLLM 0.27.1`DFlashSpeculator.propose`调用真实的`prepare_dflash_inputs` 24参数签名（每个KV组循环调用一次），新增`_310p/v2/dflash_input_ops.py`实现无Triton Torch Eager构造。CPU scheduler offsets/req-index预校验、设备resident sampled/rejected、K7 anchor+7 mask、context/query独立slot、temperature/seeds、PAD清理；按每组`block_size`查物理表，不假设六组相同。
- 新增`NPUDFlashSpeculator310V2.propose`使用`inspect.unwrap(DFlashSpeculator.propose)`恢复真正的冻结函数，再用作用域函数globals克隆注入T11（原函数经`@torch.inference_mode`装饰，不unwrap会误取Torch wrapper globals）；保留Ascend per-group attention metadata context。通用Ascend V2只新增可选`_310p_v2_speculator_factory`钩子，其他设备不改变原DFlash行为；310P候选仅接受K7。
- **NO-GO继续**：Torch CPU候选不代表310P3 `torch._assert_async/index_select/gather/copy_`可用、非因果sliding/full attention真实mask正确、DFlash分组模型KV分配及T13 sampler完整链完成；还存在冻结`GPUModelRunner.sample_tokens`对7-token draft行短宽赋值的310P内存安全风险，必须修复并审查。Worker未注册、PR#7 DRAFT、ONDEVICE NOT_RUN。

### T09二次P0审查：动态计数/词表/CPU↔设备req-index写前一致性（2026-10-25）

- 修复`post_update_ops.py`原先在`sampled_count>width`、`total_len+count>capacity`时用clamp掩盖非法写，以及`sampled_token_id`超过bin_counts词表时被clamp到合法ID错误计数的风险。现在在**任何设备状态修改之前**通过`torch._assert_async`验证device/CPU req index、count/rejected、总长/计算长、接受token非负和penalty vocab；失败直接拒绝、CPU Torch负例保证状态未变。
- **设备准入仍阻断**：`torch._assert_async`在310P3 torch-npu上的可用性/错误传播/stream异步行为没有静态及实机证据；此候选不具备事务式多设备buffer提交。T09仍BLOCKED、Worker仍NOT_IMPLEMENTED、ONDEVICE NOT_RUN。

### R2-02静态调用链补漏：新增T17～T25九项必审（2026-10-25）

- 逐读`patch_v2/patch_triton.py`发现仅替换T07/T08/T09/T11远远不够：它还把采样Gumbel/温度、penalties、min_p、bad_words、logprob、结构化输出及完整rejection_sample接到**Ascend V2 Triton实现**。明确新增T17～T24，另将`GPUModelRunner.sample_tokens`K7 `draft_tokens[idx_mapping]=draft_tokens`短宽设备写登记为T25。
- P0：Ascend V2 `rejection_sampler_utils.py`在非贪心路径`u=0`强制接受Draft，缺乏概率正确性；且T25可能跨行/跨请求写。**不得以T11和T09候选CPU绿宣称采样链闭环**。
- 准入门禁从16个T+15个ABI=31扩展为**25个T+15个ABI=40个节点**；所有新增项无READY证据，R2-A继续STATIC NO-GO / ONDEVICE NOT_RUN。历史31节点CI结果只作为旧口径记录；Worker NOT_IMPLEMENTED和PR#7 DRAFT保持不变。

### T17/T18 Eager采样安全子集候选（2026-10-25；NO-GO）

- 新增`_310p/v2/gumbel_ops.py`和`test_v2_gumbel_ops.py`：T18对温度0/1保持原值、其余按FP32除法再回写整张量；T17仅提供确定性贪心`argmax`，保持冻结ABI的int64输出；请求索引设备resident校验、随机temperature>0和FP64/processed logits明确拒绝。Torch CPU合同覆盖greedy、温度与错误先拒绝。
- **不能标T17/T18 READY**：尚未接入真实Sampler，`torch._assert_async` NPU支持未知；尤其温度>0不能用贪心替代随机Gumbel/Philox分布，且T23概率rejection仍是P0。Worker继续NOT_IMPLEMENTED，R2-A STATIC NO-GO。

### T23 Greedy K7 Rejection Eager候选（2026-10-25；STATIC NO-GO）

- 新增`_310p/v2/rejection_ops.py`和`test_v2_rejection_ops.py`，固定冻结上游`rejection_sample` 14参数ABI，纯Torch Eager处理K7+bonus=8、混合长度、按下一个logits位置取draft token、首个不匹配截断并回填Target argmax、返回int64 [N,8]+int32计数；Torch CPU gold/错误测试覆盖全部接受、首步拒绝、中途拒绝、prefill短batch和异常。
- **只实现Greedy温度=0**：温度>0/FP64/synthetic/block verification均明确拒绝，不伪造概率分布。还未替代真实`RejectionSampler._verify`全局绑定，`torch._assert_async`/NPU动态索引和随机分布均未通过静审；T23仍BLOCKED，T17～T25全部保持NO-GO。

### T07/T17/T23冻结Sampler实际作用域接线（2026-10-25；STATIC NO-GO）

- 新增`_310p/v2/sampler.py`：专属`NPUSampler310V2`对冻结`Sampler.__call__`绑定T07、`Sampler.sample`绑定T17 Greedy候选，强制禁用CUDA FlashInfer；`NPURejectionSampler310V2`对冻结`RejectionSampler.__call__`绑定T07、`_verify`绑定T23 Greedy候选。若请求logprobs，拒绝未替代T08/T22 Triton。
- 专属`NPUModelRunner310V2Candidate.load_model`作用域clone冻结`GPUModelRunner.load_model`并注入这两个候选工厂；通用Ascend/上游全局方法保持不变。源码AST合同测试验证绑定而不误报为设备通过。
- **仍NO-GO**：T19～T22/24采样参数分支与T25窄写、GPU ModelState/流/Graph/TP2和NPU`torch._assert_async`未闭环；且专属Runner未注册、T23非贪心显式拒绝。静态门禁仍0/40，ONDEVICE NOT_RUN。

### T25 310P K7 7×int64 窄写的专属V2 sample_tokens替换（CPU候选，仍NO-GO）

- 直接从冻结上游`GPUModelRunner.sample_tokens`逐段移植整段执行顺序到`NPUModelRunner310V2Candidate.sample_tokens`，仅将`self.req_states.draft_tokens[idx_mapping]=draft_tokens`改成`_310p_v2_store_draft_tokens`。保留异步ModelRunnerOutput构建、postprocess、DFlash propose、KV connector顺序及`step_eplb_after`装饰器。后续按冻结源码进行逐行静态差异审计。
- 新增`_310p/v2/draft_storage.py`，对持久8宽int64底层缓冲进行完整64B行`index_copy_`，保证K7逻辑7宽view别名与跨request重用/旧padding清零；CPU scheduler去重/合法性检查及`torch._assert_async`确保GPU设备req-index一致。对应`test_v2_draft_storage.py`覆盖多请求/复用/异常。
- **尚不满足T25 READY**：新代码仍属于未注册的候选Runner，`torch._assert_async/index_copy_`与NPU短写、异步输出、device stream和GDN/Mamba/TP2需要静审/设备验证。Worker NOT_IMPLEMENTED继续保留。

### 2026-10-25 GitHub CPU双job实测核对（最新代码SHA 213816de）

- 代码提交`213816dec35ceff8cda8059cf405ef245eb73d29`对应GitHub Actions Run [40174498389](https://github.com/10wvw01/vllm-ascend/actions/runs/40174498389) **SUCCESS**：轻量治理 **122 unittest OK（含Torch缺依赖skip）+54 pytest PASS**；独立Python3.12/Torch2.10 CPU **130 unittest OK**。同一Run静态门禁明确输出 **`R2-A STATIC-NO-GO: 0/40 mandatory nodes`**。
- 本轮新增真实候选源码：T11六组K7 prepare +冻结DFlash propose作用域绑定、T09写前动态验证、T17/T18 Greedy/温度安全子集、T23 Greedy rejection及冻结Sampler/RejectionSampler工厂局部接线；另发现T17～T25九项被旧矩阵遗漏的必经Triton/所有权风险。CPU CI只证明该SHA的CPU语义，不等于NPU算子可用、随机分布正确、Target/DFlash真实Runner或双轮静审PASS。
- **最终状态仍STATIC NO-GO / ONDEVICE NOT_RUN**；`NPUWorker310`的`NOT_IMPLEMENTED`未动、PR#7仍DRAFT。下一轮优先T19～T22/T24的非Triton真实采样与概率拒绝、T25整行K7写、ModelState/attention的T13 ABI穷尽及device ops静审。

### T25拷贝冻结sample_tokens的全局名静审保护（2026-10-25）

- `cfbc04de`新T25实现拷贝冻结`GPUModelRunner.sample_tokens`后已导入Torch、AsyncOutput、ModelRunnerOutput、pcp、step_eplb_after；作者二次静审补齐原缺失的`GrammarOutput`类型注解ABI（`from __future__ import annotations`下不会在类定义时报错，但运行时`get_type_hints`会失效）。
- 新增`test_v2_runner_copied_sample_tokens_imports.py`使用AST+symtable检查拷贝方法所有全局自由名、装饰器/类型引用、并确保K7直接窄写被64B padded存储替换。**T25仍需设备写粒度/流和复用实证，不能标READY**。

### R2首轮静态深审（代码真实范围，NO-GO）

- 已出具[R2运行时第一轮静态深审](R2_STATIC_REVIEW_ROUND1_20261025.md)，审查专属V2构造/KV多Group、T01～T04/T09/T11、Greedy sampling、T25原位缓冲/持久状态；确认8项P0缺口，覆盖不等于PASS。Worker继续fail-closed、PR仍Draft、40项证据未升级。第二轮仅在完成生产代码闭环后开展。

### 第一轮静审发现并修复T13 NPU main_stream错误继承

- 冻结GPUModelRunner@0.27.1 的`main_stream`是`@cached_property`，调用`torch.cuda.current_stream(self.device)`，不属于之前克隆父构造函数的局部globals作用域，因而不能靠`_TorchNPUProxy`保证无CUDA调用。专属`NPUModelRunner310V2Candidate`新增同名NPU cached_property以显式返回`torch.npu.current_stream(self.device)`；补AST合同`test_v2_runner_npu_main_stream.py`。这修的是**一处真实继承边界**，并非完整T13结束，Worker仍fail-closed。

### 310P V2采样T18～T24真实分支阻断修复（仍NO-GO）

- 冻结vLLM Sampler@0.27.1 的`apply_sampling_params`在有温度、penalties、bad_words、logit_bias/min_p/top-p参数时可能调用未替代的Ascend Triton内核。专属`NPUSampler310V2.apply_sampling_params`增加显式greedy/默认参数前置fail-closed，非默认直接`NotImplementedError`，不能静默当贪心推理或掉入Triton；CPU AST合约守护方法接口。**这不是T18～T24实现完成**，仅阻断危险隐式调用；正常业务如需非贪心须补正确算子及概率采样。

### P0-09 采样器嵌套UVA及StructuredOutputs CUDA Stream构造（2026-10-25；NO-GO）

- 深审冻结上游 `Sampler.__init__` 发现此前仅在Runner构造时替换`RequestState`仍不够：它还创建`SamplingStates`（5个`UvaBackedTensor`）、`PenaltiesState`（3个）、`LogitBiasState`（4个+GPU StagedWrite）、`BadWordsState`（1个+GPU StagedWrite）、`LogprobTokenIdsState`（1个+GPU StagedWrite）。而原`NPUSampler310V2.__init__`使用`super().__init__`，在真实load_model时**必定再次进入UVA构造**。另`GPUModelRunner.load_model`的`StructuredOutputsWorker`即使无grammar也在构造时调用`torch.cuda.Stream()`。
- 新增`_310p/v2/sampler_state.py`：对上述五个冻结状态构造器使用局部globals clone注入`NPUBackedTensor`/`NPUStagedWriteTensor`；`NPUSampler310V2`私有克隆`Sampler.__init__`替换所有工厂并禁止FlashInfer；`GPUModelRunner.load_model`私有clone再注入`NPUStructuredOutputsWorker310V2`，无grammar时不创建CUDA stream，有grammar明确fail-closed。没有修改上游模块全局符号。
- 新增CPU真实构造替代测试和源码合同`test_v2_sampler_uva_constructor.py`。这只消除两处新发现的构造级P0，**不是T13/T15准入完成**：采样State多buffer生命周期、异步流、310P3 Device ABI、其它继承路径与ModelState仍需两轮静审，40节点证据不升级，Worker NOT_IMPLEMENTED和PR DRAFT保持。

### T14/T16 设备拥有向量的32B整行RMW修复（2026-10-25；仍NO-GO）

- `NPUStagedWriteTensor.apply_write`原vector分支对`self._storage`逐int32/int64索引`index_copy_`，会形成短粒度持久设备写，和310P3已识别的对齐风险不符。现按32字节物理行聚合待写索引，`index_select`先读取设备当前行，`torch.where`只覆盖staged位置，再`index_copy_`整行提交；同一物理行其余已被`post_update`修改的请求状态从设备读而非使用过期CPU镜像。Torch CPU覆盖跨行、同位置last-wins、int64与尾部padding。
- 仍需NPU3`index_select/where/index_copy_`的32B粒度和同流排序静态核实及设备证据；异步失败poison保护尚非事务提交。不可标T14/T16 READY，不解除Worker fail-closed。

### P0-10 冻结DraftTokensHandler的构造级CUDA Stream/Event（2026-10-25；仍NO-GO）

- 静审`GPUModelRunner.__init__`还发现`DraftTokensHandler(self.device)`：该类位于上游独立模块，`__init__`直接调用其模块全局`torch.cuda.Stream(device)`和`torch.cuda.Event(blocking=True)`。310P Runner仅对GPU父构造的`torch`变量使用NPU代理**不会传递到DraftTokensHandler模块**；此前代码会在真正Runner构造期误入CUDA。
- 新增`_310p/v2/draft_handler.py:NPUDraftTokensHandler310V2`，冻结构造函数的局部globals clone注入`DraftTokensHandler`。无grammar的DFlash K7维持上游`get_draft_tokens`全-1回退语义，不构造CUDA流/事件；grammar或形状异常写前拒绝。CPU语义+源码合同`test_v2_draft_handler_no_cuda.py`。
- 仍需完整继承路径、NPU输出异步复制、TP2及设备生命周期静审。T13/T15、Worker、40节点门禁仍NO-GO。

### P0-11 sample_tokens继承AsyncOutput隐式CUDA Event/Stream（2026-10-25；仍NO-GO）

- 冻结`vllm/v1/worker/gpu/async_utils.py:AsyncOutput.__init__`直接调用其模块级`torch.cuda.Event(blocking=True)`、`torch.cuda.set_stream`，不会被Runner构造的NPU代理覆盖。此前复制的`sample_tokens`即使`main_stream`修为NPU，仍在每次采样时必经CUDA。
- 新增`_310p/v2/async_output.py:NPUAsyncOutput310V2`，专属`model_runner.py`通过同名`AsyncOutput`局部导入绑定：在`postprocess_sampled`修改持久Buffer之前用**阻塞D2H**取得CPU独立快照，按实际K7接受计数裁剪输出；无CUDA Event/Stream、无异步overlap承诺，grammar/logprobs/EP fault显式拒绝。Torch CPU覆盖多请求/部分接受/请求buffer复用和异常。
- 后续仍需310P3 D2H同流排序与设备API支持、AsyncOutput调度所有权、NPU异常传播、真实TP2/多KV验证；这是构造/运行时P0修复候选，不等于设备准入。Worker NOT_IMPLEMENTED、R2-A STATIC NO-GO、40节点0 READY不变。

### T26～T29四项新发现的必经CUDA/UVA路径正式纳入准入门禁（2026-10-25）

- 实际深审发现`Sampler`嵌套五个UVA状态构造器（T26）、`StructuredOutputsWorker`构造CUDA Stream（T27）、`DraftTokensHandler`构造CUDA Stream/Event（T28）、`AsyncOutput`每次采样构造CUDA Event/切换CUDA Stream（T29）。这四条不属于之前T01～T25的显式矩阵节点，虽可归入T13总风险，但不得仅靠T13一个证据笼统关闭。
- `V2_TRITON_REPLACEMENT_MATRIX.md`补T26～T29；`r2_static_admission.py`的强制清单从**25T+15ABI=40**扩展为**29T+15ABI=44**；测试中未知新增ID由T26更新为T30。T26～T29目前均有局部Eager/无CUDA候选接线及CPU合同，**全部仍BLOCKED**，未有310P3算子/双轮静审证据，门禁 **0/44 STATIC NO-GO / ONDEVICE NOT_RUN**。旧40节点CI是历史口径，不得复用。

### R2-A T16请求槽回收与待提交向量写的CPU修复（2026-10-25 17:50 北京时间；NO-GO）

- 静审发现`NPURequestState.remove_request`只检查`all_token_ids`矩阵待提交写，未检查`total_len`与`num_computed_tokens`一维向量待提交写；在token row已提交但长度/计算计数仍排队时释放slot，会允许旧请求延迟写污染后续复用的请求。现先完整检查三类buffer，再原子修改两个req映射和free_indices；向量按`StagedWritePlan._pending`的offset而非row检查，保留旧token pending错误合同。
- 增加Torch CPU合同：token row已提交、两个vector分别留有未提交写时均拒绝释放，拒绝后req映射/free列表不变，清理待提交后允许释放并复用，复用后的长度/计算计数为新请求值。此为真实代码修复，CI需在新SHA核实；**不能据此认为in-flight NPU stream/异常传播/异步提交安全**，T16仍BLOCKED，44项强制静态准入证据保持0/44、PR#7 DRAFT、Worker NOT_IMPLEMENTED、ONDEVICE NOT_RUN。下一步继续审计device-owned row写入和实际GPUModelRunner继承调用链。

- 新代码SHA `710bd56e1905cf5848de38d3fdb3172fb4c4c72a` 对应 [GitHub Actions Run 40182388545](https://github.com/10wvw01/vllm-ascend/actions/runs/40182388545) 双job **SUCCESS**：轻量138 unittest OK（13 skip）+54 pytest PASS；独立Torch2.10 CPU **158 unittest OK**，较上一SHA新增1项回收/复用合同。准入检查仍明确`R2-A STATIC-NO-GO: 0/44 mandatory nodes`、双轮审查未PASS。该CPU成功不代表NPU stream和设备生命周期可用。

### R2-A T01/T16 32B Lookahead与继承释放顺序修复（2026-10-25 19:24 北京时间；STATIC NO-GO）

- 实际代码提交 `63b93afac199127931d11ae3ea625c989c2bd114`：静审发现原`next_prefill_tokens[lookahead,req_state]`的`index_fill_(1)`和T01`index_copy_(1)`只更新每个请求的int32列，存在310P3设备持久状态短写/跨行覆盖风险。现在`NPURequestState`使用按请求分配、32B向上对齐的物理存储`_next_prefill_storage[req_state,padded_lookahead]`，对外保持冻结上游`[lookahead,req_state]`转置视图；T01先拒绝非对齐布局，再通过物理整行`index_copy_(0)`提交并保留未修改请求及padding。槽位复用整行清零，补1～9步lookahead、混合请求、负例及跨请求隔离CPU测试。**这仅消除代码层面可见的4B列写，不等于NPU index_copy_、as_strided和流安全已验证**。
- 实际代码提交 `e7158aaf8b98e2a2d7c71c2a748c2e9e501c4970`：冻结`GPUModelRunner._remove_request`先调用`ModelState.remove_request`再调用`RequestState.remove_request`，原T16待写检查发生得过晚；拒绝释放时ModelState已被修改，形成双所有者分裂。现将`NPURequestState.assert_can_release_request`做成不修改状态的预检，310P候选Runner在继承释放前先调用它，之后仍保留原上游ModelState→RequestState清理顺序；CPU负例及AST顺序合同覆盖。**设备in-flight所有权、异步失败和完整ModelState生命周期仍未闭环**。
- 最终代码SHA `e7158aaf8b98e2a2d7c71c2a748c2e9e501c4970` 对应 [GitHub Actions Run 40188687250](https://github.com/10wvw01/vllm-ascend/actions/runs/40188687250) 双job **SUCCESS**：轻量 **139 unittest OK（13 skip）+54 pytest PASS**，独立Torch2.10 CPU **162 unittest OK**。真实准入检查仍为 **`R2-A STATIC-NO-GO: 0/44 mandatory nodes`**，双轮静审未PASS，`NPUWorker310`继续`NOT_IMPLEMENTED`，PR#7继续Draft，R2-B/C/D ONDEVICE NOT_RUN。
- 下轮继续T13完整继承路径、T01其它可能短写（如输入/seq_lens缓冲）、T16多Stream/错误传播、T05/T06 KV组与ModelState真实接线、DFlash K7概率正确性及310P3设备算子静审；不得以CPU绿灯消除任何44项门禁。

### R2-A T06 多KV Group Slot Mapping 32B物理行修复（2026-10-25 19:27 北京时间；NO-GO）

- 源码提交 `0e819f23fed8ad8ab346bbaaea86d5ebcc75720d`：静审发现`NPUBlockTables.slot_mappings`原按`[group,max_num_batched_tokens]`未补齐行stride，`compute_slot_mappings`通过`slot_mappings[:, :num_tokens_padded].copy_`进行局部宽度设备写；当group行宽/活动token数非8的倍数时存在310P3 4B～28B短写、相邻KV group覆盖风险。现用`_slot_storage[group,align_up(max_tokens,8)]`，保留冻结上游逻辑视图`slot_mappings[group,max_tokens]`，先构造含PAD的完整物理行再一次整缓冲copy；空批次和dummy也改为整物理存储清零/填PAD。
- 新增Torch CPU测试覆盖两组不同物理blocksize（64/128）、最大token=5/活动token=3、尾部PAD、输出别名、空批次与dummy复用。代码SHA `0e819f23fed8ad8ab346bbaaea86d5ebcc75720d` 对应 [GitHub Actions Run 40188835484](https://github.com/10wvw01/vllm-ascend/actions/runs/40188835484) 双job **SUCCESS**：轻量 **139 unittest OK（13 skip）+54 pytest PASS**；独立Torch2.10 CPU **163 unittest OK**。
- **T06/ABI-03仍BLOCKED/NO**：`slot_mappings`的非连续逻辑视图是否符合310P3 Attention/backend的stride合同、`torch.cat/copy_/searchsorted/index_select`设备支持与同流排序、每组5 Sliding+1 Full的真实mask/不同blocksize、KV allocator/ModelState和异常传播均未证实；门禁仍 **0/44 STATIC NO-GO**，PR#7仍Draft，Worker NOT_IMPLEMENTED，ONDEVICE NOT_RUN。

### R2-A T09尾部容量 masked scatter 重复列 P0（2026-10-25；STATIC NO-GO）

- 纯Torch CPU最小复现：row width=32、old_total=31、sample width=4、accepted=1时，原`scatter_`将尾部无效列clamp到31，与真实accepted列重合，后续旧值覆写合法Token（期望7，实际旧值）。这是候选T09静默错误，不代表310P3设备已运行。
- 本分支修为accepted-only delta `scatter_add_`（无效列delta=0）；空`output_bin_counts`词表从晚检查改为写前拒绝；accepted int64超出持久int32范围在写前拒绝；新增Torch CPU尾列、异常不变更状态回归。先以此次SHA对应GitHub CI核验，不能复用旧Run。
- `torch._assert_async/scatter_add_/index_copy_`的310P3支持及异步时序仍UNKNOWN；多Buffer写异常所有权尚未封闭。T09/ABI-04仍BLOCKED/NO，**0/44 STATIC NO-GO**；Worker NOT_IMPLEMENTED、PR#7 Draft、R2-B/C/D ONDEVICE NOT_RUN。
