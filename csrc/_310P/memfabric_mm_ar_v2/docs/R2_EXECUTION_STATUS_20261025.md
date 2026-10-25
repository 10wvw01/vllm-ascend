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
