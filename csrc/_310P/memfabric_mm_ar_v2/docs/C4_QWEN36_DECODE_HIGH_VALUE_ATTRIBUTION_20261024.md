# C4｜Qwen3.6 在 310P3 长输出 Decode 的高价值性能优化方向与取证计划

> 日期：2026-10-24 · **状态：已完成源码路径审查、提出可证伪的假设、提交离线Profiler分类工具；尚无真实 Decode 分项性能 Trace，不能把候选排序当成瓶颈事实。**  
> 代码分支：`perf/310p-c4-decode-cost-attribution-20261024`（基于 SEL-01 Stock/V2 实验的代码分支，不包含 C3 CleanLines 实验改动）  
> **项目决策**：Stock保持默认。V2 Tile2048和C3均是独立候选，C4先做归因，不提前合入正式开发分支。稳定性长稳专项暂缓，只有故障阻断本次性能测量时做最小必要处理。

## 1. 为什么本轮改变优化焦点？

用户指定的唯一正式对比：`Qwen3.6-35B-A3B-w8a8 / input=4096 / output=2048 / TP=2 / batch=10 / Eager`，50条固定随机请求、无prefix cache/Graph/DFlash，固定同一设备和运行时Host SO/KVER。

SEL-01 实测成功会话 Stock 4/4，中位45.98输出tok/s；V2 Tile2048 4/5，中位46.49输出tok/s，+1.12%，正负方向不稳定；V1 0/3自然长输出成功，Decode卡住而无法量化吞吐。**默认仍为Stock，C4只应修改和分析Stock真实Eager Decode路径，不因V2少量收益而假定路线已经选好。**

**关键的Amdahl约束**：V2当前只融合大M的 Prefill目标层；Decode小M通常回Stock。4096输入/2048输出的长输出测试中，会执行约2048轮Token生成，不能把前40个 Prefill站点的少量μs节省解释成整个模型的长期加速。之前在不同负载/设备测得的局部改善，不应与SEL-01的整模型百分比直接相乘。

参考：40层（30 GDN、10 Full Attention）、每层含MoE，256专家、每Token top8，维度2048的模型配置。**模型层次数来自官方公开模型结构；现场应以所加载 `config.json` 的SHA和具体W8A8配置为准。**

## 2. 首轮审视已找到的“每生成一个Token就会重复”的代码路径

以下**按优先测量的先后顺序**排列，**不是已经量到的耗时排行**：

| ID | 对应模型与310P代码 | Decode时怎样工作（大白话） | 具体技术风险/机会 | 先查哪项证据 |
|---|---|---|---|---|
| **C4-01 / MoE** | `vllm_ascend/_310p/fused_moe/{token_dispatcher.py,moe_mlp.py,moe_comm_method.py}` | 每层把Token分给专家，再跑两次W8A8 Grouped MatMul、SwiGLU和反向合并；40层都会碰 | `npu_moe_init_routing_v2`后把expert_tokens转int64；量化GMM走两次，专用x_scale优化仅覆盖gate/up形状`(640,2048)`和down形状`(640,256)`；batch10的真实激活行数**须实测**，不能推断一定命中640特化；是否不必要的调度/重复量化/数据转换还未知 | MoE调度、GMM1/GMM2、路由/合并的事件次数和占比；记录实际input/group_list/weight布局与是否进入量化特化 |
| **C4-02 / 30层GDN** | `vllm_ascend/_310p/ops/fla/gdn_310.py` | Decode一层先做`npu_causal_conv1d_310`，再计算门控，更新`npu_recurrent_gated_delta_rule_310`状态；每Token涉及30层 | 层内可能有多次小算子、临时Tensor、状态读写；可能受Kernel Launch与状态搬运主导；不能凭源码断定是哪一项 | 30层每步`CausalConv/GDN Gating/Recurrent`核次数、各自在trace中的排序、等待间隙、状态相关内存转换 |
| **C4-03 / TP2通信** | `vllm_ascend/_310p/ops/memfabric_mm_ar_v2_serve.py`及MoE通信实现 | Decode无法命中V2大M融合；在需要跨Rank合并时走原生`tensor_model_parallel_all_reduce`与MoE集合通信 | 可能存在很多小报文通信、同步及Kernel边界，但**每步实际通信次数还没从Trace确认**，不得凭40层推断有40或80次HCCL | `HCCL AllReduce/AllGather`真实次数、每步等待、前后Kernel的跨流关系；首选Stock，不触碰MemFabric实验协议 |
| **C4-04 / 10层Full Attention** | `vllm_ascend/_310p/attention/attention_v1.py` | Decode走`_npu_paged_attention`，随着上下文由4096延伸到6144，反复读KV Cache | 可能被带宽、BlockTable读取和动态增长长度拖慢；不同输出阶段开销可能不同 | 在输出早/中/后段同batch10下对比PA延迟和访存、KV长度、BlockTable |
| **C4-05 / W8A8与非MoE线性** | `vllm_ascend/_310p/quantization/methods/w8a8_static.py`和量化GMM | 线性层非int8输入需要`vllm.quantize`，然后调用`npu_quant_matmul`；MoE另走per-token动态量化 | Decode小M算子启动/量化成本可能显著；仅排查真实存在的重复量化与layout转换，不凭源码估收益 | `quantize / npu_quant_matmul / format_cast`次数、形状、缓存预处理 |
| **C4-06 / Host/采样/调度** | `vllm_ascend/_310p/model_runner_310p.py`与vLLM Engine/采样 | 每Token需要准备元数据、提交设备任务、执行采样，CPU与设备可能互相等 | Async Scheduling已启用；实际Host空档、Python算子提交次数和H2D同步是否主导待验 | Profiler`step_trace_time.csv`与`trace_view.json`上连续Decode步CPU提交间隙，对比整步时间而非简单求和 |

**C3的位置**：其目标是V2的Prefill`CleanLines`循环控制，虽保持DCCI语义但不会直接优化Decode。鉴于SEL-01规模，本轮先做C4成本排序。C3保留独立候选，不在任何主线Profiler或Stock基线里装载。

## 3. 关键区别：算子活动量 ≠ 整模型关键路径耗时

- `kernel_details.csv` 的`Duration(us)`跨核/跨流有并发；同类Kernel时间相加只能给出“设备工作量和出现频度”，**不能**直接当作真实Decode的独占成本、TP2步耗时或整模型提升上限。
- 应同时结合`trace_view.json`、`step_trace_time.csv`、`operator_details.csv`，识别各Token迭代的设备时间线、Host排队/等待，以及通信与计算的重叠区间。
- **每步两Rank取配对完成时间**，不能拿`max(p50(rank0),p50(rank1))`冒充真实TP2配对统计；也不能根据Profiler ON下的绝对值报告线上吞吐收益。
- 任何“候选X贡献了20%时间”都要标清是**Kernel工作量占比**还是**经实验隔离得到的关键路径贡献**。
- 三个窗口：刚结束Prefill的早段Decode、输出约1K的中段、接近输出2K的后段，至少确认每段的上下文长度与运行批次，样本被混合Prefill污染则整段作废。

## 4. D0：不改正式模型的低风险取证（OpenCode现场执行）

**正式压测口径永远不变：input4096/output2048/TP2/batch10/Eager；Profiler是独立的短窗口诊断，不用于速度排名。**

1. 独占同一对310P3设备，锁仓库SHA、模型config SHA256、W8A8权重身份、CANN版本、实际Host/plugin/SO/KVER11哈希与物理设备映射。**Stock作为第一研究对象**：关闭V1/V2 MemFabric路由，继续禁用Graph和DFlash。
2. 使用原SEL-01服务和`vllm bench serve`负载参数（50请求、固定seed、输入4096/输出2048、--max-concurrency10、`--max-num-seqs10`），但**诊断场**在`vllm serve`追加：
```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/310p_c4_profile","torch_profiler_with_stack":false}'
```
3. 只在一段**确认已经进入纯Decode且运行10个请求、每个请求本步只调度1 Token**的窗口调用`POST /start_profile`，短时间采集后`POST /stop_profile`；关闭Profiling并保留原始trace。**不要从“Running=10”单独推定“纯Decode”**，需结合该步实际`num_scheduled_tokens_per_request`和请求生命周期确认。与Prefill混合的窗口不用于Decode结果。
4. 按[官方vLLM Ascend服务Profiler指南](https://docs.vllm.ai/projects/ascend/zh-cn/main/developer_guide/performance_and_debug/service_profiling_guide.html)执行`torch_npu.profiler.profiler.analyse(...)`，保存两个Rank各自的`kernel_details.csv`、`operator_details.csv`、`step_trace_time.csv`、`trace_view.json`及输入proof JSON；截图不是必要证据，保留原文件和哈希。
5. 通过轻量CPU脚本做初步类别归因（**仅Kernel工作量排序**），随后人工审阅Timeline确定主导实际等待和关键路径：
```bash
python3 -m pytest -q tests/e2e/_310p/test_c4_decode_kernel_cost.py
python3 tests/e2e/_310p/c4_decode_kernel_cost.py \
  --proof /tmp/310p_c4_profile/decode_window_proof.json \
  --rank0 /tmp/310p_c4_profile/rank0/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --rank1 /tmp/310p_c4_profile/rank1/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --output /tmp/310p_c4_profile/c4_decode_cost.json
```
6. `decode_window_proof.json`示例（`true`**只能现场有日志/trace确认后填写**）：
```json
{
  "input_tokens": 4096,
  "output_tokens": 2048,
  "tp": 2,
  "max_num_seqs": 10,
  "max_concurrency": 10,
  "mode": "eager",
  "phase": "decode",
  "decode_phase_verified": true,
  "max_num_scheduled_tokens_per_request": 1,
  "running_requests": 10,
  "source_trace": "原始时间线文件路径与SHA256",
  "source_server_log": "实际batch及请求阶段证据文件路径与SHA256",
  "rank_count": 2,
  "failed_requests": 0
}
```
7. 将C4三窗口的每Rank Kernel工作量排名、`step_trace`配对时延、Host间隙、Top20实际算子、所有`UNCLASSIFIED`名称和反例列入报告。若缺batch10纯Decode证明则判`NO-GO/INSUFFICIENT_PHASE_PROOF`，不得把Prefill或混合样本归到Decode。

注意：若三窗口采集对全场2048输出产生干扰，Profiler必须独立于正式吞吐ABBA，诊断证据只用于选热点；选择出候选后在**无Profiler**下、固定4096/2048/TP2/batch10重新比较。

## 5. 根据D0结果的研发决策：先证据，后改正式Kernel

| 证据显示的主导项目 | 下一项最小可落地的动作 | GO / NO-GO |
|---|---|---|
| **MoE GMM/路由**占主要设备工作量且出现在每步关键路径 | 先验证量化GMM`(640,...)`特化在batch10真正是否命中；检查`expert_tokens.to(int64)`转换成本、两次W8A8 GMM输出布局、动态quant是否重复；**不能直接放宽特殊形状的算子安全guard** | 修改前必须有真实shape与Kernel时间线；最小单变量代码变更+位精确+4096/2048整模型>=5% |
| **GDN/卷积/状态**主导 | 聚焦30层Decode里卷积→门控→循环状态的小Kernel拼接，逐步查可消除的layout转换或多余启动 | 必须验证state/首调/长输出位精确；不改变量生命周期或静默状态 |
| **TP2小通信**主导 | 在原生Stock路径做HCCL calls/等待分析，尊重缓存与异步交付，考虑合法且低风险的MM+AR协同或缩减调用次数 | **不能为了性能直接替换成当前有P0的MemFabric Decode路径**，必须独立隔离、具备通信安全合同 |
| **PagedAttention**随上下文增长显著增大 | 先观察KV长度、BlockTable、Paged Attention对早/中/后段斜率；再考虑数据布局或被支持的优化接口 | 限定310P3/CANN实际支持，不照搬910B/Graph优化 |
| **Host空档/提交**主导 | 控制台和worker线程剖析、消除多余Python/CPU同步与小Kernel Launch开销 | 不允许直接把Graph/投机解码测试与用户要求的Eager三臂结果混在一起 |

**停止规则**：如果Top类别不含可信的计算/通信关键路径或主要时间是Host空档，暂不进行C3及其他盲调；如果某候选单层收益不错但整模型改善低于噪声或不稳定，直接记录NO-GO；若Gate E阻断性能采集，仅先排除测试阻塞，保留Stock默认。

## 6. 对项目主账的影响

- **新任务编号C4-D0**：Decode成本归因；状态`SOURCE_READY / DEVICE_NOT_RUN`，**不新增“已完成取证”数量**。
- 原C3 CleanLines实验继续归档候选但暂不进入长模型对照。
- `SEL-01`三方因V1挂死无法完整选择，Stock仍是默认；C4先从Stock入手能避免引入MemFabric诊断变量。
- 真正的“明显正向”判据仍是同场4096/2048、TP2、batch10 Eager中连续多次**≥5%真实输出吞吐提升且TTFT/TPOT无明显倒退**；是否达到，等D0定位后再进行受控正式优化。
