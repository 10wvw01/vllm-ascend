# F1｜310P3 单卡双 Die 小M MM+AllReduce 固定成本：起步与准入门禁

> 2026-10-24 · **F1启动：已建独立实验分支，提交M10邻近矩阵基准及CPU校验，尚未上310P3实机；无新增性能收益结论。**  
> 分支：`perf/310p-f1-smallm-fixed-cost-20261024`，基于D4 Tile2048稳定可用的源码基线，但**F1使用现有T1 NZ-Direct小M独立测试ABI**；绝不通过改动大M V2正式Kernel冒充Decode路线。  
> 默认产品路径Stock；不在这条线开启V1长输出Serving（SEL-01 V1 0/3自然会话成功，Decode挂死未定因）。

## 一、为什么F1重要，为什么不能直接改小M通信协议？

最终目标是310P3单卡双Die的 `M×2048 × 2048×2048` FP16 NZ MM + AllReduce，目标Decode批规模`M≈10`。历史T1实机：

| M | Stock MM+HCCL（host同步计时） | V2 T1融合最优（历史） |
|---|---:|---:|
| 1 | 0.307ms | 0.330～0.357ms |
| 8 | 0.292ms | 0.324ms |
| 16 | 0.257ms | 0.322ms |
| 32 | 0.291ms | 0.325ms |

**M=10此前没有作为独立重点量测的正式证据**。不能用M8/M16线性插值得出M10结论。历史T1的短M固定通信往返+IB核间事件不回本；T2仅替换HCCL、不融合比Stock还慢，说明F1必须探索**完整MM+AR融合调度**而不是通信API机械替换。

小M计算很快，因此F1必须拆开以下成本，否则调参没有靶子：

1. 基础MM计算：当前T1可选1/2/4/8核，N-split写ND；8核并不一定比4核快。
2. 本端结果DCCI→Core0等待其余参与核（IB）→MemFabric`signal_at`。
3. 对端`wait_at`邮件、校验、放行Add核、对端缓存失效和Add。
4. `quiet_at`及DONE确认，准备下一次输出/对称池复用。
5. Host Launch、同步和设备流等待（不可在不同计时口径之间直接相减）。

**特别禁令**：不能先删`wait_at`、DCCI、DONE，也不能引入未经证明的多producer signal或缩减设备地址生命周期。若性能机制依赖数据不可见或不合法并发，收益无效。

## 二、F1-1：先真实量化M10及邻近分核

这批工具只做**不改变任何产品Kernel源码的初筛**：

- `test/perf/f1_run_smallm.sh`：直接复用已验证T1 `test/t1_nz_direct_trial.py` 的`full`模式，在同卡两Die跑 `M∈{1,8,10,16,32,64}`、`MM_CORES∈{4,8}`、`ADD_CORES∈{1,4,8}` 共36种组合，每组合30次，逐样本`status=0 + torch.equal`，包含建池前Stock MM+HCCL对照和G0 IB Probe。
- `test/perf/f1_smallm_analyze.py`：严格解析6×2×3完整表，缺M10、G0/正确性失败、NaN或重复行则拒绝给排名；显示每M最佳MM/Add组合。
- `test/ut/test_f1_smallm_contract.py`：CPU-only测试上述约束及错误示例。

这里**Stock在MemFabric建池之前测、T1在建池后测，并且含host同步**，适合作为筛选，不等同最终同场无Profiler Event严格因果评估。F1-1结论最多写`SCREEN_ONLY`，不能因为一场host结果+3%就将方案合入产品。

**执行入口**：

```bash
git switch perf/310p-f1-smallm-fixed-cost-20261024
python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_f1_smallm_contract.py
# 先现场核实实际两Die设备、CANN/MemFabric KVER11/SO SHA及卡对独占：
export ASCEND_RT_VISIBLE_DEVICES=6,7  # 示例，实际以310P3现场为准
export MF_SDMA_ORCH_JSON=/path/to/libmf_sdma_orch_v11.json
export F1_EXCLUSIVE_APPROVED=YES
bash csrc/_310P/memfabric_mm_ar_v2/test/perf/f1_run_smallm.sh /tmp/opencode/f1_screen_001
```

本批F1只能在F2设备作业完成并确认清洁退出后独占使用同一卡对；**F1研发分析可以同步，310P3实机不能与F2同时争抢同一双Die**。出现507014、首写丢失、IB异常或KVER错版立即保全失败场、不自动重试凑数据。

## 三、F1-2：用可验证的阶段分解决定正式优化内容

从F1-1的M10和其邻近量级，先验地提出三类可证伪实验，均在**诊断专用代码**中实施，不修改正式Kernel：

| 实验 | 要测什么 | 只能证明什么 |
|---|---|---|
| **F1-A：MM隔离** | T1 N-split纯MM已验证诊断核下的M10、MM_CORE=4/8；对比原生NZ MM | 核参与/Tile是否导致小M计算浪费，不能推断通信开销 |
| **F1-B：IB与协议独立探针** | G0 IB Fan、既有MemFabric通信/AR-only的信号、邮件与Add路径；采同一版本两Rank配对时间 | 看固定通信开销和长尾边界，不得把不同测试中位数作算术拆分声称就是核内每阶段耗时 |
| **F1-C：完整融合** | 新方案必须`MM+MemFabric+Add`单Kernel，M=10/邻近形状双Rank位精确、Event配对和同卡Stock HCCL A/B | 只能在相同数据、设备、SO和温度条件下说明完整融合能否打赢Stock |

取得F1-A/B阶段数据后，才允许提出如“低成本单core协调”“专用M10协议”“MM核数重新分配”的**具体一处一变量**代码更改；不能把历史有挂死的V1 Decode路径直接作为可推广方案。

**技术取舍前置**：若基本的必要通信握手时间已经接近或大于Stock完整MM+HCCL总时长，那么靠4核/8核 MM和Add微调不可能在M10稳定胜出。此时应停止F1单Kernel微调，明确是现有MemFabric通信合同的下限问题，再评估有证据的协议层优化或保持Stock小M回退。

## 四、F1 / F2 共用的真实模型硬性验收

当F1出现经多场ABBA稳定显著胜出的完整小M方案后，才能纳入真实Qwen3.6实验；**真实模型三方对比全部固定输入4096、输出2048、TP2、batch上限10、Eager**，并留存实际调度中每步活跃batch的取证。模型性能测量不得混用过去D4输出512。若F1持续发生小M/Decode挂死而无法获得性能数据，才先定向调查该路径的协议阻塞；不提早启动MemFabric长期稳定性治理。

只有满足逐请求完整输出、位精确和无异常，再比较Stock/V1/V2策略，直至明显且可重复的真实正向性能收益。Stock始终默认，实验版本不允许自动切换路由。

## 五、两轮静态审视的结论边界

- R1：**源码/正确性审查**：基于已有T1运行器，不修改正式算子及MemFabric协议。F1脚本只指定M10邻近形状、核数、重复次数。解析器强制6形状×2MM×3Add、status和完整PASS，否则fail-closed。
- R2：**时钟/推断边界审查**：Stock参考在建池前、T1融合在建池后且都含host同步，属于筛选性证据；不能由此宣称已经测出准确通信阶段耗时或最终赢Stock。F1-2若使用Profiler则仅诊断；最终用无Profiler、双Rank配对Event与同卡Stock比较。

**当前交付目标**：OpenCode先执行F2，F1同步准备；拿到F2第一轮结果后释放310P3双Die给F1做M10初筛。两线分别独立归档证据、GO/NO-GO；任何无法测出的收益写UNKNOWN。
