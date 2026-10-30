# 统一L1 Stock对照与性能采集

唯一入口：`python csrc/_310P/tools/L1_test/l1_bench.py`。
310P3单卡双Die/TP2，M=2048/4096，K=N=2048，FP16 ND输入、NZ29权重、完整ND SUM输出。
完整合同见[MEASUREMENT_V4.md](MEASUREMENT_V4.md)，适配见[ADAPTER_GUIDE.md](ADAPTER_GUIDE.md)。
代码/CPU验证已完成，**新的公共Tensor入口及V4测量的CANN/NPU仍NOT_RUN**。

## 以Stock为基准的真实调用层

Stock独立进程调用原生`F.linear(X,W_NZ)`再HCCL SUM；融合独立进程调用
`torch.ops.memfabric310p.mm_ar(X,W,context) -> Tensor Y`。
两者都从就绪Tensor输入开始，直到完整输出及所属Stream完成结束。
V2/V3的每次输出获取、Storage包装、租约、输入异步保活都在融合算子代码中完成，计入调用。
L1薄适配器不再预分配/预包装输出。持久池可在冷态初始化，实际每次成本不可外移。
公开入口代码：[共同入口](../../common/torch_tensor_entry.py)、
[V2实现](../../memfabric_mm_ar_v2/src/torch_adapter/tensor_entry.py)、
[V3实现](../../memfabric_mm_ar_v3/src/torch_adapter/tensor_entry.py)。
原C ABI和冻结会话作为实现层保留，Serving默认不变，模型调用点集成尚未验证。

## 三种相互独立的证据

| 路径 | 操作与边界 | 结论 |
|---|---|---|
| 功能 | 前后立即同Stream ADD；冷态连续提交保留每个Y，检查数值/格式/别名 | 下游依赖与输出生命周期；不计时 |
| ready | 每次完整MM+AR，逐调用末尾等待 | 单次接口延迟 |
| steady | B次连续提交，保留全部输出，只在组尾等待 | 有界组平均时间，不是单次尾延迟或完整模型吞吐 |

默认同时采ready与steady，B=2、输入工作集4组。每模式恰好W次预热、N次测量调用，
不因计时单位不同增减调用；默认W20/N100。尾组按实际调用数处理。
组间只做协议审计/地址检查与释放；没有逐样本CPU barrier或逐样本torch.equal。
数值检查在功能门禁及阶段最后一组；原始报告明确该范围，不标每个样本correct。
下游ADD不进入热态计时或性能Profiler。

每臂/每M/每模式/每阶段共2×R×N次调用；ready计时单位同调用数，steady单位2×R×ceil(N/B)。
对每个单位先取双Rank**总时长max**，均值为总时长之和/总调用数；尾组正确加权。
保留原始Wall/Event、不删慢样本；分位数显式标注单次或块平均范围。
Wall包含Host提交、Event与末尾等待；Event是Stream区间，不能称纯Kernel时间。
空计时校准只报告、不减除；预热收敛、开销比例、工作集字节数都输出。

每配置独立Stock ABBA；R的奇数轮为BAAB，避免固定顺序。独立Stock每轮两场。
R1快速/独占轮都是筛选；R>=3才对独立轮次均值做配对bootstrap区间。
收益判断还要求exclusive、漂移可接受、预热收敛和计时开销<=5%；不把5%漂移门槛当成1%增益证明。
模型TTFT、长稳与Graph仍须各自验证，L1永不自动签发布GO。

## 输出容量与生命周期

A2/A3保留输出上限2；A2原内部同步如实计入，即使steady也不表示Host无阻塞。
raw V2/V3的内部输出组在建池前分配，受首写地址表上限约束。
V1/unknown external目前只允许B=1，超容量在设备前阻止，不静默缩窗。
Y及所有Storage别名都释放才归还槽；未完成输入由事件保活。仅支持原Stream、Eager调用。
发生不确定异步故障保留池与输入并STOP，不重试ADD、不回退Stock。

## 操作

```bash
# 独立Stock，无融合库/环境
python csrc/_310P/tools/L1_test/l1_bench.py plan --M 2048 4096 \
  --mode shared --warmup 2 --samples 10 --rounds 1 --out /tmp/stock-plan.json
# 当前V3定点迭代：显式选择，不静默替换完整矩阵
python csrc/_310P/tools/L1_test/l1_bench.py campaign plan \
  --cases v3-5-c512-g1024 v3-6-c512-g1024 v3-7-c512-g1024 \
  --mode shared --warmup 2 --samples 10 --rounds 1 --out /tmp/v3-plan.json
```

完整43配置组/170独立TP2会话见[ROUND_FIXED_M.md](ROUND_FIXED_M.md)，
新合同`L1_TENSOR_IO_V4`、campaign schema V3须重新生成/freeze，不能重标旧结果。
V1仅准备，不测；所选V2是M2048 E1/V2-7、M4096 T8/V2-15，仍保持同批Stock对照。
各版本及历史恢复身份缺失就阻止所选批次，不用最新版冒充历史。

```bash
# 新桥与MemFabric SDK无依赖；可以V2/V3共用；只在设备服务器实际执行编译
python csrc/_310P/common/build_tensor_bridge.py \
  --out build/310p_l1/<run_id>/tensor_bridge --cann /actual/cann
# 生成实际路径模板，填写完整工件/证据，不能只改NOT_RUN为PASS
python csrc/_310P/tools/L1_test/l1_bench.py campaign template \
  --campaign /tmp/v3-plan.json --out /tmp/runtime.json
python csrc/_310P/tools/L1_test/l1_bench.py campaign freeze \
  --campaign /tmp/v3-plan.json --template /tmp/runtime.json --out /tmp/fresh-run
python csrc/_310P/tools/L1_test/l1_bench.py campaign start --out /tmp/fresh-run
python csrc/_310P/tools/L1_test/l1_bench.py campaign status --out /tmp/fresh-run
```

准入顺序仍是G0只读→CPU→私有编译→双Rank短Smoke→身份装载核验；
短Smoke须包含新Tensor入口、持有两个输出及其别名、输入保活和立即下游消费。
冻结公共Tensor Python源、Host/Device/Storage/SDK/SO/JSON和实际Kernel路由。
freeze仅计算身份，不替代实际编译和门禁证据。执行前后重核，首错STOP。

沿用服务器原有vllm-ascend与memfabric_hybrid两仓，安全原地切主线/锁定SDK分支，不新clone/worktree。
build/310p_l1/<run_id>和L1_test/experiments/<run_id>是原仓内部子目录，拒绝覆盖旧结果。
MemFabric V2/V3同源码分支；SDK兼容须分别构建/短测，不把旧SO换JSON当作证明。
详细执行及SSH/仓库约束见[OPENCODE_ROUND.md](OPENCODE_ROUND.md)。
快速shared W2/N10/R1接受占用/漂移，仅OBSERVATIONAL；不自动追加exclusive/Profiler。
正式独占需现场整批租约；守护只是占用/健康监控，不能证明硬件排他锁。

## 共用采集与结果

`--collection off`测性能；host/profiler/msprof生成off-before→诊断→off-after，各臂次数相同。
Torch Profiler只包执行组；独立msprof采整个应用，统一分析用正式组标记排除冷态/预热/检查。Host完整调用和A2/A3细分阶段均保留；
旧版本缺细分为UNKNOWN。MAC/Scalar/MTE/Vector管线计数可重叠，不能相加凑100%，
Scalar不等于wait。设备MM/等待/SDMA精确分段无可靠时钟时明确UNKNOWN。

输出包括冻结计划/manifest、双Rank样本、checkpoint、路由/Profiler原件与SHA、guard/console、
analysis、summary.json/csv、STOP/NOT_RUN清单。每配置保留独立Stock，
报告calls、timing units、分位数范围、执行边界、内部同步、预热/计时质量、轮次区间及benefit_verdict。
历史导入保留原p50和原合同，不转写成新均值。

```bash
python -m pytest -q csrc/_310P/tools/L1_test --import-mode=importlib
PYTHONPATH=csrc/_310P/tools python -m L1_test.build_docs
PYTHONPATH=csrc/_310P/tools python -m L1_test.evidence
```

CPU测试包括真实CPU Torch dispatcher、假会话/生命周期、统计负例；CPU环境需PyTorch。
35份VN-M文档位于各版本docs/versions，入口为各`docs/VERSION_INDEX.md`。


## 官方msprof与统一时间线

已补`--collection msprof`、`msprof-probe`、`freeze --msprof-tool`及`timeline`离线入口。
每Rank CLI互斥于Torch Profiler，沿用同一TP2守护和off-before/diagnostic/off-after；
工具help/版本/明确依赖SHA及实际选项冻结，不直接套另一CANN版本的参数。
两种入口都按正式组标记解析官方Chrome时间线，输出Host→Device/队列关联、任务并集、
Stream可见空档和计算/通信任务包络交叠；按显式flow ID关联，缺数据UNKNOWN，不时间最近邻配对。
可见空档不等于硬件空闲，通信包络可能包含等待，融合核内部MM/SDMA交叠和真实链路占用仍不能从总时长推算。
跨窗口持久AICPU任务单列，不把进程寿命当一次算子耗时。原始数据及应用报告保留SHA，不覆盖。
详细命令、证据解释与官方来源见[PROFILING.md](PROFILING.md)。当前新CLI/310P采集仍NOT_RUN，CPU通过不替代现场短测。
