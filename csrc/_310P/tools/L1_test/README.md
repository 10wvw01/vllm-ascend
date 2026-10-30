# 统一L1独立Stock、版本对照与性能采集框架

入口：`python csrc/_310P/tools/L1_test/l1_bench.py`。本框架不依赖V3测试入口启动，Stock可独立运行；复用现有只读占卡守护。所有融合实现只放在薄适配层，计划、计时、正确性、统计及采集共用。新工具当前仅CPU验证，CANN/NPU NOT_RUN。

## 统一输入输出及两种场景

- 固定310P3单卡双Die、TP=2、Eager；FP16 ND `X[M,2048]`、已物化NZ29 `W[2048,2048]`；输出为两Rank局部FP16线性层结果之SUM，FP16 ND `[M,2048]`。
- `ready`：调用前开始，完整输出及所属流依赖完成后结束。Stock包含原生NZ `F.linear`和HCCL SUM；融合包含所有每次必要的reserve/wrap、padding、MM、seed、通信、拷贝、完成事件等成本。输出之后实际执行残差ADD，检查下游可读，但该ADD在ready窗口外。
- `chain`：相同开始点，算子返回后立即在同一流执行真实FP16残差ADD，等下游结果完成才结束。算子与下游之间不增加stream synchronize；禁止用Host读回/重排修复输出来伪装下游可用。
- 每个样本输出均检查形状、FP16、ND/连续、位精确，以及残差ADD正确性。CPU参考先按本地FP16 MM，再Gloo FP32求两Rank和后转回FP16；参考与检查都在计时区外。
- 输入生成、NZ权重转换、持久池初始化和首写保护在热态窗口外；`cold_setup_us`另报适配器建池/首调/路由的准备时间，不冒充完整应用冷启动。池租约保留到下游结束，输出释放在窗口后。
- 不把持久预分配成本强加给使用池的实现，也不把每次必需成本移到预热期。非齐整M只有实现实际支持且spec准入才运行；padding若属于算子必须在run内部，不用错误形状凑样本。

## 预热、次数与平均公式

先有双地址冷首调及每种M独立路由审计，再对每个场景/形状执行**恰好W次预热＋N次测量**，所有臂相同W/N/R。默认W=20、N=100、R=2。每个臂启动独立TP2进程，Stock只建HCCL+CPU Gloo，融合只建CPU Gloo并使用自己的冻结SDK；V2/v11与V3/v28不能同进程混装。

每轮顺序为Stock→候选→候选→Stock（ABBA）；独立Stock则每轮两个Stock会话。每次样本i先同下标配对两个Rank，再统计：

\[
t_i=\max(t_{0,i},t_{1,i}),\qquad \bar t=\frac{1}{n}\sum_{i=0}^{n-1}t_i.
\]

每个形状/场景/采集阶段的有效n为`2×R×N`，Stock和候选一致；所有N样本保留，不删慢样本，不把先求Rank均值/中位数再max当作配对均值。主指标Wall覆盖调用、统一事件记录及最后流同步；Event为辅助，不能由Wall−Event反推出Host准备时间。另报p50/p90/p99、总体标准差、最小/最大、速比`mean_stock/mean_candidate`、时延缩短`1-mean_candidate/mean_stock`。

`operator_event_us`仍是完整算子事件窗口，不是MM本体；`consumer_event_us`是残差ADD的事件窗口。ready与chain各自比较，不把两者差额当作纯融合代价。

## 独占与共享

`exclusive`必须由现场落实覆盖整批的双Die租约；每次新进程之前、运行中及结束由既有stage_guard采样占用/健康，异常只终止自有进程组。守护不是硬件排他锁，PID命名空间及子会话间空隙限制保留。

`shared`容忍外部进程和漂移，保留原始PID/占用记录及全部样本，输出`OBSERVATIONAL_SHARED`，不因噪声挑数据或自动改为独占结论。健康异常、解析失败、协议/数值错误仍STOP；无自动ADD重试、换卡、部署四卡或杀租户。

独占性能标签还核查Stock夹逼、候选重复、跨轮及无打点前后漂移（默认5%）。超过则`INCONCLUSIVE_DRIFT`；结果执行成功不代表发布GO。

## 共用采集，不为每版另写统计脚本

`--collection off`用于无打点性能；`host`或`profiler`会生成无打点前→诊断→无打点后的完整ABBA序列，诊断相对前后基线报告扰动。各阶段样本数一致，每个会话重新预热。

| 量 | 公共采集方法 | 解释边界 |
|---|---|---|
| 完整算子、下游链 | Wall＋Event＋逐样本双Rank原始JSON | 公平比较的主数据 |
| Host调用总段 | 仅host阶段对adapter.run计时 | 含可能的Host阻塞；不是纯CPU指令时间 |
| acquire/wrap/enqueue等 | A2/A3现有phase_log；其他版本无此能力则缺失 | Host阶段可重叠设备工作；不与Kernel时长相加 |
| 原生MM/HCCL或融合Kernel | 每种M独立窗口外路由审计；可选正式样本Profiler | 必须真实命中；不能用一个M的路由证明另一个M |
| MAC/Scalar/MTE/Vector | Profiler Level1 PipeUtilization原始CSV | 管线重叠，不构成互斥阶段占比；Scalar不能直接等于wait |
| MM/seed/本地等待/peer等待/ADD提交/排空内部墙钟 | 统一schema预留，现有设备时钟无效则UNKNOWN | 310P3现有AICore周期读取失败；不返回0或伪造耗时 |
| SDMA真实链路占用 | UNKNOWN，另需可用硬件链路证据 | 提交时间、Kernel总时间都不能替代 |

仅少量版本适配负责调用本版ABI、状态读取与清理。历史实现从准确快照恢复，`external`版本提供冻结Python工厂`create_adapter(ctx, specification)`，返回对象包含`version`、`run(x,w,variant)->Tensor`、`after_sample()->dict`、`close()`。after_sample的状态审计不能替代run必须保证的输出完成依赖。工厂不能提供Stock回退；源码HEAD、适配器及所有SO/JSON都须入files，旧版本未准备好则阻止执行，绝不用最新版冒充。

Graph、MM shadow、G1C/G1D和LOOP4是独立诊断类别，拒绝混入Eager完整候选。参数扫描逐次生成不同冻结manifest/结果目录，保留配置，不挑最优点拼臂。

## 操作

创建独立Stock计划（无需任何融合版本）：

```bash
python csrc/_310P/tools/L1_test/l1_bench.py plan --M 2048 4096 \
  --warmup 20 --samples 100 --rounds 2 --mode exclusive --out /tmp/stock-plan.json
```

对照示例；共享模式换为`--mode shared`，归因加入`--collection host`或`profiler`：

```bash
python csrc/_310P/tools/L1_test/l1_bench.py plan --versions V2-1 V3-5 V3-6 V3-7 \
  --warmup 20 --samples 100 --rounds 2 --out /tmp/compare-plan.json
```

复制`runtime-template.json`填实际绝对路径、真实40位源码SHA、全部相关Host/Device SO/JSON/Python、已准入M及实际Kernel正则。Stock-only删除候选runtimes即可。先按各实现现有准入流程完成G0只读→CPU合同→私有编译→双Rank短Smoke/首写/状态/物理号→二进制装载核验，把真实门禁报告文件入admission并登记PASS；不能只改模板状态字符串。

```bash
python csrc/_310P/tools/L1_test/l1_bench.py freeze --template /tmp/actual-runtime.json --out /tmp/frozen-runtime.json
python csrc/_310P/tools/L1_test/l1_bench.py start --plan /tmp/compare-plan.json \
  --manifest /tmp/frozen-runtime.json --out /tmp/l1-new-attempt
python csrc/_310P/tools/L1_test/l1_bench.py status --out /tmp/l1-new-attempt
```

freeze只计算明确声明文件的SHA，不证明二进制可复现、自动审核G0或部署设备。框架代码、版本表和框架HEAD也冻结，执行前后重核工件；每次复测用全新目录，拒绝覆盖，后台独立session不受交互工具超时直接中止。

输出：run.json、launch.json、各作业console/guard日志、每形状每场景checkpoint、双Rank完整JSON、路由CSV/SHA、可选Profiler原件、paired.json、analysis.json、result.json。失败保留stop_rank与现场；不继续后续臂。结果不含模型TTFT/Graph验收。服务器仍使用SSH链接GitHub，私有构建，不覆盖共享安装。

旧批次可离线导入，保留其原合同和p50，不伪装成新均值：

```bash
python csrc/_310P/tools/L1_test/l1_bench.py import-legacy \
  --root csrc/_310P/memfabric_mm_ar_v3/experiments/a3-unified-shared-card-20261029-r1 \
  --out /tmp/legacy-evidence.json
```

版本索引分别位于V1/V2/V3工程的`docs/VERSION_INDEX.md`。35份统一版本文档涵盖使用、输入输出、流程图、步骤说明、耗时证据与未知项。

## 2026-10-30主线执行规则

统一框架已按用户要求直接合入`dev/310p-memfabric-mm-ar`。后续实机从主线取完整SHA，修复、脚本、原始数据和结论直接回主线，不再以工作分支或PR合入为测试前置。每轮新建隔离构建/结果目录，固定源码与SO/KVER/物理映射；并发主线有更新先核对，不强推。合入不等于设备GO，仍按G0→CPU→私有编译/短Smoke→正式采集推进。

## 目录与调用方式

框架模块、CLI、版本表、冻结模板、文档/证据生成器与CPU测试统一位于`csrc/_310P/tools/L1_test/`。
直接入口可从任意工作目录执行（脚本路径使用绝对路径即可）；Rank worker通过`L1_test.worker`模块启动，子进程的PYTHONPATH由调度器设置为仓库`csrc/_310P/tools`。
迁移后请按新主线SHA重新freeze运行manifest，记录新路径与文件SHA。

```bash
python csrc/_310P/tools/L1_test/l1_bench.py --help
python -m pytest -q csrc/_310P/tools/L1_test --import-mode=importlib
PYTHONPATH=csrc/_310P/tools python -m L1_test.build_docs
PYTHONPATH=csrc/_310P/tools python -m L1_test.evidence
```


## V2所选公共C API

V2-7/M2048与V2-15/M4096可用runtime的`api: "v2_selected_v1"`调用新公共入口，
模板为[runtime-v2-selected-template.json](runtime-v2-selected-template.json)。每份计划只含一个匹配M；
历史私有ABI仍保留，未指定api时不替换。必须冻结`library`及实际装载的`selected_device_library`和全部SDK依赖。
建池前固定输出分配，完整计时/下游消费与统一采集合同不变。
[公共接口与状态](../../memfabric_mm_ar_v2/docs/SELECTED_API.md)：新入口只完成CPU验证，CANN/NPU为NOT_RUN；
模板不是准入凭证，不自动运行设备任务。


## Stock/V1/V2/V3简易适配与一轮固定M批次

适配器按家族位于`adapter_plugins/stock.py`、`v1.py`、`v2.py`、`v3.py`；公共工厂仍是adapters.create。
[接入合同](ADAPTER_GUIDE.md)、[本轮完整矩阵与命令](ROUND_FIXED_M.md)、[OpenCode执行单](OPENCODE_ROUND.md)。
默认Stock+所选V2+V3八种迭代，M=2048/4096，43配置组/170独立TP2会话，W20/N100/R1，ready+chain。
每配置独立Stock ABBA与200配对样本，输出统一summary.json/csv及各组原始证据。
V1只补适配不测；旧V3明确冻结Python会话和SO，缺任何组在设备前整批阻止，不能新代码代旧实验。
可用`l1_bench.py campaign plan/template/freeze/start/status`，[默认CPU计划](campaign-fixed-m.json)已提交。
[本轮复核](REVIEW_FIXED_M.md)：475项CPU及36子测试通过，真实设备测验尚NOT_RUN。
