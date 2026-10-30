# 一轮Stock / V2 / V3固定形状测验

范围：310P3单卡双Die / TP=2；M固定2048、4096，K=N=2048，FP16 ND输入、NZ29权重、完整ND SUM输出。
本轮按用户新要求新测Stock、V2与V3各组，构成同批公平对照；V1只补适配器，历史查找/测量继续延期。
代码和计划已准备，**这里没有实机测量结果**。

## 默认完整矩阵

| 组 | 编号 | 配置 | 配置组数 | M |
|---|---|---|---:|---|
| Stock独立完整基线 | STOCK | NZ MM + HCCL SUM | 1 | 2048、4096 |
| V2所选 | V2-7 / V2-15 | E1 / T8公共入口 | 2 | E1=2048；T8=4096 |
| 原始V3 | V3-0 | group1024/2048 × serial/pipeline；packet1MiB | 4 | 两形状 |
| V3.1 | V3-1 | group512/1024/2048 × serial/pipeline | 6 | 两形状 |
| A2 prepare | V3-2 | classic，group512/1024/2048 | 3 | 两形状 |
| A2 merged-wave | V3-3 | merged，group512/1024/2048 | 3 | 两形状 |
| A3 START | V3-4 | compute512/1024 × comm512/1024/2048 | 6 | 两形状 |
| A3首READY | V3-5 | 同上 | 6 | 两形状 |
| A3-2 | V3-6 | 同上，七核计算/专职通信 | 6 | 两形状 |
| A3-tail | V3-7 | 同上，Core0末尾席位 | 6 | 两形状 |
| **合计** | | | **43** | |

除原始V3外，V3本轮packet4MiB。MM计算粒度与通信批量分别登记，不把参数点改名为新算法版本。
原始V3、A2 classic与A3 START要用匹配历史源码/工件；CPU模板只提供恢复线索，不证明这些版本已有本轮设备PASS。

每个候选/配置独立Stock→候选→候选→Stock，Stock独立组为两场Stock。
默认一轮R=1、每场W=20/N=100、ready与chain都跑、无打点性能collection=off：
**170个独立TP2会话**。每配置/每M/每场景/每采集阶段，Stock与候选各得到200个双Rank配对样本。
每个序号先max(rank0,rank1)，再算术均值；保留全部样本，另报p50/p90/p99/std和漂移。
不把43组的Stock合成一个跨配置/跨时间平均对照，不拿历史p50充当新均值。

Stock的边界是就绪输入→完整NZ MM+HCCL输出可用；融合涵盖所有每次必需动作。
chain会在算子后立即提交残差ADD并等其完成，不插入中间Host同步。冷态、路由及检查不计入热样本。

## 生成、冻结、启动

默认矩阵原件：[campaign-fixed-m.json](campaign-fixed-m.json)。适配/环境模板：
[runtime-campaign-template.json](runtime-campaign-template.json)，所有占位路径和NOT_RUN必须用实际材料替换。
推荐在服务器生成带实际仓库路径的新模板：

```bash
python csrc/_310P/tools/L1_test/l1_bench.py campaign plan --out /tmp/l1-round-plan.json
python csrc/_310P/tools/L1_test/l1_bench.py campaign template \
  --campaign /tmp/l1-round-plan.json --out /tmp/l1-round-runtime.json
# 按实际构建manifest填写runtime及逐组准入证据后：
python csrc/_310P/tools/L1_test/l1_bench.py campaign freeze \
  --campaign /tmp/l1-round-plan.json --template /tmp/l1-round-runtime.json --out /tmp/l1-round-new
python csrc/_310P/tools/L1_test/l1_bench.py campaign start --out /tmp/l1-round-new
python csrc/_310P/tools/L1_test/l1_bench.py campaign status --out /tmp/l1-round-new
```

plan/template仅CPU；freeze在任何设备执行前核对**全部组**的文件、形状、API、参数、源码身份及实际准入报告。
任何缺失会返回BLOCKED_BEFORE_DEVICE，不创建一份假装完整的运行批次，也不自动跳过旧版本。
start使用独立后台session；每个配置继承同一L1执行器/守护，异常STOP且不进入下一组，无自动ADD重试。

默认exclusive要求整批固定健康双Die卡；若明确接受共享漂移，plan时指定`--mode shared`，
结果会保留OBSERVATIONAL_SHARED标签。不得执行到一半自动改模式或换卡拼成绩。
逐个runtime填写admission_by_runtime和可取回的证据文件；不只把NOT_RUN字符串改成PASS。

## 可选范围

- `--profile active`：只跑主线当前V3.1、merged与三A3家族，加Stock/V2，共30组、118场。
  它是另一份明确缩减的计划，不是完整矩阵失败后的静默替代。
- `--v2-history`：另外补21组V2大M历史对照/参数扫描，完整矩阵变64组、254场。
  包含T0/D4/C3、F2六点、B1/B2、E1/E2/E3/E4；所选公共API与历史私有API分开绑定。
  小M T1与不完整诊断不硬塞进大M性能排名。
- `--collection host`或`profiler`：每组自动生成无打点前/诊断/无打点后三段；完整矩阵为510场。
  正式性能仍读无打点段，诊断保留扰动标签；设备内部阶段/SDMA占用缺证据则UNKNOWN。

## 输出与判读

每组保留prepared计划/manifest、run.json、所有guard/console、双Rank原始样本、路由、checkpoint和analysis。
整个批次输出summary.json、summary.csv、campaign-result.json；每行包含配置、M/场景/阶段/指标、
样本数、均值/分位数/std、该组Stock、速比、漂移、源码HEAD和原判定。
失败/未运行分别列STOP与NOT_RUN，不将不完整批次签成成功，不自动挑最快样本。

CPU适配器、调度器和假数据验证不等于310P实机。新入口/历史恢复版本都要先完成真实构建与双Rank短功能准入。
本轮不改Serving默认、不运行模型TTFT或Graph；算法研发继续聚焦V3迭代。
