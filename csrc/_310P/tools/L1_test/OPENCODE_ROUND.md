# 委托OpenCode：L1统一适配器固定M一轮Stock/V2/V3测验

在310P3服务器执行一轮统一L1测验，M仅2048与4096，K=N=2048、FP16、单卡双Die/TP2。
范围以主线`csrc/_310P/tools/L1_test/ROUND_FIXED_M.md`和`campaign-fixed-m.json`为准：
默认43配置组/170独立TP2会话，W20/N100/R1，ready+chain，无打点性能。
每组有自己的Stock ABBA；本轮明确重测Stock/V2以取得同批对照，V1仅准备适配，不测、不查旧数据。

## 1. 固定主线与私有环境

GitHub使用SSH：`git@github.com:10wvw01/vllm-ascend.git`，唯一研发主线`dev/310p-memfabric-mm-ar`。
先读PROJECT_310P_MEMFABRIC.md、V2 AGENTS/PROJECT_CONTROL及上述本轮说明。
保全服务器原工作区；使用隔离构建/结果目录并记录完整主线SHA/tree。
后续执行修复、数据与报告直接回主线，不创建默认实验分支、不强推、不改HaonanWu1仓库。

全批固定一张健康310P3双Die物理卡，按现场租约与G0选定并写入template的card/devices。
默认exclusive；若用户明确选shared才生成相应新计划，不在途中降级。不得杀外部租户或运行中跨卡混算。
V2/v11与V3/v28每场新TP2进程、各自SDK/LD_LIBRARY_PATH/SO/JSON，不能混装同一进程。
不替换共享安装，不绕过cache/首写/credit/IB/quiet/DONE，不启用生产Serving。

## 2. 构建与历史版本准备

先CPU合同，再按各实现原门禁私有编译、G0、双Rank短正确性/首写/状态/下游消费及装载核验。
V2编译新selected入口：M2048 E1、M4096 T8；与旧Serving/D4区别记录。
当前V3.1、merged、A3/A3-2/tail可共享同批冻结Host/Device构建与Storage适配，但各参数场仍独立进程。

原始V3-0、A2 prepare V3-2、A3 START V3-4需要恢复匹配源码与Python会话；版本登记中有historical_source_hint。
可从本仓保留Git对象提取至私有源码目录；框架仍从最新主线运行，不新建研发分支。
不能把当前SO/当前Python直接改名当旧版。保留源码HEAD/tree、源文件与实际Host/Device/Storage/SDK/KVER SHA、构建命令。
核对后才能登记historical_identity_verified。若确实无法恢复/编译/准入，列出具体阻塞并停止完整批次准备，
不擅自以active计划替换43组；不要把缺失项写成PASS。

允许修复执行/路径/构建兼容BUG，修复入主线并记录实际源码差异，重做对应构建/门禁/冻结。
不为本轮采集临时改变算法、增删同步或手工修正输出；设备异常/协议错误/错算先STOP保留现场。

## 3. 生成并填真实runtime

```bash
python csrc/_310P/tools/L1_test/l1_bench.py campaign plan --out /tmp/l1-round-plan.json
python csrc/_310P/tools/L1_test/l1_bench.py campaign template \
  --campaign /tmp/l1-round-plan.json --out /tmp/l1-round-runtime.json
```

填写每个runtime的实际源码40位SHA、Host/Device/SO、SDK/KVER JSON、route_pattern和全部files。
Stock只填原生torch_npu依赖，不附加融合环境。V3显式python_modules按所选源码填写，并纳入files。
Tensor Storage扩展亦纳入files，必须来自匹配构建。
逐runtime的admission_by_runtime填写真实PASS和原始报告文件；模板中的占位路径/NOT_RUN不能作准入。
同一配置的后端不得中途换库；固定N/W/seed，不挑点、不丢慢样本。

```bash
python csrc/_310P/tools/L1_test/l1_bench.py campaign freeze \
  --campaign /tmp/l1-round-plan.json --template /tmp/l1-round-runtime.json --out /tmp/l1-round-new
python csrc/_310P/tools/L1_test/l1_bench.py campaign start --out /tmp/l1-round-new
python csrc/_310P/tools/L1_test/l1_bench.py campaign status --out /tmp/l1-round-new
```

全量预检通过才启动，后台独立session有界运行。发现失败不重试ADD、不继续后组，保留STOP/未运行清单及双Rank原件。
不擅自增加host/profiler三段采集或旧V2历史全系列，这些会改变本轮会话数。

## 4. 回传与归档

原件、计划/manifest/SHA、G0/占用/构建/路由、全部Rank样本与summary.json/csv、campaign-result.json都归档。
大文件可制品包+SHA，但必须可取回，不能只有临时服务器路径。结果直接提交主线
`csrc/_310P/tools/L1_test/experiments/<新的唯一run_id>/`，同步PROJECT_CONTROL/PERFORMANCE_BASELINES。

先报告每组完成/STOP/NOT_RUN、实际源码/SO/SDK身份；再按M/ready/chain列完整Wall均值、p50/p90/p99、
各组独立Stock、速比、漂移。诊断段不混到正式收益；不把不同配置Stock拼成一个平均，不把失败批次写成完整PASS。
最后突出V3每次迭代与通信/计算粒度的收益或劣化，说明证据级别；不虚构设备内部阶段耗时或发布GO。
