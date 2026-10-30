# 委托OpenCode：L1统一适配器固定M一轮Stock/V2/V3测验

在310P3服务器执行一轮统一L1测验，M仅2048与4096，K=N=2048、FP16、单卡双Die/TP2。
范围以主线`csrc/_310P/tools/L1_test/ROUND_FIXED_M.md`和`campaign-fixed-m.json`为准：
完整范围43配置组/170独立TP2会话，ready+chain，无打点性能。
**用户最新执行规则：本轮先非独占快速验证，shared、W2/N10/R1；正式性能另用exclusive、W20/N100/R1。**
两种模式明确生成不同计划/结果，不自动追加独占轮；快速轮不减实验组，只减少预热和样本。
每组有自己的Stock ABBA；本轮明确重测Stock/V2以取得同批对照，V1仅准备适配，不测、不查旧数据。

## 1. 沿用两个现有仓库，不再扩散目录

GitHub使用SSH：`git@github.com:10wvw01/vllm-ascend.git`，唯一研发主线`dev/310p-memfabric-mm-ar`。
先读PROJECT_310P_MEMFABRIC.md、V2 AGENTS/PROJECT_CONTROL及上述本轮说明。
直接沿用服务器原有`vllm-ascend`目录，切换/快进到上述主线；不再clone或新增worktree。
MemFabric也沿用原有`memfabric_hybrid`目录；V2/V3统一使用已有源码分支
`fix/310p-v3-orch-deploy-helper-20261029`的`db38d75e8485d31451302219eb7a1bfeb1dd107c`，
不能拿移动分支的新HEAD替代。两仓均先核工作区及运行任务：干净且不被任务依赖才原地切换；
有未提交内容或运行任务依赖则保留现场并说明阻塞，不reset/clean、不另起仓库绕过。

该分支修复公共SDMA编排器的建池部署：把探测、CUST迁移、复制和校验整体放到fork+exec的
orch_deploy_helper中，隔离torch_npu宿主上下文的KFC故障；助手缺失/失败明确报错。
db38d75e本次差异未改ADD指令、KVER28设备核或其通信合同，分支名中的v3不要求V2另用源码分支。
模板中SDK_V11/SDK_V28是历史工件路径示例，不是“V2只能v11”的接口限制；可填同一套已通过验证的SDK。
V2改用新SDK时必须匹配头文件重建Host/Device并做双Rank短测；不把旧v11二进制直接换JSON当作兼容。
共用源码分支不等于已证明所有旧SO与新SDK二进制兼容，旧成绩继续保留原SDK身份。

只新增仓库内部子目录，记录两仓完整SHA/tree：

- 构建/SDK暂存/必要的历史源码快照统一放`vllm-ascend/build/310p_l1/<run_id>/`。
- 计划、runtime、证据和结果统一放`vllm-ascend/csrc/_310P/tools/L1_test/experiments/<run_id>/`。

这两个是原仓内子目录，不是新的仓库。每次run_id唯一，保留上一轮结果；不自动删除现有旧目录。
优先只读复用已冻结且SHA/SDK/KVER匹配的SDK和工件，不为本轮重新复制全部仓库。
如确需重建，临时源码/构建产物也收在统一build子目录；旧prepare_sdk要求fresh源码，
不得直接用会清build/output的上游脚本覆盖原MemFabric仓库。需要时修复构建路径兼容并回主线，
或报告具体阻塞，不照搬旧执行单的新顶层clone步骤。
后续执行修复、数据与报告直接回主线，不创建默认实验分支、不强推、不改HaonanWu1仓库。

全批固定一张健康、内存足够的310P3双Die物理卡，按当前G0选定并写入template的card/devices。
快速轮使用shared：允许外部进程和性能漂移，不因目标卡非空闲而拒绝；仍核健康、物理映射、
SDK身份、协议与数值，不驱赶其他任务。结果标OBSERVATIONAL_SHARED，耗时只作观察。
正式性能使用exclusive：要求整批独占窗口，保留占用与漂移判定。不得运行中改模式、跨卡混算或杀租户。
V2/V3每场新TP2进程，冻结实际SDK/LD_LIBRARY_PATH/SO/JSON；如复现时确需不同SDK，按进程隔离，不能混装。
不替换共享安装，不绕过cache/首写/credit/IB/quiet/DONE，不启用生产Serving。

## 2. 构建与历史版本准备

先CPU合同，再按各实现原门禁私有编译、G0、双Rank短正确性/首写/状态/下游消费及装载核验。
G0保留占用原件，但shared不要求空闲；短Smoke也要采用共享守护。
V3旧quick.py已有`--allow-shared-card`，使用该入口时显式传入并写明lease-note；
不得套用旧独占Smoke/空闲卡判据来阻止本轮shared，也不得伪填准入PASS。
V2编译新selected入口：M2048 E1、M4096 T8；与旧Serving/D4区别记录。
当前V3.1、merged、A3/A3-2/tail可共享同批冻结Host/Device构建与Storage适配，但各参数场仍独立进程。

原始V3-0、A2 prepare V3-2、A3 START V3-4需要恢复匹配源码与Python会话；版本登记中有historical_source_hint。
可从本仓保留Git对象提取至统一build下的历史快照子目录；框架仍从主线运行，
不为每个版本新建仓库/worktree，也不反复把主源码目录切到旧实验分支。
不能把当前SO/当前Python直接改名当旧版。保留源码HEAD/tree、源文件与实际Host/Device/Storage/SDK/KVER SHA、构建命令。
核对后才能登记historical_identity_verified。若确实无法恢复/编译/准入，列出具体阻塞并停止完整批次准备，
不擅自以active计划替换43组；不要把缺失项写成PASS。

允许修复执行/路径/构建兼容BUG，修复入主线并记录实际源码差异，重做对应构建/门禁/冻结。
不为本轮采集临时改变算法、增删同步或手工修正输出；设备异常/协议错误/错算先STOP保留现场。

## 3. 生成并填真实runtime

```bash
# 从现有vllm-ascend根目录运行；run_id示例，已有同名目录就换新编号。
RUN_DIR=csrc/_310P/tools/L1_test/experiments/l1-quick-shared-20261030-001
mkdir -p "$RUN_DIR"
python csrc/_310P/tools/L1_test/l1_bench.py campaign plan \
  --mode shared --warmup 2 --samples 10 --rounds 1 --collection off --out "$RUN_DIR/plan.json"
python csrc/_310P/tools/L1_test/l1_bench.py campaign template \
  --campaign "$RUN_DIR/plan.json" --out "$RUN_DIR/runtime.json"
```

填写每个runtime的实际源码40位SHA、Host/Device/SO、SDK/KVER JSON、route_pattern和全部files。
Stock只填原生torch_npu依赖，不附加融合环境。V3显式python_modules按所选源码填写，并纳入files。
Tensor Storage扩展亦纳入files，必须来自匹配构建。
逐runtime的admission_by_runtime填写真实PASS和原始报告文件；模板中的占位路径/NOT_RUN不能作准入。
同一配置的后端不得中途换库；固定N/W/seed，不挑点、不丢慢样本。

```bash
python csrc/_310P/tools/L1_test/l1_bench.py campaign freeze \
  --campaign "$RUN_DIR/plan.json" --template "$RUN_DIR/runtime.json" --out "$RUN_DIR/frozen"
python csrc/_310P/tools/L1_test/l1_bench.py campaign start --out "$RUN_DIR/frozen"
python csrc/_310P/tools/L1_test/l1_bench.py campaign status --out "$RUN_DIR/frozen"
```

全量预检通过才启动，后台独立session有界运行。发现失败不重试ADD、不继续后组，保留STOP/未运行清单及双Rank原件。
不擅自增加host/profiler三段采集或旧V2历史全系列，这些会改变本轮会话数。
后续正式独占轮使用新run_id，plan显式改为`--mode exclusive --warmup 20 --samples 100 --rounds 1`，
重做当前环境准入与freeze，不复用shared的计划/结果冒充独占证据。

## 4. 回传与归档

原件、计划/manifest/SHA、G0/占用/构建/路由、全部Rank样本与summary.json/csv、campaign-result.json都归档。
大文件可制品包+SHA，但必须可取回，不能只有临时服务器路径。结果直接提交主线
`csrc/_310P/tools/L1_test/experiments/<新的唯一run_id>/`，同步PROJECT_CONTROL/PERFORMANCE_BASELINES。

先报告每组完成/STOP/NOT_RUN、实际源码/SO/SDK身份；再按M/ready/chain列完整Wall均值、p50/p90/p99、
各组独立Stock、速比、漂移。诊断段不混到正式收益；不把不同配置Stock拼成一个平均，不把失败批次写成完整PASS。
最后突出V3每次迭代与通信/计算粒度的收益或劣化，说明证据级别；不虚构设备内部阶段耗时或发布GO。

## 简短委托词

目的：验证统一L1_test与Stock/V2/V3适配是否能在310P3双Die正确运行，并观察V3各实验的完整输入输出耗时。
仅复用服务器现有vllm-ascend、memfabric_hybrid两个仓库。vllm-ascend切dev/310p-memfabric-mm-ar；
V2/V3共用本文件锁定的MemFabric fix分支/db38d75e；SDK按实际兼容性验证后冻结。
先核未提交内容和运行任务，安全后原地切换，SSH拉GitHub，
不新增clone/worktree。构建放原仓build/310p_l1/<run_id>，结果放原仓
csrc/_310P/tools/L1_test/experiments/<run_id>，不删旧目录。
按本文件先做G0、CPU、匹配构建和双Rank短测，再执行完整43组：M2048/4096、ready+chain、
shared、W2/N10/R1、collection off，允许其他进程及漂移；保留健康/数值/协议门禁，首错STOP。
V1不测，不自动追加独占轮或Profiler。回传双Rank原件、summary、失败清单、两仓/SDK/SO身份和目录清单，
代码修复及实验归档直接提交上述主线；shared耗时仅作观察，不签正式性能结论。
