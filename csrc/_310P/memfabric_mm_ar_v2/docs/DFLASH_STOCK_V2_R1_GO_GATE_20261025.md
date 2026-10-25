# R1｜Qwen3.6 DFlash K=7：Stock / V2 四象限现场准入与验证合同

> 2026-10-25；唯一主线 dev/310p-memfabric-mm-ar。**只完成GitHub静态准备；310P3环境、Eager和Graph均NOT_RUN。** 本文是唯一R1执行规范；总门禁见 [ENGINEERING_GATES.md](ENGINEERING_GATES.md)。

## 1. 冻结合同与四个独立运行组

目标模型：/home/models/Qwen/Qwen3.6-35B-A3B-w8a8；草稿：/home/models/Qwen3.6-35B-A3B-DFlash；num_speculative_tokens=7，TP=2/单卡双Die。严格使用用户给出的DFlash JSON（不额外显式添加draft_tensor_parallel_size）；vLLM v0.24.0 对非mlp_speculator模型默认将未设置的draft TP解析为目标TP=2，**必须由现场实际Worker日志确认**。理论8个验证位置不是实测physical/padded M。

| 顺序 | 运行组 | 数据与裁决 |
|---|---|---|
| G0 | 只读依赖+设备核对 | 真实Python和vLLM v0.24.0 API、两模型配置、设备物理映射、两处Host SO、Device SO、KVER11、KFC/IB与HCCL健康、独占及恢复证据。未全PASS不得运行模型 |
| R1-1 | Stock+DFlash Eager smoke | 输入4096/输出32、2请求、并发1；Stock严禁实际融合 |
| R1-2 | V2+DFlash Eager smoke | 同一请求；必须观察正式V2 LinearMethod、2048～4096范围融合及首调位精确；真实双Rank kernel证据仍要补 |
| R1-3 | Stock+DFlash Eager full | 4096输入/2048输出/TP2/batch上限10/50请求/ignore-eos |
| R1-4 | V2+DFlash Eager full | 与R1-3完全同口径，记录真实Prefill命中数、Verify/Decode退回Stock数、接受率/Target Forward数/吞吐、TTFT/TPOT/ITL |
| R1-5 | Stock+DFlash FULL_DECODE_ONLY Graph smoke | 只有两组Eager审阅后才人工授权，保留target/draft/rank capture+replay、实际pad M |
| R1-6 | V2+DFlash FULL_DECODE_ONLY Graph smoke | Stock Graph通过后才诊断共存；不执行Graph完整压测。V2 eager Prefill命中不意味着V2 Kernel在Graph内部捕获重放 |

**当前正式V2 Serving仅2048≤M≤4096融合，小M Verify/Decode仍是Stock。** V2+DFlash只是首轮模型集成，不能宣称投机Decode已融合。历史V2 Graph从未完成有效验收，旧Graph A/B曾把Stock-vs-Stock误算成V2对比。Graph捕获描述符候选为8/16/32/64/80，全部是K7宽度8的整数倍；必须现场验证，不复用旧[10]配置，也不直接搬上游验收脚本K=15或其开发机硬编码路径。

## 2. 本次工程交付

- tools/310p_mm_ar/dflash_stock_preflight.py：Stock只读依赖G0，模型config、API模块路径、pip check、标志隔离；只容忍上游310P Docker有意卸载的Triton与Triton-Ascend明确缺包告警，不放行其他缺包或版本冲突。
- tools/310p_mm_ar/dflash_v2_trial.py：plan生成全参数manifest（纯CPU无硬件副作用）；execute必须同时提供已PASS的真实G0 JSON、G0批准/双Die独占/恢复方案审阅三个显式开关，否则不启动。每次新进程组启动服务与Bench，结束只处理本试验PID组，失败保存日志且绝不自动重跑；verify读取manifest、server.log、bench.json防Stock污染、缺失融合路由及未完成请求。
- tools/310p_mm_ar/test_dflash_stock_preflight.py、test_dflash_v2_trial.py：隔离Stock、K7 TP2、Graph宽度、路由真实性、G0和Graph门禁、异常终止的纯CPU合同测试。CI必须有实际运行成功记录，不替代310P设备测试。

**禁止本轮直接运行历史tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh的默认Graph配置。** 它的Graph捕获尺寸[10]、SO自动搜索和默认设备假定不符合本轮K7严格合同。

## 3. 现场G0的安全操作顺序

1. 只读检查git remote -v、git worktree list --porcelain、每个worktree的git status --porcelain、已装载Python模块实际路径和版本、现场PID/端口、npu-smi、真实物理Die号。先保全全部未提交改动、已有SO和日志。禁止在正在使用的工作树直接checkout/reset/clean；新任务使用隔离worktree，取远程唯一主线HEAD并记录完整SHA。
2. 检查设备两Die确属同一310P3、独占；记录CANN/BiSheng/Driver、真实KVER11 JSON及设备端SO、两条MemFabric Host库路径及SHA256、扩展SO与V2 Device SO哈希、进程实际maps、HCCL和IB/KFC证据。G0失败NO-GO，禁止盲目更换SO、pip install、重启设备。
3. 在**实际启动vllm所用Python环境**、隔离工作树根目录执行纯CPU测试与只读Stock探针，并保存每条完整exit code和JSON：

    python3 -m unittest discover -s tools/310p_mm_ar -p 'test_*.py' -v

    python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut

    python3 tools/310p_mm_ar/dflash_stock_preflight.py --num-speculative-tokens 7 --out /tmp/opencode/唯一run_id/g0_stock_dflash.json

4. G0 JSON verdict.static_go=true且blockers为空才可能继续；真实设备、版本与SO仍需人工核验。可选导入探针只在确认现场可安全加载torch-npu库时运行带--import-smoke的独立新JSON文件。若发现ModuleNotFoundError/ImportError先核对上游底座与我们的额外ModelSlim导入，不在本次自动修环境。

## 4. 顺序跑两组Eager，按条件进入Graph探索

设置经G0确证的物理设备对和已核对KVER11 JSON绝对路径。下列用变量代替真实值，不能原样复制假设备ID：

    ROOT=/tmp/opencode/唯一run_id
    DEVICES=经G0确认的设备号1,设备号2
    KVER11_JSON=已确认的绝对路径/libmf_sdma_orch_v11.json

每组都运行一次下述plan → execute → verify，同一个组使用同一manifest，**每组目录必须全新**：

    python3 tools/310p_mm_ar/dflash_v2_trial.py plan --arm stock --mode eager --phase smoke --devices "$DEVICES" --port 8000 --dir "$ROOT/stock_eager_smoke"

    python3 tools/310p_mm_ar/dflash_v2_trial.py execute --manifest "$ROOT/stock_eager_smoke/manifest.json" --preflight "$ROOT/g0_stock_dflash.json" --g0-approved --exclusive-approved --restore-reviewed

    python3 tools/310p_mm_ar/dflash_v2_trial.py verify --manifest "$ROOT/stock_eager_smoke/manifest.json"

Stock Eager smoke正确后，将--arm换v2，目录换$ROOT/v2_eager_smoke，并**增加--orch-json "$KVER11_JSON"**；执行后必须有V2正式路径和首调正确性证据，否则NO-GO。两组smoke PASS后分别运行--phase full，目录为stock_eager_full、v2_eager_full，严格50请求/2048输出/并发10。每组结束先检查设备清洁与服务PID，不能两个模型服务共用一对Die同时运行。

Graph仅在两组Eager已有审阅证据、Stock原始完整基线正常、设备健康时研究性开展；创建--mode graph --phase smoke的Stock和V2新目录，execute额外传入--graph-research-authorized与--eager-baselines-reviewed，**Stock Graph先于V2 Graph**。Graph出现错误即STOP并保留设备/两Rank、capture、replay、Host/Driver日志，不自动重试、更改首写保护或默认路由。

## 5. 验收与必须回传的原始证据

- Stock请求完成不等于依赖问题已全面修复：Stock需无融合选层；同模式真实K7生效，草稿接受率和Target Forward次数需单独核验。
- V2日志中的一次选层/首调bit-exact是**必要但非充分**条件；无双Rank设备真实Kernel调用与活跃请求M分布时只记“路由候选”，不能算有效算子收益。
- Graph即使全请求正确，也必须提供target/draft各Rank原生capture/replay及图内算子命中证据；目前V2 Graph仍为UNKNOWN/NOT_RUN。V2 Prefill Eager融合与Graph中的Stock Decode必须拆账。
- 完整对照只认同配置Stock+DFlash vs V2+DFlash（普通无投机4096/2048另立账），保留完整率、吞吐、TTFT/TPOT/ITL P50/P90/P99、接受率、Forward次数、真实融合调用数和解释为何收益传导有限。
- 所有manifest.json、server.log、bench.stdout.log、bench.json、verdict.json、G0 JSON、双Rank日志、设备轨迹、Graph文件、版本SO/KVER SHA、失败样本和清理/回滚验证保存。OpenCode先将代码修复和报告提交自有仓库短期任务分支，再评审合入唯一主线，同批更新PROJECT_CONTROL.md及有性能结果时的PERFORMANCE_BASELINES.md。绝不提交外部HaonanWu1仓库。

## 6. 裁决

**研发/静态阶段：PRECHECK-GO，允许进入310P3服务器做G0和CPU检查。真实硬件条件：UNKNOWN；只有现场G0全PASS才能做Stock+DFlash Eager短冒烟。** V2+DFlash Eager严格排在Stock后；Graph受控诊断排在两组Eager后。无任何新真实K7性能或Graph通过声明。

## 7. 用户原始启动参数的逐项校准（2026-10-25）

用户给定的正式业务服务端：
- Target模型绝对路径：`/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`，不同于早期推断的`/home/models/Qwen3.6-35B-A3B-w8a8`。本轮G0和R1脚本**均已改成用户路径**；现场不存在则G0失败，而不是自动选另一个。
- --host 0.0.0.0（服务端监听全部地址，现场需要提前确认网络隔离/访问控制）；--port 8000；--served-model-name qwen3.6-35b-a3b-w8a8；--tensor-parallel-size 2；--quantization ascend；--dtype float16；--max-model-len 8192；**--max-num-seqs 16**；--max-num-batched-tokens 4096；**--gpu-memory-utilization 0.90**；以及全部prefix/chunked/async、Mamba缓存、additional-config参数，已与用户基线统一。
- DFlash JSON严格为`{"method":"dflash","model":"/home/models/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":7}`；之前脚本显式增加draft TP=2，本轮已删除以匹配用户原样。vLLM v0.24.0 `SpeculativeConfig._verify_and_get_draft_tp()`会对一般draft模型沿用target TP=2，现场必须核实。
- 删除额外的`--safetensors-load-strategy eager`及Bench的`--num-warmups`选项以减少基线漂移；Bench仍使用独立的result目录、bench.json文件及更完整的TTFT/TPOT/ITL统计/明细，业务负载与用户完全相同（50请求/4096/2048/并发10/ignore-eos）。
- **刻意保留唯一重要运行模式差异**：Eager smoke和Eager full显式设`cudagraph_mode=NONE`（对应禁用Graph），用来隔离依赖、模型和V2 Eager问题；只有Graph研究组才设`FULL_DECODE_ONLY`。
- **用户的Graph `cudagraph_capture_sizes:[10]`不能直接视为K7有效配置**。我们自己的DFlash FULL Decode dispatcher用`uniform_decode_query_len=1+K=8`检验`candidate % uniform_decode_query_len == 0`；10%8=2，对K7的常规单请求验证形状无法通过；batch10理论80 token也超出单一capture尺寸10。R1为独立受控Graph实验，选择候选尺寸`[8,16,32,64,80]`，必须真机验证并记录是否遭拒、实际Padded M与Graph Replay，不擅自宣称已跑通。
- **端到端报告必须分别写明**：与用户命令字段相同、因Eager/Graph目的不同而有意差异、仅为证据管理增加的日志/目录/采样字段。不可把Eager实验的收益直接称为用户原始Graph配置性能。

## 8. R1-B 整合后 310P3 实机准入与止损合同

> 前轮 frozen G0=NO-GO、四组NOT_RUN，详见 [R1现场原始报告](R1_G0_ONDEVICE_NOGO_REPORT_20261025.md)。本附录取代前文中**过时的**vLLM v0.24 pin、Draft原始路径以及“可选导入探针”安排；保留所有现场安全门禁。**未授权自动修改服务器环境或清空占用。**

1. **源码冻结**：只从 `dev/310p-memfabric-mm-ar` 更新/创建隔离worktree，记录commit SHA、`git status`和`PYTHONPATH`。真正serve用的`vllm_ascend`必须解析为本隔离worktree而不是已有editable主工作树。更新原始报告不代表生产二进制已重新编译。
2. **目标/草稿路径**：Target `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`，Draft **现场确认** `/home/models/Qwen/Qwen3.6-35B-A3B-DFlash`。该路径与用户早期提供的不含Qwen目录版本不同；一律用实际`config.json`检测并把差异记录报告，严禁无证据自动找另一个目录。Stock和V2一致。
3. **软件依赖**：冻结已知服务器 vLLM 0.27.1（真实源码git@6e448d0）、torch2.10.0、torch-npu 2.10.0.post4、transformers5.17.0；不是重新安装0.24时代旧环境。先在真实 Python3.12.13 上用 `python tools/310p_mm_ar/dflash_stock_preflight.py --import-smoke --out <新目录>/g0_dflash.json`，确认 `static_go=true`、`blockers=[]`、`isolated_import_probe.exit_code=0` 及每条pip警告均属**明确列出的**已知状态；新增依赖/版本/导入错误立即NO-GO并保留完整原始日志。
4. **环境保护**：重新核验同一NPU43520双Die 6/7独占（不能仅凭上一轮快照）、设备表/宿主机PID/端口8000+8581、CANN/驱动、HCCL+IB/KFC安全、Host双路径SO、Device SO和KVER11真实文件及SHA；任何不一致先STOP。设备4旧残留显存/hugepages无须在本轮触碰。
5. **V2 SO准入**：隔离worktree天然缺失gitignored的`vllm_ascend_C*.so`、`libmf310p_v2_trial_device.so`、`libmf310p_device.so`；**不得**偷偷从另一worktree import或无备份替换正在使用的安装库。优先隔离编译；若确须复用现成SO，提供前后SHA、ABI及备份回滚计划并获用户明确批准后才做；Trial已增加缺失阻断。
6. **两个Eager短测及完整业务**：先Stock Eager smoke（input4096/output32、两请求），检查真实DFlash服务加载及草稿推理、K7、双Rank无异常，成功再V2 Eager smoke并验证正式V2选层/首调位精确、双Rank真实NPU Kernel命中。两者通过才分别执行Eager 4096/2048、50req、maxConcurrency10/batch10、ignoreEOS；保存接受率、Target Forward、physical/padded M、TTFT/ITL/TPOT和原始rank日志。脚本返回路由“提示”不等于融合通过；必须核设备trace。无法重现实机只报告UNKNOWN/FAIL。
7. **Graph不默认执行**：完成双组Eager审阅后另行人工批准受控Stock Graph smoke→V2 Graph smoke；Graph候选 `[8,16,32,64,80]` 仍未经实机通过。旧V2 Graph失败、Graph Stock-vs-Stock伪命中史仍有效；不自动跑长稳。
8. **首写/通信故障处理**：出现507014/507899、无错误码的静默数值损坏、双Rank阻塞、导入混树或SO不一致立即停止，不能用关KFC/cache clean/同步保护绕过；回传所有manifest、G0 JSON、两个Rank、Kernel trace、模型输出、SO加载`/proc/*/maps`与停止后环境清单。新证据只提交自有项目仓库分支，并同步PERFORMANCE_BASELINES/PROJECT_CONTROL。
