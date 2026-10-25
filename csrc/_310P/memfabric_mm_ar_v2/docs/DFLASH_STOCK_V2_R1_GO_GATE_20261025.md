# R1｜Qwen3.6 DFlash K=7：Stock / V2 四象限现场准入与验证合同

> 2026-10-25；唯一主线 dev/310p-memfabric-mm-ar。**只完成GitHub静态准备；310P3环境、Eager和Graph均NOT_RUN。** 本文是唯一R1执行规范；总门禁见 [ENGINEERING_GATES.md](ENGINEERING_GATES.md)。

## 1. 冻结合同与四个独立运行组

目标模型：/home/models/Qwen3.6-35B-A3B-w8a8；草稿：/home/models/Qwen3.6-35B-A3B-DFlash；num_speculative_tokens=7，额外显式draft_tensor_parallel_size=2（来自原上游TP2验收脚本），TP=2/单卡双Die。理论8个验证位置不是实测physical/padded M。

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
