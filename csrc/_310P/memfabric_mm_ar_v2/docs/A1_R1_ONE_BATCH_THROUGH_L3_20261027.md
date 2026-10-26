# A1-R1｜一次进场连续执行G0、M2048 Smoke、L1、L2、L3（2026-10-27）

> **用户最新明确授权：**环境恢复与隔离验证条件满足后，同一个OpenCode任务继续做首次M2048双Rank正确性Smoke；Smoke安全/数值/实际装载全部通过后，**无需再向用户逐段申请，连续推进L1 Screen、路由审计、L1 Confirm、L2真实Serving取证及L3真实Prefill验证**。L3_GO后也可按原正式4096/2048验收合同执行L4，前提是同样设备安全、独占、软件身份冻结且已有正式L4脚本在当前环境通过静审；若任何约束不满足就STOP/保全，不在现场变更授权边界。**禁止以进度压力放宽任何设备与数值闸**。
>
> 适用A1工作分支 `work/310p-mm-ar-large-m-a1-20261026`，正式目标、状态与版本依主线 [大M实验总账](https://github.com/10wvw01/vllm-ascend/blob/dev/310p-memfabric-mm-ar/csrc/_310P/memfabric_mm_ar_v2/docs/LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)。旧[R1A/R1B分段文档](A1_R1_ISOLATED_LOADER_PREFLIGHT_AND_SMOKE_GATE_20261027.md)作为具体环境证据与备份参考，**其“每段必须重新人工批准”的组织约定由本最新单批授权取代**。旧脚本自身的保护门（如U1需G0/独占/恢复与Confirm环境变量）**不可跳过**；这些变量只能在OpenCode本批已收齐真实批准条件与完整证据后设置，不允许预先设置。

## 0. 必须提前冻结的三个原则

1. **不扰动别人**：服务器正在运行其它分支。R0候选卡16384(dev2/3)必须现场重验同卡双Die健康/独占；不修改其它租户进程/共享Host双路径/正在被editable引用的插件目录。首选私有隔离装载；若唯一方法需要覆盖共享SO或全局site-packages且无明确维护窗口/全体受影响租户退出证据，**本批停在R1恢复方案，不冒险启动Kernel**。
2. **先真身份后跑数**：修复Host SO SHA256固定`1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30`，D4 Device SO`2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19`，插件`8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1`，KVER11 JSON`25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78`。**候选路径SHA通过≠实际Host/Device已加载**。从PID/maps/装载日志/CANN Kernel注册Trace证明真实身份，特别防止在装baseline Device SO`b8ca96f0...`伪装D4。
3. **无异常才续跑**：每阶段审原始Rank0/Rank1结果、SO/KVER、通信协议、日志与卡健康。任何507014、507899、孤立507018、KFC 0x2a/辅助程序错误、IB/Core7阻塞、位不精确、温控/MCU告警、外部租户新占用时立即STOP，不自动换卡、清理KFC、重启刷通过或偷偷取消测试。

## 1. 一批执行总表（减少交接，但阶段数据独立）

| 顺序 | 必须做什么 | 无异常后续条件与输出 |
|---|---|---|
| **S0 环境复原** | 主线与A1工作树身份、4卡重巡、完整SHA、CANN9.1/torch/Python ABI、无有毒环境变量、私有Host+D4+Plugin+KVER装载链；必要时隔离目录复制冻结产物及恢复验证 | CPU原仓测试全绿，卡健康独占，私有实际加载方案可证明，不得覆盖共享SO。输出`r1_loader_static_inventory.json`、真实装载审计和原始日志 |
| **S1 正式M2048 Smoke** | `bash test/perf/u1_run.sh /NEW/ABS/path large smoke`，2次独立torchrun（Stock后D4），每次两Rank、3次warmup+6次测量+尾锚，首次V2建池自动KVER11/KFC，位精确/status双Rank检查 | 运行`a1_r1_smoke_gate.py`独立CPU验收原始两个Rank JSON、冻结SO/设备、现场装载/KFC/首写/独占证据；只有输出`ALLOW_SCREEN`继续。Smoke性能不作GO |
| **S2 L1 Screen** | `bash test/perf/u1_run.sh /NEW/ABS/path large screen`：Stock与V2分会话，M1024诊断、M2048/M4096正式，双Rank完整链 Event与MM-only旁证；用`u1_analyze.py`和`a1_large_m_assess.py`离线分析 | 保存两个Rank原始数据、真实HCCL跨流完成与D4 kernel绑定调查，确认无协议异常；即便结果负收益也继续L2分析 |
| **S3 L1 Confirm** | 当S2没有安全/数据完整性阻塞且实际HCCL路由能审计时，按已有U1 `large confirm`8会话（Stock–V2–V2–Stock ×2，每M40samples），原始Rank配对与Stock漂移检查，独立HCCL completion audit | 必须真实填`U1_CONFIRM_APPROVED=YES`的现场检查证据，不能无理由强加；性能负值照记，**不要为了过原5% Candidate筛选规则而改阈值**；安全/统计不可信则不把L1记GO，但可进入不加风险的离线分析 |
| **S4 L2真实Qwen取证** | 同模型、同物理双Die/同版本，Eager input4096/output1/concurrency1；Stock与D4各1次独立 Serve，6warmup+5请求短Profiler。按[旧归因分析器](../evidence/v2_prefill_attribution_20261021/scripts/analyze_prefill_trace.py)与原始ascend_pt双RankTrace核每Rank10个o_proj+30个out_proj实际命中、物理M、Kernel、输入contiguous拷贝、HCCL真实AR完成、SDMA/IB等待和端到端依赖 | Profiler诊断L2与校准L3事件覆盖边界，**不把Profiler时延或40层求和当作L3稳定收益**。若当前Trace工具的字段/层名已变化，先静审诊断适配，不能编造40命中 |
| **S5 L3模型Prefill无Profiler四ServeABBA** | **只有S4已确认真实设备依赖/安全且实验Event开关通过CPU+最小实机自检时**，在同一卡分别执行 Stock S1→V2 F1→V2 F2→Stock S2，共4个独立Serve、各6warmup+30成功请求；输入4096/输出1/并发1/TP2/Eager、不启Profiler、开启`A1_310P_PREFILL_EVENTS_DIR`测试Only事件。纯CPU执行`a1_l3_prefill_assess.py` | 收集每Rank同一Stream Forward→Logits事件时长，L2需审定已覆盖真实关键依赖与请求对齐，才可判断`L3_GO`或`L3_INCONCLUSIVE_OR_NO_GAIN`；测量不可靠则`L3_MEASUREMENT_NOT_READY`，不能用客户端TTFT代替 |
| **S6 L4正式验收（严格条件执行）** | **仅在S5数据与L2证据完整且判`L3_GO`、同一软件/物理卡身份、零异常，并且现场已静审正式Eager 4096/2048/TP2/并发10/50请求无Profiler启动器**的情况下，顺带执行Stock–V2–V2–Stock四场；否则`L4_NOT_RUN_BY_DESIGN` | 只接受逐请求真实客户端TTFT、Stock跨会话漂移、p50/p90/p99、所有失败样本；最终发布仍受Gate E/首写限制。L3未GO绝对不跑L4 |

**为了提高单次信息密度**：阶段之间不等待ChatGPT重新回复，可由OpenCode在同一次委托按上述条件自动推进。**但一个原始证据/身份/数值/健康门禁不合格就不能跳过去**；已完成阶段证据照常上传，未能通过的后续标NOT_RUN而非放宽条件。性能不正向不是安全故障，仍应进行有诊断价值的L2/L3采样；正式L4只有L3_GO。

## 2. 这批新增工具和使用方法

- `test/perf/a1_r1_loader_readonly.py`：离线读取候选/在装Host/D4/插件/KVER SHA及ELF依赖，**不验证实际装载**；仍需现场调用/设备注册取证。
- `test/perf/a1_r1_smoke_gate.py`：纯CPU核原始Stock/V2双Rank M2048 Smoke、KFC/实际加载/独占等**外部取证**；只在`ALLOW_SCREEN`时进入S2。外部`attestation`必须有`scope=A1_R1_M2048_SMOKE_RUNTIME_ATTESTED`、物理卡映射、9个全真条件及至少6份真实非空日志/trace，禁止无根据填写True。
- `vllm_ascend/_310p/a1_prefill_event_probe.py` + `vllm_ascend/worker/model_runner_v1.py` **环境开关测试插桩**：只在`A1_310P_PREFILL_EVENTS_DIR`指向现存私有绝对目录时启用，在Forward进入与Logits输出处各记录一个NPU Event，并仅在原本自然设备完成后检查`Event.query()/elapsed_time()`，不插入同步；将完成事件写出`a1_prefill_rank{rank}_pid{pid}.jsonl`。默认开关未设时不执行事件测量，**不是替换Runner、不会将事件自动解释为完整Step**。
- `test/perf/a1_prefill_serve_session.sh`：单次私有、独立`vllm serve`，强制明确模型/卡/端口/SO身份与全新输出，Stock/V2同参数；`l2_profile`才允许profiler，`l3_event`只开上面的无Profiler事件；客户端统一`a1_prefill_http_requests.py`（固定4096输入、1输出、请求串行、逐请求日志）。
- `test/perf/a1_l3_prefill_assess.py`：只接受四Serve S1/F1/F2/S2、每Rank恰好对应36条有效4096Token预填事件、同版identity、L2 Trace/设备依赖与Rank请求对齐签核；输出与Stock漂移对照后才可能`L3_GO`。**此时L3测试代码仍需原仓CPU和现场Event API自检；不允许在采集失败后把HTTP客户端延迟填进设备Step字段**。
- `test/ut/test_a1_{r1_loader_readonly,r1_smoke_gate,prefill_event_probe,l3_prefill_assess}.py`：全部为纯CPU合同测试。**现场先执行完整原仓pytest/py_compile/bash -n**；没有本轮新实机执行记录前，一律NOT_RUN。

示例（真实路径由OpenCode经过S0审计后填写，不能从示例覆盖当前现场）：

```bash
P=csrc/_310P/memfabric_mm_ar_v2
python3 -m py_compile "$P/test/perf/"a1_{r1_smoke_gate,prefill_http_requests,l3_prefill_assess}.py \
  vllm_ascend/_310p/a1_prefill_event_probe.py vllm_ascend/worker/model_runner_v1.py
bash -n "$P/test/perf/u1_run.sh" "$P/test/perf/a1_prefill_serve_session.sh"
python3 -m pytest -q "$P/test/ut"

# S1: Smoke 仅在S0真正完成实际装载确认后执行
bash "$P/test/perf/u1_run.sh" /THIS_BATCH/ABS/smoke large smoke
python3 "$P/test/perf/a1_r1_smoke_gate.py" \
  --smoke-root /THIS_BATCH/ABS/smoke \
  --evidence-root /THIS_BATCH/ABS/runtime_proofs \
  --attestation /THIS_BATCH/ABS/runtime_proofs/attested_smoke.json \
  --out /THIS_BATCH/ABS/smoke_gate.json

# S2/S3: 若smoke_gate.json状态ALLOW_SCREEN且运行无异常
bash "$P/test/perf/u1_run.sh" /THIS_BATCH/ABS/screen large screen
# 独立路由/HCCL审计能证实完成语义时，设置U1_CONFIRM_APPROVED=YES（保留证据）
bash "$P/test/perf/u1_run.sh" /THIS_BATCH/ABS/confirm large confirm

# S4/S5在现场动态装载、运行时身份与模型Eager路由均已证明后继续：
# 对S4 stock/v2分别起一个l2_profile独立serve；
# 对S5 S1/F1/F2/S2分别起一个l3_event独立serve。
# 每次显式A1_SERVE_MODEL_PATH/A1_SERVE_MODEL_NAME/A1_SERVE_PORT/
# A1_SERVE_STORE_PORT/A1_RUNTIME_IDENTITY_JSON，保证端口空闲与每臂独占。
bash "$P/test/perf/a1_prefill_serve_session.sh" stock l2_profile L2S /THIS_BATCH/ABS/l2_stock
bash "$P/test/perf/a1_prefill_serve_session.sh" v2    l2_profile L2F /THIS_BATCH/ABS/l2_v2
bash "$P/test/perf/a1_prefill_serve_session.sh" stock l3_event S1 /THIS_BATCH/ABS/l3/sessions/S1
bash "$P/test/perf/a1_prefill_serve_session.sh" v2    l3_event F1 /THIS_BATCH/ABS/l3/sessions/F1
bash "$P/test/perf/a1_prefill_serve_session.sh" v2    l3_event F2 /THIS_BATCH/ABS/l3/sessions/F2
bash "$P/test/perf/a1_prefill_serve_session.sh" stock l3_event S2 /THIS_BATCH/ABS/l3/sessions/S2
python3 "$P/test/perf/a1_l3_prefill_assess.py" \
  --root /THIS_BATCH/ABS/l3 \
  --audit /THIS_BATCH/ABS/l3/l2_route_calibration.json \
  --out /THIS_BATCH/ABS/l3/l3_result.json
```

**上例前置条件不可省略**：私有真实加载与独占、KFC/设备安全证明、双Rank路由确认、固定模型/端口、`A1_RUNTIME_IDENTITY_JSON`必须来自本批原始文件；`l2_route_calibration.json`来自**真实L2 Trace**而不是猜测。旧`v2_prefill_capture.sh`默认0/1、8000、全局set_env和已失效配置，**禁止未经审查原样运行**。若S4 profiler引入设备问题STOP，不跳到S5。S5若没有36条事件/Rank、缺实际40目标站点或Event不足以代表完整Prefill关键路径，输出MEASUREMENT_NOT_READY，**不假装L3_GO**。

## 3. 完整验收与事后报告

一次上传**各阶段原始**`manifest.json`、rank0/rank1 JSON、session.log、真实PID装载证据/KFC两Die状态、当前四卡before/after、L1 screen+confirm分析、L2 ascend_pt双Rank profiler/每层40站点及Copy+HCCL/IB取证、L3每Rank event JSONL、ABBA四Serve原始请求与device timing、Hash与异常/停止记录。L3请求序列`S1/F1/F2/S2`为服务启动顺序，不与U1微基准阶段名混淆。保留`run_id`、源码/真实装载SHA、模型Eager参数/种子、实际请求Token数与独立会话标签。

要提交结构化结论：`G0/ISOLATION/SMOKE/L1_SCREEN/L1_CONFIRM/L2_TRACE/L3_GO_OR_NOT/L4`，并给当前D4相对完整Stock的正式M2048/4096微秒改善、真实模型目标40站点份额、两Rank Prefill Step ms改善及传导原因。任何缺字段写UNKNOWN；失败的候选不追逐“刷新直到增益出现”。

**最终准入声明**：本批一次委托的条件授权允许在所有证据和硬门合格时走完整链，**不等于准许设备异常后的自动重启、共享系统SO覆盖、跨业务抢卡、进入发布默认或跳过L3而运行L4**。实机尚未执行，所有结果目前NOT_RUN。
