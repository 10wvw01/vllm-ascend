# F1-C｜310P3双Die小M：Stock FP16 HCCL vs T1 MemFabric 真对照实验方案与实机准入

> **编写时间：2026-10-25。状态：源码/实验工具已提交，尚未实机执行；必须完成G0后方可跑TP2。**  
> 专属分支：`perf/310p-f1c-stock-fp16-hccl-vs-memfabric-abba-20261025`，基于F1-1报告完结的独立分支 `b14fbe2f863e8...`。未修改任何正式 MM+AR Kernel、Host ABI、MemFabric内存/通信协议；Stock始终保持模型默认。
>
> **项目原则：不使用F1-1的建池前FP32转换Stock 0.408ms与建池后T1 0.340ms直接认定胜出。** 本实验首要任务是验证这68µs究竟多少是实实在在的FP16 MM+HCCL对照收益。

## 1. 策略修订：不强行要求HCCL与MemFabric池在同一进程

历史 `NZ_DIRECT_TRIAL.md`、`T2_AR_ONLY_REPORT.md` 均记载：参考HCCL需要尽量在MemFabric建池前执行，曾复现混用、MTE3首写保护和HCCL端口遗留风险。

原F1-1实机报告提出“同进程/同池ABBA”，当前**没有完成HCCL与MemFabric池实时共存安全证明，不应直接照做**。因此本轮选择：

- `stock_fp16`：独立torchrun TP2，CPU gloo用于数值参考和step barrier，**另外仅本进程创建HCCL group**；计时区 `F.linear(fp16)` → `dist.all_reduce(fp16, HCCL)`，不发生`float32`转换；**不创建MemFabric池**。
- `memfabric_t1`：下一场全新torchrun TP2，CPU gloo用于参考和barrier；**不创建HCCL通信group**，仅使用T1 `mf310p_v2_trial_run_nz_direct_small`，MM8/Add4、NZ weight。继承经实机正确性验证的stream级首写保护和KVER11协议，使用相同轮换输入。
- 对**同一物理双Die和同一装机环境**重复 `A(stock)-B(T1)-B(T1)-A(stock)` 两轮（共8个独立进程会话），每场5预热+50计时样本，每次都保留真实Rank0/1。**不是同一进程内的ABBA，不是跨会话同时刻配对**；跨会话只做交叉中位与热态漂移分析。

该方案消除了“建池前Stock、建池后T1但只采一个会话”的单次时间顺序偏差，同时避免把HCCL和MemFabric放进同一个进程。两条线的Host初始化成本仍不属于NPU Event主指标，Host Wall单独展示。

## 2. 输入与数值公平性

- 优先测 **M=10**，随后只有M10严谨且取得有意义收益时才扩展 M=1/8/16/32；`K=N=2048`、TP2、FP16矩阵乘及输出，T1用与F1-1完全一致的 `npu_format_cast(weight, FRACTAL_NZ=29)`；Stock使用原生`F.linear(x,w)`（框架原生算法，不换成定制MM）。
- Rank固定种子：权重4242；输入 `1200+rank*17+M+variant*1009`，2个不同输入逐次交替，避免相同内容/地址复用掩盖旧结果。输出全部要求 `torch.equal` 等于每Rank局部FP16 F.linear结果经CPU gloo FP32规约再FP16转换后的参考；若Stock原生FP16 HCCL与参考有可解释的数值合同差异，必须**STOP并记录**，不得放宽标准通过。
- 每步采两个stream Event包围完整算子区间，`event_us`为设备事件主指标，`wall_us`为Host+同步辅助指标；**Stock HCCL可能跨流**，因此只有另一个独立诊断Trace确认`event_end`确实等待通信完成，才允许把Event指标当作严格Stock对照。Profiler只运行在另开的小诊断会话，不能混进8场正式计时。
- 双Rank每第i个采样取`max(event_r0[i], event_r1[i])`后算p50/p90/p99，不能先分Rank算p50再max。跨会话按两组ABBA成对评判。不可删除高延迟样本、错误样本或因缺日志自动重跑。

**公平性边界**：Stock`F.linear`会按原生API创建输出，而T1预分配输出，属于真实完整算子调用路径差异。必须保留该事实，不能将所有差额直接解释为MemFabric传输本身更快；如果需要隔离框架输出分配/Host开销，下一轮另做Stock`out=`形式的诊断对照，保持正式Stock基线不变。

## 3. G0：310P3服务器准入（任何一项失败都不能正式采样）

F1-1结束后**已经恢复装机旧态**：旧设备 SO `b8ca96f0...`、旧MemFabric Host SO `88d39285...`。因此不能直接从F1-1“环境无进程”推断F1-C可启动。先独占同一单卡双Die，逐项记录并**受控备份、部署、验证**：

| 检查 | 必须具备的证据 |
|---|---|
| G0-1代码 | `git switch`新F1C分支且`git pull --ff-only`；记录完整HEAD与`git diff <F1-1 HEAD> HEAD`；`git status --porcelain`确认生产源码无未经说明的修改；新分支不得含C3优化或正式V1 Decode路由变更 |
| G0-2资源 | `npu-smi info`证明物理6/7属于同卡43520、进程资源独占；有其它租户则改用符合物理映射支持且经过准入的卡对或中止；记录`ASCEND_RT_VISIBLE_DEVICES`与两个worker的local-rank映射 |
| G0-3二进制 | **设备SO**已部署D4 Tile2048诊断ON `2a19390c...`；**Python扩展**使用诊断ON `8eb65de9...` 并暴露T1 ABI；**MemFabric Host SO**在两安装路径均为物理号修复版 `1dc46d7d...`；KVER11 JSON `25e16b0a...`，设备侧AICPU KVER11已正常部署和探测。除仅SHA相符，必须证明运行进程实际加载这些路径 |
| G0-4协议与资源 | KFC正常、HCCL独立探针正确（不与MemFabric池同时启用），端口8581无残留、IB探针通过，首写/stream guard打开；`MF_KFC_BYPASS`与`MF310P_V2_NOWARMUP`必须**unset** |
| G0-5本地代码质量 | `bash -n`、`python3 -m py_compile`、`pytest test_f1c_abba_contract.py`必须实际通过；另以M10各1场Stock/Fused受控短跑确认设备正确性、两Rank数据和HCCL Event包围语义，未通过不展开8场 |
| G0-6可恢复 | 替换Host/Device SO前备份旧二进制双路径和SHA，记录回滚命令；不修改外部租户或系统任务；故障退出后不得自动切下一任务 |

**本审查是代码/流程准备，并非我已远程核验以上实机条件。** OpenCode必须先提交G0结果并现场确认后才能启动正式脚本。

## 4. 操作流程（逐字执行前请替换现场真实路径）

```bash
cd /home/w00804146/310P/vllm-ascend
git fetch origin
git switch perf/310p-f1c-stock-fp16-hccl-vs-memfabric-abba-20261025
git pull --ff-only

# G0: 先源化CANN9.1和MemFabric环境，并人工核实dev6/7独占、部署身份与备份。
bash -n csrc/_310P/memfabric_mm_ar_v2/test/perf/f1c_run_abba.sh
python3 -m py_compile csrc/_310P/memfabric_mm_ar_v2/test/perf/f1c_arm.py csrc/_310P/memfabric_mm_ar_v2/test/perf/f1c_analyze.py
python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_f1c_abba_contract.py

export ASCEND_RT_VISIBLE_DEVICES=6,7
export MF_SDMA_ORCH_JSON=/真实安装路径/libmf_sdma_orch_v11.json
export V2_HOST_LIB=/真实安装路径/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export F1C_DEVICE_SO_PATH=/真实安装路径/libmf310p_v2_trial_device.so
export F1C_CORE_SO_LIB64=/真实安装路径/libmf_hybm_core.so
export F1C_CORE_SO_PY=/另一安装路径/libmf_hybm_core.so
# ONLY after G0 hardware, binaries, diagnostic ABI, restore plan and short smoke all PASS:
export F1C_G0_APPROVED=YES F1C_EXCLUSIVE_APPROVED=YES F1C_RESTORE_PLAN_VERIFIED=YES
bash csrc/_310P/memfabric_mm_ar_v2/test/perf/f1c_run_abba.sh /tmp/opencode/f1c_m10_001 10
```

上述二进制哈希硬门槛由脚本自动校验；如现场因服务器恢复态不符导致拒绝运行，应先备份并受控部署，**不能删除哈希检查以求通过**。每场15分钟超时、30秒强杀宽限；任何错误/挂死即STOP并隔离设备，保存Rank日志、NPU/HCCL/KFC/IB现场，不自动重试或删除离群数据。

**重要先行环节**：正式8场前短跑Stock和T1各一场，既证明FP16 Stock位精确与两套后端不共进程，也通过设备Timeline/必要Trace确认Stock HCCL的stream Event尾事件覆盖真正AllReduce完成。不能因Event看起来比Wall小就主观认定测量正确。独立取证文件放入本次输出目录`route_proof/`；审计者确认后创建`audit.json`，其Schema：

```json
{
  "stock_fp16_hccl_verified": true,
  "memfabric_t1_only_verified": true,
  "no_simultaneous_hccl_memfabric": true,
  "hccl_event_stream_bracketing_verified": true,
  "evidence_files": ["route_proof/stock_trace.txt", "route_proof/fused_trace.txt"]
}
```
以上true**只允许凭实际Trace/运行日志作证后填写**；不能为了让分析器产生GO而预设。若任一项不明，保持`f1c_provisional.json`的`PROVISIONAL`状态，不能产出最终GO。

一旦审计证据齐备，再运行：

```bash
python3 csrc/_310P/memfabric_mm_ar_v2/test/perf/f1c_analyze.py \
  --sessions /tmp/opencode/f1c_m10_001/sessions --m 10 \
  --audit /tmp/opencode/f1c_m10_001/audit.json \
  --out /tmp/opencode/f1c_m10_001/f1c_audited.json
```

## 5. 门槛与下一项行动

**F1C_OPERATOR_GO（仍非真实模型GO）**：两轮ABBA每轮各自输出吞吐等效率（Stock.p50 / T1.p50 −1）均≥5%，Fused整体配对Event P90 ≤Stock的1.05倍；两Rank位精确、无协议status异常/挂死/样本缺失；完整审计证据证明Stock FP16 HCCL和T1 MemFabric各自正确命中，尤其HCCL跨流Event时间有效。

任一不达标则`NO_GO_OR_MORE_EVIDENCE`：优先诊断MM启动、通信同步和HCCL测量问题，不直接改核或协议。若真实GO，才研究生产路由、首写非Graph-safe、长期Decode稳定性；**不得将当前仅诊断T1成功视为真实模型随即可用**。

此后真实模型三方固定：Qwen3.6-35B-A3B-w8a8，input4096/output2048/TP2/batch上限10/Eager，记录实际Decode batch；与历史输出512或F1-1 0.340vs0.408 host同步数字严格隔离。

## 6. 结束交付与恢复

- 保存分支Commit SHA、G0检查与真实部署指纹、`sessions/01..08/`全Rank JSON、每场session.log、前后`npu-smi`、被诊断的HCCL/T1路由证据、机器码/扩展SO哈希、`f1c_provisional.json`及有条件的`f1c_audited.json`。
- 记录Stock/T1配对p50/p90/p99、Host Wall−Event台阶、两轮交叉的收益方向、HCCL额外FP32转换已被移除的证据。
- 最终退出前检查双Rank/KFC/HCCL资源，恢复Host SO双安装路径`88d39285...`和Device SO`b8ca96f0...`到现场原状（具体以G0备份为准），复核SHA、没有残留进程。KVER11部署按现有正确性基线保留。
- 所有报告和**脱敏原始日志**提交到本F1C分支，尤其此前F1-1欠缺的`f1_t1_stock.log`，如果服务器仍在，应补归档；不可伪造缺失数据。
- 若硬件/环境问题阻止测试，输出`F1C_G0_NO_GO_REPORT`并保留证据，不得为了测试指标修改Stock默认路由或C3源代码。

## 7. 两轮研发静态审视要求

R1代码/资源合同：`git diff`确保相对F1-1没有修改正式Kernel/Host ABI/协议；各进程只有一种通信数据面，T1预建池后完全不执行HCCL；两套权重与轮换输入一致、bit-exact/status每次必查；FP16原生Stock不在计时窗口多一次float/half转换。

R2数据合同：8场ABBA、每场2Rank×50样本、失败/缺失/脏版本一律拒绝，按index先max再p50，p90和每轮效果独立计算；GO必须依赖实际路由与HCCL stream/bracketing审计文件，而不是人工信任输出。仓库自测UT通过不等于真机G0通过。
