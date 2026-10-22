# V2 MTE2/Scalar Gate C：4 Chunk 受控隔离诊断实施及实机验收（2026-10-22）

> **当前状态：代码与静态审视完成；尚未在 310P3 编译/运行，不得标为实机 PASS。**  
> 分支：`diag/310p-v2-gate-c-controlled-isolation-20261022`，基于 Gate A/B 核验分支 `diag/310p-v2-mte2-scalar-gate-ab-20261022@7a1a0893`。  
> Gate A/B 证据：`V2_MTE2_SCALAR_GATE_AB_VERIFICATION_REPORT_20261022.md`；Host SO `88d39285...`、KVER11 JSON `25e16b0a...`、插件 `929b53c2...`。版本仍以现场重新核验为准。

## 1. 为什么是 4 Chunk，而不是新增算子形态

真实 Qwen3.6 Eager Prefill 在测量点 `M=4096`、`chunk_rows=1024`，因此**4 个 Chunk**；8 个 AI Core 全员 MM，Core0 signal，正式 `overlap=1`。

本阶段不改变正式执行路径。我们需要一个**同输入、同输出规模和同 KVER11**，但故意取消 Chunk 间错峰的 LOOP 串行诊断，用于区分“多 Chunk 执行形态本身”与“交叠实现”；不能凭一个计时差精确拆分 SDMA、IB、缓存维护的绝对时间。

## 2. 源码实现与边界

- 新增诊断专用 C API：`mf310p_v2_trial_diag_loop4(handle,x,w,y,stream)`，实现在 `src/op_host/mf310p_v2_trial_diagnostics.cpp`，声明在 `mf310p_v2_trial_diagnostics_api.h`。
- 仅允许**已有 handle** `chunk_rows=1024`，固定 chunks=4、overlap=0；无效 handle、stream、设备指针返回 -1，不启动 Kernel。
- 只调用**既有** `mf310p_v2_diag_loop_async`；新增小型首次符号/输出地址保护状态在内部 handle 中；每次发射前清零 64B status。
- 保留全部 MM、CleanLines、SDMA signal/wait、IB、Add、quiet 和 DONE。LOOP 与正式核**只有协议调度形态不等价**：每个 Chunk 完全串行后再处理下一 Chunk。
- 旧 `mf310p_v2_trial_diag` 单 Chunk API 和所有现有参数、数值语义不变。
- 正式 `mf310p_v2_trial_run`、正式 `mf310p_v2_kernel.asc`、Torch Serving 路由、MemFabric Host/Device 均未修改。
- `diag6`（IBDONE16）虽然有底层 chunks 参数，但设备 Kernel 只处理第一 Chunk，**禁止**用作 4 Chunk 全形状对照。本阶段没有为它透传 chunks。
- `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=OFF` 时诊断 TU 不链接，正式运行时不可调用新接口。

## 3. TP2 小型执行器与正确性

新增 `test/perf/gate_c_loop4.py`：
- 每个独立 `torchrun` 会话只测**一个** case，避免不同变体互相污染消息环/输出槽；
- 使用 `torch.randn` 的固定 rank-specific 输入和 NZ 权重，分别产生 FP16 Stock allreduce 参考值；
- 单 Chunk diag0 只比对**本 Rank MM**，diag1/3/4/5 只比对第一 Chunk 的**TP2 AllReduce**；正式与 LOOP4 都比对全部 4096 行；
- `torch.equal` 位精确、status=0、IB probe 必须先于采集通过，任一失败立刻停止当前会话；
- 构建/pool/首次符号/首次输出保护全部在至少 3 次预热前完成；Profiler 窗口不加入首调；
- `--profile` 只开启 torch_npu 原生 CPU/NPU profiler，采集独立设备 Kernel，而不是写新的 AI Core 时间戳；
- 原始 Trace 每 Rank 保留本地；脚本拒绝覆盖已有实验 JSON 或 Trace。

`test/perf/analyze_gate_c_trace.py` 离线要求每例 Rank0/Rank1 完整，且每个 Rank **恰好 `iters` 个对应 Kernel**；不满足则拒绝给出性能结论。

### 可比较矩阵

| case | M | chunks | 实际语义 | 可用于 |
|---|---:|---:|---|---|
| formal | 4096 | 4 | 正式核 overlap=1 | 唯一真实 V2 Kernel 形态锚点 |
| loop4 | 4096 | 4 | LOOP 串行 overlap=0 | 4 Chunk 隔离对照、交叠有效性方向 |
| diag0 | 1024 | 1 | MM-only | 单 Chunk 数学/访存基准 |
| diag1 | 1024 | 1 | NoIB | 单 Chunk 协议危险变体，仅经正确性门禁后定性 |
| diag3 | 1024 | 1 | IB | 单 Chunk IB 条件对照 |
| diag4 | 1024 | 1 | IBDONE | 单 Chunk DONE 条件对照 |
| diag5 | 1024 | 1 | 正式 Kernel 串行 | 单 Chunk 正式路径锚点 |

**禁止**：把 `4 * diag0` 叫作全形状 MM 的精确耗时；把 `loop4-formal` 当成严格 overlap 纯收益或上界；把 NOIB/IB 差额当作正式 Kernel 内 IBWait 时长；将短窗口设备核数据直接外推 50×2048 的真实 Serving。

## 4. OpenCode 实机步骤（按门禁顺序；本分支不自动执行）

**C0 设备准入**：独占 5664 双 die(dev0/dev1)；保存 git SHA、现装 Host 双路径 SO SHA、插件 .so、KVER11 JSON、设备侧 v11 helper/Trace 间接身份；确保无 v12 动态加载、无 G1-D 诊断环境残留。记录变更前备份及恢复办法。不得覆盖 Gate A/B 原始 Trace。

**C1 构建/静态**：诊断 ON 编译（与 Gate A/B 同环境）；检查 `nm -D` 新诊断符号存在且原有 C ABI 仍存在；诊断 OFF 构建不应引用该符号。运行 CPU UT：
```bash
python -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_gate_c_source_contract.py
python -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_v2_source_separation.py
```

**C2 双 Rank 正确性**：每次新进程依序尝试 `formal`、`loop4`，全部 bit-exact 后才尝试 `diag5`、`diag0`、`diag3`、`diag4`；`diag1`（NoIB）需额外确认两 Rank 预备和协议安全后再单独尝试，不得放在无人值守全矩阵中。运行例：
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/perf/gate_c_loop4.py \
  --case formal --out /tmp/opencode/gate_c_c2/formal
```
`loop4` 请分别开新进程并改 `--case loop4 --out /tmp/opencode/gate_c_c2/loop4`。

**C3 性能短窗口**：先确定 C2 全通过，再对 formal、loop4 各运行新进程（每 Rank 3+ 次预热、6+ 个有效样本）。根据证据需要小规模追加单 Chunk 案例。每个案例新增 `--profile`，输出隔离目录。示例：
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/perf/gate_c_loop4.py \
  --case formal --out /tmp/opencode/gate_c_c3/formal --profile --warmup 3 --iters 6
python csrc/_310P/memfabric_mm_ar_v2/test/perf/analyze_gate_c_trace.py \
  --root /tmp/opencode/gate_c_c3 --out /tmp/opencode/gate_c_c3/summary.json
```

**C4 一次性额外风险**：首次符号/新输出会发生 310P3 MTE3 首写保护同步，可能 0–15s；必须双 Rank 同时启动、排除预热税。不能为了“快”设置 `MF310P_V2_NOWARMUP` 绕过保护。

## 5. 强制停止条件

任何设备占用/环境漂移、507014/507018、协议状态非零、首调位精确失败、正/诊断库路径混淆、真实 Kernel 未出现、Profiler 设备 Trace 不全或时间线/样本数不符——立即停止、保存 plog/双 Rank 日志及二进制哈希，不继续下一 case。

`diag1` 的 NoIB 本身省略核间可见性保证，**如果不能证明在当前实验条件下具备合理安全边界，直接标记 NO-GO/不运行**，不用为凑齐矩阵去触碰正式运行时。

## 6. 静态审视（两轮，不等于实机证明）

**Round 1——协议、ABI、越界、生命周期**
- 新 API 只传已有 LOOP Kernel，并将 `M4096` 固定 4 个有效 Chunk；旧 ABI 保持，`diag6` 没有伪扩展；
- 诊断编译开关控制；内部 handle 小幅增加独立 bool/地址字段但无公开结构布局变化；
- 各 Chunk 的 x/ownC/y 地址偏移仍由原始 LOOP Kernel 管理，Host 不重写远端布局，不省略清缓存/同步；
- 初次发射、新 y 地址继续遵守正式 MTE3 首写保护纪律；后续复用固定输出指针，不在采样区强行全设备同步。

**Round 2——实验有效性、统计、安全**
- 每例一个 torchrun 会话，rank-specific 完整随机数据防止“只计算第一 Chunk”假通过；
- 逐次同步后比对整幅结果/单 Chunk 局部参考，不能只判断 status；双 Rank 设备 Trace 按 `iters` 严格计数；
- 单 Chunk 与 4 Chunk、串行与 overlap 的方法学边界在 JSON 和本文件显式说明；
- Graph、长压和正式 Kernel 优化均不在本轮，Gate E 的 P0 依然未关闭。

**静态结果：设计和源码已审视；CANN/Bisheng 编译、链接符号、实机正确性与稳定性目前仍为 UNKNOWN，待 OpenCode C0–C3 执行。**

## 7. Gate C 验收输出

OpenCode 提交一份实机报告和脱敏证据：
- 构建/环境/SO/JSON hashes、诊断 ON/OFF 符号与恢复记录；
- formal/loop4/各单 Chunk case 的 bit-exact + status；
- Rank0/Rank1 独立 Kernel p50/p90/max，异常样本单列，Trace 可回溯；
- 哪些诊断成本改变具有稳定复现性，哪些因变体不等价不能解释；
- 对 `~0.63ms/目标层` 额外成本是否已经收窄到值得进入正式优化的可测试假设；
- 74ms 长尾是否观察到（未重现则仅记“未重现”），Gate E P0 独立维持。

**结论门禁：没有明确且安全、可验证的优化假设，就不修改正式 V2 设计。**
