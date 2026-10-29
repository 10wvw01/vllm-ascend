# G1-C1：310P3 首停点设备阶段追踪 · 实施说明与执行门禁

> 研发基线：G1-B 诊断分支 `8f1eef3b` + G1-C 分级设计。  
> **职责边界**：本文件中的方案、Kernel、Host 代码、脚本与测试由方案侧提供；OpenCode 仅负责实机编译、依既定矩阵执行、环境兼容修复和证据归档。  
> **硬限制**：MemFabric 原版 KVER=11 不得修改；当前独立代码仅完成 GitHub 静态审查，AscendC/Bisheng 构建和设备侧 GM 可观测性尚待实机验证。

## 1. 为什么不同时升级 MemFabric

G1-B B02 在第 29 个完成 HTTP 请求后出现 TP0/TP1 507014；但是 ReqRing 总体无持续积压，已退出的 50 个 Epoch 均在约 15s 正常结束，直至 Host 首错附近仍能采样。源码也说明 AICPU **先推进 ReqHead 再批次敲 Doorbell**，所以 ReqPending=0 并不能证明全部 SDMA 完成。真正首停点仍可能在 V2 MM/IB、signal/wait/quiet、Done、设备侧 SQ 或异步下游算子。

先用仅修改 V2 的 G1-C1 Kernel 捕获 **Rank×Wave×Core×Chunk×Phase**。只有阶段停在 SDMA/Done 等待且证据不足时，后续再单独研发 G1-C2 的 MemFabric 设备 SQ/Stamp 埋点；不为本轮盲目升级到 KVER12。

## 2. 构建设计（独立诊断、默认关闭）

| 路径 | 内容 |
|---|---|
| `src/op_kernel/mf310p_v2_g1c_trace.h` | 64B 共享字段定义与阶段号（设备侧）；校验词用于丢弃撕裂快照 |
| `src/op_kernel/mf310p_v2_g1c_kernel.asc` | 新符号 `mf310p_v2_g1c_launch_async`；原 V2 执行体只增加 8 核阶段戳 |
| `src/op_host/mf310p_v2_trial_adapter.cpp` | 构建开关开启且运行时已绑定时选择新 Kernel；普通路径仍使用原 Kernel |
| `src/torch_adapter/mf310p_v2_serve.cpp` | 在 MemFabric 建池前分配对齐 8×64B 诊断 GM，双 Rank 首用预热后启动观测 |
| `src/torch_adapter/mf310p_v2_g1c_observer.cpp` | Worker 独立线程每约 0.5s D2H 取样，记录每核阶段及完整 Host 波提交元数据到 JSONL；poll_begin 先 fsync |
| `test/g3_e2e_scripts/g1c_trace_analyze.py` | 逐 Rank、每核、每波最后可信快照解析 |
| `test/g3_e2e_scripts/g1c_observability_gate.py` | 短时双 Rank 8 核可观测性与 D2H 时延门禁 |
| `test/g3_e2e_scripts/gate_e_forensic_verdict.py` | Gate E 部分完成请求、AICore timeout 不再误判成成功 |

- CMake 开关：`-DVLLM_ASCEND_310P_G1C_TRACE=ON`（默认 **OFF**）。设备和 Host 组件必须同时开启；运行时必须显式 `MF310P_G1C_TRACE=1` 才会绑定追踪路径。
- 追踪 GM 在建池前分配，每核独立 64B 对齐，缓冲生命周期直至 Worker 退出；不使用旧 `status`，不改 Torch Op、V1/V2 公开 ABI、Mailbox 协议及 SQE 配方。
- 设备进度为最新完成的 Stage 写回快照，不是线性一致事件历史。Host 同步 D2H 能否在 Kernel 未结束时正常返回，**必须先经 G2 短测确认**。对没有校验的撕裂样本不作归因。
- 提交的 `weight_ordinal` 是每个 Rank 首次见到某个 weight 地址的稳定序号，并非 Qwen 模型源码定义的绝对层号；与 `rank/seq/rows/chunk/slot` 联合判读。

## 3. Pull 最新代码（只动 vllm-ascend 的独立工作树）

```bash
VLLM=/home/w00804146/310P/vllm-ascend
BR=feat/310p-g1c-phase-trace-20261020
WT=/home/w00804146/310P/vllm-ascend-g1c

git -C "$VLLM" remote -v
git -C "$VLLM" worktree list --porcelain
git -C "$VLLM" fetch origin "refs/heads/$BR:refs/remotes/origin/$BR"
# 仅当 WT 和 local/g1c-20261020 不存在时执行；不得覆盖旧工作树：
git -C "$VLLM" worktree add -b local/g1c-20261020 "$WT" "origin/$BR"
git -C "$WT" status --porcelain=v1 -uall
git -C "$WT" rev-parse HEAD
```

**MemFabric 本轮不必更新源码，不需 `git pull`、不需编译/安装新的 MemFabric。** 直接使用服务器已恢复的 KVER11 与原版 Host SO。任何差异先反馈。

## 4. 实机执行 Gate（严格顺序；上一关失败即停止）

### G0 · 环境及源代码冻结

- 读原 B02/F01/S01 完整本地证据，记录时间、设备异常前最后有效消息。
- 全 worktree Git 状态、运行库双位置 SHA256、KVER11 launch JSON、python 实际导入路径、运行设备占用核实；原版 Host SO 预期 SHA256 `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`。
- 只读收集 G1-B 每纪 `consumed` 峰值（非均值）、`sqHeadAfter` 与请求流量关系；上报阶段前的 B02/F01 29 请求现象是否对应某个 V2 波/权重序号。不要把 29 个 HTTP 请求误算为 29 次融合算子。
- 取消 G1-B 的 `MF_SDMA_ORCH_DIAG` / Python 日志注入，禁止联开 G1-B+G1-C。

### G1 · CPU 静态校验、隔离构建及动态库身份校验

从 `$WT` 执行（保持系统原 CANN/torch-npu/MemFabric 环境）：

```bash
cd "$WT"
python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_g1c_kernel_isolation.py
python3 -m unittest discover -s csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts -p 'test_*g1c*.py' -v
python3 -m unittest discover -s csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts -p 'test_gate_e_forensic_verdict.py' -v
bash -n csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/{g1c_smoke.sh,gate_e_forensic_run.sh,gate_e_eager.sh}
```

参照已完成的 V2 G3 构建环境，**只在 G1-C 工作树**配置与构建 `_C_ascend`，示例（目录必须全新，不能覆盖既有构建）：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1
# 同时正确导出 ASCEND_HOME_PATH、MEMFABRIC_HYBRID_HOME_PATH、SOC_VERSION=ascend310p3
mkdir -p build/g1c-temp
printf 'VLLM_ASCEND_310P_G1C_TRACE:BOOL=ON\nVLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS:BOOL=ON\n' \
  > build/g1c-temp/CMakeCache.txt
python3 setup.py build_ext --inplace --build-temp build/g1c-temp
```

本项需实测核实 `nm -D` 的 `mf310p_v2_g1c_launch_async` 设备库导出与 Host SO 对应未定义符号能完整解析。核查 `python3` 实际导入 `vllm_ascend` 来自 **`$WT`** 而非旧 editable 工作树；若不是，先修正导入路径并记录证据，禁止启动长稳。不要执行 `pip install -e` 或覆盖原版 site-packages，除非说明隔离与回滚步骤并得到确认。

### G2 · 正式实机矩阵和观测门禁

1. 先做完整的构建符号检查和无诊断环境的 TP=2 最小正确性；T0、Gate B 及既有 `tp2_v2_mixed_route_correctness.py` 小规模位精确验证按原回归入口执行。没有完整通过，不得进入 620s 长稳。
2. 执行短时 G1-C 真实 Qwen serve 测试（自动单独落日志）：

```bash
cd "$WT"
unset MF_SDMA_ORCH_DIAG MF_SDMA_ORCH_EPOCH_BUDGET_S
export G1C_SMOKE_DURATION_S=180
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1c_smoke.sh G1C_SMOKE1
```

脚本会自动设置 `MF310P_G1C_TRACE=1`、为会话创建 `g1c/` 证据目录和 `PYTHONPATH` 前置（保留已有 CANN 路径）；完成后运行 `g1c_observability_gate.py`。
3. 门禁必须看到双 Rank 各 8 核真实 `seq`、阶段戳与 `submit` 对应；D2H 至少 2 次完成、无错误、无超过 1000ms 的阻塞，且无损坏快照。验证设备运行中数据在不同 `seq` 上推进，Host JSONL 已 fsync。若所有阶段戳只出现在 Kernel 退出后，**不能凭该轮证实实时可观测性**，须停下反馈并调整设计。
4. 验证诊断前后的数值结果仍 bit-exact，收集统计的性能变化和额外 Host 开销，必要时降低采样频率；不允许把时序干扰下的未复现当作稳定性修复。

### G3 · 正式 Gate E 长稳（G2 全通过后）

```bash
cd "$WT"
unset MF_SDMA_ORCH_DIAG MF_SDMA_ORCH_EPOCH_BUDGET_S
export MF310P_G1C_TRACE=1
export GATE_E_DURATION_S=620
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/gate_e_forensic_run.sh fused G1C_F01
```

失败时先保留原始现场；不得强制切分失败日志、删除 JSONL、清理外部进程或重装库。需要重复时新会话编号 `G1C_F02`，决不能覆盖已有结果。 `gate_e_forensic_verdict.py` 将部分完成请求判 FAIL，整体退出码不再等于旧 runner 的虚假 0。

### G4 · 离线首故障点判读和报告

```bash
HERE="$WT/csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts"
OUT="$WT/csrc/_310P/memfabric_mm_ar_v2/tmp/opencode/gate_e_forensics/G1C_F01"
python3 "$HERE/g1c_trace_analyze.py" "$OUT/g1c" > "$OUT/evidence/g1c_trace_summary.json"
python3 "$HERE/gate_e_forensic_verdict.py" "$OUT" --prompts-per-rep 25 \
  > "$OUT/evidence/final_verdict.json" || true
```

将每个 Rank 的最后 `submit(seq,weight_ordinal,slot,rows)` 对齐其 8 核 `(seq,chunk,phase)`，再结合双 Rank Host 首错与 B02 Epoch/设备告警时间线。输出 **首异常候选阶段**，明确观测可信程度；不得把 `SwiGlu` 后置报错或仅一条 `last stage` 当作确定根因。

## 5. 交付和回滚

- 原版 MemFabric SO、KVER11 不变；仅在 G1-C Worktree 构建/加载 V2 诊断算子。测试结束清除会话级 `MF310P_G1C_TRACE`，使用原工作树及其既有 native SO 即回到原 V2 环境，禁止直接在原工作树替换库。
- 证据 `tmp/opencode/gate_e_forensics/G1C_*/` 仅在服务器保存；若要上传供深审，先保留完整原件，再生成脱敏小包/摘要（地址、用户路径、token 均可能敏感）。
- OpenCode 报告须包含：代码 SHA、编译命令及设备 SO/Host SO 哈希、每关实际命令与结果、失败现场、8 核阶段追踪表、上下游证据、未解决项。技术方案和下一轮修复由方案负责人决定。
