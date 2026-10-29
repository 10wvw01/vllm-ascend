# V3.1 编译准入、五关门禁、六配置短测与包长诊断（2026-10-29）

执行：OpenCode（按 `V31_ADMISSION.md` 与 `OPENCODE_EXECUTION.md` 授权）。
范围：隔离环境 → CPU 合同 → 私有 MemFabric SDK → 私有 V3.1 编译（dav-2002）→
设备五关门禁（38144 双 Die TP=2）→ 4MiB 默认六配置两轮短测 → 1/2MiB 有界包长诊断。
只运行 V3.1；历史对照只读取已归档 V3.0/V2/Stock 数据，不重跑旧基线、不跑模型/TTFT/Profiler。

## 总结论

| 阶段 | 结果 |
|---|---|
| CPU 合同（V3+V2 UT） | 88 passed（前缀 HEAD）；修复后 89 passed（含新增守护窗口回归） |
| 私有 MemFabric SDK（db38d75e） | PASS；AICPU 三件套 + `orch_deploy_helper` 齐全（23 文件） |
| V3.1 私有编译（dav-2002） | PASS；双 UB、NZ2ND 临时区 static_assert、MTE2_MTE3/MTE3_MTE2 事件、Progress 数组全部编译核验 |
| 设备五关门禁 | 双 Rank 全 PASS：ib → add → protocol（含回绕）→ fused → tail；四个设备会话各独立通过 |
| 4MiB 默认六配置两轮短测 | r1（前缀 HEAD）与 r2（修复 HEAD）均 PASS；跨会话复现性 ≤1.77% |
| 1MiB 诊断轮 | 前缀 HEAD 被**守护器误报 STOP(97)**（见 BUG 记录）；修复 HEAD 重跑 PASS |
| 2MiB 诊断轮 | 修复 HEAD PASS |
| 包长结论 | 1/2/4MiB 逐配置差异 ≤~1.5%（会话噪声量级），对本组形状无实质影响 |
| 相对历史基线 | **未超过历史 V2**：M2048 比 V2 D4 慢约 21~22%；M4096 与 V2 D4 打平（−0.2%~+0.9%，在漂移带内，不可判胜）。距真实 NZ Stock：M2048 还需缩短约 34~36%，M4096 约 20~24%。相对 V3.0 有明确内部收益（g1024 M2048 −12.7%、g1024 M4096 −23.9%、g2048 M4096 −12.7%） |

## 指纹

| 项 | 值 |
|---|---|
| vllm-ascend 分支 | `work/310p-mm-ar-v3-sdma-add-20261029` |
| 前缀（交接锁定） | `632370a0396f5a7390e8653e320ff5fa143be39c` |
| 修复提交 | `2a5f3b37e04b0dd5036ae37d8fed5d8628351825`（仅守护器/worker 窗口语义，设备 .so 逐字节一致） |
| MemFabric | `fix/310p-v3-orch-deploy-helper-20261029` @ `db38d75e8485d31451302219eb7a1bfeb1dd107c`（tree `4fe624cab…`） |
| 私有 SDK 助手 SHA256 | `849f93c0b4a96877d3e839ae7ceec4b6c479a8de331213f0d7a270c1198bd310` |
| 环境 | aarch64 / CANN 9.1.0 / Python 3.12.13 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 |
| 设备 | 310P3 卡 38144 dev4/5（TP=2 单卡双 Die），全程健康 OK，无外部租户 |
| 隔离根目录 | `/tmp/opencode/310p-v31-20261029-001`（全新目录，未触碰共享 Serving 环境） |

## 轮次时间线与 BUG 记录

### R1（00:35–00:38，`default-r1-pass/`）：前缀 HEAD 默认 4MiB 轮 — PASS

CPU/SDK/私有编译全部通过后（本仓库 CPU 88 项），G0（38144 双 Die 健康、
`proc-mem` 无进程）成立，quick 全链路 PASS：五关门禁 + 六配置 × 2 轮 ×（4 预热 + 30 样本）。

### R2（00:39–00:40，`diag-1mib-guard-stop/`）：1MiB 诊断轮 — 守护器误报 STOP(97)

**现象**：建池部署窗口内，目标卡出现第 5 个进程（宿主 PID `4158731`，die 1，~103MB），
超过守护器窗口容忍上限（2 Rank + 2 helper = 4），fail-closed STOP。
无 ADD 发出、无协议错误、设备健康、无外部租户（前后采样与 nputools 日志均无外部进程）。

**根因（三层证据链）**：

1. torch_npu 在 worker **首个设备算子**处初始化其异步编译基础设施：fork 出短命 build 子进程
   （存活 ~8s）+ multiprocessing forkserver + 最多 7 个池 worker（受控复现见
   `diagnostics/build-probe3/tree.txt`）。build 子进程从已打开设备的 worker fork，
   继承 ~103MB 设备映射（与 `results-diag-1mib` STOP 快照中第 5 进程的 die/内存吻合）。
2. `kernel_meta/buildPidInfo.json` 在 00:40:01.89 记录了两个 worker 的 build 进程
   （`kernel_meta_4783940680449382290` / `kernel_meta_11417038517589120610`），
   宿主 PID `4158731` 恰落在 worker（4158452/53）与 helper（4159647/49）之间的分配窗口。
3. `nputools_LOG_ERR` 摘录（`diagnostics/`）显示守护器 00:40:05 采样仍为 4 进程、
   00:40:08 采样出现第 5 进程；设备侧 plog/device 日志（`diagnostics/device-logs/`）
   显示 helper 的 KFC 通道 00:40:06.5 初始化（与 V3.0 r3/默认轮完全相同的告警形态，
   属已知良性）。独立运行 helper 探测（rc=3 快路径，`diagnostics/kfc-probe-diag/`）
   与单进程算子复现均未出现第 5 进程——该浮现由驱动进程表异步记账触发，低概率但非零。

**结论**：第 5 进程是**本会话自有**的 torch_npu 编译基础设施子进程，非外部占用、
非设备故障、非协议错误。守护器对“合法自有形态”的建模不完整 → 误报。

**修复（`2a5f3b37e`，对齐 V3.0 R2→R3 守护器演进先例）**：

- `quick_worker.py`：窗口标记从建池前提前到首个设备算子前声明，覆盖编译基础设施
  子进程整个存活期 + MemFabric 部署助手；V3Session 返回/异常/停止路径统一撤销标记。
- `stage_guard.py`：声明窗口内容忍上限 2+2 → 2+2+2=6（新增 ≤2 个编译基础设施形态）；
  60s 持续界限、窗口外严格 2、健康/跨卡/解析 fail-closed 全部不变。
- `test_v3.py`：新增窗口内 5 进程容忍与第 7 进程立即 STOP 回归；CPU 89 passed。
- 设备 .so 逐字节不变（`v3-build` vs `v3-build-fix2` manifest 对比），性能口径零变化。

### R3（01:07 起，修复 HEAD）：三轮全部 PASS

- `default-r2-pass/`：默认 4MiB 复验轮（门禁 + 六配置 × 2 × 30 样本）PASS；
  与 r1 跨会话最大偏差 1.77%，证明守护器修复未扰动测量。
- `diag-1mib-r2-pass/`：1MiB × 六配置 × 2 ×（4 预热 + 10 样本）PASS。
- `diag-2mib-r2-pass/`：2MiB 同口径 PASS。
- 会话间 38144 均复核无进程；结束 `npu-after.txt` 卡干净。

## V3.1 性能数据（p50 µs，逐样本双 Rank max；p90/p99 为两轮中较差者）

### 4MiB 默认轮（r1/r2 为两个独立会话，各含两轮）

| M | group | r1 p50 | r2 p50 | p90(r2) | p99(r2) | wall_p50(r2) | 轮内漂移 |
|---|---|---|---|---|---|---|---|
| 2048 | 512 | 1884.1 | 1878.9 | 1981.8 | 3076.7* | 1961.6 | 0.32~3.81% |
| 2048 | 1024 | 1666.7 | 1656.0 | 1709.3 | 1922.5 | 1724.7 | 1.25~1.96% |
| 2048 | 2048 | 1690.4 | 1670.1 | 1729.3 | 1765.4 | 1758.5 | 0.75~1.52% |
| 4096 | 512 | 3192.6 | 3136.1 | 3291.2 | 3461.2 | 3225.7 | 0.20~0.85% |
| 4096 | 1024 | 2545.6 | 2560.3 | 2671.5 | 2713.8 | 2661.7 | 1.19~1.20% |
| 4096 | 2048 | 2525.8 | 2516.1 | 2591.9 | 2652.9 | 2608.2 | 0.54~1.11% |

\* r1 M2048g512 r1 轮 p99=3076.7 出现一次离群（该轮 p99 中位为 2012.8）；
r2 同配置 p99=2012.8 正常，判定为单样本离群而非系统性长尾。

### 包长诊断（修复 HEAD，每配置 4 预热 + 10 样本 × 2 轮，独立结果目录，不计入默认统计）

| M | group | 1MiB p50 | 2MiB p50 | 4MiB p50(r2) | 结论 |
|---|---|---|---|---|---|
| 2048 | 512 | 1906.9 | 1857.5 | 1878.9 | 噪声级 |
| 2048 | 1024 | 1631.7 | 1629.5 | 1656.0 | 1/2MiB 略优（~1.5%） |
| 2048 | 2048 | 1710.8 | 1714.1 | 1670.1 | 4MiB 略优（~2.5%） |
| 4096 | 512 | 3136.4 | 3165.1 | 3136.1 | 噪声级 |
| 4096 | 1024 | 2543.3 | 2564.2 | 2560.3 | 噪声级 |
| 4096 | 2048 | 2536.6 | 2519.4 | 2516.1 | 4MiB 略优（~0.8%） |

**包长结论**：[1,4]MiB 内包长对本组形状无实质影响；4MiB 默认合理。
最佳组合：M2048 → chunk1024（4MiB 或 2MiB），M4096 → chunk2048（4MiB）。

### 与历史基线对照（p50，跨批对照，不签正式性能 GO）

| M | group | V3.1 | V3.0 r3 | V2 D4 (R2 screen) | Stock (E3 真实 NZ) | V3.1 vs V3.0 | V3.1 vs V2 D4 | V3.1 vs Stock |
|---|---|---|---|---|---|---|---|---|
| 2048 | 1024 | 1656.0 | 1897.6 | 1367.9 | 1078.4 | −12.7% | **慢 21.1%** | 慢 53.6% |
| 2048 | 2048 | 1670.1 | 1709.9 | 1367.9 | 1078.4 | −2.3% | **慢 22.1%** | 慢 54.9% |
| 4096 | 1024 | 2545.6 | 3343.0 | 2539.8 | 2022.6 | −23.9% | 平（+0.2%） | 慢 25.9% |
| 4096 | 2048 | 2516.1 | 2883.2 | 2539.8 | 2022.6 | −12.7% | 平（−0.9%，漂移带内） | 慢 24.4% |

- **是否超过历史 V2：否。** M2048 全配置仍比 V2 D4 慢 21%+；M4096 最好配置与
  V2 D4 在 ±1% 内打平（跨批对照 + 轮内漂移 0.5~1.2%），不能判为超过。
- **距 Stock 差距：** M2048 需再缩短 ~34%（1656→1078），M4096 需 ~20%（2516→2022）。
- 保留限制：跨批/不同进程形态/SDK KVER28 占位自拷贝差异（`sdk_manifest` findings）与
  历史 Screen 判定限制不变；本次仅为算子级短 Screen。

## 复现命令（38144 双 Die 独占窗口内）

```bash
export V3_RUN_ROOT=/tmp/opencode/310p-v31-20261029-001   # 本轮根目录（证据已随仓库分发）
V3_PYTHON=$V3_RUN_ROOT/venv/bin/python
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd <vllm-ascend@work/310p-mm-ar-v3-sdma-add-20261029 @2a5f3b37e>
# 默认轮（4MiB、30 样本）
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/quick.py start \
  --manifest $V3_RUN_ROOT/v3-build-fix2/manifest.json --card 38144 --devices 4,5 \
  --out $V3_RUN_ROOT/results-default-r2 --timeout 900 --lease-note '<真实独占窗口说明>'
# 包长诊断（独立目录，不计入默认统计）
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/quick.py start \
  --manifest $V3_RUN_ROOT/v3-build-fix2/manifest.json --card 38144 --devices 4,5 \
  --out $V3_RUN_ROOT/results-diag-1mib-r2 --samples 10 --packet-bytes 1048576 \
  --timeout 900 --lease-note '<真实独占窗口说明>'
```

## 证据索引（本目录内，均可由仓库直接获取）

| 证据 | 位置 |
|---|---|
| 环境与 CPU | `manifests/python-environment.txt`、`manifests/cpu-tests.log` |
| SDK 构建 | `manifests/sdk_manifest.json`、`manifests/version.info`、`manifests/memfabric_build.log`、`manifests/orch_deploy_helper_build.log` |
| V3.1 构建 | `manifests/v3_build_manifest_prefix_632370a.json`、`manifests/v3_build_manifest_fix_2a5f3b37e.json` |
| 设备健康/占用 | `npu-before.txt`、`npu-before-fix1.txt`、`npu-after.txt`、`health-before.txt` |
| 失败轮（守护器 STOP） | `diag-1mib-guard-stop/`（`lease_guard.jsonl`、`lease_guard_stop.npu.txt`、残留 `deploy_window_rank*.flag`） |
| BUG 根因诊断 | `diagnostics/`（kfc 探测、单进程复现、进程树/卡表轮询、nputools 摘录、helper plog 与设备侧日志） |
| 各轮完整输出 | `default-r1-pass/`、`default-r2-pass/`、`diag-1mib-r2-pass/`、`diag-2mib-r2-pass/`（含 gate/timing/rank/REPORT/historical_comparison/lease_guard） |
| 汇总分析 | `analysis/summary.json` |

原始 JSON/日志合计 ~1.6MB，全部随本提交入库；服务器侧完整根目录
`/tmp/opencode/310p-v31-20261029-001` 保留至下轮实验开始（含 venv/源码/构建产物）。
