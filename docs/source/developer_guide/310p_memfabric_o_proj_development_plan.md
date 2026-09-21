# 310P3 o_proj + MemFabric：AI-native 开发状态与执行计划

> 这是本融合需求的**唯一当前状态面和后续开发入口**。  
> 不在这里保留逐日 bring-up 流水账；每个验收维度只保留“当前最新有效证据”。  
> Reviewed implementation baseline: `1058cd87bb5fd92d92c96d920df4afafd093eadc`
> (2026-09-21).
>
> 读文档顺序：
> 1. [需求与验收合同](310p_memfabric_o_proj_requirements.md)
> 2. [当前架构](310p_memfabric_o_proj.md)
> 3. 本文
> 4. [构建/运行手册](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)

## 1. Machine-readable 项目快照

```yaml
project: 310p-qwen36-full-attention-o-proj-memfabric
target:
  hardware: Ascend 310P3 single-card dual-die
  tp: 2
  model: Eco-Tech/Qwen3.6-35B-A3B-w8a8
  layer: full_attention self_attn.o_proj only
  runtime_dtype: fp16
  geometry: "x[M,2048] * w[2048,2048] -> y[M,2048]"
  communication: custom wgm-dev-310p MemFabric V5
implementation:
  model_dispatch: done
  direct_producer: done
  producer_blocks: 8
  local_reduce_kernel: done
  wave_ack_gate: done
  adapter_abi: 4
  single_layer_correctness: done
  v5_overlap_evidence: done
productionization:
  epoch_lifetime_root_fix: blocked_external
  current_head_eager_e2e: todo
  acl_graph: todo
  long_run: todo
  final_e2e_performance: todo
current_phase: productionization
highest_priority: MF-001
```

## 2. 当前源码开发状态

| Area | 状态 | 当前源码事实 | 最新有效证据 | 下一步 |
| --- | --- | --- | --- | --- |
| 精确 model hook | DONE | full-attention o_proj + TP2 + 4096/2048/2048 + FP16；关闭 generic reduce | source UT + modelslim 路由 | 仅防回归 |
| Build/link | DONE | 定制 MemFabric V5 direct link；CMake 自动 bisheng 编译 .asc | 真实 310P3 build/load 已通过 | 仅防回归 |
| Persistent runtime | DONE | process context、send/recv arena、scratch、warmup、poison | 单层 repeated calls | Graph/长稳继续验证 |
| Direct matmul producer | DONE | FP16 Cube matmul 直写 send arena；8-block interleaved producer | rows 1..4096 bit-exact | 仅按数据优化 |
| Mailbox/SDMA | DONE | V5 request/arrival rings + epoch SDMA | V5 overlap evidence | 等 epoch 生命周期根治 |
| Local reduce | DONE | repo-owned multi-block `mf310pAddKernel` | bit-exact；消除 per-shape GE compile | 仅防回归 |
| Cross-wave rendezvous | DONE | first-wave barrier；later wave ack/gate | repeat stress + tail spike 消除 | Graph 中验证 |
| 单层 correctness | DONE | 1/8/32/33/64/128/512/2048/4096 | max_abs_diff=0 记录 | HEAD 改功能后重跑 |
| V5 overlap | DONE | producer / waiter / add / ack task-time 数值闭环 | commit `1058cd87bb` | 功能变化后重采 |
| 单层性能 | DONE_FOR_CURRENT_KERNEL | 2048 ~2.09 ms；4096 ~4.11 ms steady median | P6 实机结果 | 不替代 E2E 性能 |
| Qwen3.6 eager E2E | STALE | 旧实现曾跑通，但早于 V5/P6 最终 HEAD | commit `c5fd1c46f7` | MF-003 重跑当前 HEAD |
| Epoch 生命周期 | BLOCKED | 当前 V5 常驻 epoch 约 25s timeout 风险 | 实机稳定复现 | MF-001 根治 |
| Host sync audit | TODO | pool 存活后不能碰 device-wide sync/HCCL | bench 仅规避 | MF-002 |
| ACL Graph | TODO | 没有当前 fused op capture/replay 验收 | 无 | MF-004/MF-005 |
| 长稳 | TODO | benchmark 通过缩短 pool lifetime 绕开 | 无生产证据 | MF-006 |
| E2E 性能 | TODO | 缺当前 HEAD TTFT/TPOT/tokens/s | 无最终数据 | MF-007 |

状态定义：

- **DONE**：当前 HEAD 有源码 + 可复核证据；
- **DONE_FOR_CURRENT_KERNEL**：局部指标完成，但不能替代最终验收；
- **STALE**：曾有证据，但不覆盖当前关键实现；
- **TODO**：本仓库可推进；
- **BLOCKED**：依赖外部 MemFabric/kernel 部署或平台动作；
- **DEFERRED**：只有数据证明需要时才做。

## 3. 当前最新证据

只保留每个维度的最新有效结果，不追加历史流水账。

### 3.1 单层 correctness

```text
tile_m=32
rows = 1, 8, 32, 33, 64, 128, 512, 2048, 4096
verdict = all PASS
max_abs_diff = 0
multi-wave / tail / repeated calls covered
```

硬件脚本：
`benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py`

### 3.2 P6 steady latency

```text
rows     median
1        ~0.95 ms
32       ~0.49 ms
128      ~0.47 ms
512      ~0.74 ms
2048     ~2.09 ms
4096     ~4.11 ms
```

这组数据只说明当前 fused kernel microbenchmark，不是 Qwen3.6 E2E 结论。

### 3.3 V5 overlap

代表 workload：

```text
rows=2048, tile_m=32, 64 chunks, 8 MiB/rank
producer median ~1680 us
waiter median   ~193 us
add             ~130 us
cycle median    ~2078 us
```

若 8 MiB 传输全部串行落在 waiter 窗口，需要约 44 GB/s；高于当前分析采用的
约 20 GB/s sustainable SDMA 假设。串行下界也高于实测 cycle，因此当前 V5
实现已满足 overlap 验收。

分析脚本：
`benchmarks/scripts/analyze_310p_memfabric_overlap.py`

### 3.4 E2E

旧 eager E2E 在 `c5fd1c46f7` 跑通过，但它早于 V5 epoch migration、
multi-block producer、自有 add、ack/gate。**状态只能记为 STALE。**

## 4. 有序任务队列

后续 AI/工程师每次只领取一个 Task ID。不要同时改算法、Graph、E2E 和文档。

### MF-001 — 根治 MemFabric epoch 生命周期（P0 / BLOCKER）

**目标**：删除“pool 必须在约 25 秒内结束”的生产限制。

**依赖**：定制 MemFabric 仓库与设备侧 AICPU kernel 部署权限。

**已知方向**：缩短 epoch 自限循环并 bump KVER，然后重新部署设备侧 kernel；
具体值必须以 MemFabric source 和实机验证为准，不在 vLLM-Ascend 中硬编码猜测。

**验收**：

- pool 生命周期明显跨过当前 timeout 窗口；
- 不出现 507901 / AICPU 被 kill；
- shutdown 语义重新验证；
- 结论回填本状态文档和 usage 的平台限制。

**Stop condition**：没有设备侧 kernel 部署权限时停止在“可复现 + patch proposal”，
不要用缩短测试时长假装完成。

### MF-002 — vLLM host device-sync 路径审计（P0）

**目标**：建池后完整 vLLM 请求路径不得触发会等待常驻 epoch 的 device-wide sync。

**方法**：

- 从首次 target o_proj forward 之后开始审计；
- 搜索/trace `torch.npu.synchronize`、`aclrtSynchronizeDevice`、会隐式
  device-sync 的 HCCL/error/finalize 路径；
- 区分当前 stream sync 与 device sync。

**验收**：当前 HEAD eager 服务多轮请求不依赖 benchmark 专用规避逻辑。

### MF-003 — 当前 HEAD Qwen3.6 eager E2E（P0）

**前置**：MF-001 至少有可用于服务生命周期的修复版本；MF-002 完成关键路径审计。

**验收**：

- TP=2 模型加载；
- 目标 full-attention o_proj 全部命中；
- linear-attention/GDN 不命中；
- 多轮 short prompt / prefill / decode 正常；
- feature-off baseline 正常；
- 保存 commit、MemFabric commit、启动命令、日志摘要。

### MF-004 — 最小 fused-op ACL Graph（P1）

先做单 op graph capture/replay，不要直接从 35B 模型开始排查。

覆盖：

- M=1；
- M=32；
- tail M=33；
- M=2048；
- multi-wave M=4096；
- repeated replay；
- stable GVA / wave generation。

失败时只最小化 graph 边界，不改计算算法。

### MF-005 — 完整 Qwen3.6 ACL Graph（P1）

前置：MF-004 PASS。

验收：移除 `--enforce-eager` 后完整模型 capture/replay、多轮请求稳定。

### MF-006 — 长稳与生命周期（P1）

前置：MF-001、MF-003、MF-005。

目标不是固定某个漂亮时长，而是证明不再依赖 25 秒 workaround，并覆盖：

- repeated requests；
- multiple M buckets；
- process lifetime；
- graceful shutdown；
- failure poison 行为。

### MF-007 — 最终 E2E 性能包（P1）

保存同机器、同参数 baseline/fused：

- single-layer latency；
- prefill / TTFT；
- decode / TPOT；
- prompt tokens/s；
- output tokens/s；
- peak memory；
- 必要时 profiler/task-time。

### MF-008 — Decode 小 M 优化（DEFERRED）

只有 MF-007 证明 decode 是主要瓶颈时才启动。

候选：N-panel 或其它小 M tiling。禁止因为“理论上可能更快”提前扩大 kernel 复杂度。

### MF-009 — Cache clean 优化（DEFERRED）

只有 profiler 显示 64B line clean 是显著瓶颈时评估。没有数据不改。

## 5. AI-native 单次迭代协议

每一轮开发必须按以下顺序：

### Step 0 — 锁定任务

先写一个 task packet：

```text
Task ID:
Goal:
Why now:
Implementation baseline:
MemFabric baseline:
Allowed files:
Do-not-touch:
Expected evidence:
Stop condition:
```

没有 Task ID 不开始改代码。

### Step 1 — 读 source，不读历史猜实现

至少打开与任务直接相关的当前源码。优先级：

```text
source > hardware behavior > current architecture doc > commit history
```

如果文档与 source 冲突，先以 source 为准，并在同一迭代修正文档。

### Step 2 — 只改最小边界

本项目禁止顺手泛化 TP>2、其它模型、其它 dtype。一个迭代最多解决一个主要假设。

### Step 3 — 分层验证

**L0 / source regression**

```bash
pytest -q tests/ut/_310p/test_memfabric_o_proj_source.py
```

**L1 / build-load**

- feature-on build；
- op registration；
- ldd/RUNPATH/ABI 检查。

**L2 / hardware single-layer**

```bash
torchrun --standalone --nproc-per-node=2   benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py   --rows 1 8 32 33 64 128 512 2048 4096
```

**L3 / device evidence**

需要 overlap/性能结论时采 profiler，并用 committed analyzer 复核。

**L4 / model E2E**

eager -> minimal graph -> full graph -> performance/long-run，禁止倒序。

### Step 4 — 只用证据更新状态

提交前必须记录：

```text
Task ID:
Code commit:
MemFabric commit:
Hardware:
Commands:
Correctness:
Latency/perf:
Profiler artifact:
Failure observed:
Conclusion:
Next task:
```

没有实际运行的数据，不把 TODO 改成 DONE。

### Step 5 — 文档只维护“当前事实”

功能变更后最多更新：

- requirements：只有需求真的改变；
- design：实现结构改变；
- 本文：状态/最新证据/下一任务改变；
- usage：命令或平台限制改变。

不要把 debug 日记追加到主文档。历史由 Git 自己保存。

## 6. AI guardrails

任何 AI agent 都必须遵守：

- 不把分支名 `w8a8` 当成当前数据通路事实；
- 不重新引入 BF16/W8A8 代码，除非需求合同被显式修改；
- 不恢复 V4 begin/publish/finish staged API；
- 不用 HCCL fallback 冒充 fused path；
- 不把 `at::add_out` 写回热路径，除非新实机数据证明收益；
- 不把 ack/gate 退回 per-wave TCP barrier；
- 不通过放宽 atol/rtol 解决 correctness；
- 不在 pool 存活期随意加入 device-wide sync；
- 不在没有 profiler/数值证据时宣称 overlap；
- 不用旧 commit 的 E2E 结果给当前 HEAD 签字；
- 不在没有数据时做 N-panel、TP>2、通用 collective 等范围扩张。

## 7. 代码修改边界

常见 Task 对应文件：

```text
model routing:
  vllm_ascend/_310p/ops/memfabric_o_proj.py
  vllm_ascend/_310p/quantization/modelslim_config.py

runtime / protocol:
  csrc/memfabric_o_proj_runtime.cpp
  csrc/memfabric_o_proj/external/memfabric310p_adapter*.{h,cpp}

device kernel:
  csrc/memfabric_o_proj/external/memfabric310p_device.asc

build:
  cmake/memfabric_310p.cmake

source invariants:
  tests/ut/_310p/test_memfabric_o_proj_source.py

hardware evidence:
  benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py
  benchmarks/scripts/analyze_310p_memfabric_overlap.py
```

修改协议/ABI 时必须同步 source regression。

## 8. 交接规则

新的 AI/工程师只需要回答四个问题即可继续：

1. 当前最高优先级 Task ID 是什么？
2. 它的 blocker/前置条件是什么？
3. 哪些文件允许改？
4. 什么证据才允许标记 DONE？

如果这四个问题无法从本文回答，说明状态文档需要先修，而不是继续堆新代码。
