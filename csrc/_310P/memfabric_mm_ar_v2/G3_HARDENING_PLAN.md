# G3 工程加固与固定开销定位执行方案（2026-10-18）

> 基线：V2 T0/T1 已有的实机报告；严格区分「API 可用 / 算法正确」与「生产可靠」。**本文件是待实机执行的计划，不是 PASS 报告。** 固定使用 TP=2、310P3 dav-2002、CANN 9.1 的已验证环境。不更改 V1，不直接进入正式 vLLM 路由。

## 0. 本轮先完成的软件修复

- `t1_nz_direct_trial.py --mode stress` 已改为 30 个**互不重复**的 M（含 1/255/256），不再每次重建同 seed RNG；
- 1000 次连续调用不再由同一个模运算耦合 MM/Add/权重，测试结束检查每个 M 与全部 mm×add×4 seed 组合实际出现；
- 原 `T1_NZ_DIRECT_REPORT.md` 的 160/160 PASS 只覆盖旧脚本的 4 个 M，报告第 9 节已经勘误。
- **此修复的 310P3 实机 PASS 仍待补测。** 先执行 `--mode stress` 并把输出日志提交仓库，确认 coverage_plan unique_m=30、seq_1000 unique_m=30、mm_add_seed=32（默认 mm=4 档、add=2 档）。

## 1. P0：MTE3 首次输出丢失——先排因，不能靠时间延迟当修复

**证据：** 单卡无池的 MM-only T1 也能复现；launch-stream sync 对 T1 起作用，但 T0 无效；T0 使用全设备同步有 0–15s 随机开销；纯延迟 50ms 也会改变结果。因而「MemFabric 建池页面映射竞争」**尚未被证明**，不能简单归咎于 Cache。官方 `DataCacheCleanAndInvalid` 主要处理 Scalar ↔ DMA 的 GM Data Cache 可见性，不应在没有证据时把它当成修复 MTE3 丢写的原语。

### A. 生成最小复现与归因矩阵

使用现有 `t1_diag_single.py` 和设备库中的 `V2DiagNzDirectMmKernel`，另增加含 **T0 MM+Add 的最小独立单卡用例**，不建 MemFabric 池，直接检查 `torch.equal(y,ref)`，并保存异常设备日志。

每组测试至少 100 次新进程/新 buffer 首写（不是同一地址重复 100 次），记录 Kernel symbol、首次真实 launch 与否、GM 地址、输出分配/预写时机、关联 stream、KFC 是否存在：

| 对照轴 | 必须比较 |
|---|---|
| 内核范围 | 单纯 GM→GM 拷贝 / 单卡 MM 直接写 C / 单卡 MM+TQue Add 写 y / T0 整体 Kernel |
| 分配时机 | y 在 device/stream 初始化后、MemFabric 建池前或后分配 |
| 初始化 | `torch.empty`、`aclrtMemsetAsync(y,...)`、设备侧实际写一次完整 y |
| 前置条件 | 不同步、调用 stream sync、**确有生产者时** Event record + stream wait、device sync（仅诊断）、50ms sleep（仅诊断） |
| 符号生命周期 | enable=0 空预热、**真实 enable=1 完整双 Rank/独立单卡热身**、更换 kernel 模板 |
| 内存重用 | 全新地址、原地址重复写、释放后缓存分配器复用相同地址 |

必要结论：确定「真实 symbol 首次使用」还是「y 新地址」还是「额外 stream 的尚未完成生产者」，并区分时间敏感型运行时缺陷与真实跨流依赖。如果出现没有 MemFabric 的单卡丢写，应携最小代码、编译日志、CANN/驱动/KVER/设备日志上报平台支持，而不是继续修改通信协议。

### B. 临时安全路线（必须实机证实才允许启用）

1. **初始化阶段预分配和预写固定输出地址**：先准备有界、可复用的 ND 输出 workspace，对所有 T0/T1 模板执行一遍**真正 enable=1 的有效计算**（必须双 Rank 配对、含完整 `signal/wait/quiet/DONE`，不能仅用 enable=0）。验证首调用与动态变 M 之后是否仍不丢写。
2. 若 T1 的输出首次使用必须在其他 Stream 已完成相关任务之后，按官方 AscendCL `aclrtRecordEvent + aclrtStreamWaitEvent` 做**有生产者依据的**设备侧依赖，而不是无条件 Host stream sync。没有真实事件依赖时，Event 不是修复手段。
3. 不在热路径保留 `aclrtSynchronizeDevice`；消除全设备同步后还要核验双 Rank 偏斜 (0/1/10/100 ms) 不再引起通信等待死锁。删除同步之前，不能宣布修复。若无法可靠消除，在生产路由中**禁用 V2 T0**并保留 Stock/V1 降级；不允许靠 `sleep` 或把 wait 超时加大掩盖异常。
4. `warmed_y` 仅按地址缓存不识别同地址不同生命周期，须明确 buffer 所有权、预热代际；同池多 Stream 并发尚不支持。Graph Capture 先不做，等首次写入可由**初始化前置步骤**安全保证以后再测。

**通过条件：** 所有候选模板 × 动态新旧输出 × 不同启动顺序，每组不少于 1000 次 bit-exact、无 MTE3 空写、无设备同步、无对端超时；记录首调耗时/P95。达到前保留 T0/T1 trial-only 开关。

## 2. P0：KFC 控制面环境修复

T0/T1 使用 dev0/dev1 上的 `MF_KFC_BYPASS`（通过 `/tmp/mf_kfc_*.json` 临时文件规避启动逻辑）不能认定部署正常。

- 新进程重启 MemFabric AICPU 服务时，先以**不启用 bypass** 的官方部署流程验证；收集 system() 失败的原始错误、进程权限、设备健康、驱动/CANN/KVER 兼容性、AICPU/KFC 日志与 pid/清理日志；
- 把 bypass 仅当作单独标记的实验，不许在生产默认启用。必要时在**健康的另一组 310P3 设备**交叉复验，区分设备环境故障与 MemFabric 代码问题；
- 正常 KFC 初始化和销毁全流程至少重复 100 次无泄漏、无初始化失败；异常后不得复用旧 pool/控制信箱。

## 3. P1：公正的性能与通信共存评估

- 单独保存两组不同定义的 Stock：`FP16 MM + FP16 HCCL AllReduce`（真实 production 性能基线）及 `FP16 MM → FP32 HCCL sum → FP16`（bit-exact 正确性参考）。旧 `t1_nz_direct_trial.py` 的 `--mode stock` 测 FP16 AR，而 `full` 内嵌 stock_ms 测 FP32 AR，**不能混成同一个基线**；性能对比必须同环境、同 stream/sync 边界、双 Rank 同统计口径。
- 性能通过条件不应只看 `M=256` 的单点；测 M={1,16,32,64,128,255,256,257,512,1024,2048,4096,8192}，重复 100+ 次中位/P95，分进程和运行期采样，规避会话间 ±0.1 ms 波动。
- 设备侧 profiler 需要测量 `MM`、`IB ready/release/done`、`SDMA signal→mail wait`、`Add`、`quiet+DONE`、Host warmup/sync。先分析固定成本，再决定是否合并或减少同步；**quiet 和波级 DONE 不得为追求速度直接删除**。
- 若要采用「小 M Stock、大 M V2」混合路由，先验证同一 TP=2 推理进程内 HCCL ↔ MemFabric 按层交替 1000 次、同 Rank 顺序一致、跨 stream 依赖和 Graph，不能只凭先算完 HCCL 参考再建池的独立微基准认定共存成功。
- V1 与 V2 当前共用单池限制，**不能不改运行时直接按 M 混用 V1/V2**。

## 4. 执行顺序 / 交付

G3-0：跑修正版 T1 stress，提交完整 stdout/raw 数据和覆盖率；G3-1：MTE3 最小复现与修复候选 A/B；G3-2：KFC 无绕过建池/析构；G3-3：同进程 HCCL/MemFabric 混用和设备侧 profiler；G3-4：完成以上之后再进入 Eager/Graph 与 Qwen3.6 TP=2。

每个关口都产出「环境、复现命令、原始失败率、修复前后、正确性矩阵、耗时分布、剩余风险」。先实证后更改生产路由，且不得修改 V1。
