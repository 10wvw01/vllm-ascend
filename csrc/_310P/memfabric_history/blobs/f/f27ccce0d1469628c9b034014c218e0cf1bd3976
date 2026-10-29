# G1-C：310P3 设备侧第一停止点取证方案与实现记录

> 状态：诊断代码已提交到隔离分支。**2026-10-20 实机执行完毕**——CANN 9.1 编译、CPU 单测、
> Gate C1 双 Rank 链路自检、Gate C2 工况复现（C2R2）与环境恢复均已完成，
> 第一停止点观测结论与全部证据见 `G1C_DEVICE_FIRST_STOP_REPORT_20261020.md`；根因仍未定。
> 禁止将本文件描述为 Gate E 修复报告。原诊断工作树、原部署动态库与 G1-B 证据均不可覆盖。

## 1. 问题定位与证据边界

G1-A F01、G1-B B02 均复现第 29 个请求附近的双 Rank AICore timeout；宿主首次于 `aclnnSwiGlu` 同步点报错，不等于 SwiGlu 为根因。G1-B Host 记录器证明请求环无持续积压、Epoch 持续正常，但不能直接观察 CP SQ 的实时 Head/Tail，也不能知道哪一个 V2 Core 先停。

静态源码风险：`submitSqe()` 的 `sqDepth-used<1` 对合法模环指针计算出的 `used∈[0,depth-1]` 恒为假；且 `ctx.sqHead` 通常仅在 Epoch 入口刷新。因此设备侧软件环状态可能不能正确反映硬件消费进度。**这是待证伪的风险，不是已确认的故障原因，不得在取证时修复容量算法。**

G1-C 核心问题：找到最早未继续推进的 *AI Core 阶段 / AICPU SQ 提交阶段 / 硬件 SQ 消费阶段*，并给出跨 Rank 时间线。

## 2. 两仓库、隔离分支与不可变基线

- V2：`10wvw01/vllm-ascend` 的 `diag/310p3-g1c-device-first-stop-20261020`，从原 G1-B 报告 HEAD `8f1eef3b` 派生；原 G1-A/B 诊断分支保持不变。
- MemFabric：`10wvw01/memfabric_hybrid_310p` 的同名 G1-C 分支，从 `eb96eced` Host 飞行记录器版本派生；`131ee5ff` 仍是原安装 core.so 的源代码基线。
- 实机当前 `libmf_hybm_core.so` 原哈希 `88d39285...`；*仅编译新工作树不会改变现网库*。MemFabric 本机 origin 为 GitCode，获取 GitHub G1-C 分支只用经核实的 github-dev。
- 不执行 reset/clean/pull 覆盖已有工作树；新增独立 worktree 和独立构建目录。两套 G1-C 产物须保存 git SHA 与 SHA256。

## 3. V2 AI Core 阶段戳：每核 64B，默认零开销

实现：`src/op_kernel/mf310p_v2_g1c_trace.h`、`mf310p_v2_kernel.asc`、`src/op_host/mf310p_v2_trial_adapter.cpp`。CMake 选项 `VLLM_ASCEND_310P_V2_G1C_TRACE=ON` 才生成注入版，设备库命名 `libmf310p_v2_trial_device_g1c.so`；默认 OFF 使用原设备库名称。

- 原有 `status[0..63]` 语义不变，后面另加 `8×64B` 进度槽，8 个 AI Core 各写独占 cache line，不修改对称内存、IB 事件或邮件协议。
- 每槽 +0 为 `(chunk << 32) | stage`；+8 为真实 Kernel 每核进入计数（stage=10 才递增）。Host 将每核输出为 `c核=代际:chunk:阶段`；这避免将上一调用的阶段 120 误认为本次执行完毕。
- 每次戳写进行必要的单缓存行清理；每个需要判断的同步点都记录“进入之前”和“返回之后”。**因此该版本增加设备 GM/Cache 操作，可能改变竞争时序，不用它做性能基线。**
- Host 仅在编译选项开启且设置 `MF310P_V2_G1C_TRACE=1` 时起 1Hz 采样线程，继承当前 ACL context，每秒只读 512B，输出 `[V2_G1C]`。采样失败记错误，绝不默认为完成。

### 阶段字典

| stage | 含义 | 不成对时重点 |
|---|---|---|
| 10 | 本核进入 Kernel | 该核尚未启动 / context / 调度 |
| 20/21 | MM 之前 / MM+Clean 完成 | 计算 / MTE |
| 30/31 | IB MM-ready 前 / 后 | MM 完成扇入 |
| 40/41（Core0） | data signal 前 / 后 | 请求环提交 |
| 50/51（Core0） | data wait 前 / 邮件确认后 | 对端 SDMA 到达 |
| 60/61 | Add 释放 IB 前 / 后 | Add 核间释放 |
| 70/71 | Add 前 / 后 | SDMA 可见性、Vector/MTE |
| 80/81 | AddDone 汇合前 / 后 | AddDone 扇入 |
| 90/91（Core0） | quiet 前 / 后 | 本 Rank SDMA 落地 |
| 100/101（Core0） | DONE signal 前 / 后 | 跨 Rank 完成握手提交 |
| 110/111（Core0） | DONE wait 前 / 后 | 对端 DONE |
| 120 | 本核退出 | 必须结合代际与完整 Rank 证据解释 |

无法仅凭阶段戳判定同步根因。Host 线程与 AI Core 在不同一致性域，D2H 读数可能落在两个阶段之间；要求连续采样与上下文信息。

## 4. MemFabric AICPU / CP SQ：64B 私有快照

实现：`src/hybm/ops/hybm_kernel/hybm_g1c_device_trace.h`、`hybm_sdma_orch_kernel.cc` 和 Host `src/hybm/csrc/entity/hybm_sdma_orchestrator.cpp`。

- 在已有 4096B Epoch 工作区 `+0x100..0x13f` 保存单 64B 快照；不改 `HybmSdmaOrchParam`（72B）、`HybmSdmaOrchResult`（24B）、SMEM 段尾双环和 SQE 配方。
- 请求被读取、SQE 三连生成、Doorbell 前后、Epoch 入口/退出都留下阶段；空闲循环隔约数千步、在批次末约每 32 封请求额外执行 HAL `SQ_HEAD/SQ_TAIL` 查询。
- 快照包含 reqSeq、软件 shadowTail / sqDepth、*最近一次设备侧硬件* Head/Tail、HAL 查询码与错误码。没有新查询时保留最后成功/失败查询结果；字段名为 `Last`，**不是 Host 采样时刻的实时硬件位置**。
- `MF_SDMA_ORCH_G1C_DEVICE_TRACE=ON` 才编译设备埋点；必须同时设置 `MF_SDMA_ORCH_KVER=12`，生成独立 `libmf_sdma_orch_v12.json`。采用 v12 名称避免 CP1 对旧 v11 SO 的 dlopen 缓存。
- Host 仅在 `MF_SDMA_ORCH_DIAG=1` + `MF_SDMA_ORCH_G1C_TRACE=1` 同时成立时每秒读一次 +0x100 并输出 `[MF_G1C]`；原 G1-B 的 1Hz 请求环与 epoch 日志继续保留。
- frame 通过 `seqBegin==seqEnd!=0` 进行基本有效性判断；帧号不是跨核/跨设备全序时钟。HAL 返回码非 0 时不把游标当成有效硬件事实。

## 5. 必须按序执行的验收 Gate

**Gate C0：编译与二进制溯源**

1. 两个 GitHub G1-C 远程分支单独 `git fetch` + `git worktree add`，旧的 `54c367ac`、`eb96eced` 工作树不碰。记录三类哈希：源码、原安装库、G1-C 新产物。
2. MemFabric 在全新构建目录启用 `-DMF_SDMA_ORCH_KVER=12 -DMF_SDMA_ORCH_G1C_DEVICE_TRACE=ON`，设备库必须包含取证段；Host 库来自同一 G1-C worktree。确认 v12 JSON 引用的设备名、部署帮助程序实际目标名、临时运行时库搜索路径全部吻合；**不能用原 v11 名字覆盖**。
3. V2 从独立构建目录启用 `-DVLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=ON -DVLLM_ASCEND_310P_V2_G1C_TRACE=ON`；验证目标链接的是 `libmf310p_v2_trial_device_g1c.so`，不是默认 SO。OFF/正式构建另测一次链接与导出符号不退化。
4. 编译阶段如果出现 dav-2002 不支持 trace 写/Host ACL context 设置等错误，先只修诊断兼容问题，记录差异；不能用跳过检查的方式宣称 Gate 通过。

**Gate C1：无长稳故障前的取证链路自检**

1. 完全备份 `/usr/local` 和 site-packages 中真实加载的 core.so；只在无外部 NPU 任务、用户工作树受保护的情况下，用测试版本临时替换并记录 SHA256。原 AICPU v11 文件不得覆盖，先部署新的 v12 到新文件名。
2. 设置 `MF_SDMA_ORCH_JSON` 指向经过核对的 v12 launch JSON，`MF_SDMA_ORCH_DIAG=1`、`MF_SDMA_ORCH_G1C_TRACE=1`、`MF310P_V2_G1C_TRACE=1`，Epoch 默认 15s；G1-B 的 INFO 日志需保留有效的 `g1b_sitecustomize.py` 级别注入及原 PYTHONPATH。
3. 先两卡跑简短 bit-exact 与混合路由：M=2047/2048/2049/4095/4096/4097，分别核验 Stock/V2、四个 chunk 轮转、8 核 stage 代际一致、每秒都有 `[V2_G1C]` / `[MF_G1C]`、`[MF_DIAG]`；不能只确认日志存在，必须确认数据有变化。
4. 如 1Hz 采样本身阻塞或导致额外 507014、MTE3 首写丢失、通信结果不一致，停止 G1-C，保存证据。恢复原库；不得以“延长 timeout”替代修复。

**Gate C2：Gate E 原工况 + 原样 Stock 对照**

执行原 G1-A 的 TP2 Eager、input=4096、output=256、并发10、620秒；一轮诊断 fused 后应先保存双 rank 的 `serve.log`、device slog/plog、`[V2_G1C]`、`[MF_G1C]`、`[MF_DIAG]` 和 SHA256，再决定是否继续。与无埋点原 F01 / B02 不混合计算故障概率；追加无诊断版本的同工况对照以核验观测扰动。出现故障先归档，不能清理设备后声称无证据。

**Gate C3：结论所需的证据组合**

- V2 Core0 卡 `50` 且 MemFabric 尚未生成对应 SQE：调查 AICPU 消费/提交；若 SQE 已发但硬件 SQ Head 不推进：调查 CP SQ / 固件；若硬件消费并到达戳完成：再查 V2 的 wait/缓存可见性。
- V2 Core0 卡 `30`、`60` 或 `80`，其他核对应阶段不一致：优先 IB 核间同步；需要所有 8 核的同代际关系。
- V2 Core0 卡 `90` / `110`：分别检查 quiet 本地完成戳 / 对端 DONE 邮件，不应将两者混淆。
- 8 核同代际均 `120`，但后续 Stock kernel 卡住：目前证据不能把故障归咎 V2；先排除旧 trace/跨调用滞后，再获取真正 stock 前置任务和 runtime 事件。
- SQ Head/Tail “不推进”只在有效的两次及以上设备侧 HAL 样本、存在待处理工作和足够时间跨度时作为异常证据；Host 请求环无积压并不证明硬件 SQ 空。

**归因成功标准：** 对齐双 Rank 最后几次调用代际、核心阶段、AICPU reqSeq 与 HAL SQ 样本、设备日志时间线，定位最早停止层；若快照滞后/不一致则写 UNKNOWN，而不是伪造单点根因。

## 6. 防误判与回退

- 本次是显式编译期诊断：V2 进度写在选中 trace 构建后始终有效；运行时 `MF310P_V2_G1C_TRACE=1` 仅控制 1Hz Host 采样线程。关闭设备侧写入必须改用 OFF 构建。
- 因 Host 采样线程和 AICPU 额外 HAL 查询会改变时间线，应同时保留原 G1-B 非侵入/低侵入结果。
- G1-C 取证代码不能直接生产启用；不更改 MM、signal/wait/quiet、IB、SQE、Mailbox/Cache line、Epoch 或 KFC 算法。
- 恢复时停止所有相关进程后，按备份哈希恢复 Host core.so 的 /usr/local 与 site-packages 两处；原 v11 不动。核对 `SHA256SUMS`、运行时加载路径、KVER JSON 和进程退出，无残留诊断环境变量。
- 实验日志只留服务器 `tmp/opencode/`，不提交公网 GitHub。原 F01/B02 证据与原 worktree 均不可覆盖。

## 7. 两轮静态审视记录（2026-10-20）

**第一轮：协议/布局与观测链审视**

- V2 8 核 trace 槽各占 64B；原 status 0..63、IB workspace、MemFabric 请求环和到达环未更动。Kernel 的原有 MM、signal/wait/quiet、Add 顺序与调用条件保留。
- AICPU trace 占 workspace +0x100 独立 64B，不与 result/overflow/stop/heartbeat 重叠；Host/D2H 仅读，SQE 三连配方、Epoch 重拉算法未更动。
- 发现并修正：Host SQ 快照最初放在 `exitReason==RUNNING` 的 `continue` **之后**，会丢失 Epoch 运行期间每秒样本；已移到判断之前。

**第二轮：反误判、跨版本、边界回归审视**

- V2 stage 增加每核真实入口代际，避免上次 stage120 冒充本轮完成；1Hz 快照包含 8 核全部 64B。
- AICPU 设备 `HEAD/TAIL` 改成保存最近一次 HAL 采样，不拿启动时的 `ctx.sqHead` 冒充实时值；批处理采样使用 **跨越 32 封请求边界** 判断，避免批量步长错过 `head%32==0`。
- 源码静态断言检查：V2 trace 编译默认 OFF、诊断 device SO 独立名；MemFabric trace 默认 OFF、编译需 v12；`#if/#endif` 平衡、双库 Host D2H 观测入口以及已有 `CheckOrTrap` 错误路径保留，均通过源码级检查。
- 新增 CPU-only pytest：V2 source/layout 与离线日志解析测试；MemFabric G1-C 设备/Host/CMake 合同检查。**测试代码已提交，但此环境无法直接执行 repo 的 pytest 或 CANN 设备编译，不能宣称测试 PASS。**

## 8. 阶段状态

源代码取证埋点和 import-free UT 已写入两仓库的独立 G1-C 分支。暂无 CANN 9.1 / dav-2002 实机构建记录，也暂无 Gate C1/C2 的真实设备结果；这两项必须由 310P3 OpenCode 在受控部署环境执行。
