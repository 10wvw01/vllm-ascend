# attempt-002 STOP 定位证据：msprof CLI 设备侧任务采集在 MemFabric 建池后停止

执行时间：2026-10-30 20:00–20:55（Asia）；卡 38144（dev 4,5，shared）。

## 现象（attempt-002 首个 V3-5 候选会话 V3-5-msprof-r0-s1）

- worker 本体全 PASS：数值 bit-exact、协议、20 个 MSTX 正式窗口全部打标成功（MSTX C API 修复后）。
- `collect_external` 的正式窗口路由核验失败 STOP：`measured msprof window lacks actual NZ/fused/HCCL route`。
- 导出的 Chrome timeline 中，正式窗口内只有 host 侧 API 事件（`Node@launch`、`Runtime@BinaryGetFunctionByName`、`aclrtLaunchKernelWithHostArgs`、`Runtime@LaunchKernelV2`），**没有任何 A3 融合核的设备侧任务事件**。
- 时间轴证据：最后一个带时长的设备任务在 1793391764199399（≈20:22:44，建池前 setup 阶段）；首个正式窗口在 1793391802479334（≈20:23:22）。建池恰好发生在两者之间（`A3Session` 建池 ~38s）。
- 原始证据：`rts_task` 表有 491 条任务提交记录（host 侧可见），但其中 433 条 `start_time=-1`、`task_type=UNKNOWN`——设备侧 HWTS 计时从未回填。
- `AI Core Freq` 采样行贯穿全程（100Hz），说明设备本身未被重启，仅**任务级 HWTS 计时采集**在建池后停止。

## 对照组与排除实验（全部在本机 310P3/CANN 9.1.0/torch_npu 2.10.0.post4 复现）

| 实验 | 命令要点 | 结论 |
|---|---|---|
| 1. 纯直调核（无建池） | `mini_kernel.asc`（bisheng dav-2002）+ ctypes `<<<8,0,stream>>>` 直调，msprof `--task-time=on --runtime-api=on --ascendcl=on --aicpu=on` | MiniAddKernel 5 次全部带 AI_CORE 时长出现在导出 timeline（12µs 级）——**直调核本身可被 msprof 采集** |
| 2. fork+exec orch_deploy_helper | 同上，但在两组直调之间 fork+exec 真实助手（无参运行，rc=1 退出） | 6 次直调全部正常采集——**fork+exec 本身与助手子进程存活/退出不杀采集** |
| 3. 真实 A3 建池 + run_pool | `probe_app.py`（A3Session a3-2 + 6 次 run_pool，torchrun TP2 双 Rank），msprof 全默认 | 全程 0 设备任务入 timeline；`hwts.db` 连 `HwtsTaskTime` 表都未建——**建池后设备侧任务采集未启动/被停** |
| 4. 实验3 + `--aicpu=off` | 排除 AICPU 插桩与 KFC 常驻 AICPU 任务互相阻塞 | 仍 0 设备任务——非 aicpu 插桩开关问题 |
| 5. 实验3 + `--aic-mode=task-based` 显式 | 排除采样/任务模式差异 | 仍 0 设备任务 |
| 6. Stock 对照（attempt-002 s0） | 同一 campaign 的 STOCK msprof 会话（无建池） | 设备任务正常（AI_CORE 195、REDUCE_ASYNC_V2/HCCL、MEMCPY 均在窗口内），20 窗口全部 `route_verified=True`，成功生成 timeline JSON/CSV——**msprof+MSTX 链路对 Stock 完整可用** |

复现脚本：本目录 `mini_kernel.asc`（实验1/2 用核）、`mini_direct_launch_app.py`、`mini_fork_helper_app.py`、`probe_app.py`+`probe_run.sh`（实验3-5，需 memfabric-sdk/私有构建路径）；`probe_baseline_run.log`、`probe_aicpu_off_run.log`、`probe_task_based_run.log` 为对应运行日志（POOL_CREATED/SETUP_DONE/RUN_POOL_OK 全 PASS，仅采集缺失）。

## 根因判定

**MemFabric 建池流程（KFC epoch 首轮下发/smem 池 UIO 配置，位于锁定提交 `db38d75e` 的 `libmf_hybm_core`/`libmf_smem` 内）与 msprof CLI 的设备侧任务计时采集互斥**：建池后同一进程的所有任务（aclnn 与直调 alike）不再产生 HWTS 计时。host 侧 API/MSTX 采集不受影响。

本 L1 框架无对应可修 BUG：路由核验 STOP 是合同的正确行为；修复需要 CANN 或 MemFabric 侧变更（MemFabric 依赖锁定 `db38d75e`，按治理需研发侧决策新分支），超出本轮"修复执行/解析 BUG"授权范围。设备内部阶段/SDMA 链路证据按合同标记 UNKNOWN。

## 附：同批 Stock 采集扰动（可观察）

同 case 内 Stock off-before vs msprof 段（wall，ready）：
- M2048：1176.2µs（s0+s3 均值）→ 1221.8µs（msprof）＝ **+3.9%**
- M4096：2098.3µs → 2176.9µs ＝ **+3.7%**

MSTX 正式窗口时长（Stock，r0）：M2048 ready 窗口 2423µs、可见设备任务并集 2195µs、可见空档 228µs；host_device_correlation=UNKNOWN_NO_RESOLVED_FLOW（导出无 s/f flow 端点，按合同保留 UNKNOWN，不做时间最近邻拼接）。
