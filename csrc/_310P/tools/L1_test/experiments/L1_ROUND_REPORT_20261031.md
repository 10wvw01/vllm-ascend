# L1 定点验证轮报告 —— 2026-10-30/31（l1-fixed-20261031-001 + l1-msprof-20261031-001/002）

目的：验证新公共 Tensor 入口（`torch.ops.memfabric310p.mm_ar`）、统一计时（L1_TENSOR_IO_V4）与 msprof 采集链路；取得 Stock/V2/V3 初步性能与瓶颈证据。
结论级别：**OBSERVATIONAL_SHARED / R1 —— 全部耗时仅作观察，不签稳定性能收益**。

## 1. 身份

| 项 | 值 |
|---|---|
| vllm-ascend | `dev/310p-memfabric-mm-ar`：起点 `ceae9e4ca1e7dce4e7ff7c565e93d95ba758af07` → 本轮结束 `c0f75b2d369c0f897d204ae09430e5b642f2500b`（4 个执行BUG/测试修复提交：b7adbbeb9 PYTHONPATH、e44a71a13 模板STOCK、df8f399a4 MSTX C API、c0f75b2d3 单测） |
| memfabric_hybrid | `fix/310p-v3-orch-deploy-helper-20261029` @ `db38d75e8485d31451302219eb7a1bfeb1dd107c`（tree `4fe624cabbf62d6475f79cc63bd19dcbedacb0e7`，GitHub镜像 10wvw01/memfabric_hybrid；GitCode 远程无此分支） |
| SDK/KVER | 复用 `310P/a3ut-20261029/memfabric-sdk`（KVER28、libmf_sdma_orch_v28.json、orch_deploy_helper sha256 `849f93c0…`；version.info CommitId=db38d75e 一致） |
| 私有SO | tensor_bridge `094afdda…`；V3 host/device 主线重建（manifest 见 l1-fixed…/admission/v3-build-manifest.json）；V2 selected 扩展（vllm_ascend_C.so + libmf310p_v2_selected_device.so，V2首次配 V28 SDK/v28 JSON） |
| msprof 工具 | `cann-9.1.0/tools/profiler/bin/msprof`（探测冻结：help 全默认可广告、msproftx=on、--version rc=255 保留原件） |
| 硬件/环境 | 310P3 卡 38144 物理 dev4,5，shared；CANN 9.1.0；torch 2.10.0+cpu / torch_npu 2.10.0.post4；双Die Health OK |

## 2. 链路验证结果

| 链路 | 结果 |
|---|---|
| 新公共Tensor入口（V2 selected + V3 python_session，输出获取/Storage包装/租约/别名保留释放/输入保活） | **PASS**（6 binding 双Rank Smoke + 22场快测全程门禁） |
| 统一计时（ready逐次/steady块均、双Rank max、尾组加权、独立下游ADD不计时） | **PASS** |
| CPU合同 | L1_test 106 passed；V3/V2 335+36 子测试 |
| 实际路由取证 | Stock: aclnnMatmulWeightNz+hcom_allReduce；V2-7 V2TrialKernel；V2-15 E4Group2048Kernel；V3-5 A3Kernel；V3-6/7 A3SeatKernel（进程maps核验：仅冻结库映射） |
| msprof CLI 链路（MSTX打标→Chrome导出→timeline解析→窗口路由） | **Stock 全通**（20窗口 route_verified、timeline JSON/CSV 产出）；**V2/V3 候选 STOP**（见§5） |

## 3. 快测 campaign（6组22场全PASS，shared W2/N10/R1，B2/WS4）

Wall/Event（µs；ready=单次、steady=块均每调用；分位数为该范围口径）：

| 配置 | M | 场景 | Wall均值 | W-p50 | W-p90 | W-p99 | Event均值 | E-p50 | E-p90 | E-p99 | 同批Stock速比(wall) | 臂漂移 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 2048 | ready | 1160.3 | 1157.8 | 1176.4 | 1184.2 | 1075.4 | 1072.4 | 1086.5 | 1107.7 | — | 0.3% |
| stock | 2048 | steady | 1068.8 | 1063.2 | 1081.6 | 1105.6 | 1032.3 | 1025.5 | 1044.6 | 1073.2 | — | 0.3% |
| stock | 4096 | ready | 2165.6 | 2110.8 | 2206.7 | 2729.2 | 2054.3 | 2023.4 | 2039.4 | 2566.7 | — | 5.8% |
| stock | 4096 | steady | 2004.9 | 2009.8 | 2016.5 | 2018.5 | 1965.4 | 1966.0 | 1975.7 | 1981.3 | — | 0.9% |
| v2-e1-m2048 | 2048 | ready | 1445.8 | 1427.1 | 1528.0 | 1576.9 | 1360.8 | 1340.7 | 1441.6 | 1484.1 | 0.824 | 8.0% |
| v2-e1-m2048 | 2048 | steady | 1300.3 | 1309.6 | 1372.5 | 1381.9 | 1259.3 | 1267.4 | 1327.9 | 1336.2 | 0.846 | 6.3% |
| v2-t8-m4096 | 4096 | ready | 2319.6 | 2309.6 | 2396.7 | 2434.2 | 2223.9 | 2223.1 | 2301.3 | 2347.1 | 0.909 | 0.5% |
| v2-t8-m4096 | 4096 | steady | 2210.6 | 2206.9 | 2279.8 | 2295.5 | 2164.4 | 2163.1 | 2221.8 | 2250.0 | 0.909 | 0.7% |
| v3-5-c512-g1024 | 2048 | ready | 1627.8 | 1631.0 | 1739.2 | 1747.7 | 1542.5 | 1557.9 | 1643.4 | 1675.1 | 0.708 | 2.7% |
| v3-5-c512-g1024 | 2048 | steady | 1513.9 | 1522.4 | 1538.8 | 1547.1 | 1476.5 | 1486.1 | 1500.0 | 1512.5 | 0.708 | 0.6% |
| v3-5-c512-g1024 | 4096 | ready | 2530.4 | 2518.8 | 2631.2 | 2729.4 | 2446.2 | 2449.0 | 2540.5 | 2654.9 | 0.868 | 8.1% |
| v3-5-c512-g1024 | 4096 | steady | 2437.0 | 2414.4 | 2508.5 | 2741.6 | 2394.9 | 2371.8 | 2467.8 | 2699.1 | 0.827 | 4.3% |
| v3-6-c512-g1024 | 2048 | ready | 1677.0 | 1661.8 | 1755.1 | 1842.6 | 1589.1 | 1575.5 | 1677.5 | 1760.6 | 0.770 | 23.6% |
| v3-6-c512-g1024 | 2048 | steady | 1538.6 | 1535.0 | 1563.6 | 1575.5 | 1496.4 | 1495.3 | 1516.8 | 1537.2 | 0.726 | 2.3% |
| v3-6-c512-g1024 | 4096 | ready | 2563.9 | 2575.9 | 2648.7 | 2674.9 | 2477.6 | 2487.9 | 2560.3 | 2574.3 | 0.838 | 2.2% |
| v3-6-c512-g1024 | 4096 | steady | 2427.2 | 2437.7 | 2500.7 | 2512.0 | 2387.5 | 2399.0 | 2454.2 | 2471.5 | 0.842 | 0.8% |
| v3-7-c512-g1024 | 2048 | ready | 2056.9 | 2054.5 | 2173.5 | 2305.2 | 1963.9 | 1946.2 | 2081.4 | 2211.1 | 0.575 | 3.0% |
| v3-7-c512-g1024 | 2048 | steady | 1862.5 | 1846.8 | 1913.3 | 1950.8 | 1823.5 | 1807.1 | 1876.7 | 1914.7 | 0.567 | 2.3% |
| v3-7-c512-g1024 | 4096 | ready | 3100.3 | 3054.8 | 3204.4 | 3275.1 | 3011.5 | 2960.4 | 3133.6 | 3198.3 | 0.677 | 2.2% |
| v3-7-c512-g1024 | 4096 | steady | 2898.5 | 2892.1 | 2988.9 | 3013.5 | 2854.4 | 2850.0 | 2944.2 | 2971.6 | 0.690 | 3.2% |

计时封套占比：ready 11.4–12.7%、steady 2.9–5.9%（>5% 的行不构成收益证据）。预热多数 STABLE_TIMING_WINDOW；漂移最高 v3-6 M2048 首场 23.6%（外部占用，shared 接受，未剔慢样本）。

方向性观察（仅观察）：
- 全部候选慢于同批 Stock；V2-15/T8 最接近（≈0.91），V3-5/V3-6 M4096 ≈0.83–0.84，V3-7/tail 明显最慢（0.57–0.69）。
- 与 2026-10-29 A2/A3 历史方向一致：完整公开接口成本（输出获取/包装/租约/双Host协商/同步）为主因结构，本轮未做归因轮。

## 4. V3 迭代观察（三A3同批）

| A3 变体 | M2048 wall ready/steady | M4096 wall ready/steady | 相对 V3-5 |
|---|---|---|---|
| V3-5 首READY | 1627.8 / 1513.9 | 2530.4 / 2437.0 | 基线 |
| V3-6 七核+专职通信 | 1677.0 / 1538.6 | 2563.9 / 2427.2 | 持平±3%（M2048首场受外部干扰23.6%漂移） |
| V3-7 Core0末尾席位 | 2056.9 / 1862.5 | 3100.3 / 2898.5 | **劣化 26–29%**（c512/g1024点） |

## 5. msprof 诊断（36场计划：5场完成 + 1场STOP + 30场NOT_RUN）

- attempt-1（l1-msprof-20261031-001）：torch_npu python mstx 包装在外部 msprof 下恒0 → 框架BUG，已修复（df8f399a4 走 CANN MSTX C API）。
- attempt-2（l1-msprof-20261031-002）：off-before 4场 ✓、Stock msprof 场 ✓（20窗口 route_verified、timeline 产出）；**V3-5 候选场 STOP**——窗口内无设备侧任务事件，按合同路由缺失即STOP。
- 根因（隔离复现，见 002/diagnostics/README.md）：**MemFabric 建池（KFC epoch 首轮下发/smem UIO 配置）后，msprof CLI 设备侧任务计时采集对本进程永久停止**（433/491 任务无 HWTS 计时；已排除 fork+exec 助手、aicpu 开关、aic 模式；纯直调核与 Stock 对照正常）。属 CANN×MemFabric(db38d75e锁定) 互斥，非本框架可修BUG；按治理交研发侧决策，本轮不改依赖、不静默跳组、不回退。
- 时间线分析（可用部分，Stock r0）：msprof 段相对 off-before wall 扰动 M2048 +3.9% / M4096 +3.7%；MSTX 窗口（M2048 ready）2423µs，可见设备任务并集 2195µs，可见空档 228µs（非硬件空闲），host_device_correlation=UNKNOWN_NO_RESOLVED_FLOW。
- **UNKNOWN（明确缺证据）**：V2/V3 各设备内部阶段（MM/READY/wait/SDMA/排空）分段、核内 MM/AR 真实交叠、SDMA 真实链路占用/带宽——msprof 设备侧窗口证据缺失，且无核内可靠时钟，按合同不拼占比。

## 6. 失败/未运行清单

| 项 | 状态 |
|---|---|
| msprof attempt-1 全部候选会话 | STOP（框架打标BUG，已修复；保留现场） |
| msprof attempt-2 v3-5 msprof 候选场(V3-5-msprof-r0-s1) | STOP（工具链互斥，保留双Rank现场+原始捕获） |
| msprof attempt-2 其余30场（v3-5 off-after×3、v3-5 msprof×2、v3-6/v3-7 全部） | NOT_RUN（首错停止本批次） |
| exclusive轮 / R≥3 bootstrap / V1 / 模型TTFT | NOT_RUN（本轮授权范围外） |

## 7. 原始数据位置

- 快测：`l1-fixed-20261031-001/quick-off/frozen/`（campaign/plan/runtime、summary.json/csv、campaign-result.json、逐场 rank0/rank1/paired/lease_guard）。
- Smoke/准入：`l1-fixed-20261031-001/smoke/`、`.../admission/`。
- 大体积原始捕获（in-process route tensorboard、msprof PROF 树，共 78 目录/约144MB）保留于本服务器工作树原位，`raw-capture-index.json` 逐目录记录 path/files/bytes/sha256。
