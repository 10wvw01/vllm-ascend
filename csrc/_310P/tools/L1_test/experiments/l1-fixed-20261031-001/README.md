# run_id: l1-fixed-20261031-001 —— L1定点验证（新公共Tensor入口 + 统一计时 + 快测 campaign）

执行日期：2026-10-30/31（服务器本地 10-30 晚）；卡 **38144**（物理 dev 4,5），shared 非独占、W2/N10/R1、batch_size=2、working_set=4、ready+steady、collection=off。

## 身份

| 项 | 值 |
|---|---|
| vllm-ascend 主线 | `dev/310p-memfabric-mm-ar`，起点 `ceae9e4ca`，本 run 结束 HEAD `c0f75b2d3`（3 个执行BUG修复 + 1 个单测修正，见下） |
| MemFabric | `fix/310p-v3-orch-deploy-helper-20261029` @ `db38d75e`（tree `4fe624cabb…`，GitHub 镜像 10wvw01/memfabric_hybrid） |
| SDK | 复用 `/home/w00804146/310P/a3ut-20261029/memfabric-sdk`（KVER28，`libmf_sdma_orch_v28.json`，orch_deploy_helper SHA `849f93c0…`，CommitId=db38d75e 未变） |
| CANN / Python / Torch | `/usr/local/Ascend/cann-9.1.0`，python3.12.13，torch 2.10.0+cpu / torch_npu 2.10.0.post4 |
| Tensor 桥 | `build/310p_l1/l1-fixed-20261031-001/tensor_bridge/mf310p_tensor_bridge.so`（SHA `094afdda…`，源=主线 pool_tensor.cpp，编译PASS） |
| V3 Host/Device | 同 build 目录 `v3-build/`（**从主线 ceae9e4ca 重建**：与历史 r8 二进制相比 8 个源文件已变更，不复用旧 SO；manifest 在 admission/） |
| V2 扩展（selected） | 同 build 目录 `v2-selected/`（vllm_ascend_C.so + libmf310p_v2_selected_device.so + trial/device；V2-7→V2TrialKernel，V2-15→E4Group2048Kernel；V2 首次使用 V28 SDK + v28 launch JSON，双Rank短测通过） |
| msprof 工具 | `/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof`，help 全默认项可广告、`--msproftx=on`，`--version` rc=255（保留原件） |

主线修复提交（执行BUG，均已带 CPU 回归）：
- `b7adbbeb9` clean_environment 保留 CANN PYTHONPATH（否则首个设备算子 AclSetCompileopt 500001/tbe 缺失）
- `e44a71a13` case-selected campaign 模板必含 STOCK runtime（否则 freeze 阻塞）
- `df8f399a4`+`c0f75b2d3` msprof 窗口打标改走 CANN MSTX C API（torch_npu python mstx 包装在外部 msprof CLI 下恒返回 0）

## 门禁

- CPU合同：L1_test 106 passed；V3/V2 335 passed + 36 subtests。
- G0：卡 38144 双Die Health OK（admission/g0*.txt），DDR ~3% 起始占用。
- 双Rank短Smoke（新公共Tensor入口，全部 PASS，真实路由取证在案）：STOCK aclnnMatmulWeightNz+hcom_allReduce；V2-7 V2TrialKernel；V2-15 E4Group2048Kernel；V3-5 A3Kernel；V3-6/V3-7 A3SeatKernel。两个Y+Storage别名保留/释放、输入保活、立即同Stream下游ADD、协议审计、首写防护均在 worker 内验证（smoke/ 下 rank0/rank1.json）。

## 快测 campaign（6组22场 TP2，全部 PASS_L1_CAMPAIGN_ONLY）

OBSERVATIONAL_SHARED（shared/R1 只观察，不签稳定性能收益）。每配置独立 Stock ABBA/BAAB 同批对照。完整均值见 `quick-off/frozen/summary.json|csv`。Wall 均值（µs，双Rank max单位/总调用数；ready=单次、steady=块均）：

| 配置 | M2048 ready | M2048 steady | M4096 ready | M4096 steady | 同批Stock速比区间 |
|---|---|---|---|---|---|
| stock | 1160.3 / 1075.4 | 1068.8 / 1032.3 | 2165.6 / 2054.3 | 2004.9 / 1965.4 | 1.000 |
| v2-e1-m2048 | 1445.8 | 1300.3 | — | — | 0.824–0.846 |
| v2-t8-m4096 | — | — | 2319.6 | 2210.6 | 0.909–0.911 |
| v3-5-c512-g1024 | 1627.8 | 1513.9 | 2530.4 | 2437.0 | 0.708–0.868 |
| v3-6-c512-g1024 | 1677.0 | 1538.6 | 2563.9 | 2427.2 | 0.726–0.842 |
| v3-7-c512-g1024 | 2056.9 | 1862.5 | 3100.3 | 2898.5 | 0.567–0.690 |

（`a/b` = 两个 Stock 场次；候选行 wall/event 均在 summary.csv。）

要点：
- 全部候选**慢于同批 Stock**：V2-15/T8 最接近（~0.91），V3-5/M4096 与 V3-6/M4096 约 0.83–0.84，V3-7/tail 最慢（0.56–0.69）。方向与 2026-10-29 A2/A3 历史一致（接口成本为主因结构未变）。
- 漂移：多数 ≤6%；stock M4096 ready 5.8%、v2-e1 7.9–8.6%、v3-6 M2048 首场 23.6%（外部占用，shared 接受，未剔样本）。计时封套占比 ready ~11–13%、steady ~3–6%（>5% 的不签明确收益，均只作观察）。
- 数值/协议/下游/首写全 PASS；每场报告保留原始 rank 样本（`quick-off/frozen/runs/`，原始 in-process route 捕获目录保留在盘未入库）。

## 目录

- `admission/`：G0、CPU日志、三份构建manifest、双Rank smoke汇总与逐binding证据。
- `smoke/`：6 个 binding 的双Rank Smoke 原始 rank 报告与 lease_guard。
- `quick-off/`：plan/runtime.actual/frozen（campaign）、summary.json/csv、campaign-result.json。
- `smoke_driver.py`：本轮引导驱动（手工装配 run.json + stage_guard，用于 smoke 引导，未改框架语义）。

## 未含 / UNKNOWN

- V1 未测（按约定）；无 exclusive 轮；shared/R1 结果不签稳定性能收益。
- 模型TTFT/Serving/Graph：NOT_RUN（不变）。
- msprof 诊断与设备内部阶段/SDMA 链路证据：见 `../l1-msprof-20261031-001`、`../l1-msprof-20261031-002`（attempt-2 因工具链互斥 STOP，设备侧证据 UNKNOWN）。
