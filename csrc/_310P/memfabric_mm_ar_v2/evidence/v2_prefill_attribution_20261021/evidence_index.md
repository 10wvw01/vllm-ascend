# V2 真实 Prefill 归因证据索引（evidence_index.md，2026-10-21）

> 生成：2026-10-21（V2 Prefill 归因轮，OpenCode @ 310P3，分支
> `perf/310p-v2-prefill-attribution-20261021`，基线 `feat/310p-memfabric-public-api`）。
> 证据分级：`observed`=日志/Trace 直录；`derived`=带推导链的算术结论。
> 原始双 Rank ascend_pt 导出（~70MB/臂）仅存服务器
> `/tmp/opencode/v2_prefill_captures/{stock,v2}_prefill_v2attrib/`；本目录只含脱敏节选。

## 1. 会话清单

| 会话（服务器本地路径） | 配置 | 结果 |
|---|---|---|
| `/tmp/opencode/v2_prefill_captures/stock_prefill_v2attrib` | Stock eager serve + torch_npu profiler，4096-in/1-out ×(6 预热+5 窗口) | 双 Rank Trace 导出成功，5/5 请求成功 |
| `/tmp/opencode/v2_prefill_captures/v2_prefill_v2attrib` | V2 eager serve（V2_SERVE=1, chunk=1024, slots=4）+ 同负载 | 双 Rank Trace 导出成功，5/5 请求成功；首调位精确校验通过 |
| L3 stock（`bench_results/v2attrib_l3_20261021/stock_eager_c1`） | 无 Profiler，4096/2048、50 prompts、并发 10 | 50/50 成功；吞吐 46.45 tok/s、TTFT mean 11341.9ms |
| L3 V2（`bench_results/v2attrib_l3_20261021/fused_eager_c2`） | 同上，V2 开（80/80 层、M=4096→V2、首调校验通过） | 50/50 成功；吞吐 45.94 tok/s、TTFT mean 11578.6ms |

全轮 0 次 507014、0 次数值错误、0 次执行路径不符（停机条件未触发）。

## 2. 目录内容

| 路径 | 内容 | 分级 |
|---|---|---|
| `provenance.json` | 分支/commit、KVER11 JSON 与 Host SO SHA256、插件 .so SHA256、构建旗标、设备映射 | observed |
| `captures/stock_requests.json`、`captures/v2_requests.json` | 预热+窗口逐请求时延（含 V2 首请求 71.29s 建池） | observed |
| `captures/v2_serve_log_excerpt.txt` | 80/80 层选择、dummy 走 Stock、M=4096 路由、双 Rank 首调校验日志（脱敏） | observed |
| `captures/stock_serve_log_excerpt.txt` | Stock 臂引擎初始化与 chunked prefill 配置（零 MemFabric 行，作对照） | observed |
| `sites/stock_r{0,1}_sites.csv` | 200 站点/臂：MM 起始/耗时、AR 耗时、MM→AR 间隙 | observed |
| `sites/v2_r{0,1}_sites.csv` | 200 站点/臂：融合核起始/耗时/Wait | observed |
| `analysis/final_attribution.json` | 步周期、站点统计、非目标 AR、Top Kernel 表 | derived |
| `analysis/kernel_internal_breakdown.json` | 融合核 vs Stock MM 的 mac/mte/scalar/vec 计数器 p50 | observed（计数器）/derived（解读） |
| `analysis/host_api_summary.json` | 双臂 acl API Top-25（时长/次数） | observed |
| `l3_regression/*_rep1_run.json` | L3 回归逐请求明细（50 ttfts + 50×2047 ITLs/臂，去除 generated_texts；路径脱敏） | observed |
| `l3_regression/*_bench.log.summary.txt` | L3 bench 汇总块（吞吐/TTFT/TPOT/ITL 分位） | observed |
| `scripts/` | 采集与分析脚本（含 `v2_prefill_capture.sh` 入库副本），可复现全部 derived 数字 | — |

## 3. 关键 observed 数字（与报告 §3/§4 对应）

- Stock 窗口时延：2.3784/2.3776/2.3760/2.3857/2.3723 s；V2：2.4824/2.4168/2.4099/2.4105/2.4120 s。
- Stock 站点 p50：MM 795us（r0）/759us（r1）；AR 1149us（r0）/1512us（r1）。
- V2 融合核 p50：2573us（r0）/2672us（r1）；p90 2965/3248us；一次性异常 74232/74827us（窗口第 9 波，双 Rank 同时）。
- 融合核内部（r0 p50）：mac 458.9 / mte2 1711.4 / scalar 872.8 / mte3 191.1 / vec 127.4us；
  Stock MatMulV3：mac 456.6 / mte2 335.9 / scalar 186.5 / mte3 91.5 / vec 194.9us。
- V2 分支输入 contiguous 拷贝：199/199 站点，p50 66us；Stock 目标 MM 前 0 次。
- 非目标 HCCL AR：42/步/臂，合计 48.4~52.0ms/步（双臂无系统性差异）。

## 4. 已知剔除项

- `HybmSdmaOrchEpoch`（AICPU 持久编排核，单实例 14.9s 跨窗口）：按持久核剔除，不计入忙时。
- V2 臂窗口内 1 次 `aclrtSynchronizeDeviceWithTimeout` 3.47s：位于 stop_profile 导出期，不在任何请求关键路径。
- 融合核 74ms 异常计入其所在请求（窗口首请求）与"含之"行，稳态统计单列。
