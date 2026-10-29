# E4 Attempt2/Attempt3 主线选择性归档（2026-10-28）

> 来源：[E4 PR #21](https://github.com/10wvw01/vllm-ascend/pull/21) `work/310p-mm-ar-e4-layered-pipeline-20261028`。
> 完整证据（含CANN profiler原始树、全部torchrun日志、构建日志）保留在E4分支 `evidence/e4_layered_20261028/`，本目录只归档主线复核所需的最小关键集。

## attempt2（完成全部允许阶段，执行HEAD `5b727e9f9`，早于守护代码合入）

- `attempt2_e4_screen_summary.json`：七臂Screen机器判定（`SCREEN_ONLY_NO_SIGNED_PERFORMANCE`）——M4096配对p50 Stock 2037.7 / E1 2261.9 / E3-G 2133.9 / e4t2 2131.7 / e4t4 2112.5 / **e4t8 2076.2** / e4p 2105.3µs。
- `attempt2_smoke_03..07_*.json`（10份）：五臂双Rank Smoke原始Rank JSON（e3g/e4t2/e4t4/e4t8/e4p，bit-exact、protocol status=0、首写成功）。
- `attempt2_progress.txt`、`attempt2_binaries_before.sha256`、`attempt2_commit.txt`：执行进度、二进制身份、执行HEAD。

## attempt3（测试保护机制验收，修复HEAD `05d679341`→证据HEAD `1bfacb251`）

- `attempt3_cpu_regression.log`：新HEAD全量CPU（原仓UT 93+9、保护专项13、E4合同9+9、静态门）。
- `attempt3_stage_guard_fault_injection.log`（28/28）、`attempt3_detached_fault_injection.log`（18/18）：无NPU真进程故障注入。
- `attempt3_guarded_nzroute_lease_guard.jsonl` + `attempt3_rank*_route.json`（4份）：受守护双Rank NZ路由全生命周期（无误杀）+ 真NZ路由4/4。
- `preflight_stop_2433615_*`：守护器preflight拦截真实单PID外部入侵的JSONL与原始快照。
- `attempt3_detached_a_lease_guard.jsonl`：campaign preflight拦截真实双PID外部任务（2620627/2621272）。
- `attempt3_detached_b_lease_guard.jsonl` + `attempt3_detached_b_stop_snapshot.txt`：**运行中拦截第三方进程（2640164）并只清理自己**的完整证据（attempt1的506263事故模式真实闭环）。
- `attempt3_g0_identity.json`、`attempt3_frozen_binaries_recheck.txt`：G0身份与四冻结SHA复核。
- `attempt3_build_manifest.json`、`attempt3_isolated_so.sha256`：隔离重编（候选SO `08f73f2c…`与attempt1/2逐字节一致）。
- `e4_card_watch_attempt3.log`、`attempt3_npu_stop_1422.txt`：35分钟外部占用时间线与14:22占用快照。

## 边界

- attempt2 Screen为30样本/臂方向性筛选，不是正式性能；attempt3无新性能数据。
- 五臂Smoke/七臂Screen在attempt3为BLOCKED（外部宿主负载分时占用16384），attempt2结果维持原判。
- 十四场Confirm从未授权亦未运行；任何后续实机前须在新HEAD重新隔离构建（manifest强绑定）。
