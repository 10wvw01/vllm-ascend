# run_id: l1-msprof-20261031-001 —— msprof 诊断 attempt-1（STOP，已定位为框架打标BUG并修复）

- 计划：3×A3（v3-5/6/7-c512-g1024）、msprof collection、shared W2/N10/R1、36场（未执行到）。
- 结果：**STOP**（campaign-result.json）。首个 msprof 会话双Rank `MSTX range_start failed`。
- 根因：torch_npu 2.10.0.post4 的 python mstx 包装（`torch_npu._C._mstx._range_start_on_host`）在外部 msprof CLI 采集下恒返回 0（隔离探针证实），L1 窗口从未进入 trace。
- 修复：主线 `df8f399a4`/`c0f75b2d3` —— 打标改走 CANN MSTX C API（`libms_tools_ext.so` 的 `mstxRangeStartA/mstxRangeEnd`，官方样例同路径；隔离探针证实 C API 打点进入导出 timeline）。
- 本目录保留 STOP 现场（frozen/runs/v3-5-c512-g1024/V3-5-msprof-r0-s0 的 stop_rank*.json 与原始捕获）；复跑使用新 attempt 目录 `l1-msprof-20261031-002`。
