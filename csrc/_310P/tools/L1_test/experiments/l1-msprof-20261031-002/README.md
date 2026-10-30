# run_id: l1-msprof-20261031-002 —— msprof 诊断 attempt-2（STOP，工具链互斥已定界）

- 身份：与 l1-fixed-20261031-001 同一套工件/SDK/准入（含双Rank Smoke证据），框架 HEAD `c0f75b2d3`（含 MSTX C API 修复）；msprof 工具冻结（msprof-tool.json）。
- 计划：`--cases v3-5-c512-g1024 v3-6-c512-g1024 v3-7-c512-g1024 --collection msprof --mode shared --warmup 2 --samples 10 --rounds 1`，36场 TP2（每配置 ABBA × off-before/msprof/off-after）。
- 实际进度：
  1. v3-5 off-before 4场全 PASS（paired.json 齐）。
  2. **V3-5-msprof-r0-s0（Stock臂）完整 PASS**：MSTX 20 窗口打标成功、Chrome 导出齐、`collect_external` 生成 rank*.profiled.json 与 timeline-rank*.json/csv，20 窗口全部 `route_verified=True`（aclnnMatmulWeightNz + hcom allReduce 在窗口内）——msprof→MSTX→导出→解析→窗口路由链路对 Stock 全通。
  3. **V3-5-msprof-r0-s1（V3-5候选臂）STOP**：worker 数值/协议/窗口全 PASS，但窗口内无任何设备侧任务事件 → `measured msprof window lacks actual NZ/fused/HCCL route`。按合同 STOP，保留双Rank现场与原始捕获（frozen/runs/v3-5-c512-g1024/V3-5-msprof-r0-s1/）。
- 根因定界：见 `diagnostics/README.md` —— **MemFabric 建池（KFC epoch 首轮下发/smem UIO 配置）后，msprof CLI 的设备侧任务计时采集对本进程永久停止**（aclnn 与直调核 alike；`rts_task` 491 条中 433 条无 HWTS 计时；排除 fork+exec 助手、aicpu 开关、aic-mode；纯直调核与 Stock 对照均正常）。属 CANN×MemFabric(锁定 db38d75e) 互斥，非本框架可修BUG；按治理需研发侧决策，本轮不擅改依赖。
- 交付：36场中 5 场完成 + 1 场 STOP 现场 + 9 场 NOT_RUN（v3-5 剩余 msprof/off-after 与 v3-6/v3-7 全部）。V2/V3 设备内部阶段、SDMA 真实链路占用、核内 MM/AR 交叠：**UNKNOWN**（无可靠链路证据，按合同不拼占比）。
- 可用附加证据（Stock 同批）：msprof 段相对 off-before 的 wall 扰动 M2048 +3.9%、M4096 +3.7%；MSTX 窗口（Stock r0，M2048 ready）窗口 2423µs、可见设备任务并集 2195µs、可见空档 228µs；host_device_correlation=UNKNOWN_NO_RESOLVED_FLOW。

## 目录

- `plan.json` / `runtime.actual.json` / `msprof-tool.json` / `frozen/`：完整冻结身份与已执行会话原件（大体积原始 PROF 捕获保留在盘、未入 git）。
- `diagnostics/`：STOP 根因复现脚本与日志（最小直调核、fork+exec 助手对照、真实 A3 建池探针、aicpu/aic-mode 变体）。
