# BATCH-R0机器可读证据格式校验记录（2026-10-27）

> run_id `a1-r0-20261027-001`，审核仅检查GitHub已提交文件，不在310P服务器运行新实验、不更改NPU结果。

## 发现与纠正

- 2026-10-27 R0初次取证归档的`source_provenance.json`中，`source_static_verification_server.hybm_gva.cpp.line`录入为`78-80`（非JSON合法数值），阻碍自动解析。
- 已执行**唯一文字级修正**：`"line": 78-80` → `"line": "78-80"`；该字段本来就是**源码行号范围**，绝非计算式或实机测量值。原始错误版本永久保留在Git提交历史；修正提交`e0f501236ea78214d0cb67e5019ad640e54a77a3`，未改变任何设备环境、二进制SHA、原始CPU日志或实验结论。
- 修正后分别按标准JSON解析器成功校验六个文件：`source_provenance.json`、`build_manifest.json`、`card_selection.json`、`cpu_results.json`、`environment_inventory_sanitized.json`、`evidence_index.json`。六份均有`schema`标识。
- `source_provenance.json`的`captured_at_local`保留原现场提供的`2026-10-26T18:0x Asia...`文本；**这不是一个准确时间戳，不得用于时间计算或宣称精确采样时刻**。需要准确取证时间时参考`environment_inventory_sanitized.json:capture_at_utc`（实际键`captured_at_utc`，报告2026-10-26T18:05:35.651953Z）及服务器原始日志，若与当前环境不符以当次新快照为准。

**判定**：格式修正完成，原取证事实有效；此次修复不代表新CPU、G0、Kernel或TTFT测试。
