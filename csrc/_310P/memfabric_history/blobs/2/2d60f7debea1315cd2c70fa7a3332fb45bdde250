# V3开发交接验证记录（2026-10-29）

## 已完成的本地检查

| 检查 | 结果 | 边界 |
|---|---|---|
| V3 CPU合同 | 29/29 PASS | 512组随机协议调度、40位wire、全部M规划、历史报告负例、外部进程STOP隔离、便携C++合同 |
| 私有SDK收集回归 | PASS（包含在29项中） | Host/设备分离、三件套完整性、同名异构库拒绝 |
| V2既有CPU回归 | 54/54 PASS | 正式V2代码未修改 |
| 两套合计 | 83/83 PASS | 不含任何NPU测试 |
| Host公共头语法 | PASS | 使用MemFabric `b9d46f97`真实SMEM头及ACL声明stub，g++ C++17/Werror |
| Host生命周期stub链接执行 | PASS | 验证IB前拒绝、首次同步、热态无重复同步、非法形状/跨流拒绝及销毁；不是ACL实机 |
| Ruff（仓库固定hook） | PASS | 新Python文件 |
| 新README/REVIEW Markdownlint | PASS | 使用仓库固定0.45.0 registry包 |
| Actionlint | PASS | 新CPU workflow |
| Codespell、Python包结构、diff空白 | PASS | 变更文件/包结构 |

命令：

```bash
python -m pytest -q csrc/_310P/memfabric_mm_ar_v3/test \
  csrc/_310P/memfabric_mm_ar_v2/test/ut --import-mode=importlib
python -m ruff check csrc/_310P/memfabric_mm_ar_v3 \
  vllm_ascend/_310p/ops/memfabric_mm_ar_v3.py
```

`bash format.sh ci`已经尝试，但全量pre-commit初始化Markdown hook时，环境npm拒绝git来源包（EALLOWGIT）。
改用registry同版本Markdownlint单独检查新文档通过；项目旧入口/控制台仍有原有MD056/MD028，
没有为本次V3修改历史表格/格式。不能将此记录写作全仓format CI全部通过。

## 现场仍需执行

**BiSheng/CANN编译：NOT_RUN；V3-0/1/2设备执行：NOT_RUN；算子收益：UNKNOWN。**
本环境没有CANN或310P3，无法证明设备编译、MTE/IB可见性、FP16舍入与资源正确性。

OpenCode按README进行私有编译，然后运行一次有界V3-only快测；前关通过再继续后关。
默认不运行V2/Stock性能臂、不启动模型、不做Confirm，不改变生产默认。
回传`manifest.json`、`result.json`、`REPORT.md`、`historical_comparison.json`、
双Rank JSON/console日志/守护日志；若STOP还必须回传失败门禁及npu-smi现场。

## 上游同步与交接补充

GitCode上游已核查到`ba3e7f42`，GitHub镜像已更新为`b9d46f97`，完整tree相同。
源公共接口未变，Host物理号修复已纳入；KVER28仍保持原值。
新增prepare_sdk.py与OPENCODE_EXECUTION.md，固定两个源码仓库及私有Host/Device构建部署顺序。
SDK准备脚本只做主机编译/文件收集，不执行run安装器或预部署设备；相关CPU正负例通过。
服务器编译、设备部署与V3快速验证仍待OpenCode执行，没有新增设备PASS。
