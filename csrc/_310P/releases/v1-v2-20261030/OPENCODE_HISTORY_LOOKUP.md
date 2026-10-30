# 委托OpenCode：只读查找已成功的大M V1封版身份

> 用户已延期：下一次实机测试时再查V1历史身份；当前不执行本查找任务，不因本任务启动额外实验。

最新范围为用户确认的M=2048～4096、K=N=2048、FP16、310P3单卡双Die/TP2。
V1从已经成功完成的大M算子测试中选，允许选历史实现，不强求最新全M版本。
本次**不重跑任何实验**：不启动模型、benchmark、torchrun、设备Kernel、Graph或部署；
不替换SO，不改SDK，不清理/切换现有工作区。只读取既有文件、Git对象和哈希，可创建独立归档目录。

仓库使用SSH连接GitHub，项目唯一主线`10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`。
先读本目录README及V1/V2的RELEASE_SELECTION，按ADR-038覆盖首轮审核中的旧前置要求。

## 必需查找：V1已成功的大M q扫描及对应历史快照

2026-09-29的VALIDATION_REPORT R2/R3记录q=1/2/4正确性通过，成功大M测试中q=4更快；
M4096约q1=3.31ms/q2=2.73ms/q4=2.36ms。已按性能选定q4/batch_m1024，现补齐该成功版本的准确身份。

请只找已有材料，回答：

1. 成功完成的M2048/4096 q扫描原件、双Rank正确性日志和原始性能CSV/日志。
   保留预热/N、计时边界、Rank聚合与统计口径；摘要没有的信息写UNKNOWN，不从当前脚本反推。
2. 对应历史源码完整HEAD/tree和当时未提交修改、q、实际批流水/单波路径及执行模式。
   不要求匹配当前最新ABI7；找到实际跑成功的版本即可。不要用当前默认q1替换历史q4。
3. 对应Device/Host/插件/SDK/KVER身份及SHA256、构建manifest或可复现的原始构建映射。
   旧SO若已丢失，明确区分仅有源码/构建日志/报告，不用当前安装SO冒充。

无需补齐小M、N-split、Decode、全模型Eager长稳，也无需查SEL-01之后的故障修复来解锁本轮选择。
失败实验不参与成功记录的性能排名，已有原件保留；不要删除或改写。

优先查已知位置，不做无界全盘扫描：

- `/tmp/opencode/phase4/`：`op_q*.csv`及同轮成功功能/正确性、环境和构建记录；
  报告还记录`phase2_q{1,2,4}.log`、`phase3_functional4.log`，具体目录以实际文件为准。
- `/tmp/opencode/`内已有q-sweep/层级性能/构建归档。
- `/home/w00804146/310P/vllm-ascend/`现有构建与ignored资产，尤其
  `csrc/_310P/memfabric_mm_ar_v2/tmp/opencode/20261020_310p3_asset_audit/`的bundle、SHA256SUMS及资产索引。

## V2已按性能选定，无需查找或重跑

按用户性能优先原则，V2 M2048选E1、M4096选E4-T8。原始14份Rank JSON已抽出并复算完整Wall，
来源/源码/实际SO身份见PERFORMANCE_SELECTION.json。D4只保留历史/构建基线身份。
不要求E4后续Confirm来解锁选择；本轮无需再查V2。

## 回传与归档

输出表：`问题 → 找到/未找到/未执行 → 原件绝对路径 → SHA256 → 源码/SO/KVER → 能证明什么`。
提供未修改原件及逐文件SHA256清单；大文件可制品包加SHA与可取回链接，不只给临时路径。
报告、索引和可入库原件直接归档主线`csrc/_310P/releases/v1-v2-20261030/history_lookup/`，
同步PROJECT_CONTROL/PERFORMANCE_BASELINES；并发主线更新先核对，不强推。
不要改正式算子或宣告生产GO；研发侧按成功的大M结果完成历史版本选择与源码封版。
