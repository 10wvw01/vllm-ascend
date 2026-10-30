# 委托OpenCode：只读查找V1/V2封版所需的既有历史数据

目标是复用历史结果选定V1/V2正式算子。本次**不重跑任何实验**：不启动模型、benchmark、torchrun、设备Kernel、Graph或部署；不替换SO，不改SDK，不清理/切换现有工作区。只读取既有文件、Git对象和文件哈希，可创建独立归档输出目录。

仓库使用SSH连接GitHub，项目唯一主线`10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`。
先读主线`csrc/_310P/releases/v1-v2-20261030/README.md`及两个RELEASE_SELECTION说明。
已在主线历史快照中的five-arm/E2评估、实体E3/e4 attempt2数据无需重复寻找或测量。

## 一、V1：最新完整ABI7的参数最优与身份对应

当前六个核心文件与public-api `fa1c42179818f5516b390d8bce3d39a9d280808f`完全一致；最新单波M-split改动在2026-10-16（30306a0a），不能仅靠2026-09-29旧报告选q。

请找已有材料，回答：

1. q=1/2/4（batch_m=256*q）的原始性能/正确性记录，覆盖已有的小M、M1024/2048/4096及tail点即可；不要为凑矩阵运行新测试。
2. `SMALL_M_PATH`、`SINGLE_WAVE_MAX_M`（1024/2048）、Graph/Eager、trace开关的实际值。区分单波、N-split和批流水，不能把它们当作不同二进制乱比较。
3. 每份结果对应的完整源码HEAD/tree、是否有未提交修改、实际插件/V1 Device/Host SO、MemFabric版本及KVER完整SHA256、物理卡/Die与计时边界。已知SEL-01 V1 Device前缀8447a55a；需要具体原件与构建映射，不以当前安装版本推断旧版本。
4. SEL-01之后是否已有同类decode自旋/507014的修复与完整成功记录。若没有，明确写“未发现后续修复/有效完成证据”，保留V1 Eager 0/3失败事实；不要用短Smoke或旧Graph成功覆盖它。

优先查这些已知位置（不存在则记不存在，不展开无界全盘扫描）：

- `/tmp/opencode/phase4/`：op_q*.csv、op_small_q*.log、phase2_q*.log、Graph/Eager场次。
- `/tmp/opencode/sel01_20261024_run02/`：manifest、原始双Rank/服务日志、路由、环境和失败记录。
- `/tmp/opencode/`中已有small-M/nsplit/single-wave/q-sweep后续目录及日志。
- `/home/w00804146/310P/vllm-ascend/`现有构建/ignored记录；其`csrc/_310P/memfabric_mm_ar_v2/tmp/opencode/20261020_310p3_asset_audit/`资产备份及SHA256SUMS、bundle索引。

## 二、V2：是否存在E4-T8之后的有效复核

已知2026-10-28 E4 attempt2仅一场30样本：M4096 T8=2076.211µs、G=2133.945µs、E1=2261.888µs；其进度写明Confirm未执行。attempt3b没有T8数据。

请找后续已有的T8/G/E1（包含T4也可）重复/平衡对照或Confirm、以及若存在的E1/D4 M2048复核：

- 完整Rank0/Rank1原始JSON/CSV、每样本序号、形状、warmup/iters、正确性/status。
- 同批Stock前后、候选重复、漂移、守护/外部占用和STOP记录；不能只带汇总p50。
- 构建manifest、源码/私有Host/Device SO与SDK/KVER SHA，前后哈希。已知E4 Device SHA为`08f73f2c1936b990a398643243c321582ca39f85cc8b1c80a21efd68cb8d92b2`；不同工件分批登记，不能混拼。
- 若后续复核未执行或已丢失，明确区分“从未执行”“只找到汇总”“原件不存在”“尚未找到”，不自动补跑。

优先查`/tmp/opencode/e4-campaign-20261028/`、既有e4 campaign attempt2/3/3b/confirm目录及`/tmp/opencode/e4-build-attempt*-20261028/`；再看当前310P项目工作区中的未提交实验归档。

## 三、回传格式与归档

先输出简明表：`问题 → 找到/未找到/未执行 → 文件绝对路径 → SHA256 → 源码/SO/KVER → 能证明什么/不能证明什么`。
提供未修改的原件及逐文件SHA256清单；大文件可以制品包加SHA，但必须可取回，不能只给临时路径。
保留失败数据。报告、索引和可入库原件直接归档主线`csrc/_310P/releases/v1-v2-20261030/history_lookup/`，并更新PROJECT_CONTROL/PERFORMANCE_BASELINES；并发主线更新先核对，不强推。
不要选最终winner或改正式算子；研发侧读完原件后完成最终选型与封版。
