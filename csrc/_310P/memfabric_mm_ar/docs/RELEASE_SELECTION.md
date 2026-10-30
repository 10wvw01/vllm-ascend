# V1 版本规整与封版选择

2026-10-30；本轮只复核历史，不重跑实机。

## 源码已锁定

主线`csrc/_310P/memfabric_mm_ar/`的六个`.cpp/.h/.asc`文件与
`feat/310p-memfabric-public-api` @ `fa1c42179818f5516b390d8bce3d39a9d280808f`逐Git blob完全一致。
对应现有public API / ABI7实现，不把目录外同名旧prototype替换进来。
具体哈希见[统一审核清单](../../releases/v1-v2-20261030/SELECTION_AUDIT.json)。

| 登记条目 | 实际关系 | 规整决定 |
|---|---|---|
| V1-0 批流水 | 完整实现的大M生产/传输/本地ADD路径 | 保留在同一个ABI7实现中 |
| V1-1 小M N-split | 同一runtime按形状选择；每层权重切片缓存 | 保留；不是另一个可单独取代V1-0的版本 |
| V1-2 单波M-split | 同一runtime中间M路径；默认ceiling2048，可配置1024 | 保留；须按实际配置找历史数据 |
| V1-3 generation READY | 同一实现的就绪通知优化 | 保留；不能与V1-0拆成互斥选项 |
| V1-4 Graph预热/生命周期 | 相同实现的另一执行模式 | 单列证据，不能拿Graph成功替代Eager成功 |
| V1-5 standalone原型 | 历史旧入口 | 保留快照，不选为默认正式候选 |

## 为什么还不能宣称“最佳正式V1已封版”

1. 2026-09-29报告有q=1/2/4及Graph/Eager结果，但早于2026-10-16单波M-split改动；文档当时默认q=2，当前入口默认q=1。不能把旧q=4大M最快直接套给最新全M实现。
2. 当前ABI7还包含small-M开关、single-wave ceiling等设置。选择最优必须找到实际源码/二进制及配置与样本的对应关系，不能仅凭最新提交名。
3. [SEL-01](../../memfabric_mm_ar_v2/docs/SEL01_STOCK_V1_V2_EAGER_4096_2048_TP2_BATCH10_REPORT_20261024.md)中V1 q=1三次自然尝试均未完成，发生decode相位自旋/超时。它没有有效完成的Eager性能成绩；不把这些失败解释成性能快慢，也不能写成生产可用。

因此当前锁定的是**完整ABI7参考源码**；最优配置与最终正式winner仍待历史查找。若服务器没有后续修复成功数据，就明确封存为带已知限制的历史参考，不能假称稳定正式发布。

## 要找的历史证据

只找已存在的q=1/2/4、N-split/单波开关与形状数据，完整双Rank原始结果、源码HEAD、实际Device/Host/SDK/KVER SHA；同时查SEL-01之后是否存在同类故障修复及完整成功记录。
详细查找范围及回传格式见[OpenCode任务](../../releases/v1-v2-20261030/OPENCODE_HISTORY_LOOKUP.md)。不为此重新运行benchmark、模型或Graph。
