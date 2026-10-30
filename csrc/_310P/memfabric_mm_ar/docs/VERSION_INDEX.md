# V1 设计版本索引

统一采用VN-M编号，保留旧名称；同一实现的不同参数点不冒充新架构。
C2选型/D2设计/D3编码/D4实测属于同一个AddTile2048实现，统一V2-1；C3指2026-10-24的CleanLines unroll4候选。
Graph、MM shadow及G1C/G1D诊断单独列明，不能混入Eager完整MM+AR收益统计。

| 编号 | 原名称 | 执行适配 | 类别 |
|---|---|---|---|
| [V1-0](versions/V1-0.md) | 公共API批流水 / cooperative MM | v1 | 完整候选/历史实现 |
| [V1-1](versions/V1-1.md) | 小M N-split串行分支 | v1 | 完整候选/历史实现 |
| [V1-2](versions/V1-2.md) | 中M M-split单波分支 | v1 | 完整候选/历史实现 |
| [V1-3](versions/V1-3.md) | generation READY / 去逐波清零 | v1 | 完整候选/历史实现 |
| [V1-4](versions/V1-4.md) | Graph预热与生命周期版本 | external | graph |
| [V1-5](versions/V1-5.md) | 早期standalone o_proj原型 | external | 完整候选/历史实现 |

框架合同见仓库`csrc/_310P/tools/L1_test/README.md`。旧设计、实验数据和失败记录保持原文；这些文档补充统一导航，不覆盖原记录。
