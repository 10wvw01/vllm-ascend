# V3 设计版本索引

统一采用VN-M编号，保留旧名称；同一实现的不同参数点不冒充新架构。
C2选型/D2设计/D3编码/D4实测属于同一个AddTile2048实现，统一V2-1；C3指2026-10-24的CleanLines unroll4候选。
Graph、MM shadow及G1C/G1D诊断单独列明，不能混入Eager完整MM+AR收益统计。

| 编号 | 原名称 | 执行适配 | 类别 |
|---|---|---|---|
| [V3-0](versions/V3-0.md) | 原始V3 SDMA ADD R/S + COPY Gather | external | 完整候选/历史实现 |
| [V3-1](versions/V3-1.md) | V3.1 Reduce/Gather重排 | v3 | 完整候选/历史实现 |
| [V3-2](versions/V3-2.md) | V3.2方案A / A2 prepare版 | external | 完整候选/历史实现 |
| [V3-3](versions/V3-3.md) | A2 merged-wave单控制集合 | v32 | 完整候选/历史实现 |
| [V3-4](versions/V3-4.md) | A3固定S+独立Y START版 | external | 完整候选/历史实现 |
| [V3-5](versions/V3-5.md) | 原A3 首READY合并 | a3 | 完整候选/历史实现 |
| [V3-6](versions/V3-6.md) | A3-2 七核计算+专职通信 | a3-2 | 完整候选/历史实现 |
| [V3-7](versions/V3-7.md) | A3-tail Core0末尾计算 | a3-tail | 完整候选/历史实现 |

框架合同见仓库`csrc/_310P/tools/L1_test/README.md`。旧设计、实验数据和失败记录保持原文；这些文档补充统一导航，不覆盖原记录。
