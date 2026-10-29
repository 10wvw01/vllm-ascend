# V2 设计版本索引

统一采用VN-M编号，保留旧名称；同一实现的不同参数点不冒充新架构。
C2选型/D2设计/D3编码/D4实测属于同一个AddTile2048实现，统一V2-1；C3指2026-10-24的CleanLines unroll4候选。
Graph、MM shadow及G1C/G1D诊断单独列明，不能混入Eager完整MM+AR收益统计。

| 编号 | 原名称 | 执行适配 | 类别 |
|---|---|---|---|
| [V2-0](versions/V2-0.md) | T0 / Tile1024 / D1-A正式基线 | v2 | 完整候选/历史实现 |
| [V2-1](versions/V2-1.md) | C2选型→D2设计→D3编码→D4实测 / AddTile2048 | v2 | 完整候选/历史实现 |
| [V2-2](versions/V2-2.md) | C3 CleanLines四路展开 | external | 完整候选/历史实现 |
| [V2-3](versions/V2-3.md) | F2 Chunk/Overlap扫描 | v2 | 完整候选/历史实现 |
| [V2-4](versions/V2-4.md) | T1小M固定成本候选 | external | 完整候选/历史实现 |
| [V2-5](versions/V2-5.md) | B1 原生NZ MM + MemFabric AR-only | v2_b1 | 完整候选/历史实现 |
| [V2-6](versions/V2-6.md) | B2 full MM后AR | v2_private | 完整候选/历史实现 |
| [V2-7](versions/V2-7.md) | E1 DCCI_ENTIRE | v2_private | 完整候选/历史实现 |
| [V2-8](versions/V2-8.md) | E2-R | v2_private | 完整候选/历史实现 |
| [V2-9](versions/V2-9.md) | E2-P | v2_private | 完整候选/历史实现 |
| [V2-10](versions/V2-10.md) | E2-P512 | v2_private | 完整候选/历史实现 |
| [V2-11](versions/V2-11.md) | E3-G group2048 / 两次1024 MM | v2_private | 完整候选/历史实现 |
| [V2-12](versions/V2-12.md) | E3-T group2048 / 单次2048 MM | v2_private | 完整候选/历史实现 |
| [V2-13](versions/V2-13.md) | E4-T2 AddTile2048 | v2_private | 完整候选/历史实现 |
| [V2-14](versions/V2-14.md) | E4-T4 AddTile4096 | v2_private | 完整候选/历史实现 |
| [V2-15](versions/V2-15.md) | E4-T8 AddTile8192 | v2_private | 完整候选/历史实现 |
| [V2-16](versions/V2-16.md) | E4-P packet-ready ADD | v2_private | 完整候选/历史实现 |
| [V2-17](versions/V2-17.md) | MM shadow站点诊断 | external | diagnostic |
| [V2-18](versions/V2-18.md) | G1C 阶段进度取证 | external | diagnostic |
| [V2-19](versions/V2-19.md) | G1D minimal forensics | external | diagnostic |
| [V2-20](versions/V2-20.md) | LOOP4 串行MM+AR隔离对照 | external | diagnostic |

框架合同见仓库`tools/310p_mm_ar/l1/README.md`。旧设计、实验数据和失败记录保持原文；这些文档补充统一导航，不覆盖原记录。
