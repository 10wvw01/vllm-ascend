# V1/V2 版本规整与封版审核（2026-10-30）

当前完成源码/历史证据锁定与候选归类，**最终最优版本封版仍待历史证据补齐**。没有安排重跑，也没有更改正式构建、Serving路由或算子代码。

- `SELECTION_AUDIT.json`：37个源码/构建输入的Git blob与SHA256、9项证据来源、四批机器评估及未决项。
- `SOURCE_SHA256SUMS`：从仓库根目录执行`sha256sum -c csrc/_310P/releases/v1-v2-20261030/SOURCE_SHA256SUMS`可核对源码。
- [V1规整与选型](../../memfabric_mm_ar/docs/RELEASE_SELECTION.md)。
- [V2规整与选型](../../memfabric_mm_ar_v2/docs/RELEASE_SELECTION.md)。
- [交给OpenCode的只读历史查找任务](OPENCODE_HISTORY_LOOKUP.md)。

## 当前可作出的决定

| 代际 | 规整结果 | 可以锁定什么 | 最终封版阻塞 |
|---|---|---|---|
| V1 | 主线六个算子源码文件与public-api `fa1c4217`逐blob一致；小M/单波/批流水/generation是同一ABI7实现的能力 | 当前完整ABI7源码身份；旧prototype不作为优先候选 | 最新q/模式最优配置尚缺完整原始对照映射；SEL-01 Eager 0/3完成，未找到后续匹配修复成功证据 |
| V2 | D4仍为当前构建参考；E1、E3-G、E4-T8按形状及证据级别分类 | D4/E1/E3/E4源码与历史E4构建manifest SHA完全一致；已有测试可复用 | E1偏向M2048，E3-G在M4096有两场稳定证据，E4-T8仅单场Screen更快；不能把它称为所有形状的正式最优 |

**源码规整/锁定、性能最优选定、生产发布是不同状态。** 本次manifest不写虚假的正式winner、不创建final标签；补齐历史证据后再填写唯一选择、明确适用形状/模式和依赖版本。

## 已找到的数据，OpenCode无需重复收集

五臂与E2的机器评估已在主线历史快照中完整保全（并非丢失）：

- five-arm assessment Git blob：`248e6be528ecddf8c39792d68011df19f2a4b1ec`。
- E2 assessment Git blob：`91c2070fd7e0bb19fb24c910f4f77ac75a9b017f`。

可在仓库使用`git cat-file blob <SHA>`读取；来源路径、分支完整HEAD及SHA256均在本目录JSON。
E3 assessment及E4 attempt2 Screen/attempt3b部分数据已在主线实体目录，保留原始漂移和Screen状态。

不将旧p50改成新框架算术均值，不混合不同批次绝对耗时求一个总冠军，不以新框架尚未上机为由要求V1/V2重测。
