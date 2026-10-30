# V1/V2 大M性能优先选型与源码封存（2026-10-30）

范围固定为用户确认的**M=2048～4096，310P3单卡双Die/TP2、K=N=2048、FP16**。
按现有成功结果选性能好的版本；不因文件名为“正式”、实验命名或验证轮数较多而改选更慢版本。
V1/V2均不重跑，失败实验不参加成功记录的性能排名，原件与适用模式限制保留。

## 已选结果

| 代际/测试点 | 性能选择 | 证据与封存状态 |
|---|---|---|
| V1 大M | 历史成功q=4、batch_m1024 | 旧成功q扫描优于q1/q2；M4096约2.36ms。配置已选，准确源码/二进制映射延期到下次实机时查找 |
| V2 M2048 | V2-7 / E1 | E4 attempt2完整调用Wall均值1394.627µs，Event p50 1279.596µs；选定并锁源码 |
| V2 M4096 | V2-15 / E4-T8 | 同批Wall均值2175.690µs，Event p50 2076.211µs；选定并锁源码，保留单轮Screen标签 |
| V2 M2049～4095 | 尚无逐点最优记录 | 不凭两个端点发明切换阈值或声称整个连续区间已有最优实测 |

V2数字均由同一批每臂30样本的原始双Rank数据复算：每个序号先取Rank max，再算均值/分位数。
完整调用Wall含接口调用、提交及流完成等待，未去掉任何样本。历史Event p50保留原名，未冒充均值。
T8与T4的M4096 Wall均值仅差0.128µs，实际上接近持平；T8中位耗时更低。
本次按观测性能选择T8，不声称已证明长期或统计上稳定优于T4。

D4是现有CMake正式构建参考，但后续five-arm M4096同批E1已快于D4；**D4不因现有正式身份入选**。
E3-G保留为回归/历史对照，不因其复核轮数更多而覆盖本次T8选择。Stock为另列对照，不能把代际内部选择写成超过Stock。

## 封存与交付边界

- `PERFORMANCE_SELECTION.json`：V2原始30样本复算、完整Wall/Event统计、按M选择、工件manifest、14份原件来源及哈希。
- `evidence/e4_attempt2/`：14份未修改的原始Rank JSON，逐Git blob与主线历史快照一致。
- `SELECTION_AUDIT.json`：37个源码/构建输入身份、历史批次和本次性能选择。`formal_winner`表示选型决定，生产启用仍为false。
- `SOURCE_SHA256SUMS`：首轮锁定的全部37个审核输入。
- `V2_SELECTED_SOURCE_SHA256SUMS`：选中V2设备源/共用依赖/私有入口和历史构建脚本的源码封存清单。
- [V1选型](../../memfabric_mm_ar/docs/RELEASE_SELECTION.md)、[V2选型](../../memfabric_mm_ar_v2/docs/RELEASE_SELECTION.md)。
- [OpenCode只读补齐V1成功版本身份](OPENCODE_HISTORY_LOOKUP.md)。该任务已按用户指令延期到下一次实机；当前只完成V2，不补测、不强求最新ABI7。

**V2已完成实测点的性能选择与源码封存；V1配置已选，完整历史源码封版待身份映射。**
本次保留原有二进制/ABI和独立调用方式，不把新拼装的运行时混合路由冒称已经实机通过。
不更改Serving默认或设备安装，也不发布虚假的生产GO/连续区间最优结论。

## 已有历史证据，无需重复查找

five-arm assessment Git blob：`248e6be528ecddf8c39792d68011df19f2a4b1ec`；
E2 assessment Git blob：`91c2070fd7e0bb19fb24c910f4f77ac75a9b017f`。
E3及E4原件也已在主线/无损历史快照，来源见审核JSON，不再要求OpenCode重复收集。
不跨批次拼绝对耗时排名；V1旧报告不是当前V2同批对照。

## V2代码梳理交付

所选实现现有公共C入口与独立构建清单：[代码图谱](../../memfabric_mm_ar_v2/docs/CODE_MAP.md)、
[接口与L1使用](../../memfabric_mm_ar_v2/docs/SELECTED_API.md)、[源码包身份](V2_CODE_PACKAGE.json)。
selected句柄固定M2048/E1或M4096/T8，不用实验kind或全局预热地址表。
E1/E4设备源未改；旧D4/Serving和历史脚本保留。新增Host/CMake/L1代码全量CPU452项+36子测试通过，
新组合CANN/NPU为NOT_RUN，不将原SO成绩转记为新SO测量。
当前`V2_SELECTED_SOURCE_SHA256SUMS`扩展为新代码包输入清单；`SOURCE_SHA256SUMS`及审计source_files
保留首轮历史基底身份（包括旧CMake），不作为新构建的全量校验。V1查找延期，不是本轮待办。
