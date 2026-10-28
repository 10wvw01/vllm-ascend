# E3 2026-10-28：主线关键原始证据归档清单（只读）

> 项目唯一主线`dev/310p-memfabric-mm-ar`。来源实验分支`work/310p-mm-ar-e3-chunk2048-20261028`、源码/真实测试HEAD`3e58ae389784d21eca5fb4bf18bad0fc32de6279`、报告归档HEAD`78a95d6e6c60f017b7769e8030b33a58f063f062`，PR [#20](https://github.com/10wvw01/vllm-ascend/pull/20)。本目录所有文件按UTF-8原始内容逐文件从该提交复制，没有重算/编辑原始采样；下述Git blob SHA检查在拷贝后逐目录对照**全部同名文件一致**。

## 本主线已保全的原始证据

| 目录 | 主线已保全文件数 | 原始证据类型 |
|---|---:|---|
| [机器评估](e3_assessment.json) | 1 | 原始`E3_CHUNK2048_ASSESS_V1`完整JSON、两形状配对统计、每场p50/p99 |
| [G0](g0/) | 5 | 设备板卡/健康/四卡巡检、真实身份JSON、冻结SO/KVER SHA |
| [正式八场+Smoke](campaign_full/sessions/) | 28 | **16份正式双Rank JSON（每场100样本）、4份G/T双RankSmoke、八份测后NPU健康日志** |
| [四组Stock真实NZ路由](campaign_full/routes/) | 8 | Rank0/1×M2048/4096，真实CANN kernel_details CSV与核验JSON；只保留最小可复核原生NZ/HCCL证明文件 |
| [Campaign环境及身份](campaign_full/) | 7 | progress、commit、git_status、二进制before/after SHA、测前后NPU巡检 |
| [首版L0C异常](bugfix1_l0c_overflow/) | 2 | 原始BUG说明和首次Smoke失败进度；实机异常原因见[报告](../../docs/A1_E3_CHUNK2048_实机报告_20261028.md) |
| [正式中文实机报告](../../docs/A1_E3_CHUNK2048_实机报告_20261028.md) | 1 | 从E3 PR#20原文复制 |
| **合计** | **52份** | 同名Git blob SHA一致；本归档说明另外计1份 |

**原始全部统计不因复制改变**：M4096真NZ Stock2022.553/E1 2271.836/E3-G2119.506/E3-T2267.090µs；E3-G vs E1 +7.187%，比Stock慢96.953µs。M2048 E1跨场漂移6.192%>5%，整批`INCONCLUSIVE_DRIFT`。所有E3-G/T在修复后Smoke及八场中正确，但新候选L2/L3/L4仍`NOT_RUN`。

## 未进入本批主线归档、不能伪称已归档的资料

1. **深层原生CANN profiler二进制/trace_view原包**：体积与冗余较大，原实验分支内仍保留；主线已经保全四组可审计的`kernel_details.csv`与对应SHA证明JSON。若将来要做微观Kernel排序分析，仍应查看PR#20实验分支或服务器原件。
2. **首次507015原始PLog、`smoke_e3t.log`等**：E3报告有引用，但`bugfix1_l0c_overflow/`在原GitHub分支只有README与progress，原件没有公开；需现场OpenCode补采（若仍存在）或永久标注`SERVER_ONLY_OR_MISSING`，不能编造。
3. **`build/`两版manifest/构建日志和`card_watch_30s.log`**：E3报告引用而原GitHub证据目录未见；需现场核实。现有`g0`身份、`campaign_full/binaries_before/after.sha256`和逐场NPU日志已能证明本次报告所述的核心版本身份及逐场输出，但不能替代构建日志/连续30s原始巡检。
4. **E3实验Kernel/Host/测试源和PR**：依然在PR#20及其后继E4分支，不属于已采纳的正式生产代码；不要把E3工作分支整个merge进主线。真正默认Stock不会因为这份归档改变。

## 审计、回滚和后续规则

- 本目录和主线`docs/PERFORMANCE_BASELINES.md`/ADR-020/`PROJECT_CONTROL.md`即便PR#20关闭也可独立读取**全部52份本轮关键可复核文本数据**。
- 如有后续源分支更新，禁止直接覆盖主线归档；新实验须独立`run_id`和目录，原数据保持不变，校验Git blob SHA后另行审签。
- 在E4新增真实结果之前，不能将其源级静态审查/纯CPU合约的通过写成任何310P3/BiSheng/Serving收益。

## 独立数值交叉核验（原始Rank样本对机器评估）

将主线16份正式Rank JSON逐场读取，针对M2048和M4096分别把同索引rank0/rank1的100个NPU Event样本取`max(rank0[i],rank1[i])`，再取各场p50，与本主线`e3_assessment.json.raw_sessions[].Ms[].paired_p50_us`逐项比对：**八场×两形状共16个结果全部一致，最大仅约0.001µs显示精度舍入差**；每个样本规模100、非Stock结果均为`bit_exact=true,status=0`。本交叉核验没有改变原始JSON或从漂移失败样本挑选新的结果；M2048整体`INCONCLUSIVE_DRIFT`、M4096稳定的机器合同结论保持原样。
