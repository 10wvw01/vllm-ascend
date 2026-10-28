# E4 Attempt1｜2026-10-28主线原始证据保全（STOP，无E4测速）

> 这是唯一项目主线 `dev/310p-memfabric-mm-ar` 中的选择性**原始文本镜像**，不合入E4实验Kernel/Host/过时工作分支治理文档。源分支`work/310p-mm-ar-e4-layered-pipeline-20261028`，执行HEAD`1e20d3522cc9dd8040b257fae8d887e7d12f3b5e`，结果归档提交`5b727e9f9153e8292c6022a14b63e974dfb0db03`，[PR #21](https://github.com/10wvw01/vllm-ascend/pull/21)，run_id=`e4-layered-20261028-001`。

## 同源证据保全清单

原实验分支按路径逐文件复制**23份原始文本证据**，没有重新生成或修改原始样本，逐目录核对来源和主线 **Git blob SHA完全一致**：

| 类别 | 数量 | 主线原始内容 |
|---|---:|---|
| [中文实机报告](A1_E4_第一轮实机验证报告_20261028.md) | 1 | 原始STOP及阶段状态 |
| [G0](g0/g0_identity.json) | 1 | 目标卡/双Die、CANN、原Host/Device/Plugin/KVER SHA、隔离工作区 |
| [CPU](cpu/) | 2 | CPU回归日志，2049形状规划审核JSON |
| [build](build/) | 2 | 真实BiSheng构建manifest、私有SO SHA清单 |
| [真实NZ路由](campaign_attempt1/routes/) | 8 | Rank0/1×M2048/4096，各原生route JSON和CANN kernel_details.csv |
| [Smoke与终止现场](campaign_attempt1/sessions/) | 4 | E3-G Rank0/1原始Smoke JSON、E3-G日志、E4-T2048被SIGTERM打断日志 |
| [Campaign](campaign_attempt1/) | 3 | STOP进度、测前SO SHA、原始30秒`npu-smi`巡检含PID506263 |
| [STOP](stop_incident/) | 2 | 逐时事件取证、终止时NPU状态 |
| **总计** | **23** | 所有原样复制文件均已按SHA验证 |

原分支另有深层NZ Profiler、空白或冗余编译/host日志、更多G0设备输出，本主线只选择可独立复核STOP和已通过门禁的最小原始闭环。**不能声称主线已镜像原分支全部设备数据或全部Build日志**。

## 已证事实与限制

- G0曾PASS：card16384 dev2/dev3同卡双Die，真实装载Host`1dc46d7d…`、正式D4`2a19390c…`、插件`8eb65de9…`、KVER11`25e16b0a…`；CANN9.1 BiSheng dav-2002。CPU原仓80 passed、新E4合同9 passed（另有subtests）、Python/Shell语法、项目静态gate通过；2048～4096共2049整数M的**仅CPU**规划器通过。
- E4全部四变体（T2048/T4096/T8192/PacketAdd）在单一私有SO内成功**编译**，含Vector UB48KiB的8192路径，**不等于实机运行正确**。
- Stock原生NZ+HCCL路由Rank0/1×M2048/4096均核对，四组CANN CSV证明`aclnnMatmulWeightNz`/hcom，0 TransData。
- E3-G旧候选双RankSmoke在M2048/4096 bit-exact/status0；同一轮M4096事件样本~2.3ms与~6.5–6.8ms剧烈交替**被外部并发占卡污染，只用于正确性，不作为性能锚点或有效E3-G性能变化**。
- 目标卡G0时空闲，首份run中巡检UTC06:29:39已有外部PID506263跨四卡占用并持续增长；本轮`e4t2` torchrun又被执行工具SIGTERM打断，未执行设备Kernel，**两事件相互独立**。后续E4 T4/T8/PacketAdd Smoke与七臂Screen**NOT_RUN**；14场Confirm未获授权亦NOT_RUN。未知对E3-G/Stock的实际增益；不出现E4数值。
- **下轮P0工程准入**：先取得真实连续单卡双Die独占租约并重G0；修改/测试本候选分支的`e4_campaign.sh`只读巡检，使异步外部新PID被识别后立刻以STOP标记、阻断新阶段、保全现有日志；不能依靠“30s日志有人之后再看”。运行器采用真正的独立session启动和进度轮询，避免上层工具超时TERM，确保监控和测量进程同时退出。绝不自动清KFC、杀外部PID、换卡或覆盖共享SO。环境不合格就停在G0，**无需再重跑CPU/构建可复用产物时也必须严格核对源码与二进制manifest HEAD/SHA**。
- **E4连续M**仅离线算法规划，真实变长设备Kernel和Serving选路未实现；Stock生产默认和L2/L3/L4 NOT_RUN均不改变。

**结论：E4工程编译准入已通过、设备算法性能没有证据；本轮的STOP合规，**但“巡检只记录不强制停止”是下一轮执行前必须补的门禁缺陷。不得把本次E3-G污染Smoke耗时或E3旧轮p50写成E4新优化成绩。
