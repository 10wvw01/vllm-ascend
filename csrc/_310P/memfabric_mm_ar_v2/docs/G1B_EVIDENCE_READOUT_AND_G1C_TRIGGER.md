# G1-B 判读规则与 G1-C 进入条件（离线准备稿）

> 2026-10-20。只供方案分析和 G1-B 结束后的归因；不修改正在运行的诊断分支或实机配置。  
> **本次新增 Python 脚本是纯离线工具**，不需 CANN/NPU/重编译，不得在本轮进行设备侧协议修复。

## 一、冻结的可观测量

| 日志 | 写入来源 | 可以证明 | 不可以证明 |
|---|---|---|---|
| `[MF_DIAG] tick epoch` 的 reqHead / reqTail | Host 每约 1s 对段内两游标做 16B D2H 快照 | 请求环在采样时有无前进迹象 | CP 硬件 SQ head/tail、Doorbell 成功、Cache Line 丢更新 |
| `[MF_DIAG] exit epoch` 的 exitReason / mainRet / consumed | Host 在读到非 RUNNING 时打印设备结果区 | 当前结果区的终态及错误报告 | 每个 kernel 必然已经从 Host stream 队列中完成同步 |
| `[MF_DIAG] start epoch` | Host `LaunchEpoch` 调用成功后打印 | Host 发起了新 Epoch 的提交 | 新 Epoch 已获得 AICPU 调度并进入主循环 |
| 507014 + 双 Rank plog/slog | Host/设备日志 | 在特定同步/故障检测点报告超时 | 该同步点一定是最早卡住的 kernel |
| `Get event num has exceeded 1000` | AICPU sharder 告警 | 告警发生并可能与排队相关 | 仅凭告警就确认 SQ 覆写或 MemFabric 根因 |

请求环布局：ReqTail=0x800、ReqHead=0x808，深度 64；二者位于同 64B Cache Line，不同写入者导致快照可能短暂不一致。**SQ 环是另一套队列，单靠 G1-B 不可将两者混淆。**

## 二、固定判断矩阵

1. **Epoch 先报告 ERROR**（exitReason=2、mainRet 非零），随后请求环积压或双 Rank timeout：优先排查 H2 设备编排异常；以明确的时间线判定是否是先因，不得以一次重拉就视为已恢复。
2. **请求环 tail 持续增加、head 在正积压下多秒不动**：表明可能存在消费停滞，需进一步调查 Epoch 消费与门铃路径；不自动等同于 SQ 满，优先申请 G1-C 的实时硬件 SQ 快照。
3. **Epoch 均按约 15s 正常续命，head/tail 持续推进直到观测中断**：H2 证据减弱，但不能彻底排除数据面延迟；进一步考虑 H4，G1-C 应优先补 Core0/其他核阶段戳。
4. **出现多个极短间隔 start/exit**：审查结果区的 RUNNING 写入与 Host 轮询的竞态，以及是否出现异常重复排队；`start` 只是 Host 下发，不作为设备已启动证据。
5. **没有复现且所有日志平稳**：只给出“本次未复现”，不宣称 Gate E 修复或 SQ 缺陷消除。
6. **日志在 timeout 前静默**：以最后成功日志时刻作为“最后可观测活动”，不能把它直接当作第一停摆时刻。

上述判断必须分别提供 dev0/dev1 的时序和原始日志坐标，区分“事实 / 可能性 / 排除证据 / 尚缺证据”。

## 三、离线分析命令（仅 G1-B 完成、日志上传或可访问后）

```bash
cd csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts

# G1-B Host 飞行日志的摘要：按 dev 分析游标、异常退出和积压停滞
python3 g1b_epoch_trace.py /path/to/G1B_F01/serve.log

# Gate E 的真实请求完成情况（F01 类 partial result 必须 FAIL）
# 原工况每轮 25 个请求，新增 Stock Graph 工况则为 50
python3 gate_e_forensic_verdict.py /path/to/G1B_F01 --prompts-per-rep 25
# 退出码 0=未观测到故障，1=FAIL，2=UNKNOWN（证据不足）

# 不依赖 CANN 的规则回归
python3 -m unittest test_g1b_epoch_trace.py test_gate_e_forensic_verdict.py -v
```

`g1b_epoch_trace.py` 的积压阈值是连续 5 次有效采样、reqPending>0 且 reqHead 不前进，只属于**调查线索**。单次无效游标样本不能直接认定缓存丢写。

## 四、如需 G1-C，下一层观察必须是什么

不提前修改设备侧代码。只有在 G1-B 结果说明 Host 请求环、Epoch 不足以确定首故障点时，再在独立实验分支制定可关闭设备诊断版：

- **控制面**：独立 AICPU 设备产物版本号（防 CP1 同名缓存），快照 SQ 实际 head/tail、depth、每封邮件需 3 SQE 的容量检查前后值、Doorbell 返回码及诊断错误码。
- **计算面**：V2 每核独立 64B 对齐进度槽，至少区分 MM→IB Ready→Signal→Wait→Add→Quiet→DONE；Core0 标记 chunk 与阶段，其他 7 核标记 MM 与 IB 阶段。
- **判据**：对齐双 Rank 和控制面的最后成功阶段，确定卡住方向；不得只根据最终 Trap/SwiGlu 报错推断。
- **保护**：诊断对现有协议、Shared Memory 布局和生产执行路径的侵入需单独静态审视，并独立于基线验证；未完成设计评审前不部署。

**当前 G1-B 正在执行，不允许为了用上此文档的判定器而在运行中 pull / checkout / 重装运行库。**
