# E4第一轮实机：运行中外卡占用STOP事件取证（2026-10-28）

> run_id: e4-layered-20261028-001｜分支 work/310p-mm-ar-e4-layered-pipeline-20261028 @ 1e20d3522
> 本文件记录E4 campaign第一次尝试中止的完整因果链、设备占用证据与止损依据。

## 1. 时间线（容器内时钟，UTC+00:00，宿主npu-smi同源）

| 时刻 | 事件 | 证据 |
|---|---|---|
| ~06:25 | G0完成：卡16384（dev2/3）健康、无进程、hugepages 0/0，四冻结SHA与u1_spec一致 | g0/npu_smi_g0.txt |
| 06:29:3x | campaign以私有环境启动（NZ路由→Smoke→Screen；E4_CONFIRM_APPROVED未设置） | campaign_attempt1/progress.txt |
| 06:29:50–06:30:02 | 真实NZ路由4/4证明：两Rank×M2048/4096各6次aclnnMatmulWeightNz+6次hcom、0 TransData | campaign_attempt1/routes/ |
| 06:30:10–06:31:36 | E3-G双Rank Smoke PASS：M2048/4096、6样本/臂/rank、bit-exact、status=0、首写与池创建成功 | campaign_attempt1/sessions/smoke_03_e3g_rank*.json |
| ~06:28–06:29 | **宿主级外部进程PID 506263出现并在全部4卡8个Die开始分配内存**（首份30s巡检快照06:29:39已含506263：5664dev0 2739MB、16384双Die、38144、43520） | campaign_attempt1/card_watch_30s.log 第1样本 |
| 06:31:36 | e4t2 Smoke torchrun启动 | sessions/smoke_e4t2.log |
| 06:31:37-38 | **本会话执行工具因120s超时向进程组发送SIGTERM**，torchrun与两个worker在启动后1秒内被杀，未执行任何设备Kernel（0条ARM=记录） | sessions/smoke_e4t2.log |
| 06:35–07:01+ | 506263持续占用并线性增长：16384双Die 2839→2959MB（约+20MB/5min），hugepages 0→1439/1439满；四卡同步增长，判定为存活的宿主租户服务而非本会话残留 | stop_incident/npu_stop_*.txt |
| ~07:02 | 按合同"设备被占用→立即STOP"正式停止，保全全部现场 | 本文件 |

## 2. PID 506263的定性证据

1. **不在容器PID命名空间**：本会话运行于Docker容器（/.dockerenv，仅31个可见PID）；506263及3589835/11744/12149均只在宿主npu-smi可见、/proc不可达，无法也禁止杀灭。
2. **非本会话残留**：506263同时出现在本会话从未触碰的5664/38144/43520各Die；本会话worker只使用ASCEND_RT_VISIBLE_DEVICES=2,3。死进程的内存记账不会持续增长。
3. **存活且增长**：16384双Die等量增长（2819→2959MB，+10 hugepages/min），四卡8 Die同步增长；43520两Die与5664dev1因他租户占满仅少量。
4. **非E3历史遗留**：E3整晚campaign的card_watch_30s.log中16384全程无任何外部PID、hugepages 0/0；506263为今日新出现。R2 attempt2/3曾记录dev2/dev3"短暂"外部占用，本轮为持续增长型，与R2模式不同。

## 3. 为什么停止而不是换卡或等待

- 委托硬性约束与ADR-014：运行中设备被占用必须STOP保全，禁止自动换卡、盲目重试；ADR-014仅允许G0阶段的卡级健康/占用失败换卡重G0。
- 四卡现状（07:01快照）：5664被3589835(42753MB)+506263占用；16384被506263占用且hugepages 1439/1439满；38144 dev4残留6873hugepages满+506263；43520 dev7持续Alarm+11744/12149。**无任何一张健康且独占的双Die卡可选。**
- 16384的hugepages已满：即使506263立刻退出且记账回收，也须先确认回收；若不回收则MemFabric建池无法进行。
- 继续Screen会把507263竞争计入性能样本，违反独占租约合同，必然产出INCONCLUSIVE数据（R2 Confirm漂移79%/663%即为外部进程轮换污染的先例）。

## 4. 本轮执行侧自身失误（如实记录）

- e4t2 Smoke的torchrun在启动1秒后被**本会话bash执行工具的120秒超时SIGTERM误杀**：nohup只防SIGHUP，不防进程组TERM。该失误与506263占用相互独立：即使不被误杀，campaign也会在506263共存状态下继续跑（campaign脚本30s巡检只记录不检测外部进程），产生违反独占合同的Screen数据。
- 修复与预防：下一轮campaign必须以setsid全脱管方式启动+独立轮询；建议（本轮未改，属合同脚本变更需审查）在campaign巡检循环加入"16384出现非本run进程→STOP"的自动门禁。

## 5. 中止时各阶段状态

| 阶段 | 状态 |
|---|---|
| G0四卡只读+SHA身份 | PASS（g0_identity.json） |
| CPU/静态门禁 | PASS（cpu/cpu_regression.log） |
| E1/E3-G/E4四实例隔离编译 | PASS（build/manifest.json，AddTile8192编译通过） |
| 真实NZ路由4/4 | PASS（routes/，含CSV） |
| E3-G双Rank Smoke | PASS（bit-exact/status0/首写） |
| E4-T2048/4096/8192、E4-PacketAdd Smoke | **NOT_RUN**（e4t2启动即被外部SIGTERM，无设备执行；其余未开始） |
| 七臂Screen | NOT_RUN（设备占用STOP） |
| 十四场Confirm | NOT_RUN（未授权，E4_CONFIRM_APPROVED从未设置） |

## 6. 现场保全清单

- /tmp/opencode/e4-campaign-20261028/：原始campaign目录（含被杀e4t2日志）原样保留
- /tmp/opencode/e4-build/：隔离编译产物与manifest原样保留
- evidence/e4_layered_20261028/campaign_attempt1/：全量复制归档
- stop_incident/npu_stop_*.txt：06:35、07:01两份npu-smi快照
- 本会话未杀灭任何宿主进程、未清除KFC、未修改任何共享SO或系统文件
