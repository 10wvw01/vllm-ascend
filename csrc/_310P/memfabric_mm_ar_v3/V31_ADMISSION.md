# V3.1 开发与设备准入状态

## 已实现

- chunk512/1024/2048，沿用V2 MM tiling；2048保持两个1024规模子调用。
- 当前Reduce先提交，输出延后两组排空；跨组READY/REDUCE/GATHER统一分发。
- 每包校验epoch、phase、group、packet、dst、len及重复位图。
- 收齐Reduce立即提交Gather，所有源全波独立，末尾quiet+DONE保护下一波复用。
- 双16KiB UB交替copy，MTE2_MTE3保证数据就绪，MTE3_MTE2保证缓冲复用；事件动态申请并归还。
- 默认4MiB包，quick支持`--packet-bytes`选择1048576/2097152/4194304。
- 默认六配置：M2048/4096×chunk512/1024/2048，两轮30样本；旧性能臂不运行。

## 已验证与待验证

CPU88项通过，包括直接提取实际C++设备Progress的异步FIFO模拟：
32种调度×5组数×3包数=480场，验证双Rank偏斜、逐元素和、源未提前覆写及完整排空；
并拒绝错误epoch/dst/group/packet和重复消息。既有V2 CPU合同保留。
两轮源码审查核对mail顺序、发送源生命周期、首尾排空及MTE事件复用。

**当前没有CANN/BiSheng/NPU，设备编译和真实MTE事件行为未验证。**
G++协议模拟不能证明AscendC编译或设备安全准入，因此当前状态为CPU_PASS/CANN_PENDING。

## OpenCode执行顺序

沿用[原执行单](OPENCODE_EXECUTION.md)的隔离、固定HEAD、助手SDK、独占和守护合同。
MemFabric固定db38d75e，SDK助手不可缺失；原私有SDK可在完整manifest复核后复用。
V3必须针对本次完整提交重编，使用新的build/result目录，不改生产环境。

1. 先重新执行CPU检查和`tools/build.py`私有CANN编译。核查双UB与NZ2ND资源、
   MTE2_MTE3/MTE3_MTE2事件及Progress数组在dav-2002的编译结果；失败保留日志并修复执行BUG。
2. **设备编译成功、源码/全部SO/助手manifest一致且G0独占健康成立后**，才能运行设备门禁。
3. 执行quick的IB→ADD→全部chunk串行/流水协议→融合→尾块；任何失败停止当前轮并保留证据。
4. 全门禁通过才做六配置两轮短测，默认4MiB包，按原方法逐样本取Rank max。
   单独包长诊断用新的结果目录和`--packet-bytes`，不覆盖正式默认轮。
5. 对照本次V3.0原始记录及历史V2/Stock，报告完整算子p50/p90/p99、漂移和失败轮。
   不把CPU通过、代码交叠图或相对旧V3加速直接当成超越V2的结论。

当前可交付给服务器的是编译准入任务；满足第1/2步后再进入设备执行。
