# 统一输入输出计时与A3阶段归因

## 固定合同

`ND_X_NZ_W_TO_READY_ND_Y_V1`：输入X为ND FP16[M,2048]、W为已物化NZ29 FP16[2048,2048]。
开始位于Tensor接口调用前；结束位于ND输出Tensor返回且所属流/必要通信依赖完成后。
Stock执行F.linear+HCCL SUM；V2执行D4 trial完整MM+AR并返回持久输出Tensor；
V3返回预分配输出Tensor；A3返回独立Storage包装的池输出Tensor。
允许实现持久预分配，禁止把每调用必需的reserve、wrap、seed、拷贝、检查或完成事件移到窗口外。
这统一功能和调用层级，不要求不同实现产生相同的内存分配成本。

主指标Host Wall为调用前（含起始Event record）至所属流同步返回；辅助指标Event包住同一调用。
Host与设备可交叠，Event可能含Host下发空档，Wall−Event不等于Host准备耗时。
正确性/status回读、下游ND消费验证、统计、输出释放在窗口后；输入生成、权重转换、初始化和预热在窗口前。
本轮只测真实M2048/4096，无输入padding；未来非对齐M必须在同一op内纳入padding和输出整理。
不包含模型其他算子、Graph、TTFT或冷启动，不将热态实验外推发布稳定性。

## 对照与进程隔离

统一单臂入口tools/unified.py；完整入口tools/unified_campaign.py。
每臂新torchrun进程，Stock只建HCCL和CPU Gloo，不装载MemFabric；V2只建Gloo，使用独立冻结D4/KVER11；
V3/A3只建Gloo，沿用锁定MemFabric db38d75e/KVER28。不得同进程混装两代SDK或HCCL/MemFabric数据面。
Stock有一次窗口外的轻量路由采集，必须看到真实aclnnMatmulWeightNz和HCCL，否则STOP。

完整批次11会话：Stock前、V2前、V3、三A3×compute512、三A3×compute1024、V2后、Stock后。
Stock/V2固定comm标签1024，候选逐项保留comm512/1024/2048，不挑最优点替代整表。
每会话两轮、每配置4预热+30正式样本；第二轮反转配置顺序。A3每会话四阶段：
无打点前→仅Host打点→仅Device打点→无打点后。其他臂无打点前/后。
每个样本完整正确性检查；每轮先同下标取双Rank max，再计算p50/p90/p99，禁止先取两Rank中位再max。

paired.json检查Rank身份、构建manifest、合同、全部配置/轮次/样本与有效数值；
包含前后及跨轮漂移、两类打点相对无打点基线的扰动。漂移>5%不判稳定收益。
打点扰动>5%时设备归因只作为受扰动样本描述，不将其绝对时间当无打点路径的真实占比。
campaign.json对Stock/V2外层前后漂移也检查；任何子会话失败停止，不拼接不同manifest。
批次status表示工具执行与证据完整，不自动签性能或发布GO。

## A3打点布局与含义

私有Host→Device ABI新增trace地址，必须同版重编；公共run ABI不变，默认关闭。
Device有独立TRACE=false/true模板实例；关闭路径不读取设备时钟、不写trace。
公共trace_control/fetch复用既有API，A3缓冲8192B；旧V32缓冲仍4096B。

| U64位置 | 所有者 | 内容 |
|---|---|---|
| 0..6 | Core0 | magic A3T2、M、compute、comm、epoch、rank、profile |
| 16+core×8 | 各核独占cacheline | 入口、最终核间会合前、会合后、输出可见后 |
| 80+seat×8 | 每座位生产核独占cacheline | 生产开始、MM结束、S可见、seed/Y可见、epoch发布结束 |
| 592+g×16 | Core0 | 就绪等待前后、READY提交前后、peer READY确认、ADD提交完成、最后观察到的入站ADD完成；tail另记强制补算前 |
| 720..724 | Core0 | Drain前后、quiet后、DONE后、尾部（最终quiet已移除，退出前） |

最多64座位、8通信批次；每核和每座位独占64B，避免trace本身出现cacheline共享写回覆盖。
座位trace在完成时写回，Core0控制区与各核末尾区在退出前写回；不改SDK私有队列。
原A3本地就绪等待按compute波记录，A3-2/tail按通信批次记录；解析器分别解释。

Host独立阶段：validate/reap、reserve、wrap、enqueue、输入完成事件提交。
这些阶段在仅Host对照中启用；不与Device打点混在同一批样本，避免无法区分扰动来源。

Device输出保留原始1024个U64/样本，并核对magic、几何、profile及单会话递增epoch。
按同核计数差得到每座位MM、S可见、seed/Y可见和发布开销，各核生产阶段占自身total的比例，
以及Core0本地等待、READY提交/收件泵、ADD提交、Drain/quiet/DONE与最终join。
MM包含Matmul内部搬运和写出，不冒充纯Cube周期；seed包含必要可见性处理。
READY等待包含消费穿插消息，ADD提交包含可能的环背压；Drain不是独立纯链路带宽测试。

设备计数未经频率校准仅标ticks，不猜微秒。不同核/不同Die时钟未校准，不直接相减。
所有核累计MM与SDMA重叠，不相加画100%饼图，不用fused−partial估计MM本体。
已有原始时间戳可观察同核先后和暴露尾部；跨核绝对时间线/真实链路占用仍需独立校准或硬件Profiler，
本工具不宣称已经测得它们。最后入站完成只代表本端观察时间；出站全部完成由quiet与DONE交换保证。

## 服务器操作

先遵循OPENCODE_EXECUTION.md和A3_ADMISSION.md：私有SDK、Host/Device与Tensor适配重建；
对新SHA重跑原A3/七核/末尾的quick六门禁。新trace模板还须在device阶段数值通过、缓冲不越界后才采样。
整个11会话需现场真实独占租约，子会话间隙不宣称有连续守护；占卡/告警/未知状态立即STOP。
保持SSH GitHub克隆，不改共享安装，不部署四卡，不杀外部租户，不自动重试ADD或换卡。

V2不自动重建旧发行版。先按历史G0确认D4源码身份、实际设备SO、Host/plugin、KVER11、物理号修复，
提供精确冻结spec（路径与SHA来自已审核产物，不使用示例值）：

```json
{
  "kind": "D4_TRIAL_KVER11",
  "source_head": "<已审核D4源码完整SHA>",
  "library": "<导出mf310p_v2_trial_*的绝对SO路径>",
  "sha256": "<SO SHA256>",
  "dependencies": {"<所有相关Host/Device SO和v11 launch JSON绝对路径>": "<各自SHA256>"},
  "environment": {
    "LD_LIBRARY_PATH": "<隔离V2及CANN依赖搜索路径，不含V3 SDK>",
    "MF_SDMA_ORCH_JSON": "<冻结的绝对路径>/libmf_sdma_orch_v11.json",
    "MEMFABRIC_HYBRID_HOME_PATH": "<V2 SDK>",
    "MEMFABRIC_HYBRID_EXTEND_LIB_PATH": "<V2库目录>"
  }
}
```

新构建先build_tensor冻结Storage，再freeze_v2；所有会话共用同一最终manifest。
若V2产物缺失或身份不明，STOP回传，不拿历史数字填本轮对照。

```bash
python csrc/_310P/memfabric_mm_ar_v3/tools/freeze_v2.py \
  --manifest <新构建>/manifest.json --spec <审核后的V2冻结spec.json>
python csrc/_310P/memfabric_mm_ar_v3/tools/unified_campaign.py start \
  --manifest <新构建>/manifest.json --out <新结果目录> \
  --card 38144 --devices 4,5 --lease-note '<现场覆盖整批的独占租约>'
python csrc/_310P/memfabric_mm_ar_v3/tools/unified_campaign.py status --out <新结果目录>
```

若已授权其他物理卡，card与两个直接物理DeviceID须一致；不设置可见号映射。
单臂调试使用tools/unified.py start（同参数另加--arm及--compute-rows），不得以单臂替代正式对照。
Host/Device或协议实现BUG修复后记录新SHA、重编重门禁、新完整批次，不放宽精度、不混拼数据。
归档manifest、V2 spec、实际maps、Stock路由CSV、所有checkpoint/失败现场、原始双Rank JSON、paired.json和campaign.json。
大体积trace可归档制品并提供可访问地址及SHA，不能只留下服务器临时路径。

## 当前验证边界

本地CPU/source/mock验证不模拟实际缓存、CANN模板资源或设备计数行为。
新Device模板编译、路由采集适配、V2库装载、数值与实机归因全部NOT_RUN。
不把旧版本六会话正确性当成新trace版本已获设备GO；现场仍须按顺序编译、门禁、对照。

本地最终验证：V3/V2 CPU与源码回归、治理测试合计179项PASS（其中治理38项），
Ruff、静态项目合同及git diff --check通过。全仓format.sh ci因未安装pre-commit未完成。
这些结果不替代新设备模板编译、生命周期/数值门禁和真实归因采集。
