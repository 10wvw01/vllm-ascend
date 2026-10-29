# 统一L1框架复核与交接

2026-10-30，基底主线8e8c4def。范围仅工具、设计文档及历史数据读取；未修改Kernel、Host ABI、通信协议、Serving或MemFabric依赖。

## 第一轮：调用边界、生命周期与执行隔离

- ready从就绪Tensor调用前到完整输出完成；chain立即提交真实残差ADD，中间没有额外stream synchronize。必要的每次准备/包装/拷贝/seed/事件仍在run里。
- 所有臂恰好W预热、N有效样本；首写保护、持久池及路由检查另列。检查在窗口后；输出保留到下游结束。
- 每作业新torchrun TP2进程。Stock仅HCCL+Gloo，融合仅Gloo；清除继承的MemFabric/旧trace/LD_PRELOAD环境，并核实际映射库。V2/V3各用自己的冻结运行时。
- 按每个M独立核实际NZ/fused/HCCL路由。框架及调用依赖源码和所有已声明工件冻结，实际执行要求源码已提交。声明文件不等于可复现构建证明，G0/真实二进制审计继续负责后者。
- 调度器调用现有占卡/健康守护，故障停止自有进程组，保留现场，不ADD重试或继续下一臂；共享模式仍不豁免健康/协议/数值。
- 原A3/7核/tail图明确生产与通信依赖；Core0兼计算不画成另一个独立调度线程。

## 第二轮：统计、诊断扰动与证据真实性

- 反向Rank离群样本回归证明必须先取每样本双Rank max再均值；不先平均Rank、不删慢样本。
- 校验完整计划、作业、合同、runtime身份、全部形状/场景、W/N、样本编号、FP16与真实下游正确性。缺Rank/缺ABBA/混配置/非有限时长均失败。
- 同时检查Stock夹逼、候选重复、跨轮和前后无打点漂移；诊断相对两侧均值报扰动。共享结果与受扰动诊断分别标记，不签发布GO。
- Profiler正式N次采样有独立W预热，路由采集在性能外。各形状/场景独立目录，避免覆盖同名CSV。
- 各版Kernel统计与可用Host细段归到公共报告；核内MM/seed/wait/ADD/drain与SDMA链路缺证据标UNKNOWN，不返回0、不相加为100%。
- 历史p50仍是p50，36行Host范围与来源SHA可复算；6份最终CSV不升级成18份完整采集证据。
- 35份文档的来源和相对链接校验通过；历史blob未展开时直接核Git对象存在性，不强制恢复废止源码。

## 验证与实机状态

CPU回归使用：

```bash
PYTHONPATH=tools/310p_mm_ar:csrc/_310P/memfabric_mm_ar_v2/test/perf:csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts:csrc/_310P/memfabric_mm_ar_v3/test \
python -m pytest -q tools/310p_mm_ar csrc/_310P/memfabric_mm_ar_v2/test csrc/_310P/memfabric_mm_ar_v3/test --import-mode=importlib
```

新增框架21项测试包含计划/统计污染、漂移、上下游边界、runtime/SO篡改、trace环境泄漏、历史别名、CSV管线、文档链接、完整模拟TP2调度及首失败STOP。
全量CPU回归428项及36个子测试通过；静态项目入口和变更台账检查通过。按用户要求，本轮不执行任何格式检查。
新框架CANN/NPU、真实Kernel/SDK装载、Profiler版本适配及设备数值/生命周期测试仍NOT_RUN。

## 服务器最小下一步

先用SSH拉取审核提交，在隔离目录核SHA，不动共享安装。按原各版执行单完成G0、CPU、私有编译和短Smoke；Stock-only不要求融合私有编译。
填写实际runtime-template并冻结证据；先W2/N2/R1短试，验证Stock-only、D4、A3三候选的ready/chain均正确和路由真实，再执行正式W20/N100/R2。
独占/共享由现场明确选择，整批每臂同样N，完成全部前后对照。历史实现没有匹配工件/工厂时先报Blocked，不用最新版填空。
归档双Rank原始JSON、路由、maps/SO/KVER/设备物理号、guard日志及恢复后的SO SHA。此文不构成已获得实机GO。
