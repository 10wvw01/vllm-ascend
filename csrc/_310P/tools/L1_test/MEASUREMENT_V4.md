# L1专业测量合同V4

本轮以Stock的公开Tensor调用层为基准；修改融合算子的Host Tensor入口和测量框架，
不修改原C ABI、设备Kernel、通信协议或Serving默认。历史V3合同数据保留原身份。

## 实际统一的调用边界

| 对象 | 计时内入口 | 输出/输入生命周期 |
|---|---|---|
| Stock | `F.linear(X,W_NZ)` → `dist.all_reduce(Y)` | 框架正常分配Y，HCCL完成依赖包含在同Stream窗口 |
| V2所有内建入口 | `torch.ops.memfabric310p.mm_ar(X,W,context)` | 算子内部获取持久槽、创建Storage/Tensor租约、保活异步输入 |
| V3所有内建入口 | 同一个公开Torch入口，按context调用冻结算法会话 | raw版由公开入口管理槽；A2/A3由原算子池获取/包装/保活 |

这是实际Torch dispatcher实现，不是给ctypes结果改标签。实现位于
`csrc/_310P/common/torch_tensor_entry.py`及V2/V3各自`src/torch_adapter/tensor_entry.py`；
L1适配器只加载公开入口并传X/W，不分配输出、不提前包装Y、不代办每调用同步。
原始C/Session保留为实现细节与历史复现接口；新成绩对应新的公开Tensor入口。

持久建池、权重NZ物化、输出底层池分配、Event首次记录属于明确的冷态准备；
每次验证、槽获取、Storage包装、输入保活和实现原有同步均在公开调用内。
不强迫融合算子采用Stock的分配策略；策略差异是算子实现的一部分。
公开入口只支持已验证的原Stream/Eager/FP16 ND X与NZ29 W；不代表已接入模型调用点或Graph。
输出及其Storage别名持有租约，全部释放后才可复用；异步失败保留不确定输入和池，停止双Rank。

## 四条路径

1. 功能门禁：采样前后完整数值/格式及立即同Stream下游ADD，不计时；冷态另做保留所有Y的连续提交检查。
2. ready：逐调用记录完整输入到输出延迟，每次末尾同步；保留组内全部Y到组尾。
3. steady：连续提交一组，保留全部Y，只在组尾记录完成并等待；每组总时长除调用数。
4. host/profiler：独立诊断段及前后无打点对照，Profiler范围只包执行，不包状态、数值或ADD检查。

N始终是每臂每轮每M每模式的总算子调用数，W是预热总调用数。
默认`batch_size=2`、`working_set=4`；尾组按实际调用数计，不补调用或丢输出。
ready有N个计时单位；steady有ceil(N/B)个计时单位，其分位数是**块平均每调用时间**，不是单次p99。
两Rank先对同计时单位总时长取max，再以总时间/总调用数求均值：

\[
\bar t=\frac{\sum_b\max(T_{0,b},T_{1,b})}{\sum_b c_b}.
\]

组内无CPU barrier、协议D2H、torch.equal和下游ADD；ready的逐次末尾同步属于其定义。
CPU barrier只在预热/采样阶段起点；组尾读取协议状态及检查输出地址不重叠，不计时。
数值校验只在采样前后与各阶段最后一组执行；**不声称每个计时样本已独立数值核验**。
因此steady是有界连续提交基准，组间仍有审计/释放，不是无限持续推理吞吐。

A2/A3最多同时保留2个输出；A2原有内部同步如实保留/标记，故不承诺其Host完全异步。
raw V2/V3冷态配置持久输出组，受64地址首写表及32窗口上限约束。
V1/未知external尚未审计连续输出，窗口只允许1；显式请求B=1，不静默缩窗口或删版本。
超过容量在设备前拒绝，算子内也拒绝覆盖活跃输出。旧地址首写保护保留。

## 测量质量和结论强度

- Wall包含公开调用、统一Event记录及末尾Stream等待；Event称Stream区间，可能包含Host提交空档，不称纯Kernel时间。
- 独立空计时封套校准保存原件及开销比例，**不减去校准值**。超过5%不签明确收益；池容量不能为了摊薄时钟而绕开。
- 权重固定NZ，输入默认4组轮换；记录字节数与不自动清缓存，不冒充模型全站点工作集。
- 预热记录最后8个计时单位的范围/中位数；不足6个单位记短预热未验证。时间收敛不等于核频/温度已稳定。
- 轮间ABBA/BAAB交替。N是相关调用次数；独立轮次R才用于配对bootstrap，不把200个调用当200次独立实验。
- R<3只作Screen；R>=3给出2000次配对轮次bootstrap的95%区间，区间不消除系统偏差。
- shared永远只作观察；诊断不签收益。独占仍须漂移、预热、时钟开销与置信区间共同通过。
- 完整43组保留；`campaign plan --cases ...`明确选择配置供日常V3迭代，每配置仍有自己的Stock对照。
- 所有结果仅针对登记公开算子入口；模型调用点集成、真实Qwen Step/TTFT与Graph仍NOT_RUN。

## 准入

新合同`L1_TENSOR_IO_V4` / campaign schema `MF310P_L1_FIXED_M_CAMPAIGN_V3`要求重新生成计划和冻结。
新增TorchNPU Storage桥必须实际编译，原C/Device工件及公共Tensor Python源都须冻结。
快速shared W2/N10/R1；正式独占仍单独授权执行，R1也只是性能筛选，不自动追加R3。
CPU假后端与真实CPU Torch dispatcher测试只验证Host语义；CANN/NPU数值/路由/首写/连续提交仍交OpenCode验证。
