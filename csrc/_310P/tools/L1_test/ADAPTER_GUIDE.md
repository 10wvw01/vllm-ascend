# L1_test 算子接入合同

统一工厂`create(version,spec,ctx,collection)`；测试框架提供预热、计时、数值门禁、双Rank配对和统计。
**以Stock的Tensor调用层为基准**，V2/V3每调用的工作必须落在融合算子内，不能由测试适配器代办。

| 家族 | L1薄适配 | 真正执行入口 |
|---|---|---|
| Stock | adapter_plugins/stock.py | 原生NZ F.linear → HCCL SUM，正常输出分配 |
| V1 | adapter_plugins/v1.py | 原V1 Tensor op；本轮不测，连续窗口尚限1 |
| V2 | adapter_plugins/v2.py | 公共torch.ops.memfabric310p.mm_ar → V2算子Tensor实现 → 原C ABI |
| V3 | adapter_plugins/v3.py | 同一公共Torch入口 → V3算子Tensor实现 → 冻结算法会话 |

## 调用与生命周期

`ctx`提供torch、dist、linear、rank、url、stream、device、shapes；可显式指定内部输出槽数。
`run(x,w)`返回完整FP16 ND Y，并在原Stream建立后续可读依赖；Host不需要在算子与下游之间同步。
兼容参数`variant`不再控制V2/V3输出槽，输出选择由算子持有。

每次的验证、输出获取、Storage包装、租约、输入保活、padding/copy/seed/同步都在算子入口内。
持久底层池可以冷态建立，不能把每次必要动作提前到测试代码。
共同实现：[torch_tensor_entry.py](../../common/torch_tensor_entry.py)；
[V2](../../memfabric_mm_ar_v2/src/torch_adapter/tensor_entry.py)；
[V3](../../memfabric_mm_ar_v3/src/torch_adapter/tensor_entry.py)。不导入L1框架，可以独立构造调用。

raw V2/V3由公开入口管理内部输出槽，每次wrap为TorchNPU Storage，Storage别名共同持租约；
内部原始Tensor保持首写地址身份。输入由完成Event保活，失败保留不确定波的所有引用。
A2/A3已有的输出池/输入保活保留，获取/包装在dispatcher调用内；A2内部同步如实计时。
桥扩展来自既有pool_tensor.cpp，可用`common/build_tensor_bridge.py`独立构建并供V2/V3共用，不链接MemFabric。

`after_block()`（兼容`after_sample()`）在已完成组的计时及Profiler之外检查协议状态、回收完成输入。
Host诊断返回`{'host_calls':[每调用相位dict,...]}`；原始计时行保留adapter_call_including_blocking。
`close()`只能在输出及别名释放、完成确认后销毁；失败池不做重试/隐式集合清理。

## 测量与能力

合同L1_TENSOR_IO_V4默认ready+steady，N是算子调用数，B默认2，工作集默认4。
调用层相同，内部分配/同步策略可以不同且须报告。超过实现输出保留上限在设备前拒绝，不缩窗口。
所有Y在组内保留到完成，不忽略中间Y；组间检查地址不重叠/协议。
前后数值/立即下游ADD、冷态连续提交、每阶段末组数值检查为不计时功能证据。
干净采样不逐样本比较，不伪造每调用correct；不把有界steady结果当模型TTFT。

## V2构建身份

`api=v2_selected_v1`固定V2-7/M2048、V2-15/M4096；原selected C ABI及E1/T8设备源不变。
`api=v2_trial_v1`用于匹配T0/D4/C3/F2；私有B/E候选仍冻结private_library/symbol/kind。
这些ABI都由算子公开Tensor入口包装，不能直接用L1裸指针out调用绕开输出成本。
必须冻结新公共源、TorchNPU桥、实际Host/Device/SDK工件；新入口性能须单独测量。

## V3历史恢复

`api=python_session_v1`明确冻结`python_modules`及其文件SHA：

| 家族 | 精确模块角色 |
|---|---|
| raw V3 | memfabric_mm_ar_v3 |
| A2 classic/merged | memfabric_mm_ar_v3、memfabric_mm_ar_v32 |
| A3 START/首READY/7+1/tail | memfabric_mm_ar_v3、memfabric_mm_ar_a3 |

隔离包按路径与SHA加载，相对导入不能拿当前base冒充旧实现。
V3-0/2/4还须核对历史源码/二进制并登记historical_identity_verified；公共Tensor外层取当前主线，算法内层取匹配历史工件。
raw V3 packet1MiB，其余本轮4MiB；classic/merged不能通过改参数偷换VN-M。

新增实验优先复用算子家族，只在versions.json及campaign登记参数；若ABI确变，在算子源码内实现新入口，L1继续薄调用。
未知external保留冻结`create_adapter(ctx,specification)`扩展点，只允许已审计边界；不回退Stock。
Graph/MM shadow/G1专用诊断不混入Eager候选。

[完整测量说明](MEASUREMENT_V4.md)、[矩阵](ROUND_FIXED_M.md)、[OpenCode执行单](OPENCODE_ROUND.md)。
