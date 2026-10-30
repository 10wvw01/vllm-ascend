# 固定M统一适配器与批次复核（2026-10-30）

下述首两轮是L1_TENSOR_IO_V2历史审核；当前计时合同V3的修订与验证见末节，旧chain结论不沿用。

基底：dev/310p-memfabric-mm-ar @ 9b2e00875a35e8dc09af25a7d5a6128579414297。
本轮实现Stock/V1/V2/V3薄适配器，准备Stock+所选V2+V3八种迭代的完整固定M矩阵。
V1不进入本轮测量；无CANN/NPU新执行，无算子Kernel/Host/协议改动。

## 第一轮：调用与测量合同

- Stock严格一次NZ MM+HCCL SUM；V1一次Tensor op，q参数明确传递。
- V2绑定基础/私有/所选API；仅B1绑定AR-only符号，诊断OFF不阻止基础路径。
- V2 raw输出在建池前分配；V3 raw输出和探针存储同样提前，pool-return不额外分配闲置输出。
- 原始V3使用1MiB packet，其他本轮V3使用4MiB；serial/pipeline、classic/merged、A3席位均显式登记。
- V3-0/2/4从冻结模块加载，独立包隔离相对import，不可落到全局当前V3Session。
- 适配器只有run/after_sample/close，不增设版本私有计时和统计，不隐藏必需同步、包装、seed或拷贝。
- 原ready/chain、正确性和逐样本双Rank max后均值公式保持，设备阶段耗时缺失仍UNKNOWN。

## 第二轮：矩阵、隔离与失败

- 完整43组/170会话；active30组/118会话；可选V2历史64组/254会话；host/profiler完整三段510会话。
- 每配置独立ABBA，所选V2与历史私有V2单独runtime_key；没有跨配置合并Stock或混淆参数。
- W20/N100/R1，两形状ready+chain，默认每臂每形状每场景200配对样本。
- 所有工件/历史身份/准入在设备前统一预检，任一缺失整批BLOCKED，不静默删实验。
- 框架身份覆盖每个插件模块；bundle固定每份plan/manifest；启动后不允许改API/参数/库。
- SDK v11/v28按进程隔离，Stock不加载融合依赖；沿用物理卡守护和SDK/Device实际装载检查。
- 首个错误STOP，不重试通信波、不进入下一组；汇总失败也写明确STOP，保留原始状态与未运行清单。
- 每组输出源码HEAD、配置、对应Stock、均值/分位数/std、漂移和原判定，诊断不冒充正式收益。

## CPU验证

```text
python -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test \
  csrc/_310P/memfabric_mm_ar_v3/test tools/310p_mm_ar csrc/_310P/tools/L1_test
475 passed, 36 subtests passed
```

本轮新增23项薄适配/历史模块隔离/完整矩阵/冻结/STOP/汇总回归。
CMake测试通过当前Python环境寻找cmake，去掉对临时虚拟环境绝对路径的依赖。
35份版本文档重新生成，V3历史分支现在明确说明内置冻结session适配；CLI可从任意目录生成计划。
未运行格式检查。所有PASS仅指CPU假后端、结构与元数据合同，不代表SDK/CANN/NPU准入。

## 下游仅做不计时验证：L1_TENSOR_IO_V3复核

基底主线df6256ff。用户明确下游消费无需计时。

第一轮核对边界：timed只接受算子调用、Stream、两个Event与时钟，无consumer参数；
热态W/N及路由/性能Profiler内无下游ADD，Wall/Event只覆盖一次完整MM+AR及最终等待。
独立validate_downstream执行op→同Stream ADD→最后一次同步，框架不读时钟或记录Event。
每M/双地址在采样前后均验证数值和协议；输出租约保留至消费完成，不将必需Host准备移出run。

第二轮核对证据：计划只准ready，合同V3与campaign schema V2拒绝旧chain计划/结果。
聚合要求双Rank全部M、双地址、before/after下游PASS且timed=false；缺失、重复、错算、
旧consumer_event_us等字段拒绝。汇总只包含wall_us/event_us与PASS_UNTIMED，不生成消费耗时/速比。
完整矩阵仍43组/170会话，形状/ABBA/样本均值公式及shared/exclusive约定不变。

CPU回归482项及36子测试通过，其中L1框架59项；新增7项覆盖旧合同拒绝、
下游无时钟/事件且依赖顺序不变、门禁证据遗漏/重复/伪计时及旧指标混入。
35份VN-M文档与默认计划重新生成。无新CANN/NPU执行，不查格式，不签生产GO。

## L1_TENSOR_IO_V4：调用层对齐与专业测量复核

基底0bca98ff45a2afbf415ae3f800de3071f58503cb。用户要求专业优化，并明确以Stock为基准，
V2/V3不对等部分必须在融合算子内修正。此前V3合同只删除下游计时，不解决调用层不一致；
本节及MEASUREMENT_V4.md覆盖前述测量做法，历史记录不重标。

第一轮静审核对调用/生命周期：V2/V3输出获取、Storage包装、租约和输入保活迁入算子src，
公开入口是真实torch.ops.memfabric310p.mm_ar，L1插件只有薄加载/调用。
raw底层Tensor保持持久首写身份，返回Storage别名持租约；A2/A3保留既有池语义/同步。
输入事件完成才释放引用，后端/事件失败poison；上下文强注册至显式close，避免调用方丢弃对象后
在途内存被GC；无隐式集合析构。原C ABI、C++/ASC Kernel和冻结Session算法未改。
桥扩展复用已实机验证的pool_tensor.cpp，但新V2/raw用途本轮设备仍NOT_RUN。

第二轮静审核对测量/证据：默认ready+有界steady，N恒为调用数，所有组输出保留至完成。
组内无CPU barrier/数值/状态检查，组尾审计和阶段末数值如实声明；不存在每样本correct伪证据。
A2/A3超2容量设备前拒绝，不偷偷扩大池/缩窗。尾块按调用数加权，steady分位数标块平均。
空计时封套只报告不减除；预热收敛不冒充频率证书。ABBA/BAAB轮间交替，bootstrap重采独立轮次。
shared观察、R1筛选、诊断不签收益；外部工厂调用层未审计不得签公平收益。
Profiler只包执行组，Host细分完整保留（包括同组多个phase_log，不再只pop最后一次）。

验证：新增真实CPU Torch dispatcher捕获证明mm_ar包住后端计算；
假后端验证wrap处于dispatch内、两个输出/别名、输入事件保活、owner丢弃安全、满池不覆盖、
组尾等待、尾块、精确调用数、容量/旧字段/伪校准拒绝及独立轮次结论。
最终完整CPU回归记录见下方及V2_CODE_PACKAGE.json的V4修订；无CANN/NPU新执行，不查格式。

最终验证：497项CPU测试及36子测试通过，其中L1为74项，含真实torch 2.10.0+cpu dispatcher；项目元数据门禁PASS。CANN/NPU、模型调用点与新入口性能仍NOT_RUN。
