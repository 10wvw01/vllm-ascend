# A3设计审核、实现与实机准入

## 交付内容

设计见[A3_DESIGN](A3_DESIGN.md)。实现A3Session.run_pool、Host profile3301、
完整S席位、独立Y Storage、设备START、NZ权重64/128行MM、显式seed、Direct ADD和最终排空。
compute_rows=512/1024与comm_rows=512/1024/2048独立。MM仍是已经验证的候选tiling，
不宣称“最佳MM”或已经超越Stock；本轮不增加ND权重转换或Host matmul调用。

正常reserve/run无Host allgather、无流同步；Header是Kernel参数，在同一Kernel内发布START。
输入用最多64个预初始化完成事件非阻塞回收；Storage持有槽，只有最后别名释放才标记可复用。
A3资源不足/包装失败为终止错误：仍提交START ready=0，双方在数据计算前失败。
可恢复busy不是A3 run_pool接口；旧A2接口独立保留，不混用。同流限制必须遵守。

## 两轮设计与源码审核

第一轮（依赖和地址）：逐行席位不重叠，两个Y和一个S位于用户区48MiB内，
控制cacheline紧随其后；单独的IB/标志分配不接触SDK私有环。
START先发送再等待，比较所有参数及双方admit，之后才放行工作核。
seed写入完成早于READY，入站ADD只写Y，S全波不变；最终quiet保护下一调用复用S。

第二轮（交错、边界和生命周期）：MM完成范围到通信范围的覆盖检查；compute大于或小于comm；
最多8范围、每范围64包，整波START+READY+DATA+DONE最多522封小于1024到达槽；
请求环64槽由公开signal背压，不能假设提交即完成。
单消费者校验phase/epoch/group/packet/dst/len/status及重复位；合法旧ADD完成可穿插READY。
显式核查busy旧租约不被释放、包装回调仅一次、事件失败封禁、事件池耗尽仍通知对端。
修正了实现中的源码指纹漏列新适配器、A3关闭profile及测试符号覆盖问题。

## 本地证据与实机阻塞项

实际A3Progress及A3Start在异步FIFO模型执行512场成功数据交错及1024场失败握手，
检查延迟源读取下的逐元素和值、S未变、完整排空、单边admit失败和参数错位双方拒绝。
实际Host经mock ACL/SMEM编译，检查固定槽、warm、ready=0和热路径同步/集合调用计数不增加。
Python适配器测试输入事件回收、槽忙、包装接管前后失败、事件错误、容量上限。
这些均为CPU模型，不模拟真实缓存、NZ性能、驱动或设备资源。

**本地无CANN/BiSheng、TorchNPU与NPU设备，因此新设备编译、资源、START可见性、
数值、生命周期及性能全部NOT_RUN。不签设备GO。**
A2的设备通过仅支持复用其已锁定Storage工厂适配，不能代替A3验证。

## OpenCode一次性服务器执行入口（用户委托后）

继承[OPENCODE_EXECUTION](OPENCODE_EXECUTION.md)的隔离/G0/守护/停止规则。
固定新完整HEAD，新建目录，MemFabric仍锁db38d75e8485d31451302219eb7a1bfeb1dd107c，
只用既有public API，不修改对方仓库、不改变共享安装/CP1/KVER。

1. 运行V3与V2 CPU/源码合同及项目检查。
2. `tools/build.py`私有编译，核对新A3设备函数、START 64B、UB/NZ2ND/双MTE缓冲、
   两段IB和生产标志地址、CacheLine::SINGLE_CACHE_LINE及标量退避在dav-2002的合法性。
3. `tools/build_tensor.py --manifest <build>/manifest.json --out <fresh-tensor-dir>`，
   使用现场锁定的TorchNPU Storage factory路径；冻结全部源码/SDK/helper/SO指纹。
4. 健康且连续独占的G0成立后，detached运行：
   `tools/quick.py start --a3 --manifest <build>/manifest.json --out <fresh-result-dir> --card 38144 --devices 4,5`
   实际卡须符合获准独占窗口。使用status轮询，不前台长阻塞。
5. 六门禁：IB、裸ADD、输出Storage/切片/双槽/单边持有close、协议/单发布单元/回绕、
   融合/下游ND消费、尾块。真实忙槽及包装失败为终止性START错误，负面故障只在CPU回归，
   不在健康设备会话中故意触发Trap再继续。
6. 全部门禁通过才自动六配置M2048/4096×comm512/1024/2048，compute默认1024，
   4MiB包、4预热+30样本、两轮。协议门禁另覆盖compute512及交叉粒度；
   默认六配置不是三种MM tiling，报告要分别标compute_rows/comm_rows。
7. 完整计时含本地reserve/Storage、START、seed、MM、ADD、完成事件record和最终完成；
   不计一次性initialize但单列冷启动。事件record开销不能隐藏，报告Host Wall与配对Rank max。
8. 比较历史A2/V3.1/V2/真实NZ Stock，保留跨批限制，不重跑旧性能臂，
   不追加独立归因/模型/TTFT/Graph/Confirm/重型Profiler。

任一不确定ADD/协议/设备异常停止双方、保留现场，禁止原会话继续或重试ADD；
独占/健康丢失不绕过守护、不杀租户、不自动换卡。实现BUG按新SHA/新目录重编重门禁。
原始双Rank、失败/编译日志、全部指纹、G0/守护及复现命令远程归档，不能只列服务器临时目录。

本地汇总：V3/V2 CPU与源码合同116项PASS；项目治理38项PASS；Ruff、静态项目合同、
增量台账与git diff --check通过。全仓format.sh ci因未安装pre-commit未完成，不能宣称全仓CI通过。
