# A3定稿：固定完整S、独立Y、NZ MM与通信粒度解耦

状态：设计合同；实现与准入结果记录在A3_ADMISSION.md。无新增设备收益结论。

## 目标、边界与格式

310P3单卡双Die/TP2，优化Qwen3.6已确认FP16 RowParallel目标路径；K/N=2048，M<=4096。
单用户Kernel完成NZ权重MM、seed、SDMA Direct ADD及排空，最终完整ND输出；不改Serving默认。
输入ND、权重原生NZ常驻、S/Y为ND。B=NZ不等于达到原生NZ MatMul性能；目前可复用的
64/128行计算tiling是候选，不宣称最佳。无需且不得调用Host aclnnMatmul冒充单Kernel内MM。
保留完整K后FP16舍入与原数值门禁，seed和输出管理计入完整性能。

## 地址、席位与生命周期

每Rank最大Arena=16MiB。Y0=[0,16MiB)，Y1=[16,32MiB)，S=[32,48MiB)。
A3控制源/目的各独占64B，位于48MiB之后，与旧控制及SDK reserved tail不重叠。
保留64MiB对称申请，初始化核查有效几何；地址上限不得接触reserved tail。
S一次Kernel中每行只生产一次，不是环形区；chunk/后续通信不能覆盖其它席位。
Y由Storage及切片持有，最终输出不copy到另一完整缓冲。
后续Kernel复用S前由拥有流顺序及前波最终quiet保证源已读完，不新增Host流同步。
Y槽=(下一epoch-1)%2，不做Host求空闲交集；只在本地租约为空时创建本次持有者。

普通路径是异步返回Tensor；若槽忙、包装失败或输入在途容量耗尽，仍提交ready=0的START，
让双方在MM/ADD之前失败；该会话为终止性错误，禁止重试。可恢复busy不是本接口合同，
旧V32显式同步接口保留但不允许在同一会话混用。不能悄悄覆盖活跃槽。
输入持有到拥有流完成事件，事件query非阻塞回收；元数据、事件对象及引用总量有界。

## 计算与通信解耦

compute_rows=512/1024，对应每核64/128行固定MM；comm_rows=512/1024/2048。
compute_rows和comm_rows独立设置，M分别整除两者；真实尾行由显式padding处理并计入成本。
第q次计算：row=q*compute_rows+core*(compute_rows/8)，八核共同产生一段连续ND行。
每核完成完整MM写S，MTE读取S初始化Y，保证写出可见后发布独占cacheline标志(epoch)。
工作核不等Core0放行；Core0只汇总当前生产单元，无每chunk反复fan-out。
当所有覆盖生产单元已就绪，Core0按comm_rows形成连续READY范围；不足范围不发，尾部必须整除
或显式padding。compute_rows>comm_rows时，一次生产可释放多个连续范围。
只合并完整K、S和Y均已完成、地址连续的范围，不按ND列tile字节总数误合包。

## 设备控制与阻塞收件合同

Host正常路径没有allgather/prepare/流同步；仅本地reserve、Storage创建、参数提交和完成事件。
一次性warm预热所有A3符号、两个Y槽，保留310P首写防护；warm完成前禁止launch。
Core0在同一用户Kernel入口写64B START(epoch,slot,rows,compute,comm,mode,ready)，
双方均signal后wait，严格校验mail和payload；START/READY/DATA/DONE均携带本波epoch。
只有双方有效才放行所有核；无效Trap并由外部守护停止双方，不返回可消费的成功输出。

Core0唯一消费者：每个范围先send READY，收件分发器等待对应peer READY时可以消费旧ADD完成。
peer READY成立后只提交一次完整S范围到对端Y的ADD；不为回收S逐chunk等待ADD完成。
固定S免除了覆盖等待，但public wait仍可能阻塞于对端READY；不能宣称完全异步producer调度。
每个范围的packet数<=64、范围数<=8；默认4MiB。到达队列和请求队列按公开1024/64容量有界，
允许合法跨范围READY/ADD交错，拒绝重复/旧epoch/错误地址、长度、phase及未提交/发布状态。
同一发送FIFO中最多一波READY/DATA流；不添加Gather。

## 完成条件

每核全部MM和seed完成；每个通信范围已提交一次ADD、收到全部期望ADD完成；
出站quiet->双方DONE->最终quiet->全核放行及输出可见性，之后Kernel结束。
ADD读S但写Y，S不被任何ADD修改；Y首次ADD前已有本地种子，之后不再被MM或seed重写。
不在不确定ADD后重试；任何会话/协议/设备失败停止双方。
缓存：结果发布保持必要全局数据可见性；独占标志读取/写入仅做单cacheline维护并渐进退避，
不得每次poll整缓存。不能用volatile取代可见性和跨引擎完成。

## 审核与验收

审核1：地址不重叠、S/Y生命周期、START失败、Direct ADD依赖、seed首次写、帧尾源完成。
审核2：MM/通信交叉粒度、FIFO接收交错、credit/范围上界、单单元和尾部排空、异步输入引用。
实际设备调度器需CPU异步FIFO/延迟读源/乱速测试，Host热路径禁止集合与同步的回归；
之后CANN/Storage编译、G0/双Die独占、IB/ADD/协议/生命周期/融合/尾块门禁，再完整短测。
不追加独立设备归因轮；复用历史V2/Stock，跨批结果不得签同批正式性能GO。
