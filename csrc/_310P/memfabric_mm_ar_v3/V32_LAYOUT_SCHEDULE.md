# V3.2 内存布局与融合排布推导（2026-10-29）

## 结论与范围

这是可编码设计，不是已实现或已证明最优的Kernel。只重构MM+AR内部及必要的SMEM输出适配，
不改Runner/DFlash/模型，保留TP2、FP16、NZ权重、完整ND输出、单用户Kernel和chunk512/1024/2048。
不追加独立设备归因轮，直接实现后做完整正确性与性能验证，旧V2/Stock继续复用历史数据。

关键改变：单份具有独立输出生命周期的池内Y，同时承担本地MM结果、ADD累加和Gather目的地。
去掉最终全量copy；不额外分配半份发送源S。各核独立发布整行带完成，取消逐chunk全核会合。
上一轮提出的Y+S不是主方案：TP2的ADD完成→Gather依赖已足以保护源，独立S只增加半份存储。

## 参考与硬约束

1. FLUX设计：<https://github.com/bytedance/flux/blob/main/docs/design.md>
   GEMM输出tile接入ReduceScatter、计算顺序按通信依赖重排；不照搬GPU warp/TMA能力。
2. NVIDIA NVSHMEM示例：<https://docs.nvidia.com/nvshmem/api/latest/examples.html#gemm-allreduce-fused-kernel-example>
   对称堆输出、每个GEMM tile触发collective、显式完成和同步驻留要求；NVLS不是310P可用能力。
3. Ascend GetTensorC：<https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetTensorC.html>
   Atlas推理AI Core不支持异步模式；NZ/ND支持必须区分接口及搬运路径。
4. Ascend WaitIterateAll：<https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/WaitIterateAll.html>
   Atlas推理AI Core不支持。不能以异步整块Matmul为设计前提。
5. 稳定产品文档交叉核查：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0639.html>
   同样明确推理AI Core不支持GetTensorC异步。以上master预览并非服务器9.1头文件的替代证明。

本地锁定的SMEM公开Host头有smem_shm_create返回GVA及对称大小查询，没有可直接用于
普通NPU Tensor的外部内存注册接口。不能把普通torch.empty指针直接作为远端ADD目的地。
公开设备接口目前只有阻塞signal/wait/quiet；不得在算子中读取私有环来补非阻塞能力。

## 确定采用的布局：每个输出槽只有Y

记N=K=2048，元素2字节，M<=4096，S_total=M*4096字节。
每Rank每个活跃输出槽分配Y[S_total]，正常行主序[M,2048]，不按rank打包。
MM所有部分结果直接写Y，没有额外source区、seed copy或清零。

令C为chunk，g=floor(row/C)，u=row%C，owner=0当u<C/2，否则owner=1。
Y(row)=Y_base+row*4096；两Rank通过统一几何计算对端Y。

| Rank | MM写入 | ADD发送源 | ADD接收目标 | Gather发送源 | Gather接收目标 |
|---|---|---|---|---|---|
| 0 | Y全体本地部分结果 | Y每chunk后半 | Y前半 | 最终Y前半 | Y后半 |
| 1 | Y全体本地部分结果 | Y每chunk前半 | Y后半 | 最终Y后半 | Y前半 |

源只要求在传输消费前不变，不要求整波不变。owner与nonowner半区不重叠。
区域及标志独立对齐，用户meta不与SMEM reserved tail重叠。
不预先zero Y：全部位置先由MM覆盖；旧代次不能充当种子。
不向完整K归约及FP16舍入之前的片段触发ADD。
不把MM原子写与SDMA ADD混合使用来省READY，跨引擎并发原子语义没有证据支持。

## 计算与传输粒度

基本单元是完整N列的连续行带，不是baseN=256的列tile。
原因：ND列tile跨行有stride；公共SDMA只接受连续src/dst/len，按列tile发会退化为大量逐行请求。
首版沿用已实机验证的ND Matmul落盘路径，不假设GetTensorC能在310P直接提供任意ND异步epilogue。

| chunk C | 每核行数 | 每次完成行带h | 每核子MM数 | 行带字节 | 每chunk行带数 |
|---|---|---|---|---|---|
| 512 | 64 | 64 | 1 | 256KiB | 8 |
| 1024 | 128 | 128 | 1 | 512KiB | 8 |
| 2048 | 256 | 128 | 2 | 512KiB | 16 |

保留baseM64/128、baseN256、baseK64，2048不得扩大baseM到256。
每核计算地址：row=g*C+core*(C/8)+j*h。core0~3是owner0行，core4~7是owner1行。
每核每完成一个h行子MM、全部输出写完并满足缓存可见性后，独立发布该行带完成。
core1~7无需等待core0，不在每个chunk上执行fan-in/fan-out。

首次实现仍用每次h行同步MM直接写ND目标；Core0在其子MM边界pump消息和就绪标志。
这保证可以编码，不宣称同一Core0的标量通信与同步MM真正同时运行。
已提交SDMA可与其他7核后续MM并行；Core0算完自身行带后负责排空其他核的通信。
进一步进入Iterate/GetTensorC微epilogue时，必须提供NZ2ND、UB及完整K/FP16舍入证明，
且不得引入额外完整GM中转；不作为首版成败的隐藏依赖。

## 片段状态机与源安全证明

对连续行带i，owner为o，另一Rank为n：

1. owner MM完整写Y_o[i] → 写出完成及DCCI → TARGET_READY(epoch,i)。
2. nonowner MM完整写Y_n[i] → 写出完成及DCCI → SOURCE_READY(epoch,i)。
3. 本地SOURCE_READY与收到的TARGET_READY成立，才发一次ADD：Y_n[i] → Y_o[i]。
4. owner收到该ADD的有效完成邮件后，发Gather：Y_o[i] → Y_n[i]。
5. nonowner收到Gather邮件后，该行带在两端均为最终结果。

关键先后链为：

`MM_n(i) → ADD读取Y_n(i)结束 → owner收到ADD完成 → Gather覆盖Y_n(i)`。

因此Gather不会提前修改发送源，独立S没有消除额外依赖的价值。
owner的种子写也严格早于ADD；owner在ADD后不再做本行带MM或修改最终值。
两个方向处理不同owner区域，不构成双端对同一值原地互加的错误算法。

TARGET_READY可以先于SOURCE_READY，亦可相反；producer不等待远端即可继续计算。
依赖是MM→READY→ADD→Gather的单向链，没有“先等待再生成对方所需数据”的环。
最终输出不能在本行带Gather之前被下游读取；普通Tensor接口在完整Kernel结束后返回可消费结果。
FP16参考为round_fp16(round_fp16(local_MM)+round_fp16(peer_MM))，保持完整K和既有数值门禁。

邮件校验epoch/slot_generation/phase/range/dst/len/status；bitmap拒绝重复ADD或重复完成。
不对未确认是否执行的ADD重试，错误双Rank停止并保留现场。

## 主方案A：现有阻塞API，固定有序流水

MemFabric由其他团队负责；主方案不新增API、不改库、不假设提供异步收件。
单Y布局、输出Storage生命周期、各核独立producer和完整数值合同保持不变。
core1~7按自己的行带顺序继续后续MM，不参加每chunk反复的全核会合。
Core0每到chunk g，先完成本核该chunk的MM，并确认该chunk本地各行带已发布就绪，
保证接下来阻塞不会阻止本chunk任何必要本地数据的生成。工作核无需等Core0放行。

两端严格使用同一发送顺序；每个阶段按固定地址顺序发完该组所有包，包数由几何唯一决定。
以下R表示READY，D表示ADD Reduce，G表示COPY Gather：

```text
for g = 0 .. chunks-1:
    完成本地chunk g的必要MM及写出可见性
    send R(g)
    if g >= 2: receive G(g-2)
    if g >= 1:
        receive D(g-1)
        send G(g-1)
    receive R(g)
    send D(g)

if chunks >= 2: receive G(chunks-2)
receive D(chunks-1)
send G(chunks-1)
receive G(chunks-1)
final quiet + DONE + 输出完成
```

其中receive是调用现有wait_at接收预期包数，每封严格核对epoch/phase/chunk/packet/dst/len。
chunks=1也有完整排空路径，不构造不存在的负序号。

发送FIFO为`R0,D0,R1,G0,D1,R2,G1,D2,...,Glast`。
因此g>=2时未收邮件的顺序恰好是`G(g-2),D(g-1),R(g)`，与上述阻塞收件顺序一致。
双方均先发本轮R；等待D(g-1)前，D(g-1)已经由双方上一轮提交；
等待G(g-2)前，G(g-2)已在上一轮D(g-2)收齐后提交。因此没有互相等未发任务的循环。

G(g-1)在提交D(g)之前入队，可与后续MM及队列传输交叠。
不声称Core0阻塞MM期间能处理邮件，也不声称单SDMA FIFO的两项数据任务能同时执行。
这是协议顺序约束换取兼容性，允许head-of-line等待，但不依赖额外MemFabric能力。
相较异步方案，主方案仍保留Core0对本chunk本地就绪的汇总等待；不能把它称为逐tile完全自驱。
主要收益来源为单Y去最终copy、减少反复全核会合和确定的Gather提前提交。

本地验证了1/2/4/8个chunk×256种随机rank/网络进度，共1024场FIFO协议调度，
全部按预期收件且完整排空。这不是CANN/NPU正确性或吞吐证明。

## 备选方案B：团队提供公共try_wait后，逐片段自驱

仅在上游明确提供并完成接口合同审核后启用，不作为A的交付依赖。
Core0在子MM边界pump已到消息与本地行带标志，owner行带可独立发TARGET_READY，
本地SOURCE_READY及远端TARGET_READY成立即可发ADD，ADD完成即可发Gather。
仍保留唯一mail消费者、合法交错分发、就绪范围合包和无饿死调度。

期望的公共try_wait：EMPTY立即返回且不推进head；有邮件时消费一封；保留错误/溢出语义。
不读取私有环、不缩短阻塞wait预算伪造轮询、不用TIMEOUT替代EMPTY；不另开第二个收件核。
A/B在会话启动前固定协议模式，双方交换并验证一致；一波中不切换顺序，不能混用两套消息流。
B提供更细推进的潜力，也增加小消息和状态机开销，不能默认比A快。

## 备选B的范围合包；A保持固定分包

库内每封signal仍有数据/到达戳/完成戳SQE及游标维护，小片段越多越可能损失收益。
行带是就绪和正确性单位，传输范围可合并：仅合并两端连续Y地址、同owner且全已就绪的片段。
不为等满4MiB阻塞已就绪的早片段；在当前pump批次内尽量合并，单包上限4MiB。
范围标签+起始行带索引+固定片段长度描述整包，按同一几何推导覆盖bitmap；禁止合并跨owner孔洞。
chunk2048的第一个128行子调用先完成时，允许存在暂不连续的已就绪带，不能误合并覆盖未完成行。
控制READY同样允许描述连续范围；不逐行、逐列或逐元素发控制邮件。

## 输出生命周期是算法合同的一部分

Y必须来自SMEM池，不依赖普通Tensor注册。C++扩展将本Rank有效Y映射包装为ND NPU Tensor，
指定实际device/stride和自定义Storage持有者；保持池和槽在整个输出生命周期内存活。
torch.from_blob或等价Storage包装的NPU行为是必须实现并验证的准入项，尚无本轮实机证明。

Y仅在最终quiet+peer DONE、本波输出完成、下游读取完成且Tensor持有释放后复用。
保留多个输出并连续调用时必须各有独立槽；切片/别名同样持有槽；close不得销毁仍被持有的池。
槽耗尽在进入设备波前明确背压/报资源不足，不允许按epoch取模静默覆盖。
采用有界多槽池和generation，双方选择同一通信槽且各自可复用前才能启动新波。
跨流支持不默许：首版限制拥有流，其他流使用须显式记录consumer event后才释放。

新输出返回式接口作为主实现。外部out兼容接口若必须copy，就把它作为完整物化路径单独报告，
不能把省copy的数值用于声称既有外部out接口也获得相同收益。
模型下游消费仍得到完整ND Tensor，不擅自把AllReduce改为只输出ReduceScatter。

## 能证明什么、不能证明什么

设每Rank结果S_total字节：跨Die数据仍为S_total/2 ADD+S_total/2 Gather，合计S_total，不声称减半。
忽略控制、缓存和事务放大，当前路径的结果数据写入包括MM S_total、入站传输S_total、copy S_total；
新路径为MM S_total、入站传输S_total，省掉完整copy的一读一写及相应MTE/UB/会合。
MM每Rank写Y一份，逻辑结果存储每活跃槽为一份；多输出槽、控制和库工作区另计。
这与旧V3.1的原地Reduce/Gather安全关系一致；创新在输出分配、生命周期及producer调度，
不是重新发明两阶段AllReduce，也不把Y+S布局本身当性能来源。

本地纯CPU所有权模型已执行600种随机事件交错，覆盖两M×三chunk、任意MM完成顺序与FIFO传输；
验证每元素和值、owner种子先于ADD、延迟读取源时仍未被覆写、Gather及全部片段排空。
该模型不模拟真实CANN缓存、Core0性能、Tensor包装或队列饥饿，不等于设备准入。

仍不能数学证明快于V2：控制消息、Core0 pump开销及MM受扰动可能抵消copy节省。
但这是围绕inlineReduce和最终结果落点的完整数据流重构，不是继续堆叠V2后处理流水。
A/B新Kernel及输出适配均未实现；B的公共接口也未获上游承诺，性能UNKNOWN。

## 实施边界与验收

两个我方仓库：vllm-ascend实现单Y输出槽布局、独立producer、Core0固定顺序调度、Tensor槽与门禁；
A沿用现有MemFabric锁定版本；B须由负责团队提供公共非阻塞收件并验收，不擅自修改其私有协议。
保持旧接口可用，保存V3.1原始证据，新路径显式选择，不启Serving默认。
实现完成后直接进行输出持有/别名/连续波、各种chunk、数值与协议门禁及完整性能短测，
不加独立设备归因轮，不重跑V2/Stock。继续以完整ND可消费输出为终点，报告pool-return和out-copy差别。
通过实际服务器CANN编译、NPU Storage有效性、全部数值/生命周期门禁后才签设备准入。

## 2026-10-29方案A编码更新

A已编码，见[V32_ADMISSION](V32_ADMISSION.md)。实现为按核完整chunk发布独占缓存行，
Core0逐chunk汇总，其他核持续向前；B的逐行带自驱未实现。
两输出槽复用前执行原流同步与双Rank空闲交集allgather，该完整接口成本必须进入时延。
Storage包装采用通用from_blob并补TorchNPU StorageDescHelper描述；仅普通from_blob不算可用。
新设备编译/Storage/数值/性能仍NOT_RUN，上文“尚未编码”仅描述推导时点。
