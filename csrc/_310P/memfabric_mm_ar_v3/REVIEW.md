# V3两轮静态审查与验证边界

## 第一轮：数据与通信协议

审查对象为V3新源码，不继承V2/E4的设备PASS身份。

| 检查 | 结论 |
|---|---|
| 每个输出元素的MM部分结果 | 每Rank计算一次，直接写本地工作区 |
| 每个输出元素的跨Rank求和 | 在其唯一owner执行一次FP16 ADD |
| Reduce发送源和接收目的 | 在同一Rank的不同半区，不重叠 |
| ADD目的初值 | 本地MM/partial搬运完成、Flush、8核IB、双向READY后才允许ADD |
| Gather覆盖对端旧源 | 本端收到Reduce完成邮件，证明该对端源已消费；随后才可COPY回去 |
| Gather发送源复用 | 保持到quiet，全部输出搬运完成且DONE后才可跨调用覆写 |
| 流水数据 | 组间地址独立，只有Reduce与下一组MM交叠；没有循环槽位提前复用 |
| 邮件 | 单生产/消费方，FIFO，逐封校验phase/epoch/group/packet/dst/len |
| 标签 | 40位；24位epoch用尽拒绝继续，不静默回绕 |
| 控制缓存行 | 不与MM工作区、Raw源、MemFabric保留尾部共行 |
| IB | 每个FanOut事件只有一个消费者；CopyDone后再发布CopyConsumed，复用前消费完 |

CPU随机调度规格检查串行/流水的非对称进度、源版本与结果，含512个配置/随机种子组合。
第一轮修正：尾块输出需按3072行分配；限制分包计数避免6位packet溢出；设备Tag函数显式设备限定。

## 第二轮：工程、停止与性能证据

- Host仅消费SMEM公共API，不链接旧vLLM扩展，不访问MemFabric内部环或SQE。
- 同一CANN流；返回前由调用者等待完成；首写符号/新输出地址保留设备同步与双Rank会合。
- V3Session保持Tensor分配存活，独立输出不与输入共享storage；Graph和失败Session拒绝继续。
- 错误不重发ADD，不在异常清理中进入新的collective或设备同步，交由有界外部守护停止。
- E4守护器按文件回收自`5cff1b2d`，增加设备阶段截止时间；STOP只发给自己创建的PGID。
- 独立session后台启动和真实result.json落盘，避免执行工具120秒超时中断现场。
- 物理DeviceID直接选择，不使用可见设备重映射；CANN物理映射API不匹配则建池前停止。
- 构建头/依赖/源码/安装SO/launch JSON和真实装载映射分别记录；启动拒绝过期manifest。
- 默认只测V3、两次短轮，历史对照不会生成Stock/V2新性能数据；CPU Gloo参考只用于正确性。
- 逐Rank逐样本配对max后算分位；缺门禁、缺形状、重复case、错序样本、非有限时间均拒绝报告。
- 所有完整算子成本计入Event窗口：Host下发空档、控制邮件、ADD、Gather、输出搬运；不称纯kernel周期。
- 历史来源/原判定/文件SHA保留；不自动将跨批差值签为正式收益；不触发模型或Confirm长测。

第二轮修正：直接物理号选择替代visible remap；拒绝不完整性能矩阵；显式标记KVER28占位自拷贝差异。

## 本地完成与现场待执行

本地可验证：CPU协议与报告负面回归、真实SMEM公共头配合ACL声明的Host语法、便携C++合同、Python lint。
当前没有BiSheng/CANN/NPU，不能验证设备指令、资源、首写、缓存、FP16舍入或性能。
现场顺序固定：私有CANN编译成功 → G0/IB → ADD数值 → 协议 → 融合正确性 → V3短测。
前一关失败不执行后续阶段。当前版本不是Serving默认，不是生产发布候选。

## 上游同步后复核

上游ba3e7f42与GitHub镜像b9d46f97的整树一致，新增的两份Host文件仅修正SetAccess物理号。
V3公共头/ADD wire/布局/KVER不变；Host语法以同步后的真实SMEM头再次通过。
prepare_sdk.py显式从新构建复制Host动态库及AICPU三件套，设备库不混入Host lib64。
脚本拒绝旧构建目录、源码树不符和SDK位于源码目录内，避免上游wrapper删除其他任务产物。
新增完整性/缺件/冲突回归，V3 CPU29项+既有V2 CPU54项合计83项通过。
