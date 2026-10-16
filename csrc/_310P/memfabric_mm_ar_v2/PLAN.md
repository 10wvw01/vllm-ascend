# memfabric_mm_ar_v2 研发计划（2026-10-16 重编）

> 依据新版 DESIGN.md。旧版多生产者 Request Ring、TAIL proxy、tag/ACK 每 Lane 的实施任务全部作废。V1 不改动。

## T0：第一版实验原型（源码阶段）

参考官方 MatmulAllReduceCustom 的 MM 分核、块完成会合、Commit 后继续计算模式，换用 310P MemFabric 公共 signal/wait/quiet。

**实验边界**：K=N=2048、FP16、NZ 权重、TP=2，M=chunkRows×chunks，chunkRows∈{256,512,1024}、M≤8192；单 Kernel，8 核 M-split，Core0 兼职通信；C0/C1 对称静态空间无 chunk 复用；模式 SERIAL/OVERLAP 切换。每 wave 一次 quiet 和 DONE 对称握手；无 torch 模型路由。

- 已新增实验性源文件：mf310p_v2_trial_device.asc、mf310p_v2_trial_adapter.cpp、mf310p_v2_trial_adapter_api.h、cmake/memfabric_310p_v2.cmake。
- 仅环境变量 VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1 时构建，必须同时启用 V1 MemFabric 构建。
- C API 用于独立双 Rank 试验。**不表示已经编译或实机验证。**

## G0：官方 IBSet/IBWait 能力验证（先于完整融合）

1. 在 dav-2002 bisheng 编译实验设备库，确认 kernel_operator_block_sync_intf.h 与 KERNEL_TYPE_AICORE 支持 IBSet/IBWait。
2. 双核单 Kernel 执行 eventID 0/1 往返 1000、10000 次；对应 GM 工作区初始化清零；检查完成计数与两次调用复用。
3. 检查 8 核 7→1 扇入、Core0 1→7 使用七个不同 eventID 的同步正确性；量化事件时延。
4. 若 G0 失败，暂停 G1，先查官方 ABI 与核模式；不得转为直写 MemFabric 环。

**验收**：编译 0 error、无死锁、全部同步检查通过。

## G1：最小串行正确性

- 单次 M=256/512/1024、CHUNK 同 M，两个 Rank 各自 MM 写入对应对称 C 段；
- Core0 signal 一封，双方 wait；8 核 Add 后本波 DONE；检查 bit-exact 和所有邮件字段；
- 重复调用 1000 次，确认静态 C 段复用、IB event 复用和缓存可见性；
- 单 Kernel 与 V1 的 stock MatMul+HCCL 单波对照，记录暴露 wait。

## G2：大 M 交叠

- M=2048/4096/8192，chunkRows=256/512/1024；比较 SERIAL 与 OVERLAP；
- 用 profiler 证实 MM(k+1) 与 SDMA(k) 的重叠；
- 分解 MM、IB、signal、wait、Add、quiet、DONE；确认 AICore 资源及 UB 足够；
- 不允许在一个 Rank 的 Core0 等待另一个 Rank 尚不能 signal 的数据。

## G3：工程补全和产品化（当前未实现）

1. 增加任意 M 尾块，M>8192 多 wave；
2. 官方 MatmulBlock 的 M/N 二维分核小 M 优化；MM 1/2/4/8 核与 Add 核数分离；
3. 增加严格的内核资源检查、错误恢复和 graph replay 状态治理；
4. 接入 PyTorch/vLLM 的独立显式 feature flag，避免覆盖 V1；
5. 按 M=1/16/64/255/256/257/512/1024/2048/4096/8192 与真实 Qwen3.6-35B-A3B 的 40 层做 A/B，Graph ×1000 + 10min+ eager/graph 长稳；
6. 与 V1、stock 比较 ITL/TTFT/tok/s，通过验收后才讨论默认启用。

> 当前 PLAN 的状态是“源码原型，未编译”。发布门槛不允许用静态审视代替 310P3 实机。
