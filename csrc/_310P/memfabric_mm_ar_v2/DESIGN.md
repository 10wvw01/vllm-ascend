# memfabric_mm_ar_v2 设计方案：单 Kernel 双引擎融合

> **设计版本：2026-10-16 双引擎方案；T0 已在 2026-10-17 完成 G0/G1/G2 实机验证；T1 小 M N-split 源码待验证。**  
> **硬件限定：310P3 / dav-2002，TP=2，CANN 9.1，Qwen3.6-35B-A3B-w8a8。**  
> V1 `csrc/_310P/memfabric_mm_ar/` 冻结；V2 独立实现，按需复用 V1 已验证的 MemFabric 建池、几何信息和 PyTorch 适配能力。本文**替换**旧版「8 Lane 对称通信、直接改写请求环」方案。**PLAN.md 已按双引擎原型重编；T0 尚未通过实机验收。**

## 1. 设计目标与已确认合同

本期目标是把 Qwen3.6 中未量化的 FP16 `self_attn.o_proj`（10 层）与 GDN `linear_attn.out_proj`（30 层）的 **MatMul + TP AllReduce** 融合到一次 AICore Kernel launch 中。

两 Rank 沿 K 维分片，而非沿 M 维简单拼接：

```text
Rank 0: X0[M,2048] × W0[2048,2048] → C0[M,2048]
Rank 1: X1[M,2048] × W1[2048,2048] → C1[M,2048]
Both:   Y[M,2048] = C0 + C1
```

- 计算与通信数据均为 FP16；权重采用已加载的 NZ 格式，输出为 ND。
- **计算引擎**：AICore 执行局部 MM 和本地向量 Add。
- **AllReduce 通信引擎**：MemFabric 对称内存 + 异步 SDMA 负责双向交换部分结果；到达之后计算引擎 Add。
- **单 Kernel、两条执行路线**：低开销串行与大 M 分块交叠。不得为了实现两条路线退回多 Kernel 编排。
- 只为真实数据依赖付费：`MM(chunk)完成 → signal(chunk)`，`wait(chunk)完成 → Add(chunk)`，以及整个静态工作区复用前的完成条件。
- 核数、分核方式、分块大小及两条路线切换点都以 310P3 实测选型，不预设「8 核 MM + 8 核 Add」必然最优。

目标基线：**stock FP16 NZ MatMul + HCCL AllReduce**、V1。必须分别考核 Decode 小 M 延迟、Prefill 大 M 时延与真实模型的 Graph/Eager 端到端稳定性。

## 2. 运行时架构：两个引擎，官方原语优先

```text
┌─────────────── 单个 AICore Kernel ───────────────┐
│                                                   │
│ 计算引擎：MM(chunk) ───────────────→ Add(chunk)    │
│                │                     ↑            │
│                │ MM 已完成            │ 对端到达    │
│                ↓                     │            │
│ 通信调度：Core0 signal(chunk) ──→ wait(chunk)     │
│                │                     ↑            │
└────────────────┼─────────────────────┼────────────┘
                 └── MemFabric SDMA ────┘
                     对称内存双向交换
```

**MemFabric 数据面仅调用公共 API：**

| API | 功能及调用时机 | 不应混淆的保证 |
|---|---|---|
| `smem_shm_sdma_signal(gva,src,dst,len,imm)` | 唯一通信提交者 Core0 为一个连续 chunk 提交异步数据搬运 | 返回并不等于搬运完成，也不意味着对端 Add 已完成 |
| `smem_shm_sdma_wait(gva)` | Core0 消费按序到达的邮件；检验 `status/dst/len/imm` 后允许 Add | 只证明**本端收到了对端数据**，不是对端已消费本端数据 |
| `smem_shm_sdma_quiet(gva)` | 确认**本端提交的发送**已经落地，发送源可复写 | **不能**单独证明对端 Add 完毕或对端 recv 可覆写 |

热点调用可缓存 `smem_shm_sdma_reserved(gva)`，使用公共 `*_at` 变体。**禁止直接操作 MemFabric 请求环、TAIL/HEAD、到达环、SQE 和内部布局；禁止自定义多生产者协议。** 只有 Core0 执行 `signal/wait/quiet`，但 Core0 仍正常参与 MM。

**核间同步优先 Ascend C 官方 `IBSet/IBWait`**，不用自写 GM ready-cell/影子计数器作为默认方案。两者在官方产品支持表包含 Atlas 推理系列 AI Core，但 dav-2002、当前混合/单体编译模式与多次复用的具体行为需编译及实机 probe；不是 A2 的硬件 `CrossCoreSetFlag/WaitFlag`。不能混淆核间完成通知与 GM↔SDMA 的缓存可见性。

参考：
- [MemFabric 310P 公共原语](https://github.com/10wvw01/memfabric_hybrid/blob/wgm-dev-310p/src/smem/include/device/smem_shm_aicore_base_sdma.h)
- [IBSet 官方文档](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0202.html)
- [IBWait 官方文档](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0203.html)

## 3. 对称内存即计算结果汇合区

每张卡各有一个对称内存段，**同样的逻辑偏移，不同的本地物理存储**：

```text
                 Rank 0                              Rank 1
       ┌────────────────────────┐         ┌────────────────────────┐
offset │ C0: 本卡 MM 直接写入  │ --SDMA-->│ C0: 接收 Rank0 的结果 │
       ├────────────────────────┤         ├────────────────────────┤
       │ C1: 接收 Rank1 的结果 │ <--SDMA--│ C1: 本卡 MM 直接写入  │
       └────────────────────────┘         └────────────────────────┘
                   │                                  │
             Add(C0,C1) → Y0                    Add(C0,C1) → Y1
```

- MM 直接写本 Rank 所属的 C 区；SDMA 直接写对端对应的 C 区；Add 从本地两个区读出后写到输出 `Y`。没有额外本地 staging/pack/unpack。
- 对称内存**不等于**两卡自动共享缓存；仍须跨卡 SDMA 传输一次，且通信读写要遵守缓存可见性。
- chunk 沿 **M 行**切分；同一波每块占自己独立、连续的 `[rows,2048]` ND 位置。**单波内不循环复用 send/recv chunk 槽**，从根源取消逐块 credit/ACK。
- Core0 收到邮件不需要按 Lane 转发数据，只发布必要的本核「接收完成」通知。
- MemFabric 自有段尾保留区由库管理，不分配给算子。官方 IB 同步 GM/UB workspace、错误码、波级控制邮件和输出 `Y` 与数据区分配明确隔离。

**初始空间预算（每 Rank）：** `windowM=8192`，`C0=8192×2048×2=32 MiB`，`C1=32 MiB`，数据区总计 **64 MiB**；另需同步工作区、控制区、对齐和池管理保留区。M 超过窗口时按 wave 顺序复用整片窗口；窗口大小不是性能常量，按内存预算配置。

**缓存边界：** MM 各核在通知完成前，确保自己写入的 C 行对 SDMA 可读（V1 已验证按行 cache clean）。`wait` 成功后 Add 读取接收区前，必须保证 SDMA→AICore 可见性，重复使用的地址还要防止读到旧缓存。优先采用官方缓存维护 API，并以 dav-2002 probe 确定最小操作范围；`IBSet/IBWait` **不能自动代替**跨引擎 cache 维护。

## 4. Chunk 取值：继承 V1 已证实的 4 MiB 单邮件能力

固定 `K=N=2048, FP16`，单块传输大小为 `chunkRows × 2048 × 2`：

| chunkRows | 单次 signal 数据量 | 证据与定位 |
|---|---:|---|
| 256 | 1 MiB | V1 `q=1` 实机验证 |
| 512 | 2 MiB | V1 `q=2` 实机验证、默认基线 |
| **1024** | **4 MiB** | **V1 `q=4` 实机正确性与性能均已验证** |

V1 的 `mf310p_device_launch_direct_producer_async` 支持 `BATCH_M=1024`，调用公共 `smem_shm_sdma_signal(..., batchBytes, ...)`；测试报告中 q=4（1024）通过多 M 正确性，并在 M=4096 获得明显的算子级收益。

**不要将 MemFabric 示例里的「8×128 KiB、单次 ≤1 MiB」测试规模当成接口硬性上限。** 在当前 V1 运行栈中，1024 行（4 MiB）已经由实机证明可工作；超过 4 MiB 的候选必须再做传输正确性和性能验证。

- 串行和交叠 **不与固定 CHUNK 绑定**：初期 A/B 比较 256/512/1024。
- 小 M 允许按有效行数计算/传输；必要的 tile 对齐或 staged 输入只在确有硬件要求时采用，禁止沿用「M<256 一律补到256」作为设计前提。
- 大 M 优先实验 `512/1024`；是否用「短首块 + 长后续块」由 `T_MM`、`T_SDMA` 和每块同步成本决定。
- `M=2048` 是旧 V2 的路线分界**候选值**，不是硬约束。最终依据端到端实测调整。

## 5. 计算引擎：MM 与 Add 核数解耦

借鉴官方 [MatmulAllReduceCustom](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/23_matmul_all_reduce_custom/MatmulAllReduceCustom) 的 `MatmulBlock`：按 `baseM/baseN` 将 C 分成基本块，再向多个核分配任务；小 M 不应为采用 8 核 M-split 而进行大量无效补零。不能直接搬用 A2 的 HCCL API 或硬件 `CrossCoreSetFlag/WaitFlag`。

| 数据规模 | MM 初始候选 | Add 初始候选 | 关注点 |
|---|---|---|---|
| M=1～64 | 1/2/4/8 核，N-split/GEMV 优先 | 1 核优先 | Decode 延迟；避免无效计算和接收广播 |
| M=65～255 | 2/4/8 核，N-split 与精确 M tiling | 1/2 核 | 权重切片、按 ND 连续地址写入成本 |
| M=256～2048 | 4/8 核，M/N 切块对照 | 1/2/4/8 核 | 同步与计算负载均衡 |
| M>2048 | 8 核 M-split 为基线，补测 4 核 | 4/8 核并测单核 | MM/SDMA/Add overlap 与带宽 |

这张表是**实验矩阵**，不是最终选择。

- 核间协作采用「工作核完成 MM → Core0 汇总 → 一封 signal」；不需要所有核互等。
- Add 只有在 `wait` 完成后消费对应数据。小 M 优先 Core0 独自 Add（省去接收完成广播）；大 M 再用多核 Add。
- 若 N-split，各核应把结果按最终 ND 地址写入 C 区；需要验证写回步长、cache line 所有权与 NZ 权重切片成本。通信必须仍能读到一个连续 chunk。
- MM 和 Add 共享 Kernel 内资源，需要分别确认 `Matmul` 与向量 TQue/UB 资源可同时满足编译与执行约束；各核在进入下一块前应确保其上一个 Add 的依赖已满足。

## 6. 单 Kernel 两条路线及最小同步

### 6.1 串行：低 M / 少块

```text
计算核：  MM(k) ── 发送前本地完成通知 ───── Add(k) ── ...
Core0：           汇总 → signal(k) → wait(k) → 放行 Add
SDMA：                        对端交换数据
```

单 chunk 不追求没有用处的交叠，优先减少 MM 核数、cache 维护、同步与暴露 wait；多个 chunk 仍按串行次序执行。**只启动一个 Kernel**。

### 6.2 交叠：足够多的 chunk

```text
计算核          Core0 / MemFabric
MM(0)         → signal(0) ───────── SDMA(0)
MM(1)         → signal(1) ───────── SDMA(1)
Add(0)        ← wait(0)
MM(2)         → signal(2) ───────── SDMA(2)
Add(1)        ← wait(1)
 ...
Add(last)     ← wait(last)
```

这只是偏序约束而非严格锁步时间表：`MM(k+1)` 优先遮蔽 `SDMA(k)`，但 `wait(k)` 必须发生于 `Add(k)` 前；Core0 若先等待数据导致后续 `signal` 无法推进，则需改变其局部调度，不能形成两卡循环互等。

**每 chunk 仅两类本地通知：**

1. **7→1 MM 完成扇入**：非 Core0 核完成本块工作并发布 `IBSet`，Core0 对相应源核 `IBWait` 后 signal。Core0 自己的 MM 完成由本核流水依赖保证。
2. **1→参与 Add 核的到达放行**：Core0 执行 `wait` 后，给每个需要参与 Add 的核建立独立的 `IBSet/IBWait` 事件配对；**不得多个核消费同一个一次性标志**。若官方 API 的事件容量/复用合同不能满足，则优先测试 Core0 单核 Add 或降低 Add 核数，而非立刻自制事件协议。

官方 IB 的 GM workspace 要按逻辑核数和 eventID 范围配置，初始清零；`isAIVOnly`、MIX 模式下的索引语义、同一 event 多轮消费和 Graph replay 必须先通过专项 probe。尤其逻辑 block 数不得超过实际核数，否则可能发生调度死锁。

### 6.3 静态工作区跨 wave/调用的复用：**一次波级握手**

**只有 `quiet()` 不够**：它证明自己的 SDMA 已落地，却不能证明对端已做完 Add 并不再读取其接收区。

基准实现采用**每个 wave 一次对称完成握手**：

1. 各核做完本波所有 Add，Core0 在完成必要的本地 Add-done 会合后执行 `quiet()`。
2. Core0 用公共 `signal()` 发送**一封 8B 完成通知邮件**至对端的独立控制区（恒定不改写的源字节；通过 `imm` 或顺序校验 wave 标识）。
3. Core0 用公共 `wait()` 消费对端完成邮件；此时各端都已消费本波 C 数据，且本端发送已经落地。
4. 只有此后才能在下一 wave 或下一次调用覆写同一 C0/C1 区。两侧都 **先发送完成通知、再等待对端通知**，避免对称死锁。

该握手**不在每个 chunk 执行**。同一 wave 的所有 chunk 使用不同地址，因此 Send/Recv 可独立推进，无需三槽循环、逐 chunk credit、tagCell、ackCell、shadowCounter。若整次调用在一个 window 内完成，一次调用只付出一对完成通知和一次收尾 `quiet`。

**跨调用额外注意：**同一 Rank 若由多个 stream 并发使用该池，需要上层串行化或按 stream 独立工作区。Graph replay 不能跳过波级握手，也不能把「旧邮件」视为当前 wave；对端失联/报错必须 fail-stop，不允许复用污染的池。

## 7. 可实现的数据流伪代码（调度级，不是可直接编译的 AscendC）

```cpp
rank = tp_rank;  core = GetBlockIdx();
for (wave : waves) {
    for (k : wave.chunks) {
        MM_to_symmetric_C(rank, k);           // 参与 MM 的计算核
        // 官方核内流水完成 + 必要 cache clean
        IB_publish_mm_done(core, k);          // 需 probe 证实重用语义
        if (core == 0) {
            IB_gather_mm_done(k);
            mf_signal(data_src(rank,k), peer_dst(rank,k),
                      chunk_bytes(k), DATA_TAG(k));
        }

        if (strategy == OVERLAP && k > wave.first) {
            if (core == 0) {
                mail = mf_wait();              // 消费对端上一 chunk 的邮件
                check_data_mail(mail, k - 1);
                IB_release_add(k - 1);
            } else {
                IB_wait_add_release_if_needed(k - 1);
            }
            ADD_symmetric_C_to_output(k - 1);
        } else if (strategy == SERIAL) {
            // 本 chunk signal 后直接 wait → Add
            // Core0 或 Add 核数由本 M 的配置决定
            wait_release_add(k);
            ADD_symmetric_C_to_output(k);
        }
    }
    if (strategy == OVERLAP) {
        wait_release_add(wave.last); ADD_symmetric_C_to_output(wave.last);
    }
    IB_gather_all_add_done_if_needed();
    if (core == 0) {
        mf_quiet();                            // 本 Rank 所有 data 已落地
        mf_signal(CONST_CTRL_SRC, peer_ctrl, 8, WAVE_DONE);
        check_wave_done_mail(mf_wait());       // 双 Rank 均先 signal 再 wait
    }
    // 必要的本地放行后，才能复用下一 wave 的工作区
}
```

真正编码时须补齐「Core0/其余核是否同一轮执行 wait_release_add」及「波尾本核复用放行」的逐核时序，不得把上述抽象函数当作隐含全核 barrier。同步对象属于本 Kernel，本地核不能在 Core0 完成远端 wave 握手前提前开始下一 wave 的 MM。

## 8. 生命周期、Graph 与异常

- **建池**：复用已验证的 MemFabric Host 创建/几何校验，分配静态对称区与官方 IB 工作区；保持 V1/V2 的单池互斥规则，不能在同进程未经隔离复用同一通信邮箱。
- **初始状态**：官方 IB GM 工作区初始化为零，所有同步事件配对执行；清零和重复消费的正确语义由 probe 后固定，不能未经验证声称 graph replay 自动安全。
- **Graph**：在 capture 前完成建池、权重预处理、设备库和全部选中模板的 eager warmup；形状参数可固定，但邮件和事件需能跨 replay 正确演进。双 Rank 的相应调用顺序一致。
- **通信安全**：所有数据/完成邮件校验 `status/dst/len/imm`，超时、到达溢出、同步不匹配、`quiet` 失败均记录错误并拒绝重用池，不得静默 fallback 已部分传输的波。
- **长稳**：V1 曾在连续 eager 负载下观察到 MemFabric AICPU epoch 续命停摆与 AICore 超时；V2 单 Kernel 不视为自动修复，须重新覆盖 eager、Graph、10min+ 持续负载。
- **精度**：FP16 MM、两端数据交换及 Add 必须通过与 stock/V1 的正确性回归；bit-exact 作为目标，具体参考要固定舍入和输入分布并实测确认。

## 9. 工程实现范围与验证关口

```text
csrc/_310P/memfabric_mm_ar_v2/
    DESIGN.md                           本文件
    PLAN.md                             后续按新方案重编
    mf310p_v2_device.asc                 单 Kernel：MM + MemFabric + Add
    mf310p_v2_adapter_api.h              参数和上下文 ABI
    mf310p_v2_adapter.cpp                MemFabric 公共 Host 适配
    memfabric_mm_ar_v2_runtime.cpp       M/tiling/route、Graph、生命周期
    memfabric_mm_ar_v2_binding.cpp       Torch op 注册
    memfabric_mm_ar_v2_torch_adpt.h      Tensor 合同校验
```

**先验证能力，再优化性能**：

| 关口 | 必须回答的问题 | 产物/通过条件 |
|---|---|---|
| G0 官方同步 API | dav-2002 能否正确编译、执行 `IBSet/IBWait` 扇入/单发多收；事件多轮复用、MIX/Graph 是否安全 | 最小双向同步 probe + 延迟分布；不能依赖尚未证实的广播合同 |
| G1 最简串行 | MM 直接写对称区，单邮件 `signal/wait`，Add 正确；波级复用握手安全 | 双 Rank、1～1024 行、跨调用重复 correctness |
| G2 分核效率 | 1/2/4/8 核 MM 与 Add 的真实增益，N/M 切块、cache clean 成本 | 以 310P3 profiler 分解 MM/同步/Add，选定小 M 路径 |
| G3 交叠 | 256/512/1024 行 chunk 的 SDMA/计算重叠，不发生跨 Rank 循环等待 | M=2048～8192，逐块时序证明与 profiler 时间线 |
| G4 跨波及 Graph | `M>windowM` 静态地址复用、尾块、连续 replay、异常处理 | M=8193/16384/65536，graph×1000，双 Rank 长稳 |
| G5 发布判定 | 真实 Qwen3.6 的 40 层启用后是否优于 V1 与 stock | 层级延迟、Decode ITL、Prefill、输出吞吐、稳定性齐全 |

**P0 风险列表：**

1. `IBSet/IBWait` 的扇入/一对多事件配对、重用与 MIX 模式约束尚未实机证实；必须先 G0。
2. C0/C1 的 GM/cache 清洁纪律及 Add 的读取可见性必须经过对称内存实机检查。
3. 交叠中 Core0 同时承担 MM/signal/wait 时可能出现调度阻塞；每条等待边必须有已发布的前驱事件。
4. 波级 ACK 是「接收区已消费」而非仅传输完成的证明，不可省略；`quiet` 也不可代替。
5. 单 Kernel 小 M 的阻塞 `wait` 可能暴露之前多 Kernel launch 帮忙掩盖的通信等待，必须与 stock/V1 实测比较。

## 10. 相关源码与证据

- [V1 源码、已验证的 256/512/1024 单次 signal](../memfabric_mm_ar/memfabric310p_device.asc)
- [V1 实机验证报告：q=1/2/4 正确性与性能](../memfabric_mm_ar/VALIDATION_REPORT.md)
- [官方 MatmulAllReduceCustom 分核样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/23_matmul_all_reduce_custom/MatmulAllReduceCustom)
- [MemFabric 公共 SDMA API](https://github.com/10wvw01/memfabric_hybrid/blob/wgm-dev-310p/src/smem/include/device/smem_shm_aicore_base_sdma.h)

> **本版决策摘要：** 双引擎单 Kernel、串行/交叠双路线、静态 C0/C1 对称内存、Core0 单生产者、IBSet/IBWait 官方优先、逐 chunk 只发送数据邮件、每 wave 一次复用完成握手、256/512/1024 三档分块候选。原版 8 Lane 直写请求环、tag/ack cell、TAIL proxy、三槽滚动复用等设计全部废弃。


## 11. 首轮可运行原型与已知限制（T0）

本设计的第一轮**试验代码**参照官方 MatmulAllReduceCustom 的「多核 MM → 同步 → 通信提交」思路，放在本目录的 `mf310p_v2_trial_device.asc` 等文件中，见 [TRIAL.md](TRIAL.md) 和 [PLAN.md](PLAN.md)。**它不是本设计全部能力的交付**：

- 当前仅 `chunkRows=256/512/1024`、`M` 为 CHUNK 正整数倍且 `M<=8192`、8 核 M-split；串行和交叠两种模式可选。
- Core0 负责一封连续数据邮件/块；官方 `IBSet/IBWait` 负责核间依赖；`quiet + DONE` 负责波级复用。
- 还没有小 M N-split、任意 M 尾块、多波、自适应核数、PyTorch/vLLM 接线或正式 Graph 验证。
- **代码未在 310P3 上编译/运行**。首个实机步骤必须是 G0 的 `IBSet/IBWait` 官方 API 契约探针，然后才进行融合正确性和性能测试；现有 V1 路由、参数和行为完全不变。


## 12. T1 小矩阵主线：原始 NZ / 直接 ND（**未实机验证**）

上一版草拟的「照搬 V1 8 核 N-split + NZ 权重分片 + 输入 Staging + Blocked C」**不作为 V2 主线**，相关代码已从 T1 主路径撤除；V1 方案仅保留为独立性能对照，**V1 本身未修改**。

T1 主路径参照官方 `MatmulBlock::CalcGMOffset`：对于 `B=NZ,transposed=true`，沿输出 N 的列偏移为 `colOffset×16` 个 FP16 元素；C=ND 时直接定位 `row×2048+colOffset`。具体落地：

- 小 M 输入 `X[M,2048]` 直接传 MM（`M=1..256`），权重直接采用**原始完整 NZ** `W`，不做 8 MiB 切片、输入 Staging 或 blocked 输出重排。
- `mmCores∈{1,2,4,8}`，每个 MM 核处理 `N/mmCores` 连续列；单核 static tiling 只计算这一 N 片。通过 `SetOrgShape(M,2048,2048,2048,2048)` 维护 B 的 NZ 原始全列步长与 C 的 ND 行步长；`SetSingleShape(M,2048/mmCores,2048)` 缩到真实 M，避免补齐输入。
- MM 各核**直接写对称区中最终 ND `[M,2048]` 地址**。每一核只清洁其负责的 64B 对齐列片；Core0 汇总 `IBWait` 后一次 `signal(M×2048×2)`。
- 收到对端数据后 `addCores∈{1,2,4,8}` 按整行分配工作。Core0 单核 Add 时无需额外释放/完成广播；多核 Add 使用官方 IB 独立事件；所有 Add 都直接按 ND 读写。
- 单次调用最后由 Core0 做 `quiet` 与波级双向 DONE，避免下一次调用复写当前 C 区。所有 MemFabric API 仍是公共 `signal/wait/quiet`。
- `SetOrgShape/SetSingleShape` 配合**原始 NZ 子矩阵 + ND 非紧凑行步长**目前属于 **310P3 待验证的关键能力**，不是已证明正确。T1 对应 `mf310p_v2_trial_run_nz_direct_small` 原型只通过源码审视，不接生产路由。

**性能衡量：** stock FP16 MM+HCCL、V1 已验证的小 M N-split、V2 T0 256 行 M-split 均作参考。先 G0 编译与 bit-exact；若官方 Matmul API 在 dav-2002 不能直接写 C(ND) 跨列子块，则记录编译/数值证据，回退至官方支持的 Matmul 分块能力；不得默默引入 V1 blocked 布局替代主设计。

测试及边界：[NZ_DIRECT_TRIAL.md](NZ_DIRECT_TRIAL.md)、[bench_nz_direct_small.py](bench_nz_direct_small.py)、[PLAN.md](PLAN.md)。

> T0 的 G0/G1/G2 实机通过与 T1 的直接-NZ 正确性是两组独立证据。首次新输出指针 `aclrtSynchronizeDevice` 是旧版已知临时规避，尚非 Graph 安全的产品方案。
