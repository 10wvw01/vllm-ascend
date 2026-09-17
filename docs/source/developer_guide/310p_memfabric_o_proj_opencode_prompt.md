# OpenCode 后续研发提示词

下面整段内容可以直接复制给运行在 Ascend 310P3 服务器上的 OpenCode。建议让 OpenCode 在 vLLM-Ascend 仓库根目录启动，并确保它可以读取本机 `wgm-dev-310p` MemFabric 源码、CANN 环境和编译日志。

```text
你正在 Ascend 310P3 实机上继续开发一个 vLLM-Ascend 融合算子项目。请把自己当作负责把当前 WIP 真正编译、跑通、验证并继续优化到最终验收的资深 Ascend/vLLM/C++/AscendC 工程师，而不是只做代码审阅。

============================================================
一、需求目标锁定：这是最高优先级
============================================================

开始任何修改之前，第一份必须完整阅读的文件是：

docs/source/developer_guide/310p_memfabric_o_proj_requirements.md

它是本项目的需求基线和最终验收合同。

资料冲突时按以下优先级处理：

1. 需求目标文档；
2. Ascend 310P3 实机事实和 wgm-dev-310p 定制 MemFabric 真实源码/ABI；
3. 设计文档；
4. 研发计划；
5. 当前代码和历史假设。

如果真实硬件/定制 MemFabric 证明当前设计或代码有问题，应修改设计/代码继续满足需求，不能为了保住当前实现而降低项目目标。

你必须始终记住最终要交付的不是“一个能跑的通信 demo”，而是：

Eco-Tech/Qwen3.6-35B-A3B-w8a8
+ Ascend 310P3 单物理卡双 die
+ TP=2
+ 仅 full-attention self_attn.o_proj
+ 310P static W8A8 -> BF16 数值语义保持
+ 使用 wgm-dev-310p 定制 MemFabric
+ o_proj matmul 与 TP reduction 由一条融合路径拥有
+ tile 级 MM[t+1] 与 communication/reduce[t] 真正 overlap
+ repeated-wave 安全
+ eager 正确
+ ACL Graph 正确
+ profiler 能证明 overlap
+ 有 baseline/fused 可重复性能数据
+ 最终具有正向性能收益

不得自行把目标降级成以下任意一种：

- 普通 HCCL allreduce；
- 只完成 MemFabric 通信样例；
- 只完成 Phase-1 correctness prototype；
- 放弃真正 overlap；
- 只支持 eager；
- 只做到“编译成功”；
- 只做 benchmark、不接 Qwen3.6；
- 扩散去做 TP>2、其它模型或无关框架重构。

Phase-1 tiled npu_quant_matmul + copy + MemFabric 是必要 bring-up 阶段，但不是最终性能形态。最终目标仍需要 Phase-2 direct tiled W8A8 producer，直接写 symmetric send tile 并 notify，尽量去掉 local copy 和 per-tile ACLNN matmul launch 开销。

每完成一个阶段，在继续之前都必须问自己：

“我刚完成的工作是否让项目更接近上述最终目标和需求文档的验收条件？”

如果答案是否定的，请停止扩散，回到需求主线。

============================================================
二、仓库、模型和硬件
============================================================

仓库：
https://github.com/10wvw01/vllm-ascend

工作分支：
feat/310p-w8a8-o-proj-memfabric-ar

不要切回 main，不要丢弃现有提交，不要重写与本功能无关的代码。

目标模型：
Eco-Tech/Qwen3.6-35B-A3B-w8a8

目标硬件：
Ascend 310P3
单物理卡、双 die
Tensor Parallel = 2

必须以 npu-smi 的实际逻辑 NPU/die 映射为准。

============================================================
三、核心优化目标
============================================================

只针对 Qwen3.6/Qwen3.5 MoE text trunk 的 full-attention self_attn.o_proj。

普通路径：

  W8A8 local o_proj matmul
      -> generic TP allreduce

目标路径：

  MM[t] -> publish[t] -> MemFabric SDMA[t] -> peer arrival[t] -> local reduce[t]
    |
    +----------------------------------------------------------> MM[t+1]

硬要求：

MM[t+1] 不等待 SDMA/reduce[t]。

只允许在安全 buffer reuse、wave boundary 和 operator 最终完成时进行必要等待。

最终 op 返回前必须保证所有 compute/send/receive/reduce 已完成。

是否真的 overlap 必须由 310P profiler timeline 证明，不能因为代码里“没有显式 wait”就宣布完成。

============================================================
四、通信依赖
============================================================

必须使用定制 MemFabric Hybrid：

https://gitcode.com/GDD_ESCC/memfabric_hybrid/tree/wgm-dev-310p

这是本功能实际使用的 MemFabric 实现。

不要：

- 编译安装官方/upstream MemFabric 来实现本功能；
- 做成 upstream MemFabric + 310P plugin；
- 凭经验猜 upstream lib/soname/API；
- 新增独立 adapter .so；
- 用 runtime dlopen 隐藏真实链接关系。

正确关系应为：

wgm-dev-310p source
    -> build/install customized MemFabric
    -> vLLM-Ascend direct include/link customized install

HCCL 只允许作为 correctness/latency reference，不能成为融合算子内部通信路径。

============================================================
五、开始修改前必须阅读
============================================================

按顺序完整阅读：

1. docs/source/developer_guide/310p_memfabric_o_proj_requirements.md
2. docs/source/developer_guide/310p_memfabric_o_proj.md
3. docs/source/developer_guide/310p_memfabric_o_proj_development_plan.md
4. docs/source/user_guide/feature_guide/310p_memfabric_o_proj_usage.md
5. AGENTS.md
6. 当前分支全部 memfabric_o_proj 相关源码

重点源码：

- vllm_ascend/_310p/ops/memfabric_o_proj.py
- vllm_ascend/_310p/quantization/methods/w8a8_static.py
- csrc/memfabric_o_proj_binding.cpp
- csrc/memfabric_o_proj_runtime.cpp
- csrc/memfabric_o_proj/external/memfabric310p_adapter_api.h
- csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp
- csrc/memfabric_o_proj/external/memfabric310p_device.asc
- cmake/memfabric_310p.cmake
- benchmarks/scripts/bench_310p_memfabric_o_proj.py
- tests/ut/_310p/test_memfabric_o_proj_source.py

============================================================
六、当前已经完成，不要无理由推倒重来
============================================================

当前已有：

- feature gate 默认关闭；
- 仅 qwen3_5_moe_text；
- 仅 *.layers.N.self_attn.o_proj；
- config.layer_types[N] == full_attention；
- TP=2；
- local K=2048；
- output N=2048；
- 310P static W8A8；
- BF16 output；
- 命中后关闭 generic RowParallelLinear allreduce；
- custom MemFabric direct-link 构建框架；
- persistent runtime state；
- symmetric send/recv non-alias arenas；
- dedicated reduce stream；
- failed wave runtime poison；
- Phase-1 tiled npu_quant_matmul -> copy send -> publish -> SDMA -> reduce；
- AICore publish kernel；
- correctness-baseline BF16 reduce；
- source regression；
- TP=2 MemFabric communication correctness/latency benchmark。

这些是现有基础，不代表都已经在真实 310P 上验证正确。

============================================================
七、永远不能破坏的 correctness invariants
============================================================

1. local send buffer 在 peer SDMA 可能读取期间必须 immutable。
2. recv/final buffer 与 send buffer 不得发生危险 alias。
3. rank0-only quant_bias，只能加一次，不能两 rank 各加后再求和。
4. 保持现有 310P W8A8 dequant/output dtype/FRACTAL_NZ 语义。
5. non-target layer 必须保持原路径。
6. feature off 必须保持原路径，并且不能要求 custom MemFabric。
7. tile 之间不能加 host barrier/wait 破坏 overlap。
8. repeated wave 不能依赖一次性 non-zero flag。
9. 不允许 silent fallback 到 HCCL 后仍报告 fusion enabled。
10. performance 结论必须来自实机 benchmark/profiler。

============================================================
八、当前最重要的未知项
============================================================

以下内容必须从本机 wgm-dev-310p 真实源码确认，不能猜：

1. smem_shm_aicore_sdma.h 的真实 API/常量；
2. smem_shm_sdma_submit/wait/get_workspace/get_result 的真实 declaration；
3. submit 背后的 AICPU mailbox/SQE orchestration；
4. chunk source offset 推进方式；
5. destination offset 推进方式；
6. arrival flag 地址推进方式；
7. flag 写入值、完成顺序和 cache visibility；
8. mailbox/flag reuse/reset 语义；
9. 是否真的存在 MemFabric control barrier；
10. example 08 的 .asc 真实编译命令及产物形式；
11. custom install 实际生成的 libraries 和 link order。

已知 example 08 host 侧使用 MPI_Barrier，因此当前代码中的 smem_shm_control_barrier 假设必须核实；没有真实源码依据就不能继续把它当成最终方案。

============================================================
九、阶段 1：环境盘点
============================================================

立即执行并保存输出：

- git status
- git rev-parse HEAD
- npu-smi info
- python3 --version
- cmake --version
- torch version
- torch-npu version
- vLLM version
- vLLM-Ascend location/version
- echo $ASCEND_HOME_PATH
- echo $SOC_VERSION
- 定位本机 wgm-dev-310p 源码目录和 commit
- 定位 custom MemFabric install prefix
- find install prefix 下的 smem*.h、*.so、*.a

把结果写入 bring-up log，并把最终确认值同步进项目文档。

============================================================
十、阶段 2：读定制 MemFabric 真实实现
============================================================

找到：

08_310p_aicore_aicpu_sdma.asc

以及它关联的 header、AICPU implementation、CMake/build/install 脚本。

必须回答并记录：

- notify mailbox slot 格式；
- submit(chunks) 的真实工作方式；
- src GVA 来源；
- dst offset 递增；
- flag offset 递增；
- SDMA 完成后何时写 flag；
- poll_flag 成功值；
- flag/mailbox 是否必须手工清；
- 多 wave 的官方 reset/reuse 协议；
- 是否存在 MemFabric barrier；
- example 08 的 MPI_Barrier 在 vLLM worker 中应如何替代；
- .asc 的真实 compiler/build rule；
- 最终 libraries 和 link order。

只有源码证明存在的 API 才可以使用。

============================================================
十一、阶段 3：先把真实 build 打通
============================================================

按 custom MemFabric example 08 的真实 build rule 编译：

csrc/memfabric_o_proj/external/memfabric310p_device.asc

不要猜 .asc compiler 参数。

根据真实服务器环境设置：

SOC_VERSION=ascend310p3
VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
VLLM_ASCEND_310P_MEMFABRIC_ROOT=<真实 custom install prefix>
VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES=<真实 library list>
VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT=<真实 device object/stub>
VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32

如果真实 device build 产物不是普通 .o，修改 cmake/memfabric_310p.cmake 适配真实产物，不要伪造 object。

编译问题按以下顺序解决：

header/signature
-> device compile
-> launch stub
-> linker symbols
-> runtime loader/RPATH

每次保留关键原始错误，并做最小修复。

阶段结束前执行一次“需求目标锁定检查”：build 通了只是里程碑，不是项目完成。

============================================================
十二、阶段 4：修正 repeated-wave 同步
============================================================

根据 wgm-dev-310p 真实语义修正 wave reuse。

目标必须保证：

上一 wave 两 rank 都结束
-> 双 rank rendezvous
-> 各自安全清理 local mailbox/arrival state
-> 双 rank rendezvous
-> 新 wave 才允许 publish

同时绝对不能在 tile 间 barrier。

若 MemFabric 没有自己的 barrier，优先研究 vLLM/torch.distributed/store 的 wave-boundary rendezvous；不要直接把 MPI 变成生产依赖，除非项目负责人明确决定。

验收：repeated wave 至少连续 1000 次无 hang、无 mismatch、无 stale flag。

============================================================
十三、阶段 5：先跑独立通信 benchmark
============================================================

运行：

torchrun --standalone --nproc-per-node=2 \
  benchmarks/scripts/bench_310p_memfabric_o_proj.py

至少覆盖：

M = 1, 8, 32, 64, 128, 512, 2048

要求：

- rank0/rank1 都 PASS；
- bilateral exchange 正确；
- local reduce 正确；
- 与 HCCL reference allclose；
- tail chunk 正确；
- repeated-wave 1000+ 稳定；
- 无偶发 hang。

如果失败，先修通信，不启动 35B 模型。

通信 benchmark PASS 也不代表最终目标完成，继续进入 W8A8 和模型集成。

============================================================
十四、阶段 6：Phase-1 W8A8 单层 correctness
============================================================

比较：

reference = local npu_quant_matmul + TP allreduce
fused     = tiled npu_quant_matmul + MemFabric exchange + local reduce

检查：

- BF16 numerical correctness；
- rank0-only quant_bias；
- 多种 M；
- tail M；
- repeated calls；
- feature gate off fallback；
- full-attention 命中；
- linear-attention/GDN 不命中。

如果缺少单层 hardware test，新增可重复运行的 test/benchmark。

不要通过无理由放宽 tolerance 掩盖逻辑错误。

============================================================
十五、阶段 7：启动 Qwen3.6
============================================================

模型：
Eco-Tech/Qwen3.6-35B-A3B-w8a8

第一次只做 TP=2 eager correctness：

--tensor-parallel-size 2
--quantization ascend
--dtype bfloat16
--enforce-eager

第一次不要额外开启 EP、sequence-parallel MoE 或其它实验优化，减少变量。

确认：

- full_attention self_attn.o_proj 确实命中；
- linear_attn/GDN 不命中；
- 短 prompt 正常；
- prefill 正常；
- 多轮请求不 hang；
- feature-off baseline 正常。

只在 eager 能跑不等于完成，后面仍必须验证 ACL Graph。

============================================================
十六、阶段 8：Profiler 验证真正 overlap
============================================================

correctness 稳定后，用 profiler timeline 证明：

MM[t+1]

与：

SDMA[t] / peer receive / local reduce[t]

存在真实设备侧时间重叠。

如果没有 overlap，依次排查：

- stream dependency；
- hidden synchronization；
- npu_quant_matmul host/device blocking；
- cache clean；
- publish kernel；
- AICPU orchestration；
- SDMA；
- tile_m。

没有 timeline 证据时，不得把 overlap 标记为完成。

============================================================
十七、阶段 9：Phase-2 final direct W8A8 producer
============================================================

Phase-1 完全稳定以后，继续完成最终性能形态。

目标：

one AscendC/CATLASS-level W8A8 tiled producer
    -> compute/dequant BF16 tile
    -> rank0-only quant_bias
    -> direct write symmetric send[t]
    -> cache clean
    -> smem_shm_sdma_notify(t)
    -> immediately continue MM[t+1]

需要尽量移除：

- npu_quant_matmul output -> send arena local copy；
- per-tile ACLNN matmul launch overhead。

先搜索当前 CANN/CATLASS/vLLM-Ascend 中真实可复用的 310P INT8 matmul primitive，不要为了形式从零手写复杂 Cube matmul。

先把大 M 的 M-row tiling 做稳定；decode 小 M 再用实机数据决定是否加入 N-panel tiling。

Phase-2 完成后重新做单层 correctness、profiler 和性能对比。

============================================================
十八、阶段 10：reduce 优化
============================================================

当前 scalar BF16 reduce 只是 correctness baseline。

协议稳定后再逐步优化：

vector/UB
-> multicore
-> double buffer

每一步都必须和 reference 做 correctness，并用 profiler/benchmark 判断是否真的有效。

============================================================
十九、阶段 11：ACL Graph
============================================================

Eager 通过后去掉：

--enforce-eager

验证：

- capture/replay；
- stable MemFabric GVA；
- stable workspace；
- repeated graph replay；
- wave generation/flag reuse；
- 不同 M/graph bucket；
- repeated request。

如果 graph-only 失败，先做融合 op 最小 graph reproducer。

============================================================
二十、最终性能验收
============================================================

必须记录 baseline 和 fused：

- MemFabric/HCCL reference communication microbenchmark；
- target o_proj microbenchmark；
- representative prefill；
- representative decode；
- 至少一组 Qwen3.6 end-to-end 指标。

最终目标是可重复、可解释的正向性能收益。

如果 correctness 全部通过但 fused 没有正向收益，项目仍是“correctness 完成，性能目标未完成”，继续依据 profiler 找瓶颈，不能直接宣布整个项目完成。

============================================================
二十一、提交要求
============================================================

直接在当前 feature branch 工作。

每个逻辑阶段使用小而清晰的 commit，例如：

fix(310p): match custom MemFabric SDMA ABI
fix(310p): make MemFabric wave reset race-free
build(310p): compile MemFabric device kernel with custom toolchain
test(310p): add TP2 repeated-wave hardware coverage
perf(310p): vectorize MemFabric BF16 reduce
feat(310p): add direct W8A8 tiled producer
docs(310p): record verified 310P bring-up commands

不要把所有变化塞进一个巨大 commit。

============================================================
二十二、文档维护要求
============================================================

每完成一个阶段立即更新：

- docs/source/developer_guide/310p_memfabric_o_proj_development_plan.md
- docs/source/user_guide/feature_guide/310p_memfabric_o_proj_usage.md
- 必要时更新设计文档

需求目标文档：

docs/source/developer_guide/310p_memfabric_o_proj_requirements.md

是项目验收基线。不要自行修改其中硬目标来迁就实现；只有项目负责人明确改变需求时才修改它。

把服务器实际验证出的未知内容写回文档：

- MemFabric commit；
- build/install command；
- library names；
- .asc compile command；
- barrier 最终实现；
- submit/offset/flag semantics；
- NPU topology；
- 最终 vLLM-Ascend build command；
- 最终 vllm serve command；
- benchmark correctness；
- profiler timeline 结论；
- baseline/fused 性能。

============================================================
二十三、工作方式
============================================================

不要只告诉我“下一步应该做什么”。

请实际执行：

检查
-> 阅读真实源码
-> 编译
-> 观察原始错误
-> 定位根因
-> 做最小修改
-> 重新编译
-> 实测
-> benchmark/profiler
-> commit
-> 更新文档

每遇到失败：

1. 保留关键原始错误；
2. 给出根因；
3. 做最小修复；
4. 重新编译/测试；
5. 记录结果。

如果确实被路径、权限或外部源码完全阻塞，只告诉我：

1. 唯一阻塞项是什么；
2. 已经检查过什么；
3. 我最少需要提供什么。

除此之外，优先自行从服务器环境、仓库源码、MemFabric 源码、build log 和 profiler 中发现答案。

最终只有在需求目标文档中的 build、communication correctness、W8A8 correctness、Qwen3.6 E2E、profiler overlap、ACL Graph 和 performance 验收都具备证据后，才能认为该融合算子项目真正完成。
```