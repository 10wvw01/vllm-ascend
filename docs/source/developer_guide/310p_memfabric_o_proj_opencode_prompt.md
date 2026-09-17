# OpenCode 后续研发提示词

下面整段内容可以直接复制给运行在 Ascend 310P3 服务器上的 OpenCode。建议让 OpenCode 在 vLLM-Ascend 仓库根目录启动，并确保它可以读取本机 `wgm-dev-310p` MemFabric 源码、CANN 环境和编译日志。

```text
你正在 Ascend 310P3 实机上继续开发一个 vLLM-Ascend 融合算子项目。请把自己当作负责把当前 WIP 代码真正编译、跑通、验证并继续优化的资深 Ascend/vLLM/C++/AscendC 工程师，而不是只做代码审阅。

【仓库与分支】
仓库：https://github.com/10wvw01/vllm-ascend
工作分支：feat/310p-w8a8-o-proj-memfabric-ar
不要切回 main，不要丢弃现有提交，不要重写与本功能无关的代码。

【目标模型】
ModelScope: Eco-Tech/Qwen3.6-35B-A3B-w8a8
目标模型目录在服务器上可能已经下载；如果没有，再按项目使用文档下载。

【目标硬件/拓扑】
Ascend 310P3
单物理卡、双 die
Tensor Parallel = 2
必须以 npu-smi 的实际逻辑 NPU/die 映射为准。

【核心优化目标】
只针对 Qwen3.6/Qwen3.5 MoE text trunk 的 full-attention self_attn.o_proj：

normal:
  W8A8 local o_proj matmul -> TP allreduce

target:
  MM[t] -> publish[t] -> MemFabric SDMA[t] -> peer arrival[t] -> local reduce[t]
    |
    +----------------------------------------------------------> MM[t+1]

要求 compute/communication 真正 overlap：MM[t+1] 不等待通信 tile[t]。最终 op 返回前才 join 全部 compute/communication/reduce。

【通信依赖】
必须使用定制 MemFabric Hybrid 分支：
https://gitcode.com/GDD_ESCC/memfabric_hybrid/tree/wgm-dev-310p

这是独立编译安装的定制分支，不是官方现有 MemFabric ABI，不允许凭经验猜官方 lib 名、soname、install 目录或 API。

vLLM-Ascend 应直接 include/link 该定制安装产物；不要重新引入独立 adapter .so 或 runtime dlopen。

【首先必须阅读】
开始修改前完整阅读：
1. docs/source/developer_guide/310p_memfabric_o_proj.md
2. docs/source/developer_guide/310p_memfabric_o_proj_development_plan.md
3. docs/source/user_guide/feature_guide/310p_memfabric_o_proj_usage.md
4. AGENTS.md
5. 当前 PR/分支的所有 memfabric_o_proj 相关代码

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

【当前已经完成，不要推倒重来】
- feature gate 默认关闭；
- 只命中 qwen3_5_moe_text；
- 只命中 *.layers.N.self_attn.o_proj；
- 额外检查 config.layer_types[N] == full_attention；
- TP=2、Qwen3.6 固定 shape、310P static W8A8、BF16 限制；
- 命中后关闭 generic RowParallelLinear allreduce；
- custom MemFabric direct-link 构建框架；
- persistent runtime state；
- send/recv 非 alias symmetric arenas；
- dedicated reduce stream；
- failure poison；
- Phase-1 tiled npu_quant_matmul -> copy send -> publish -> SDMA -> reduce；
- AICore publish + correctness-baseline reduce；
- source regression；
- 独立 TP=2 MemFabric correctness/latency benchmark。

【当前最重要的未知项】
这些必须从本机 wgm-dev-310p 源码确认，不能猜：
1. smem_shm_aicore_sdma.h 的真实 API/常量；
2. smem_shm_sdma_submit/wait/get_workspace/get_result 的真实 declaration；
3. submit 背后的 AICPU mailbox/SQE orchestration；
4. chunk src/dst offset 推进方式；
5. arrival flag 的地址推进、写入值、可见性和复用语义；
6. 是否真的存在可用的 MemFabric control barrier；已知 example 08 host 侧使用 MPI_Barrier，因此当前代码中的 smem_shm_control_barrier 假设必须核实；
7. example 08 的 .asc 真实编译命令、object/stub 形式；
8. custom install 实际生成的 libraries 和链接顺序。

【工作原则】
- 先实机编译和 correctness，再优化。
- 遇到 API 不匹配，先读定制 MemFabric 源码，不要发明 wrapper 来掩盖未知语义。
- 不要把 HCCL 放进融合实现；HCCL 只允许作为 correctness/latency reference。
- 不要让 local reduce 修改仍可能被 peer SDMA 读取的 send buffer。
- quant_bias 只能 rank0 加一次。
- 必须保持现有 310P W8A8 dequant/output dtype/FRACTAL_NZ 语义。
- 不要为了“编译通过”删除同步/错误检查。
- 不要在 tile 间加入阻塞式 host wait；这会破坏 overlap 目标。
- 所有性能结论必须来自 310P profiler/benchmark，不要凭代码结构声称已经 overlap。
- 每次重要修复都更新研发计划/使用说明，记录真实命令和环境。

【第一阶段：环境盘点】
立即执行并保存输出：
- git status
- git rev-parse HEAD
- npu-smi info
- python3 --version
- cmake --version
- python3 打印 torch/torch_npu/vllm 版本
- echo $ASCEND_HOME_PATH
- echo $SOC_VERSION
- 定位本机 wgm-dev-310p 源码目录和 commit
- 定位 custom MemFabric install prefix
- find install prefix 下的 smem*.h、*.so、*.a

把结果写入一个本地 bring-up log，必要时同步到研发计划文档。

【第二阶段：读定制 MemFabric 真实实现】
在 wgm-dev-310p 源码中找到 example 08：08_310p_aicore_aicpu_sdma.asc 以及关联 CMake/build 脚本。

重点回答并记录：
- AICore notify mailbox slot 格式是什么？
- submit(chunks) 是一次启动一个持续消费 mailbox 的 AICPU task，还是 host 每 chunk submit？
- src GVA 来自 mailbox 还是固定 base + offset？
- dst 如何按 chunk 递增？
- flag 如何按 chunk 递增？
- SDMA 完成后何时写 flag？
- poll_flag 的成功值是什么？是否需要清 flag？
- mailbox slot 是否需要手工清？
- 连续多 wave 的官方/示例 reset 协议是什么？
- example 的 MPI_Barrier 能否在 vLLM worker 场景替换为现有 torch.distributed/store rendezvous？
- 有没有 MemFabric 自己的 barrier API？只有源码证明确实存在才使用。

【第三阶段：先把 build 打通】
按 custom MemFabric example 08 的真实 build rule 编译：
csrc/memfabric_o_proj/external/memfabric310p_device.asc

不要猜 .asc compiler 参数。

然后设置真实：
- VLLM_ASCEND_310P_MEMFABRIC_ROOT
- VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES
- VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT
- SOC_VERSION=ascend310p3
- VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1

使用项目使用说明中的受控方式构建 vLLM-Ascend。

编译失败时逐个解决，不要一次大改。每次保留完整 compiler/linker error。
优先级：header/signature -> device compile -> linker -> runtime loader。

如果 device .asc 的真实 build 产物不是普通 .o，请调整 cmake/memfabric_310p.cmake 适配真实产物，而不是伪造 object。

【第四阶段：修正 reusable-wave 同步】
这是 correctness 最高风险项。

根据你刚读到的 wgm-dev-310p 真实语义修正 mf310p_prepare_wave()。
目标必须保证：
- 上一 wave 两 rank 都结束后才能清本地 state；
- 两 rank 都完成清理后，才能允许新 wave publish；
- 新 peer notify 不能被对端 late memset 清掉；
- 不在 tile 间 barrier；
- repeated wave 1000+ 次不 hang、不读 stale flag。

若 MemFabric 无 barrier，可优先复用 vLLM 已有 process group/store 做 wave-boundary rendezvous；不要引入 MPI 作为 vLLM 生产依赖，除非项目明确决定这么做。

【第五阶段：先跑独立通信 benchmark】
运行：
torchrun --standalone --nproc-per-node=2 benchmarks/scripts/bench_310p_memfabric_o_proj.py

先覆盖 rows: 1,8,32,64,128,512,2048。
通过后做 repeat=1000 stress。

要求：
- 两 rank 都 PASS；
- 与 HCCL reference allclose；
- repeated wave 稳定；
- tail chunk 正确；
- 无偶发 hang。

如果失败，先修通信，不启动 35B 模型。

【第六阶段：Phase-1 W8A8 单层 correctness】
构造或从模型抽取目标 o_proj，比较：
reference = local npu_quant_matmul + TP allreduce
fused = tiled npu_quant_matmul + MemFabric exchange + local reduce

检查：
- BF16 allclose；
- rank0-only quant_bias；
- 多种 M；
- tail M；
- repeated calls；
- feature gate off fallback。

如果仓库还没有单层硬件 test，请新增一个清晰、可重复运行的 tests/benchmarks 脚本。

【第七阶段：启动 Qwen3.6】
模型：Eco-Tech/Qwen3.6-35B-A3B-w8a8
第一次只做 TP=2 eager correctness：
- --tensor-parallel-size 2
- --quantization ascend
- --dtype bfloat16
- --enforce-eager
- 不额外开启 EP/sequence-parallel MoE

确认：
- full_attention self_attn.o_proj 命中；
- linear_attn/GDN 不命中；
- 短请求正常；
- prefill 正常；
- 多轮请求不 hang；
- feature off baseline 正常。

【第八阶段：Profiler 验证 overlap】
只有 correctness 稳定后才开始。
必须用 profiler timeline 证明：
MM[t+1] 与 SDMA[t]/reduce[t] 有真实设备侧重叠。

如果没有 overlap，逐个排查：
- stream dependency；
- hidden sync；
- npu_quant_matmul host/device blocking；
- cache clean 是否串行化；
- publish launch overhead；
- AICPU orchestration；
- tile_m。

不要在没有 timeline 的情况下报告“已经 overlap”。

【第九阶段：Phase-2 direct producer】
Phase-1 完全稳定后，再实现一个 AscendC/CATLASS-level W8A8 tiled producer：
- 直接写 symmetric send arena；
- dequant/output BF16 与现有 310P 路径一致；
- rank0-only quant_bias；
- 每 tile 完成后 clean + smem_shm_sdma_notify；
- producer 继续下一 tile；
- 移除 local copy；
- 尽量减少 per-tile launch。

先做大 M 的 M-row tiling。decode 小 M 再评估 N-panel tiling。

不要从零手写复杂 Cube MM，先搜索 CANN/CATLASS/vLLM-Ascend 当前版本可复用的 310P INT8 matmul primitive，并以编译和 profiler 结果决定方案。

【第十阶段：reduce 优化】
当前标量 BF16 reduce 只是 correctness baseline。
通信稳定后再逐步改：vector/UB -> multicore -> double buffer。
每步都和 reference 做 correctness。

【第十一阶段：ACL graph】
Eager 通过后去掉 --enforce-eager。
验证 capture/replay、稳定地址、不同 M bucket 和 repeated request。
如果 graph-only 失败，先做融合 op 最小 graph reproducer。

【提交要求】
你可以直接修改当前 feature branch代码。
每个逻辑阶段使用清晰 commit，例如：
- fix(310p): match custom MemFabric SDMA ABI
- fix(310p): make MemFabric wave reset race-free
- build(310p): compile MemFabric device kernel with custom toolchain
- test(310p): add TP2 repeated-wave hardware coverage
- perf(310p): vectorize MemFabric BF16 reduce
- feat(310p): add direct W8A8 tiled producer
- docs(310p): record verified 310P bring-up commands

不要把所有变化塞进一个巨大 commit。

【文档维护要求】
每完成一个阶段立即更新：
- docs/source/developer_guide/310p_memfabric_o_proj_development_plan.md
- docs/source/user_guide/feature_guide/310p_memfabric_o_proj_usage.md
必要时同步设计文档。

尤其要把原先未知的内容替换成服务器实际验证内容：
- MemFabric commit
- build/install command
- library names
- .asc compile command
- barrier API/方案
- submit/flag semantics
- 最终 vLLM build command
- 最终 model serve command
- benchmark/profiler result

【输出方式】
不要只告诉我“应该怎么做”。请实际执行检查、编译、测试和修改。
每遇到失败：
1. 贴出关键原始错误；
2. 给出根因；
3. 做最小修复；
4. 重新编译/测试；
5. 记录结果。

如果某一步因为缺少路径、权限或外部源码完全无法继续，请明确指出唯一阻塞项以及我需要提供的最小信息；除此之外尽量自行从本机仓库、环境和日志中发现信息，不要频繁向我确认已经能从代码中得到的内容。

最终目标是把当前 WIP 变成：可重复构建、可启动 Qwen3.6、correctness 可证明、overlap 有 profiler 证据、性能可量化的 310P3 TP=2 融合算子实现。
```
