# V2 单 Kernel 原型（构建默认关闭）

目的：使用官方 MatmulAllReduceCustom 的计算分核+完成会合调度，并在 310P3 上用 MemFabric 3 个公共通信原语实现双 Rank 全量 AllReduce。当前为实验性代码，**尚未进行 CANN 编译和实机正确性验证，不会影响 V1 serving**。

## 构建

设置 MEMFABRIC_HYBRID_HOME_PATH 指向已安装的 wgm-dev-310p，CANN 9.1、bisheng dav-2002，然后在编译时同时设置：

- VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
- VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1

设备库输出 libmf310p_v2_trial_device.so，Native C API 声明在 mf310p_v2_trial_adapter_api.h。此版**未注册 PyTorch op，也未自动连接模型路由**。

## 双 Rank 调用流程

调用方先完成本 Rank 的 aclInit / aclrtSetDevice / stream 初始化，两卡 store_url 相同，调用相同顺序的 API：

    mf310p_v2_trial_t* h = nullptr;
    CHECK(mf310p_v2_trial_create(rank, store_url, 512, &h));
    CHECK(mf310p_v2_trial_ib_probe(h, 1000, stream));
    CHECK(mf310p_v2_trial_run(h, x, w_nz, y, 4096, 1, stream));
    CHECK(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
    uint64_t status = ~0ULL;
    CHECK(mf310p_v2_trial_check_status(h, &status));
    CHECK(status == 0);
    CHECK(mf310p_v2_trial_destroy(h));

传入的 x/w_nz/y 均为 device pointer，x 形状 [M,2048]，w 为已预处理 NZ 权重，y 是独立输出，不得 alias C0/C1；rank=0/1，M 必须是 chunkRows 的正整数倍且不超过 8192。overlap=0 串行；overlap=1 则 MM(k) 掩盖 SDMA(k-1)。

从 M=256、chunkRows=256、overlap=0 的正确性开始测，千万不要直接用这版替换生产路由。

## 注意

- 一个进程只能启用一套 MemFabric 池：**不可同时运行 V1 serving 和这个实验**。
- G0 的官方 IBSet/IBWait 探针必须先通过；一旦陷入 hang，优先检查 isAIVOnly 的实际编译模式和 GM/UB workspace 约束。
- Kernel 目前固定 8 核、全 chunk、整波 8192 行；tail、动态 M、graph replay 的错误路径以及多 wave 均未验证。
- 本原型没有单独提交可执行 C++ runner；上述 C API 用于下一轮连接最小双 Rank 测试程序。
