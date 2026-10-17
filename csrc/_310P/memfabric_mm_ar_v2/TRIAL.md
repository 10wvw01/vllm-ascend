# V2 单 Kernel 原型（构建默认关闭）

目的：使用官方 MatmulAllReduceCustom 的计算分核+完成会合调度，并在 310P3 上用 MemFabric 3 个公共通信原语实现双 Rank 全量 AllReduce。实验性代码，不注册 PyTorch op、不接模型路由；2026-10-17 已在 dav-2002（CANN 9.1，TP=2）完成 G0/G1/G2 实机位精确验证（详见同目录《20261017实机测验及修复.md》）。

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

## 首次启动预热（run 内建，调用方无需处理）

实测 dav-2002 平台缺陷：MemFabric 建池**之后**新分配的输出缓冲，其页面对 AICore 的 MTE3（向量队列）写首次访问会被**静默丢弃**——内核其余部分（标量写、Matmul GM 写、IB/SDMA/DONE 全协议）全部正常落地，仅 TQue 输出消失，且无任何错误码。池内存与建池前分配的缓冲不受影响。

因此 `mf310p_v2_trial_run` 内部跟踪输出缓冲指针：y 首次出现（或更换）时先执行一次 `aclrtSynchronizeDevice()`（全设备同步，等待建池后台流完成地址映射）再发射。流级同步、SDMA memset、额外内核发射均无效果（均已实测）。设置环境变量 `MF310P_V2_NOWARMUP=1` 可关闭该行为（仅测试用）。

从 M=256、chunkRows=256、overlap=0 的正确性开始测，千万不要直接用这版替换生产路由。

## 注意

- 一个进程只能启用一套 MemFabric 池：**不可同时运行 V1 serving 和这个实验**。
- G0 的官方 IBSet/IBWait 探针必须先通过；一旦陷入 hang，优先检查 isAIVOnly 的实际编译模式和 GM/UB workspace 约束。
- Kernel 目前固定 8 核、全 chunk、整波 8192 行；tail、动态 M、graph replay 的错误路径以及多 wave 均未验证。
- 诊断脚手架：`mf310p_v2_trial_diag(h, x, w, y, 0..7, stream)` 提供分级隔离内核（纯 MM / +TQue Add / +IB / +DONE 握手 / 生产内核 / 16 参编组 / 循环结构复刻），供分阶段排障；`mf310p_v2_trial_addresses/ib_addr/status_addr/status_words` 暴露关键地址与状态。

## T1 小 M / 原始 NZ 直接 ND 原型（2026-10-17/18 首轮实机验证 PASS）

- C API：`mf310p_v2_trial_run_nz_direct_small(handle,x,w_nz,y,M,mm_cores,add_cores,stream)`。
- `M=1..256`；`mm_cores/add_cores∈{1,2,4,8}` 相互独立；无权重切片、输入 staging 和 blocked 输出。
- 官方 `MatmulBlock` NZ B 列偏移规则与 `SetOrgShape/SetSingleShape` 组合已实机验证（单卡 1028/1028 + 双卡 160 组合位精确）；`basicM` 须 ≤128（UB 工作区合同，见报告 BUG-1）。
- 参考 [NZ_DIRECT_TRIAL.md](NZ_DIRECT_TRIAL.md) 的阶段性命令、[bench_nz_direct_small.py](bench_nz_direct_small.py)、[t1_nz_direct_trial.py](t1_nz_direct_trial.py)、[t1_diag_single.py](t1_diag_single.py)。
- T0 的 G0/G1/G2 实测与 T1 的实机验证是两组独立证据；V1 N-split 只保留为独立基线。结果与遗留问题见 [T1_NZ_DIRECT_REPORT.md](T1_NZ_DIRECT_REPORT.md)。
