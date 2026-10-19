/* 独立 V2 实验 ABI，当前不直接接入 vLLM 模型分发。 */
#pragma once
#include <stdint.h>
#if defined(__GNUC__)
#define MF310P_V2_API __attribute__((visibility("default")))
#else
#define MF310P_V2_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mf310p_v2_trial mf310p_v2_trial_t;

/* 调用方必须先选择 310P3 设备并创建 ACL 上下文。 */
MF310P_V2_API int mf310p_v2_trial_create(int rank, const char* store_url,
                          uint32_t chunk_rows, mf310p_v2_trial_t** out);
/* 必须先通过 G0 探针，包含两次 AICore 启动的 IB 事件对复用验证。 */
MF310P_V2_API int mf310p_v2_trial_ib_probe(mf310p_v2_trial_t* handle,
                            uint32_t repetitions, void* acl_stream);
/* x/w/y 均为设备指针，形状依次为 [M,2048]、NZ[2048,2048]、[M,2048]。原型要求 M>0、M 能被 chunk_rows 整除且 M≤8192。两个 Rank 必须按相同的 M、chunk_rows、overlap 和调用顺序执行。接口只负责入队，验证时须同步 Stream 并检查 status。 */
MF310P_V2_API int mf310p_v2_trial_run(mf310p_v2_trial_t* handle, uint64_t x,
                        uint64_t w, uint64_t y, uint32_t m,
                        uint32_t overlap, void* acl_stream);
/* T1 原始 NZ 权重直接 ND 输出：精确支持 M=1～256，不切片权重、不暂存输入；MM/Add 核数独立取 1/2/4/8。仍为实验 C API，不由 vLLM 自动路由。 */
MF310P_V2_API int mf310p_v2_trial_run_nz_direct_small(
    mf310p_v2_trial_t* handle, uint64_t x, uint64_t w, uint64_t y,
    uint32_t m, uint32_t mm_cores, uint32_t add_cores,
    void* acl_stream);
/* T2 非融合基线：框架原生 MM 在同一 ACL Stream 上先生成连续 ND FP16 的 native_partial[M,2048]。AR-only 先暂存进对称 C，再经 MemFabric signal/wait、Vector Add 与波级 DONE 完成规约。支持 M=1～8192、Add 核数 1/2/4/8，完整计入暂存成本；禁止 native_partial 与 output 重叠，或同一上下文跨 Stream 并发。 */
MF310P_V2_API int mf310p_v2_trial_run_ar_only(
    mf310p_v2_trial_t* handle, uint64_t native_partial, uint64_t output,
    uint32_t m, uint32_t add_cores, void* acl_stream);

MF310P_V2_API int mf310p_v2_trial_check_status(mf310p_v2_trial_t* handle,
                                uint64_t* protocol_status);
/* 测试辅助：同步 D2H 读取本 Rank 的 SEND 对称区；该 DDR 视图与 MemFabric AICPU SDMA 引擎可见内容一致。 */
MF310P_V2_API int mf310p_v2_trial_read_send_arena(
    mf310p_v2_trial_t* handle, void* host_dst, uint32_t bytes);
/* 测试辅助：分阶段隔离融合流水线。diag=0 为每核单 Chunk MM 直接写 y，不通信；diag=1 为 MM 写 ownC 后执行一次 signal/wait 与 Add，跳过核间 IB 会合（Core0 自身 MM 完成后即可继续）。 */
MF310P_V2_API int mf310p_v2_trial_diag(mf310p_v2_trial_t* handle, uint64_t x,
                         uint64_t w, uint64_t y, uint32_t diag,
                         void* acl_stream);
MF310P_V2_API int mf310p_v2_trial_destroy(mf310p_v2_trial_t* handle);
#ifdef __cplusplus
}
#endif
