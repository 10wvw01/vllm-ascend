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
// 不透明句柄封装池、设备工作区和 Rank 几何；对外 ABI 不泄漏 V1/V2 内部状态字段。
typedef struct mf310p_v2_trial mf310p_v2_trial_t;

/* 调用方必须先选择 310P3 设备并创建 ACL 上下文。 */
/** @brief 初始化 TP2 对称池和 IB/status 空间，跨 Rank 交换最终几何。
 * @param rank 为 0/1；store_url 跨 Rank rendezvous；chunk_rows 仅 256/512/1024。
 * @param out 输出不透明句柄；失败时清理并置空。@return 0 成功，否则错误码。
 * @why Host 统一执行控制屏障和布局交换，避免调用者使用未刷新 GVA。
 */
MF310P_V2_API int mf310p_v2_trial_create(int rank, const char* store_url,
                          uint32_t chunk_rows, mf310p_v2_trial_t** out);
/* 必须先通过 G0 探针，包含两次 AICore 启动的 IB 事件对复用验证。 */
/** @brief 运行正式 Serving 必经的 IB 同步安全门禁。
 * @param handle 已创建句柄；repetitions 复用轮数；acl_stream 发射流。
 * @return 0 且设置探针通过标记，非 0 时正式 run 拒绝执行。
 */
MF310P_V2_API int mf310p_v2_trial_ib_probe(mf310p_v2_trial_t* handle,
                            uint32_t repetitions, void* acl_stream);
/* x/w/y 均为设备指针，形状依次为 [M,2048]、NZ[2048,2048]、[M,2048]。原型要求 M>0、M 能被 chunk_rows 整除且 M≤8192。两个 Rank 必须按相同的 M、chunk_rows、overlap 和调用顺序执行。接口只负责入队，验证时须同步 Stream 并检查 status。 */
/** @brief 将完整 chunk 倍数的一波融合计算入队到 ACL stream。
 * @param handle 已过 IB 探针的句柄；x,w,y FP16 设备地址，分别为输入/权重/输出。
 * @param m 发射行数（>0、≤8192 且为 chunk_rows 倍数）；overlap 为 0/1。
 * @param acl_stream 执行流；调用方管理 Tensor 生命周期及 Rank 顺序。
 * @return 0=成功入队，错误返回非零；设备完成与协议状态必须另查。
 */
MF310P_V2_API int mf310p_v2_trial_run(mf310p_v2_trial_t* handle, uint64_t x,
                        uint64_t w, uint64_t y, uint32_t m,
                        uint32_t overlap, void* acl_stream);
/** @brief 将异步 Kernel 协议错误码读回 Host。
 * @param handle 上下文；protocol_status 输出 64 位码，0 表示未记录错误。
 * @return 0=读取成功；不替代流同步，更不代表数值必然正确。
 */
MF310P_V2_API int mf310p_v2_trial_check_status(mf310p_v2_trial_t* handle,
                                uint64_t* protocol_status);
/** @brief 释放 V2 上下文拥有的设备区与 MemFabric 池。
 * @param handle 句柄，可空。@return 0；要求使用者先保证在途流完成。
 */
MF310P_V2_API int mf310p_v2_trial_destroy(mf310p_v2_trial_t* handle);
#ifdef __cplusplus
}
#endif
