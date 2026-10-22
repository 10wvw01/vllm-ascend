#pragma once
// Serving 正式路径与调试状态检查共用的上下文布局。
#include <ATen/ATen.h>
#include <acl/acl_rt.h>
#include <mutex>
#include <string>
#include <vector>
#include "mf310p_v2_trial_adapter_api.h"
namespace vllm_ascend {
/**
 * @brief Worker 级 V2 Serving 状态，使用 mutex 序列化池/输出槽复用。
 * @field handle Host 上下文；rank/chunk_rows/execution_stream 作为建立后固定的几何与流约束。
 * @field poisoned/failure 首次失败后阻止请求继续触碰可能损坏的对称池。
 * @field output_slots 固定生命周期输出 Tensor；input_stage 非对齐尾 M 的常驻输入暂存。
 * @why 返回的 narrow 视图延长 Storage 使用期，不能只按循环下标重用输出。
 */
struct V2ServeState {
    std::mutex mutex;
    mf310p_v2_trial_t* handle = nullptr;
    int rank = -1;
    uint32_t chunk_rows = 0;
    aclrtStream execution_stream = nullptr;
    bool poisoned = false;
    std::string failure;
    // 每个输出槽的内存分配贯穿建池前后整个生命周期。返回的 narrow
    // 视图会额外持有 StorageImpl 引用，模型仍引用输出时
    // 禁止回收该槽，否则会覆盖尚未消费的结果。
    std::vector<at::Tensor> output_slots;
    at::Tensor input_stage;
};
/** @brief 取得 Worker 内唯一 Serving 状态引用。
 * @return V2ServeState&；由调用方使用 mutex 管理并发。
 */
V2ServeState& v2_serve_state();
} // vllm_ascend 命名空间
