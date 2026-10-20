#pragma once
// Serving 正式路径与调试状态检查共用的上下文布局。
#include <ATen/ATen.h>
#include <acl/acl_rt.h>
#include <mutex>
#ifdef MF310P_G1C_TRACE
#include <atomic>
#include <thread>
#endif
#include <string>
#include <vector>
#include "mf310p_v2_trial_adapter_api.h"
namespace vllm_ascend {
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
#ifdef MF310P_G1C_TRACE
    // 诊断槽必须在 MemFabric 建池前分配，并贯穿整个 Worker 生命周期。
    at::Tensor g1c_trace_storage;
    std::thread g1c_observer;
    std::atomic<bool> g1c_stop{false};
    int g1c_fd = -1;
    int g1c_device = -1;
#endif
};
V2ServeState& v2_serve_state();
#ifdef MF310P_G1C_TRACE
// 仅诊断构建可见，不暴露新的 Torch Op 或模型接口。
void g1c_start_observer(V2ServeState& s);
void g1c_stop_observer(V2ServeState& s);
void g1c_log_submit(V2ServeState& s, uint32_t rows, uint32_t padded,
                    uint64_t weight, uint64_t output, aclrtStream stream);
#endif
} // vllm_ascend 命名空间
