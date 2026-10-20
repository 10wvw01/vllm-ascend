#pragma once
// 正式与诊断 Host 共享的上下文内存布局；不要修改公共 C ABI。
#include "mf310p_v2_trial_adapter_api.h"
#include "../../../memfabric_mm_ar/memfabric310p_adapter_api.h"
#include <acl/acl_rt.h>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>
struct mf310p_v2_trial {
    mf310p_context_t* v1_pool = nullptr;
    mf310p_layout_t layout{};
    uint32_t rank = 0;
    // G1-C 按环境变量启用的独立飞行记录器；默认不申请资源/不创建线程。
    // 每 Core 专属 64B GM，最后进度由后台线程直接落到磁盘文件。
    void* g1c_trace_gm = nullptr;
    uint32_t g1c_device_id = 0;
    std::atomic<bool> g1c_stop{false};
    std::atomic<uint64_t> g1c_seq{0};
    std::thread g1c_monitor;
    std::mutex g1c_file_mutex;
    FILE* g1c_file = nullptr;
    uint32_t chunk_rows = 0;
    void* ib = nullptr;
    void* status = nullptr;
    void* probe_result = nullptr;
    bool ib_probed = false;
    // 服务输出缓冲区必须在整个使用期间持续存活。
    // 旧版本只记录上一次输出地址，每次槽位轮换都会
    // 再次触发长达 0～15 秒的设备同步，即使地址已预热。
    // 改为有界地址集合；仅凭地址相同不能保证分配生命周期一致。
    uint64_t warmed_y = 0;
    uint64_t trial_warmed_ys[64] = {};
    uint32_t trial_warmed_y_next = 0;
    bool trial_symbol_warmed = false;
    bool trial_symbol_launched = false;
    bool nz_direct_symbol_warmed[5][4] = {};
    bool ar_only_symbol_warmed = false;
    bool ar_only_symbol_launched = false;
    /* 使用有界循环集合记录完成首写保护的地址。测试会轮换多个输出和暂存缓冲区，单地址缓存导致每轮重复同步。限制：缓存无法识别已释放后又复映射的地址；调用方必须保持缓冲区存活，必要时重新执行预热同步。 */
    uint64_t ar_only_warmed_ys[64] = {};
    uint32_t ar_only_warmed_y_next = 0;
    uint64_t ar_only_warmed_partials[64] = {};
    uint32_t ar_only_warmed_partial_next = 0;

};
namespace {
constexpr uint32_t kWindowRows = 8192;
constexpr uint64_t kPoolSize = 96ULL * 1024 * 1024;
constexpr uint64_t kStatusSize = 64;
constexpr uint64_t kProbeResultBytes = 8 * 64;
} // 公共常量仍采用内部链接，避免生成额外导出符号
