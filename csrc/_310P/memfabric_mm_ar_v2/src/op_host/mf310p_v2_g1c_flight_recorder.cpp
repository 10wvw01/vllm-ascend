// G1-C 设备阶段飞行记录器：不接触 MemFabric wire、SQ 或生产数据面。
#include "mf310p_v2_trial_internal.h"
#include "mf310p_v2_g1c_trace_layout.h"

#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <new>
#include <string>
#include <thread>
#include <unistd.h>

namespace {
uint64_t nowMs()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

void observe(mf310p_v2_trial* s)
{
    // 独立线程必须绑定当前 Worker 的设备；禁止在此做全设备同步。
    const aclError bind = aclrtSetDevice(static_cast<int32_t>(s->g1c_device_id));
    {
        std::lock_guard<std::mutex> lock(s->g1c_file_mutex);
        std::fprintf(s->g1c_file, "G1C_BIND ms=%llu rank=%u dev=%u rc=%d\n",
                     static_cast<unsigned long long>(nowMs()), s->rank,
                     s->g1c_device_id, static_cast<int>(bind));
        std::fflush(s->g1c_file);
    }
    if (bind != ACL_SUCCESS) return;
    while (!s->g1c_stop.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (s->g1c_stop.load(std::memory_order_acquire)) break;

        // 若 D2H 被运行中计算核阻塞，至少上一条 SNAP 仍已持久化。
        // 独立 PRE_READ 行供测试观察：是否因采样本身卡住。
        {
            std::lock_guard<std::mutex> lock(s->g1c_file_mutex);
            std::fprintf(s->g1c_file, "G1C_PRE_READ ms=%llu seq=%llu\n",
                         static_cast<unsigned long long>(nowMs()),
                         static_cast<unsigned long long>(
                             s->g1c_seq.load(std::memory_order_acquire)));
            std::fflush(s->g1c_file);
        }

        std::array<MF310PG1CTraceSlot, MF310P_G1C_CORES> slots{};
        const aclError ret = aclrtMemcpy(
            slots.data(), sizeof(slots), s->g1c_trace_gm, sizeof(slots),
            ACL_MEMCPY_DEVICE_TO_HOST);
        const uint64_t ms = nowMs();
        std::lock_guard<std::mutex> lock(s->g1c_file_mutex);
        std::fprintf(s->g1c_file, "G1C_SNAP ms=%llu seq_latest=%llu rc=%d",
                     static_cast<unsigned long long>(ms),
                     static_cast<unsigned long long>(
                         s->g1c_seq.load(std::memory_order_acquire)),
                     static_cast<int>(ret));
        if (ret == ACL_SUCCESS) {
            for (uint32_t c = 0; c < MF310P_G1C_CORES; ++c) {
                const auto& row = slots[c];
                std::fprintf(
                    s->g1c_file, " core%u=%08x:%u:%llu:%u:%u:%u",
                    c, row.magic, row.core,
                    static_cast<unsigned long long>(row.seq),
                    row.phase, row.chunk, row.step);
            }
        }
        std::fprintf(s->g1c_file, "\n");
        std::fflush(s->g1c_file);
    }
}
} // 匿名命名空间

int mf310p_g1c_prepare(mf310p_v2_trial* s)
{
    const char* enable = std::getenv("MF310P_G1C_TRACE");
    if (!enable || std::strcmp(enable, "1") != 0) return 0;
    const char* dir = std::getenv("MF310P_G1C_TRACE_DIR");
    if (!dir || dir[0] != '/' || dir[1] == '\0') return -801; // 必须为已存在绝对路径
    char path[4096] = {};
    const int n = std::snprintf(
        path, sizeof(path), "%s/g1c_rank%u_pid%ld.log", dir,
        s->rank, static_cast<long>(getpid()));
    if (n < 0 || static_cast<size_t>(n) >= sizeof(path)) return -802;
    s->g1c_file = std::fopen(path, "wx"); // 防意外覆盖历史证据
    if (!s->g1c_file) return -803;
    if (aclrtGetDevice(reinterpret_cast<int32_t*>(&s->g1c_device_id)) != ACL_SUCCESS)
        return -804;
    const aclError alloc = aclrtMalloc(
        &s->g1c_trace_gm, MF310P_G1C_TRACE_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc != ACL_SUCCESS) return -805;
    if (aclrtMemset(s->g1c_trace_gm, MF310P_G1C_TRACE_BYTES,
                    0, MF310P_G1C_TRACE_BYTES) != ACL_SUCCESS)
        return -806;
    std::fprintf(s->g1c_file, "G1C_PREPARE ms=%llu rank=%u dev=%u pid=%ld "
                 "trace_ptr=%p bytes=%llu\n",
                 static_cast<unsigned long long>(nowMs()), s->rank,
                 s->g1c_device_id, static_cast<long>(getpid()), s->g1c_trace_gm,
                 static_cast<unsigned long long>(MF310P_G1C_TRACE_BYTES));
    std::fflush(s->g1c_file);
    return 0;
}

int mf310p_g1c_start(mf310p_v2_trial* s)
{
    if (!s->g1c_file) return 0;
    try {
        s->g1c_monitor = std::thread(observe, s);
        return 0;
    } catch (const std::exception&) {
        return -807;
    }
}

void mf310p_g1c_submit(mf310p_v2_trial* s, uint64_t seq,
                       uint64_t input, uint64_t weight, uint64_t output,
                       uint32_t m, uint32_t overlap)
{
    if (!s->g1c_file) return;
    std::lock_guard<std::mutex> lock(s->g1c_file_mutex);
    std::fprintf(s->g1c_file,
                 "G1C_SUBMIT ms=%llu rank=%u seq=%llu m=%u chunk_rows=%u "
                 "overlap=%u x=0x%llx w=0x%llx y=0x%llx\n",
                 static_cast<unsigned long long>(nowMs()), s->rank,
                 static_cast<unsigned long long>(seq), m, s->chunk_rows,
                 overlap, static_cast<unsigned long long>(input),
                 static_cast<unsigned long long>(weight),
                 static_cast<unsigned long long>(output));
    std::fflush(s->g1c_file);
}

void mf310p_g1c_stop(mf310p_v2_trial* s)
{
    s->g1c_stop.store(true, std::memory_order_release);
    if (s->g1c_monitor.joinable()) s->g1c_monitor.join();
    if (s->g1c_trace_gm) {
        (void)aclrtFree(s->g1c_trace_gm);
        s->g1c_trace_gm = nullptr;
    }
    if (s->g1c_file) {
        std::fprintf(s->g1c_file, "G1C_STOP ms=%llu\n",
                     static_cast<unsigned long long>(nowMs()));
        std::fflush(s->g1c_file);
        std::fclose(s->g1c_file);
        s->g1c_file = nullptr;
    }
}
