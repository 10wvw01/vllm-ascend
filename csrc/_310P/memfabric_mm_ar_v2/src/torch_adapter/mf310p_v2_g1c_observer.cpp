// G1-C Worker 内独立 Host 观察线程：仅读取每核独占 64B GM 诊断槽。
// 不依赖 Kernel 正常退出，采样先落盘；实机须验证 D2H 不等待计算流。
#include "mf310p_v2_serve_internal.h"
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>

namespace vllm_ascend {
namespace {
struct alignas(64) G1CSlot {
    uint64_t words[8] = {};
};
static_assert(sizeof(G1CSlot) == 64, "每核必须独占一条 64B 缓存行");
constexpr uint64_t kMagic = 0x4731435048415345ULL;
uint64_t wall_ms()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
void observe_loop(V2ServeState* s) noexcept
{
    const int device_rc = static_cast<int>(aclrtSetDevice(s->g1c_device));
    dprintf(s->g1c_fd, "{\"type\":\"observer_start\",\"time_ms\":%llu,\"device_rc\":%d}\n",
            static_cast<unsigned long long>(wall_ms()), device_rc);
    fsync(s->g1c_fd);
    if (device_rc != 0) return;

    alignas(64) std::array<G1CSlot, 8> slots{};
    const auto* gm = s->g1c_trace_storage.const_data_ptr();
    while (!s->g1c_stop.load(std::memory_order_acquire)) {
        const auto begin = std::chrono::steady_clock::now();
        const uint64_t t = wall_ms();
        // 先落盘，若同步 D2H 阻塞，至少可识别最后一次发起读取的时间。
        dprintf(s->g1c_fd, "{\"type\":\"poll_begin\",\"time_ms\":%llu}\n",
                static_cast<unsigned long long>(t));
        fsync(s->g1c_fd);
        const aclError rc = aclrtMemcpy(slots.data(), sizeof(slots),
                                       gm, sizeof(slots), ACL_MEMCPY_DEVICE_TO_HOST);
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - begin).count();
        dprintf(s->g1c_fd,
                "{\"type\":\"poll_end\",\"time_ms\":%llu,\"rc\":%d,\"duration_ms\":%lld}\n",
                static_cast<unsigned long long>(wall_ms()),
                static_cast<int>(rc), static_cast<long long>(duration));
        if (rc == ACL_SUCCESS) {
            for (uint32_t c = 0; c < slots.size(); ++c) {
                const auto& v = slots[c].words;
                if (v[0] != kMagic && v[0] != 0) {
                    dprintf(s->g1c_fd,
                            "{\"type\":\"invalid_slot\",\"time_ms\":%llu,\"core\":%u,\"magic\":%llu}\n",
                            static_cast<unsigned long long>(wall_ms()), c,
                            static_cast<unsigned long long>(v[0]));
                    continue;
                }
                dprintf(s->g1c_fd,
                        "{\"type\":\"snap\",\"time_ms\":%llu,\"core\":%u,\"magic\":%llu,"
                        "\"seq\":%llu,\"chunk\":%llu,\"phase\":%llu,\"step\":%llu}\n",
                        static_cast<unsigned long long>(wall_ms()), c,
                        static_cast<unsigned long long>(v[0]),
                        static_cast<unsigned long long>(v[1]),
                        static_cast<unsigned long long>(v[2]),
                        static_cast<unsigned long long>(v[3]),
                        static_cast<unsigned long long>(v[5]));
            }
        }
        fsync(s->g1c_fd);
        // 失败也继续低频尝试；超时期间至少已有 poll_begin 记录。
        for (int i = 0; i < 5 && !s->g1c_stop.load(std::memory_order_acquire); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    dprintf(s->g1c_fd, "{\"type\":\"observer_stop\",\"time_ms\":%llu}\n",
            static_cast<unsigned long long>(wall_ms()));
    fsync(s->g1c_fd);
}
} // 匿名命名空间

void g1c_start_observer(V2ServeState& s)
{
    TORCH_CHECK(s.handle != nullptr && s.g1c_trace_storage.defined(),
                "G1-C trace requires initialized context and preallocated GM");
    TORCH_CHECK(!s.g1c_observer.joinable() && s.g1c_fd == -1,
                "G1-C observer already active");
    const char* dir = std::getenv("MF310P_G1C_TRACE_DIR");
    TORCH_CHECK(dir != nullptr && dir[0] == '/',
                "G1-C requires absolute MF310P_G1C_TRACE_DIR (existing directory)");
    TORCH_CHECK(aclrtGetDevice(&s.g1c_device) == ACL_SUCCESS,
                "G1-C cannot determine worker device");
    char path[4096] = {};
    const int n = std::snprintf(path, sizeof(path), "%s/g1c_dev%d_rank%d_pid%ld.jsonl",
                                dir, s.g1c_device, s.rank, static_cast<long>(getpid()));
    TORCH_CHECK(n > 0 && static_cast<size_t>(n) < sizeof(path),
                "G1-C trace path overflow");
    // O_EXCL 禁止覆盖历史证据；目录由执行包装器预创建。
    const int fd = open(path, O_CREAT | O_EXCL | O_WRONLY | O_APPEND | O_CLOEXEC, 0600);
    TORCH_CHECK(fd >= 0, "G1-C trace log open failed: ", path, " errno=", errno);
    s.g1c_fd = fd;
    s.g1c_stop.store(false, std::memory_order_release);
    dprintf(fd, "{\"type\":\"trace_open\",\"time_ms\":%llu,\"rank\":%d,\"device\":%d}\n",
            static_cast<unsigned long long>(wall_ms()), s.rank, s.g1c_device);
    fsync(fd);
    try {
        s.g1c_observer = std::thread(observe_loop, &s);
    } catch (...) {
        close(fd);
        s.g1c_fd = -1;
        throw;
    }
}
void g1c_log_submit(V2ServeState& s, uint32_t rows, uint32_t padded,
                    uint64_t weight, uint64_t output, aclrtStream stream)
{
    if (s.g1c_fd < 0) return;
    const uint64_t seq = mf310p_v2_trial_g1c_sequence(s.handle);
    int slot = -1;
    for (size_t i = 0; i < s.output_slots.size(); ++i)
        if (reinterpret_cast<uint64_t>(s.output_slots[i].data_ptr()) == output) {
            slot = static_cast<int>(i);
            break;
        }
    dprintf(s.g1c_fd,
            "{\"type\":\"submit\",\"time_ms\":%llu,\"rank\":%d,\"seq\":%llu,"
            "\"rows\":%u,\"padded\":%u,\"chunk_rows\":%u,"
            "\"weight\":%llu,\"output\":%llu,\"slot\":%d,\"stream\":%llu}\n",
            static_cast<unsigned long long>(wall_ms()), s.rank,
            static_cast<unsigned long long>(seq), rows, padded, s.chunk_rows,
            static_cast<unsigned long long>(weight),
            static_cast<unsigned long long>(output), slot,
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)));
    if (seq % 32 == 0) fsync(s.g1c_fd);
}
void g1c_stop_observer(V2ServeState& s)
{
    s.g1c_stop.store(true, std::memory_order_release);
    if (s.g1c_observer.joinable()) s.g1c_observer.join();
    if (s.g1c_fd >= 0) {
        fsync(s.g1c_fd);
        close(s.g1c_fd);
        s.g1c_fd = -1;
    }
}
} // namespace vllm_ascend
