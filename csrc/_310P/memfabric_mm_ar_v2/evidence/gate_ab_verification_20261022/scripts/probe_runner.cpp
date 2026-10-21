// Standalone runner for the Gate B timing probe. Pure ACL runtime; no
// MemFabric, no HCCL, no serving involvement. Device 0 only, idle die.
#include <acl/acl_rt.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>

extern "C" int mf_timing_probe_async(
    uint64_t out, uint64_t magic, uint64_t launchId, aclrtStream stream);

static int check(int rc, const char* what)
{
    if (rc != 0) { printf("ERR %s rc=%d\n", what, rc); return 1; }
    return 0;
}

int main()
{
    if (check(aclInit(nullptr), "aclInit")) return 1;
    if (check(aclrtSetDevice(0), "SetDevice")) return 1;
    aclrtStream stream = nullptr;
    if (check(aclrtCreateStream(&stream), "CreateStream")) return 1;
    const size_t bytes = 8 * 8 * sizeof(uint64_t);  // 8 cores x 64B
    void* dev = nullptr;
    if (check(aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc")) return 1;
    if (check(aclrtMemset(dev, bytes, 0, bytes), "Memset")) return 1;

    const int kLaunches = 6;
    uint64_t host[kLaunches][64];
    double wall_ms[kLaunches];
    uint64_t magic = 0xC0DE2026ULL;
    for (int i = 0; i < kLaunches; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (check(mf_timing_probe_async(
                reinterpret_cast<uint64_t>(dev), magic, i, stream), "launch")) return 1;
        if (check(aclrtSynchronizeStream(stream), "Sync")) return 1;
        auto t1 = std::chrono::steady_clock::now();
        wall_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (check(aclrtMemcpy(host[i], bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H")) return 1;
        if (i < kLaunches - 1) {
            struct timespec ts { 0, 200 * 1000 * 1000 };  // 200ms between launches
            nanosleep(&ts, nullptr);
        }
    }

    printf("{\"launches\": %d, \"records\": [", kLaunches);
    for (int i = 0; i < kLaunches; ++i) {
        printf("%s{\"launch\": %d, \"wall_ms\": %.3f, \"cores\": [",
               i ? ", " : "", i, wall_ms[i]);
        for (int c = 0; c < 8; ++c) {
            const uint64_t* s = &host[i][c * 8];
            printf("%s{\"core\": %llu, \"c0\": %llu, \"c1\": %llu, \"c2\": %llu, "
                   "\"magic\": %llu, \"acc\": %llu, \"launch_id\": %llu}",
                   c ? ", " : "",
                   (unsigned long long)s[0], (unsigned long long)s[1],
                   (unsigned long long)s[2], (unsigned long long)s[3],
                   (unsigned long long)s[4], (unsigned long long)s[5],
                   (unsigned long long)s[6]);
        }
        printf("]}");
    }
    printf("]}\n");

    aclrtFree(dev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
