/* 独立 V2 试验适配层：仅复用 V1 的 MemFabric 对称内存公共 Host API。 */
#include "mf310p_v2_trial_adapter_api.h"
#include "../../../memfabric_mm_ar/memfabric310p_adapter_api.h"
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdlib>
#include <new>

extern "C" int mf310p_v2_trial_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_ib_probe_async(
    uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_ib_fan_probe_async(
    uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_nz_direct_small_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_ar_stage_launch_async(
    uint64_t, uint64_t, uint64_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_ar_only_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" uint32_t mf310p_v2_trial_ib_workspace_bytes();
extern "C" int mf310p_v2_diag_mm_async(
    uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_diag_v1mm_async(
    uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_diag_noib_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_diag_ib_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_diag_ibdone_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t,
    uint32_t, aclrtStream);
extern "C" int mf310p_v2_diag_ibdone16_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t,
    uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_diag_loop_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t,
    uint32_t, uint32_t, uint32_t, aclrtStream);

struct mf310p_v2_trial {
    mf310p_context_t* v1_pool = nullptr;
    mf310p_layout_t layout{};
    uint32_t rank = 0;
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
uint32_t small_template(uint32_t m)
{
    if (m <= 16) return 16;
    if (m <= 32) return 32;
    if (m <= 64) return 64;
    if (m <= 128) return 128;
    return 256;
}
uint32_t small_ti(uint32_t templ)
{
    if (templ == 16) return 0;
    if (templ == 32) return 1;
    if (templ == 64) return 2;
    if (templ == 128) return 3;
    return 4;
}
uint32_t small_ni(uint32_t cores)
{
    if (cores == 1) return 0;
    if (cores == 2) return 1;
    if (cores == 4) return 2;
    return 3;
}

/* 探针为每核分配 64B 结果槽。dav-2002 缺少跨核 L1 一致性，压缩的每核 4B 写入会在缓存回写时丢失。 */
constexpr uint64_t kProbeResultBytes = 8 * 64;

void free_allocations(mf310p_v2_trial* state)
{
    if (state->ib) aclrtFree(state->ib);
    if (state->status) aclrtFree(state->status);
    if (state->probe_result) aclrtFree(state->probe_result);
    if (state->v1_pool) mf310p_destroy(state->v1_pool);
}
}

extern "C" int mf310p_v2_trial_create(
    int rank, const char* store_url, uint32_t chunk_rows, mf310p_v2_trial_t** out)
{
    if (!out || !store_url || (rank != 0 && rank != 1) ||
        (chunk_rows != 256 && chunk_rows != 512 && chunk_rows != 1024))
        return -1;
    *out = nullptr;
    auto* s = new (std::nothrow) mf310p_v2_trial();
    if (!s) return -2;
    s->rank = static_cast<uint32_t>(rank);
    s->chunk_rows = chunk_rows;
    int err = mf310p_create(rank, 2, store_url, kPoolSize, kWindowRows,
                           chunk_rows, &s->v1_pool);
    if (!err) err = mf310p_control_barrier(s->v1_pool);
    if (!err) err = mf310p_exchange_geometry(s->v1_pool);
    /* exchange_geometry 完成之后才可以读取最新布局；expected_recv_base 和 expected_credit_dst 是交换过程中填入的对端 GVA，遵循 V1 交换后的布局刷新纪律。 */
    if (!err) err = mf310p_get_layout(s->v1_pool, &s->layout);
    if (!err) {
        err = static_cast<int>(aclrtMalloc(
            &s->ib, mf310p_v2_trial_ib_workspace_bytes(),
            ACL_MEM_MALLOC_HUGE_FIRST));
    }
    if (!err) err = static_cast<int>(aclrtMalloc(
        &s->status, kStatusSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!err) err = static_cast<int>(aclrtMalloc(
        &s->probe_result, kProbeResultBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!err) err = static_cast<int>(aclrtMemset(
        s->ib, mf310p_v2_trial_ib_workspace_bytes(), 0,
        mf310p_v2_trial_ib_workspace_bytes()));
    if (!err) err = static_cast<int>(aclrtMemset(
        s->status, kStatusSize, 0, kStatusSize));
    if (!err) err = static_cast<int>(aclrtMemset(
        s->probe_result, kProbeResultBytes, 0, kProbeResultBytes));
    if (!err) err = static_cast<int>(aclrtMemset(
        reinterpret_cast<void*>(s->layout.ack_slot), 8, 0, 8));
    if (err) {
        free_allocations(s);
        delete s;
        return err;
    }
    *out = s;
    return 0;
}

extern "C" int mf310p_v2_trial_ib_probe(
    mf310p_v2_trial_t* s, uint32_t repetitions, void* stream)
{
    if (!s || !stream || repetitions == 0 || repetitions > 100000)
        return -1;
    aclrtStream acl_stream = static_cast<aclrtStream>(stream);
    int err = mf310p_v2_ib_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result),
        repetitions, 1, acl_stream);
    if (!err) err = static_cast<int>(aclrtSynchronizeStream(acl_stream));
    uint32_t slots[8 * 16] = {};  /* 8 个核各占 64B，即每核 16 个 uint32。 */
    if (!err) err = static_cast<int>(aclrtMemcpy(
        slots, sizeof(slots), s->probe_result, sizeof(slots),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (!err && (slots[0] != repetitions || slots[16] != repetitions))
        return -2;
    // 真实 MM Kernel 执行 8→1 就绪、1→7 释放、7→1 完成会合。
    // 启用前需单独对该设备符号进行两次预热。
    if (!err) err = mf310p_v2_ib_fan_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_fan_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result), 0, 0, acl_stream);
    if (!err) err = mf310p_v2_ib_fan_probe_async(
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe_result),
        repetitions, 1, acl_stream);
    if (!err) err = static_cast<int>(aclrtSynchronizeStream(acl_stream));
    if (!err) err = static_cast<int>(aclrtMemcpy(
        slots, sizeof(slots), s->probe_result, sizeof(slots),
        ACL_MEMCPY_DEVICE_TO_HOST));
    bool all_ok = err == 0;
    for (uint32_t c = 0; c < 8; ++c) {
        all_ok = all_ok && slots[16 * c] == repetitions;
    }
    s->ib_probed = all_ok;
    return err ? err : (all_ok ? 0 : -3);
}

extern "C" int mf310p_v2_trial_run(
    mf310p_v2_trial_t* s, uint64_t x, uint64_t w, uint64_t y,
    uint32_t m, uint32_t overlap, void* stream)
{
    if (!s || !s->ib_probed || !stream || !x || !w || !y ||
        !m || m > kWindowRows || m % s->chunk_rows || overlap > 1) {
        return -1;
    }
    auto acl_stream = static_cast<aclrtStream>(stream);
    const auto& l = s->layout;
    const bool rank0 = s->rank == 0;
    // 逻辑对称输出：C0 从偏移 0 开始，C1 从 arena_bytes 开始。
    const uint64_t ownC = rank0 ? l.send_arena : l.recv_arena;
    const uint64_t peerC = rank0 ? l.recv_arena : l.send_arena;
    const uint64_t peerDst = rank0 ? l.peer_segment : l.peer_recv_arena;
    const uint64_t expectedRecvDst =
        rank0 ? l.expected_recv_base : l.expected_recv_base - l.arena_bytes;
    /* T0 BUG#3 首写保护及 2026-10-17 实测补充：建池后新分配缓冲区的首次 MTE3 输出可能静默丢失；只有 aclrtSynchronizeDevice 能缓解，Stream 同步和 enable=0 预热无效。此外，V2TrialKernel 的首次真实发射即使写入长期预热的 y 也可能丢失结果，因此每个符号首次真实执行也需设备同步。建池后全设备同步可能耗时 0～15 秒，双 Rank 偏斜曾引发 AICore 等待超时。T1 小 M 路径使用开销较小的 Stream 级守护。 */
    const bool first_real_symbol_launch = !s->trial_symbol_launched;
    if (!s->trial_symbol_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int rc = mf310p_v2_trial_launch_async(
                l.pool_base, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                s->chunk_rows, 1, 0, 0, acl_stream);
            if (rc != 0) return rc;
        }
        s->trial_symbol_warmed = true;
    }
    bool output_warmed = false;
    for (uint64_t known : s->trial_warmed_ys) {
        if (known == y) {
            output_warmed = true;
            break;
        }
    }
    if (getenv("MF310P_V2_NOWARMUP") == nullptr &&
        (first_real_symbol_launch || !output_warmed)) {
        const int sync_err = static_cast<int>(aclrtSynchronizeDevice());
        if (sync_err) return sync_err;
        // 服务为该上下文的整个生命周期固定输出分配。
        // 试验调用方也必须保证已标记为预热的地址保持存活，
        // 释放后重新分配到相同地址的情况不能
        // 由该地址缓存识别。
        if (!output_warmed) {
            s->trial_warmed_ys[s->trial_warmed_y_next] = y;
            s->trial_warmed_y_next =
                (s->trial_warmed_y_next + 1) % 64;
        }
    }
    s->warmed_y = y;
    s->trial_symbol_launched = true;
    const int clear_err = static_cast<int>(aclrtMemsetAsync(
        s->status, kStatusSize, 0, kStatusSize, acl_stream));
    if (clear_err) return clear_err;
    return mf310p_v2_trial_launch_async(
        l.pool_base, x, w, y, ownC, peerC, peerDst, expectedRecvDst,
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->status),
        l.ack_slot, l.peer_ack_slot, l.expected_credit_dst,
        s->chunk_rows, m / s->chunk_rows, overlap, 1, acl_stream);
}


/* T1：直接使用完整 NZ 权重，将精确 M 行结果写入对称 ND 输出；不切分缓存权重、不暂存输入、不散排输出。MM 核数与 Add 核数独立设置为 1/2/4/8。一次公共 SDMA 消息传输 M*2048*2 字节。 */
extern "C" int mf310p_v2_trial_run_nz_direct_small(
    mf310p_v2_trial_t* s, uint64_t x, uint64_t w, uint64_t y,
    uint32_t m, uint32_t mm_cores, uint32_t add_cores, void* stream)
{
    if (!s || !s->ib_probed || !stream || !x || !w || !y ||
        m == 0 || m > 256 ||
        (mm_cores != 1 && mm_cores != 2 &&
         mm_cores != 4 && mm_cores != 8) ||
        (add_cores != 1 && add_cores != 2 &&
         add_cores != 4 && add_cores != 8))
        return -1;

    const auto acl_stream = static_cast<aclrtStream>(stream);
    const auto& l = s->layout;
    const bool rank0 = s->rank == 0;
    const uint64_t ownC = rank0 ? l.send_arena : l.recv_arena;
    const uint64_t peerC = rank0 ? l.recv_arena : l.send_arena;
    const uint64_t peerDst = rank0 ? l.peer_segment : l.peer_recv_arena;
    const uint64_t expectedRecvDst =
        rank0 ? l.expected_recv_base :
                l.expected_recv_base - l.arena_bytes;
    const uint32_t templ = small_template(m);
    bool& warmed = s->nz_direct_symbol_warmed[small_ti(templ)]
                                               [small_ni(mm_cores)];
    if (!warmed) {
        for (int i = 0; i < 2; ++i) {
            const int rc = mf310p_v2_nz_direct_small_launch_async(
                l.pool_base, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, templ, 0, mm_cores, add_cores, 0, acl_stream);
            if (rc != 0) return rc;
        }
        warmed = true;
    }
    /* 与 run() 使用同类首次 MTE3 写保护，但仅同步启动 Stream，不使用池存活时可能耗时 0～15 秒的 aclrtSynchronizeDevice。临时试验机制，不适用于 Graph 捕获；调用方须逐次验证。 */
    if (getenv("MF310P_V2_NOWARMUP") == nullptr && s->warmed_y != y) {
        const int sync_err = static_cast<int>(aclrtSynchronizeStream(
            acl_stream));
        if (sync_err) return sync_err;
        s->warmed_y = y;
    }
    const int err = static_cast<int>(aclrtMemsetAsync(
        s->status, kStatusSize, 0, kStatusSize, acl_stream));
    if (err) return err;
    return mf310p_v2_nz_direct_small_launch_async(
        l.pool_base, x, w, y, ownC, peerC, peerDst, expectedRecvDst,
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->status),
        l.ack_slot, l.peer_ack_slot, l.expected_credit_dst,
        templ, m, mm_cores, add_cores, 1, acl_stream);
}


/* T2 传输隔离基线：同一 Stream 先运行框架原生 MM，再由该 ABI 将局部结果搬到对称 C，启动独立 AR-only 内核。D2D 与 AR 两阶段都必须计入与 Stock 的性能比较。尚未验证普通 PyTorch Tensor 可直接用作 SDMA 源，因此保留必要暂存，且不接入模型路由。 */
extern "C" int mf310p_v2_trial_run_ar_only(
    mf310p_v2_trial_t* s, uint64_t nativePartial, uint64_t y,
    uint32_t m, uint32_t addCores, void* stream)
{
    if (!s || !s->ib_probed || !stream || !nativePartial || !y ||
        nativePartial == y || m == 0 || m > kWindowRows ||
        (addCores != 1 && addCores != 2 &&
         addCores != 4 && addCores != 8))
        return -1;

    const auto acl_stream = static_cast<aclrtStream>(stream);
    const auto& l = s->layout;
    const bool rank0 = s->rank == 0;
    const uint64_t ownC = rank0 ? l.send_arena : l.recv_arena;
    const uint64_t peerC = rank0 ? l.recv_arena : l.send_arena;
    const uint64_t peerDst = rank0 ? l.peer_segment : l.peer_recv_arena;
    const uint64_t expectedRecvDst =
        rank0 ? l.expected_recv_base :
                l.expected_recv_base - l.arena_bytes;
    constexpr uint64_t kNativeRowBytes = 2048ULL * sizeof(uint16_t);
    const uint64_t bytes = static_cast<uint64_t>(m) * kNativeRowBytes;
    if (bytes > l.arena_bytes) return -1;

    if (!s->ar_only_symbol_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int rc = mf310p_v2_ar_only_launch_async(
                l.pool_base, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, s->chunk_rows, addCores, 0, acl_stream);
            if (rc) return rc;
        }
        for (int i = 0; i < 2; ++i) {
            const int rc = mf310p_v2_ar_stage_launch_async(
                0, 0, 0, 0, acl_stream);
            if (rc) return rc;
        }
        s->ar_only_symbol_warmed = true;
    }

    /* 2026-10-18 T2 实测首写守护：AR-only 符号首次真实执行会丢失 MTE3 输出，Stream 同步无效；建池后新分配的原生 MM 暂存源可能读到垃圾数据，协议 status 仍为零。沿用 T0 已验证的全设备同步，在首次真实运行以及出现未保护的输出/暂存地址时触发；使用 64 项有界地址集合。建池后可能耗时 0～15 秒，稳定地址无重复成本。仅供试验，不兼容 Graph。 */
    const bool partial_guard =
        getenv("MF310P_V2_AR_ONLY_PARTIAL_GUARD") == nullptr;
    bool y_known = false;
    for (uint32_t k = 0; k < 64; ++k) {
        if (s->ar_only_warmed_ys[k] == y) { y_known = true; break; }
    }
    bool partial_known = !partial_guard;
    for (uint32_t k = 0; k < 64 && partial_guard; ++k) {
        if (s->ar_only_warmed_partials[k] == nativePartial) {
            partial_known = true; break;
        }
    }
    if (getenv("MF310P_V2_AR_ONLY_NOWARMUP") == nullptr &&
        (!s->ar_only_symbol_launched || !y_known || !partial_known)) {
        const int err = static_cast<int>(aclrtSynchronizeDevice());
        if (err) return err;
        if (!y_known) {
            s->ar_only_warmed_ys[s->ar_only_warmed_y_next] = y;
            s->ar_only_warmed_y_next =
                (s->ar_only_warmed_y_next + 1) % 64;
        }
        if (partial_guard && !partial_known) {
            s->ar_only_warmed_partials[s->ar_only_warmed_partial_next] =
                nativePartial;
            s->ar_only_warmed_partial_next =
                (s->ar_only_warmed_partial_next + 1) % 64;
        }
    }
    s->ar_only_symbol_launched = true;

    // 原生 MM 已由调用方入队到当前 Stream。
    // 暂存采用 AICore V2ArStageKernel，不使用 aclrtMemcpyAsync：
    // 该平台拷贝引擎的读取与同 Stream AICore
    // 写入并非一致（T2 曾复现旧数据，status=0）。暂存内核结尾
    // 通过 CleanLines 清理 ownC，使后续 AICPU SDMA signal 能看到
    // 新暂存数据，遵循 T0 已验证的规则。同 Stream 顺序
    // 还保证 AR-only 内核在暂存完成后才启动。
    // 因此必须保持上述暂存与通信 Kernel 的串行依赖。
    {
        /* 空闲窗口后的首次 MTE3 写可能被丢弃：设备同步或长 Stream 同步间隔后，单次暂存启动可无写出，导致两个 Rank 复现上一轮结果。连续启动更可靠，故在真实执行之前添加两次牺牲性启动；M=1 曾发现两次仍偶发失败。该行为属于 T0 BUG#3 同类风险，enable=0 预热无效。牺牲性输出会被同 Stream 的真实启动覆盖。 */
        const int sacrificial = mf310p_v2_ar_stage_launch_async(
            ownC, nativePartial, bytes, 1, acl_stream);
        if (sacrificial) return sacrificial;
        const int sacrificial2 = mf310p_v2_ar_stage_launch_async(
            ownC, nativePartial, bytes, 1, acl_stream);
        if (sacrificial2) return sacrificial2;
    }
    const int stage_err = mf310p_v2_ar_stage_launch_async(
        ownC, nativePartial, bytes, 1, acl_stream);
    if (stage_err) return stage_err;
    const auto clear = aclrtMemsetAsync(
        s->status, kStatusSize, 0, kStatusSize, acl_stream);
    if (clear != ACL_SUCCESS) return static_cast<int>(clear);

    return mf310p_v2_ar_only_launch_async(
        l.pool_base, ownC, peerC, y, peerDst, expectedRecvDst,
        reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->status),
        l.ack_slot, l.peer_ack_slot, l.expected_credit_dst,
        m, s->chunk_rows, addCores, 1, acl_stream);
}

extern "C" int mf310p_v2_trial_check_status(
    mf310p_v2_trial_t* s, uint64_t* status)
{
    if (!s || !status) return -1;
    return static_cast<int>(aclrtMemcpy(
        status, sizeof(*status), s->status, sizeof(*status),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

/* 测试辅助：以 DDR 视图读取本 Rank 的发送区。 */
extern "C" int mf310p_v2_trial_read_send_arena(
    mf310p_v2_trial_t* s, void* host_dst, uint32_t bytes)
{
    if (!s || !host_dst || !bytes ||
        bytes > static_cast<uint32_t>(s->layout.arena_bytes))
        return -1;
    const uint64_t ownC = s->rank == 0 ? s->layout.send_arena
                                       : s->layout.recv_arena;
    return static_cast<int>(aclrtMemcpy(
        host_dst, bytes, reinterpret_cast<void*>(ownC), bytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
}

/* 测试辅助：暴露对称内存关键地址供诊断。 */
extern "C" int mf310p_v2_trial_addresses(
    mf310p_v2_trial_t* s, uint64_t* out /* 地址字段：pool_base、send_arena、recv_arena、peer_segment、peer_recv_arena、expected_recv_base、arena_bytes。 */)
{
    if (!s || !out) return -1;
    const mf310p_layout_t& l = s->layout;
    out[0] = l.pool_base;
    out[1] = l.send_arena;
    out[2] = l.recv_arena;
    out[3] = l.peer_segment;
    out[4] = l.peer_recv_arena;
    out[5] = l.expected_recv_base;
    out[6] = l.arena_bytes;
    return 0;
}

/* 测试辅助：返回 IB 工作区和状态缓冲区的设备地址。 */
extern "C" uint64_t mf310p_v2_trial_ib_addr(mf310p_v2_trial_t* s)
{
    return s ? reinterpret_cast<uint64_t>(s->ib) : 0;
}

extern "C" uint64_t mf310p_v2_trial_status_addr(mf310p_v2_trial_t* s)
{
    return s ? reinterpret_cast<uint64_t>(s->status) : 0;
}

/* 测试辅助：读取完整 64B 状态区（8 个字）。 */
extern "C" int mf310p_v2_trial_status_words(
    mf310p_v2_trial_t* s, uint64_t* out8)
{
    if (!s || !out8) return -1;
    return static_cast<int>(aclrtMemcpy(
        out8, 8 * sizeof(uint64_t), s->status, 8 * sizeof(uint64_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

namespace {
/* 计算对称区地址的公共逻辑，供 run() 与 diag() 复用。 */
struct SymmetricAddr {
    uint64_t ownC, ownPeerC, peerDst, expectedRecvDst;
};

SymmetricAddr symmetric_addr(const mf310p_layout_t& l, uint32_t rank)
{
    const bool rank0 = rank == 0;
    SymmetricAddr a;
    a.ownC = rank0 ? l.send_arena : l.recv_arena;
    a.ownPeerC = rank0 ? l.recv_arena : l.send_arena;
    a.peerDst = rank0 ? l.peer_segment : l.peer_recv_arena;
    a.expectedRecvDst =
        rank0 ? l.expected_recv_base : l.expected_recv_base - l.arena_bytes;
    return a;
}
} // 匿名命名空间

extern "C" int mf310p_v2_trial_diag(
    mf310p_v2_trial_t* s, uint64_t x, uint64_t w, uint64_t y,
    uint32_t diag, void* stream_ptr)
{
    if (!s || !s->ib_probed || !stream_ptr || !x || !w || !y || diag > 7)
        return -1;
    auto acl_stream = static_cast<aclrtStream>(stream_ptr);
    const SymmetricAddr a = symmetric_addr(s->layout, s->rank);
    const int clear_err = static_cast<int>(aclrtMemsetAsync(
        s->status, kStatusSize, 0, kStatusSize, acl_stream));
    if (clear_err) return clear_err;
    if (diag == 0 || diag == 2) {
        auto launch = diag == 0 ? mf310p_v2_diag_mm_async
                                : mf310p_v2_diag_v1mm_async;
        for (int i = 0; i < 2; ++i) {
            const int rc = launch(x, w, y, s->chunk_rows, 0, acl_stream);
            if (rc) return rc;
        }
        const int werr = static_cast<int>(aclrtSynchronizeStream(acl_stream));
        if (werr) return werr;
        return launch(x, w, y, s->chunk_rows, 1, acl_stream);
    }
    if (diag == 7) {
        const SymmetricAddr a7 = symmetric_addr(s->layout, s->rank);
        auto l7 = [&](uint32_t en) -> int {
            return mf310p_v2_diag_loop_async(
                s->layout.pool_base, x, w, y, a7.ownC, a7.ownPeerC, a7.peerDst,
                a7.expectedRecvDst, reinterpret_cast<uint64_t>(s->ib),
                reinterpret_cast<uint64_t>(s->status), s->layout.ack_slot,
                s->layout.peer_ack_slot, s->layout.expected_credit_dst,
                s->chunk_rows, 1, 0, en, acl_stream);
        };
        for (int i = 0; i < 2; ++i) {
            const int rc = l7(0);
            if (rc) return rc;
        }
        const int werr7 = static_cast<int>(aclrtSynchronizeStream(acl_stream));
        if (werr7) return werr7;
        return l7(1);
    }
    if (diag == 6) {
        const SymmetricAddr a6 = symmetric_addr(s->layout, s->rank);
        auto l6 = [&](uint32_t en) -> int {
            return mf310p_v2_diag_ibdone16_async(
                s->layout.pool_base, x, w, y, a6.ownC, a6.ownPeerC, a6.peerDst,
                a6.expectedRecvDst, reinterpret_cast<uint64_t>(s->ib),
                reinterpret_cast<uint64_t>(s->status), s->layout.ack_slot,
                s->layout.peer_ack_slot, s->layout.expected_credit_dst,
                s->chunk_rows, 1, 0, en, acl_stream);
        };
        for (int i = 0; i < 2; ++i) {
            const int rc = l6(0);
            if (rc) return rc;
        }
        const int werr6 = static_cast<int>(aclrtSynchronizeStream(acl_stream));
        if (werr6) return werr6;
        return l6(1);
    }
    if (diag == 5) {
        /* 通过正式启动入口执行真正的 V2TrialKernel，使用与 IbDone 诊断相同的单 Chunk 串行形状。 */
        const SymmetricAddr a5 = symmetric_addr(s->layout, s->rank);
        for (int i = 0; i < 2; ++i) {
            const int rc = mf310p_v2_trial_launch_async(
                s->layout.pool_base, x, w, y, a5.ownC, a5.ownPeerC,
                a5.peerDst, a5.expectedRecvDst,
                reinterpret_cast<uint64_t>(s->ib),
                reinterpret_cast<uint64_t>(s->status),
                s->layout.ack_slot, s->layout.peer_ack_slot,
                s->layout.expected_credit_dst, s->chunk_rows, 1, 0, 0,
                acl_stream);
            if (rc) return rc;
        }
        const int werr5 = static_cast<int>(aclrtSynchronizeStream(acl_stream));
        if (werr5) return werr5;
        return mf310p_v2_trial_launch_async(
            s->layout.pool_base, x, w, y, a5.ownC, a5.ownPeerC, a5.peerDst,
            a5.expectedRecvDst, reinterpret_cast<uint64_t>(s->ib),
            reinterpret_cast<uint64_t>(s->status), s->layout.ack_slot,
            s->layout.peer_ack_slot, s->layout.expected_credit_dst,
            s->chunk_rows, 1, 0, 1, acl_stream);
    }
    if (diag == 1 || diag == 3 || diag == 4) {
        auto warm = [&](int rep) -> int {
            if (diag == 1)
                return mf310p_v2_diag_noib_async(
                    s->layout.pool_base, x, w, y, a.ownC, a.ownPeerC,
                    a.peerDst, a.expectedRecvDst,
                    reinterpret_cast<uint64_t>(s->status), s->chunk_rows,
                    rep, acl_stream);
            if (diag == 3)
                return mf310p_v2_diag_ib_async(
                    s->layout.pool_base, x, w, y, a.ownC, a.ownPeerC,
                    a.peerDst, a.expectedRecvDst,
                    reinterpret_cast<uint64_t>(s->ib),
                    reinterpret_cast<uint64_t>(s->status), s->chunk_rows,
                    rep, acl_stream);
            return mf310p_v2_diag_ibdone_async(
                s->layout.pool_base, x, w, y, a.ownC, a.ownPeerC, a.peerDst,
                a.expectedRecvDst, reinterpret_cast<uint64_t>(s->ib),
                reinterpret_cast<uint64_t>(s->status), s->layout.ack_slot,
                s->layout.peer_ack_slot, s->layout.expected_credit_dst,
                s->chunk_rows, rep, acl_stream);
        };
        for (int i = 0; i < 2; ++i) {
            const int rc = warm(0);
            if (rc) return rc;
        }
        const int werr = static_cast<int>(aclrtSynchronizeStream(acl_stream));
        if (werr) return werr;
        return warm(1);
    }
    return -1;
}

extern "C" int mf310p_v2_trial_destroy(mf310p_v2_trial_t* s)
{
    if (!s) return 0;
    free_allocations(s);
    delete s;
    return 0;
}
