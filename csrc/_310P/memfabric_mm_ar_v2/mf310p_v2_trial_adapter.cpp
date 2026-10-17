/* Standalone V2 trial adapter: public V1 MemFabric pool host API only. */
#include "mf310p_v2_trial_adapter_api.h"
#include "../memfabric_mm_ar/memfabric310p_adapter_api.h"
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
    uint64_t warmed_y = 0;
};

namespace {
constexpr uint32_t kWindowRows = 8192;
constexpr uint64_t kPoolSize = 96ULL * 1024 * 1024;
constexpr uint64_t kStatusSize = 64;
/* Probe results use one 64B slot per core (dav-2002 has no inter-core L1
 * coherence; packed 4B-per-core stores lose writes at L1 writeback). */
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
    /* Layout must be (re)read AFTER exchange_geometry: expected_recv_base /
     * expected_credit_dst carry peer-space GVAs filled only by the exchange
     * (mirrors the V1 runtime's post-exchange geometry refresh). */
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
    uint32_t slots[8 * 16] = {};  /* 8 cores x 64B slot = 16 uint32 each */
    if (!err) err = static_cast<int>(aclrtMemcpy(
        slots, sizeof(slots), s->probe_result, sizeof(slots),
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (!err && (slots[0] != repetitions || slots[16] != repetitions))
        return -2;
    // The real MM kernel uses 8->1 ready, 1->7 release, 7->1 done.
    // Warm this distinct device symbol twice before enabling.
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
    // Symmetric logical output: C0 at offset 0, C1 at offset arena_bytes.
    const uint64_t ownC = rank0 ? l.send_arena : l.recv_arena;
    const uint64_t peerC = rank0 ? l.recv_arena : l.send_arena;
    const uint64_t peerDst = rank0 ? l.peer_segment : l.peer_recv_arena;
    const uint64_t expectedRecvDst =
        rank0 ? l.expected_recv_base : l.expected_recv_base - l.arena_bytes;
    /* Measured dav-2002 hazard: the FIRST full launch of the symbol whose
     * MTE3 (vector-queue) output writes target a buffer allocated after
     * the MemFabric pool was created silently drops those writes - scalar
     * stores, the matmul GM writes and the whole mail protocol still land.
     * Pool memory and pre-pool allocations are immune; a full-device
     * synchronize before the first launch into a given output buffer makes
     * it correct (stream syncs, SDMA writes and extra symbol launches
     * measured: none help). Output buffers change rarely; sync then. */
    if (getenv("MF310P_V2_NOWARMUP") == nullptr && s->warmed_y != y) {
        const int sync_err = static_cast<int>(aclrtSynchronizeDevice());
        if (sync_err) return sync_err;
        s->warmed_y = y;
    }
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

extern "C" int mf310p_v2_trial_check_status(
    mf310p_v2_trial_t* s, uint64_t* status)
{
    if (!s || !status) return -1;
    return static_cast<int>(aclrtMemcpy(
        status, sizeof(*status), s->status, sizeof(*status),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

/* Test scaffolding: expose key symmetric addresses for diagnostics. */
extern "C" int mf310p_v2_trial_addresses(
    mf310p_v2_trial_t* s, uint64_t* out /* pool_base, send_arena, recv_arena,
        peer_segment, peer_recv_arena, expected_recv_base, arena_bytes */)
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

/* Test scaffolding: device addresses of the IB workspace and status buffer. */
extern "C" uint64_t mf310p_v2_trial_ib_addr(mf310p_v2_trial_t* s)
{
    return s ? reinterpret_cast<uint64_t>(s->ib) : 0;
}

extern "C" uint64_t mf310p_v2_trial_status_addr(mf310p_v2_trial_t* s)
{
    return s ? reinterpret_cast<uint64_t>(s->status) : 0;
}

/* Test scaffolding: read the full 64B status region (8 words). */
extern "C" int mf310p_v2_trial_status_words(
    mf310p_v2_trial_t* s, uint64_t* out8)
{
    if (!s || !out8) return -1;
    return static_cast<int>(aclrtMemcpy(
        out8, 8 * sizeof(uint64_t), s->status, 8 * sizeof(uint64_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

namespace {
/* Symmetric-region address computation shared by run() and diag(). */
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
} // namespace

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
        /* Launch the REAL V2TrialKernel through the production launcher
         * with the same one-chunk serial shape the IbDone diag uses. */
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
