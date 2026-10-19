/* Standalone V2 trial adapter: public V1 MemFabric pool host API only. */
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
    // Persistent serving outputs must stay allocated for their whole use.
    // The previous single-address guard reissued a 0..15s device sync for
    // every round-robin slot even if that address had already been warmed.
    // Keep a bounded set; pointer identity alone is NOT a lifetime proof.
    uint64_t warmed_y = 0;
    uint64_t trial_warmed_ys[64] = {};
    uint32_t trial_warmed_y_next = 0;
    bool trial_symbol_warmed = false;
    bool trial_symbol_launched = false;
    bool nz_direct_symbol_warmed[5][4] = {};
    bool ar_only_symbol_warmed = false;
    bool ar_only_symbol_launched = false;
    /* Bounded circular sets of first-write-guard-cleared addresses. The
     * benchmark rotates several output/staging buffers; a single
     * last-address cache would re-synchronize (0-15 s) on every rotation.
     * Known limitation (same as T0): address caching cannot detect a
     * freed-and-remapped buffer; callers keep buffers alive or accept a
     * re-warm sync after VMM remap. */
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
    /* T0 BUG #3 guard for the large-M trial kernel, extended by two
     * measured findings (2026-10-17):
     *  1. The first MTE3 output write into a buffer allocated after pool
     *     creation is silently dropped; only a full-device synchronize
     *     heals it (launch-stream syncs and enable=0 symbol warmups do
     *     not - G1 first-call regression, y all zeros with status 0).
     *  2. The FIRST REAL execution of a V2TrialKernel symbol can drop
     *     its MTE3 output writes even into long-warmed buffers (G2
     *     chunk=1024 first call into a y written by 200+ prior calls of
     *     the chunk=256/512 symbols). enable=0 warmups do not prime the
     *     MTE path, so the first real launch of each symbol also needs
     *     the device synchronize.
     * Cost: with the pool active a device synchronize takes a random
     * 0-15 s (it waits for the AICPU SDMA orchestrator epoch), so this
     * fires once per trial handle and once per output-buffer change.
     * Known intermittent risk: a rank-asymmetric slow synchronize can
     * delay one rank's launch past the peer kernel's in-kernel mail
     * wait (observed once as an aicore timeout in a 1000-rep G1 run).
     * The T1 small-M path uses a cheaper stream-sync guard instead. */
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
        // Serving pins registered output allocations for the context's
        // entire lifetime. Trial callers MUST also keep an address alive
        // after it has been marked warmed; free/realloc at the same address
        // is NOT detected by this cache.
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


/* T1: original full NZ weight, exact-M direct ND symmetric output.
 * No NZ weight-slice cache; no input staging or output scatter.
 * mm_cores and add_cores are independent {1,2,4,8} parameters.
 * A single public SDMA mail carries M*2048*2 bytes.
 */
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
    /* Same first-MTE3-write guard as run(): launch-stream synchronize,
     * NOT aclrtSynchronizeDevice (measured 0-15 s with the pool active).
     * Temporary and NOT safe for Graph Capture; callers verify per call. */
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


/* T2 transport-isolation benchmark: native MM is an ordinary framework
 * operator run BEFORE this ABI on the same stream. This routine performs
 * an explicit D2D copy (native partial -> symmetric C) followed by one
 * AR-only device kernel; both stages must be included in comparisons
 * with native MM+HCCL.
 *
 * No model routing and no direct-source SDMA assumption. D2D is mandatory
 * until ordinary torch allocation registration has been separately proved.
 */
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

    /* First-write guard, corrected by 2026-10-18 T2 measurements:
     *  - the FIRST REAL launch of the AR-only symbol drops its MTE3 output
     *    writes (y keeps its prefill, status=0) and a launch-stream
     *    synchronize does NOT cure it, exactly like the T0 kernel (unlike
     *    the T1 kernel, where stream sync was sufficient);
     *  - a fresh D2D staging source page (nativePartial first allocated
     *    after pool creation) stages garbage: 42/42 wrong (status=0) with
     *    the partial guard disabled, 42/42 bit-exact with it enabled.
     * Use the T0-proven full-device synchronize on the first real launch
     * and on every output / staging-source address not yet cleared (bounded
     * 64-entry sets; benchmark rotates 4 y slots + per-M staging buffers).
     * Known cost: with the pool active a device synchronize can take a
     * random 0-15 s (AICPU SDMA orchestrator epoch); steady-state calls
     * with already-cleared addresses pay nothing. NOT Graph-safe,
     * trial-only. */
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

    // Native MM has already been enqueued by the caller on this stream.
    // Staging uses the AICore V2ArStageKernel, NOT aclrtMemcpyAsync:
    // the copy engine's reads are not coherent with same-stream AICore
    // writes on this stack (2026-10-18 T2 measurements: stale staging
    // with varying content, status=0). The stage kernel ends with
    // CleanLines of ownC so the following AICPU SDMA signal observes the
    // staged bytes (T0-proven discipline). Same-stream ordering then
    // guarantees the AR-only kernel starts after staging completes.
    {
        /* The first MTE3-storing launch(es) after an idle window can
         * drop their stores (measured: after the guard's 0-15 s device
         * synchronize or any per-call stream-sync gap, a single stage
         * launch can stage nothing — both ranks then reproduce the
         * PREVIOUS call's result exactly; back-to-back relaunches land).
         * Two sacrificial launches precede the real one: a measured
         * 1/128 case at M=1 showed even a pair occasionally both drop.
         * Same hazard class as T0 BUG#3; enable=0 warmups do not prime
         * the MTE path. V1's launch-twice discipline, hardened; the
         * sacrificial output is overwritten by the real launch on the
         * same stream. */
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

/* Test scaffolding: DDR-view readback of this rank's send arena. */
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
