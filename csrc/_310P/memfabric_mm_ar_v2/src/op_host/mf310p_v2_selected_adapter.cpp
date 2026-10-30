// SPDX-License-Identifier: Apache-2.0
// Fixed-shape entry for performance-selected E1 and E4-T8. Device sources and
// the legacy D4 context layout remain unchanged; no experimental kind argument.
#include "mf310p_v2_selected_api.h"
#include "mf310p_v2_trial_internal.h"
#include <array>
#include <new>

extern "C" int mf310p_v2_dcci_entire_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_e4_group2048_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);

struct mf310p_v2_selected {
    mf310p_v2_trial_t* pool = nullptr;
    mf310p_v2_selected_plan_t plan{};
    aclrtStream stream = nullptr;
    bool probed = false;
    bool poisoned = false;
    bool symbol_warmed = false;
    bool real_launched = false;
    std::array<uint64_t, 64> warmed_outputs{};
    uint32_t next_output = 0;
};

namespace {
int fail(mf310p_v2_selected_t* s, int rc)
{
    if (rc) s->poisoned = true;
    return rc;
}

int invoke(mf310p_v2_selected_t* s, uint64_t x, uint64_t w, uint64_t y,
           uint32_t enable)
{
    const auto* pool = s->pool;
    const auto& l = pool->layout;
    const bool rank0 = pool->rank == 0;
    const uint64_t own = rank0 ? l.send_arena : l.recv_arena;
    const uint64_t peer = rank0 ? l.recv_arena : l.send_arena;
    const uint64_t dst = rank0 ? l.peer_segment : l.peer_recv_arena;
    const uint64_t expected = rank0 ? l.expected_recv_base
                                    : l.expected_recv_base - l.arena_bytes;
    const bool e1 = s->plan.version == MF310P_V2_SELECTED_E1;
    const auto launch = e1 ? mf310p_v2_dcci_entire_launch_async
                           : mf310p_v2_e4_group2048_launch_async;
    // Historical E1 overlap=1; E4 variant=2 means AddTile8192, not kind=6.
    const uint32_t mode = e1 ? 1u : 2u;
    return launch(l.pool_base, x, w, y, own, peer, dst, expected,
                  reinterpret_cast<uint64_t>(pool->ib),
                  reinterpret_cast<uint64_t>(pool->status), l.ack_slot,
                  l.peer_ack_slot, l.expected_credit_dst, s->plan.group_rows,
                  s->plan.m / s->plan.group_rows, mode, enable, s->stream);
}
} // namespace

extern "C" int mf310p_v2_selected_get_plan(
    uint32_t m, mf310p_v2_selected_plan_t* plan)
{
    if (!plan) return MF310P_V2_SELECTED_INVALID;
    *plan = {};
    if (m == 2048) {
        *plan = {MF310P_V2_SELECTED_ABI, m, MF310P_V2_SELECTED_E1,
                 1024, 1024, 2048};
    } else if (m == 4096) {
        *plan = {MF310P_V2_SELECTED_ABI, m, MF310P_V2_SELECTED_E4_T8,
                 1024, 2048, 8192};
    } else {
        return MF310P_V2_SELECTED_UNSUPPORTED;
    }
    return 0;
}

extern "C" int mf310p_v2_selected_create(
    int rank, const char* url, uint32_t m, mf310p_v2_selected_t** out)
{
    if (!out) return MF310P_V2_SELECTED_INVALID;
    *out = nullptr;
    if (!url || !*url || (rank != 0 && rank != 1))
        return MF310P_V2_SELECTED_INVALID;
    mf310p_v2_selected_plan_t plan{};
    const int planned = mf310p_v2_selected_get_plan(m, &plan);
    if (planned) return planned; // Reject before a collective or allocation.
    auto* s = new (std::nothrow) mf310p_v2_selected_t;
    if (!s) return -2;
    s->plan = plan;
    const int rc = mf310p_v2_trial_create(rank, url, 1024, &s->pool);
    if (rc) {
        delete s;
        return rc;
    }
    *out = s;
    return 0;
}

extern "C" int mf310p_v2_selected_ib_probe(
    mf310p_v2_selected_t* s, uint32_t repetitions, void* stream)
{
    if (!s || !stream || repetitions == 0 || repetitions > 100000)
        return MF310P_V2_SELECTED_INVALID;
    if (s->poisoned) return MF310P_V2_SELECTED_POISONED;
    if (s->probed) return MF310P_V2_SELECTED_INVALID;
    s->stream = static_cast<aclrtStream>(stream);
    const int rc = mf310p_v2_trial_ib_probe(s->pool, repetitions, stream);
    if (!rc) s->probed = true;
    return fail(s, rc);
}

extern "C" int mf310p_v2_selected_run(
    mf310p_v2_selected_t* s, uint64_t x, uint64_t w, uint64_t y, void* stream)
{
    if (!s || !stream || !x || !w || !y)
        return MF310P_V2_SELECTED_INVALID;
    if (s->poisoned) return MF310P_V2_SELECTED_POISONED;
    if (!s->probed || !s->pool->ib_probed)
        return MF310P_V2_SELECTED_NOT_READY;
    if (s->stream != static_cast<aclrtStream>(stream))
        return MF310P_V2_SELECTED_STREAM_MISMATCH;
    if (!s->pool->v1_pool || s->pool->chunk_rows != 1024 ||
        s->pool->layout.arena_bytes < static_cast<uint64_t>(s->plan.m) * 4096)
        return MF310P_V2_SELECTED_INVALID;
    if (!s->symbol_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int rc = invoke(s, 0, 0, 0, 0);
            if (rc) return fail(s, rc);
        }
        s->symbol_warmed = true;
    }
    bool known = false;
    for (const auto address : s->warmed_outputs) {
        if (address == y) { known = true; break; }
    }
    // Keep the tested first-write guard. No environment bypass or per-call
    // steady-state synchronization; warmed outputs must keep their storage.
    if (!s->real_launched || !known) {
        const int rc = static_cast<int>(aclrtSynchronizeDevice());
        if (rc) return fail(s, rc);
        if (!known) {
            s->warmed_outputs[s->next_output] = y;
            s->next_output = (s->next_output + 1) % s->warmed_outputs.size();
        }
    }
    const int clear = static_cast<int>(aclrtMemsetAsync(
        s->pool->status, kStatusSize, 0, kStatusSize, s->stream));
    if (clear) return fail(s, clear);
    const int rc = invoke(s, x, w, y, 1);
    if (!rc) s->real_launched = true;
    return fail(s, rc);
}

extern "C" int mf310p_v2_selected_check_status(
    mf310p_v2_selected_t* s, uint64_t* status)
{
    if (!s || !status) return MF310P_V2_SELECTED_INVALID;
    const int rc = mf310p_v2_trial_check_status(s->pool, status);
    if (rc || *status) s->poisoned = true;
    return rc;
}

extern "C" int mf310p_v2_selected_destroy(mf310p_v2_selected_t* s)
{
    if (!s) return 0;
    if (s->stream) {
        const int rc = static_cast<int>(aclrtSynchronizeStream(s->stream));
        if (rc) return fail(s, rc); // Never free a potentially live wave.
    }
    uint64_t status = 0;
    const int checked = mf310p_v2_selected_check_status(s, &status);
    if (checked) return checked;
    if (status) return MF310P_V2_SELECTED_PROTOCOL;
    const int rc = mf310p_v2_trial_destroy(s->pool);
    if (rc) return fail(s, rc);
    delete s; // Warmup state belongs to this lifetime, not a global pointer map.
    return 0;
}
