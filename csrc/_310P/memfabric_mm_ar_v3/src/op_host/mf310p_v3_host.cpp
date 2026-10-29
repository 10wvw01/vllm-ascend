// SPDX-License-Identifier: Apache-2.0
// Standalone public-SMEM adapter; no vLLM/torch or MemFabric-private ABI dependency.
#include "../mf310p_v3_api.h"
#include "../mf310p_v3_protocol.h"
#include <acl/acl_rt.h>
#include <smem.h>
#include <smem_shm.h>
#include <array>
#include <cstdio>
#include <new>
#include <mutex>

extern "C" int mf310p_v3_device_run(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v3_device_raw(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t,
    uint32_t, aclrtStream);
extern "C" int mf310p_v3_device_ib(uint64_t, uint64_t, uint32_t, uint32_t, aclrtStream);

struct mf310p_v3_context {
    smem_shm_t pool = nullptr;
    bool smem = false, shm = false, ibReady = false;
    uint32_t rank = 0, packet = 0, epoch = 0, protocol = 31;
    int32_t device = -1;
    uint64_t gva = 0, work = 0, peer = 0, expectedWork = 0, expectedCtrl = 0;
    void *ib = nullptr, *status = nullptr, *probe = nullptr;
    aclrtStream stream = nullptr;
    std::array<bool, 9> warmed{};
    std::array<uint64_t, 64> outputs{};
    uint32_t outputCount = 0;
    std::mutex leasesMutex;
    std::array<uint32_t, 2> leases{};
    std::array<uint32_t, 2> generations{}, slotRows{}, slotGroups{}, slotModes{};
    bool poolMode = false, legacyRun = false;
    std::array<bool, 16> poolWarmed{};
};

namespace {
using namespace mfv3;
int Error(const char* op, int rc)
{
    if (rc) std::fprintf(stderr, "[mf310p-v3] %s failed rc=%d\n", op, rc);
    return rc;
}
int BindStream(mf310p_v3_context* s, void* raw)
{
    if (!s || !raw) return Error("null context/stream", -1);
    int32_t device = -1;
    const int rc = aclrtGetDevice(&device);
    if (rc || device != s->device) return Error("owning device mismatch", rc ? rc : -2);
    auto stream = static_cast<aclrtStream>(raw);
    if (s->stream && s->stream != stream) return Error("single-stream contract", -3);
    s->stream = stream;
    return 0;
}
int Launch(mf310p_v3_context* s, uint64_t x, uint64_t w, uint64_t y, uint32_t rows,
           uint32_t group, uint32_t pipeline, uint32_t mode, uint32_t enabled)
{
    return mf310p_v3_device_run(s->gva, x, w, y, s->work, s->peer,
        reinterpret_cast<uint64_t>(s->ib), reinterpret_cast<uint64_t>(s->status),
        s->work + kCtrlSrcOff, s->peer + kCtrlDstOff, s->expectedWork, s->expectedCtrl,
        rows, group, s->rank, s->epoch, s->packet, pipeline, mode, enabled, s->stream);
}
int LaunchRaw(mf310p_v3_context* s, uint64_t src, uint64_t seed, uint64_t y,
              uint32_t bytes, uint32_t enabled)
{
    return mf310p_v3_device_raw(s->gva, src, seed, y, s->work, s->work + kArenaBytes, s->peer,
        reinterpret_cast<uint64_t>(s->ib), reinterpret_cast<uint64_t>(s->status),
        s->work + kCtrlSrcOff, s->peer + kCtrlDstOff, s->expectedWork, s->expectedCtrl,
        bytes, s->epoch, enabled, s->stream);
}
int FirstWrite(mf310p_v3_context* s, uint32_t symbol, uint32_t group, uint32_t mode, uint64_t y)
{
    bool outputKnown = false;
    for (uint32_t i = 0; i < s->outputCount; ++i) outputKnown |= s->outputs[i] == y;
    if (s->warmed[symbol] && outputKnown) return 0;
    if (!outputKnown && s->outputCount == s->outputs.size()) return Error("output lifetime slots exhausted", -4);
    for (int i = 0; i < 2; ++i) {
        const int rc = mode == RAW ? LaunchRaw(s, 0, 0, 0, 4096, 0)
                                  : Launch(s, 0, 0, 0, group, group, 0, mode, 0);
        if (rc) return Error("no-op symbol prewarm", rc);
    }
    // Keep 310P MTE3 cold-first-write protection. Never in the hot measured path.
    int rc = aclrtSynchronizeDevice();
    if (!rc) rc = smem_shm_control_barrier(s->pool);
    if (rc) return Error("first-write device sync/barrier", rc);
    s->warmed[symbol] = true;
    if (!outputKnown) s->outputs[s->outputCount++] = y;
    return 0;
}
int NextEpoch(mf310p_v3_context* s)
{
    if (s->epoch == kMaxEpoch) return Error("40-bit epoch exhausted; drain and recreate BOTH ranks", -5);
    ++s->epoch;
    return Error("status clear", aclrtMemsetAsync(s->status, 64, 0, 64, s->stream));
}
int Geometry(mf310p_v3_context* s)
{
    const uint64_t symm = smem_shm_get_symmetric_size(s->pool);
    struct Info { uint64_t base, symm; uint32_t abi, packet, protocol; };
    const Info mine{s->gva, symm, kAbi, s->packet, s->protocol};
    std::array<Info, 2> peers{};
    int rc = smem_shm_control_allgather(s->pool, reinterpret_cast<const char*>(&mine), sizeof(mine),
                                      reinterpret_cast<char*>(peers.data()), sizeof(peers));
    if (rc) return Error("geometry allgather", rc);
    if (!symm || (symm & (symm - 1)) || symm < kPoolBytes) return Error("symmetric geometry", -6);
    for (const auto& p : peers) {
        if (p.symm != symm || p.abi != kAbi || p.packet != s->packet || p.protocol != s->protocol) return Error("peer configuration", -7);
    }
    s->work = s->gva + symm * s->rank;
    s->peer = s->gva + symm * (1 - s->rank);
    // Mail addresses are sender-space GVAs, not necessarily our own mapping.
    s->expectedWork = peers[1 - s->rank].base + symm * s->rank;
    s->expectedCtrl = s->expectedWork + kCtrlDstOff;
    return 0;
}
int Initialize(mf310p_v3_context* s, const char* url)
{
    int rc = smem_init(0);
    if (rc) return Error("smem_init", rc);
    s->smem = true;
    smem_shm_config_t config{};
    rc = smem_shm_config_init(&config);
    if (rc) return Error("config_init", rc);
    config.startConfigStoreServer = s->rank == 0;
    rc = smem_shm_init(url, 2, s->rank, static_cast<uint16_t>(s->device), &config);
    if (rc) return Error("smem_shm_init", rc);
    s->shm = true;
    void* base = nullptr;
    s->pool = smem_shm_create(0, 2, s->rank, kPoolBytes, SMEMS_DATA_OP_SDMA, 0, &base);
    if (!s->pool || !base || !smem_shm_sdma_get_workspace(s->pool)) return Error("SDMA pool not ready", -8);
    s->gva = reinterpret_cast<uint64_t>(base);
    rc = Geometry(s);
    if (!rc) rc = aclrtMalloc(&s->ib, 2 * kIbBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!rc) rc = aclrtMalloc(&s->status, 64, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!rc) rc = aclrtMalloc(&s->probe, 8 * 64, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!rc) rc = aclrtMemset(s->ib, 2 * kIbBytes, 0, 2 * kIbBytes);
    if (!rc) rc = aclrtMemset(s->status, 64, 0, 64);
    if (!rc) rc = aclrtMemset(reinterpret_cast<void*>(s->work + kCtrlSrcOff), 128, 0, 128);
    return Error("workspace allocation/init", rc);
}
} // namespace

extern "C" uint32_t mf310p_v3_abi() { return mfv3::kAbi; }
static int Create(int rank, const char* url, uint32_t packet, uint32_t protocol, mf310p_v3_context** out)
{
    if (!out || !url || rank < 0 || rank > 1 || !mfv3::PacketValid(packet)) return Error("create arguments", -1);
    *out = nullptr;
    auto* s = new (std::nothrow) mf310p_v3_context();
    if (!s) return Error("context allocation", -2);
    s->protocol = protocol;
    s->rank = static_cast<uint32_t>(rank);
    s->packet = packet;
    int rc = aclrtGetDevice(&s->device);
    if (!rc && (s->device < 0 || s->device > UINT16_MAX)) rc = -3;
    if (!rc) rc = Initialize(s, url);
    if (rc) {
        mf310p_v3_destroy(s);
        return Error("create", rc);
    }
    *out = s;
    return 0;
}
extern "C" int mf310p_v3_create(int rank, const char* url, uint32_t packet, mf310p_v3_context** out)
{ return Create(rank, url, packet, 31, out); }
extern "C" int mf310p_v32_create(int rank, const char* url, uint32_t packet, mf310p_v3_context** out)
{ return Create(rank, url, packet, 32, out); }
extern "C" int mf310p_v3_ib_probe(mf310p_v3_context* s, uint32_t reps, void* raw)
{
    int rc = BindStream(s, raw);
    if (rc) return rc;
    if (!reps || reps > 100000) return Error("IB repetitions", -1);
    for (int i = 0; i < 2 && !rc; ++i) rc = mf310p_v3_device_ib(
        reinterpret_cast<uint64_t>(s->ib), reinterpret_cast<uint64_t>(s->probe), 0, 0, s->stream);
    if (!rc) rc = aclrtSynchronizeDevice();
    if (!rc) rc = smem_shm_control_barrier(s->pool);
    if (!rc) rc = mf310p_v3_device_ib(reinterpret_cast<uint64_t>(s->ib),
        reinterpret_cast<uint64_t>(s->probe), reps, 1, s->stream);
    if (!rc) rc = aclrtSynchronizeStream(s->stream);
    std::array<uint32_t, 8 * 16> result{};
    if (!rc) rc = aclrtMemcpy(result.data(), sizeof(result), s->probe, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint32_t c = 0; c < 8 && !rc; ++c) if (result[c * 16] != reps) rc = -2;
    s->ibReady = rc == 0;
    return Error("official IB probe", rc);
}
extern "C" int mf310p_v3_run(mf310p_v3_context* s, uint64_t x, uint64_t w, uint64_t y,
    uint32_t rows, uint32_t group, uint32_t pipeline, uint32_t mode, void* raw)
{
    using namespace mfv3;
    int rc = BindStream(s, raw);
    if (rc) return rc;
    if (s->poolMode) return Error("cannot mix V3.1 and V3.2 sessions", -20);
    if (!s->ibReady || !x || !y || x == y || (mode == MATMUL && !w) || mode > PARTIAL ||
        pipeline > 1 || !ShapeValid(rows, group) || PacketCount(group * kRowBytes / 2, s->packet) > 64)
        return Error("run contract", -1);
    s->legacyRun = true;
    const uint32_t index = group == 256 ? 0 : group == 512 ? 1 : group == 1024 ? 2 : 3;
    rc = FirstWrite(s, index + mode * 4, group, mode, y);
    if (!rc) rc = NextEpoch(s);
    if (!rc) rc = Launch(s, x, w, y, rows, group, pipeline, mode, 1);
    return Error("enqueue fused/protocol kernel", rc);
}
extern "C" int mf310p_v3_probe_add(mf310p_v3_context* s, uint64_t src, uint64_t seed, uint64_t y,
                                   uint32_t bytes, void* raw)
{
    int rc = BindStream(s, raw);
    if (rc) return rc;
    if (s->poolMode) return Error("ADD probe must precede pool output", -20);
    if (!s->ibReady || !src || !seed || !y || !mfv3::PacketValid(bytes)) return Error("ADD probe contract", -1);
    rc = FirstWrite(s, 8, 0, mfv3::RAW, y);
    if (!rc) rc = NextEpoch(s);
    if (!rc) rc = LaunchRaw(s, src, seed, y, bytes, 1);
    return Error("enqueue FP16 ADD probe", rc);
}
extern "C" int mf310p_v3_status(mf310p_v3_context* s, uint64_t* status)
{
    if (!s || !status) return Error("status arguments", -1);
    return Error("status D2H", aclrtMemcpy(status, sizeof(*status), s->status, sizeof(*status), ACL_MEMCPY_DEVICE_TO_HOST));
}
// Collective allocation is deliberately outside the kernel. A released Storage
// becomes reusable only after the owning stream drains on BOTH ranks.
extern "C" int mf310p_v32_acquire(mf310p_v3_context* s, uint32_t rows, uint32_t group,
    uint32_t mode, void* raw, uint32_t* slot, uint32_t* generation, uint64_t* address)
{
    int rc = BindStream(s, raw);
    if (rc || !slot || !generation || !address) return rc ? rc : -1;
    if (!s->ibReady || s->protocol != 32 || s->legacyRun || !mfv3::ShapeValid(rows, group) || group == 256 || mode > 1 ||
        mfv3::PacketCount(group * mfv3::kRowBytes / 2, s->packet) > 64)
        return Error("pool acquire contract", -21);
    rc = aclrtSynchronizeStream(s->stream);
    if (rc) return Error("pool reuse stream completion", rc);
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    struct Info { uint32_t freeMask, rows, group, mode, epoch, packet, protocol; };
    uint32_t freeMask = 0;
    for (uint32_t i = 0; i < 2; ++i) if (!s->leases[i] && s->generations[i] != UINT32_MAX) freeMask |= 1u << i;
    Info mine{freeMask, rows, group, mode, s->epoch, s->packet, 32};
    std::array<Info, 2> peers{};
    rc = smem_shm_control_allgather(s->pool, reinterpret_cast<const char*>(&mine), sizeof(mine),
                                  reinterpret_cast<char*>(peers.data()), sizeof(peers));
    if (rc) return Error("slot collective", rc);
    for (const auto& p : peers) {
        if (p.rows != rows || p.group != group || p.mode != mode || p.epoch != s->epoch ||
            p.packet != s->packet || p.protocol != 32) return Error("pool peer configuration", -22);
        freeMask &= p.freeMask;
    }
    if (!freeMask) return Error("pool slots held on either rank; release outputs/aliases", -23);
    *slot = (freeMask & 1) ? 0 : 1;
    s->leases[*slot] = 1;
    s->slotRows[*slot] = rows;
    s->slotGroups[*slot] = group;
    s->slotModes[*slot] = mode;
    *generation = ++s->generations[*slot];
    *address = s->work + uint64_t{*slot} * mfv3::kArenaBytes;
    s->poolMode = true;
    return 0;
}
extern "C" int mf310p_v32_release(mf310p_v3_context* s, uint32_t slot, uint32_t generation)
{
    if (!s || slot >= 2) return -1;
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    if (!s->leases[slot] || s->generations[slot] != generation) return -2;
    s->leases[slot] = 0; // no device call in a Storage destructor
    return 0;
}
extern "C" int mf310p_v32_run(mf310p_v3_context* s, uint64_t x, uint64_t w,
    uint32_t rows, uint32_t group, uint32_t mode, uint32_t slot, uint32_t generation, void* raw)
{
    int rc = BindStream(s, raw);
    if (rc) return rc;
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    if (!s->poolMode || slot >= 2 || s->leases[slot] != 1 || s->generations[slot] != generation ||
        s->slotRows[slot] != rows || s->slotGroups[slot] != group || s->slotModes[slot] != mode ||
        !x || (mode == 0 && !w) || mode > 1 || group == 256 || !mfv3::ShapeValid(rows, group)) return -24;
    const uint64_t off = uint64_t{slot} * mfv3::kArenaBytes;
    const uint64_t y = s->work + off;
    // Reject all pool aliases as inputs: caller must not feed a live slot back
    // into this prototype's Matmul or protocol-only write path.
    if ((x >= s->work && x < s->work + 2 * mfv3::kArenaBytes) ||
        (w >= s->work && w < s->work + 2 * mfv3::kArenaBytes)) return -25;
    const uint32_t index = group == 512 ? 1 : group == 1024 ? 2 : 3;
    // Separate symbol first-write, including each slot's output address.
    const uint32_t warmIndex = slot * 8 + index + mode * 4;
    for (int i = 0; !s->poolWarmed[warmIndex] && i < 2 && !rc; ++i)
        rc = mf310p_v3_device_run(s->gva, 0, 0, y, y, s->peer + off,
            reinterpret_cast<uint64_t>(s->ib), reinterpret_cast<uint64_t>(s->status),
            s->work + mfv3::kCtrlSrcOff, s->peer + mfv3::kCtrlDstOff,
            s->expectedWork + off, s->expectedCtrl, group, group, s->rank, s->epoch,
            s->packet, 2, mode, 0, s->stream);
    if (!s->poolWarmed[warmIndex] && !rc) {
        rc = aclrtSynchronizeDevice();
        if (!rc) rc = smem_shm_control_barrier(s->pool);
        s->poolWarmed[warmIndex] = rc == 0;
    }
    if (!rc) { s->leases[slot] = 2; rc = NextEpoch(s); }
    if (!rc) rc = mf310p_v3_device_run(s->gva, x, w, y, y, s->peer + off,
        reinterpret_cast<uint64_t>(s->ib), reinterpret_cast<uint64_t>(s->status),
        s->work + mfv3::kCtrlSrcOff, s->peer + mfv3::kCtrlDstOff,
        s->expectedWork + off, s->expectedCtrl, rows, group, s->rank, s->epoch,
        s->packet, 2, mode, 1, s->stream);
    return Error("enqueue V3.2 fixed FIFO pool output", rc);
}
extern "C" int mf310p_v3_destroy(mf310p_v3_context* s)
{
    if (!s) return 0;
    for (auto held : s->leases) if (held) return Error("live output Storage prevents close", -26);
    int rc = 0;
    // Caller must finish its stream first. Stop the MemFabric pool before freeing its backing data.
    if (s->pool) rc = smem_shm_destroy(s->pool, 0);
    if (s->shm) smem_shm_uninit(0);
    if (s->smem) smem_uninit();
    for (void* p : {s->ib, s->status, s->probe}) {
        if (p) { const int r = aclrtFree(p); if (!rc) rc = r; }
    }
    delete s;
    return Error("destroy", rc);
}
