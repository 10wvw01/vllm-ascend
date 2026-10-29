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

extern "C" int mf310p_a3_device_run(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);

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
    bool poolMode = false, legacyRun = false, failed = false;
    std::array<bool, 16> poolWarmed{};
    bool a3Warmed = false;
    std::array<uint32_t, 2> slotComm{};
};

namespace {
using namespace mfv3;
constexpr uint32_t kPlanAProtocol = 3203, kA3Protocol = 3301;
constexpr uint64_t kA3SourceOff = 2 * kArenaBytes, kA3CtrlOff = 3 * kArenaBytes;
enum ControlOp : uint32_t { ACQUIRE_SLOT = 1, PREPARE_WAVE = 2, CLOSE_SESSION = 3 };
struct ControlInfo {
    uint32_t op, protocol, epoch, flags, rows, group, mode, slot, generation;
};
// Identical envelope size for every collective prevents acquire/close/prepare
// mismatches from accidentally being interpreted as another control operation.
// Slot and generation are reconciliation payloads since 3203 (merged wave):
// the speculative acquire path may disagree before prepare converges them.
int Exchange(mf310p_v3_context* s, const ControlInfo& mine, std::array<ControlInfo, 2>& peers, bool strictSlot)
{
    int rc = smem_shm_control_allgather(s->pool, reinterpret_cast<const char*>(&mine), sizeof(mine),
                                      reinterpret_cast<char*>(peers.data()), sizeof(peers));
    if (!rc) for (const auto& p : peers) {
        if (p.op != mine.op || p.protocol != mine.protocol || p.epoch != mine.epoch ||
            p.rows != mine.rows || p.group != mine.group || p.mode != mine.mode ||
        (strictSlot && p.slot != mine.slot)) { rc = -22; break; }
    }
    if (rc) s->failed = true;
    return rc;
}
int Error(const char* op, int rc)
{
    if (rc) std::fprintf(stderr, "[mf310p-v3] %s failed rc=%d\n", op, rc);
    return rc;
}
int BindStream(mf310p_v3_context* s, void* raw)
{
    if (!s || !raw) return Error("null context/stream", -1);
    if (s->failed) return Error("failed session; no further device operations", -27);
    int32_t device = -1;
    const int rc = aclrtGetDevice(&device);
    if (rc || device != s->device) return Error("owning device mismatch", rc ? rc : -2);
    auto stream = static_cast<aclrtStream>(raw);
    if (s->stream && s->stream != stream) return Error("single-stream contract", -3);
    s->stream = stream;
    return 0;
}
int LaunchA3(mf310p_v3_context* s, uint64_t x, uint64_t w, uint32_t rows, uint32_t compute,
             uint32_t comm, uint32_t mode, uint32_t slot, uint32_t admit, uint32_t enabled)
{
    return mf310p_a3_device_run(s->gva, x, w, s->work + slot * kArenaBytes,
        s->work + kA3SourceOff, s->peer + slot * kArenaBytes,
        reinterpret_cast<uint64_t>(s->ib), reinterpret_cast<uint64_t>(s->status),
        s->work + kA3CtrlOff, s->peer + kA3CtrlOff + 64,
        s->expectedWork + slot * kArenaBytes, s->expectedWork + kA3CtrlOff + 64,
        rows, compute, comm, s->rank, s->epoch ? s->epoch : 1, s->packet, mode, slot, admit, enabled, s->stream);
}
bool A3Shape(mf310p_v3_context* s, uint32_t rows, uint32_t compute, uint32_t comm, uint32_t mode)
{
    return s->protocol == kA3Protocol && s->ibReady && s->a3Warmed && mode <= 1 &&
        (compute == 512 || compute == 1024) && (comm == 512 || comm == 1024 || comm == 2048) &&
        ShapeValid(rows, compute) && rows % comm == 0 && PacketCount(comm * kRowBytes, s->packet) <= 64;
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
{ return Create(rank, url, packet, kPlanAProtocol, out); }
extern "C" int mf310p_a3_create(int rank, const char* url, uint32_t packet, mf310p_v3_context** out)
{ return Create(rank, url, packet, kA3Protocol, out); }
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
    if (s->poolMode || s->protocol != 31) return Error("cannot mix V3.1 and V3.2 sessions", -20);
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
    if (s->failed) return Error("failed session", -27);
    if (!s->ibReady || s->protocol != kPlanAProtocol || s->legacyRun || !mfv3::ShapeValid(rows, group) || group == 256 || mode > 1 ||
        mfv3::PacketCount(group * mfv3::kRowBytes / 2, s->packet) > 64)
        return Error("pool acquire contract", -21);
    rc = aclrtSynchronizeStream(s->stream);
    if (rc) { s->failed = true; return Error("pool reuse stream completion", rc); }
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    uint32_t freeMask = 0;
    for (uint32_t i = 0; i < 2; ++i) if (!s->leases[i] && s->generations[i] != UINT32_MAX) freeMask |= 1u << i;
    ControlInfo mine{ACQUIRE_SLOT, kPlanAProtocol, s->epoch, freeMask, rows, group, mode, 0, 0};
    std::array<ControlInfo, 2> peers{};
    rc = Exchange(s, mine, peers, true);
    if (rc) return Error("slot collective/configuration", rc);
    for (const auto& p : peers) freeMask &= p.flags;
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
// Merged-wave speculative acquire: takes the lease and picks the slot without
// any control allgather. Both ranks use the same deterministic rule (lowest
// free index), so picks agree whenever the free sets allow it; prepare is the
// single allgather that reconciles them and falls back classically (-29) when
// asymmetric holds made the picks diverge.
extern "C" int mf310p_v32_wave_acquire(mf310p_v3_context* s, uint32_t rows, uint32_t group,
    uint32_t mode, void* raw, uint32_t* slot, uint32_t* generation, uint64_t* address)
{
    int rc = BindStream(s, raw);
    if (rc || !slot || !generation || !address) return rc ? rc : -1;
    if (s->failed) return Error("failed session", -27);
    if (!s->ibReady || s->protocol != kPlanAProtocol || s->legacyRun || !mfv3::ShapeValid(rows, group) || group == 256 || mode > 1 ||
        mfv3::PacketCount(group * mfv3::kRowBytes / 2, s->packet) > 64)
        return Error("pool acquire contract", -21);
    rc = aclrtSynchronizeStream(s->stream);
    if (rc) { s->failed = true; return Error("pool reuse stream completion", rc); }
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    uint32_t freeMask = 0;
    for (uint32_t i = 0; i < 2; ++i) if (!s->leases[i] && s->generations[i] != UINT32_MAX) freeMask |= 1u << i;
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
// Called on BOTH ranks even if local Tensor wrapping failed. No ADD may
// launch until every rank has an output Storage and a matching live lease.
// Since 3203 this is the single control allgather of a merged wave: ranks
// that speculated different slots (asymmetric free masks) get a recoverable
// -29 and fall back to the classic intersection acquire for this wave.
extern "C" int mf310p_v32_prepare(mf310p_v3_context* s, uint32_t slot, uint32_t generation,
                                   uint32_t ready, void* raw)
{
    int rc = BindStream(s, raw);
    if (rc) return rc;
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    if (s->failed || slot >= 2 || s->protocol != kPlanAProtocol) return -27;
    const bool local = ready && s->leases[slot] == 1 && s->generations[slot] == generation;
    ControlInfo mine{PREPARE_WAVE, kPlanAProtocol, s->epoch, local ? 1u : 0u,
                     s->slotRows[slot], s->slotGroups[slot], s->slotModes[slot], slot, generation};
    std::array<ControlInfo, 2> peers{};
    rc = Exchange(s, mine, peers, false);
    if (rc) { s->failed = true; return Error("wave prepare failed; stop BOTH ranks", rc); }
    for (const auto& p : peers) if (!p.flags) { rc = -27; break; }
    if (rc) { s->failed = true; return Error("wave prepare failed; stop BOTH ranks", rc); }
    if (peers[0].slot != peers[1].slot) {
        // Speculative picks diverged. Keep the speculative lease so the caller's
        // Storage deleter releases it cleanly; the wave restarts classically.
        // Generation stays a local lease token: speculation may burn it
        // asymmetrically, so it is never compared across ranks.
        return Error("slot speculation mismatch; retry with collective intersection", -29);
    }
    s->leases[slot] = 2; // prepared exactly once
    return 0;
}
extern "C" int mf310p_v32_run(mf310p_v3_context* s, uint64_t x, uint64_t w,
    uint32_t rows, uint32_t group, uint32_t mode, uint32_t slot, uint32_t generation, void* raw)
{
    int rc = BindStream(s, raw);
    if (rc) return rc;
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    if (s->failed || !s->poolMode || slot >= 2 || s->leases[slot] != 2 || s->generations[slot] != generation ||
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
    if (!rc) { s->leases[slot] = 3; rc = NextEpoch(s); }
    if (!rc) rc = mf310p_v3_device_run(s->gva, x, w, y, y, s->peer + off,
        reinterpret_cast<uint64_t>(s->ib), reinterpret_cast<uint64_t>(s->status),
        s->work + mfv3::kCtrlSrcOff, s->peer + mfv3::kCtrlDstOff,
        s->expectedWork + off, s->expectedCtrl, rows, group, s->rank, s->epoch,
        s->packet, 2, mode, 1, s->stream);
    if (rc) s->failed = true;
    return Error("enqueue V3.2 fixed FIFO pool output", rc);
}
// Cold-path only. No frame may launch before the symbol/slot prewarm barrier.
extern "C" int mf310p_a3_warm(mf310p_v3_context* s, void* raw)
{
    int rc = BindStream(s, raw);
    if (rc) return rc;
    if (s->protocol != kA3Protocol || !s->ibReady || s->poolMode) return -30;
    if (s->a3Warmed) return 0;
    for (uint32_t slot = 0; slot < 2 && !rc; ++slot)
        for (uint32_t compute : {512u, 1024u}) for (uint32_t mode = 0; mode < 2; ++mode)
            for (uint32_t i = 0; i < 2 && !rc; ++i)
                rc = LaunchA3(s, 0, 0, compute, compute, compute, mode, slot, 0, 0);
    if (!rc) rc = aclrtSynchronizeDevice();
    if (!rc) rc = smem_shm_control_barrier(s->pool);
    if (!rc) s->a3Warmed = true;
    else s->failed = true;
    return Error("A3 cold prewarm", rc);
}
// No ACL synchronization or collective call in reserve or launch's hot path.
extern "C" int mf310p_a3_reserve(mf310p_v3_context* s, uint32_t rows, uint32_t compute,
    uint32_t comm, uint32_t mode, void* raw, uint32_t* slot, uint32_t* generation, uint64_t* address)
{
    int rc = BindStream(s, raw);
    if (rc || !slot || !generation || !address) return rc ? rc : -1;
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    if (!A3Shape(s, rows, compute, comm, mode) || s->epoch == kMaxEpoch) return -30;
    *generation = s->epoch + 1;
    *slot = s->epoch % 2;
    *address = s->work + *slot * kArenaBytes;
    if (s->leases[*slot]) return -23; // caller must enqueue START ready=0, not silently retry
    s->leases[*slot] = 1;
    s->generations[*slot] = *generation;
    s->slotRows[*slot] = rows; s->slotGroups[*slot] = compute;
    s->slotModes[*slot] = mode; s->slotComm[*slot] = comm;
    return 0;
}
extern "C" int mf310p_a3_run(mf310p_v3_context* s, uint64_t x, uint64_t w,
    uint32_t rows, uint32_t compute, uint32_t comm, uint32_t mode,
    uint32_t slot, uint32_t generation, uint32_t admit, void* raw)
{
    int rc = BindStream(s, raw);
    if (rc) return rc;
    std::lock_guard<std::mutex> guard(s->leasesMutex);
    if (!A3Shape(s, rows, compute, comm, mode) || slot > 1 || s->epoch == kMaxEpoch ||
        generation != s->epoch + 1 || slot != s->epoch % 2) return -30;
    const bool lease = s->leases[slot] == 1 && s->generations[slot] == generation &&
        s->slotRows[slot] == rows && s->slotGroups[slot] == compute && s->slotComm[slot] == comm && s->slotModes[slot] == mode;
    const bool alias = (x >= s->work && x < s->work + 3 * kArenaBytes) ||
                       (w >= s->work && w < s->work + 3 * kArenaBytes);
    const uint32_t ready = admit && lease && x && (mode || w) && !alias;
    rc = NextEpoch(s);
    if (!rc) { s->poolMode = true; if (ready) s->leases[slot] = 3;
        rc = LaunchA3(s, x, w, rows, compute, comm, mode, slot, ready, 1); }
    if (rc || !ready) s->failed = true;
    return Error("A3 enqueue", rc);
}
extern "C" int mf310p_v3_destroy(mf310p_v3_context* s)
{
    if (!s) return 0;
    if (s->failed) return Error("failed session: preserve evidence, do not destroy/retry", -27);
    if ((s->protocol == kPlanAProtocol || s->protocol == kA3Protocol) && s->ibReady) {
        std::lock_guard<std::mutex> guard(s->leasesMutex);
        bool held = false;
        for (auto lease : s->leases) held |= lease != 0;
        ControlInfo mine{CLOSE_SESSION, s->protocol, s->epoch, held ? 1u : 0u, 0, 0, 0, 0, 0};
        std::array<ControlInfo, 2> peers{};
        const int rc = Exchange(s, mine, peers, true);
        if (rc) return Error("collective close agreement", rc);
        for (const auto& p : peers) if (p.flags) return Error("live output on either rank prevents close", -26);
    } else for (auto held : s->leases) if (held) return Error("live output Storage prevents close", -26);
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
