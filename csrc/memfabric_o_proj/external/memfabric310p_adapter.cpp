/*
 * Bridge for the installed wgm-dev-310p MemFabric implementation (V5,
 * origin/wgm-dev-310p mailbox-ring epoch API).
 * This file intentionally includes the customized MemFabric headers; the main
 * vllm_ascend_C extension does not.
 */
#include "memfabric310p_adapter_api.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <smem.h>
#include <smem_shm.h>

#include <cstdint>
#include <cstring>
#include <new>

namespace {

/*
 * V5 reserved tail of each rank's physical segment hosting the SDMA mailbox
 * rings (request ring + arrival ring; see smem_shm_sdma_layout.h). The
 * layout header is device-oriented but layout-only; mirror the constant the
 * same way the customized MemFabric host implementation does. CMake verifies
 * the header exists in the selected install as a build-input contract.
 */
constexpr uint64_t kSdmaReservedRegionSize = 48ULL * 1024ULL;

/* P6 Fix C wave-acknowledgement slot size. */
constexpr uint64_t kAckSlotBytes = 8ULL;

/* Implemented by memfabric310p_device.asc and compiled with the customized
 * MemFabric/AscendC toolchain. */
extern "C" int mf310p_device_launch_direct_producer_async(
    uint64_t pool_base,
    uint64_t x,
    uint64_t w,
    uint64_t send_arena,
    uint64_t peer_recv_arena,
    uint32_t m_bucket,
    uint32_t first_chunk,
    uint32_t chunk_count,
    uint32_t a_advance_rows,
    uint64_t chunk_bytes,
    uint64_t first_seq,
    aclrtStream stream);

extern "C" int mf310p_device_wait_mails_async(
    uint64_t pool_base,
    uint32_t chunks,
    aclrtStream stream);

extern "C" int mf310p_device_warmup_producer_async(
    uint64_t pool_base,
    uint32_t m_bucket,
    uint64_t chunk_bytes,
    aclrtStream stream);

extern "C" int mf310p_device_warmup_waiter_async(
    uint64_t pool_base,
    aclrtStream stream);

extern "C" int mf310p_device_add_async(
    uint64_t out,
    uint64_t send_arena,
    uint64_t recv_arena,
    uint64_t elems,
    uint32_t block_count,
    aclrtStream stream);

extern "C" int mf310p_device_warmup_add_async(
    uint64_t arena,
    aclrtStream stream);

extern "C" int mf310p_device_ack_async(
    uint64_t pool_base,
    uint64_t ack_slot,
    uint64_t peer_ack_slot,
    uint64_t imm,
    uint32_t enable,
    aclrtStream stream);

extern "C" int mf310p_device_gate_async(
    uint64_t pool_base,
    uint64_t ack_slot,
    uint64_t expected_imm,
    uint32_t enable,
    aclrtStream stream);

/* o_proj output width shared by the fused producer kernels (FP16 elements).
 * The whole phase-2 path is deliberately o_proj-geometry-specific. */
constexpr uint32_t kOProjWidthElems = 2048;

bool is_valid_producer_bucket(uint32_t m_bucket)
{
    return m_bucket >= 16 && m_bucket <= 4096 && (m_bucket & (m_bucket - 1)) == 0;
}

} // namespace

struct mf310p_context {
    int rank = -1;
    int world_size = 0;
    uint64_t local_size = 0;
    smem_shm_t shm = nullptr;
    mf310p_layout_t layout{};
};

extern "C" uint32_t mf310p_adapter_abi_version(void)
{
    return VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION;
}

extern "C" int mf310p_create(
    int rank,
    int world_size,
    const char* store_url,
    uint64_t local_size,
    uint64_t arena_bytes,
    uint64_t chunk_bytes,
    mf310p_context_t** out_ctx)
{
    if (out_ctx == nullptr || store_url == nullptr || world_size != 2 ||
        rank < 0 || rank >= world_size || chunk_bytes == 0 ||
        arena_bytes < chunk_bytes ||
        arena_bytes / chunk_bytes < VLLM_ASCEND_MF310P_MAX_CHUNKS) {
        return -1;
    }

    auto* ctx = new (std::nothrow) mf310p_context();
    if (ctx == nullptr) {
        return -2;
    }
    ctx->rank = rank;
    ctx->world_size = world_size;
    ctx->local_size = local_size;

    int ret = smem_init(0);
    if (ret != 0) {
        delete ctx;
        return ret;
    }

    smem_shm_config_t cfg;
    ret = smem_shm_config_init(&cfg);
    if (ret != 0) {
        smem_uninit();
        delete ctx;
        return ret;
    }
    cfg.startConfigStoreServer = (rank == 0);

    ret = smem_shm_init(
        store_url,
        static_cast<uint32_t>(world_size),
        static_cast<uint32_t>(rank),
        static_cast<uint16_t>(rank),
        &cfg);
    if (ret != 0) {
        smem_uninit();
        delete ctx;
        return ret;
    }

    void* gva = nullptr;
    ctx->shm = smem_shm_create(
        0,
        static_cast<uint32_t>(world_size),
        static_cast<uint32_t>(rank),
        local_size,
        SMEMS_DATA_OP_SDMA,
        0,
        &gva);
    if (ctx->shm == nullptr || gva == nullptr) {
        smem_shm_uninit(0);
        smem_uninit();
        delete ctx;
        return -3;
    }

    const uint64_t symmetric_size = smem_shm_get_symmetric_size(ctx->shm);
    const uint64_t own_segment =
        reinterpret_cast<uint64_t>(gva) + symmetric_size * rank;
    const uint64_t peer_segment =
        reinterpret_cast<uint64_t>(gva) + symmetric_size * (1 - rank);

    /* send + recv/final + the 8-byte P6 ack slot + the V5 SDMA mailbox
     * reserved tail must fit in one physical contribution. The two arenas
     * use identical offsets on both ranks, which lets the producer signal
     * directly at peer recv_arena and the ack kernel at peer ack_slot. */
    if (2 * arena_bytes + kAckSlotBytes + kSdmaReservedRegionSize >
        local_size) {
        smem_shm_destroy(ctx->shm, 0);
        smem_shm_uninit(0);
        smem_uninit();
        delete ctx;
        return -4;
    }

    /* V5 host-side readiness check: the only data-plane query left. NULL
     * means the AICPU epoch kernel is not ready (e.g. the launch json was
     * not found). */
    void* workspace = smem_shm_sdma_get_workspace(ctx->shm);
    if (workspace == nullptr) {
        smem_shm_destroy(ctx->shm, 0);
        smem_shm_uninit(0);
        smem_uninit();
        delete ctx;
        return -5;
    }

    ctx->layout.pool_base = reinterpret_cast<uint64_t>(gva);
    ctx->layout.own_segment = own_segment;
    ctx->layout.peer_segment = peer_segment;
    ctx->layout.symmetric_size = symmetric_size;
    ctx->layout.local_size = local_size;
    ctx->layout.send_arena = own_segment;
    ctx->layout.recv_arena = own_segment + arena_bytes;
    ctx->layout.peer_recv_arena = peer_segment + arena_bytes;
    ctx->layout.ack_slot = own_segment + 2 * arena_bytes;
    ctx->layout.peer_ack_slot = peer_segment + 2 * arena_bytes;
    ctx->layout.sdma_workspace = reinterpret_cast<uint64_t>(workspace);
    ctx->layout.arena_bytes = arena_bytes;
    ctx->layout.chunk_bytes = chunk_bytes;
    ctx->layout.max_chunks = VLLM_ASCEND_MF310P_MAX_CHUNKS;

    *out_ctx = ctx;
    return 0;
}

extern "C" int mf310p_destroy(mf310p_context_t* ctx)
{
    if (ctx == nullptr) {
        return 0;
    }
    int first_ret = 0;
    const int destroy_ret = smem_shm_destroy(ctx->shm, 0);
    if (destroy_ret != 0) {
        first_ret = destroy_ret;
    }
    smem_shm_uninit(0);
    smem_uninit();
    delete ctx;
    return first_ret;
}

extern "C" int mf310p_get_layout(
    mf310p_context_t* ctx,
    mf310p_layout_t* out_layout)
{
    if (ctx == nullptr || out_layout == nullptr) {
        return -1;
    }
    *out_layout = ctx->layout;
    return 0;
}

extern "C" int mf310p_control_barrier(mf310p_context_t* ctx)
{
    if (ctx == nullptr) {
        return -1;
    }
    return smem_shm_control_barrier(ctx->shm);
}

extern "C" int mf310p_direct_producer_async(
    mf310p_context_t* ctx,
    uint64_t x,
    uint64_t w,
    uint32_t m_bucket,
    uint32_t first_chunk,
    uint32_t chunk_count,
    uint32_t a_advance_rows,
    uint64_t first_seq,
    void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr || x == 0 || w == 0 ||
        !is_valid_producer_bucket(m_bucket) || chunk_count == 0 ||
        static_cast<uint64_t>(first_chunk) + chunk_count >
            ctx->layout.max_chunks ||
        static_cast<uint64_t>(m_bucket) * kOProjWidthElems * sizeof(uint16_t) >
            ctx->layout.chunk_bytes ||
        (chunk_count > 1 && a_advance_rows < m_bucket)) {
        return -1;
    }
    return mf310p_device_launch_direct_producer_async(
        ctx->layout.pool_base,
        x,
        w,
        ctx->layout.send_arena,
        ctx->layout.peer_recv_arena,
        m_bucket,
        first_chunk,
        chunk_count,
        a_advance_rows,
        ctx->layout.chunk_bytes,
        first_seq,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_wait_mails_async(
    mf310p_context_t* ctx,
    uint32_t chunks,
    void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr || chunks == 0 ||
        chunks > ctx->layout.max_chunks) {
        return -1;
    }
    return mf310p_device_wait_mails_async(
        ctx->layout.pool_base,
        chunks,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_producer_async(
    mf310p_context_t* ctx,
    uint32_t m_bucket,
    void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr ||
        !is_valid_producer_bucket(m_bucket) ||
        static_cast<uint64_t>(m_bucket) * kOProjWidthElems * sizeof(uint16_t) >
            ctx->layout.chunk_bytes) {
        return -1;
    }
    return mf310p_device_warmup_producer_async(
        ctx->layout.pool_base,
        m_bucket,
        ctx->layout.chunk_bytes,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_waiter_async(mf310p_context_t* ctx, void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_warmup_waiter_async(
        ctx->layout.pool_base,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_add_async(
    mf310p_context_t* ctx,
    uint64_t out,
    uint64_t elems,
    void* acl_stream)
{
    constexpr uint32_t kAddAlignElems = 128;
    constexpr uint32_t kAddMaxBlocks = 8;
    if (ctx == nullptr || acl_stream == nullptr || out == 0 ||
        elems == 0 || (elems % kAddAlignElems) != 0 ||
        elems > ctx->layout.max_chunks * ctx->layout.chunk_bytes /
                    sizeof(uint16_t)) {
        return -1;
    }
    /* 2048 elems = one o_proj row; each block should own at least one row. */
    const uint64_t rows = elems / kOProjWidthElems;
    uint32_t block_count = kAddMaxBlocks;
    if (rows < kAddMaxBlocks) {
        block_count = static_cast<uint32_t>(rows);
    }
    return mf310p_device_add_async(
        out,
        ctx->layout.send_arena,
        ctx->layout.recv_arena,
        elems,
        block_count,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_add_async(mf310p_context_t* ctx, void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_warmup_add_async(
        ctx->layout.send_arena,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_ack_async(
    mf310p_context_t* ctx,
    uint64_t imm,
    void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_ack_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        ctx->layout.peer_ack_slot,
        imm,
        1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_gate_async(
    mf310p_context_t* ctx,
    uint64_t expected_imm,
    void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_gate_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        expected_imm,
        1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_ack_gate_async(mf310p_context_t* ctx, void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr) {
        return -1;
    }
    const auto stream = reinterpret_cast<aclrtStream>(acl_stream);
    int ret = mf310p_device_ack_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        ctx->layout.peer_ack_slot,
        0,
        0,
        stream);
    if (ret != 0) {
        return ret;
    }
    return mf310p_device_gate_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        0,
        0,
        stream);
}
