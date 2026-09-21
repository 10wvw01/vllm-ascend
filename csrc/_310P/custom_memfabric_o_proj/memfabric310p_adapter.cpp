/*
 * Public-API bridge for wgm-dev-310p MemFabric.
 *
 * MemFabric is treated as an external black box.  This file may use public
 * host headers, but it must not depend on mailbox/ring/reserved-region layout
 * or on AICPU/orchestrator implementation details.
 */
#include "memfabric310p_adapter_api.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <smem.h>
#include <smem_shm.h>

#include <cstdint>
#include <new>

namespace {

constexpr uint64_t kAckSlotBytes = 8ULL;
constexpr uint32_t kOProjWidthElems = 2048;
constexpr uint64_t kProducerReadyStrideBytes = 64ULL;
constexpr uint64_t kProducerControlBytes =
    VLLM_ASCEND_MF310P_MAX_CHUNKS * kProducerReadyStrideBytes;
constexpr uint64_t kProtocolStatusBytes = 64ULL;

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
    uint64_t producer_control,
    uint64_t protocol_status,
    aclrtStream stream);

extern "C" int mf310p_device_wait_mails_async(
    uint64_t pool_base,
    uint64_t recv_arena,
    uint64_t chunk_bytes,
    uint64_t protocol_status,
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
    uint64_t protocol_status,
    uint32_t enable,
    aclrtStream stream);

extern "C" int mf310p_device_gate_async(
    uint64_t pool_base,
    uint64_t ack_slot,
    uint64_t protocol_status,
    uint32_t enable,
    aclrtStream stream);

extern "C" int mf310p_device_quiet_async(
    uint64_t pool_base,
    uint64_t protocol_status,
    uint32_t enable,
    aclrtStream stream);

bool is_valid_producer_bucket(uint32_t m_bucket)
{
    return m_bucket >= 16 && m_bucket <= 4096 &&
           (m_bucket & (m_bucket - 1)) == 0;
}

void destroy_partial(mf310p_context_t* opaque);

} // namespace

struct mf310p_context {
    int rank = -1;
    int world_size = 0;
    uint64_t local_size = 0;
    bool smem_inited = false;
    bool shm_inited = false;
    smem_shm_t shm = nullptr;
    void* producer_control = nullptr;
    void* protocol_status = nullptr;
    int device_id = -1;
    mf310p_layout_t layout{};
};

namespace {

void destroy_partial(mf310p_context_t* opaque)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr) {
        return;
    }
    if (ctx->protocol_status != nullptr) {
        (void)aclrtFree(ctx->protocol_status);
        ctx->protocol_status = nullptr;
    }
    if (ctx->producer_control != nullptr) {
        (void)aclrtFree(ctx->producer_control);
        ctx->producer_control = nullptr;
    }
    if (ctx->shm != nullptr) {
        (void)smem_shm_destroy(ctx->shm, 0);
        ctx->shm = nullptr;
    }
    if (ctx->shm_inited) {
        smem_shm_uninit(0);
        ctx->shm_inited = false;
    }
    if (ctx->smem_inited) {
        smem_uninit();
        ctx->smem_inited = false;
    }
    delete ctx;
}

} // namespace

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
        rank < 0 || rank >= world_size || local_size == 0 ||
        chunk_bytes == 0 || arena_bytes < chunk_bytes ||
        arena_bytes / chunk_bytes < VLLM_ASCEND_MF310P_MAX_CHUNKS) {
        return -1;
    }
    *out_ctx = nullptr;

    const uint64_t app_bytes = 2 * arena_bytes + kAckSlotBytes;
    /* Reject an impossible application layout before asking MemFabric to
     * create the external pool. The remaining suffix stays opaque to vLLM. */
    if (app_bytes >= local_size) {
        return -2;
    }

    auto* ctx = new (std::nothrow) mf310p_context();
    if (ctx == nullptr) {
        return -3;
    }
    ctx->rank = rank;
    ctx->world_size = world_size;
    ctx->local_size = local_size;

    int ret = smem_init(0);
    if (ret != 0) {
        delete ctx;
        return ret;
    }
    ctx->smem_inited = true;

    int32_t device_id = -1;
    const aclError device_ret = aclrtGetDevice(&device_id);
    if (device_ret != ACL_SUCCESS || device_id < 0 ||
        device_id > static_cast<int32_t>(UINT16_MAX)) {
        destroy_partial(ctx);
        return device_ret == ACL_SUCCESS ? -4 : static_cast<int>(device_ret);
    }
    ctx->device_id = device_id;

    smem_shm_config_t cfg;
    ret = smem_shm_config_init(&cfg);
    if (ret != 0) {
        destroy_partial(ctx);
        return ret;
    }
    cfg.startConfigStoreServer = (rank == 0);

    ret = smem_shm_init(
        store_url,
        static_cast<uint32_t>(world_size),
        static_cast<uint32_t>(rank),
        static_cast<uint16_t>(device_id),
        &cfg);
    if (ret != 0) {
        destroy_partial(ctx);
        return ret;
    }
    ctx->shm_inited = true;

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
        destroy_partial(ctx);
        return -7;
    }

    /*
     * Public readiness contract only.  The returned workspace is intentionally
     * not retained or exposed: non-null means the 310P SDMA subsystem is ready.
     */
    if (smem_shm_sdma_get_workspace(ctx->shm) == nullptr) {
        destroy_partial(ctx);
        return -6;
    }

    const uint64_t symmetric_size = smem_shm_get_symmetric_size(ctx->shm);
    if (symmetric_size == 0) {
        destroy_partial(ctx);
        return -5;
    }

    if (app_bytes > symmetric_size) {
        destroy_partial(ctx);
        return -8;
    }

    const uint64_t pool_base = reinterpret_cast<uint64_t>(gva);
    const uint64_t own_segment =
        pool_base + symmetric_size * static_cast<uint64_t>(rank);
    const uint64_t peer_segment =
        pool_base + symmetric_size * static_cast<uint64_t>(1 - rank);

    /*
     * vLLM owns only a compact prefix of each rank's public symmetric segment.
     * It makes no assumptions about how MemFabric uses any other bytes.
     */
    ctx->layout.pool_base = pool_base;
    ctx->layout.own_segment = own_segment;
    ctx->layout.peer_segment = peer_segment;
    ctx->layout.symmetric_size = symmetric_size;
    ctx->layout.local_size = local_size;
    ctx->layout.send_arena = own_segment;
    ctx->layout.recv_arena = own_segment + arena_bytes;
    ctx->layout.peer_recv_arena = peer_segment + arena_bytes;
    ctx->layout.ack_slot = own_segment + 2 * arena_bytes;
    ctx->layout.peer_ack_slot = peer_segment + 2 * arena_bytes;
    ctx->layout.arena_bytes = arena_bytes;
    ctx->layout.chunk_bytes = chunk_bytes;
    ctx->layout.max_chunks = VLLM_ASCEND_MF310P_MAX_CHUNKS;

    /* vLLM-private cross-AICore ready flags. One cache line per chunk avoids
     * false sharing between MM workers and the single communication
     * coordinator. This buffer is unrelated to MemFabric's internal state. */
    ret = aclrtMalloc(
        &ctx->producer_control,
        kProducerControlBytes,
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS || ctx->producer_control == nullptr) {
        destroy_partial(ctx);
        return -9;
    }
    ret = aclrtMalloc(
        &ctx->protocol_status,
        kProtocolStatusBytes,
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS || ctx->protocol_status == nullptr) {
        destroy_partial(ctx);
        return -10;
    }

    *out_ctx = ctx;
    return 0;
}

extern "C" int mf310p_destroy(mf310p_context_t* opaque)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr) {
        return 0;
    }
    int first_ret = 0;
    if (ctx->protocol_status != nullptr) {
        const aclError ret = aclrtFree(ctx->protocol_status);
        if (ret != ACL_SUCCESS && first_ret == 0) {
            first_ret = static_cast<int>(ret);
        }
        ctx->protocol_status = nullptr;
    }
    if (ctx->producer_control != nullptr) {
        const aclError ret = aclrtFree(ctx->producer_control);
        if (ret != ACL_SUCCESS && first_ret == 0) {
            first_ret = static_cast<int>(ret);
        }
        ctx->producer_control = nullptr;
    }
    if (ctx->shm != nullptr) {
        const int ret = smem_shm_destroy(ctx->shm, 0);
        if (ret != 0) {
            first_ret = ret;
        }
        ctx->shm = nullptr;
    }
    if (ctx->shm_inited) {
        smem_shm_uninit(0);
        ctx->shm_inited = false;
    }
    if (ctx->smem_inited) {
        smem_uninit();
        ctx->smem_inited = false;
    }
    delete ctx;
    return first_ret;
}

extern "C" int mf310p_get_layout(
    mf310p_context_t* opaque,
    mf310p_layout_t* out_layout)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || out_layout == nullptr) {
        return -1;
    }
    *out_layout = ctx->layout;
    return 0;
}

extern "C" int mf310p_debug_protocol_status(
    mf310p_context_t* opaque,
    uint64_t* out_status)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        out_status == nullptr) {
        return -1;
    }
    const aclError ret = aclrtMemcpy(
        out_status,
        sizeof(*out_status),
        ctx->protocol_status,
        sizeof(*out_status),
        ACL_MEMCPY_DEVICE_TO_HOST);
    return ret == ACL_SUCCESS ? 0 : static_cast<int>(ret);
}

extern "C" int mf310p_control_barrier(mf310p_context_t* opaque)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->shm == nullptr) {
        return -1;
    }
    return smem_shm_control_barrier(ctx->shm);
}

extern "C" int mf310p_prepare_wave_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        acl_stream == nullptr) {
        return -1;
    }
    const aclError ret = aclrtMemsetAsync(
        ctx->protocol_status,
        kProtocolStatusBytes,
        0,
        kProtocolStatusBytes,
        reinterpret_cast<aclrtStream>(acl_stream));
    return ret == ACL_SUCCESS ? 0 : static_cast<int>(ret);
}

extern "C" int mf310p_init_credit_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        acl_stream == nullptr) {
        return -1;
    }
    const auto stream = reinterpret_cast<aclrtStream>(acl_stream);
    int ret = mf310p_device_ack_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        ctx->layout.peer_ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        1,
        stream);
    if (ret != 0) {
        return ret;
    }
    return mf310p_device_quiet_async(
        ctx->layout.pool_base,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        1,
        stream);
}

extern "C" int mf310p_direct_producer_async(
    mf310p_context_t* opaque,
    uint64_t x,
    uint64_t w,
    uint32_t m_bucket,
    uint32_t first_chunk,
    uint32_t chunk_count,
    uint32_t a_advance_rows,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr || x == 0 || w == 0 ||
        !is_valid_producer_bucket(m_bucket) || chunk_count == 0 ||
        static_cast<uint64_t>(first_chunk) + chunk_count >
            ctx->layout.max_chunks ||
        static_cast<uint64_t>(m_bucket) * kOProjWidthElems *
                sizeof(uint16_t) >
            ctx->layout.chunk_bytes ||
        (chunk_count > 1 && a_advance_rows < m_bucket)) {
        return -1;
    }

    const auto stream = reinterpret_cast<aclrtStream>(acl_stream);
    const aclError clear_ret = aclrtMemsetAsync(
        ctx->producer_control,
        kProducerControlBytes,
        0,
        kProducerControlBytes,
        stream);
    if (clear_ret != ACL_SUCCESS) {
        return static_cast<int>(clear_ret);
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
        reinterpret_cast<uint64_t>(ctx->producer_control),
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        stream);
}

extern "C" int mf310p_wait_mails_async(
    mf310p_context_t* opaque,
    uint32_t chunks,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr || chunks == 0 ||
        chunks > ctx->layout.max_chunks) {
        return -1;
    }
    return mf310p_device_wait_mails_async(
        ctx->layout.pool_base,
        ctx->layout.recv_arena,
        ctx->layout.chunk_bytes,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        chunks,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_producer_async(
    mf310p_context_t* opaque,
    uint32_t m_bucket,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr ||
        !is_valid_producer_bucket(m_bucket) ||
        static_cast<uint64_t>(m_bucket) * kOProjWidthElems *
                sizeof(uint16_t) >
            ctx->layout.chunk_bytes) {
        return -1;
    }
    return mf310p_device_warmup_producer_async(
        ctx->layout.pool_base,
        m_bucket,
        ctx->layout.chunk_bytes,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_waiter_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_warmup_waiter_async(
        ctx->layout.pool_base,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_add_async(
    mf310p_context_t* opaque,
    uint64_t out,
    uint64_t elems,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    constexpr uint32_t kAddAlignElems = 128;
    constexpr uint32_t kAddMaxBlocks = 8;
    if (ctx == nullptr || acl_stream == nullptr || out == 0 ||
        elems == 0 || (elems % kAddAlignElems) != 0 ||
        elems > ctx->layout.max_chunks * ctx->layout.chunk_bytes /
                    sizeof(uint16_t)) {
        return -1;
    }

    const uint64_t rows = elems / kOProjWidthElems;
    const uint32_t block_count =
        rows < kAddMaxBlocks ? static_cast<uint32_t>(rows) : kAddMaxBlocks;
    if (block_count == 0) {
        return -1;
    }

    return mf310p_device_add_async(
        out,
        ctx->layout.send_arena,
        ctx->layout.recv_arena,
        elems,
        block_count,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_add_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_warmup_add_async(
        ctx->layout.send_arena,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_ack_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_ack_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        ctx->layout.peer_ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_gate_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_gate_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_quiet_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        acl_stream == nullptr) {
        return -1;
    }
    return mf310p_device_quiet_async(
        ctx->layout.pool_base,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_warmup_protocol_async(
    mf310p_context_t* opaque,
    void* acl_stream)
{
    auto* ctx = reinterpret_cast<mf310p_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        acl_stream == nullptr) {
        return -1;
    }
    const auto stream = reinterpret_cast<aclrtStream>(acl_stream);
    int ret = mf310p_device_ack_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        ctx->layout.peer_ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        0,
        stream);
    if (ret != 0) {
        return ret;
    }
    ret = mf310p_device_gate_async(
        ctx->layout.pool_base,
        ctx->layout.ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        0,
        stream);
    if (ret != 0) {
        return ret;
    }
    return mf310p_device_quiet_async(
        ctx->layout.pool_base,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        0,
        stream);
}
