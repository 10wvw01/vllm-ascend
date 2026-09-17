/*
 * Adapter for the separately built wgm-dev-310p MemFabric SDK.
 * This file intentionally includes the customized SDK headers; the main
 * vllm_ascend_C extension does not.
 */
#include "memfabric310p_adapter_api.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <smem.h>
#include <smem_shm.h>
#include <smem_shm_aicore_sdma.h>

#include <cstdint>
#include <cstring>
#include <new>

namespace {

/* Implemented by memfabric310p_device.asc and compiled with the customized
 * MemFabric/AscendC toolchain. */
extern "C" int mf310p_device_publish_chunk_async(
    void* workspace,
    uint64_t send_arena,
    uint32_t chunk_idx,
    uint64_t chunk_bytes,
    uint64_t valid_bytes,
    aclrtStream stream);

extern "C" int mf310p_device_launch_reduce_consumer_async(
    uint64_t send_arena,
    uint64_t recv_arena,
    uint64_t arrival_flags,
    uint32_t chunks,
    uint64_t chunk_bytes,
    aclrtStream stream);

constexpr uint64_t kFlagBytes =
    static_cast<uint64_t>(VLLM_ASCEND_MF310P_MAX_CHUNKS) * sizeof(uint64_t);

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

    /* send + recv/final + MemFabric reserved flag region must fit in one
     * physical contribution.  The two arenas use identical offsets on both
     * ranks, which lets submit_wave point directly at peer recv_arena. */
    if (2 * arena_bytes + SMEM_SHM_SDMA_FLAG_REGION_SIZE > local_size ||
        kFlagBytes > SMEM_SHM_SDMA_FLAG_REGION_SIZE) {
        smem_shm_destroy(ctx->shm, 0);
        smem_shm_uninit(0);
        smem_uninit();
        delete ctx;
        return -4;
    }

    void* workspace = smem_shm_sdma_get_workspace(ctx->shm);
    if (workspace == nullptr) {
        smem_shm_destroy(ctx->shm, 0);
        smem_shm_uninit(0);
        smem_uninit();
        delete ctx;
        return -5;
    }

    ctx->layout.own_segment = own_segment;
    ctx->layout.peer_segment = peer_segment;
    ctx->layout.symmetric_size = symmetric_size;
    ctx->layout.send_arena = own_segment;
    ctx->layout.recv_arena = own_segment + arena_bytes;
    ctx->layout.peer_recv_arena = peer_segment + arena_bytes;
    ctx->layout.arrival_flags =
        own_segment + local_size - SMEM_SHM_SDMA_FLAG_REGION_SIZE;
    ctx->layout.peer_arrival_flags =
        peer_segment + local_size - SMEM_SHM_SDMA_FLAG_REGION_SIZE;
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

extern "C" int mf310p_reset_mailbox(mf310p_context_t* ctx)
{
    if (ctx == nullptr) {
        return -1;
    }
    void* workspace = reinterpret_cast<void*>(ctx->layout.sdma_workspace);
    aclError acl_ret = aclrtMemset(
        workspace,
        SMEM_SHM_SDMA_WS_MAILBOX_REGION_SIZE,
        0,
        SMEM_SHM_SDMA_WS_MAILBOX_REGION_SIZE);
    if (acl_ret != ACL_SUCCESS) {
        return static_cast<int>(acl_ret);
    }

    /* Flags are reused only after the previous wave has joined. Clearing them
     * here gives every wave a fresh non-zero arrival transition. */
    acl_ret = aclrtMemset(
        reinterpret_cast<void*>(ctx->layout.arrival_flags),
        kFlagBytes,
        0,
        kFlagBytes);
    return acl_ret == ACL_SUCCESS ? 0 : static_cast<int>(acl_ret);
}

extern "C" int mf310p_submit_wave(mf310p_context_t* ctx, uint32_t chunks)
{
    if (ctx == nullptr || chunks == 0 || chunks > ctx->layout.max_chunks) {
        return -1;
    }
    return smem_shm_sdma_submit(
        ctx->shm,
        reinterpret_cast<void*>(ctx->layout.peer_recv_arena),
        reinterpret_cast<void*>(ctx->layout.peer_arrival_flags),
        chunks);
}

extern "C" int mf310p_wait_wave(mf310p_context_t* ctx)
{
    return ctx == nullptr ? -1 : smem_shm_sdma_wait(ctx->shm);
}

extern "C" int mf310p_get_result(
    mf310p_context_t* ctx,
    uint32_t* main_ret,
    uint32_t* stage,
    uint32_t* sq_head)
{
    if (ctx == nullptr || main_ret == nullptr || stage == nullptr ||
        sq_head == nullptr) {
        return -1;
    }
    return smem_shm_sdma_get_result(ctx->shm, main_ret, stage, sq_head);
}

extern "C" int mf310p_publish_chunk_async(
    mf310p_context_t* ctx,
    uint32_t chunk_idx,
    uint64_t valid_bytes,
    void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr ||
        chunk_idx >= ctx->layout.max_chunks || valid_bytes == 0 ||
        valid_bytes > ctx->layout.chunk_bytes || (valid_bytes % 4) != 0) {
        return -1;
    }
    return mf310p_device_publish_chunk_async(
        reinterpret_cast<void*>(ctx->layout.sdma_workspace),
        ctx->layout.send_arena,
        chunk_idx,
        ctx->layout.chunk_bytes,
        valid_bytes,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mf310p_launch_reduce_consumer_async(
    mf310p_context_t* ctx,
    uint32_t chunks,
    void* acl_stream)
{
    if (ctx == nullptr || acl_stream == nullptr || chunks == 0 ||
        chunks > ctx->layout.max_chunks) {
        return -1;
    }
    return mf310p_device_launch_reduce_consumer_async(
        ctx->layout.send_arena,
        ctx->layout.recv_arena,
        ctx->layout.arrival_flags,
        chunks,
        ctx->layout.chunk_bytes,
        reinterpret_cast<aclrtStream>(acl_stream));
}
