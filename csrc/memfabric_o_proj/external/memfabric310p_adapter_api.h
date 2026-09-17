/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stable ABI between vllm-ascend and the separately built/installed customized
 * MemFabric 310P adapter.  Nothing in this header depends on MemFabric headers.
 *
 * The adapter itself is built against the wgm-dev-310p MemFabric SDK and may
 * change internally when that SDK changes.  vllm-ascend only dlopens this ABI.
 */
#define VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION 1u
#define VLLM_ASCEND_MF310P_MAX_CHUNKS 64u

typedef struct mf310p_context mf310p_context_t;

typedef struct mf310p_layout {
    uint64_t own_segment;
    uint64_t peer_segment;
    uint64_t symmetric_size;

    /* Two non-aliasing arenas in each rank's symmetric segment. */
    uint64_t send_arena;
    uint64_t recv_arena;
    uint64_t peer_recv_arena;

    /* Arrival flags are local; peer_arrival_flags is the SDMA notify target. */
    uint64_t arrival_flags;
    uint64_t peer_arrival_flags;

    /* Device-side AICore/AICPU cooperation workspace owned by MemFabric. */
    uint64_t sdma_workspace;

    uint64_t arena_bytes;
    uint64_t chunk_bytes;
    uint32_t max_chunks;
    uint32_t reserved;
} mf310p_layout_t;

/* Returns VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION. */
uint32_t mf310p_adapter_abi_version(void);

/*
 * Initialize one TP rank. world_size is deliberately fixed to 2 by the caller.
 * local_size is this rank's physical contribution to the symmetric pool.
 */
int mf310p_create(
    int rank,
    int world_size,
    const char* store_url,
    uint64_t local_size,
    uint64_t arena_bytes,
    uint64_t chunk_bytes,
    mf310p_context_t** out_ctx);

int mf310p_destroy(mf310p_context_t* ctx);
int mf310p_get_layout(mf310p_context_t* ctx, mf310p_layout_t* out_layout);

/* Clear mailbox state before reusing a wave. Pool flags use epoch/slot policy. */
int mf310p_reset_mailbox(mf310p_context_t* ctx);

/*
 * Arm the AICPU/SDMA consumer for one wave. The adapter sends local send_arena
 * chunks to the peer's recv_arena and writes the peer arrival flags.
 */
int mf310p_submit_wave(mf310p_context_t* ctx, uint32_t chunks);
int mf310p_wait_wave(mf310p_context_t* ctx);

/* Optional debug/status data from the customized MemFabric orchestration. */
int mf310p_get_result(
    mf310p_context_t* ctx,
    uint32_t* main_ret,
    uint32_t* stage,
    uint32_t* sq_head);

/*
 * Enqueue a tiny AICore producer operation on the supplied ACL stream:
 *   clean(send chunk) -> smem_shm_sdma_notify(mailbox slot, src_gva, words)
 * The stream value is aclrtStream cast to void* to keep this ABI header-free.
 */
int mf310p_publish_chunk_async(
    mf310p_context_t* ctx,
    uint32_t chunk_idx,
    uint64_t valid_bytes,
    void* acl_stream);

/*
 * Enqueue the local reduction consumer on a separate ACL stream. It polls each
 * local arrival flag and performs recv[chunk] += send[chunk] without modifying
 * send while the peer SDMA may still be reading it.
 */
int mf310p_launch_reduce_consumer_async(
    mf310p_context_t* ctx,
    uint32_t chunks,
    void* acl_stream);

#ifdef __cplusplus
}
#endif
