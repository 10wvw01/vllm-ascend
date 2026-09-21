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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stable ABI between vLLM-Ascend and its internal bridge to the installed
 * wgm-dev-310p MemFabric package.
 *
 * ABI v5 treats MemFabric as a black box.  The bridge uses only public host
 * APIs and the AscendC device library uses only public signal/wait/quiet
 * primitives.  No mailbox/ring/reserved-region state is exposed here.
 */
#define VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION 5u
#define VLLM_ASCEND_MF310P_MAX_CHUNKS 64u

typedef struct mf310p_context mf310p_context_t;

typedef struct mf310p_layout {
    uint64_t pool_base;
    uint64_t own_segment;
    uint64_t peer_segment;
    uint64_t symmetric_size;
    uint64_t local_size;

    /* vLLM-owned application data inside each symmetric rank segment. */
    uint64_t send_arena;
    uint64_t recv_arena;
    uint64_t peer_recv_arena;
    uint64_t ack_slot;
    uint64_t peer_ack_slot;

    uint64_t arena_bytes;
    uint64_t chunk_bytes;
    uint32_t max_chunks;
    uint32_t reserved;
} mf310p_layout_t;

uint32_t mf310p_adapter_abi_version(void);

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

/*
 * Initialization rendezvous.  It is public-API only and is not part of the
 * per-chunk data plane.
 */
int mf310p_control_barrier(mf310p_context_t* ctx);

/*
 * Phase-B fused producer. One AICore block is the communication coordinator
 * and is the only caller of public smem_shm_sdma_signal(); the remaining
 * blocks compute interleaved o_proj chunks and publish vLLM-owned ready flags.
 * This preserves chunk-level MM/SDMA overlap without depending on MemFabric
 * mailbox/ring internals or multi-producer behavior.
 */
int mf310p_direct_producer_async(
    mf310p_context_t* ctx,
    uint64_t x,
    uint64_t w,
    uint32_t m_bucket,
    uint32_t first_chunk,
    uint32_t chunk_count,
    uint32_t a_advance_rows,
    void* acl_stream);

/* Public quiet() followed by public wait() for each peer chunk. */
int mf310p_wait_mails_async(
    mf310p_context_t* ctx,
    uint32_t chunks,
    void* acl_stream);

int mf310p_warmup_producer_async(
    mf310p_context_t* ctx,
    uint32_t m_bucket,
    void* acl_stream);

int mf310p_warmup_waiter_async(mf310p_context_t* ctx, void* acl_stream);

int mf310p_add_async(
    mf310p_context_t* ctx,
    uint64_t out,
    uint64_t elems,
    void* acl_stream);

int mf310p_warmup_add_async(mf310p_context_t* ctx, void* acl_stream);

/*
 * Application-level cross-wave arena credit.  These wrappers are implemented
 * exclusively with public MemFabric signal()/wait() calls; they do not expose
 * or depend on MemFabric mailbox internals.
 */
int mf310p_ack_async(
    mf310p_context_t* ctx,
    uint64_t imm,
    void* acl_stream);

int mf310p_gate_async(
    mf310p_context_t* ctx,
    uint64_t expected_imm,
    void* acl_stream);

int mf310p_warmup_ack_gate_async(mf310p_context_t* ctx, void* acl_stream);

#ifdef __cplusplus
}
#endif
