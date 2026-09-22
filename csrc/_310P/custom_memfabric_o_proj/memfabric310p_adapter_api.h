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
 * ABI v6 treats MemFabric as an opaque transport. The bridge uses only public
 * host APIs and the AscendC device library uses only public signal/wait/quiet
 * primitives. No MemFabric mailbox/ring/reserved-region state is exposed.
 */
#define VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION 6u
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
    uint64_t producer_control;
    uint64_t debug_flags;
    uint64_t expected_credit_dst;
    uint64_t expected_recv_base;
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
int mf310p_debug_protocol_status(
    mf310p_context_t* ctx,
    uint64_t* out_status);

/* One-time host rendezvous during protocol initialization only. */
int mf310p_control_barrier(mf310p_context_t* ctx);

/*
 * One-time per-process GVA geometry exchange over the MemFabric control
 * network. The smem mapping base is chosen per process, so mail dst fields
 * carry peer-space GVAs; receivers must validate against them. Fills
 * layout.expected_credit_dst / layout.expected_recv_base.
 */
int mf310p_exchange_geometry(mf310p_context_t* ctx);

/*
 * Clear vLLM-owned per-wave protocol status. Ready flags are cleared
 * independently before each producer launch.
 */
int mf310p_prepare_wave_async(mf310p_context_t* ctx, void* acl_stream);

/*
 * Seed exactly one fixed application credit into the peer's arrival FIFO and
 * quiet it. Called once after both ranks have created the same MemFabric pool.
 */
int mf310p_init_credit_async(mf310p_context_t* ctx, void* acl_stream);

/*
 * Fused producer: block 0 is the sole MemFabric signal() caller; remaining
 * blocks compute interleaved o_proj chunks and publish vLLM-owned ready flags.
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

/*
 * Public quiet() followed by public wait() for every peer data chunk, with
 * strict public-mail validation (dst/len/imm/status).
 */
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
 * Application-level arena credit. A fixed tag is sufficient because public
 * wait() is FIFO: every wave consumes one credit before overwriting the peer
 * recv arena and publishes one new credit after its local add.
 */
int mf310p_ack_async(mf310p_context_t* ctx, void* acl_stream);
int mf310p_gate_async(mf310p_context_t* ctx, void* acl_stream);
int mf310p_quiet_async(mf310p_context_t* ctx, void* acl_stream);

int mf310p_warmup_protocol_async(mf310p_context_t* ctx, void* acl_stream);

#ifdef __cplusplus
}
#endif
