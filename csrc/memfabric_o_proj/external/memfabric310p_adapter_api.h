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
 * Stable ABI between vllm-ascend and the bridge built against the installed
 * wgm-dev-310p MemFabric implementation. Nothing in this header depends on
 * MemFabric headers.
 *
 * ABI v3 targets the V5 MemFabric (origin/wgm-dev-310p mailbox-ring epoch
 * API): kernels drive the AICPU orchestrator themselves via
 * smem_shm_sdma_signal_at/wait_at/quiet_at and only need the symmetric pool
 * base (gva), so the V4 workspace/mailbox/arrival-flag plumbing is gone.
 *
 * The bridge may change internally when the customized MemFabric changes;
 * vllm-ascend only depends on this small C ABI.
 */
#define VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION 4u
#define VLLM_ASCEND_MF310P_MAX_CHUNKS 64u

typedef struct mf310p_context mf310p_context_t;

typedef struct mf310p_layout {
    /* Symmetric pool base (gva). Kernels derive their own reserved-region
     * handle from it via smem_shm_sdma_reserved(); host code must not. */
    uint64_t pool_base;

    uint64_t own_segment;
    uint64_t peer_segment;
    uint64_t symmetric_size;
    /* This rank's physical contribution (the reserved tail sits at
     * own_segment + local_size - 48 KiB). Debug tooling uses it. */
    uint64_t local_size;

    /* Two non-aliasing arenas in each rank's symmetric segment. */
    uint64_t send_arena;
    uint64_t recv_arena;
    uint64_t peer_recv_arena;

    /* 8-byte wave-acknowledgement slots after the arenas (P6 Fix C): the
     * peer's ack kernel signals into our slot after its reduced-output add
     * completed; our gate kernel consumes that mail before the next wave's
     * producer may overwrite the peer's recv arena. Content is never read -
     * the arrival mail's imm/dst carry the protocol. */
    uint64_t ack_slot;
    uint64_t peer_ack_slot;

    /* Device-side SDMA mailbox rings owned by the MemFabric epoch kernel.
     * Host-side value is only a readiness marker (non-zero once the epoch
     * kernel is ready); there is no host data-plane access anymore. */
    uint64_t sdma_workspace;

    uint64_t arena_bytes;
    uint64_t chunk_bytes;
    uint32_t max_chunks;
    uint32_t reserved;
} mf310p_layout_t;

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

/*
 * Control-network barrier on the pool (smem_shm_control_barrier). Used at
 * wave boundaries together with a preceding host-side stream sync: both
 * ranks must have finished every arena access of the previous wave
 * (including the reduced-output add that reads recv) before either rank's
 * next-wave signals may overwrite the peer's recv arena. The barrier is
 * per wave, never per chunk. See the V5 migration notes in the
 * development plan.
 */
int mf310p_control_barrier(mf310p_context_t* ctx);

/*
 * Phase-2 direct producer. Enqueue the fused AscendC FP16 matmul
 * (A ND [m_bucket, 2048] x B NZ [2048, 2048] transposed) that writes chunks
 * [first_chunk, first_chunk + chunk_count) of the send arena directly.
 * Per chunk: matmul -> line-clean -> smem_shm_sdma_signal_at toward the
 * peer's recv arena, so the AICPU epoch kernel can move it while the kernel
 * continues with the next chunk.
 *
 * m_bucket selects the static-tiling kernel instantiation (power of two in
 * [16, 4096]) and must satisfy m_bucket * 2048 * 2 <= chunk_bytes (the slot
 * capacity) and m_bucket >= valid rows of every chunk it covers. When
 * chunk_count > 1, a_advance_rows (the x row stride between consecutive
 * chunks, normally tile_m) must be >= m_bucket so chunk reads do not overlap.
 */
int mf310p_direct_producer_async(
    mf310p_context_t* ctx,
    uint64_t x,
    uint64_t w,
    uint32_t m_bucket,
    uint32_t first_chunk,
    uint32_t chunk_count,
    uint32_t a_advance_rows,
    uint64_t first_seq,
    void* acl_stream);

/*
 * Enqueue the waiter kernel on the supplied ACL stream:
 *   smem_shm_sdma_quiet_at (all of this rank's signals landed at the peer)
 *   -> smem_shm_sdma_wait_at x chunks (receive this rank's own mails).
 *
 * Everything enqueued after it on the same stream (e.g. the host-visible
 * reduction of send + recv) may read both arenas safely.
 */
int mf310p_wait_mails_async(
    mf310p_context_t* ctx,
    uint32_t chunks,
    void* acl_stream);

/*
 * Side-effect-free warmup for one M-bucket producer instantiation: enqueues
 * the chunk_count == 0 kernel shape (the kernel returns before touching any
 * memory). On dav-2002 the first launch of each kernel symbol from a freshly
 * loaded .so is a silent no-op, so callers must warm each bucket twice
 * before its first real wave.
 */
int mf310p_warmup_producer_async(
    mf310p_context_t* ctx,
    uint32_t m_bucket,
    void* acl_stream);

/* Same warmup contract for the waiter kernel (chunks == 0 shape). */
int mf310p_warmup_waiter_async(mf310p_context_t* ctx, void* acl_stream);

/*
 * Enqueue the multi-block vector reduce out[0, elems) = send + recv on the
 * supplied ACL stream. Shape-agnostic replacement for at::add_out (which
 * pays a per-output-shape GE compile on first use); correctly-rounded FP16
 * add, bit-exact with the FP32-math reference. elems must be a multiple of
 * 128 and fit one wave (max_chunks * tile_m * 2048).
 */
int mf310p_add_async(
    mf310p_context_t* ctx,
    uint64_t out,
    uint64_t elems,
    void* acl_stream);

/* Same double-launch warmup contract for the add kernel (elems == 0). */
int mf310p_warmup_add_async(mf310p_context_t* ctx, void* acl_stream);

/*
 * P6 Fix C device-side wave rendezvous (replaces the per-wave host join /
 * control barrier, which costs ~150 us steady state with rare ~40 ms TCP
 * spikes):
 *
 *   ack  - after this rank's reduced-output add for wave `imm` completed
 *          (stream-ordered), signal 8 bytes into the peer's ack slot with
 *          imm = wave index.
 *   gate - before this rank's next-wave producer, consume one arrival mail
 *          and verify it is the peer's ack for the previous wave
 *          (expected_imm) targeting our ack slot. Waiting for the ack means
 *          the peer's add has completed, so our signals may safely
 *          overwrite the peer's recv arena.
 *
 * The arrival-ring FIFO order [data x N, ack] per wave matches the
 * [waiter, gate] consumption order on the receiving rank; wave indices and
 * chunk counts are identical on both ranks by construction. The very first
 * wave still uses mf310p_control_barrier once as the pool-creation
 * rendezvous.
 */
int mf310p_ack_async(
    mf310p_context_t* ctx,
    uint64_t imm,
    void* acl_stream);

int mf310p_gate_async(
    mf310p_context_t* ctx,
    uint64_t expected_imm,
    void* acl_stream);

/* Same double-launch warmup contract for the ack/gate kernels (enable == 0
 * side-effect-free shape). */
int mf310p_warmup_ack_gate_async(mf310p_context_t* ctx, void* acl_stream);

#ifdef __cplusplus
}
#endif
