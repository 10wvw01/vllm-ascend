// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mf310p_v3_context mf310p_v3_context;
uint32_t mf310p_v3_abi(void);
// One context per process; caller selects its NPU before create. TP=2 only.
int mf310p_v3_create(int rank, const char* store_url, uint32_t packet_bytes, mf310p_v3_context** out);
// All run methods enqueue on one fixed stream. Buffers must live through completion.
// group_rows=256/512/1024/2048; rows is padded to a whole group by the caller.
// mode=0 fused MM, mode=1 protocol-only on precomputed FP16 partials.
int mf310p_v3_run(mf310p_v3_context* ctx, uint64_t x, uint64_t weight_nz, uint64_t y,
                 uint32_t rows, uint32_t group_rows, uint32_t pipeline, uint32_t mode, void* stream);
// Both ranks contribute src to the peer's seed. Result is local_seed + peer_src.
int mf310p_v3_probe_add(mf310p_v3_context* ctx, uint64_t src, uint64_t seed, uint64_t y,
                       uint32_t bytes, void* stream);
// Mandatory one-time official IB fan-in/fan-out probe before data launches.
int mf310p_v3_ib_probe(mf310p_v3_context* ctx, uint32_t repetitions, void* stream);
int mf310p_v3_status(mf310p_v3_context* ctx, uint64_t* status);
int mf310p_v32_create(int rank, const char* store_url, uint32_t packet_bytes, mf310p_v3_context** out);
// V3.2 Plan A: two collective output slots, fixed stream and TP2 only.
// acquire is collective and synchronizes prior consumers before slot reuse.
int mf310p_v32_acquire(mf310p_v3_context*, uint32_t rows, uint32_t group, uint32_t mode,
    void* stream, uint32_t* slot, uint32_t* generation, uint64_t* address);
// release only marks a Storage lease free; it never touches the device.
int mf310p_v32_release(mf310p_v3_context*, uint32_t slot, uint32_t generation);
// Collective prepare after Storage wrapping, ready=0 if wrapping failed.
int mf310p_v32_prepare(mf310p_v3_context*, uint32_t slot, uint32_t generation, uint32_t ready, void* stream);
int mf310p_v32_run(mf310p_v3_context*, uint64_t x, uint64_t weight_nz, uint32_t rows,
    uint32_t group, uint32_t mode, uint32_t slot, uint32_t generation, void* stream);
// A3: cold warm is collective; hot reserve/run are local and asynchronous.
int mf310p_a3_create(int rank, const char*, uint32_t packet, mf310p_v3_context**);
int mf310p_a3_warm(mf310p_v3_context*, void* stream);
int mf310p_a3_reserve(mf310p_v3_context*, uint32_t rows, uint32_t compute, uint32_t comm,
    uint32_t mode, void* stream, uint32_t* slot, uint32_t* generation, uint64_t* address);
// reserve busy (-23) still returns a token. Submit admit=0 to stop BOTH ranks at START.
int mf310p_a3_run(mf310p_v3_context*, uint64_t x, uint64_t weight_nz, uint32_t rows,
    uint32_t compute, uint32_t comm, uint32_t mode, uint32_t slot, uint32_t generation,
    uint32_t admit, void* stream);
// Must be called after stream completion, on the owning device.
// V3.2: collective close; held output on EITHER rank returns -26 on BOTH ranks.
int mf310p_v3_destroy(mf310p_v3_context* ctx);
#ifdef __cplusplus
}
#endif
