// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mf310p_v2_trial_adapter_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mf310p_v2_selected mf310p_v2_selected_t;

enum {
    MF310P_V2_SELECTED_ABI = 1,
    MF310P_V2_SELECTED_E1 = 7,
    MF310P_V2_SELECTED_E4_T8 = 15,
    MF310P_V2_SELECTED_INVALID = -21001,
    MF310P_V2_SELECTED_UNSUPPORTED = -21002,
    MF310P_V2_SELECTED_NOT_READY = -21003,
    MF310P_V2_SELECTED_POISONED = -21004,
    MF310P_V2_SELECTED_STREAM_MISMATCH = -21005,
    MF310P_V2_SELECTED_PROTOCOL = -21006
};

typedef struct mf310p_v2_selected_plan {
    uint32_t abi;
    uint32_t m;
    uint32_t version;       // VN-M ID: V2-7 or V2-15, not a kernel mode.
    uint32_t mm_rows;       // NZ MM invocation size, always 1024.
    uint32_t group_rows;    // Communication/readiness group: 1024 or 2048.
    uint32_t add_elements;  // FP16 vector ADD tile: 2048 or 8192.
} mf310p_v2_selected_plan_t;

// CPU-only query. Only M=2048/E1 and M=4096/E4-T8 have selected evidence.
MF310P_V2_API int mf310p_v2_selected_get_plan(
    uint32_t m, mf310p_v2_selected_plan_t* plan);

// One handle has one fixed M/variant, its own pool and one execution stream.
// Both ranks must create/probe/run/destroy matching handles in the same order.
// Use distinct rendezvous URLs for separate live handles. No Graph capture or
// concurrent calls. Allocate and initialize X/W/Y before create; keep them alive
// until the execution stream completes. Do not recycle warmed output storage.
MF310P_V2_API int mf310p_v2_selected_create(
    int rank, const char* store_url, uint32_t m, mf310p_v2_selected_t** out);
MF310P_V2_API int mf310p_v2_selected_ib_probe(
    mf310p_v2_selected_t* handle, uint32_t repetitions, void* acl_stream);

// X[M,2048] ND FP16, W[2048,2048] NZ29 FP16, Y[M,2048] ND FP16.
// Buffers must not alias. Enqueues the full MM+AR; output is usable by the next
// operation on this stream. Host reads require caller synchronization. Retains
// historical symbol/first-write guards and the per-call status clear.
MF310P_V2_API int mf310p_v2_selected_run(
    mf310p_v2_selected_t* handle, uint64_t x, uint64_t w, uint64_t y,
    void* acl_stream);

// Call after stream completion. Nonzero protocol status poisons the handle.
// A backend failure also poisons it: do not retry the communication wave.
MF310P_V2_API int mf310p_v2_selected_check_status(
    mf310p_v2_selected_t* handle, uint64_t* status);

// Synchronizes the bound stream before freeing. On synchronization/status
// failure preserves the handle and its resources for fail-stop teardown.
MF310P_V2_API int mf310p_v2_selected_destroy(mf310p_v2_selected_t* handle);

#ifdef __cplusplus
}
#endif
