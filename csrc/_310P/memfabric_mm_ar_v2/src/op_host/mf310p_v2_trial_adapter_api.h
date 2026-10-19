/* Experimental standalone V2 ABI, NOT wired into vLLM model dispatch. */
#pragma once
#include <stdint.h>
#if defined(__GNUC__)
#define MF310P_V2_API __attribute__((visibility("default")))
#else
#define MF310P_V2_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mf310p_v2_trial mf310p_v2_trial_t;

/* Caller must have selected a 310P3 device and created an ACL context. */
MF310P_V2_API int mf310p_v2_trial_create(int rank, const char* store_url,
                          uint32_t chunk_rows, mf310p_v2_trial_t** out);
/* G0 must pass; includes IB event-pair reuse check (2 AICore kernel). */
MF310P_V2_API int mf310p_v2_trial_ib_probe(mf310p_v2_trial_t* handle,
                            uint32_t repetitions, void* acl_stream);
/* x/w/y are device pointers: [M,2048], NZ[2048,2048], [M,2048].
 * Only M > 0, M % chunk_rows == 0 and M <= 8192 in this first version.
 * Both ranks must invoke with identical M, chunk_rows, overlap and order.
 * Enqueue-only API; use stream sync + check_status for validation.
 */
MF310P_V2_API int mf310p_v2_trial_run(mf310p_v2_trial_t* handle, uint64_t x,
                        uint64_t w, uint64_t y, uint32_t m,
                        uint32_t overlap, void* acl_stream);
/* T1 direct-ND trial: original NZ weights, exact M=1..256;
 * no weight slicing or staging; independent {1,2,4,8} MM/Add cores.
 * Still an experimental native C API, not routed from vLLM.
 */
MF310P_V2_API int mf310p_v2_trial_run_nz_direct_small(
    mf310p_v2_trial_t* handle, uint64_t x, uint64_t w, uint64_t y,
    uint32_t m, uint32_t mm_cores, uint32_t add_cores,
    void* acl_stream);
/* T2 non-fused baseline: native framework MM runs first on the SAME ACL
 * stream and produces contiguous ND FP16 native_partial[M,2048].
 * AR-only stages it into pool C with D2D memcpy then performs MemFabric
 * signal/wait + vector Add + wave DONE. M=1..8192, addCores=1/2/4/8.
 * Asynchronous, experimental; includes mandatory staging cost.
 * The caller must not alias native_partial and output or run two streams
 * against the same trial context concurrently.
 */
MF310P_V2_API int mf310p_v2_trial_run_ar_only(
    mf310p_v2_trial_t* handle, uint64_t native_partial, uint64_t output,
    uint32_t m, uint32_t add_cores, void* acl_stream);

MF310P_V2_API int mf310p_v2_trial_check_status(mf310p_v2_trial_t* handle,
                                uint64_t* protocol_status);
/* Test scaffolding: synchronous D2H readback of this rank's SEND arena
 * (where AR-only staging lands). The sync copy reads device DDR, i.e.
 * exactly the view the MemFabric AICPU SDMA engine observes. */
MF310P_V2_API int mf310p_v2_trial_read_send_arena(
    mf310p_v2_trial_t* handle, void* host_dst, uint32_t bytes);
/* Test scaffolding: staged isolation of the fused pipeline.
 * diag=0: per-core MM of one chunk straight into y (no communication).
 * diag=1: MM into ownC + single signal/wait exchange + Add, without the
 *         inter-core IB rendezvous (core0 proceeds after its own MM). */
MF310P_V2_API int mf310p_v2_trial_diag(mf310p_v2_trial_t* handle, uint64_t x,
                         uint64_t w, uint64_t y, uint32_t diag,
                         void* acl_stream);
MF310P_V2_API int mf310p_v2_trial_destroy(mf310p_v2_trial_t* handle);
#ifdef __cplusplus
}
#endif
