/*
 * 310P3 TP=2 MemFabric runtime for the fused o_proj overlap pipeline.
 *
 * When VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ is enabled, vllm_ascend_C is
 * compiled and linked directly against the V5 MemFabric installation produced
 * by wgm-dev-310p (origin/wgm-dev-310p, mailbox-ring epoch API). Loading is
 * by direct link time dependency only; no bridge shared object is ever opened
 * dynamically at runtime. The small mf310p_* C ABI remains only as an
 * internal source boundary between vLLM-Ascend and the customized MemFabric
 * APIs.
 *
 * Per call (single fused op, no host-side staging anymore):
 *   for each wave (capacity batching, no host syncs between chunks):
 *     join        - sync current stream + control barrier: both ranks
 *                   finished every arena access of the previous wave,
 *                   including the reduced-output add that reads recv;
 *     producer    - fused AscendC FP16 matmul kernel(s): per chunk
 *                   matmul -> 64B-line clean -> signal toward peer recv;
 *     waiter      - quiet (my signals landed) + wait x chunks (peer chunks
 *                   landed in my recv);
 *     add         - out[wave] = send[wave] + recv[wave] on the same stream.
 */
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/string_view.h>
#include <torch/extension.h>

#ifdef ASCEND_PLATFORM_310P
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "memfabric_o_proj/external/memfabric310p_adapter_api.h"

namespace vllm_ascend {

/* Defined below (both build variants); forward-declared for the atexit hook. */
void memfabric_o_proj_shutdown();

#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
namespace {

constexpr int64_t kOProjWidth = 2048;
constexpr uint64_t kDefaultLocalPoolBytes = 32ULL * 1024ULL * 1024ULL;
constexpr const char* kDefaultStoreUrl = "tcp://127.0.0.1:8581";

uint64_t parse_u64_env(const char* name, uint64_t fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    TORCH_CHECK(
        errno == 0 && end != value && *end == '\0',
        "Invalid ", name, "=", value);
    return static_cast<uint64_t>(parsed);
}

struct RuntimeState {
    std::mutex mutex;
    mf310p_context_t* ctx = nullptr;
    mf310p_layout_t layout{};
    int tp_rank = -1;
    int64_t tile_m = 0;
    bool poisoned = false;
    std::string failure_reason;

    /* Direct producer state (allocated lazily, tile_m-sized). */
    uint64_t producer_scratch = 0;        /* device VA, tile_m*2048*2 bytes */
    uint32_t producer_warmed_buckets = 0; /* bit (log2(bucket)-4) per bucket */
    bool waiter_warmed = false;

    ~RuntimeState()
    {
        if (producer_scratch != 0) {
            (void)aclrtFree(reinterpret_cast<void*>(producer_scratch));
        }
        if (ctx != nullptr) {
            (void)mf310p_destroy(ctx);
        }
    }
};

RuntimeState& runtime_state()
{
    static RuntimeState state;
    return state;
}

void init_context_locked(
    RuntimeState& state,
    const at::Tensor& x,
    int64_t tp_rank,
    int64_t tile_m)
{
    TORCH_CHECK(
        mf310p_adapter_abi_version() == VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION,
        "310P MemFabric internal adapter ABI mismatch: expected ",
        VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION,
        ", got ", mf310p_adapter_abi_version());

    if (state.ctx != nullptr) {
        TORCH_CHECK(state.tp_rank == tp_rank,
                    "MemFabric runtime rank changed from ", state.tp_rank,
                    " to ", tp_rank);
        TORCH_CHECK(state.tile_m == tile_m,
                    "MemFabric runtime tile_m changed from ", state.tile_m,
                    " to ", tile_m,
                    ". Restart the worker when changing tile size.");
        return;
    }

    TORCH_CHECK(tp_rank == 0 || tp_rank == 1,
                "MemFabric runtime only supports TP=2 rank 0/1, got ", tp_rank);
    TORCH_CHECK(tile_m > 0, "tile_m must be positive");

    const uint64_t chunk_bytes =
        static_cast<uint64_t>(tile_m) * kOProjWidth * sizeof(at::Half); /* FP16 payload */
    const uint64_t arena_bytes =
        static_cast<uint64_t>(VLLM_ASCEND_MF310P_MAX_CHUNKS) * chunk_bytes;
    const uint64_t local_pool_bytes = parse_u64_env(
        "VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES", kDefaultLocalPoolBytes);
    const char* store_url = std::getenv("VLLM_ASCEND_310P_MEMFABRIC_STORE_URL");
    if (store_url == nullptr || *store_url == '\0') {
        store_url = kDefaultStoreUrl;
    }

    const int ret = mf310p_create(
        static_cast<int>(tp_rank),
        2,
        store_url,
        local_pool_bytes,
        arena_bytes,
        chunk_bytes,
        &state.ctx);
    TORCH_CHECK(ret == 0 && state.ctx != nullptr,
                "mf310p_create failed with ret=", ret);

    const int layout_ret = mf310p_get_layout(state.ctx, &state.layout);
    TORCH_CHECK(layout_ret == 0,
                "mf310p_get_layout failed with ret=", layout_ret);
    TORCH_CHECK(state.layout.chunk_bytes == chunk_bytes,
                "MemFabric returned unexpected chunk_bytes");
    TORCH_CHECK(state.layout.max_chunks == VLLM_ASCEND_MF310P_MAX_CHUNKS,
                "MemFabric returned unexpected max_chunks");
    TORCH_CHECK(state.layout.pool_base != 0,
                "MemFabric returned an invalid pool base");

    state.tp_rank = static_cast<int>(tp_rank);
    state.tile_m = tile_m;
    (void)x;

    /*
     * LIFO atexit teardown: the process-persistent MemFabric context must be
     * destroyed while the ACL runtime is still alive. Registering here (after
     * torch_npu's import-time registrations) makes this handler run first, so
     * the RuntimeState static destructor - which may otherwise run after the
     * NPU runtime was finalized and segfault - becomes a no-op. Verified on
     * the real 310P3 server: without this, both ranks SIGSEGV at exit.
     */
    static bool atexit_registered = false;
    if (!atexit_registered) {
        atexit_registered = true;
        (void)std::atexit([]() { memfabric_o_proj_shutdown(); });
    }
}

void check_runtime_healthy(const RuntimeState& state)
{
    TORCH_CHECK(
        !state.poisoned,
        "310P MemFabric o_proj runtime is poisoned after a previous failed wave. "
        "Restart both TP workers before reusing it. First failure: ",
        state.failure_reason);
}

at::Tensor wrap_pool_tensor(
    uint64_t address,
    int64_t tile_m,
    const at::Tensor& x)
{
    const int64_t rows =
        static_cast<int64_t>(VLLM_ASCEND_MF310P_MAX_CHUNKS) * tile_m;
    const auto options = x.options().dtype(at::kHalf); /* FP16 payload */
    return at_npu::native::from_blob(
        reinterpret_cast<void*>(address),
        {rows, kOProjWidth},
        [](void*) {},
        options,
        x.device());
}

/* ---- Direct producer helpers ---- */

constexpr uint32_t kProducerMinBucket = 16;
constexpr uint32_t kProducerMaxBucket = 4096;

bool is_producer_bucket(uint32_t value)
{
    return value >= kProducerMinBucket && value <= kProducerMaxBucket &&
           (value & (value - 1)) == 0;
}

/* Smallest producer bucket covering `rows`; callers guarantee
 * rows <= tile_m <= kProducerMaxBucket so this cannot overflow. */
uint32_t producer_bucket_for_rows(uint32_t rows)
{
    uint32_t bucket = kProducerMinBucket;
    while (bucket < rows) {
        bucket <<= 1;
    }
    return bucket;
}

uint32_t producer_bucket_bit(uint32_t bucket)
{
    return 1u << (__builtin_ctz(bucket) - 4);
}

/*
 * The first launch of each kernel instantiation from a freshly loaded .so is
 * a silent binary-eager-load no-op on dav-2002, so every M-bucket must be
 * warmed twice before its first real wave. Both warmup launches use the
 * chunk_count == 0 shape, which returns before touching any memory.
 */
void warm_producer_bucket_locked(
    RuntimeState& state,
    uint32_t bucket,
    aclrtStream stream)
{
    const uint32_t bit = producer_bucket_bit(bucket);
    if (state.producer_warmed_buckets & bit) {
        return;
    }
    for (int i = 0; i < 2; ++i) {
        const int ret = mf310p_warmup_producer_async(
            state.ctx,
            bucket,
            reinterpret_cast<void*>(stream));
        TORCH_CHECK(ret == 0,
                    "mf310p_warmup_producer_async failed for bucket ",
                    bucket, " with ret=", ret);
    }
    const aclError sync_ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(sync_ret == ACL_SUCCESS,
                "producer warmup sync failed, ret=", sync_ret);
    state.producer_warmed_buckets |= bit;
}

/* Same double-launch warmup contract for the single waiter kernel. */
void warm_waiter_locked(RuntimeState& state, aclrtStream stream)
{
    if (state.waiter_warmed) {
        return;
    }
    for (int i = 0; i < 2; ++i) {
        const int ret = mf310p_warmup_waiter_async(
            state.ctx, reinterpret_cast<void*>(stream));
        TORCH_CHECK(ret == 0,
                    "mf310p_warmup_waiter_async failed with ret=", ret);
    }
    const aclError sync_ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(sync_ret == ACL_SUCCESS,
                "waiter warmup sync failed, ret=", sync_ret);
    state.waiter_warmed = true;
}

} // namespace

void memfabric_o_proj_shutdown()
{
#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.ctx == nullptr) {
        return;
    }
    /*
     * An in-flight wave is unsafe to destroy (outstanding mailbox state);
     * leak instead of tearing down mid-flight in a dying process. The
     * wave-boundary join inside the fused op means a healthy shutdown between
     * calls is always safe.
     */
    if (state.producer_scratch != 0) {
        (void)aclrtFree(reinterpret_cast<void*>(state.producer_scratch));
        state.producer_scratch = 0;
    }
    (void)mf310p_destroy(state.ctx);
    state.ctx = nullptr;
#endif
}

/*
 * Debug-only observability for the V5 mailbox rings. Synchronously copies
 * both ranks' reserved regions (48 KiB each, D2H via the copy engine - it
 * does not depend on the possibly-stuck compute stream) and returns the
 * protocol-relevant words so a hung wave can be dissected live:
 * does the epoch consume (reqHead), did the doorbell ring and the SQE trio
 * execute (quiet stamps / arrival stamps), did the data land (arena words),
 * and the same for the peer side (its region is mapped in our VA space).
 */
at::Tensor memfabric_o_proj_debug_snapshot()
{
    constexpr int64_t kWords = 24;
    at::Tensor out = at::zeros({kWords}, at::TensorOptions().dtype(at::kLong));
    int64_t* w = out.data_ptr<int64_t>();

#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.ctx == nullptr) {
        return out;
    }
    const mf310p_layout_t& layout = state.layout;
    constexpr uint64_t kReserved = 48ULL * 1024ULL;
    /* Mirrors smem_shm_sdma_layout.h offsets. */
    constexpr uint64_t kReqTail = 0x800 / 8;
    constexpr uint64_t kReqHead = 0x808 / 8;
    constexpr uint64_t kQuiet = 0x810 / 8;
    constexpr uint64_t kArrStamp = 0xA20 / 8;
    constexpr uint64_t kArrMail = 0x2A20 / 8;
    constexpr uint64_t kArrHead = 0xAA20 / 8;
    const uint64_t words_per_region = kReserved / 8;

    std::vector<uint64_t> own(words_per_region, 0);
    std::vector<uint64_t> peer(words_per_region, 0);
    const uint64_t own_reserved = layout.own_segment + layout.local_size - kReserved;
    const uint64_t peer_reserved = layout.peer_segment + layout.local_size - kReserved;
    bool own_ok = aclrtMemcpy(own.data(), kReserved,
                              reinterpret_cast<void*>(own_reserved), kReserved,
                              ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    bool peer_ok = aclrtMemcpy(peer.data(), kReserved,
                               reinterpret_cast<void*>(peer_reserved), kReserved,
                               ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;

    auto arena_word = [](uint64_t addr, bool* ok) -> uint64_t {
        uint64_t v = 0;
        if (ok != nullptr && !*ok) {
            return 0;
        }
        const aclError ret = aclrtMemcpy(&v, sizeof(v),
                                         reinterpret_cast<void*>(addr), sizeof(v),
                                         ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS && ok != nullptr) {
            *ok = false;
        }
        return v;
    };

    const int64_t magic = 0x4D463135  /* "MF15" marker: dump is live */
                          + (own_ok ? 1 << 8 : 0) + (peer_ok ? 1 << 16 : 0);
    w[0] = magic;
    w[1] = state.tp_rank;
    w[2] = static_cast<int64_t>(own[kReqTail]);
    w[3] = static_cast<int64_t>(own[kReqHead]);
    w[4] = static_cast<int64_t>(own[kQuiet]);
    w[5] = static_cast<int64_t>(own[kQuiet + 1]);
    w[6] = static_cast<int64_t>(own[kArrStamp]);
    w[7] = static_cast<int64_t>(own[kArrHead]);
    w[8] = static_cast<int64_t>(own[kArrMail]);
    w[9] = static_cast<int64_t>(own[kArrMail + 1]);
    w[10] = static_cast<int64_t>(own[kArrMail + 2]);
    w[11] = static_cast<int64_t>(own[kArrMail + 3]);
    w[12] = static_cast<int64_t>(arena_word(layout.send_arena, &own_ok));
    w[13] = static_cast<int64_t>(arena_word(layout.recv_arena, &own_ok));
    w[14] = static_cast<int64_t>(peer[kReqTail]);
    w[15] = static_cast<int64_t>(peer[kReqHead]);
    w[16] = static_cast<int64_t>(peer[kQuiet]);
    w[17] = static_cast<int64_t>(peer[kArrStamp]);
    w[18] = static_cast<int64_t>(peer[kArrHead]);
    w[19] = static_cast<int64_t>(peer[kArrMail]);
    w[20] = static_cast<int64_t>(peer[kArrMail + 3]);
    w[21] = static_cast<int64_t>(arena_word(layout.peer_recv_arena, &peer_ok));
    w[22] = static_cast<int64_t>(arena_word(layout.peer_segment, &peer_ok));
    w[23] = 0;
#endif
    return out;
}

at::Tensor memfabric_direct_o_proj_allreduce_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t tp_rank,
    int64_t tile_m)
{
    /*
     * Fused direct producer on the V5 epoch API. The kernel instantiations
     * are M-bucket specialized (static tiling), so tile_m must be a power of
     * two in [16, 4096]: a chunk slot holds tile_m rows, and every kernel
     * writes m_bucket <= tile_m rows into it.
     */
    TORCH_CHECK(tile_m >= kProducerMinBucket && tile_m <= kProducerMaxBucket &&
                    (tile_m & (tile_m - 1)) == 0,
                "direct MemFabric o_proj requires tile_m to be a power of two "
                "in [16, 4096], got ",
                tile_m);
    TORCH_CHECK(x.is_contiguous(),
                "direct MemFabric o_proj requires contiguous x, got strides ",
                x.strides());
    TORCH_CHECK(weight.is_contiguous(),
                "direct MemFabric o_proj requires contiguous weight");

    const int64_t num_tokens = x.size(0);
    if (num_tokens == 0) {
        return at::empty({0, kOProjWidth}, x.options());
    }

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    check_runtime_healthy(state);
    init_context_locked(state, x, tp_rank, tile_m);

    /* Tail staging buffer: the partial last chunk reads bucket-padded rows,
     * which may exceed x's valid rows, so it reads from this scratch copy
     * instead. */
    const uint64_t scratch_bytes =
        static_cast<uint64_t>(tile_m) * kOProjWidth * sizeof(at::Half);
    if (state.producer_scratch == 0) {
        void* scratch = nullptr;
        const aclError malloc_ret = aclrtMalloc(
            &scratch, scratch_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(malloc_ret == ACL_SUCCESS && scratch != nullptr,
                    "producer scratch aclrtMalloc failed, ret=", malloc_ret);
        state.producer_scratch = reinterpret_cast<uint64_t>(scratch);
    }

    at::Tensor output =
        at::empty({num_tokens, kOProjWidth}, x.options());
    const uint64_t weight_ptr =
        reinterpret_cast<uint64_t>(weight.const_data_ptr());
    const int64_t max_rows_per_wave =
        static_cast<int64_t>(VLLM_ASCEND_MF310P_MAX_CHUNKS) * tile_m;
    const at::Tensor send_pool =
        wrap_pool_tensor(state.layout.send_arena, tile_m, x);
    const at::Tensor recv_pool =
        wrap_pool_tensor(state.layout.recv_arena, tile_m, x);

    for (int64_t wave_start = 0; wave_start < num_tokens;
         wave_start += max_rows_per_wave) {
        const int64_t wave_rows =
            std::min<int64_t>(max_rows_per_wave, num_tokens - wave_start);
        const uint32_t chunks =
            static_cast<uint32_t>((wave_rows + tile_m - 1) / tile_m);
        const uint32_t rows_last =
            static_cast<uint32_t>(wave_rows - (chunks - 1) * tile_m);
        const bool has_tail = rows_last != static_cast<uint32_t>(tile_m);
        const uint32_t full_count = has_tail ? chunks - 1 : chunks;
        const uint32_t tail_bucket =
            has_tail ? producer_bucket_for_rows(rows_last) : 0;

        try {
            aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

            /*
             * Wave-boundary rendezvous: synchronize this rank's stream (every
             * previous arena access, including the reduced-output add, is
             * complete) then control-barrier with the peer. Without it, a
             * fast rank's next-wave signals could overwrite the slow rank's
             * recv arena while its add still reads it. There is no host
             * synchronization between chunks inside the wave.
             */
            int ret = mf310p_join_previous_call(
                state.ctx, reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0,
                        "mf310p_join_previous_call failed with ret=", ret);

            /* Lazy per-symbol warmup before the wave is armed. */
            if (full_count > 0) {
                warm_producer_bucket_locked(
                    state, static_cast<uint32_t>(tile_m), stream);
            }
            if (has_tail) {
                warm_producer_bucket_locked(state, tail_bucket, stream);
            }
            warm_waiter_locked(state, stream);

            const uint64_t x_wave = reinterpret_cast<uint64_t>(
                x.const_data_ptr()) +
                static_cast<uint64_t>(wave_start) * kOProjWidth *
                    sizeof(at::Half);

            if (full_count > 0) {
                ret = mf310p_direct_producer_async(
                    state.ctx,
                    x_wave,
                    weight_ptr,
                    static_cast<uint32_t>(tile_m),
                    0,
                    full_count,
                    static_cast<uint32_t>(tile_m),
                    reinterpret_cast<void*>(stream));
                TORCH_CHECK(ret == 0,
                            "full-chunk mf310p_direct_producer_async failed, "
                            "ret=",
                            ret);
            }

            if (has_tail) {
                /* Deterministic tail slot: zero the whole scratch, copy the
                 * valid tail rows into it, and zero the slot rows the
                 * bucket kernel does not write (rows beyond the valid tail
                 * are never read downstream, but the peer reduces the full
                 * fixed-size chunk). */
                const aclError mem_ret = aclrtMemsetAsync(
                    reinterpret_cast<void*>(state.producer_scratch),
                    scratch_bytes,
                    0,
                    scratch_bytes,
                    stream);
                TORCH_CHECK(mem_ret == ACL_SUCCESS,
                            "tail scratch memset failed, ret=", mem_ret);
                const aclError copy_ret = aclrtMemcpyAsync(
                    reinterpret_cast<void*>(state.producer_scratch),
                    scratch_bytes,
                    reinterpret_cast<const void*>(
                        x_wave +
                        static_cast<uint64_t>(full_count) * tile_m *
                            kOProjWidth * sizeof(at::Half)),
                    static_cast<uint64_t>(rows_last) * kOProjWidth *
                        sizeof(at::Half),
                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                    stream);
                TORCH_CHECK(copy_ret == ACL_SUCCESS,
                            "tail scratch copy failed, ret=", copy_ret);

                const uint64_t tail_slot =
                    state.layout.send_arena +
                    static_cast<uint64_t>(chunks - 1) *
                        state.layout.chunk_bytes;
                const uint64_t pad_bytes = static_cast<uint64_t>(
                    tile_m - tail_bucket) *
                    kOProjWidth * sizeof(at::Half);
                if (pad_bytes > 0) {
                    const aclError pad_ret = aclrtMemsetAsync(
                        reinterpret_cast<void*>(
                            tail_slot +
                            static_cast<uint64_t>(tail_bucket) * kOProjWidth *
                                sizeof(at::Half)),
                        pad_bytes,
                        0,
                        pad_bytes,
                        stream);
                    TORCH_CHECK(pad_ret == ACL_SUCCESS,
                                "tail slot pad memset failed, ret=", pad_ret);
                }

                ret = mf310p_direct_producer_async(
                    state.ctx,
                    state.producer_scratch,
                    weight_ptr,
                    tail_bucket,
                    chunks - 1,
                    1,
                    tail_bucket,
                    reinterpret_cast<void*>(stream));
                TORCH_CHECK(ret == 0,
                            "tail mf310p_direct_producer_async failed, ret=",
                            ret);
            }

            /* Quiet (my signals landed) + wait x chunks (peer chunks landed
             * in my recv). Everything after this on the stream may read both
             * arenas. */
            ret = mf310p_wait_mails_async(
                state.ctx, chunks, reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0,
                        "mf310p_wait_mails_async failed with ret=", ret);

            /* Reduced result into ordinary NPU storage so the symmetric
             * arenas can be safely reused by later layers. Stream-ordered
             * after the waiter, so both arenas are complete. */
            at::Tensor out_narrow = output.narrow(0, wave_start, wave_rows);
            at::add_out(
                out_narrow,
                send_pool.narrow(0, 0, wave_rows),
                recv_pool.narrow(0, 0, wave_rows));
        } catch (const std::exception& exc) {
            /* A partially executed wave is unsafe to reuse (outstanding
             * mailbox/SDMA state is not recoverable without a rank-wide
             * restart); poison the process-local runtime and rethrow. */
            state.poisoned = true;
            if (state.failure_reason.empty()) {
                state.failure_reason = exc.what();
            }
            throw;
        }
    }

    return output;
}

#else  // VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ

namespace {
[[noreturn]] void memfabric_not_built()
{
    TORCH_CHECK(
        false,
        "vllm_ascend_C was built without the 310P customized MemFabric o_proj "
        "runtime. Build/install wgm-dev-310p MemFabric first, set "
        "VLLM_ASCEND_310P_MEMFABRIC_ROOT and required build variables, then "
        "rebuild vLLM-Ascend.");
    std::abort();
}
} // namespace

at::Tensor memfabric_direct_o_proj_allreduce_impl(
    const at::Tensor&,
    const at::Tensor&,
    int64_t,
    int64_t)
{
    memfabric_not_built();
}

void memfabric_o_proj_shutdown()
{
    /* Feature-off builds never create the context; nothing to tear down. */
}

#endif // VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ

} // namespace vllm_ascend
#endif // ASCEND_PLATFORM_310P
