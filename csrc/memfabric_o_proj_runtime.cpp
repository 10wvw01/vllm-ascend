/*
 * 310P3 TP=2 MemFabric runtime for the fused o_proj overlap pipeline.
 *
 * When VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ is enabled, vllm_ascend_C is
 * compiled and linked directly against the wgm-dev-310p MemFabric public
 * SHM/SDMA API. Loading is
 * by direct link time dependency only; no bridge shared object is ever opened
 * dynamically at runtime. The small mf310p_* C ABI remains only as an
 * internal source boundary between vLLM-Ascend and the customized MemFabric
 * APIs.
 *
 * Per call (single fused op, no host-side staging anymore):
 *   for each wave (capacity batching, no host syncs between chunks):
 *     join        - first wave only: stream sync + control barrier as the
 *                   pool-creation rendezvous; later waves use the gate
 *                   kernel (waits for the peer's ack mail of the previous
 *                   wave, proving its reduced-output add that reads recv
 *                   completed - only then may our signals overwrite its
 *                   recv arena);
 *     producer    - multi-block AscendC FP16 matmul followed by a
 *                   single-block public MemFabric signal publisher;
 *     waiter      - public quiet + public wait x chunks (peer chunks
 *                   landed in my recv);
 *     add         - out[wave] = send[wave] + recv[wave] on the same stream;
 *     ack         - signal 8B (imm = wave index) to the peer's ack slot
 *                   after the add, consumed by the peer's next-wave gate.
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
#include <chrono>
#include <cstdio>
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

/*
 * P6 phase tracing (debug only): VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE=1
 * prints per-wave host-side wall time of each protocol phase. Only the
 * host-blocking phases (join sync/barrier, warmup sync) are accurate
 * device-side waits; enqueue times measure host launch overhead.
 */
bool trace_enabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE");
        return v != nullptr && *v != '\0' && *v != '0';
    }();
    return enabled;
}

int64_t trace_now_us()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void trace_wave(
    int rank,
    int64_t rows,
    uint32_t chunks,
    int64_t join_sync_us,
    int64_t join_barrier_us,
    int64_t warmup_us,
    int64_t enqueue_us,
    int64_t prejoin_us)
{
    std::fprintf(
        stderr,
        "[mf310p-trace] rank=%d rows=%lld chunks=%u join_sync_us=%lld "
        "join_barrier_us=%lld warmup_us=%lld enqueue_us=%lld prejoin_us=%lld "
        "total_us=%lld\n",
        rank, static_cast<long long>(rows), chunks,
        static_cast<long long>(join_sync_us),
        static_cast<long long>(join_barrier_us),
        static_cast<long long>(warmup_us),
        static_cast<long long>(enqueue_us),
        static_cast<long long>(prejoin_us),
        static_cast<long long>(join_sync_us + join_barrier_us + warmup_us +
                               enqueue_us + prejoin_us));
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
    bool add_warmed = false;
    /* Completed-wave counter driving the ack/gate rendezvous: the ack for
     * wave N carries imm = N, and the gate before wave N+1 expects it.
     * Identical on both ranks by the same symmetry. */
    uint64_t wave_count = 0;
    bool ack_gate_warmed = false;

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

/* Same double-launch warmup contract for the waiter and add kernels. */
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
    if (!state.add_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int ret = mf310p_warmup_add_async(
                state.ctx, reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0,
                        "mf310p_warmup_add_async failed with ret=", ret);
        }
        state.add_warmed = true;
    }
    if (!state.ack_gate_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int ret = mf310p_warmup_ack_gate_async(
                state.ctx, reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0,
                        "mf310p_warmup_ack_gate_async failed with ret=", ret);
        }
        state.ack_gate_warmed = true;
    }
    if (state.add_warmed || state.ack_gate_warmed) {
        const aclError sync_ret = aclrtSynchronizeStream(stream);
        TORCH_CHECK(sync_ret == ACL_SUCCESS,
                    "add/ack-gate warmup sync failed, ret=", sync_ret);
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
     * An in-flight wave is unsafe to destroy (outstanding communication state);
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
 * Debug-only application snapshot. MemFabric internals are intentionally
 * opaque: report only public pool geometry, vLLM-owned arenas and data words.
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

    auto arena_word = [](uint64_t addr, bool* ok) -> uint64_t {
        uint64_t v = 0;
        const aclError ret = aclrtMemcpy(
            &v, sizeof(v), reinterpret_cast<void*>(addr), sizeof(v),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS && ok != nullptr) {
            *ok = false;
        }
        return v;
    };

    bool own_ok = true;
    bool peer_ok = true;
    w[0] = 0x4D465035; /* MFP5 */
    w[1] = state.tp_rank;
    w[2] = static_cast<int64_t>(layout.pool_base);
    w[3] = static_cast<int64_t>(layout.symmetric_size);
    w[4] = static_cast<int64_t>(layout.local_size);
    w[5] = static_cast<int64_t>(layout.send_arena);
    w[6] = static_cast<int64_t>(layout.recv_arena);
    w[7] = static_cast<int64_t>(layout.peer_recv_arena);
    w[8] = static_cast<int64_t>(layout.ack_slot);
    w[9] = static_cast<int64_t>(layout.peer_ack_slot);
    w[10] = static_cast<int64_t>(layout.arena_bytes);
    w[11] = static_cast<int64_t>(layout.chunk_bytes);
    w[12] = static_cast<int64_t>(state.wave_count);
    w[13] = static_cast<int64_t>(arena_word(layout.send_arena, &own_ok));
    w[14] = static_cast<int64_t>(arena_word(layout.recv_arena, &own_ok));
    w[15] = static_cast<int64_t>(arena_word(layout.peer_recv_arena, &peer_ok));
    w[16] = own_ok ? 1 : 0;
    w[17] = peer_ok ? 1 : 0;
#endif
    return out;
}

at::Tensor memfabric_direct_o_proj_allreduce_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t tp_rank,
    int64_t tile_m)
{
    /* P6 tracing: entry timestamp covers validation + output allocation +
     * scratch alloc + pool-tensor wrapping (first wave only reports it). */
    const int64_t t_entry = trace_enabled() ? trace_now_us() : 0;
    /*
     * Public-API producer path. The kernel instantiations
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

            const int64_t t_join0 = trace_now_us();
            /*
             * Wave-boundary rendezvous (P6 Fix C): the very first wave uses
             * a stream sync + control barrier as the pool-creation
             * rendezvous. Every later wave uses the device-side gate
             * kernel, which waits for the peer's ack mail proving the
             * peer's reduced-output add completed - only then may this
             * rank's signals overwrite the peer's recv arena. This removes
             * the per-wave host barrier (~150 us steady state, rare ~40 ms
             * TCP spikes) and keeps waves fully pipelined on the stream.
             */
            int64_t t_join1 = t_join0;
            int ret = 0;
            if (state.wave_count == 0) {
                const aclError sync_ret = aclrtSynchronizeStream(stream);
                TORCH_CHECK(sync_ret == ACL_SUCCESS,
                            "first-wave stream join failed, ret=", sync_ret);
                t_join1 = trace_now_us();
                ret = mf310p_control_barrier(state.ctx);
                TORCH_CHECK(ret == 0,
                            "mf310p_control_barrier failed with ret=", ret);
            } else {
                ret = mf310p_gate_async(
                    state.ctx,
                    state.wave_count - 1,
                    reinterpret_cast<void*>(stream));
                TORCH_CHECK(ret == 0,
                            "mf310p_gate_async failed with ret=", ret);
            }
            const int64_t t_join2 = trace_now_us();

            /* Lazy per-symbol warmup before the wave is armed. */
            if (full_count > 0) {
                warm_producer_bucket_locked(
                    state, static_cast<uint32_t>(tile_m), stream);
            }
            if (has_tail) {
                warm_producer_bucket_locked(state, tail_bucket, stream);
            }
            warm_waiter_locked(state, stream);
            const int64_t t_warm = trace_now_us();

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

            /*
             * Reduced result into ordinary NPU storage via the repo-owned
             * multi-block vector-add kernel (shape-agnostic; at::add_out
             * would pay a per-output-shape GE compile on first use, ~90 ms
             * measured). Stream-ordered after the waiter, so both arenas
             * are complete.
             */
            ret = mf310p_add_async(
                state.ctx,
                reinterpret_cast<uint64_t>(
                    output.const_data_ptr()) +
                    wave_start * kOProjWidth * sizeof(at::Half),
                static_cast<uint64_t>(wave_rows) * kOProjWidth,
                reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0, "mf310p_add_async failed with ret=", ret);

            /* Application-level wave acknowledgement. Once the peer consumes
             * this public MemFabric mail, our next wave may overwrite its recv
             * arena. MemFabric's internal sequencing is opaque to vLLM. */
            ret = mf310p_ack_async(
                state.ctx,
                state.wave_count,
                reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0, "mf310p_ack_async failed with ret=", ret);
            state.wave_count += 1;

            if (trace_enabled()) {
                const int64_t t_enq = trace_now_us();
                trace_wave(
                    state.tp_rank,
                    wave_rows,
                    chunks,
                    t_join1 - t_join0,
                    t_join2 - t_join1,
                    t_warm - t_join2,
                    t_enq - t_warm,
                    wave_start == 0 ? t_join0 - t_entry : 0);
            }
        } catch (const std::exception& exc) {
            /* A partially executed wave is unsafe to reuse. Treat the
             * external communication subsystem as opaque, poison this
             * process-local context, and restart both TP workers. */
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
        "runtime. Install wgm-dev-310p MemFabric, source its set_env.sh, "
        "enable VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1, then rebuild "
        "vLLM-Ascend.");
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
