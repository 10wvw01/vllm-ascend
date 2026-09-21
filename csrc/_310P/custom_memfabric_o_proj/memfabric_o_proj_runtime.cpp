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
 *     producer    - one communication coordinator overlaps public MemFabric
 *                   signal() with seven interleaved AscendC MM workers;
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
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "memfabric310p_adapter_api.h"

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
 * Optional host enqueue tracing. These timestamps are not device execution
 * timings; use the CANN profiler for MM/SDMA overlap measurements.
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
    bool protocol_warmed = false;
    bool protocol_initialized = false;
    aclrtStream eager_stream = nullptr;
    bool graph_capture_seen = false;

    /* Teardown is owned by the atexit hook while ACL is still alive. */
    ~RuntimeState() = default;
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
        const int atexit_ret =
            std::atexit([]() { memfabric_o_proj_shutdown(); });
        TORCH_CHECK(atexit_ret == 0,
                    "failed to register MemFabric shutdown hook");
        atexit_registered = true;
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

/* Same double-launch warmup contract for waiter/add/protocol kernels. */
void warm_waiter_locked(RuntimeState& state, aclrtStream stream)
{
    if (state.waiter_warmed) return;

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
    if (!state.protocol_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int ret = mf310p_warmup_protocol_async(
                state.ctx, reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0,
                        "mf310p_warmup_protocol_async failed with ret=", ret);
        }
        state.protocol_warmed = true;
    }
    const aclError sync_ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(sync_ret == ACL_SUCCESS,
                "waiter/add/protocol warmup sync failed, ret=", sync_ret);
    state.waiter_warmed = true;
}

bool is_current_stream_capturing()
{
    return c10_npu::currentStreamCaptureStatusMayInitCtx() ==
           c10_npu::CaptureStatus::Active;
}

void validate_execution_stream_locked(
    RuntimeState& state,
    aclrtStream stream,
    bool capturing)
{
    TORCH_CHECK(stream != nullptr, "MemFabric requires a valid ACL stream");

    /*
     * torch.npu.graph() intentionally switches to an internal non-default
     * capture stream. The graph context synchronizes the device before that
     * transition, so it is a framework-managed quiescent stream handoff, not
     * arbitrary concurrent eager use. Keep the eager contract strict while
     * allowing that capture side stream.
     */
    if (capturing) {
        state.graph_capture_seen = true;
        return;
    }

    if (state.eager_stream == nullptr) {
        state.eager_stream = stream;
        return;
    }
    TORCH_CHECK(
        state.eager_stream == stream,
        "310P MemFabric o_proj eager execution is single-stream by contract. "
        "The first eager call used stream=",
        reinterpret_cast<void*>(state.eager_stream),
        ", current stream=",
        reinterpret_cast<void*>(stream),
        ". ACL Graph capture side streams are handled separately.");
}

void require_capture_ready_locked(
    const RuntimeState& state,
    uint32_t full_bucket,
    bool has_full,
    uint32_t tail_bucket,
    bool has_tail)
{
    TORCH_CHECK(
        state.ctx != nullptr && state.protocol_initialized &&
            state.producer_scratch != 0 && state.waiter_warmed &&
            state.add_warmed && state.protocol_warmed,
        "310P MemFabric o_proj entered ACL Graph capture before runtime "
        "initialization/warmup completed. Run the normal eager warmup/profile "
        "path before graph capture.");

    if (has_full) {
        const uint32_t bit = producer_bucket_bit(full_bucket);
        TORCH_CHECK(
            state.producer_warmed_buckets & bit,
            "310P MemFabric producer bucket ", full_bucket,
            " was not warmed before ACL Graph capture.");
    }
    if (has_tail) {
        const uint32_t bit = producer_bucket_bit(tail_bucket);
        TORCH_CHECK(
            state.producer_warmed_buckets & bit,
            "310P MemFabric tail bucket ", tail_bucket,
            " was not warmed before ACL Graph capture.");
    }
}

void initialize_protocol_locked(RuntimeState& state, aclrtStream stream)
{
    if (state.protocol_initialized) return;

    warm_waiter_locked(state, stream);

    int ret = mf310p_prepare_wave_async(
        state.ctx, reinterpret_cast<void*>(stream));
    TORCH_CHECK(ret == 0,
                "mf310p_prepare_wave_async(init) failed with ret=", ret);

    ret = mf310p_control_barrier(state.ctx);
    TORCH_CHECK(ret == 0,
                "mf310p_control_barrier(init) failed with ret=", ret);

    ret = mf310p_init_credit_async(
        state.ctx, reinterpret_cast<void*>(stream));
    TORCH_CHECK(ret == 0,
                "mf310p_init_credit_async failed with ret=", ret);

    const aclError sync_ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(sync_ret == ACL_SUCCESS,
                "initial MemFabric credit sync failed, ret=", sync_ret);
    state.protocol_initialized = true;
}

} // namespace

void memfabric_o_proj_shutdown()
{
#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.ctx == nullptr) return;

    if (state.graph_capture_seen) {
        /*
         * NPUGraph replay may execute the captured model on whichever stream
         * is current at replay time, and replay bypasses this host function.
         * We therefore cannot prove all replay streams quiescent here without
         * a device-wide synchronization (which this integration deliberately
         * avoids). Keep the external pool process-lifetime instead of risking
         * destruction with in-flight graph work.
         */
        std::fprintf(
            stderr,
            "[mf310p] graph-used context kept process-lifetime; skip unsafe "
            "MemFabric destroy\n");
        return;
    }

    if (state.eager_stream != nullptr) {
        /* The last healthy wave ends with an asynchronous credit signal.
         * Enqueue public quiet before the final stream sync so every local
         * MemFabric request, including that credit, is proven landed before
         * pool destruction. */
        const int quiet_ret = mf310p_quiet_async(
            state.ctx, reinterpret_cast<void*>(state.eager_stream));
        if (quiet_ret != 0) {
            state.poisoned = true;
            if (state.failure_reason.empty()) {
                state.failure_reason =
                    "shutdown quiet enqueue failed; MemFabric resources "
                    "intentionally leaked until process exit";
            }
            std::fprintf(
                stderr,
                "[mf310p] skip unsafe destroy after quiet enqueue ret=%d\n",
                quiet_ret);
            return;
        }

        const aclError sync_ret = aclrtSynchronizeStream(state.eager_stream);
        if (sync_ret != ACL_SUCCESS) {
            state.poisoned = true;
            if (state.failure_reason.empty()) {
                state.failure_reason =
                    "shutdown stream synchronization failed; MemFabric "
                    "resources intentionally leaked until process exit";
            }
            std::fprintf(
                stderr,
                "[mf310p] skip unsafe destroy after stream sync failure ret=%d\n",
                static_cast<int>(sync_ret));
            return;
        }
    }

    if (state.producer_scratch != 0) {
        (void)aclrtFree(reinterpret_cast<void*>(state.producer_scratch));
        state.producer_scratch = 0;
    }
    const int destroy_ret = mf310p_destroy(state.ctx);
    if (destroy_ret != 0) {
        std::fprintf(stderr, "[mf310p] mf310p_destroy ret=%d\n", destroy_ret);
    }
    state.ctx = nullptr;
    state.eager_stream = nullptr;
    state.graph_capture_seen = false;
    state.protocol_initialized = false;
    state.protocol_warmed = false;
    state.waiter_warmed = false;
    state.add_warmed = false;
    state.producer_warmed_buckets = 0;
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
    w[0] = 0x4D465036; /* MFP6 */
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
    w[12] = state.protocol_initialized ? 1 : 0;
    w[13] = static_cast<int64_t>(arena_word(layout.send_arena, &own_ok));
    w[14] = static_cast<int64_t>(arena_word(layout.recv_arena, &own_ok));
    w[15] = static_cast<int64_t>(arena_word(layout.peer_recv_arena, &peer_ok));
    w[16] = own_ok ? 1 : 0;
    w[17] = peer_ok ? 1 : 0;
    uint64_t protocol_status = 0;
    const int status_ret =
        mf310p_debug_protocol_status(state.ctx, &protocol_status);
    w[18] = static_cast<int64_t>(protocol_status);
    w[19] = status_ret == 0 ? 1 : 0;
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

    const auto npu_stream = c10_npu::getCurrentNPUStream();
    aclrtStream stream = npu_stream.stream();
    const bool capturing = is_current_stream_capturing();

    if (capturing) {
        TORCH_CHECK(
            state.ctx != nullptr,
            "310P MemFabric context must be created by eager warmup before "
            "ACL Graph capture.");
    }
    init_context_locked(state, x, tp_rank, tile_m);
    validate_execution_stream_locked(state, stream, capturing);

    /* Tail staging buffer: the partial last chunk reads bucket-padded rows,
     * which may exceed x's valid rows, so it reads from this scratch copy
     * instead. */
    const uint64_t scratch_bytes =
        static_cast<uint64_t>(tile_m) * kOProjWidth * sizeof(at::Half);
    if (state.producer_scratch == 0) {
        TORCH_CHECK(
            !capturing,
            "310P MemFabric producer scratch must be allocated by eager "
            "warmup before ACL Graph capture.");
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

    if (capturing) {
        TORCH_CHECK(
            state.protocol_initialized,
            "310P MemFabric fixed-credit protocol must be initialized before "
            "ACL Graph capture.");
    } else {
        initialize_protocol_locked(state, stream);
    }

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
            /* Finish one-time local kernel preparation before consuming the
             * peer credit. A warmup failure must not spend a credit token. */
            const int64_t t_warm0 = trace_now_us();
            if (capturing) {
                require_capture_ready_locked(
                    state,
                    static_cast<uint32_t>(tile_m),
                    full_count > 0,
                    tail_bucket,
                    has_tail);
            } else {
                if (full_count > 0) {
                    warm_producer_bucket_locked(
                        state, static_cast<uint32_t>(tile_m), stream);
                }
                if (has_tail) {
                    warm_producer_bucket_locked(state, tail_bucket, stream);
                }
            }
            const int64_t t_warm = trace_now_us();

            const int64_t t_join0 = t_warm;
            /*
             * Graph-safe fixed application credit. Every wave consumes one
             * token before reusing peer recv; the previous wave publishes the
             * next token after its local add.
             */
            int ret = mf310p_prepare_wave_async(
                state.ctx, reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0,
                        "mf310p_prepare_wave_async failed with ret=", ret);
            ret = mf310p_gate_async(
                state.ctx, reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0,
                        "mf310p_gate_async failed with ret=", ret);
            const int64_t t_join2 = trace_now_us();

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
                reinterpret_cast<void*>(stream));
            TORCH_CHECK(ret == 0, "mf310p_ack_async failed with ret=", ret);

            if (trace_enabled()) {
                const int64_t t_enq = trace_now_us();
                trace_wave(
                    state.tp_rank,
                    wave_rows,
                    chunks,
                    0,
                    t_join2 - t_join0,
                    t_warm - t_warm0,
                    t_enq - t_join2,
                    wave_start == 0 ? t_warm0 - t_entry : 0);
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
