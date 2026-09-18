/*
 * 310P3 TP=2 MemFabric runtime for the staged o_proj overlap pipeline.
 *
 * When VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ is enabled, vllm_ascend_C is
 * compiled and linked directly against the MemFabric installation produced by
 * wgm-dev-310p.  Loading is by direct link time dependency only; no bridge shared
 * object is ever opened dynamically at runtime.  The small mf310p_* C ABI remains only as an internal source
 * boundary between vLLM-Ascend and the customized MemFabric APIs.
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

#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <string>

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
    aclrtStream reduce_stream = nullptr;
    int tp_rank = -1;
    int64_t tile_m = 0;
    bool wave_active = false;
    bool poisoned = false;
    uint32_t active_chunks = 0;
    std::string failure_reason;

    ~RuntimeState()
    {
        if (reduce_stream != nullptr) {
            (void)aclrtDestroyStream(reduce_stream);
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
        static_cast<uint64_t>(tile_m) * kOProjWidth * sizeof(at::BFloat16);
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

    const aclError stream_ret = aclrtCreateStream(&state.reduce_stream);
    TORCH_CHECK(stream_ret == ACL_SUCCESS,
                "aclrtCreateStream(reduce) failed, ret=", stream_ret);

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
    const auto options = x.options().dtype(at::kBFloat16);
    return at_npu::native::from_blob(
        reinterpret_cast<void*>(address),
        {rows, kOProjWidth},
        [](void*) {},
        options,
        x.device());
}

} // namespace

std::tuple<at::Tensor, at::Tensor> memfabric_o_proj_begin(
    const at::Tensor& x,
    int64_t tp_rank,
    int64_t tile_m,
    int64_t chunks)
{
    TORCH_CHECK(x.is_privateuseone(), "x must be an NPU tensor");
    TORCH_CHECK(chunks > 0 && chunks <= VLLM_ASCEND_MF310P_MAX_CHUNKS,
                "chunks must be in [1, ", VLLM_ASCEND_MF310P_MAX_CHUNKS,
                "], got ", chunks);

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    check_runtime_healthy(state);
    init_context_locked(state, x, tp_rank, tile_m);
    TORCH_CHECK(!state.wave_active,
                "MemFabric o_proj begin called while previous wave is active");

    /*
     * Join all previously enqueued compute-stream work before the wave-boundary
     * rendezvous. Without this, a fast peer could start overwriting this rank's
     * recv arena (wave N+1 SDMA) while a still-pending local kernel from wave N
     * (e.g. the reduced-result output copy) is reading it. This is a wave-boundary
     * wait only; tiles inside the wave remain unsynchronized.
     */
    const aclError join_ret =
        aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream());
    TORCH_CHECK(join_ret == ACL_SUCCESS,
                "wave-boundary compute stream join failed, ret=", join_ret);

    /* Wave-boundary barriers are allowed. There is no barrier between tiles. */
    int ret = mf310p_prepare_wave(state.ctx);
    TORCH_CHECK(ret == 0, "mf310p_prepare_wave failed with ret=", ret);

    /* Consumer may poll before peer data arrives; that is intentional. */
    ret = mf310p_launch_reduce_consumer_async(
        state.ctx,
        static_cast<uint32_t>(chunks),
        reinterpret_cast<void*>(state.reduce_stream));
    TORCH_CHECK(ret == 0,
                "mf310p_launch_reduce_consumer_async failed with ret=", ret);

    /* Arm AICPU/SDMA before producer notifications start arriving. */
    ret = mf310p_submit_wave(state.ctx, static_cast<uint32_t>(chunks));
    TORCH_CHECK(ret == 0, "mf310p_submit_wave failed with ret=", ret);

    state.wave_active = true;
    state.active_chunks = static_cast<uint32_t>(chunks);

    return {
        wrap_pool_tensor(state.layout.send_arena, tile_m, x),
        wrap_pool_tensor(state.layout.recv_arena, tile_m, x),
    };
}

void memfabric_o_proj_publish(const at::Tensor& send, int64_t chunk_idx)
{
    TORCH_CHECK(send.is_privateuseone(), "send must be an NPU tensor");
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    check_runtime_healthy(state);
    TORCH_CHECK(state.wave_active, "MemFabric publish without active wave");
    TORCH_CHECK(chunk_idx >= 0 &&
                    chunk_idx < static_cast<int64_t>(state.active_chunks),
                "chunk_idx out of active wave range: ", chunk_idx);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    const int ret = mf310p_publish_chunk_async(
        state.ctx,
        static_cast<uint32_t>(chunk_idx),
        state.layout.chunk_bytes,
        reinterpret_cast<void*>(stream));
    TORCH_CHECK(ret == 0,
                "mf310p_publish_chunk_async failed for chunk ", chunk_idx,
                " with ret=", ret);
}

void memfabric_o_proj_finish(const at::Tensor& recv)
{
    TORCH_CHECK(recv.is_privateuseone(), "recv must be an NPU tensor");
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    check_runtime_healthy(state);
    TORCH_CHECK(state.wave_active, "MemFabric finish without active wave");

    int ret = mf310p_wait_wave(state.ctx);
    TORCH_CHECK(ret == 0, "mf310p_wait_wave failed with ret=", ret);

    const aclError sync_ret = aclrtSynchronizeStream(state.reduce_stream);
    TORCH_CHECK(sync_ret == ACL_SUCCESS,
                "reduce stream synchronization failed, ret=", sync_ret);

    uint32_t main_ret = 0;
    uint32_t stage = 0;
    uint32_t sq_head = 0;
    ret = mf310p_get_result(state.ctx, &main_ret, &stage, &sq_head);
    TORCH_CHECK(ret == 0,
                "mf310p_get_result failed with ret=", ret);
    TORCH_CHECK(main_ret == 0,
                "MemFabric AICPU/SDMA pipeline failed: main_ret=", main_ret,
                ", stage=", stage, ", sq_head=", sq_head);

    state.wave_active = false;
    state.active_chunks = 0;
}

void memfabric_o_proj_shutdown()
{
#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.ctx == nullptr) {
        return;
    }
    /* An armed wave is unsafe to destroy (outstanding SDMA/SQE/flag state);
     * leak instead of tearing down mid-flight in a dying process. */
    if (state.wave_active) {
        state.poisoned = true;
        return;
    }
    if (state.reduce_stream != nullptr) {
        (void)aclrtSynchronizeStream(state.reduce_stream);
        (void)aclrtDestroyStream(state.reduce_stream);
        state.reduce_stream = nullptr;
    }
    (void)mf310p_destroy(state.ctx);
    state.ctx = nullptr;
#endif
}

void memfabric_o_proj_mark_failed(
    const at::Tensor& recv,
    c10::string_view reason)
{
    TORCH_CHECK(recv.is_privateuseone(), "recv must be an NPU tensor");
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> guard(state.mutex);

    /*
     * Do not attempt to clear flags, destroy SHM, or synchronize an unknown
     * partial wave here. A peer may still be polling or AICPU may still own
     * SQEs. Mark the process-local runtime permanently unusable and require a
     * coordinated two-rank worker restart.
     */
    state.poisoned = true;
    state.wave_active = false;
    state.active_chunks = 0;
    if (state.failure_reason.empty()) {
        state.failure_reason.assign(reason.data(), reason.size());
    }
}

at::Tensor memfabric_w8a8_o_proj_allreduce_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& deq_scale,
    const c10::optional<at::Tensor>& quant_bias,
    int64_t tp_rank,
    int64_t tile_m)
{
    /*
     * Phase-2 direct AscendC/CATLASS tiled producer is not implemented yet.
     * The binding declares this entry point so the op schema exists, but
     * calling it must fail fast instead of silently falling back to HCCL.
     * Model traffic uses the phase-1 staged begin/publish/finish pipeline.
     */
    (void)x;
    (void)weight;
    (void)deq_scale;
    (void)quant_bias;
    (void)tp_rank;
    (void)tile_m;
    TORCH_CHECK(
        false,
        "Phase-2 direct MemFabric W8A8 o_proj producer is not implemented yet. "
        "The phase-1 staged pipeline (memfabric_o_proj_begin/publish/finish) "
        "is the supported path on this build.");
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

std::tuple<at::Tensor, at::Tensor> memfabric_o_proj_begin(
    const at::Tensor&, int64_t, int64_t, int64_t)
{
    memfabric_not_built();
}

void memfabric_o_proj_publish(const at::Tensor&, int64_t)
{
    memfabric_not_built();
}

void memfabric_o_proj_finish(const at::Tensor&)
{
    memfabric_not_built();
}

void memfabric_o_proj_mark_failed(const at::Tensor&, c10::string_view)
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
