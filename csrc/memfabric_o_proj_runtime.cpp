/*
 * Runtime bridge from vllm_ascend_C to the independently built customized
 * MemFabric 310P adapter.  This file has no dependency on MemFabric headers or
 * libraries: it uses a small stable C ABI and dlopen/dlsym.
 */
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#ifdef ASCEND_PLATFORM_310P
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "memfabric_o_proj/external/memfabric310p_adapter_api.h"

namespace vllm_ascend {
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

const char* required_env(const char* name)
{
    const char* value = std::getenv(name);
    TORCH_CHECK(
        value != nullptr && *value != '\0',
        name,
        " is required when the 310P MemFabric o_proj path is enabled. It must "
        "point to the separately built vllm_ascend_memfabric310p_adapter.so.");
    return value;
}

template <typename T>
T load_symbol(void* handle, const char* name)
{
    dlerror();
    void* ptr = dlsym(handle, name);
    const char* err = dlerror();
    TORCH_CHECK(ptr != nullptr && err == nullptr,
                "Missing symbol ", name, " in customized MemFabric adapter: ",
                (err == nullptr ? "unknown error" : err));
    return reinterpret_cast<T>(ptr);
}

struct AdapterFns {
    void* so = nullptr;
    decltype(&mf310p_adapter_abi_version) abi_version = nullptr;
    decltype(&mf310p_create) create = nullptr;
    decltype(&mf310p_destroy) destroy = nullptr;
    decltype(&mf310p_get_layout) get_layout = nullptr;
    decltype(&mf310p_reset_mailbox) reset_mailbox = nullptr;
    decltype(&mf310p_submit_wave) submit_wave = nullptr;
    decltype(&mf310p_wait_wave) wait_wave = nullptr;
    decltype(&mf310p_get_result) get_result = nullptr;
    decltype(&mf310p_publish_chunk_async) publish_chunk_async = nullptr;
    decltype(&mf310p_launch_reduce_consumer_async) launch_reduce_consumer_async = nullptr;
};

struct RuntimeState {
    std::mutex mutex;
    AdapterFns api;
    mf310p_context_t* ctx = nullptr;
    mf310p_layout_t layout{};
    aclrtStream reduce_stream = nullptr;
    int tp_rank = -1;
    int64_t tile_m = 0;
    bool wave_active = false;
    uint32_t active_chunks = 0;

    ~RuntimeState()
    {
        if (reduce_stream != nullptr) {
            (void)aclrtDestroyStream(reduce_stream);
        }
        if (ctx != nullptr && api.destroy != nullptr) {
            (void)api.destroy(ctx);
        }
        if (api.so != nullptr) {
            (void)dlclose(api.so);
        }
    }
};

RuntimeState& runtime_state()
{
    static RuntimeState state;
    return state;
}

void load_adapter_locked(RuntimeState& state)
{
    if (state.api.so != nullptr) {
        return;
    }
    const char* path = required_env("VLLM_ASCEND_310P_MEMFABRIC_ADAPTER_SO");
    state.api.so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(state.api.so != nullptr,
                "Failed to dlopen customized MemFabric adapter ", path, ": ",
                dlerror());

    state.api.abi_version = load_symbol<decltype(state.api.abi_version)>(
        state.api.so, "mf310p_adapter_abi_version");
    state.api.create = load_symbol<decltype(state.api.create)>(
        state.api.so, "mf310p_create");
    state.api.destroy = load_symbol<decltype(state.api.destroy)>(
        state.api.so, "mf310p_destroy");
    state.api.get_layout = load_symbol<decltype(state.api.get_layout)>(
        state.api.so, "mf310p_get_layout");
    state.api.reset_mailbox = load_symbol<decltype(state.api.reset_mailbox)>(
        state.api.so, "mf310p_reset_mailbox");
    state.api.submit_wave = load_symbol<decltype(state.api.submit_wave)>(
        state.api.so, "mf310p_submit_wave");
    state.api.wait_wave = load_symbol<decltype(state.api.wait_wave)>(
        state.api.so, "mf310p_wait_wave");
    state.api.get_result = load_symbol<decltype(state.api.get_result)>(
        state.api.so, "mf310p_get_result");
    state.api.publish_chunk_async = load_symbol<decltype(state.api.publish_chunk_async)>(
        state.api.so, "mf310p_publish_chunk_async");
    state.api.launch_reduce_consumer_async =
        load_symbol<decltype(state.api.launch_reduce_consumer_async)>(
            state.api.so, "mf310p_launch_reduce_consumer_async");

    TORCH_CHECK(
        state.api.abi_version() == VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION,
        "Customized MemFabric adapter ABI mismatch: expected ",
        VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION,
        ", got ", state.api.abi_version());
}

void init_context_locked(
    RuntimeState& state,
    const at::Tensor& x,
    int64_t tp_rank,
    int64_t tile_m)
{
    load_adapter_locked(state);
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

    const int ret = state.api.create(
        static_cast<int>(tp_rank),
        2,
        store_url,
        local_pool_bytes,
        arena_bytes,
        chunk_bytes,
        &state.ctx);
    TORCH_CHECK(ret == 0 && state.ctx != nullptr,
                "mf310p_create failed with ret=", ret);

    const int layout_ret = state.api.get_layout(state.ctx, &state.layout);
    TORCH_CHECK(layout_ret == 0,
                "mf310p_get_layout failed with ret=", layout_ret);
    TORCH_CHECK(state.layout.chunk_bytes == chunk_bytes,
                "MemFabric adapter returned unexpected chunk_bytes");
    TORCH_CHECK(state.layout.max_chunks == VLLM_ASCEND_MF310P_MAX_CHUNKS,
                "MemFabric adapter returned unexpected max_chunks");

    const aclError stream_ret = aclrtCreateStream(&state.reduce_stream);
    TORCH_CHECK(stream_ret == ACL_SUCCESS,
                "aclrtCreateStream(reduce) failed, ret=", stream_ret);

    state.tp_rank = static_cast<int>(tp_rank);
    state.tile_m = tile_m;
    (void)x; // x fixes the device through the caller's current NPU context.
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
    init_context_locked(state, x, tp_rank, tile_m);
    TORCH_CHECK(!state.wave_active,
                "MemFabric o_proj begin called while previous wave is active");

    int ret = state.api.reset_mailbox(state.ctx);
    TORCH_CHECK(ret == 0, "mf310p_reset_mailbox failed with ret=", ret);

    /* Consumer may poll before data arrives; that is intentional. */
    ret = state.api.launch_reduce_consumer_async(
        state.ctx,
        static_cast<uint32_t>(chunks),
        reinterpret_cast<void*>(state.reduce_stream));
    TORCH_CHECK(ret == 0,
                "mf310p_launch_reduce_consumer_async failed with ret=", ret);

    /* Arm AICPU/SDMA before producer notifications start arriving. */
    ret = state.api.submit_wave(state.ctx, static_cast<uint32_t>(chunks));
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
    TORCH_CHECK(state.wave_active, "MemFabric publish without active wave");
    TORCH_CHECK(chunk_idx >= 0 &&
                    chunk_idx < static_cast<int64_t>(state.active_chunks),
                "chunk_idx out of active wave range: ", chunk_idx);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    const int ret = state.api.publish_chunk_async(
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
    TORCH_CHECK(state.wave_active, "MemFabric finish without active wave");

    int ret = state.api.wait_wave(state.ctx);
    TORCH_CHECK(ret == 0, "mf310p_wait_wave failed with ret=", ret);

    const aclError sync_ret = aclrtSynchronizeStream(state.reduce_stream);
    TORCH_CHECK(sync_ret == ACL_SUCCESS,
                "reduce stream synchronization failed, ret=", sync_ret);

    uint32_t main_ret = 0;
    uint32_t stage = 0;
    uint32_t sq_head = 0;
    ret = state.api.get_result(state.ctx, &main_ret, &stage, &sq_head);
    TORCH_CHECK(ret == 0,
                "mf310p_get_result failed with ret=", ret);
    TORCH_CHECK(main_ret == 0,
                "MemFabric AICPU/SDMA pipeline failed: main_ret=", main_ret,
                ", stage=", stage, ", sq_head=", sq_head);

    state.wave_active = false;
    state.active_chunks = 0;
}

} // namespace vllm_ascend
#endif // ASCEND_PLATFORM_310P
