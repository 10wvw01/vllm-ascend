/*
 * Experimental TP=2 Qwen V2 serving bridge.
 *
 * No V1 runtime or device kernel is modified. The already device-validated
 * V2 trial MM+SDMA+Add kernel is wrapped as one PyTorch op, with an explicit
 * >=2048-token routing contract handled by the Python linear method.
 *
 * IMPORTANT: first-write MTE3 mitigation is provisional, not silicon fix.
 * Pool outputs are allocated BEFORE MemFabric creation and remain pinned for
 * this worker lifetime. Each output address is exercised with a real two-rank
 * wave during eager initialization; T0's device-sync guard runs ONLY during
 * this initialization, not repeatedly as the four pinned slots rotate.
 * The current platform may still fail for a new kernel/runtime combination:
 * correctness and startup skew MUST be tested on 310P3.
 */
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "mf310p_v2_trial_adapter_api.h"

namespace vllm_ascend {
void memfabric_mm_ar_v2_shutdown();
namespace {

constexpr int64_t kWidth = 2048;
constexpr int64_t kMaxWindowRows = 8192;
constexpr int64_t kThreshold = 2048;
constexpr int kDefaultPoolSlots = 4;

struct V2ServeState {
    std::mutex mutex;
    mf310p_v2_trial_t* handle = nullptr;
    int rank = -1;
    uint32_t chunk_rows = 0;
    aclrtStream execution_stream = nullptr;
    bool poisoned = false;
    std::string failure;
    // Each slot has a fixed pre-pool allocator lifetime. A returned narrow
    // view owns a StorageImpl ref; a slot may be reused only after all views
    // have gone. Reusing still-live outputs is a correctness bug.
    std::vector<at::Tensor> output_slots;
    at::Tensor input_stage;
};

V2ServeState& state()
{
    static V2ServeState s;
    return s;
}

int configured_slot_count()
{
    const char* v = std::getenv("VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS");
    if (v == nullptr || *v == '\0') return kDefaultPoolSlots;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    TORCH_CHECK(end != v && *end == '\0' && n >= 2 && n <= 16,
                "V2 output slots must be within [2,16], got ", v);
    return static_cast<int>(n);
}

const char* configured_store_url()
{
    const char* v = std::getenv("VLLM_ASCEND_310P_MEMFABRIC_STORE_URL");
    return v && *v ? v : "tcp://127.0.0.1:8581";
}

void checked(int rc, const char* op)
{
    TORCH_CHECK(rc == 0, "V2 MemFabric serving ", op, " failed (rc=", rc,
                "). Restart BOTH TP=2 workers; a partially failed "
                "SDMA wave cannot be safely continued.");
}

void check_protocol(V2ServeState& s, const char* where)
{
    uint64_t status = ~0ULL;
    checked(mf310p_v2_trial_check_status(s.handle, &status), where);
    TORCH_CHECK(status == 0, "V2 serving ", where,
                " protocol status=", status);
}

void warm_outputs_locked(V2ServeState& s, const at::Tensor& x,
                         const at::Tensor& w, aclrtStream stream)
{
    // m >= 2048 is guaranteed by the public serving API. Limit the warm
    // wave to two chunks: use the same real MM/SMM protocol as steady state.
    const uint32_t warm_rows = 2u * s.chunk_rows;
    TORCH_CHECK(x.size(0) >= static_cast<int64_t>(warm_rows),
                "V2 bootstrap requires at least two complete chunks");
    const uint64_t input = reinterpret_cast<uint64_t>(x.const_data_ptr());
    const uint64_t weight = reinterpret_cast<uint64_t>(w.const_data_ptr());
    for (auto& slot : s.output_slots) {
        checked(mf310p_v2_trial_run(
                    s.handle, input, weight,
                    reinterpret_cast<uint64_t>(slot.data_ptr()),
                    warm_rows, 1, reinterpret_cast<void*>(stream)),
                "real-wave output prewarm");
        const aclError sync = aclrtSynchronizeStream(stream);
        TORCH_CHECK(sync == ACL_SUCCESS,
                    "V2 prewarm stream sync failed: ", sync);
        check_protocol(s, "prewarm");
    }
}

void init_locked(V2ServeState& s, const at::Tensor& x,
                 const at::Tensor& w, int64_t tp_rank,
                 uint32_t chunk_rows, aclrtStream stream)
{
    if (s.handle != nullptr) {
        TORCH_CHECK(s.rank == tp_rank && s.chunk_rows == chunk_rows,
                    "V2 serving context geometry changed; restart workers");
        TORCH_CHECK(s.execution_stream == stream,
                    "V2 serving is single-stream only; mixed streams "
                    "must be explicitly synchronized and reinstrumented");
        return;
    }

    // Output slots and the padded input are allocated *before* create().
    // This is a platform first-MTE3-write mitigation, not a proven cure.
    // All allocations are pinned in this state until worker shutdown.
    const int count = configured_slot_count();
    s.output_slots.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        s.output_slots.push_back(
            at::empty({kMaxWindowRows, kWidth}, x.options()));
    }
    s.input_stage = at::zeros({kMaxWindowRows, kWidth}, x.options());
    // Ensure all pre-pool buffer preparations are actually complete.
    const aclError init_sync = aclrtSynchronizeStream(stream);
    TORCH_CHECK(init_sync == ACL_SUCCESS,
                "V2 pre-pool allocation/init sync failed ", init_sync);
    s.rank = static_cast<int>(tp_rank);
    s.chunk_rows = chunk_rows;
    s.execution_stream = stream;

    checked(mf310p_v2_trial_create(s.rank, configured_store_url(),
                                   chunk_rows, &s.handle),
            "pool create / geometry exchange");
    static bool shutdown_hook_registered = false;
    if (!shutdown_hook_registered) {
        TORCH_CHECK(std::atexit(&memfabric_mm_ar_v2_shutdown) == 0,
                    "V2 failed to register worker shutdown hook");
        shutdown_hook_registered = true;
    }
    checked(mf310p_v2_trial_ib_probe(s.handle, 1000u,
                                     reinterpret_cast<void*>(stream)),
            "8-core IB probe");
    warm_outputs_locked(s, x, w, stream);
}

at::Tensor reserve_slot_locked(V2ServeState& s)
{
    for (auto& slot : s.output_slots) {
        // storage() returns a temporary Storage handle: its extra ref
        // brings the idle count to exactly two (state slot + temporary).
        // A narrow/reshape result held by model code increments that count.
        // Never return an in-use slot, including across layer calls.
        if (slot.storage().use_count() == 2) {
            return slot;
        }
    }
    TORCH_CHECK(false,
        "V2 pooled outputs exhausted: all ", s.output_slots.size(),
        " slots are still referenced by model tensors. Increase "
        "VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS (max 16), "
        "or reduce in-flight requests. Refusing unsafe alias/reuse.");
    return at::Tensor();  // unreachable after TORCH_CHECK; satisfy C++ diagnostics
}

void run_window_locked(V2ServeState& s, const at::Tensor& x,
                       const at::Tensor& w, int64_t start, int64_t rows,
                       at::Tensor& output, aclrtStream stream)
{
    const int64_t padded = std::max<int64_t>(
        kThreshold,
        ((rows + static_cast<int64_t>(s.chunk_rows) - 1) /
         s.chunk_rows) * s.chunk_rows);
    TORCH_CHECK(padded >= kThreshold && padded <= kMaxWindowRows,
                "V2 invalid padded window size ", padded);
    // For an aligned, contiguous full window, avoid staging entirely.
    // For unaligned tails, use the resident stage tensor (row-local MM
    // semantics: stale data above rows is never returned to the caller).
    at::Tensor source;
    if (rows == padded) {
        source = x.narrow(0, start, rows);
    } else {
        // A torch NPU vector copy is used instead of raw aclrtMemcpyAsync;
        // T2 showed the latter can observe stale producer data on 310P3.
        // This stage is same-stream and serialized before the V2 kernel.
        s.input_stage.narrow(0, 0, rows).copy_(
            x.narrow(0, start, rows));
        source = s.input_stage;
    }
    checked(mf310p_v2_trial_run(
                s.handle,
                reinterpret_cast<uint64_t>(source.const_data_ptr()),
                reinterpret_cast<uint64_t>(w.const_data_ptr()),
                reinterpret_cast<uint64_t>(output.data_ptr()),
                static_cast<uint32_t>(padded), 1,
                reinterpret_cast<void*>(stream)),
            "MM+MemFabric overlap");

    // Status is surfaced by the device fail-stop path asynchronously.
    // Host synchronization on every invocation would destroy the measured
    // overlap and Graph compatibility; the test harness must drain and
    // inspect status explicitly under stress and before process shutdown.
}

}  // namespace

at::Tensor memfabric_mm_ar_v2_allreduce(
    const at::Tensor& x, const at::Tensor& weight, int64_t tp_rank,
    int64_t chunk_rows)
{
    TORCH_CHECK(x.is_privateuseone() && weight.is_privateuseone(),
                "V2 requires NPU activation and weight");
    TORCH_CHECK(x.scalar_type() == at::kHalf &&
                    weight.scalar_type() == at::kHalf,
                "V2 only supports FP16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == kWidth &&
                weight.dim() == 2 && weight.size(0) == kWidth &&
                weight.size(1) == kWidth,
                "V2 geometry requires X[M,2048], W_NZ[2048,2048]");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "V2 requires contiguous X and NZ W");
    TORCH_CHECK(x.size(0) >= kThreshold,
                "V2 invoked on small M: ", x.size(0),
                " (small M must use native MM + HCCL)");
    TORCH_CHECK(tp_rank == 0 || tp_rank == 1, "V2 requires TP=2 rank 0/1");
    TORCH_CHECK(chunk_rows == 256 || chunk_rows == 512 ||
                    chunk_rows == 1024,
                "V2 chunk rows must be 256/512/1024");
    TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx() !=
                    c10_npu::CaptureStatus::Active,
                "V2 large-M prefill during ACL Graph capture is not yet "
                "supported. Configure FULL_DECODE_ONLY with capture "
                "sizes below M=2048; decode uses Stock HCCL.");
    const auto stream = c10_npu::getCurrentNPUStream().stream();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    TORCH_CHECK(!s.poisoned, "V2 serving runtime poisoned: ", s.failure);
    try {
        init_locked(s, x, weight, tp_rank,
                    static_cast<uint32_t>(chunk_rows), stream);
        const int64_t rows = x.size(0);
        if (rows <= kMaxWindowRows) {
            // Use a reusable output allocation whose first device write was
            // already exercised while initializing the MemFabric context.
            at::Tensor slot = reserve_slot_locked(s);
            run_window_locked(s, x, weight, 0, rows, slot, stream);
            return slot.narrow(0, 0, rows);
        }
        // Rare >8192 packed-prefill: sequential windows preserve TP
        // protocol ordering, and copy to framework-owned contiguous output.
        // Each window's copy must be enqueued before slot reuse.
        at::Tensor full = at::empty({rows, kWidth}, x.options());
        for (int64_t start = 0; start < rows; start += kMaxWindowRows) {
            const int64_t valid =
                std::min<int64_t>(kMaxWindowRows, rows - start);
            at::Tensor slot = reserve_slot_locked(s);
            run_window_locked(s, x, weight, start, valid, slot, stream);
            full.narrow(0, start, valid).copy_(
                slot.narrow(0, 0, valid));
        }
        return full;
    } catch (const std::exception& e) {
        s.poisoned = true;
        s.failure = e.what();
        throw;
    }
}

int64_t memfabric_mm_ar_v2_debug_status()
{
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    TORCH_CHECK(s.handle != nullptr, "V2 diagnostic requires an initialized pool");
    const aclError sync = aclrtSynchronizeStream(s.execution_stream);
    TORCH_CHECK(sync == ACL_SUCCESS, "V2 diagnostic stream sync failed: ", sync);
    uint64_t protocol = ~0ULL;
    checked(mf310p_v2_trial_check_status(s.handle, &protocol),
            "debug status read");
    if (protocol != 0) {
        s.poisoned = true;
        s.failure = "device protocol status " + std::to_string(protocol);
    }
    return static_cast<int64_t>(protocol);
}

void memfabric_mm_ar_v2_shutdown()
{
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.handle) return;
    if (s.execution_stream != nullptr)
        (void)aclrtSynchronizeStream(s.execution_stream);
    (void)mf310p_v2_trial_destroy(s.handle);
    s.handle = nullptr;
    s.input_stage.reset();
    s.output_slots.clear();
}

}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def(
        "memfabric_mm_ar_v2_allreduce(Tensor x, Tensor weight, "
        "int tp_rank, int chunk_rows) -> Tensor");
    ops.impl("memfabric_mm_ar_v2_allreduce", torch::kPrivateUse1,
             &vllm_ascend::memfabric_mm_ar_v2_allreduce);
    ops.def("memfabric_mm_ar_v2_debug_status() -> int");
    ops.impl("memfabric_mm_ar_v2_debug_status",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::memfabric_mm_ar_v2_debug_status);
    ops.def("memfabric_mm_ar_v2_shutdown() -> ()");
    ops.impl("memfabric_mm_ar_v2_shutdown",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::memfabric_mm_ar_v2_shutdown);
}
