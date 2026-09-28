#include <ATen/ATen.h>
#include <torch/library.h>

#ifdef ASCEND_PLATFORM_310P
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

#include <cstdint>

#include "../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
namespace {
constexpr int64_t kWidth = 2048;
constexpr uint32_t kExpectedRuntimeAbi = 1u;

bool IsCapturing()
{
    return c10_npu::currentStreamCaptureStatusMayInitCtx() ==
           c10_npu::CaptureStatus::Active;
}

using RuntimeAbiFn = uint32_t (*)();
using RuntimeShutdownFn = int (*)();
using RuntimeDebugFn = int (*)(uint64_t*, size_t);

RuntimeAbiFn RuntimeAbi()
{
    return reinterpret_cast<RuntimeAbiFn>(
        GetOpApiFuncAddr("mfmc2RuntimeAbiVersion"));
}

}  // namespace

bool memfabric_matmul_allreduce_available()
{
    const auto phase1 =
        GetOpApiFuncAddr("aclnnMemFabricMatmulAllReduceGetWorkspaceSize");
    const auto phase2 = GetOpApiFuncAddr("aclnnMemFabricMatmulAllReduce");
    const auto abi = RuntimeAbi();
    return phase1 != nullptr && phase2 != nullptr && abi != nullptr &&
           abi() == kExpectedRuntimeAbi;
}

at::Tensor memfabric_matmul_allreduce(const at::Tensor& x,
                                      const at::Tensor& weight,
                                      int64_t tp_rank,
                                      int64_t batch_basem_count)
{
    TORCH_CHECK(memfabric_matmul_allreduce_available(),
                "MemFabricMatmulAllReduce custom OPP is not installed or has "
                "an incompatible runtime ABI. Install the memfabric_mc2 .run "
                "package and source its set_env.bash before starting vLLM.");
    TORCH_CHECK(x.is_privateuseone(), "x must be an NPU tensor");
    TORCH_CHECK(weight.is_privateuseone(), "weight must be an NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be FP16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be FP16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == kWidth,
                "MemFabricMatmulAllReduce expects x=[M,2048], got ", x.sizes());
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == kWidth &&
                    weight.size(1) == kWidth,
                "MemFabricMatmulAllReduce expects weight=[2048,2048], got ",
                weight.sizes());
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be logically contiguous");
    TORCH_CHECK(tp_rank == 0 || tp_rank == 1,
                "MemFabricMatmulAllReduce supports TP rank 0/1 only");
    TORCH_CHECK(batch_basem_count == 1 || batch_basem_count == 2 ||
                    batch_basem_count == 4,
                "batch_basem_count must be one of {1,2,4}");

    const int64_t rows = x.size(0);
    at::Tensor output = at::empty({rows, kWidth}, x.options());
    if (rows == 0) return output;

    const uint64_t x_addr =
        reinterpret_cast<uint64_t>(x.const_data_ptr());
    const uint64_t weight_addr =
        reinterpret_cast<uint64_t>(weight.const_data_ptr());
    const uint64_t output_addr =
        reinterpret_cast<uint64_t>(output.data_ptr());
    const bool capturing = IsCapturing();

    EXEC_NPU_CMD(aclnnMemFabricMatmulAllReduce,
                 x_addr,
                 weight_addr,
                 rows,
                 tp_rank,
                 batch_basem_count,
                 output_addr,
                 capturing);
    return output;
}

void memfabric_matmul_allreduce_shutdown()
{
    auto fn = reinterpret_cast<RuntimeShutdownFn>(
        GetOpApiFuncAddr("mfmc2RuntimeShutdown"));
    if (fn == nullptr) return;
    const int ret = fn();
    TORCH_CHECK(ret == 0, "mfmc2RuntimeShutdown failed, ret=", ret);
}

at::Tensor memfabric_matmul_allreduce_debug_snapshot()
{
    constexpr int64_t kWords = 24;
    at::Tensor out = at::zeros({kWords}, at::TensorOptions().dtype(at::kLong));
    auto fn = reinterpret_cast<RuntimeDebugFn>(
        GetOpApiFuncAddr("mfmc2RuntimeDebugSnapshot"));
    if (fn == nullptr) return out;
    const int ret = fn(reinterpret_cast<uint64_t*>(out.data_ptr<int64_t>()),
                       static_cast<size_t>(kWords));
    TORCH_CHECK(ret == 0, "mfmc2RuntimeDebugSnapshot failed, ret=", ret);
    return out;
}

}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("memfabric_matmul_allreduce("
            "Tensor x, Tensor weight, int tp_rank, int batch_basem_count) -> Tensor");
    ops.impl("memfabric_matmul_allreduce", torch::kPrivateUse1,
             &vllm_ascend::memfabric_matmul_allreduce);

    ops.def("memfabric_matmul_allreduce_available() -> bool");
    ops.impl("memfabric_matmul_allreduce_available",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::memfabric_matmul_allreduce_available);

    ops.def("memfabric_matmul_allreduce_shutdown() -> ()");
    ops.impl("memfabric_matmul_allreduce_shutdown",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::memfabric_matmul_allreduce_shutdown);

    ops.def("memfabric_matmul_allreduce_debug_snapshot() -> Tensor");
    ops.impl("memfabric_matmul_allreduce_debug_snapshot",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::memfabric_matmul_allreduce_debug_snapshot);
}
#endif
