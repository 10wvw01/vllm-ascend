/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TORCH_ADPT_H
#define VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>
#include <torch/extension.h>

namespace vllm_ascend {

/* Fused o_proj matmul + TP=2 reduction using only the installed
 * wgm-dev-310p MemFabric public SHM/SDMA API. The single opaque op lives in
 * vllm_ascend_C; MemFabric transport internals remain outside vLLM. */
at::Tensor memfabric_direct_o_proj_allreduce_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t tp_rank,
    int64_t tile_m);

/* Explicitly tear down the process-persistent MemFabric context while the ACL
 * runtime is still alive. Also registered via atexit on first use; workers
 * should call this on shutdown for deterministic teardown. */
void memfabric_o_proj_shutdown();

/* Debug-only application snapshot of public pool geometry and vLLM-owned
 * arena words. It deliberately exposes no MemFabric internal state. */
at::Tensor memfabric_o_proj_debug_snapshot();

inline at::Tensor memfabric_direct_o_proj_allreduce(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t tp_rank,
    int64_t tile_m)
{
    TORCH_CHECK(x.is_privateuseone(), "x must be an NPU tensor");
    TORCH_CHECK(weight.is_privateuseone(), "weight must be an NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf,
                "x must be FP16, got ", x.scalar_type());
    TORCH_CHECK(weight.scalar_type() == at::kHalf,
                "weight must be FP16, got ", weight.scalar_type());
    TORCH_CHECK(x.dim() == 2,
                "x must be 2D [M, 2048], got dim=", x.dim());
    TORCH_CHECK(x.size(1) == 2048,
                "TP=2 Qwen3.5/3.6 o_proj expects K_local=2048, got ",
                x.size(1));
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == 2048 && weight.size(1) == 2048,
                "TP=2 Qwen3.5/3.6 o_proj expects weight [2048, 2048], got ",
                weight.sizes());
    TORCH_CHECK(tp_rank == 0 || tp_rank == 1,
                "only TP rank 0/1 is supported, got ", tp_rank);
    TORCH_CHECK(tile_m > 0, "tile_m must be positive, got ", tile_m);

    return memfabric_direct_o_proj_allreduce_impl(x, weight, tp_rank, tile_m);
}

} // namespace vllm_ascend

#endif // VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TORCH_ADPT_H
