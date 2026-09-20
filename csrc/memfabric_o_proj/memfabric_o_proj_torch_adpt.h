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

/* Phase-1 staged pipeline. These functions live in vllm_ascend_C, which links
 * the repo-owned adapter (compiled against the installed wgm-dev-310p
 * MemFabric) directly at build time. */
std::tuple<at::Tensor, at::Tensor> memfabric_o_proj_begin(
    const at::Tensor& x,
    int64_t tp_rank,
    int64_t tile_m,
    int64_t chunks);

void memfabric_o_proj_publish(
    const at::Tensor& send,
    int64_t chunk_idx);

void memfabric_o_proj_finish(const at::Tensor& recv);

/* Mark a partially started wave unusable. A failed device/AICPU/SDMA pipeline
 * cannot be safely reused because outstanding polling/SQE/flag state is not
 * recoverable without a rank-wide restart. */
void memfabric_o_proj_mark_failed(
    const at::Tensor& recv,
    c10::string_view reason);

/* Explicitly tear down the process-persistent MemFabric context while the ACL
 * runtime is still alive. Also registered via atexit on first use; workers
 * should call this on shutdown for deterministic teardown. */
void memfabric_o_proj_shutdown();

/* Reserved phase-2 direct-MM entry point (owner decision A, 2026-09-18: the
 * target o_proj is unquantized BF16). Once the AscendC BF16 tiled matmul
 * writes directly into the symmetric send arena, Python will switch from the
 * staged begin/publish/finish path to this single opaque op. */
#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
at::Tensor memfabric_direct_o_proj_allreduce_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t tp_rank,
    int64_t tile_m);
#endif

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

#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
    return memfabric_direct_o_proj_allreduce_impl(x, weight, tp_rank, tile_m);
#else
    TORCH_CHECK(
        false,
        "Direct MemFabric BF16 o_proj producer is not built (feature off). "
        "The phase-1 staged pipeline uses memfabric_o_proj_begin/publish/"
        "finish instead.");
    return at::Tensor(); /* unreachable; keeps non-void signature well-formed */
#endif
}

} // namespace vllm_ascend

#endif // VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TORCH_ADPT_H
