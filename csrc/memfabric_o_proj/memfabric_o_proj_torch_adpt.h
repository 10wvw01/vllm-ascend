/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TORCH_ADPT_H
#define VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <torch/extension.h>

namespace vllm_ascend {

#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
at::Tensor memfabric_w8a8_o_proj_allreduce_impl(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& deq_scale,
    const c10::optional<at::Tensor>& quant_bias,
    int64_t tp_rank,
    int64_t tile_m);
#endif

inline at::Tensor memfabric_w8a8_o_proj_allreduce(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& deq_scale,
    const c10::optional<at::Tensor>& quant_bias,
    int64_t tp_rank,
    int64_t tile_m)
{
    TORCH_CHECK(x.is_privateuseone(), "x must be an NPU tensor");
    TORCH_CHECK(weight.is_privateuseone(), "weight must be an NPU tensor");
    TORCH_CHECK(deq_scale.is_privateuseone(), "deq_scale must be an NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kChar,
                "x must be int8, got ", x.scalar_type());
    TORCH_CHECK(weight.scalar_type() == at::kChar,
                "weight must be int8, got ", weight.scalar_type());
    TORCH_CHECK(x.dim() == 2,
                "x must be 2D [M, 2048], got dim=", x.dim());
    TORCH_CHECK(x.size(1) == 2048,
                "TP=2 Qwen3.5/3.6 o_proj expects K_local=2048, got ",
                x.size(1));
    TORCH_CHECK(tp_rank == 0 || tp_rank == 1,
                "only TP rank 0/1 is supported, got ", tp_rank);
    TORCH_CHECK(tile_m > 0, "tile_m must be positive, got ", tile_m);
    TORCH_CHECK(deq_scale.numel() == 2048,
                "o_proj deq_scale must have 2048 elements, got ",
                deq_scale.numel());
    if (quant_bias.has_value()) {
        TORCH_CHECK(tp_rank == 0,
                    "quant_bias must only be supplied by TP rank 0");
        TORCH_CHECK(quant_bias->numel() == 2048,
                    "o_proj quant_bias must have 2048 elements, got ",
                    quant_bias->numel());
    }

#ifdef VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ
    return memfabric_w8a8_o_proj_allreduce_impl(
        x, weight, deq_scale, quant_bias, tp_rank, tile_m);
#else
    TORCH_CHECK(
        false,
        "memfabric_w8a8_o_proj_allreduce is registered, but this "
        "vllm_ascend_C build does not contain the 310P MemFabric runtime. "
        "Rebuild with VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ=ON.");
#endif
}

} // namespace vllm_ascend

#endif // VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TORCH_ADPT_H
