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

#include <torch/library.h>

#ifdef ASCEND_PLATFORM_310P
#include "memfabric_o_proj/memfabric_o_proj_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def(
        "memfabric_w8a8_o_proj_allreduce("
        "Tensor x, Tensor weight, Tensor deq_scale, Tensor? quant_bias, "
        "int tp_rank, int tile_m) -> Tensor");
    ops.impl(
        "memfabric_w8a8_o_proj_allreduce",
        torch::kPrivateUse1,
        &vllm_ascend::memfabric_w8a8_o_proj_allreduce);
}
#endif
