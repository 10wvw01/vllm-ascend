/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <torch/library.h>

#ifdef ASCEND_PLATFORM_310P
#include "memfabric_o_proj/memfabric_o_proj_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    /* Phase-1 staged pipeline used for correctness/overlap validation. */
    ops.def(
        "memfabric_o_proj_begin("
        "Tensor x, int tp_rank, int tile_m, int chunks) -> (Tensor send, Tensor recv)");
    ops.impl(
        "memfabric_o_proj_begin",
        torch::kPrivateUse1,
        &vllm_ascend::memfabric_o_proj_begin);

    ops.def("memfabric_o_proj_publish(Tensor send, int chunk_idx) -> ()");
    ops.impl(
        "memfabric_o_proj_publish",
        torch::kPrivateUse1,
        &vllm_ascend::memfabric_o_proj_publish);

    ops.def("memfabric_o_proj_finish(Tensor recv) -> ()");
    ops.impl(
        "memfabric_o_proj_finish",
        torch::kPrivateUse1,
        &vllm_ascend::memfabric_o_proj_finish);

    ops.def("memfabric_o_proj_mark_failed(Tensor recv, str reason) -> ()");
    ops.impl(
        "memfabric_o_proj_mark_failed",
        torch::kPrivateUse1,
        &vllm_ascend::memfabric_o_proj_mark_failed);

    /* Deterministic worker-lifetime teardown (also auto-registered via
     * atexit on first context creation). The schema is declared once and
     * implemented for both the PrivateUse1 key and a catch-all key so it is
     * callable without a tensor argument on any backend. */
    ops.def("memfabric_o_proj_shutdown() -> ()");
    ops.impl(
        "memfabric_o_proj_shutdown",
        torch::kPrivateUse1,
        &vllm_ascend::memfabric_o_proj_shutdown);
    ops.impl(
        "memfabric_o_proj_shutdown",
        c10::DispatchKey::CompositeExplicitAutograd,
        &vllm_ascend::memfabric_o_proj_shutdown);

    /* Reserved phase-2 single opaque op for direct AscendC matmul->SHM. */
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
