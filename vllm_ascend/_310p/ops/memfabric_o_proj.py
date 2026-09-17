# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P3 TP=2 MemFabric fused full-attention o_proj helpers.

The fused path is intentionally narrow.  It is only enabled for Qwen3.5/3.6
MoE full-attention ``self_attn.o_proj`` layers running static W8A8 with TP=2.
All other layers keep the existing RowParallelLinear implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import logger

from vllm_ascend import envs

_QWEN35_MOE_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
_EXPECTED_INPUT_SIZE = 4096
_EXPECTED_INPUT_SIZE_PER_PARTITION = 2048
_EXPECTED_OUTPUT_SIZE = 2048
_EXPECTED_TP_SIZE = 2


@dataclass(frozen=True)
class MemFabricOProjPlan:
    """Static contract for the first 310P3 fused implementation."""

    tile_m: int
    input_size_per_partition: int = _EXPECTED_INPUT_SIZE_PER_PARTITION
    output_size: int = _EXPECTED_OUTPUT_SIZE
    tp_size: int = _EXPECTED_TP_SIZE

    def chunks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        return (num_tokens + self.tile_m - 1) // self.tile_m


def get_memfabric_o_proj_plan() -> MemFabricOProjPlan:
    tile_m = int(envs.VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M)
    if tile_m <= 0:
        raise ValueError(
            "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M must be positive, "
            f"got {tile_m}"
        )
    return MemFabricOProjPlan(tile_m=tile_m)


def _is_target_model() -> bool:
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return False
    text_config = vllm_config.model_config.hf_text_config
    return getattr(text_config, "model_type", None) == _QWEN35_MOE_TEXT_MODEL_TYPE


def should_enable_memfabric_o_proj(layer: torch.nn.Module) -> bool:
    """Return whether ``layer`` matches the deliberately narrow fused contract."""

    if not envs.VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ:
        return False
    if not _is_target_model():
        return False

    prefix = getattr(layer, "prefix", "")
    if not prefix.endswith(".self_attn.o_proj"):
        return False

    if getattr(layer, "tp_size", None) != _EXPECTED_TP_SIZE:
        return False
    if getattr(layer, "input_size", None) != _EXPECTED_INPUT_SIZE:
        return False
    if getattr(layer, "input_size_per_partition", None) != _EXPECTED_INPUT_SIZE_PER_PARTITION:
        return False
    if getattr(layer, "output_size", None) != _EXPECTED_OUTPUT_SIZE:
        return False

    quant_method = getattr(layer, "quant_method", None)
    scheme = getattr(quant_method, "quant_method", None)
    if scheme is None or scheme.__class__.__name__ != "AscendW8A8LinearMethod310":
        return False

    return True


def configure_memfabric_o_proj(layer: torch.nn.Module) -> bool:
    """Mark a target RowParallelLinear so the fused op owns the TP reduction."""

    enabled = should_enable_memfabric_o_proj(layer)
    setattr(layer, "_ascend_310p_memfabric_o_proj", enabled)
    if not enabled:
        return False

    if not getattr(layer, "reduce_results", False):
        raise RuntimeError(
            "MemFabric o_proj fusion expects RowParallelLinear(reduce_results=True) "
            f"before patching, got {getattr(layer, 'prefix', '<unknown>')}"
        )

    # The fused operator returns the already-reduced TP=2 result, therefore the
    # generic RowParallelLinear must not launch its own all-reduce afterwards.
    layer.reduce_results = False
    plan = get_memfabric_o_proj_plan()
    logger.info_once(
        "Enable 310P3 TP=2 MemFabric W8A8 full-attention o_proj fusion "
        "(tile_m=%d).",
        plan.tile_m,
    )
    return True


def is_memfabric_o_proj_configured(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, "_ascend_310p_memfabric_o_proj", False))


def memfabric_w8a8_o_proj_allreduce(
    layer: torch.nn.Module,
    x_q: torch.Tensor,
    quant_bias: torch.Tensor | None,
    tp_rank: int,
) -> torch.Tensor:
    """Invoke the opaque fused operator after input activation quantization.

    ``x_q`` is already int8.  The C++/Ascend implementation owns tiled matmul,
    MemFabric exchange and the final local reduction.  Bias is supplied only by
    TP rank 0, preserving RowParallelLinear's existing semantics.
    """

    if tp_rank not in (0, 1):
        raise RuntimeError(f"MemFabric o_proj fusion only supports TP rank 0/1, got {tp_rank}")
    if x_q.dtype != torch.int8:
        raise TypeError(f"MemFabric o_proj expects int8 activation, got {x_q.dtype}")

    plan = get_memfabric_o_proj_plan()
    op_namespace = getattr(torch.ops, "_C_ascend", None)
    op = getattr(op_namespace, "memfabric_w8a8_o_proj_allreduce", None) if op_namespace is not None else None
    if op is None:
        raise RuntimeError(
            "vllm_ascend_C was built without the 310P MemFabric o_proj operator. "
            "Rebuild vllm-ascend with the MemFabric operator enabled."
        )

    return op(
        x_q,
        layer.weight.data,
        layer.deq_scale,
        quant_bias,
        int(tp_rank),
        int(plan.tile_m),
    )
