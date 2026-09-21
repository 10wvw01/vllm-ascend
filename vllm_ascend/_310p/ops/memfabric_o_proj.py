# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P3 TP=2 MemFabric fused full-attention o_proj helpers.

The fused path is intentionally narrow. It is only enabled for Qwen3.5/3.6
MoE full-attention ``self_attn.o_proj`` layers that are *unquantized*
(the real Qwen3.6-35B-A3B-w8a8 checkpoint keeps them FLOAT; owner decision A,
2026-09-18) running with TP=2. All other layers keep the existing
RowParallelLinear implementation. The checkpoint carries no torch_dtype, so
vLLM runs it as FP16 on 310P (BF16 NZ linear is unsupported by this CANN
anyway); the fused path therefore exchanges FP16 partial results.

The fused op runs on the V5 customized MemFabric (origin/wgm-dev-310p,
mailbox-ring epoch API): one fused AscendC FP16 matmul kernel per wave
writes each chunk straight into the symmetric send arena (matmul -> 64B-line
clean -> signal), the AICPU epoch kernel moves it to the peer die while the
producer continues, and a waiter kernel (quiet + wait x chunks) joins the
wave before the host-side ``add_out`` reduces send + recv into the output.
The M-bucket specialized kernel recipe was validated bit-exact against
``F.linear`` for m = 1..4096 on dav-2002; see the development plan P5 notes
for the details.
"""

from __future__ import annotations

import re
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
_MAX_CHUNKS_PER_WAVE = 64
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class MemFabricOProjPlan:
    """Static contract for the first 310P3 fused implementation."""

    tile_m: int
    input_size_per_partition: int = _EXPECTED_INPUT_SIZE_PER_PARTITION
    output_size: int = _EXPECTED_OUTPUT_SIZE
    tp_size: int = _EXPECTED_TP_SIZE
    max_chunks_per_wave: int = _MAX_CHUNKS_PER_WAVE

    @property
    def max_rows_per_wave(self) -> int:
        return self.tile_m * self.max_chunks_per_wave

    @property
    def chunk_bytes(self) -> int:
        # Communication payload is FP16 [tile_m, 2048].
        return self.tile_m * self.output_size * 2

    def chunks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        return (num_tokens + self.tile_m - 1) // self.tile_m


def get_memfabric_o_proj_plan() -> MemFabricOProjPlan:
    tile_m = int(envs.VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M)
    if tile_m <= 0:
        raise ValueError(f"VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M must be positive, got {tile_m}")
    return MemFabricOProjPlan(tile_m=tile_m)


def _target_text_config():
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return None
    text_config = vllm_config.model_config.hf_text_config
    if getattr(text_config, "model_type", None) != _QWEN35_MOE_TEXT_MODEL_TYPE:
        return None
    return text_config


def _is_full_attention_prefix(prefix: str, text_config) -> bool:
    """Cross-check prefix against Qwen3.5/3.6's configured hybrid layer type."""

    if not prefix.endswith(".self_attn.o_proj"):
        return False

    match = _LAYER_INDEX_RE.search(prefix)
    if match is None:
        return False
    layer_idx = int(match.group(1))

    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None or layer_idx >= len(layer_types):
        return False
    return layer_types[layer_idx] == "full_attention"


def should_enable_memfabric_o_proj(layer: torch.nn.Module) -> bool:
    """Return whether ``layer`` matches the deliberately narrow fused contract."""

    if not envs.VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ:
        return False

    text_config = _target_text_config()
    if text_config is None:
        return False

    prefix = getattr(layer, "prefix", "")
    if not _is_full_attention_prefix(prefix, text_config):
        return False

    if getattr(layer, "tp_size", None) != _EXPECTED_TP_SIZE:
        return False
    if getattr(layer, "input_size", None) != _EXPECTED_INPUT_SIZE:
        return False
    if getattr(layer, "input_size_per_partition", None) != _EXPECTED_INPUT_SIZE_PER_PARTITION:
        return False
    if getattr(layer, "output_size", None) != _EXPECTED_OUTPUT_SIZE:
        return False
    # The model runs FP16 on 310P (checkpoint has no torch_dtype and BF16 NZ
    # linear is unsupported here); the exchange/reduce kernel is FP16-only.
    if getattr(layer, "params_dtype", None) != torch.float16:
        return False

    # Owner decision A (2026-09-18): the target checkpoint's full-attention
    # o_proj is unquantized BF16, routed through the 310P unquantized linear
    # method. Accept either that method or the MemFabric dispatch subclass of
    # it (during quant-method routing quant_method may still be None).
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is not None:
        method_name = type(quant_method).__name__
        if method_name not in ("AscendUnquantizedLinearMethod", "MemFabricOProjLinearMethod310"):
            return False
        nested = getattr(quant_method, "quant_method", None)
        if nested is not None and type(nested).__name__ != "AscendUnquantizedLinearMethod":
            return False

    return True


def configure_memfabric_o_proj(layer: torch.nn.Module) -> bool:
    """Mark a target RowParallelLinear so the fused op owns the TP reduction."""

    enabled = should_enable_memfabric_o_proj(layer)
    layer._ascend_310p_memfabric_o_proj = enabled
    if not enabled:
        return False

    if not getattr(layer, "reduce_results", False):
        raise RuntimeError(
            "MemFabric o_proj fusion expects RowParallelLinear(reduce_results=True) "
            f"before patching, got {getattr(layer, 'prefix', '<unknown>')}"
        )

    # The fused path returns an already-reduced TP=2 result, so the generic
    # RowParallelLinear must not launch its HCCL all-reduce afterwards.
    layer.reduce_results = False
    plan = get_memfabric_o_proj_plan()
    logger.info_once(
        "Enable 310P3 TP=2 MemFabric unquantized full-attention o_proj pipeline "
        "(tile_m=%d, chunk_bytes=%d, FP16 exchange).",
        plan.tile_m,
        plan.chunk_bytes,
    )
    return True


def is_memfabric_o_proj_configured(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, "_ascend_310p_memfabric_o_proj", False))


def memfabric_o_proj_allreduce(
    layer: torch.nn.Module,
    x: torch.Tensor,
    tp_rank: int,
) -> torch.Tensor:
    """Run the fused MM/SDMA/reduce pipeline (unquantized, FP16).

    One opaque custom op per call: per wave it launches the fused AscendC
    producer (matmul tiles straight into the symmetric send arena, one
    mailbox signal per chunk), the waiter kernel (quiet + wait x chunks),
    and the stream-ordered ``add_out`` that reduces send + recv into ordinary
    NPU storage so the arenas can be safely reused by later layers.
    """

    if tp_rank not in (0, 1):
        raise RuntimeError(f"MemFabric o_proj fusion only supports TP rank 0/1, got {tp_rank}")
    if x.dtype != torch.float16:
        raise TypeError(f"MemFabric o_proj expects FP16 activation, got {x.dtype}")
    if x.dim() != 2 or x.shape[1] != _EXPECTED_INPUT_SIZE_PER_PARTITION:
        raise ValueError(f"MemFabric o_proj expects x=[M, 2048], got {tuple(x.shape)}")
    if layer.params_dtype != torch.float16:
        raise TypeError(f"MemFabric o_proj reduction currently requires FP16 output, got {layer.params_dtype}")

    plan = get_memfabric_o_proj_plan()

    direct = getattr(torch.ops._C_ascend, "memfabric_direct_o_proj_allreduce", None)
    if direct is None:
        raise RuntimeError("vllm_ascend_C was built without the fused 310P MemFabric o_proj op")
    return direct(x, layer.weight.data, int(tp_rank), int(plan.tile_m))


def make_memfabric_o_proj_linear_method():
    """Build the MemFabric dispatch linear method (lazy import, no cycles).

    The returned class subclasses the 310P unquantized linear method so the
    FLOAT-routed o_proj keeps its exact weight handling (NZ cast) while its
    apply() takes over matmul+TP-reduction for eligible layers.
    """
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

    class MemFabricOProjLinearMethod310(AscendUnquantizedLinearMethod):
        """Unquantized linear method that owns o_proj matmul + TP reduction.

        Selected by the 310P modelslim router for layers matching the
        MemFabric o_proj prefix/geometry contract; configure-time checks in
        ``configure_memfabric_o_proj`` remain the single eligibility source
        and can still reject this layer (leaving the stock behavior).
        """

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            super().process_weights_after_loading(layer)
            configure_memfabric_o_proj(layer)

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if is_memfabric_o_proj_configured(layer):
                if bias is not None:
                    # Qwen o_proj is bias-free; a biased RowParallel o_proj
                    # would need explicit rank0-only handling before fusion.
                    raise RuntimeError("MemFabric o_proj fusion does not support biased o_proj")
                from vllm.distributed import get_tensor_model_parallel_rank

                return memfabric_o_proj_allreduce(
                    layer=layer,
                    x=x,
                    tp_rank=get_tensor_model_parallel_rank(),
                )
            return super().apply(layer, x, bias)

    return MemFabricOProjLinearMethod310
