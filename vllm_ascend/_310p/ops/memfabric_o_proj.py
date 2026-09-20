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

Phase 1 uses per-tile BF16 matmul (``F.linear``/``unquantized_gemm``
semantics, NZ weight layout included), stages each finished tile into
MemFabric symmetric memory, publishes it immediately, and lets AICPU/SDMA
plus a separate AICore reduce stream consume earlier tiles while the compute
stream continues with later matmuls. Phase 2 will replace only the producer
with a direct AscendC BF16 matmul that writes the SHM send arena.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
        # Phase-1/final communication payload is BF16 [tile_m, 2048].
        return self.tile_m * self.output_size * 2

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
    setattr(layer, "_ascend_310p_memfabric_o_proj", enabled)
    if not enabled:
        return False

    if not getattr(layer, "reduce_results", False):
        raise RuntimeError(
            "MemFabric o_proj fusion expects RowParallelLinear(reduce_results=True) "
            f"before patching, got {getattr(layer, 'prefix', '<unknown>')}"
        )

    # The staged/final fused path returns an already-reduced TP=2 result, so the
    # generic RowParallelLinear must not launch its HCCL all-reduce afterwards.
    layer.reduce_results = False
    plan = get_memfabric_o_proj_plan()
    logger.info_once(
        "Enable 310P3 TP=2 MemFabric W8A8 full-attention o_proj pipeline "
        "(tile_m=%d, chunk_bytes=%d).",
        plan.tile_m,
        plan.chunk_bytes,
    )
    return True


def is_memfabric_o_proj_configured(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, "_ascend_310p_memfabric_o_proj", False))


def _runtime_ops():
    namespace = getattr(torch.ops, "_C_ascend", None)
    if namespace is None:
        raise RuntimeError("vllm_ascend_C custom-op namespace is unavailable")
    begin = getattr(namespace, "memfabric_o_proj_begin", None)
    publish = getattr(namespace, "memfabric_o_proj_publish", None)
    finish = getattr(namespace, "memfabric_o_proj_finish", None)
    mark_failed = getattr(namespace, "memfabric_o_proj_mark_failed", None)
    if begin is None or publish is None or finish is None or mark_failed is None:
        raise RuntimeError(
            "vllm_ascend_C was built without the staged 310P MemFabric o_proj runtime"
        )
    return begin, publish, finish, mark_failed


def memfabric_o_proj_allreduce(
    layer: torch.nn.Module,
    x: torch.Tensor,
    tp_rank: int,
) -> torch.Tensor:
    """Run the phase-1 MM/SDMA/reduce overlap pipeline (unquantized, FP16).

    For each wave:
      1. arm peer receive/reduce and local AICPU/SDMA orchestration;
      2. compute one FP16 matmul tile (``F.linear`` semantics, matching the
         unquantized fallback exactly, NZ weight layout included);
      3. copy that tile into its immutable symmetric-memory send slot;
      4. publish the slot; SDMA can move it while the next matmul executes;
      5. after the wave joins, copy reduced recv/final rows to ordinary NPU
         storage so the symmetric arenas can be safely reused by later layers.

    The extra local copies are intentionally temporary. The phase-2 producer
    writes matmul results directly into ``send`` and removes them.
    """

    if tp_rank not in (0, 1):
        raise RuntimeError(f"MemFabric o_proj fusion only supports TP rank 0/1, got {tp_rank}")
    if x.dtype != torch.float16:
        raise TypeError(f"MemFabric o_proj expects FP16 activation, got {x.dtype}")
    if x.dim() != 2 or x.shape[1] != _EXPECTED_INPUT_SIZE_PER_PARTITION:
        raise ValueError(
            "MemFabric o_proj expects x=[M, 2048], "
            f"got {tuple(x.shape)}"
        )
    if layer.params_dtype != torch.float16:
        raise TypeError(
            "Phase-1 MemFabric o_proj reduction currently requires FP16 output, "
            f"got {layer.params_dtype}"
        )

    plan = get_memfabric_o_proj_plan()
    begin, publish, finish, mark_failed = _runtime_ops()

    num_tokens = int(x.shape[0])
    if num_tokens == 0:
        return torch.empty(
            (0, _EXPECTED_OUTPUT_SIZE),
            dtype=layer.params_dtype,
            device=x.device,
        )

    # Ordinary NPU result storage prevents a later o_proj wave from overwriting
    # a symmetric recv arena that downstream ops may still be consuming.
    output = torch.empty(
        (num_tokens, _EXPECTED_OUTPUT_SIZE),
        dtype=layer.params_dtype,
        device=x.device,
    )

    for wave_start in range(0, num_tokens, plan.max_rows_per_wave):
        wave_rows = min(plan.max_rows_per_wave, num_tokens - wave_start)
        chunks = plan.chunks_for_tokens(wave_rows)
        send, recv = begin(x, int(tp_rank), int(plan.tile_m), int(chunks))

        try:
            for chunk_idx in range(chunks):
                local_row = chunk_idx * plan.tile_m
                rows = min(plan.tile_m, wave_rows - local_row)
                global_row = wave_start + local_row

                x_tile = x.narrow(0, global_row, rows)
                # Same op as the unquantized fallback (torch.ops.vllm.
                # unquantized_gemm == F.linear), so numerics are identical
                # including the 310P NZ weight layout.
                y_tile = F.linear(x_tile, layer.weight.data)

                # Each SDMA slot has fixed 128KB when tile_m=32. A partial tail
                # is zero padded so the consumer reduces a fixed-size chunk.
                send_tile = send.narrow(0, local_row, plan.tile_m)
                if rows != plan.tile_m:
                    send_tile.zero_()
                send_tile.narrow(0, 0, rows).copy_(y_tile)

                # publish() is enqueued on the same NPU stream after MM/copy.
                # This only orders producer -> communication. It never waits for
                # SDMA, so MM[t+1] can immediately be enqueued.
                publish(send, int(chunk_idx))

            # Final join only after all producer tiles have been submitted.
            finish(recv)
            output.narrow(0, wave_start, wave_rows).copy_(recv.narrow(0, 0, wave_rows))
        except BaseException as exc:
            # A partially armed SDMA/reduce wave is unsafe to reuse. Poison the
            # process-local runtime; both TP workers must be restarted.
            try:
                mark_failed(recv, str(exc))
            except BaseException:
                pass
            raise

    return output


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
                    raise RuntimeError(
                        "MemFabric o_proj fusion does not support biased o_proj"
                    )
                from vllm.distributed import get_tensor_model_parallel_rank

                return memfabric_o_proj_allreduce(
                    layer=layer,
                    x=x,
                    tp_rank=get_tensor_model_parallel_rank(),
                )
            return super().apply(layer, x, bias)

    return MemFabricOProjLinearMethod310
