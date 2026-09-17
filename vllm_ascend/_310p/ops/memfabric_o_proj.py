# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P3 TP=2 MemFabric fused full-attention o_proj helpers.

The fused path is intentionally narrow. It is only enabled for Qwen3.5/3.6
MoE full-attention ``self_attn.o_proj`` layers running static W8A8 with TP=2.
All other layers keep the existing RowParallelLinear implementation.

Phase 1 uses the existing 310P ``npu_quant_matmul`` in M tiles, stages each
finished tile into MemFabric symmetric memory, publishes it immediately, and
lets AICPU/SDMA plus a separate AICore reduce stream consume earlier tiles while
the compute stream continues with later matmuls. Phase 2 will replace only the
producer with a direct AscendC W8A8 matmul that writes the SHM send arena.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.logger import logger

from vllm_ascend import envs

_QWEN35_MOE_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
_EXPECTED_INPUT_SIZE = 4096
_EXPECTED_INPUT_SIZE_PER_PARTITION = 2048
_EXPECTED_OUTPUT_SIZE = 2048
_EXPECTED_TP_SIZE = 2
_MAX_CHUNKS_PER_WAVE = 64


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
    # The first device-side reduction kernel is deliberately BF16-only.
    if getattr(layer, "params_dtype", None) != torch.bfloat16:
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
    if begin is None or publish is None or finish is None:
        raise RuntimeError(
            "vllm_ascend_C was built without the staged 310P MemFabric o_proj runtime"
        )
    return begin, publish, finish


def memfabric_w8a8_o_proj_allreduce(
    layer: torch.nn.Module,
    x_q: torch.Tensor,
    quant_bias: torch.Tensor | None,
    tp_rank: int,
) -> torch.Tensor:
    """Run the phase-1 MM/SDMA/reduce overlap pipeline.

    For each wave:
      1. arm peer receive/reduce and local AICPU/SDMA orchestration;
      2. compute one W8A8 matmul tile;
      3. copy that tile into its immutable symmetric-memory send slot;
      4. publish the slot; SDMA can move it while the next matmul executes;
      5. after the wave joins, copy reduced recv/final rows to ordinary NPU
         storage so the symmetric arenas can be safely reused by later layers.

    The extra local copies are intentionally temporary. The phase-2 producer
    writes matmul results directly into ``send`` and removes them.
    """

    if tp_rank not in (0, 1):
        raise RuntimeError(f"MemFabric o_proj fusion only supports TP rank 0/1, got {tp_rank}")
    if x_q.dtype != torch.int8:
        raise TypeError(f"MemFabric o_proj expects int8 activation, got {x_q.dtype}")
    if x_q.dim() != 2 or x_q.shape[1] != _EXPECTED_INPUT_SIZE_PER_PARTITION:
        raise ValueError(
            "MemFabric o_proj expects x_q=[M, 2048], "
            f"got {tuple(x_q.shape)}"
        )
    if layer.params_dtype != torch.bfloat16:
        raise TypeError(
            "Phase-1 MemFabric o_proj reduction currently requires BF16 output, "
            f"got {layer.params_dtype}"
        )

    plan = get_memfabric_o_proj_plan()
    begin, publish, finish = _runtime_ops()

    num_tokens = int(x_q.shape[0])
    if num_tokens == 0:
        return torch.empty(
            (0, _EXPECTED_OUTPUT_SIZE),
            dtype=layer.params_dtype,
            device=x_q.device,
        )

    # Ordinary NPU result storage prevents a later o_proj wave from overwriting
    # a symmetric recv arena that downstream ops may still be consuming.
    output = torch.empty(
        (num_tokens, _EXPECTED_OUTPUT_SIZE),
        dtype=layer.params_dtype,
        device=x_q.device,
    )

    for wave_start in range(0, num_tokens, plan.max_rows_per_wave):
        wave_rows = min(plan.max_rows_per_wave, num_tokens - wave_start)
        chunks = plan.chunks_for_tokens(wave_rows)
        send, recv = begin(x_q, int(tp_rank), int(plan.tile_m), int(chunks))

        for chunk_idx in range(chunks):
            local_row = chunk_idx * plan.tile_m
            rows = min(plan.tile_m, wave_rows - local_row)
            global_row = wave_start + local_row

            x_tile = x_q.narrow(0, global_row, rows)
            y_tile = torch_npu.npu_quant_matmul(
                x_tile,
                layer.weight.data,
                layer.deq_scale,
                bias=quant_bias,
                output_dtype=layer.params_dtype,
            )

            # Each SDMA slot has fixed 128KB when tile_m=32. A partial tail is
            # zero padded so the consumer can always reduce a fixed-size chunk.
            send_tile = send.narrow(0, local_row, plan.tile_m)
            if rows != plan.tile_m:
                send_tile.zero_()
            send_tile.narrow(0, 0, rows).copy_(y_tile)

            # publish() is enqueued on the same NPU stream after the MM/copy.
            # It only orders producer -> communication; it does not wait for
            # SDMA, so the next loop iteration can immediately enqueue MM[t+1].
            publish(send, int(chunk_idx))

        # Host join for the correctness baseline: local outgoing AICPU/SDMA and
        # peer-arrival reduce stream must both be complete before arena reuse.
        finish(recv)
        output.narrow(0, wave_start, wave_rows).copy_(recv.narrow(0, 0, wave_rows))

    return output
