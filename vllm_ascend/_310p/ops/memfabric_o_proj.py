# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P3 TP=2 routing for the independently installed MemFabric MC2 operator.

vLLM owns only eligibility/routing/fallback policy. The actual 8-core FP16
matmul, MemFabric transport, SUM reduction, arena/protocol lifecycle and graph
capture contract live in the ``memfabric_matmul_allreduce`` CANN custom OPP.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from contextlib import contextmanager
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
_BASE_M = 256
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class MemFabricOProjPlan:
    """Static routing contract for the external 310P3 MC2 operator."""

    batch_basem_count: int
    base_m: int = _BASE_M
    input_size_per_partition: int = _EXPECTED_INPUT_SIZE_PER_PARTITION
    output_size: int = _EXPECTED_OUTPUT_SIZE
    tp_size: int = _EXPECTED_TP_SIZE

    @property
    def batch_m(self) -> int:
        return self.base_m * self.batch_basem_count

    @property
    def batch_bytes(self) -> int:
        return self.batch_m * self.output_size * 2

    def batches_for_tokens(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        return (num_tokens + self.batch_m - 1) // self.batch_m


def get_memfabric_o_proj_plan() -> MemFabricOProjPlan:
    q = int(envs.VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT)
    if q not in (1, 2, 4):
        raise ValueError(
            "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT "
            f"must be one of 1/2/4, got {q}"
        )
    return MemFabricOProjPlan(batch_basem_count=q)


def _external_op_available() -> bool:
    """Whether the installed custom OPP is visible and ABI-compatible."""

    availability = getattr(
        torch.ops._C_ascend, "memfabric_matmul_allreduce_available", None
    )
    if availability is None:
        return False
    try:
        return bool(availability())
    except Exception:
        return False


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
    """Return whether the layer may be routed to the external MC2 operator."""

    if not envs.VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ:
        return False
    # Never disable generic HCCL reduction unless the independently installed
    # operator is already discoverable through CANN's custom-op API loader.
    if not _external_op_available():
        logger.warning_once(
            "310P MemFabric o_proj was requested but the memfabric_mc2 custom "
            "OPP is not installed/visible or has an incompatible runtime ABI; "
            "falling back to stock matmul + HCCL all-reduce."
        )
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
    if getattr(layer, "params_dtype", None) != torch.float16:
        return False

    quant_method = getattr(layer, "quant_method", None)
    if quant_method is not None:
        method_name = type(quant_method).__name__
        if method_name not in (
            "AscendUnquantizedLinearMethod",
            "MemFabricOProjLinearMethod310",
        ):
            return False
        nested = getattr(quant_method, "quant_method", None)
        if nested is not None and type(nested).__name__ != "AscendUnquantizedLinearMethod":
            return False

    return True


def configure_memfabric_o_proj(layer: torch.nn.Module) -> bool:
    """Mark a target RowParallelLinear so the external MC2 op owns TP SUM."""

    enabled = should_enable_memfabric_o_proj(layer)
    layer._ascend_310p_memfabric_o_proj = enabled
    if not enabled:
        return False

    if not getattr(layer, "reduce_results", False):
        raise RuntimeError(
            "MemFabric o_proj fusion expects RowParallelLinear(reduce_results=True) "
            f"before patching, got {getattr(layer, 'prefix', '<unknown>')}"
        )

    layer.reduce_results = False
    plan = get_memfabric_o_proj_plan()
    logger.info_once(
        "Enable external 310P3 TP=2 MemFabricMatmulAllReduce "
        "(runtime ABI=1, base_m=%d, batch_basem_count=%d, batch_m=%d, "
        "batch_bytes=%d).",
        plan.base_m,
        plan.batch_basem_count,
        plan.batch_m,
        plan.batch_bytes,
    )
    return True


def is_memfabric_o_proj_configured(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, "_ascend_310p_memfabric_o_proj", False))


_pool_protocol_started = False


def memfabric_o_proj_pool_started() -> bool:
    """Whether this worker has issued at least one external fused call."""

    return _pool_protocol_started


_warmup_fallback_depth = 0


@contextmanager
def memfabric_o_proj_warmup_fallback():
    """Optionally keep engine dummy/profile runs on stock matmul + HCCL."""

    global _warmup_fallback_depth
    if envs.VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK:
        _warmup_fallback_depth += 1
        try:
            yield
        finally:
            _warmup_fallback_depth -= 1
    else:
        yield


def memfabric_o_proj_warmup_fallback_active() -> bool:
    return _warmup_fallback_depth > 0


def memfabric_o_proj_allreduce(
    layer: torch.nn.Module,
    x: torch.Tensor,
    tp_rank: int,
) -> torch.Tensor:
    """Call the independently installed MatMul+MemFabric-AllReduce operator."""

    global _pool_protocol_started

    if tp_rank not in (0, 1):
        raise RuntimeError(
            f"MemFabric o_proj fusion only supports TP rank 0/1, got {tp_rank}"
        )
    if x.dtype != torch.float16:
        raise TypeError(f"MemFabric o_proj expects FP16 activation, got {x.dtype}")
    if x.dim() != 2 or x.shape[1] != _EXPECTED_INPUT_SIZE_PER_PARTITION:
        raise ValueError(
            f"MemFabric o_proj expects x=[M, 2048], got {tuple(x.shape)}"
        )
    if layer.params_dtype != torch.float16:
        raise TypeError(
            "MemFabric o_proj reduction currently requires FP16 output, "
            f"got {layer.params_dtype}"
        )

    direct = getattr(torch.ops._C_ascend, "memfabric_matmul_allreduce", None)
    if direct is None or not _external_op_available():
        raise RuntimeError(
            "memfabric_mc2 custom OPP is not installed/visible or its runtime "
            "ABI is incompatible"
        )

    plan = get_memfabric_o_proj_plan()
    if envs.VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE:
        ev0, ev1 = _record_fused_call_events()
        out = direct(
            x,
            layer.weight.data,
            int(tp_rank),
            int(plan.batch_basem_count),
        )
        _finish_fused_call_events(ev0, ev1, x.shape[0])
    else:
        out = direct(
            x,
            layer.weight.data,
            int(tp_rank),
            int(plan.batch_basem_count),
        )
    _pool_protocol_started = True
    return out


_DEVICE_MONITOR_LOCK = threading.Lock()
_DEVICE_MONITOR_EVENTS: deque = deque(maxlen=128)
_DEVICE_MONITOR_SEQ = 0
_DEVICE_MONITOR_STARTED = False


def _record_fused_call_events():
    global _DEVICE_MONITOR_SEQ
    ev0 = torch.npu.Event(enable_timing=True)
    ev0.record()
    return ev0, torch.npu.Event(enable_timing=True)


def _finish_fused_call_events(ev0, ev1, rows):
    global _DEVICE_MONITOR_SEQ
    ev1.record()
    with _DEVICE_MONITOR_LOCK:
        _DEVICE_MONITOR_EVENTS.append(
            (_DEVICE_MONITOR_SEQ, ev0, ev1, rows, time.time())
        )
        if _DEVICE_MONITOR_SEQ == 0:
            logger.info("[mf310p-monitor] first fused call wall=%.3f", time.time())
        _DEVICE_MONITOR_SEQ += 1
        _maybe_start_device_monitor_locked()


def _maybe_start_device_monitor_locked():
    global _DEVICE_MONITOR_STARTED
    if _DEVICE_MONITOR_STARTED:
        return
    _DEVICE_MONITOR_STARTED = True
    threading.Thread(target=_device_monitor_loop, daemon=True).start()


def _device_monitor_loop():
    while True:
        time.sleep(0.5)
        while True:
            with _DEVICE_MONITOR_LOCK:
                if not _DEVICE_MONITOR_EVENTS:
                    break
                seq, ev0, ev1, rows, t_enq = _DEVICE_MONITOR_EVENTS[0]
                if not ev1.query():
                    break
                _DEVICE_MONITOR_EVENTS.popleft()
            try:
                elapsed_ms = ev0.elapsed_time(ev1)
            except Exception:
                elapsed_ms = -1.0
            if elapsed_ms > 500.0:
                logger.warning(
                    "[mf310p-slow] seq=%d rows=%d device_ms=%.1f enq_wall=%.3f",
                    seq,
                    rows,
                    elapsed_ms,
                    t_enq,
                )


def make_memfabric_o_proj_linear_method():
    """Build the Qwen routing method around the generic external operator."""

    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

    class MemFabricOProjLinearMethod310(AscendUnquantizedLinearMethod):
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Preserve the existing 310P unquantized path, including NZ weight
            # conversion, before deciding whether to route the layer.
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
                    raise RuntimeError(
                        "MemFabric o_proj fusion does not support biased o_proj"
                    )
                if (
                    memfabric_o_proj_warmup_fallback_active()
                    or x.shape[0] < envs.VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M
                ):
                    from vllm.distributed import tensor_model_parallel_all_reduce

                    out = super().apply(layer, x, bias)
                    return tensor_model_parallel_all_reduce(out)

                from vllm.distributed import get_tensor_model_parallel_rank

                return memfabric_o_proj_allreduce(
                    layer=layer,
                    x=x,
                    tp_rank=get_tensor_model_parallel_rank(),
                )
            return super().apply(layer, x, bias)

    return MemFabricOProjLinearMethod310
