# SPDX-License-Identifier: Apache-2.0
"""Opt-in 310P3 V2 serving route: 2048<=M<=4096 fused, otherwise Stock.

V1 serving remains unchanged when this feature is disabled. The native
large-M op uses the already-tested V2 T0 one-kernel path, with a bounded
pre-pool output buffer lease and exact-M tail padding at the C++ boundary.

Current safety contract: only FULL_DECODE_ONLY graph (capture sizes <2048)
is supported; large-M fused prefill is eager. Production enablement still
requires G3 first-write, KFC and TP skew evidence from real 310P3 tests.
"""
from __future__ import annotations

import torch
from vllm.logger import logger

from vllm_ascend import envs

MIN_FUSED_M = 2048
MAX_FUSED_M = 4096  # the configured 4096-token Forward budget
_pool_started = False
_logged_stock = False
_logged_fused = False


def v2_pool_started() -> bool:
    return _pool_started


def v2_threshold_route(m: int) -> str:
    """Pure route decision, intentionally independent from V1's q/batch size."""
    if m < 0:
        raise ValueError(f"negative M: {m}")
    # Forward M may exceed this serving profile's expected 4096-token
    # budget. Stock must remain a safe, non-fused escape hatch: do NOT
    # split it into multiple V2 waves or abort normal inference.
    return "v2" if MIN_FUSED_M <= m <= MAX_FUSED_M else "stock"


def make_memfabric_mm_ar_v2_linear_method():
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
    from vllm_ascend._310p.ops.memfabric_mm_ar import (
        configure_memfabric_mm_ar,
        is_memfabric_mm_ar_configured,
        memfabric_mm_ar_warmup_fallback_active,
    )

    class MemFabricMmArV2LinearMethod310(AscendUnquantizedLinearMethod):
        """Only selected FP16 RowParallelLinear o_proj/out_proj are routed."""

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            super().process_weights_after_loading(layer)
            configure_memfabric_mm_ar(layer)

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            global _pool_started, _logged_stock, _logged_fused

            if not is_memfabric_mm_ar_configured(layer):
                return super().apply(layer, x, bias)
            if bias is not None:
                raise RuntimeError("310P V2 fused o_proj does not support bias")

            # Profile/dummy-run and any M outside [2048,4096] go through
            # the original native MM + HCCL. Decode can still be captured
            # by the existing FULL_DECODE_ONLY graph route.
            if memfabric_mm_ar_warmup_fallback_active() or (
                v2_threshold_route(int(x.shape[0])) == "stock"
            ):
                if not _logged_stock:
                    logger.info("[mf310p-v2] M=%d outside [2048,4096] (or dummy warmup): native MM + HCCL", x.shape[0])
                    _logged_stock = True
                from vllm.distributed import tensor_model_parallel_all_reduce
                partial = super().apply(layer, x, bias)
                return tensor_model_parallel_all_reduce(partial)

            op = getattr(torch.ops._C_ascend, "memfabric_mm_ar_v2_allreduce", None)
            if op is None:
                raise RuntimeError(
                    "310P V2 serving op missing. Rebuild with both "
                    "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1 and "
                    "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1."
                )
            from vllm.distributed import get_tensor_model_parallel_rank

            chunk = int(envs.VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS)
            if chunk not in (256, 512, 1024):
                raise ValueError(f"invalid V2 chunk_rows={chunk}")
            out = op(
                x.contiguous(), layer.weight.data,
                int(get_tensor_model_parallel_rank()), chunk,
            )
            _pool_started = True
            if not _logged_fused:
                logger.info("[mf310p-v2] M=%d within [2048,4096]: V2 MM+MemFabric (chunk=%d)", x.shape[0], chunk)
                _logged_fused = True
            # The V2 op already performs a TP=2 reduction. Generic HCCL
            # MUST NOT run again on this branch.
            return out

    return MemFabricMmArV2LinearMethod310
