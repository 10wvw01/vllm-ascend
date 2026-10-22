# SPDX-License-Identifier: Apache-2.0
"""Opt-in 310P3 V2 serving route: 2048<=M<=4096 fused, otherwise Stock.

V1 serving remains unchanged when this feature is disabled. The native
large-M op uses the already-tested V2 T0 one-kernel path, with a bounded
pre-pool output buffer lease and exact-M tail padding at the C++ boundary.

Current safety contract: only FULL_DECODE_ONLY graph (capture sizes <2048)
is supported; large-M fused prefill is eager. Production enablement still
requires G3 first-write, KFC and TP skew evidence from real 310P3 tests.

The first fused call of each tail shape class (chunk-aligned vs padded)
is verified bit-exact against the stock MM + HCCL path. A mismatch raises
immediately (fail-stop) instead of letting a cold-pool wave silently
corrupt model outputs: the 2026-10-18 cold-orchestrator incident produced
garbage outputs with protocol status 0 on the first loads of a freshly
deployed kernel.
"""
from __future__ import annotations

import torch
from vllm.logger import logger

from vllm_ascend import envs

# Serving 的实际路由窗口：只对 2048≤M≤4096 的合格 FP16 层使用融合。
# 与 T0 Kernel 独立试验的 8192 行上限不同；Graph 小 M 依旧回退 Stock。
MIN_FUSED_M = 2048
MAX_FUSED_M = 4096  # the configured 4096-token Forward budget
_pool_started = False
_logged_stock = False
_logged_fused = False
_logged_oversize_stock = False
_verified_aligned = False
_verified_padded = False


# 作用：向 Runner hook 报告本 Worker 是否已发起 V2 池生命周期。
# 入参：无。返回：布尔值；首次 V2 op 返回后置 True，不代表异步波已完成。
# 原因：池常驻时避免 device-wide sync 等待 MemFabric 编排流。
def v2_pool_started() -> bool:
    return _pool_started


# 作用：将 M 行数映射到 V2/Stock；不与 V1 q 或 batch_M 混用。
# 入参：m 为本次 Linear 的 M。返回："v2" 或 "stock"；负数抛 ValueError。
# 原因：硬件验证范围之外仍必须有完整 MM+HCCL 的安全退路。
def v2_threshold_route(m: int) -> str:
    """Pure route decision, intentionally independent from V1's q/batch size."""
    if m < 0:
        raise ValueError(f"negative M: {m}")
    # Forward M may exceed this serving profile's expected 4096-token
    # budget. Stock must remain a safe, non-fused escape hatch: do NOT
    # split it into multiple V2 waves or abort normal inference.
    return "v2" if MIN_FUSED_M <= m <= MAX_FUSED_M else "stock"


# 作用：延迟创建继承 AscendUnquantizedLinearMethod 的 V2 LinearMethod 子类。
# 入参：无。返回：MemFabricMmArV2LinearMethod310 类（不是立即执行 Kernel）。
# 原因：保留原始 FP16/NZ 权重加载，又允许目标层接管 MM+TP 规约，避免循环 import。
def make_memfabric_mm_ar_v2_linear_method():
    from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
    from vllm_ascend._310p.ops.memfabric_mm_ar import (
        configure_memfabric_mm_ar,
        is_memfabric_mm_ar_configured,
        memfabric_mm_ar_warmup_fallback_active,
    )

    # 只用于满足 Qwen3.6 指定 RowParallelLinear 条件的层，其余层不应改变原来的 LinearMethod。
    class MemFabricMmArV2LinearMethod310(AscendUnquantizedLinearMethod):
        """Only selected FP16 RowParallelLinear o_proj/out_proj are routed."""

        # 作用：继承 Stock 的权重处理（含 NZ 布局），再对满足资格的层标记融合。
        # 入参：self 当前 LinearMethod；layer 已加载权重的目标层。返回：None；副作用是资格标记。
        # 原因：不为 V2 单独复制权重转换逻辑，统一在层加载后关闭上层重复 TP Reduce。
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            super().process_weights_after_loading(layer)
            configure_memfabric_mm_ar(layer)

        # 作用：执行资格与 M 路由，或退回完整 Stock；融合后首次按尾块类型做位精确对照。
        # 入参：self LinearMethod、layer 目标层、x=[M,2048] FP16 Tensor、bias 可选偏置。
        # 返回：已完成 TP2 规约的输出 Tensor；非合格层交给父类按原合同执行。
        # 原因：reduce_results 已被关闭，Stock 退路必须自行补 HCCL；V2 不能再重复 HCCL。
        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            global _pool_started, _logged_stock, _logged_fused
            global _logged_oversize_stock, _verified_aligned, _verified_padded

            # 本层若未被资格检查选中，不私自接管 TP 规约，保持父类的正常路径。
            if not is_memfabric_mm_ar_configured(layer):
                return super().apply(layer, x, bias)
            if bias is not None:
                raise RuntimeError("310P V2 fused o_proj does not support bias")

            # Profile/dummy-run and any M outside [2048,4096] go through
            # the original native MM + HCCL. Decode can still be captured
            # by the existing FULL_DECODE_ONLY graph route.
            # profile/dummy 和 M 窗口以外走本地 MM+HCCL；仅调试开关允许的 warmup 才启用回退。
            # 特别注意：layer.reduce_results=False，因此必须在这里显式调用 TP all-reduce。
            if memfabric_mm_ar_warmup_fallback_active() or (
                v2_threshold_route(int(x.shape[0])) == "stock"
            ):
                if x.shape[0] > MAX_FUSED_M and not _logged_oversize_stock:
                    logger.warning(
                        "[mf310p-v2] unexpected Forward M=%d > %d; "
                        "falling back to original MM + HCCL AR (not V2)",
                        x.shape[0], MAX_FUSED_M,
                    )
                    _logged_oversize_stock = True
                if not _logged_stock:
                    logger.info("[mf310p-v2] M=%d outside [2048,4096] (or dummy warmup): native MM + HCCL", x.shape[0])
                    _logged_stock = True
                from vllm.distributed import tensor_model_parallel_all_reduce
                partial = super().apply(layer, x, bias)
                return tensor_model_parallel_all_reduce(partial)

            # PyTorch Op 由 C++ TORCH_LIBRARY_FRAGMENT 注册，返回的结果已包含对端局部 C。
            op = getattr(torch.ops._C_ascend, "memfabric_mm_ar_v2_allreduce", None)
            if op is None:
                raise RuntimeError(
                    "310P V2 serving op missing. Rebuild with both "
                    "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1 and "
                    "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1."
                )
            from vllm.distributed import get_tensor_model_parallel_rank

            # 仅允许 Kernel 已编译的 3 档静态 tiling；真实 Rank 来自 TP 通信组。
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

            # Numerical gate for the first fused call of each tail shape
            # class. Both ranks run this symmetrically; the 2-rank FP16 sum
            # is order-independent, so bit-exact equality with the stock
            # path is the expected steady-state behavior (proven by the
            # TP=2 mixed-route test). A cold pool/transport must fail
            # loudly here instead of silently corrupting model outputs.
            # 首次融合校验分别覆盖整块和尾块暂存两类路径。
            # 协议 status==0 不保证首次输出正确；位精确对照失败直接抛错并要求双 Rank 重启。
            aligned = (int(x.shape[0]) % chunk) == 0
            needs_verify = (aligned and not _verified_aligned) or (
                (not aligned) and not _verified_padded)
            if needs_verify:
                from vllm.distributed import tensor_model_parallel_all_reduce
                reference = tensor_model_parallel_all_reduce(
                    super().apply(layer, x, bias))
                if not torch.equal(out, reference):
                    bad = int((out != reference).sum().item())
                    raise RuntimeError(
                        "[mf310p-v2] first fused call verification FAILED "
                        f"(M={x.shape[0]}, aligned={aligned}, mismatched "
                        f"elements={bad}). The MemFabric pool is in a cold/"
                        "corrupted state; restart BOTH TP=2 workers (and "
                        "re-check the orchestrator kernel deployment)."
                    )
                if aligned:
                    _verified_aligned = True
                else:
                    _verified_padded = True
                logger.info(
                    "[mf310p-v2] first fused call verified bit-exact vs "
                    "stock (M=%d, aligned=%s)", x.shape[0], aligned)
            # The V2 op already performs a TP=2 reduction. Generic HCCL
            # MUST NOT run again on this branch.
            # 最终输出是已规约的 Y=C0+C1，返回后绝对不能再走 RowParallelLinear HCCL。
            return out

    return MemFabricMmArV2LinearMethod310
