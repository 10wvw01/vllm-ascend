# SPDX-License-Identifier: Apache-2.0
"""310P3 TP=2 unquantized-BF16 o_proj single-layer correctness benchmark.

Compares the Phase-1 MemFabric fused path against the stock unquantized
reference on one dual-die 310P3 (owner decision A, 2026-09-18: the target
checkpoint keeps full-attention o_proj as FLOAT/BF16):

  reference = full-M F.linear(local, weight) + TP=2 allreduce (FP32 math)
  fused     = tiled F.linear + MemFabric exchange + local reduce

The fused path is the production entry point used by
``MemFabricOProjLinearMethod310.apply`` for target full-attention o_proj
layers. Semantics under test: BF16 activation, 310P FRACTAL_NZ runtime weight
layout (via the stock unquantized process_weights path), BF16 output, and the
BF16 -> F32 add -> BF16 round reduce kernel.

Run:

  VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 \
  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py
"""

from __future__ import annotations

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401  (registers the _C_ascend ops)
from vllm_ascend._310p.ops.memfabric_o_proj import memfabric_o_proj_allreduce
from vllm_ascend.utils import maybe_trans_nz

K_LOCAL = 2048
N_OUT = 2048


def _make_layer(rank: int, device: torch.device) -> types.SimpleNamespace:
    """Mimic AscendUnquantizedLinearMethod.process_weights_after_loading."""

    layer = types.SimpleNamespace()

    # vllm linear weight is [out, in] = [N, K_local] FP16; the 310P runtime
    # layout applies FRACTAL_NZ (kept logical shape) - exactly what the stock
    # unquantized apply()/F.linear consumes.
    weight = torch.randn(N_OUT, K_LOCAL, dtype=torch.float16, device=device)
    layer.weight = types.SimpleNamespace(data=maybe_trans_nz(weight))
    layer.params_dtype = torch.float16
    return layer


def _reference(layer, x):
    # Same F.linear as the stock unquantized path, full M.
    y = F.linear(x, layer.weight.data)
    # 310P HCCL has no FP16 allreduce with matching rounding; reduce in FP32
    # and round back, which is exactly the fused kernel's FP16 -> F32 add ->
    # FP16 round semantics.
    y = y.float()
    dist.all_reduce(y)
    return y.to(torch.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+",
                        default=[1, 8, 32, 33, 64, 128, 512, 2048])
    parser.add_argument("--tile-m", type=int,
                        default=int(os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M", "32")))
    parser.add_argument("--repeat", type=int, default=20,
                        help="repeated fused calls per rows value")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("This benchmark requires exactly TP=2")

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")

    if rank == 0:
        print(f"tile_m={args.tile_m} repeat={args.repeat}")
        print("rows\tverdict\tmax_abs_diff\tn_bad\tfused_ms")

    layer = _make_layer(rank, device)

    for rows in args.rows:
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        torch.manual_seed(42 + rows)
        x = torch.randn(rows, K_LOCAL, dtype=torch.float16, device=device)

        ref = _reference(layer, x)
        torch.npu.synchronize()

        import time
        t0 = time.perf_counter()
        fused = memfabric_o_proj_allreduce(layer=layer, x=x, tp_rank=rank)
        torch.npu.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1e3

        diff = (fused.float() - ref.float()).abs()
        max_diff = diff.max().item()
        n_bad = (diff > (args.atol + args.rtol * ref.float().abs())).sum().item()
        ok = torch.allclose(fused, ref, rtol=args.rtol, atol=args.atol)

        # Repeated calls: verify every call to catch wave-reuse issues.
        for call_idx in range(args.repeat - 1):
            out = memfabric_o_proj_allreduce(layer=layer, x=x, tp_rank=rank)
            if not torch.allclose(out, ref, rtol=args.rtol, atol=args.atol):
                n_bad += (out.float() - ref.float()).abs().numel()
                ok = False
        torch.npu.synchronize()

        verdict = "PASS" if ok else "FAIL"
        if rank == 0:
            print(f"{rows}\t{verdict}\t{max_diff:.6f}\t{n_bad}\t{fused_ms:.3f}")
        if not ok:
            raise AssertionError(
                f"FP16 o_proj fused mismatch rows={rows} max_abs_diff={max_diff}")

    torch.ops._C_ascend.memfabric_o_proj_shutdown()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
