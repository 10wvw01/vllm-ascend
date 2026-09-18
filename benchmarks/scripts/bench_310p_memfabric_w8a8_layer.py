# SPDX-License-Identifier: Apache-2.0
"""310P3 TP=2 W8A8 o_proj single-layer correctness benchmark.

Compares the Phase-1 MemFabric fused path against the existing 310P W8A8
reference on one dual-die 310P3:

  reference = full-M npu_quant_matmul(local) + TP=2 allreduce
  fused     = tiled npu_quant_matmul + MemFabric exchange + local reduce

The fused path is the production entry point used by
``AscendW8A8LinearMethod310.apply`` for target full-attention o_proj layers.
Quantization semantics under test: int8 activation, 310P FRACTAL_NZ runtime
weight layout, int64 deq_scale, rank0-only int32 quant_bias, BF16 output.

Run:

  VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 \
  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  LD_LIBRARY_PATH=/opt/memfabric-wgm-dev-310p/lib64:$LD_LIBRARY_PATH \
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/scripts/bench_310p_memfabric_w8a8_layer.py
"""

from __future__ import annotations

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend._310p.ops.memfabric_o_proj import memfabric_w8a8_o_proj_allreduce
from vllm_ascend.utils import maybe_trans_nz

K_LOCAL = 2048
N_OUT = 2048


def _make_layer(rank: int, device: torch.device) -> types.SimpleNamespace:
    """Mimic AscendW8A8LinearMethod310.process_weights_after_loading."""

    layer = types.SimpleNamespace()

    # Canonical quantized weight [N, K] int8, then the 310P runtime layout:
    # FRACTAL_NZ + transpose(0, 1) - exactly what apply() consumes.
    weight = torch.randint(
        -127, 128, (N_OUT, K_LOCAL), dtype=torch.int8, device=device)
    layer.weight = types.SimpleNamespace(
        data=maybe_trans_nz(weight).transpose(0, 1))

    # int64 deq_scale and int32 quant_bias match get_perchannel_param.
    layer.deq_scale = torch.randint(
        1 << 19, 1 << 21, (N_OUT,), dtype=torch.int64, device=device)
    layer.quant_bias = torch.randint(
        -(1 << 28), 1 << 28, (N_OUT,), dtype=torch.int32, device=device)
    layer.params_dtype = torch.bfloat16
    return layer


def _reference(layer, x_q, quant_bias):
    # Same npu_quant_matmul as the non-fused 310P path, full M.
    y = torch_npu.npu_quant_matmul(
        x_q,
        layer.weight.data,
        layer.deq_scale,
        bias=quant_bias,
        output_dtype=layer.params_dtype,
    )
    # 310P HCCL has no BF16 allreduce; reduce in FP32 and round back, which is
    # exactly the fused kernel's BF16 -> F32 add -> BF16 semantics.
    y = y.float()
    dist.all_reduce(y)
    return y.to(torch.bfloat16)


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
    quant_bias = layer.quant_bias if rank == 0 else None

    for rows in args.rows:
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        torch.manual_seed(42 + rows)
        x_q = torch.randint(
            -127, 128, (rows, K_LOCAL), dtype=torch.int8, device=device)

        ref = _reference(layer, x_q, quant_bias)
        torch.npu.synchronize()

        import time
        t0 = time.perf_counter()
        fused = memfabric_w8a8_o_proj_allreduce(
            layer=layer, x_q=x_q, quant_bias=quant_bias, tp_rank=rank)
        torch.npu.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1e3

        diff = (fused.float() - ref.float()).abs()
        max_diff = diff.max().item()
        n_bad = (diff > (args.atol + args.rtol * ref.float().abs())).sum().item()
        ok = torch.allclose(fused, ref, rtol=args.rtol, atol=args.atol)

        # Repeated calls: verify every call to catch wave-reuse issues.
        for call_idx in range(args.repeat - 1):
            out = memfabric_w8a8_o_proj_allreduce(
                layer=layer, x_q=x_q, quant_bias=quant_bias, tp_rank=rank)
            if not torch.allclose(out, ref, rtol=args.rtol, atol=args.atol):
                n_bad += (out.float() - ref.float()).abs().numel()
                ok = False
        torch.npu.synchronize()

        verdict = "PASS" if ok else "FAIL"
        if rank == 0:
            print(f"{rows}\t{verdict}\t{max_diff:.6f}\t{n_bad}\t{fused_ms:.3f}")
        if not ok:
            raise AssertionError(
                f"W8A8 o_proj fused mismatch rows={rows} max_abs_diff={max_diff}")

    torch.ops._C_ascend.memfabric_o_proj_shutdown()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
