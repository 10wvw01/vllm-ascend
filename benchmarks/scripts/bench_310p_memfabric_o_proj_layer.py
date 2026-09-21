# SPDX-License-Identifier: Apache-2.0
"""310P3 TP=2 FP16 o_proj single-layer correctness benchmark.

The target checkpoint keeps full-attention o_proj as an unquantized FLOAT
layer; this 310P3 path runs FP16.

Reference:
    local F.linear -> FP32 TP sum -> FP16

Fused:
    7 MM workers + 1 communication coordinator
    -> MemFabric public signal/wait/quiet
    -> local FP16 add

This benchmark intentionally computes the HCCL reference before the first fused
call so correctness measurement does not depend on HCCL/MemFabric coexistence.
Coexistence is a separate hardware-validation gate.

Run:

  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 \
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py \
    --rows 1 8 32 33 64 128 512 2048 4096 --repeat 20
"""

from __future__ import annotations

import argparse
import os
import time
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend._310p.ops.memfabric_o_proj import memfabric_o_proj_allreduce
from vllm_ascend.utils import maybe_trans_nz

K_LOCAL = 2048
N_OUT = 2048


def _make_layer(rank: int, device: torch.device) -> types.SimpleNamespace:
    layer = types.SimpleNamespace()
    weight = torch.randn(N_OUT, K_LOCAL, dtype=torch.float16, device=device)
    layer.weight = types.SimpleNamespace(data=maybe_trans_nz(weight))
    layer.params_dtype = torch.float16
    return layer


def _reference(layer, x):
    y = F.linear(x, layer.weight.data).float()
    dist.all_reduce(y)
    return y.to(torch.float16)


def _stream_sync() -> None:
    torch.npu.current_stream().synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 8, 32, 33, 64, 128, 512, 2048, 4096],
    )
    parser.add_argument(
        "--tile-m",
        type=int,
        default=int(
            os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M", "32")
        ),
    )
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
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
        print(
            "rows\tverdict\tmax_abs_diff\tn_bad\tfirst_ms"
            "\tmin_ms\tmed_ms\tmax_ms"
        )

    layer = _make_layer(rank, device)

    # Materialize all HCCL references before MemFabric context creation.
    inputs: dict[int, torch.Tensor] = {}
    refs: dict[int, torch.Tensor] = {}
    for rows in args.rows:
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        torch.manual_seed(42 + rows)
        x = torch.randn(rows, K_LOCAL, dtype=torch.float16, device=device)
        inputs[rows] = x
        refs[rows] = _reference(layer, x)
    torch.npu.synchronize()
    dist.barrier()

    for rows in args.rows:
        x = inputs[rows]
        ref = refs[rows]

        t0 = time.perf_counter()
        fused = memfabric_o_proj_allreduce(layer=layer, x=x, tp_rank=rank)
        _stream_sync()
        first_ms = (time.perf_counter() - t0) * 1e3

        diff = (fused.float() - ref.float()).abs()
        max_diff = diff.max().item()
        n_bad = (
            diff > (args.atol + args.rtol * ref.float().abs())
        ).sum().item()
        ok = torch.allclose(fused, ref, rtol=args.rtol, atol=args.atol)

        steady_ms = []
        for _ in range(args.repeat - 1):
            t1 = time.perf_counter()
            out = memfabric_o_proj_allreduce(layer=layer, x=x, tp_rank=rank)
            _stream_sync()
            steady_ms.append((time.perf_counter() - t1) * 1e3)
            if not torch.allclose(out, ref, rtol=args.rtol, atol=args.atol):
                ok = False
                n_bad += (
                    (out.float() - ref.float()).abs()
                    > (args.atol + args.rtol * ref.float().abs())
                ).sum().item()

        steady_ms.sort()
        verdict = "PASS" if ok else "FAIL"
        if rank == 0:
            if steady_ms:
                med = steady_ms[len(steady_ms) // 2]
                print(
                    f"{rows}\t{verdict}\t{max_diff:.6f}\t{n_bad}"
                    f"\t{first_ms:.3f}\t{steady_ms[0]:.3f}"
                    f"\t{med:.3f}\t{steady_ms[-1]:.3f}"
                )
            else:
                print(
                    f"{rows}\t{verdict}\t{max_diff:.6f}\t{n_bad}"
                    f"\t{first_ms:.3f}"
                )

        if not ok:
            raise AssertionError(
                f"FP16 o_proj fused mismatch rows={rows} "
                f"max_abs_diff={max_diff}"
            )

    if rank == 0:
        print("ALL PASS", flush=True)

    # Eager benchmark owns the context and can exercise the current teardown
    # contract explicitly. Do not suppress teardown errors.
    torch.ops._C_ascend.memfabric_o_proj_shutdown()


if __name__ == "__main__":
    main()
