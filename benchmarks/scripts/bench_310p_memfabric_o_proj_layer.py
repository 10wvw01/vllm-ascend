# SPDX-License-Identifier: Apache-2.0
"""310P3 TP=2 unquantized FP16 o_proj single-layer correctness benchmark.

Compares the fused MemFabric V5 path against the stock unquantized reference
on one dual-die 310P3 (owner decision A, 2026-09-18: the target checkpoint
keeps full-attention o_proj as FLOAT; the model runs FP16 on 310P):

  reference = full-M F.linear(local, weight) + TP=2 allreduce (FP32 math)
  fused     = one fused AscendC FP16 matmul per wave writing the symmetric
              send arena directly (matmul -> line-clean -> mailbox signal),
              AICPU/SDMA exchange, waiter kernel (quiet + wait), stream-
              ordered add_out reduce

V5 integration constraints honored by this benchmark (see the development
plan P5 notes):

  1. While the MemFabric pool exists, a perpetual AICPU epoch kernel is
     alive; any device-wide synchronize (torch.npu.synchronize) would wait
     for it forever. All fused-phase waits therefore use
     ``torch.npu.current_stream().synchronize()`` only.
  2. Pool lifetime must stay well under the epoch launch-timeout window
     (~25 s on this box). All HCCL reference work therefore happens in a
     first pass *before* the pool is created (the pool is created lazily by
     the first fused call).

Run:

  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 \
  MF_SDMA_ORCH_JSON=<prefix>/hybm/aicpu_kernel/libmf_sdma_orch_v6.json \
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
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


def _stream_sync() -> None:
    # Never torch.npu.synchronize() here: the perpetual V5 epoch AICPU task
    # would make a device-wide sync hang (see module docstring).
    torch.npu.current_stream().synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 33, 64, 128, 512, 2048])
    parser.add_argument("--tile-m", type=int, default=int(os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M", "32")))
    parser.add_argument("--repeat", type=int, default=20, help="repeated fused calls per rows value")
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

    # ---- Pass 1: inputs + references (HCCL phase, pool does not exist yet).
    # Deterministic inputs per rows value; identical seeds on both ranks keep
    # the exchanged partials reproducible.
    inputs: dict[int, torch.Tensor] = {}
    refs: dict[int, torch.Tensor] = {}
    for rows in args.rows:
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        torch.manual_seed(42 + rows)
        x = torch.randn(rows, K_LOCAL, dtype=torch.float16, device=device)
        inputs[rows] = x
        refs[rows] = _reference(layer, x)
    # Device-wide sync is safe here (no pool / no epoch yet) and makes sure
    # every reference is materialized before the fused phase starts.
    torch.npu.synchronize()
    dist.barrier()

    # ---- Pass 2: fused calls. The first call lazily creates the MemFabric
    # pool; from here on only stream-level syncs are allowed.
    pool_alive = False
    for rows in args.rows:
        x = inputs[rows]
        ref = refs[rows]

        t0 = time.perf_counter()
        fused = memfabric_o_proj_allreduce(layer=layer, x=x, tp_rank=rank)
        _stream_sync()
        fused_ms = (time.perf_counter() - t0) * 1e3
        pool_alive = True

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
        _stream_sync()

        verdict = "PASS" if ok else "FAIL"
        if rank == 0:
            print(f"{rows}\t{verdict}\t{max_diff:.6f}\t{n_bad}\t{fused_ms:.3f}")
        if not ok:
            raise AssertionError(f"FP16 o_proj fused mismatch rows={rows} max_abs_diff={max_diff}")

    # Exit contract on V5 (see development plan P5 notes): while the pool is
    # alive, torch_npu HCCL collectives (barrier/allreduce) internally do a
    # device-wide synchronize, which waits for the perpetual epoch AICPU task
    # and dies with 507901 once the epoch launch-timeout kills it. The
    # compute results are already validated per rank above, so after a
    # best-effort pool shutdown the benchmark exits without any further
    # collectives or device work.
    if rank == 0:
        print("ALL PASS", flush=True)
    if pool_alive:
        with contextlib.suppress(Exception):
            torch.ops._C_ascend.memfabric_o_proj_shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
