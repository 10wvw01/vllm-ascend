# SPDX-License-Identifier: Apache-2.0
"""310P3 TP=2 MemFabric o_proj communication bring-up benchmark.

Run on one dual-die 310P3 after building vLLM-Ascend against the customized
wgm-dev-310p MemFabric installation, for example:

  VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 \
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/scripts/bench_310p_memfabric_o_proj.py

This script intentionally tests the SDMA/reduce pipeline independently of the
W8A8 matmul first. HCCL is used only as a numerical/timing reference; the tested
MemFabric path itself does not use HCCL.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401

WIDTH = 2048


def _sync() -> None:
    torch.npu.synchronize()


def _make_local(rows: int, rank: int, device: torch.device) -> torch.Tensor:
    # Values are exactly representable enough for a deterministic BF16 sum while
    # still varying by row/rank. Keep magnitude small to avoid reduction noise.
    row = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    col = torch.arange(WIDTH, device=device, dtype=torch.float32).unsqueeze(0)
    value = (rank + 1) * 0.25 + (row % 31) * 0.001 + (col % 17) * 0.0001
    return value.to(torch.bfloat16)


def _runtime_ops():
    ns = torch.ops._C_ascend
    return (
        ns.memfabric_o_proj_begin,
        ns.memfabric_o_proj_publish,
        ns.memfabric_o_proj_finish,
        ns.memfabric_o_proj_mark_failed,
    )


def _memfabric_once(
    local: torch.Tensor,
    rank: int,
    tile_m: int,
) -> torch.Tensor:
    begin, publish, finish, mark_failed = _runtime_ops()
    rows = local.shape[0]
    chunks = math.ceil(rows / tile_m)
    dummy_x = torch.empty((max(rows, 1), WIDTH), dtype=torch.int8, device=local.device)
    send, recv = begin(dummy_x, rank, tile_m, chunks)

    try:
        for chunk_idx in range(chunks):
            row0 = chunk_idx * tile_m
            valid_rows = min(tile_m, rows - row0)
            send_tile = send.narrow(0, row0, tile_m)
            if valid_rows != tile_m:
                send_tile.zero_()
            send_tile.narrow(0, 0, valid_rows).copy_(local.narrow(0, row0, valid_rows))
            publish(send, chunk_idx)
        finish(recv)
        return recv.narrow(0, 0, rows).clone()
    except BaseException as exc:
        try:
            mark_failed(recv, str(exc))
        except BaseException:
            pass
        raise


def _time_ms(fn, warmup: int, repeat: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    _sync()

    samples = []
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter_ns()
        fn()
        _sync()
        samples.append((time.perf_counter_ns() - t0) / 1e6)

    return statistics.median(samples), min(samples), max(samples)


def _hccl_once(local: torch.Tensor) -> torch.Tensor:
    out = local.clone()
    dist.all_reduce(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 64, 128, 512, 2048])
    parser.add_argument("--tile-m", type=int, default=int(os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M", "32")))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"This benchmark requires exactly TP=2, got WORLD_SIZE={world_size}")

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")

    if rank == 0:
        print(f"tile_m={args.tile_m}, warmup={args.warmup}, repeat={args.repeat}")
        print("rows\tcorrect\tmemfabric_med_ms\thccl_med_ms\tmemfabric_min_ms\thccl_min_ms")

    for rows in args.rows:
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        local = _make_local(rows, rank, device)

        # Correctness is checked before timing. Both ranks must execute the same
        # MemFabric wave sequence or the customized SHM protocol can deadlock.
        mf_out = _memfabric_once(local, rank, args.tile_m)
        ref = _hccl_once(local)
        _sync()
        correct = torch.allclose(mf_out, ref, rtol=args.rtol, atol=args.atol)
        if not correct:
            diff = (mf_out.float() - ref.float()).abs().max().item()
            raise AssertionError(f"MemFabric mismatch rows={rows}, max_abs_diff={diff}")

        mf_med, mf_min, _ = _time_ms(
            lambda: _memfabric_once(local, rank, args.tile_m),
            args.warmup,
            args.repeat,
        )
        hccl_med, hccl_min, _ = _time_ms(
            lambda: _hccl_once(local),
            args.warmup,
            args.repeat,
        )

        # Report the slower rank for end-to-end TP latency.
        stats = torch.tensor([mf_med, hccl_med, mf_min, hccl_min], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"{rows}\tPASS\t{stats[0].item():.4f}\t{stats[1].item():.4f}"
                f"\t{stats[2].item():.4f}\t{stats[3].item():.4f}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
