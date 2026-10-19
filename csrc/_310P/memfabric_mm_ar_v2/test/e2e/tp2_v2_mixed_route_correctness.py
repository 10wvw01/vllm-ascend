# SPDX-License-Identifier: Apache-2.0
"""G3 TP=2 route: Stock for M<2048 or M>4096, V2 for 2048..4096.

Includes live HCCL *after* V2 MemFabric pool initialization, exact/padded
tails in 2048..4096 and unexpected larger M Stock fallback. TP=2 only.

This tests the C++ V2 serving op and its run-time protocol status directly;
the vLLM layer-level routing is separately tested by the serve functional
test with MEMFABRIC_MM_AR_E2E_V2=1.
"""
from __future__ import annotations

import argparse
import gc
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.utils import maybe_trans_nz
from vllm_ascend._310p.ops.memfabric_mm_ar_v2_serve import v2_threshold_route


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+",
                   default=[2047, 2048, 2049, 3000, 3073, 4095, 4096, 4097, 5000])
    p.add_argument("--chunk-rows", type=int, choices=(256, 512, 1024),
                   default=1024)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    if any(m < 1 for m in args.rows) or args.reps < 1:
        p.error("M must be positive and reps >=1")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        p.error("torchrun --nproc-per-node=2 is required")
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dev = torch.device(f"npu:{local}")
    dist.init_process_group("hccl")
    group = dist.new_group(backend="gloo")

    torch.manual_seed(812 + rank)
    w = maybe_trans_nz(torch.randn(2048, 2048, dtype=torch.float16, device=dev))
    all_m = sorted(set(args.rows + [1, 10, 2047, 2048, 2049, 3000, 3073, 4095, 4096, 4097, 5000]))
    xs, fp32_refs, stock_refs = {}, {}, {}
    # All FP32/HCCL correctness references are computed *before* pool creation.
    # The FP16 Stock baseline is also computed here for subsequent live
    # post-pool Stock/HCCL coexistence validation.
    for m in all_m:
        torch.manual_seed(6300 + rank * 67 + m)
        x = torch.randn(m, 2048, dtype=torch.float16, device=dev)
        xs[m] = x
        part = F.linear(x, w)
        fp32_ref = part.float()
        dist.all_reduce(fp32_ref)
        fp32_refs[m] = fp32_ref.half()
        stock = F.linear(x, w)
        dist.all_reduce(stock)
        stock_refs[m] = stock
    torch.npu.current_stream().synchronize()
    dist.barrier(group=group)

    plan = [2048, 1, 2049, 10, 2047, 3000, 3073, 4095, 4096, 4097, 5000]
    plan.extend(args.rows)
    plan *= args.reps
    ok = 0
    try:
        for iteration, m in enumerate(plan):
            x = xs[m]
            if v2_threshold_route(m) == "stock":
                out = F.linear(x, w)
                dist.all_reduce(out)
                expected = stock_refs[m]
                path = "stock_hccl_after_pool"
            else:
                out = torch.ops._C_ascend.memfabric_mm_ar_v2_allreduce(
                    x, w, rank, args.chunk_rows
                )
                expected = fp32_refs[m]
                path = "v2_fused"
            torch.npu.current_stream().synchronize()
            if v2_threshold_route(m) == "v2":
                status = int(torch.ops._C_ascend.memfabric_mm_ar_v2_debug_status())
                assert status == 0, (
                    f"protocol status={status} rank={rank} m={m} i={iteration}"
                )
            if not torch.equal(out, expected):
                delta = (out.float() - expected.float()).abs()
                raise AssertionError(
                    f"rank={rank} i={iteration} M={m} path={path}: "
                    f"bad={(out != expected).sum().item()} "
                    f"max_abs={delta.max().item()}"
                )
            ok += 1
            if rank == 0:
                print(f"PASS i={iteration} M={m} route={path}", flush=True)
            del out
            # Ensure previous pooled output views really go out of scope.
            gc.collect()
        dist.barrier(group=group)
        if rank == 0:
            print(f"TP2 V2 mixed-route PASS: {ok} calls including live HCCL", flush=True)
    finally:
        torch.npu.current_stream().synchronize()
        # No V1 serving context exists in this process.
        torch.ops._C_ascend.memfabric_mm_ar_v2_shutdown()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
