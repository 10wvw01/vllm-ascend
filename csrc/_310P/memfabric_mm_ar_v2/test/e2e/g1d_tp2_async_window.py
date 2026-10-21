#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-D2 双 Rank 异步窗口对照：不改 MemFabric 队列/游标，不制造人工损坏。

仅对比同一 Stream 上若干 V2 调用连续入队与逐调用同步。
禁止同一窗口持有超过实际 V2_OUTPUT_SLOTS 的未释放输出引用。
默认窗口为 1，先验收正确性，再扩大窗口；每窗口后同步和位精确校验。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.utils import maybe_trans_nz


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--chunk-rows", type=int, choices=(256, 512, 1024), default=1024)
    p.add_argument("--window", type=int, default=1, help="同步前连续 V2 调用数，最大 4")
    p.add_argument("--calls", type=int, default=128)
    p.add_argument("--output-slots", type=int, default=4)
    p.add_argument("--add-stock", action="store_true", help="每窗口完成后插入 Stock/HCCL")
    args = p.parse_args()
    if (not 2048 <= args.rows <= 4096 or args.window < 1
            or args.output_slots < 1 or args.window > args.output_slots
            or args.output_slots > 4 or args.calls < 1):
        p.error("仅 2048<=M<=4096; 1<=window<=output-slots<=4; calls>=1")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        p.error("仅允许 torchrun --nproc-per-node=2")
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dev = torch.device(f"npu:{local}")
    dist.init_process_group("hccl")
    ctrl = dist.new_group(backend="gloo")
    pending = []
    completed = 0
    t0 = time.monotonic()
    try:
        torch.manual_seed(4260 + rank)
        w = maybe_trans_nz(torch.randn(2048, 2048, device=dev, dtype=torch.float16))
        x = torch.randn(args.rows, 2048, device=dev, dtype=torch.float16)
        # 在 MemFabric 首次建池之前生成跨 Rank 参考。
        local_part = F.linear(x, w)
        reference = local_part.float()
        dist.all_reduce(reference)
        reference = reference.half()
        torch.npu.current_stream().synchronize()
        dist.barrier(group=ctrl)
        for index in range(args.calls):
            out = torch.ops._C_ascend.memfabric_mm_ar_v2_allreduce(
                x, w, rank, args.chunk_rows
            )
            pending.append((index, out))
            if len(pending) < args.window and index + 1 < args.calls:
                continue
            torch.npu.current_stream().synchronize()
            st = int(torch.ops._C_ascend.memfabric_mm_ar_v2_debug_status())
            if st != 0:
                raise AssertionError(f"rank={rank} status={st} after index={index}")
            for i, result in pending:
                if not torch.equal(result, reference):
                    raise AssertionError(
                        f"rank={rank} index={i} M={args.rows} bit-exact failed"
                    )
                completed += 1
            pending.clear()
            del out
            gc.collect()
            if args.add_stock:
                stock = F.linear(x, w)
                dist.all_reduce(stock)
                torch.npu.current_stream().synchronize()
                # Stock 参考不是 FP32 累加口径：只确认形状/有限性，不混为融合数值判据。
                assert stock.shape == reference.shape
                assert bool(torch.isfinite(stock).all().item())
                del stock
            if rank == 0 and (completed <= 8 or completed % 128 == 0
                              or completed == args.calls):
                print(json.dumps({
                    "tag": "G1D2_PROGRESS", "completed_calls": completed,
                    "window": args.window, "rows": args.rows,
                    "chunk_rows": args.chunk_rows,
                    "theoretical_mf_requests_per_rank":
                        completed * ((args.rows + args.chunk_rows - 1) // args.chunk_rows + 1),
                    "elapsed_s": round(time.monotonic() - t0, 3),
                }, ensure_ascii=False), flush=True)
        dist.barrier(group=ctrl)
        if rank == 0:
            print(f"G1D2 PASS calls={completed} window={args.window} "
                  f"rows={args.rows} elapsed_s={time.monotonic()-t0:.3f}", flush=True)
    finally:
        # 若失败，尽量回收；真正设备挂死时由外部 timeout/设备日志取证负责。
        pending.clear()
        torch.npu.current_stream().synchronize()
        torch.ops._C_ascend.memfabric_mm_ar_v2_shutdown()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
