#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""U1真实NZ-Stock短Profiler探针：单进程组仅HCCL+CPU Gloo，不混MemFabric池。

运行前由人工先完成G0/隔离/卡健康，输出Chrome诊断Trace，不用Profiler时延做ABBA。
对实际aclnnMatmulWeightNz命中必须额外导出CANN kernel_details.csv并执行
a1_nz_route_gate.py；如果现场版本没法导出则不能给NZ模型等价性PASS。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from u1_spec import D4_DEVICE_SHA, PLUGIN_SHA, MEMFABRIC_HOST_SHA, KVER_SHA

K = N = 2048
MS = (2048, 4096)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=6)
    args = p.parse_args()
    if (not args.out.is_absolute() or not args.out.is_dir()
            or args.iters < 4 or args.iters > 10):
        p.error("existing absolute private output dir and iters 4..10")
    if (os.environ.get("U1_G0_APPROVED") != "YES"
            or os.environ.get("U1_EXCLUSIVE_APPROVED") != "YES"
            or os.environ.get("U1_RESTORE_PLAN_VERIFIED") != "YES"):
        p.error("G0, exclusive and restore hard gates must be approved")
    if any(k in os.environ for k in ("MF_KFC_BYPASS", "MF310P_V2_NOWARMUP")):
        p.error("no safety bypass")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        p.error("two workers required")
    for key, value in (
        ("U1_DEVICE_SHA256", D4_DEVICE_SHA),
        ("U1_HOST_SHA256", MEMFABRIC_HOST_SHA),
        ("U1_PLUGIN_SHA256", PLUGIN_SHA),
        ("U1_KVER_SHA256", KVER_SHA),
    ):
        if os.environ.get(key) != value:
            p.error(f"unverified runtime identity {key}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    cpu = dist.new_group(backend="gloo")
    result = []
    try:
        from torch_npu.profiler import ProfilerActivity, profile
        torch.manual_seed(4242)
        weight = torch.randn(N, K, device=device, dtype=torch.float16)
        weight_nz = torch_npu.npu_format_cast(weight, 29)
        stream = torch.npu.current_stream()
        for m in MS:
            torch.manual_seed(1200 + rank * 17 + m)
            x = torch.randn(m, K, device=device, dtype=torch.float16)
            ref = F.linear(x, weight).float().cpu()
            dist.all_reduce(ref, group=cpu)
            ref = ref.to(device, dtype=torch.float16)
            stream.synchronize()
            def invoke():
                y = F.linear(x, weight_nz)
                dist.all_reduce(y, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
                stream.synchronize()
                if not torch.equal(y, ref):
                    raise AssertionError(f"rank{rank} M{m}: NZ-Stock differs from Gloo FP32 reference")
            for _ in range(5):
                dist.barrier(group=cpu)
                invoke()
            trace = args.out / f"rank{rank}_m{m}_chrome.json"
            if trace.exists():
                raise FileExistsError(trace)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                record_shapes=True, with_stack=False
            ) as prof:
                for _ in range(args.iters):
                    dist.barrier(group=cpu)
                    invoke()
            prof.export_chrome_trace(str(trace))
            if not trace.is_file() or trace.stat().st_size == 0:
                raise RuntimeError("profiler failed to export device activity")
            result.append({"M": m, "rank": rank, "profile_chrome": trace.name,
                           "profile_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                           "bit_exact": True, "iters": args.iters,
                           "route": "NOT_VERIFIED_UNTIL_CANN_KERNEL_DETAILS_CSV"})
        dest = args.out / f"nz_probe_rank{rank}.json"
        with dest.open("x") as f:
            json.dump({"schema": "A1_NZ_STOCK_SHORT_PROFILE_V1", "rank": rank,
                       "workload": "M2048+4096/K2048/N2048/HCCL-TP2",
                       "full_NZ_matmul_route_verified": False,
                       "results": result}, f, indent=2)
            f.write("\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
