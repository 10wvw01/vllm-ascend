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
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
import vllm_ascend.vllm_ascend_C as extension
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
    if any(k in os.environ for k in ("MF_KFC_BYPASS", "MF310P_V2_NOWARMUP", "MF310P_V2_AR_ONLY_NOWARMUP")):
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
    host_path = os.environ.get("V2_HOST_LIB", "")
    if (not host_path or Path(host_path).resolve() != Path(extension.__file__).resolve()
            or hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest() != PLUGIN_SHA):
        p.error("actual imported plugin identity is not approved")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 2 or not all(x.isdecimal() for x in visible) or visible[0] == visible[1]:
        p.error("pin verified same-card dual-die physical devices")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    # Serving等价启动：镜像vllm_ascend serve的model_runner_v1.py:229与
    # _310p/worker_310p.py:44。缺失时F.linear(x,weight_nz)实测走通用MatMul+
    # TransData（ND输出~1830µs）而非真实站点的aclnnMatmulWeightNz（~770µs），
    # 路由门会正确拒绝该假NZ基线。
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    cpu = dist.new_group(backend="gloo")
    result = []
    try:
        from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler
        # 镜像vllm_ascend TorchNPUProfilerWrapper的_ExperimentalConfig：
        # 没有它CANN分析只输出任务级精简CSV（无aclnn名/形状/格式），
        # 路由门无法证明真实NZ Kernel。
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None)
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
            # 同一个Rank/M单独采集，原生tensorboard_handler才产生解析后的
            # ASCEND_PROFILER_OUTPUT/kernel_details.csv。不得伪造或从Chrome猜表。
            trace_dir = args.out / f"rank{rank}_m{m}_profiler"
            trace_dir.mkdir(mode=0o700, exist_ok=False)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                with_stack=False,
                experimental_config=experimental_config,
                on_trace_ready=tensorboard_trace_handler(
                    str(trace_dir), worker_name=f"rank{rank}_m{m}")
            ) as prof:
                for _ in range(args.iters):
                    dist.barrier(group=cpu)
                    invoke()
            # CANN离线分析可能滞后于trace导出；等待真正解析出的CSV。
            candidates = []
            for _ in range(60):
                candidates = [
                    f for f in trace_dir.rglob("kernel_details.csv")
                    if f.is_file() and f.stat().st_size > 100
                    and "ASCEND_PROFILER_OUTPUT" in f.parts
                ]
                if len(candidates) == 1:
                    break
                time.sleep(2)
            if len(candidates) != 1:
                raise RuntimeError(
                    f"rank{rank} M{m} expected 1 real CANN kernel_details.csv, "
                    f"found {len(candidates)}; preserve {trace_dir} and STOP "
                    "(CANN9.1 profile parser may require offline analyse)")
            real_csv = candidates[0]
            trace = args.out / f"rank{rank}_m{m}_kernel_details.csv"
            with real_csv.open("rb") as inp, trace.open("xb") as dst:
                shutil.copyfileobj(inp, dst)
            if hashlib.sha256(trace.read_bytes()).digest() != hashlib.sha256(
                    real_csv.read_bytes()).digest():
                raise RuntimeError("CANN CSV copy hash mismatch")
            result.append({"M": m, "rank": rank,
                           "raw_profiler_directory": trace_dir.name,
                           "CANN_kernel_details_source": str(real_csv.relative_to(args.out)),
                           "CANN_kernel_details_csv": trace.name,
                           "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                           "bit_exact": True, "iters": args.iters,
                           "route": "UNVERIFIED_UNTIL_FOUR_RANK_M_ROUTE_GATE"})
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
