#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""五臂原生NZ Stock真实路由证据；短Profiler仅证明Kernel选择，不参与性能计时。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch_npu.profiler import profile,ProfilerActivity,tensorboard_trace_handler

from u1_spec import PLUGIN_SHA


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if not a.out.is_absolute() or not a.out.is_dir():
        p.error("existing absolute output directory")
    if int(os.environ.get("WORLD_SIZE","0"))!=2:
        p.error("two-rank NZ route required")
    rank=int(os.environ["RANK"])
    local=int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    torch.npu.config.allow_internal_format=True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group(backend="hccl")
    try:
        import vllm_ascend.vllm_ascend_C as ext
        if hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest()!=PLUGIN_SHA:
            raise RuntimeError("production extension SHA mismatch")
        stream=torch.npu.current_stream()
        torch.manual_seed(4242)
        w=torch.randn(2048,2048,dtype=torch.float16,device=f"npu:{local}")
        nz=torch_npu.npu_format_cast(w,29)
        stream.synchronize()
        for m in (2048,4096):
            torch.manual_seed(1200+rank*17+m)
            x=torch.randn(m,2048,dtype=torch.float16,device=f"npu:{local}")
            for i in range(3):
                y=F.linear(x,nz)
                dist.all_reduce(y)
                stream.synchronize()
            dist.barrier()
            target=a.out/f"rank{rank}_m{m}"
            if target.exists():raise FileExistsError(target)
            target.mkdir(mode=0o700)
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
                         record_shapes=True,with_stack=False,
                         on_trace_ready=tensorboard_trace_handler(
                             str(target),worker_name=f"rank{rank}_m{m}")):
                for _ in range(6):
                    y=F.linear(x,nz)
                    dist.all_reduce(y)
                    stream.synchronize()
            files=[q for q in target.rglob("kernel_details.csv")
                   if "ASCEND_PROFILER_OUTPUT" in q.parts and q.is_file()
                   and q.stat().st_size>100]
            if len(files)!=1:
                raise RuntimeError("native route: exactly one CANN CSV required")
            csvpath=a.out/f"rank{rank}_m{m}_kernel_details.csv"
            with files[0].open("rb") as source,csvpath.open("xb") as out:
                shutil.copyfileobj(source,out)
            with csvpath.open(encoding="utf-8-sig",newline="") as fh:
                rows=list(csv.DictReader(fh))
            mm=[q for q in rows if q.get("Name","").startswith("aclnnMatmulWeightNz_")]
            hccl=[q for q in rows if "hcom_allReduce" in q.get("Name","")]
            bad=[q for q in rows if "TransData" in q.get("Name","")]
            if len(mm)<3 or len(hccl)<3 or bad:
                raise RuntimeError(
                    f"NZ direct Stock route failed: MM={len(mm)} AR={len(hccl)} TransData={len(bad)}")
            def times(records):
                vals=[float(r["Duration(us)"]) for r in records]
                if any(not math.isfinite(v) or v<=0 for v in vals):
                    raise RuntimeError("invalid CANN kernel duration")
                return vals
            mm_vals=times(mm);hccl_vals=times(hccl)
            evidence={
                "schema":"FIVE_ARM_REAL_NZ_HCCL_ROUTE_V1",
                "rank":rank,"M":m,"nz_mm_kernel_hits":len(mm),
                "hcom_allreduce_kernel_hits":len(hccl),
                "transdata_hits":0,"native_nz":True,
                "mm_kernel_p50_us":round(statistics.median(mm_vals),4),
                "hccl_kernel_p50_us":round(statistics.median(hccl_vals),4),
                "csv_sha256":hashlib.sha256(csvpath.read_bytes()).hexdigest(),
                "profile":True,
                "note":"route-only, never substitute pure kernel profiler medians for full-chain no-profiler p50"
            }
            (a.out/f"rank{rank}_m{m}_route.json").write_text(
                json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=="__main__":
    main()
