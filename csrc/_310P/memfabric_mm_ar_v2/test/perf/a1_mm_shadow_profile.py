#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""独立Native/Shadow短Profiler; 与无Profiler正式计时完全分离。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch_npu.profiler import profile,ProfilerActivity,tensorboard_trace_handler

from a1_mm_shadow_arm import digest, K, N, MS, SHADOW_SYMBOL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("native","nd"),required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--shadow-so",type=Path,required=True)
    p.add_argument("--build-manifest",type=Path,required=True)
    args=p.parse_args()
    if (int(os.getenv("WORLD_SIZE","0"))!=2 or not args.out.is_absolute()
            or not args.out.is_dir() or list(args.out.iterdir())):
        p.error("two ranks; empty precreated absolute output required")
    manifest=json.loads(args.build_manifest.read_text(encoding="utf-8"))
    if manifest.get("so_sha256")!=digest(args.shadow_so):
        p.error("binary identity mismatch")
    rank=int(os.environ["RANK"])
    local=int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    torch.npu.config.allow_internal_format=True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group(backend="gloo")
    try:
        import ctypes
        lib=ctypes.CDLL(str(args.shadow_so))
        fn=getattr(lib,SHADOW_SYMBOL)
        fn.argtypes=[ctypes.c_uint64]*3+[ctypes.c_uint32]*3+[ctypes.c_void_p]
        fn.restype=ctypes.c_int
        stream=torch.npu.current_stream()
        raw=ctypes.c_void_p(stream.npu_stream)
        torch.manual_seed(4242)
        w=torch.randn(N,K,device=f"npu:{local}",dtype=torch.float16)
        nz=torch_npu.npu_format_cast(w,29)
        for m in MS:
            torch.manual_seed(1200+rank*17+m)
            x=torch.randn(m,K,device=f"npu:{local}",dtype=torch.float16)
            ref=F.linear(x,w)
            out=torch.empty_like(ref)
            def call():
                if args.arm=="native":
                    return F.linear(x,nz)
                rc=fn(x.data_ptr(),nz.data_ptr(),out.data_ptr(),m,0,1,raw)
                if rc:raise RuntimeError(f"shadow route rc={rc}")
                return out
            if args.arm=="nd":
                for _ in range(2):
                    if fn(0,0,0,m,0,0,raw):
                        raise RuntimeError("MM shadow no-op prewarm failed")
            for _ in range(5):
                y=call()
                stream.synchronize()
                if not torch.equal(y,ref):
                    raise AssertionError(f"rank{rank} M{m} route correctness failed")
            dist.barrier()
            folder=args.out/f"rank{rank}_m{m}_profiler"
            folder.mkdir(mode=0o700,exist_ok=False)
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
                         record_shapes=True,with_stack=False,
                         on_trace_ready=tensorboard_trace_handler(
                             str(folder),worker_name=f"rank{rank}_m{m}")):
                for _ in range(6):
                    y=call()
                    stream.synchronize()
            traces=[p for p in folder.rglob("kernel_details.csv")
                    if p.is_file() and "ASCEND_PROFILER_OUTPUT" in p.parts
                    and p.stat().st_size>100]
            if len(traces)!=1:
                raise RuntimeError(f"expected 1 parsed CANN CSV rank{rank}/M{m}; found {len(traces)}")
            dest=args.out/f"rank{rank}_m{m}_kernel_details.csv"
            with traces[0].open("rb") as inp,dest.open("xb") as outfp:
                shutil.copyfileobj(inp,outfp)
            with dest.open(encoding="utf-8-sig",newline="") as f:
                raw_rows=list(csv.DictReader(f))
            names=[row.get("Name","") for row in raw_rows]
            if args.arm=="native":
                found=sum(name.startswith("aclnnMatmulWeightNz_") for name in names)
                bad=[name for name in names if "TransData" in name]
            else:
                found=sum("V2MmShadowKernel" in name for name in names)
                bad=[]
            if found<3 or bad:
                raise RuntimeError(
                    f"route invalid {args.arm} rank{rank} M{m}: actual kernels={found}, unexpected TransData={bad[:3]}")
            destmeta=args.out/f"rank{rank}_m{m}_route.json"
            destmeta.write_text(json.dumps({
                "schema":"A1_MM_SHADOW_ROUTE_V1","arm":args.arm,"rank":rank,"M":m,
                "kernel_hits":found,"csv_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
                "shadow_sha256":manifest["so_sha256"],"bit_exact":True,
                "CANN_original":str(traces[0].relative_to(args.out))
            },ensure_ascii=False,indent=2)+"\n")
            dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=="__main__":
    main()
