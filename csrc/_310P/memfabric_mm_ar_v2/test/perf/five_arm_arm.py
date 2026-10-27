#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""310P3五层完整算子实测：Stock / B1 native+MF / B2 full MM+MF / D4 / E1。
每个arm独立TP2进程；所有耗时均包含完整MM到可用AR SUM输出的热态窗口。
G0身份、设备独占、首写、异常和候选SO的装载门禁不可跳过。
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
import vllm_ascend.vllm_ascend_C as extension

from u1_spec import D4_DEVICE_SHA, MEMFABRIC_HOST_SHA, PLUGIN_SHA, KVER_SHA

K = N = 2048
MS = (2048,4096)
SCHEDULE = ("stock","b1","b2","d4","e1","e1","d4","b2","b1","stock")
WARMUP = 20
ITERS = 100


def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=range(1,len(SCHEDULE)+1),required=True)
    p.add_argument("--phase", choices=("smoke","confirm"),default="confirm")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--build-manifest",type=Path,default=None)
    p.add_argument("--store-url",default=os.environ.get(
        "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL",""))
    p.add_argument("--warmup",type=int,default=WARMUP)
    p.add_argument("--iters",type=int,default=ITERS)
    a=p.parse_args()
    a.arm=SCHEDULE[a.stage-1]
    if not a.out.is_absolute() or not a.out.is_dir():
        p.error("existing absolute output directory required")
    if a.phase=="confirm" and (a.warmup<20 or a.iters<100 or a.iters>500):
        p.error("confirm requires >=20 warmup and >=100 samples")
    if a.phase=="smoke" and (a.warmup<3 or a.iters<6 or a.iters>20):
        p.error("smoke requires 3 warmup and 6-20 samples")
    if int(os.getenv("WORLD_SIZE","0"))!=2 or os.getenv("RANK") not in ("0","1"):
        p.error("true TP2 two processes required")
    prefix="smoke_" if a.phase=="smoke" else ""
    if (a.out/f"{prefix}{a.stage:02d}_{a.arm}_rank{os.environ['RANK']}.json").exists():
        p.error("no overwriting saved evidence")
    if any(x in os.environ for x in (
        "MF_KFC_BYPASS","MF310P_V2_NOWARMUP","MF310P_V2_AR_ONLY_NOWARMUP",
        "MF310P_V2_AR_ONLY_PARTIAL_GUARD","MF_SDMA_ORCH_EPOCH_BUDGET_S")):
        p.error("first-write/KFC/safety bypass forbidden")
    identity={"U1_DEVICE_SHA256":D4_DEVICE_SHA,
              "U1_PLUGIN_SHA256":PLUGIN_SHA,
              "U1_HOST_SHA256":MEMFABRIC_HOST_SHA,
              "U1_KVER_SHA256":KVER_SHA}
    if any(os.environ.get(k)!=v for k,v in identity.items()):
        p.error("G0 verified production D4/Host/Plugin/KVER identity required")
    if sha(Path(extension.__file__))!=PLUGIN_SHA:
        p.error("actual imported C extension hash changed")
    formal=Path(os.environ.get("U1_DEVICE_SO_PATH","/nonexistent"))
    if not formal.is_file() or sha(formal)!=D4_DEVICE_SHA:
        p.error("actual frozen D4 Device SO mismatch")
    if not os.environ.get("MF_SDMA_ORCH_JSON","").endswith(
            "/libmf_sdma_orch_v11.json"):
        p.error("KVER11 loader configuration not verified")
    if a.arm!="stock":
        if not a.store_url.startswith("tcp://127.0.0.1:"):
            p.error("verified private TCP local MF rendezvous needed")
        actual=Path(os.environ.get("V2_HOST_LIB","/nonexistent"))
        if not actual.is_file() or actual.resolve()!=Path(extension.__file__).resolve():
            p.error("trial Host library must match actual imported extension")
    if a.arm in ("b2","e1"):
        if a.build_manifest is None or not a.build_manifest.is_absolute():
            p.error("B2/E1 isolated build manifest required")
        meta=json.loads(a.build_manifest.read_text(encoding="utf-8"))
        if meta.get("schema")!="FIVE_ARM_E_DCCI_ENTIRE_BUILD_V1":
            p.error("wrong isolated candidate manifest schema")
        for key,which in (("candidate_sha256","candidate_so"),
                          ("private_host_sha256","private_host_so")):
            lib=Path(meta.get(which,"/nonexistent"))
            if not lib.is_file() or sha(lib)!=meta.get(key):
                p.error(f"{which} identity mismatch")
        a.meta=meta
    else:
        a.meta=None
    return a


def bind_trial(lib):
    P=ctypes.c_void_p; U=ctypes.c_uint64; I=ctypes.c_uint32
    sigs={
        "mf310p_v2_trial_create":([ctypes.c_int,ctypes.c_char_p,I,ctypes.POINTER(P)],ctypes.c_int),
        "mf310p_v2_trial_ib_probe":([P,I,P],ctypes.c_int),
        "mf310p_v2_trial_run":([P,U,U,U,I,I,P],ctypes.c_int),
        "mf310p_v2_trial_run_ar_only":([P,U,U,I,I,P],ctypes.c_int),
        "mf310p_v2_trial_check_status":([P,ctypes.POINTER(U)],ctypes.c_int),
        "mf310p_v2_trial_destroy":([P],ctypes.c_int),
    }
    for name,(args,res) in sigs.items():
        f=getattr(lib,name);f.argtypes=args;f.restype=res
    return lib


def need(rc,label):
    if rc: raise RuntimeError(f"{label} failed: {rc}")


def pct(data,q):
    s=sorted(data)
    pos=(len(s)-1)*q
    lo=int(pos)
    hi=min(lo+1,len(s)-1)
    return s[lo]+(s[hi]-s[lo])*(pos-lo)


def main():
    a=parse()
    rank=int(os.environ["RANK"])
    local=int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    torch.npu.config.allow_internal_format=True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    dev=torch.device(f"npu:{local}")
    # 和U1一样禁止同一进程装载HCCL数据面+MemFabric建池。
    if a.arm=="stock":
        dist.init_process_group(backend="hccl")
        group=dist.new_group(backend="gloo")
        hccl=dist.group.WORLD
    else:
        dist.init_process_group(backend="gloo")
        group=dist.group.WORLD
        hccl=None
    stream=torch.npu.current_stream()
    raw=ctypes.c_void_p(stream.npu_stream)
    handle=ctypes.c_void_p()
    trial=None
    candidate=None
    failure=None
    try:
        torch.manual_seed(4242)
        w=torch.randn(N,K,dtype=torch.float16,device=dev)
        w_nz=torch_npu.npu_format_cast(w,29)
        # 必须在MemFabric建池前完成全部CPU FP32跨Rank参考与输入变体。
        inputs={}
        for m in MS:
            xset=[]; refs=[]
            for variant in range(2):
                torch.manual_seed(1200+rank*17+m+variant*1009)
                x=torch.randn(m,K,dtype=torch.float16,device=dev)
                local_partial=F.linear(x,w).float().cpu()
                dist.all_reduce(local_partial,group=group)
                xset.append(x)
                refs.append(local_partial.to(device=dev,dtype=torch.float16))
            inputs[m]=(xset,refs)
        outputs={m:[torch.empty((m,N),dtype=torch.float16,device=dev)
                    for _ in range(2)] for m in MS}
        stream.synchronize()  # NZ29首次物化边界
        if a.arm!="stock":
            trial=bind_trial(ctypes.CDLL(str(extension.__file__)))
            dist.barrier(group=group)
            need(trial.mf310p_v2_trial_create(
                rank,a.store_url.encode(),1024,ctypes.byref(handle)),"pool create")
            if not handle.value: raise RuntimeError("null MemFabric pool")
            need(trial.mf310p_v2_trial_ib_probe(handle,1000,raw),"IB real probe")
            stream.synchronize()
            if a.arm in ("b2","e1"):
                candidate=ctypes.CDLL(str(a.meta["private_host_so"]))
                f=candidate.mf310p_five_arm_run_private
                f.argtypes=[ctypes.c_void_p]+[ctypes.c_uint64]*3+[
                    ctypes.c_uint32]*2+[ctypes.c_void_p]
                f.restype=ctypes.c_int
                proc_maps=Path("/proc/self/maps").read_text(encoding="utf-8")
                for loc in (a.meta["candidate_so"],a.meta["private_host_so"]):
                    if str(Path(loc).resolve()) not in proc_maps:
                        raise RuntimeError("private candidate binary not mapped: "+loc)
        results={}
        for m in MS:
            xs,refs=inputs[m]
            warm=a.warmup
            n=a.iters
            vals=[]; wall_samples=[]
            ev0=torch.npu.Event(enable_timing=True)
            ev1=torch.npu.Event(enable_timing=True)
            # Stable output slot per input; AR-only B1 allocates native Partial.
            for i in range(warm+n):
                variant=i%2
                x=xs[variant];ref=refs[variant]
                output=outputs[m][variant]
                dist.barrier(group=group)
                t0=time.perf_counter_ns()
                ev0.record(stream)
                partial=None
                if a.arm=="stock":
                    got=F.linear(x,w_nz)
                    dist.all_reduce(got,op=dist.ReduceOp.SUM,group=hccl)
                elif a.arm=="b1":
                    partial=F.linear(x,w_nz)
                    # T2正确性合同：生产者输出->stage需同流同步；
                    # 此强制同步必须包含在Event/Wall窗口内，绝不隐藏。
                    stream.synchronize()
                    need(trial.mf310p_v2_trial_run_ar_only(
                        handle,partial.data_ptr(),output.data_ptr(),m,8,raw),
                        "B1 native+MF AR-only")
                    got=output
                elif a.arm=="d4":
                    need(trial.mf310p_v2_trial_run(
                        handle,x.data_ptr(),w_nz.data_ptr(),
                        output.data_ptr(),m,1,raw),"D4")
                    got=output
                else:
                    kind=1 if a.arm=="b2" else 2
                    need(f(handle,x.data_ptr(),w_nz.data_ptr(),
                           output.data_ptr(),m,kind,raw),
                         "private B2/E1 candidate")
                    got=output
                ev1.record(stream)
                stream.synchronize()
                wall=(time.perf_counter_ns()-t0)/1000.
                elapsed=ev0.elapsed_time(ev1)*1000.
                if not math.isfinite(elapsed) or elapsed<=0:
                    raise RuntimeError("invalid NPU event")
                if a.arm!="stock":
                    status=ctypes.c_uint64()
                    need(trial.mf310p_v2_trial_check_status(
                        handle,ctypes.byref(status)),"protocol status read")
                    if status.value:raise RuntimeError(
                        f"protocol status={status.value} arm={a.arm} M={m} i={i}")
                # 每一调用都要变内容bit-exact，不凭稳定固定输入掩盖旧数据。
                if not torch.equal(got,ref):
                    raise AssertionError(f"bit-exact FAIL Rank{rank} {a.arm} M{m} i={i}")
                if i>=warm:
                    vals.append(round(elapsed,4))
                    wall_samples.append(round(wall,4))
            results[str(m)]={
                "M":m,"rank":rank,"arm":a.arm,"bit_exact":True,"protocol_status":0,
                "warmup":warm,"iters":n,"npu_event_us":vals,"host_wall_us":wall_samples,
                "event_p50_us":round(statistics.median(vals),3),
                "event_p90_us":round(pct(vals,.9),3),
                "event_p99_us":round(pct(vals,.99),3),
            }
            print(f"ARM={a.arm} rank={rank} M={m} samples={len(vals)} PASS",flush=True)
        prefix="smoke_" if a.phase=="smoke" else ""
        out=a.out/f"{prefix}{a.stage:02d}_{a.arm}_rank{rank}.json"
        with out.open("x",encoding="utf-8") as fh:
            json.dump({
                "schema":"A1_FIVE_ARM_TP2_V1","stage":a.stage,"arm":a.arm,"phase":a.phase,
                "rank":rank,"TP":2,"weight_format":"NZ29",
                "K":K,"N":N,"Ms":list(MS),"stock_route_required":"aclnnMatmulWeightNz",
                "jit_compile":False,"allow_internal_format":True,
                "data_variants":2,"reference":"CPU_GLOO_FP32_SUM_TO_FP16",
                "host_sha256":MEMFABRIC_HOST_SHA,
                "production_device_sha256":D4_DEVICE_SHA,
                "candidate":a.meta if a.arm in ("b2","e1") else None,
                "profile":False,"results":results,
                "metric":"local same-rank Event for complete MM+AR; TP2 analysis pairs rank maxima",
                "limits":"B1 includes required stage, 2 sacrificial launches and producer stream sync; "
                         "B2 uses one MM phase then logical AR internally packetized",
            },fh,ensure_ascii=False,indent=2)
            fh.write("\n")
        dist.barrier(group=group)
    except BaseException as ex:
        failure=ex
        raise
    finally:
        if handle.value:
            try:
                stream.synchronize()
                need(trial.mf310p_v2_trial_destroy(handle),"pool destroy")
            except BaseException:
                if failure is None: raise
        if dist.is_initialized():dist.destroy_process_group()


if __name__=="__main__":
    main()
