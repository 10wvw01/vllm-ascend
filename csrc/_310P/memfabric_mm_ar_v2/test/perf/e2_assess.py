#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""十场双Rank E2性能对照：Stock/E1/E2-R/E2-P(1024,512)，不混旧D4漂移。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SCHEDULE=("stock","e1","e2r","e2p","e2p512","e2p512","e2p","e2r","e1","stock")
MS=(2048,4096)
MAX_DRIFT=0.05


def percentile(data,q):
    seq=sorted(data)
    t=(len(seq)-1)*q
    lo=int(t)
    hi=min(lo+1,len(seq)-1)
    return seq[lo]+(seq[hi]-seq[lo])*(t-lo)


def checked(path,stage,arm,rank):
    obj=json.loads(path.read_text(encoding="utf-8"))
    expected={
        "schema":"E2_PIPELINE_TP2_V1","stage":stage,"arm":arm,
        "rank":rank,"TP":2,"K":2048,"N":2048,
        "weight_format":"NZ29","jit_compile":False,
        "allow_internal_format":True,"profile":False,
        "reference":"CPU_GLOO_FP32_SUM_TO_FP16"
    }
    if any(obj.get(k)!=v for k,v in expected.items()):
        raise ValueError(f"identity or measurement mismatch: {path}")
    results=obj.get("results",{})
    if set(results)!=set(map(str,MS)):
        raise ValueError("not both target M values")
    for m in MS:
        r=results[str(m)]
        if (r.get("M")!=m or r.get("rank")!=rank or
            r.get("arm")!=arm or r.get("bit_exact") is not True or
            r.get("protocol_status")!=(None if arm=="stock" else 0) or r.get("warmup",0)<20 or
            r.get("iters",0)<100):
            raise ValueError(f"incorrect/corrupt Rank: {path} M{m}")
        vals=r.get("npu_event_us")
        wall=r.get("host_wall_us")
        if not isinstance(vals,list) or len(vals)!=r["iters"]:
            raise ValueError("missing event samples")
        if not isinstance(wall,list) or len(wall)!=len(vals):
            raise ValueError("missing wall samples")
        if any(type(v) not in (float,int) or not math.isfinite(v) or v<=0
               for v in vals+wall):
            raise ValueError("NaN/zero/negative sample")
    return obj


def verify_native_routes(route_dir):
    evidence=[]
    if route_dir is None:
        return evidence
    for m in MS:
        for rank in (0,1):
            jp=route_dir/f"rank{rank}_m{m}_route.json"
            cp=route_dir/f"rank{rank}_m{m}_kernel_details.csv"
            if not jp.is_file() or not cp.is_file():
                raise ValueError("native NZ route files missing")
            obj=json.loads(jp.read_text(encoding="utf-8"))
            if (obj.get("schema")!="FIVE_ARM_REAL_NZ_HCCL_ROUTE_V1" or
                obj.get("rank")!=rank or obj.get("M")!=m or
                obj.get("native_nz") is not True or
                obj.get("nz_mm_kernel_hits",0)<3 or
                obj.get("hcom_allreduce_kernel_hits",0)<3 or
                obj.get("transdata_hits")!=0 or
                obj.get("csv_sha256")!=hashlib.sha256(cp.read_bytes()).hexdigest()):
                raise ValueError("NZ kernel/HCCL real route proof invalid")
            evidence.append(obj)
    return evidence


def analyze(folder, route_dir=None):
    routes=verify_native_routes(route_dir)
    data={m:{arm:[] for arm in set(SCHEDULE)} for m in MS}
    raw=[]
    heads=[]
    for stage,arm in enumerate(SCHEDULE,1):
        pair=[
            checked(folder/f"{stage:02d}_{arm}_rank{rank}.json",stage,arm,rank)
            for rank in (0,1)
        ]
        if arm in ("e1","e2r","e2p","e2p512"):
            meta0=pair[0].get("candidate")
            meta1=pair[1].get("candidate")
            if not meta0 or not meta1 or meta0!=meta1:
                raise ValueError("candidate E1/E2 binary identity mismatch across ranks")
            heads.append(meta0.get("work_head"))
        stage_record={"stage":stage,"arm":arm,"Ms":{}}
        for m in MS:
            v0=pair[0]["results"][str(m)]["npu_event_us"]
            v1=pair[1]["results"][str(m)]["npu_event_us"]
            paired=[max(a,b) for a,b in zip(v0,v1)]
            entry={
                "paired_p50_us":round(statistics.median(paired),3),
                "paired_p90_us":round(percentile(paired,.9),3),
                "paired_p99_us":round(percentile(paired,.99),3),
                "r0_p50_us":round(statistics.median(v0),3),
                "r1_p50_us":round(statistics.median(v1),3),
                "n":len(paired),"rank0_sha256":hashlib.sha256(
                    (folder/f"{stage:02d}_{arm}_rank0.json").read_bytes()).hexdigest(),
                "rank1_sha256":hashlib.sha256(
                    (folder/f"{stage:02d}_{arm}_rank1.json").read_bytes()).hexdigest(),
            }
            stage_record["Ms"][str(m)]=entry
            data[m][arm].append(entry["paired_p50_us"])
        raw.append(stage_record)
    if len(set(heads))!=1:
        raise ValueError("E1/E2 candidate SHA drift across sessions")
    comps=[]
    status="E2_FULL_CHAIN_TIMING_VALID"
    for m in MS:
        by=data[m]
        if any(len(x)!=2 for x in by.values()):
            raise ValueError("wrong balanced E2 ten-session matrix")
        drift={arm:round(max(x)/min(x)-1,5) for arm,x in by.items()}
        good=all(q<=MAX_DRIFT for q in drift.values())
        if not good:status="INCONCLUSIVE_DRIFT"
        t={arm:round(statistics.median(x),3) for arm,x in by.items()}
        a=t["stock"]
        e1=t["e1"]
        r=t["e2r"]
        p=t["e2p"]
        p512=t["e2p512"]
        comps.append({
            "M":m,"stable":good,"arm_session_p50_us":by,
            "arm_time_us":t,"drift":drift,
            "E2_R_vs_E1_pct":round((e1/r-1)*100,3),
            "E2_P_1024_vs_E1_pct":round((e1/p-1)*100,3),
            "E2_P_512_vs_E1_pct":round((e1/p512-1)*100,3),
            "E2_R_vs_Stock_pct":round((a/r-1)*100,3),
            "E2_P_1024_vs_Stock_pct":round((a/p-1)*100,3),
            "E2_P_512_vs_Stock_pct":round((a/p512-1)*100,3),
            "E1_vs_Stock_pct":round((a/e1-1)*100,3),
            "best_E2_arm":min(("e2r","e2p","e2p512"),key=t.__getitem__),
            "best_E2_p50_us":min(r,p,p512),
            "best_E2_improvement_vs_E1_us":round(e1-min(r,p,p512),3),
            "best_E2_remaining_to_Stock_us":round(min(r,p,p512)-a,3),
            "all_arm_session_p99_us":{
                arm:[raw[j]["Ms"][str(m)]["paired_p99_us"] for j,a0 in enumerate(SCHEDULE) if a0==arm]
                for arm in ("stock","e1","e2r","e2p","e2p512")
            },
        })
    return {
        "schema":"E2_PIPELINE_ASSESS_V1",
        "status":status,
        "stock_native_NZ_route_verified":len(routes)==4,
        "native_route_evidence":routes,
        "E1_E2_device_correctness_on_new_binary":"BIT_EXACT_IN_ALL_10_SESSIONS" if status!="INCONCLUSIVE_DRIFT" else "BIT_EXACT_BUT_DRIFT",
        "candidate_work_head":heads[0],
        "comparisons":comps,"raw_sessions":raw,
        "limitations":[
            "E2-R replaces 1-to-7 Add release with a hierarchical tree, E2-P additionally introduces READY-fenced receive pre-invalidation.",
            "E2-P/512 versus E2-P/1024 changes both MM tile and packet count; cannot isolate pure SDMA or pure overlap with simple subtraction.",
            "E1 is recompiled from exactly the same immutable source in the E2 candidate SO; compare E1/E2 in balanced independent processes.",
            "No kernel phase timestamp is inferred by subtracting arm p50, and no periodic-tail cause is claimed without device timeline.",
            "Hardware tests are NPU-specific, Graph/Serving are separately NOT_RUN.",
        ]
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions",type=Path,required=True)
    p.add_argument("--route-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if not a.sessions.is_absolute() or not a.sessions.is_dir() or not a.out.is_absolute() or a.out.exists():
        p.error("existing absolute session directory and fresh absolute output path required")
    if not a.route_dir.is_absolute() or not a.route_dir.is_dir():
        p.error("verified independent absolute route directory required")
    result=analyze(a.sessions,a.route_dir)
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write("\n")
    print(result["status"])


if __name__=="__main__":
    main()
