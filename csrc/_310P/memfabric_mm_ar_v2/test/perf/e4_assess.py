#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""十四场双Rank E4对照：Stock/E1/E3-G/三档AddTile/半组Add，仅Confirm可签性能。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SCHEDULE=("stock","e1","e3g","e4t2","e4t4","e4t8","e4p","e4p","e4t8","e4t4","e4t2","e3g","e1","stock")
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
        "schema":"E4_LAYERED_TP2_V1","stage":stage,"arm":arm,
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
        if arm in ("e1","e3g","e4t2","e4t4","e4t8","e4p"):
            meta0=pair[0].get("candidate")
            meta1=pair[1].get("candidate")
            if not meta0 or not meta1 or meta0!=meta1:
                raise ValueError("E1/E3/E4 binary identity mismatch across ranks")
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
        raise ValueError("E1/E3/E4 candidate SHA drift across sessions")
    comps=[]
    status="E4_LAYERED_TIMING_VALID"
    for m in MS:
        by=data[m]
        if any(len(x)!=2 for x in by.values()):
            raise ValueError("wrong balanced E4 fourteen-session matrix")
        drift={arm:round(max(x)/min(x)-1,5) for arm,x in by.items()}
        good=all(q<=MAX_DRIFT for q in drift.values())
        if not good:status="INCONCLUSIVE_DRIFT"
        t={arm:round(statistics.median(x),3) for arm,x in by.items()}
        a=t["stock"];e=t["e1"];g=t["e3g"]
        arms=("e4t2","e4t4","e4t8","e4p")
        # 只筛选说明，不将不同阶段p50差值冒充内核阶段时间。
        comparisons={
            arm:{
                "vs_E3G_pct":round((g/t[arm]-1)*100,3),
                "delta_vs_E3G_us":round(t[arm]-g,3),
                "vs_Stock_pct":round((a/t[arm]-1)*100,3),
                "delta_vs_Stock_us":round(t[arm]-a,3),
            } for arm in arms
        }
        best=min(arms,key=lambda arm:t[arm])
        comps.append({
            "M":m,"stable":good,
            "arm_session_p50_us":by,"arm_time_us":t,"drift":drift,
            "E3_G_vs_E1_pct":round((e/g-1)*100,3),
            "E3_G_vs_Stock_pct":round((a/g-1)*100,3),
            "e4_vs_baselines":comparisons,
            "best_E4_arm":best,"best_E4_time_us":t[best],
            "best_E4_delta_vs_Stock_us":round(t[best]-a,3),
            "per_session_p99_us":{
                arm:[raw[j]["Ms"][str(m)]["paired_p99_us"]
                     for j,a0 in enumerate(SCHEDULE) if a0==arm]
                for arm in set(SCHEDULE)
            },
        })
    return {
        "schema":"E4_LAYERED_ASSESS_V1",
        "status":status,
        "stock_native_NZ_route_verified":len(routes)==4,
        "native_route_evidence":routes,
        "E1_E3_E4_device_correctness_on_new_binary":"BIT_EXACT_IN_ALL_14_SESSIONS" if status!="INCONCLUSIVE_DRIFT" else "BIT_EXACT_BUT_DRIFT",
        "candidate_work_head":heads[0],
        "comparisons":comps,"raw_sessions":raw,
        "limitations":[
            "E4 AddTile controls FP16 elements per vector call, not rows.",
            "E4 packet-ready candidate leaves Core0 as only mail consumer and schedules core1-3 or4-7 Add when that packet arrives.",
            "E4 still waits for complete 2048-row producer MM before sending first packet; it does not address early producer overlap.",
            "Only M2048/M4096 have experimental executables; E4-B offline planner cannot be counted as device support for intermediate M.",
            "No kernel phase time or Serving/TTFT inference from per-arm p50.",
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
