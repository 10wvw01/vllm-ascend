#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MM-only 对照的严格CPU验收；不把影子MM数字冒充D4融合内部精确阶段时长。"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

BASE = ("native", "nd", "nd", "native", "native", "nd", "nd", "native")
MEMORY = ("sym", "sym_clean", "sym_clean", "sym")
MS = (2048, 4096)
MAX_STOCK_DRIFT = 0.05


def pct(values, q: float) -> float:
    v = sorted(values)
    k = (len(v)-1)*q
    lo = int(k)
    hi = min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(k-lo)


def read_session(directory: Path, arm: str, sha: str) -> dict:
    parsed = []
    for rank in (0,1):
        path = directory / f"{arm}_rank{rank}.json"
        if not path.is_file():
            raise ValueError(f"missing original Rank {path}")
        r = json.loads(path.read_text(encoding="utf-8"))
        keys = {"schema":"A1_MM_SHADOW_RANK_V1","arm":arm,"rank":rank,
                "world_size":2,"shadow_sha256":sha,
                "d4_sha256":"2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19",
                "weight_layout":"NZ29","profile":False}
        if any(r.get(k)!=v for k,v in keys.items()):
            raise ValueError(f"{directory.name}/rank{rank} identity mismatch")
        if (r.get("sym_pool_created") is not (arm in ("sym","sym_clean"))
                or len(r.get("results",[])) != 2):
            raise ValueError("memory pool or M results mismatch")
        parsed.append(r)
    stats = {}
    for m in MS:
        series = []
        for rank,rec in enumerate(parsed):
            result = next((x for x in rec["results"] if x.get("M")==m),None)
            if result is None or result.get("rank") != rank or result.get("arm") != arm:
                raise ValueError("wrong M/rank/arm")
            if (result.get("bit_exact") is not True or result.get("warmup",0)<20
                    or result.get("iters",0)<100):
                raise ValueError("insufficient correctness/warmup/sample proof")
            vals = result.get("samples",[])
            if not isinstance(vals,list) or len(vals)!=result["iters"]:
                raise ValueError("raw samples missing")
            for i,v in enumerate(vals):
                if (v.get("i") != i or v.get("variant") != i%2 or
                        any(type(v.get(k)) not in (int,float) or
                            not math.isfinite(v[k]) or v[k]<=0
                            for k in ("event_us","wall_us"))):
                    raise ValueError(f"bad sample {directory.name}/rank{rank}/M{m}/#{i}")
            series.append([v["event_us"] for v in vals])
        paired=[max(a,b) for a,b in zip(*series)]
        stats[str(m)]={"paired_p50_us":round(statistics.median(paired),3),
                       "paired_p90_us":round(pct(paired,.9),3),
                       "paired_p99_us":round(pct(paired,.99),3),
                       "rank0_p50_us":round(statistics.median(series[0]),3),
                       "rank1_p50_us":round(statistics.median(series[1]),3)}
    return stats


def evaluate(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError("missing campaign root")
    build=json.loads((root/"build_manifest.json").read_text(encoding="utf-8"))
    if build.get("schema") != "A1_MM_SHADOW_BUILD_V1" or not build.get("so_sha256"):
        raise ValueError("not actual source pinned MM shadow")
    sha=build["so_sha256"]
    # 主对照必须有完全独立于无Profiler计时的4份原生/影子Kernel明细。
    route_proofs=[]
    for arm in ("native","nd"):
        for rank in (0,1):
            for m in MS:
                folder=root/"routes"/arm
                meta_path=folder/f"rank{rank}_m{m}_route.json"
                csv_path=folder/f"rank{rank}_m{m}_kernel_details.csv"
                if not meta_path.is_file() or not csv_path.is_file():
                    raise ValueError("missing independent raw kernel route proof")
                proof=json.loads(meta_path.read_text(encoding="utf-8"))
                expected={"schema":"A1_MM_SHADOW_ROUTE_V1","arm":arm,
                          "rank":rank,"M":m,"shadow_sha256":sha,
                          "bit_exact":True}
                if any(proof.get(k)!=v for k,v in expected.items()):
                    raise ValueError("native/shadow profiler route identity mismatch")
                import hashlib
                durations=proof.get("device_kernel_durations_us",[])
                if (proof.get("kernel_hits",0)<3 or
                    not isinstance(durations,list) or
                    len(durations)!=proof["kernel_hits"] or
                    any(type(x) not in (float,int) or not math.isfinite(x) or x<=0
                        for x in durations) or
                    type(proof.get("device_kernel_p50_us")) not in (float,int) or
                    abs(statistics.median(durations)-proof["device_kernel_p50_us"])>0.01 or
                    proof.get("csv_sha256")!=hashlib.sha256(csv_path.read_bytes()).hexdigest()):
                    raise ValueError("not enough native/shadow real durations or CSV hash mismatch")
                route_proofs.append(proof)
    all_sessions=BASE+MEMORY
    stages=sorted(x.name for x in (root/"sessions").iterdir())
    expected=[f"{i+1:02d}_{arm}" for i,arm in enumerate(all_sessions)]
    if stages!=sorted(expected):
        raise ValueError("not exact ABBA8+memoryABBA4 run; never cherry-pick samples")
    values={m:[] for m in MS}
    for i,arm in enumerate(all_sessions):
        path=root/"sessions"/f"{i+1:02d}_{arm}"
        stats=read_session(path,arm,sha)
        for m in MS:
            values[m].append({"stage":i+1,"arm":arm,**stats[str(m)]})
    output=[]
    valid=True
    for m in MS:
        rows=values[m]
        by={arm:[x["paired_p50_us"] for x in rows if x["arm"]==arm]
            for arm in set(all_sessions)}
        if any(len(by[arm])!=n for arm,n in (
                ("native",4),("nd",4),("sym",2),("sym_clean",2))):
            raise ValueError("unexpected ABBA arm distribution")
        drift={arm:round(max(v)/min(v)-1,6) for arm,v in by.items()}
        stable=all(d<=MAX_STOCK_DRIFT for d in drift.values())
        valid=valid and stable
        median={arm:round(statistics.median(v),3) for arm,v in by.items()}
        native,shadow=median["native"],median["nd"]
        device_kernel={}
        for arm in ("native","nd"):
            durations=[q["device_kernel_p50_us"] for q in route_proofs
                       if q["arm"]==arm and q["M"]==m]
            if len(durations)!=2:
                raise ValueError("missing one Rank device kernel duration")
            device_kernel[arm]=round(max(durations),4) # 双Rank关键耗时保守选取
        output.append({"M":m,"native_nz_mm_p50_us":native,
            "short_profiler_kernel_rankmax_p50_us":device_kernel,
            "short_profiler_shadow_minus_native_us":round(
                device_kernel["nd"]-device_kernel["native"],4),
            "v2_shadow_nd_p50_us":shadow,
            "delta_shadow_minus_native_us":round(shadow-native,3),
            "shadow_slowdown_pct":round((shadow/native-1)*100,3),
            "sym_vs_nd_us":round(median["sym"]-shadow,3),
            "clean_increment_vs_sym_us":round(median["sym_clean"]-median["sym"],3),
            "arm_p50_us":median,"session_drift":drift,
            "all_armsgate_pass":stable,"raw_sessions":rows})
    return {"schema":"A1_MM_SHADOW_ASSESS_V1",
        "status":"MM_ONLY_COMPARISON_VALID" if valid else "INCONCLUSIVE_DRIFT",
        "comparisons":output,
        "native_stock_route_verified":True,
        "native_and_shadow_route_proofs":route_proofs,
        "real_full_d4_internal_mm_time":"UNKNOWN",
        "formal_D4_phase_causality_proven":False,
        "limits":[
            "No MemFabric signal/wait, IB and Add in MM shadow; these are not formal D4 stage timings.",
            "Symmetric-GM pool lifetime and first-write sync are not inside Event samples.",
            "Only actual native aclnnMatmulWeightNz profiler and bit-exact proof qualify native baseline.",
            "Across isolated sessions, per-index two-rank max is a conservative logical pair after Gloo barrier.",
            "NPU Event includes stream launch scheduling; independent CANN kernel duration is a second diagnostic metric, not substituted into ABBA.",
            "Differences of median memory modes are controlled perturbations, not exact causal components.",
        ]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if not a.root.is_absolute() or not a.out.is_absolute() or a.out.exists():
        p.error("absolute root / fresh absolute output only")
    result=evaluate(a.root)
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write("\n")
    print(result["status"])


if __name__=="__main__":
    main()
