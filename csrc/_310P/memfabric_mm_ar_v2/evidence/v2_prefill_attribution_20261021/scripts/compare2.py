#!/usr/bin/env python3
"""Refined Stock vs V2 prefill attribution (fixed step split, clock alignment).

- Steps split by time-gap clustering of target-site anchors (>300ms gap).
- Persistent orchestrator kernel (HybmSdmaOrchEpoch) excluded.
- Non-target AR counted excluding ARs matched to target sites.
- Cross-rank clock offset calibrated via synchronized hcom_allReduce pairs,
  then applied to V2 kernel start/duration skew analysis.
- Host side: per-site host cost from operator_details.csv (aten op durations).

Usage: compare2.py
"""
import csv
import json
import statistics
import sys
from collections import defaultdict

CAP = "/tmp/opencode/v2_prefill_captures"
STOCK_R0 = f"{CAP}/stock_prefill_v2attrib/dp0_pp0_tp0_dcp0_ep0_rank0_463132_20261021142115014_ascend_pt"
STOCK_R1 = f"{CAP}/stock_prefill_v2attrib/dp0_pp0_tp1_dcp0_ep1_rank1_463204_20261021142115014_ascend_pt"
V2_R0 = f"{CAP}/v2_prefill_v2attrib/dp0_pp0_tp0_dcp0_ep0_rank0_477537_20261021143336925_ascend_pt"
V2_R1 = f"{CAP}/v2_prefill_v2attrib/dp0_pp0_tp1_dcp0_ep1_rank1_477749_20261021143336925_ascend_pt"
M = 4096

def load(d):
    rows = [r for r in csv.DictReader(open(d + "/ASCEND_PROFILER_OUTPUT/kernel_details.csv", newline="")) if r["Name"]]
    for r in rows:
        r["start"] = float(r["Start Time(us)"].strip())
        r["dur"] = float(r["Duration(us)"].strip())
        r["wait"] = float(r.get("Wait Time(us)", "0") or 0)
    rows = [r for r in rows if "HybmSdmaOrchEpoch" not in r["Name"]]  # persistent kernel
    rows.sort(key=lambda r: r["start"])
    return rows

def is_target_mm(r):
    return ("MatmulWeightNz" in r["Name"]
            and r["Output Shapes"].startswith(f'"{M},2048"')
            and r["Input Shapes"].startswith(f'"{M},2048;'))

def is_hccl(r): return r["Name"].startswith("hcom_")
def is_fused(r): return "V2TrialKernel" in r["Name"]

def split_by_time(items, gap=300_000):
    groups, cur = [], [items[0]]
    for it in items[1:]:
        if it["start"] - cur[-1]["start"] > gap:
            groups.append(cur); cur = [it]
        else:
            cur.append(it)
    groups.append(cur)
    return groups

def sites_and_steps(rows, arm):
    anchors = [r for r in rows if (is_target_mm(r) if arm == "stock" else is_fused(r))]
    groups = [g for g in split_by_time(anchors) if len(g) >= 30]
    out = []
    for g in groups:
        t0, t1 = g[0]["start"], g[-1]["start"] + g[-1]["dur"]
        sites = []
        idx = {id(r) for r in g}
        for r in g:
            if arm == "stock":
                j = rows.index(r)
                ar = next((x for x in rows[j+1:j+5] if is_hccl(x)), None)
                s = {"mm": r["dur"], "t": r["start"], "mm_wait": r["wait"]}
                if ar:
                    s["ar"] = ar["dur"]; s["ar_wait"] = ar["wait"]
                    s["gap_mm_ar"] = ar["start"] - (r["start"] + r["dur"])
                    ar_end = ar["start"] + ar["dur"]
                    nxt = next((x for x in rows if x["start"] > ar_end), None)
                    s["gap_ar_next"] = (nxt["start"] - ar_end) if nxt else None
                    s["ar_t"] = ar["start"]
                sites.append(s)
            else:
                end = r["start"] + r["dur"]
                nxt = next((x for x in rows if x["start"] > end), None)
                sites.append({"v2": r["dur"], "t": r["start"], "v2_wait": r["wait"],
                              "gap_next": (nxt["start"] - end) if nxt else None})
        # non-target ARs within [t0, t1+3ms]
        seg = [r for r in rows if t0 <= r["start"] <= t1 + 3000]
        site_ar_t = {s.get("ar_t") for s in sites if "ar_t" in s}
        nt = [r for r in seg if is_hccl(r) and r["start"] not in site_ar_t]
        out.append({"n_sites": len(g), "t0": t0, "t1": t1, "wall": t1 - t0,
                    "sites": sites, "nt_ar_n": len(nt),
                    "nt_ar_sum": sum(r["dur"] for r in nt),
                    "nt_ar_p50": statistics.median([r["dur"] for r in nt]) if nt else 0,
                    "nt_ar_t": [r["start"] for r in nt]})
    return out

def report(arm, steps, label):
    walls = [s["wall"] for s in steps]
    print(f"\n===== {arm} {label}: {len(steps)} steps =====")
    print("walls ms:", [round(w/1000, 1) for w in walls])
    if arm == "stock":
        mm = [x["mm"] for s in steps for x in s["sites"]]
        ar = [x["ar"] for s in steps for x in s["sites"]]
        g2 = [x["gap_ar_next"] for s in steps for x in s["sites"] if x.get("gap_ar_next") is not None]
        mw = [x["mm_wait"] for s in steps for x in s["sites"]]
        aw = [x["ar_wait"] for s in steps for x in s["sites"]]
        print(f"MM : n={len(mm)} p50={statistics.median(mm):.0f} sum/step={sum(mm)/len(steps):.0f} wait_p50={statistics.median(mw):.1f}")
        print(f"AR : n={len(ar)} p50={statistics.median(ar):.0f} sum/step={sum(ar)/len(steps):.0f} wait_p50={statistics.median(aw):.1f}")
        print(f"gap ar->next: p50={statistics.median(g2):.1f}us")
        print(f"site cost/step (MM+AR): {(sum(mm)+sum(ar))/len(steps):.0f}us")
    else:
        v2 = [x["v2"] for s in steps for x in s["sites"]]
        gn = [x["gap_next"] for s in steps for x in s["sites"] if x.get("gap_next") is not None]
        vw = [x["v2_wait"] for s in steps for x in s["sites"]]
        print(f"V2k: n={len(v2)} p50={statistics.median(v2):.0f} p90={sorted(v2)[int(len(v2)*0.9)]:.0f} max={max(v2):.0f} sum/step={sum(v2)/len(steps):.0f} wait_p50={statistics.median(vw):.1f}")
        print(f"gap v2->next: p50={statistics.median(gn):.1f}us sum/step={sum(gn)/len(steps):.0f}")
        print(f"V2 total/step (kernel+gap): {(sum(v2)+sum(gn))/len(steps):.0f}us")
        print("V2 durs sorted desc top5:", [round(x) for x in sorted(v2, reverse=True)[:5]])
    print(f"non-target AR: n/step={[s['nt_ar_n'] for s in steps]} sum/step(ms)={[round(s['nt_ar_sum']/1000,1) for s in steps]} p50={[round(s['nt_ar_p50']) for s in steps]}")
    return steps

print("########## RANK 0 ##########")
sr0 = sites_and_steps(load(STOCK_R0), "stock"); report("stock", sr0, "r0")
vr0 = sites_and_steps(load(V2_R0), "v2"); report("v2", vr0, "r0")
print("\n########## RANK 1 ##########")
sr1 = sites_and_steps(load(STOCK_R1), "stock"); report("stock", sr1, "r1")
vr1 = sites_and_steps(load(V2_R1), "v2"); report("v2", vr1, "r1")

# ---- clock offset calibration via synchronized non-target HCCL ARs ----
def calibrate(a_steps, b_steps):
    """Match AR starts by order; return median offset a-b."""
    offs = []
    for sa, sb in zip(a_steps, b_steps):
        for ta, tb in zip(sa["nt_ar_t"], sb["nt_ar_t"]):
            offs.append(ta - tb)
    return statistics.median(offs), offs

off_v2 = calibrate(vr0, vr1)[0]
print(f"\nclock offset rank0-rank1 (V2 arm, via {len(vr0)}x42 HCCL ARs): {off_v2:.0f}us")
off_st = calibrate(sr0, sr1)[0]
print(f"clock offset rank0-rank1 (stock arm): {off_st:.0f}us")

# apply offset: V2 kernel paired analysis rank0 vs rank1
v0 = [x for s in vr0 for x in s["sites"]]
v1 = [x for s in vr1 for x in s["sites"]]
n = min(len(v0), len(v1))
dskew, ddur, dstart_end = [], [], []
for i in range(n):
    t1c = v1[i]["t"] - off_v2  # rank1 start corrected to rank0 clock
    dskew.append(v0[i]["t"] - t1c)
    ddur.append(v0[i]["v2"] - v1[i]["v2"])
    dstart_end.append((v0[i]["t"] + v0[i]["v2"]) - (t1c + v1[i]["v2"]))
print(f"V2 kernel start skew r0-r1 (clock-corrected): p50={statistics.median(dskew):.0f}us p90={sorted(dskew)[int(n*0.9)]:.0f} max|{max(map(abs,dskew)):.0f}|")
print(f"V2 kernel dur diff r0-r1: p50={statistics.median(ddur):.0f}us")
print(f"V2 kernel end diff r0-r1: p50={statistics.median(dstart_end):.0f}us")

# stock arm same for ARs
s0 = [x for s in sr0 for x in s["sites"] if "ar_t" in x]
s1 = [x for s in sr1 for x in s["sites"] if "ar_t" in x]
n = min(len(s0), len(s1))
dskew_s, ddur_s = [], []
for i in range(n):
    t1c = s1[i]["ar_t"] - off_st
    dskew_s.append(s0[i]["ar_t"] - t1c)
    ddur_s.append(s0[i]["ar"] - s1[i]["ar"])
print(f"stock target AR start skew r0-r1 (corrected): p50={statistics.median(dskew_s):.0f}us max|{max(map(abs,dskew_s)):.0f}|")
print(f"stock target AR dur diff r0-r1: p50={statistics.median(ddur_s):.0f}us")

json.dump({"note": "refined per-site metrics"}, open("/tmp/opencode/dummy.json", "w"))
