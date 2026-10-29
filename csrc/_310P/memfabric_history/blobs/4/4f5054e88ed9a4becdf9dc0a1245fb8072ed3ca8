#!/usr/bin/env python3
"""Stock vs V2 prefill comparative attribution from ascend_pt kernel_details.

Splits steps by target-site groups (40 sites/step), computes per-step wall,
target-site costs (Stock: MM+AR+gaps; V2: fused kernel+gap), non-target AR,
stream busy/idle, and cross-rank fused-kernel skew.

Usage: compare_prefill.py <stock_ascend_pt_dir> <v2_ascend_pt_dir> [sites_per_step]
"""
import csv
import json
import statistics
import sys
from collections import defaultdict

STOCK_DIR, V2_DIR = sys.argv[1], sys.argv[2]
SITES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
M = 4096

def load(d):
    rows = [r for r in csv.DictReader(open(d + "/ASCEND_PROFILER_OUTPUT/kernel_details.csv", newline="")) if r["Name"]]
    for r in rows:
        r["start"] = float(r["Start Time(us)"].strip())
        r["dur"] = float(r["Duration(us)"].strip())
        r["wait"] = float(r.get("Wait Time(us)", "0") or 0)
        r["stream"] = r["Stream ID"]
    rows.sort(key=lambda r: r["start"])
    return rows

def is_target_mm(r):
    return ("MatmulWeightNz" in r["name"] if "name" in r else "MatmulWeightNz" in r["Name"]) \
        and r["Output Shapes"].startswith(f'"{M},2048"') and r["Input Shapes"].startswith(f'"{M},2048')

def is_hccl(r):
    return r["Name"].startswith("hcom_")

def is_fused(r):
    return "V2TrialKernel" in r["Name"]

def split_steps(rows, sites_idx):
    """sites_idx: indices of target-site anchor kernels; split every SITES."""
    steps = []
    for i in range(0, len(sites_idx), SITES):
        chunk = sites_idx[i:i + SITES]
        a, b = chunk[0], chunk[-1]
        # step = from just after previous step's last anchor to this step's last anchor + margin
        lo = 0 if i == 0 else sites_idx[i - 1] + 1
        hi = b + 12  # include a few kernels after last site (AR/norm tail)
        steps.append((lo, hi, chunk))
    return steps

def analyze(d, arm):
    rows = load(d)
    for r in rows:
        r["name"] = r["Name"]
    anchors = []
    for i, r in enumerate(rows):
        if arm == "stock" and is_target_mm(r):
            anchors.append(i)
        elif arm == "v2" and is_fused(r):
            anchors.append(i)
    steps = split_steps(rows, anchors)
    out = {"arm": arm, "dir": d, "steps": []}
    for lo, hi, chunk in steps:
        seg = rows[lo:hi]
        t0, t1 = seg[0]["start"], max(r["start"] + r["dur"] for r in seg)
        st = {"wall_us": round(t1 - t0, 1), "kernels": len(seg), "sites": []}
        # per-site extraction
        for ai in chunk:
            r = rows[ai]
            if arm == "stock":
                ar = next((x for x in rows[ai + 1:ai + 5] if is_hccl(x)), None)
                site = {"mm_dur": r["dur"], "mm_wait": r["wait"],
                        "mm_stream": r["stream"], "t": r["start"]}
                if ar:
                    site["ar_dur"] = ar["dur"]
                    site["ar_wait"] = ar["wait"]
                    site["gap_mm_ar"] = ar["start"] - (r["start"] + r["dur"])
                    ar_end = ar["start"] + ar["dur"]
                    nxt = next((x for x in rows if x["start"] > ar_end), None)
                    site["gap_ar_next"] = nxt["start"] - ar_end if nxt else None
                st["sites"].append(site)
            else:
                end = r["start"] + r["dur"]
                nxt = next((x for x in rows if x["start"] > end), None)
                st["sites"].append({"v2_dur": r["dur"], "v2_wait": r["wait"],
                                    "stream": r["stream"], "t": r["start"],
                                    "gap_next": (nxt["start"] - end) if nxt else None})
        # non-target ARs in segment
        site_ts = set(s["t"] for s in st["sites"])
        nt_ar = [r for r in seg if is_hccl(r)]
        st["nontarget_ar"] = {"n": len(nt_ar), "sum": round(sum(r["dur"] for r in nt_ar), 1),
                              "p50": round(statistics.median([r["dur"] for r in nt_ar]), 1) if nt_ar else 0}
        # stream busy/idle
        byst = defaultdict(list)
        for r in seg:
            if r["stream"] != "N/A":
                byst[int(r["stream"])].append((r["start"], r["start"] + r["dur"]))
        si = {}
        for s, ivs in byst.items():
            ivs.sort()
            merged = []
            for b, e in ivs:
                if merged and b <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([b, e])
            busy = sum(e - b for b, e in merged)
            gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
            si[s] = {"n": len(ivs), "busy": round(busy, 1), "idle": round((t1 - t0) - busy, 1),
                     "gaps_sum": round(sum(gaps), 1), "gaps_top3": [round(g, 1) for g in sorted(gaps, reverse=True)[:3]]}
        st["streams"] = dict(sorted(si.items(), key=lambda x: -x[1]["busy"])[:5])
        out["steps"].append(st)
    return out, rows

def summarize(res, arm):
    steps = res["steps"]
    walls = [s["wall_us"] for s in steps]
    print(f"\n===== {arm} ({len(steps)} steps) =====")
    print(f"step wall us: {[round(w) for w in walls]}  mean={statistics.mean(walls):.0f}")
    if arm == "stock":
        mm = [s["mm_dur"] for st in steps for s in st["sites"]]
        ar = [s["ar_dur"] for st in steps for s in st["sites"] if "ar_dur" in s]
        g1 = [s["gap_mm_ar"] for st in steps for s in st["sites"] if "gap_mm_ar" in s]
        g2 = [s["gap_ar_next"] for st in steps for s in st["sites"] if s.get("gap_ar_next") is not None]
        mw = [s["mm_wait"] for st in steps for s in st["sites"]]
        aw = [s["ar_wait"] for st in steps for s in st["sites"] if "ar_wait" in s]
        print(f"target MM : n={len(mm)} sum={sum(mm):.0f} p50={statistics.median(mm):.0f} max={max(mm):.0f} wait_p50={statistics.median(mw):.1f}")
        print(f"target AR : n={len(ar)} sum={sum(ar):.0f} p50={statistics.median(ar):.0f} max={max(ar):.0f} wait_p50={statistics.median(aw):.1f}")
        print(f"gaps: mm->ar p50={statistics.median(g1):.1f}us | ar->next p50={statistics.median(g2):.1f}us max={max(g2):.0f}")
        per_step = (sum(mm) + sum(ar)) / len(steps)
        print(f"target site cost/step: {(sum(mm)+sum(ar))/len(steps):.0f}us  (MM {sum(mm)/len(steps):.0f} + AR {sum(ar)/len(steps):.0f})")
    else:
        v2 = [s["v2_dur"] for st in steps for s in st["sites"]]
        gn = [s["gap_next"] for st in steps for s in st["sites"] if s.get("gap_next") is not None]
        vw = [s["v2_wait"] for st in steps for s in st["sites"]]
        print(f"V2 kernel: n={len(v2)} sum={sum(v2):.0f} p50={statistics.median(v2):.0f} p90={sorted(v2)[int(len(v2)*0.9)]:.0f} max={max(v2):.0f} wait_p50={statistics.median(vw):.1f}")
        print(f"gap v2->next: p50={statistics.median(gn):.1f}us max={max(gn):.0f}")
        print(f"V2 cost/step: {sum(v2)/len(steps):.0f}us")
        top = sorted(v2, reverse=True)[:5]
        print(f"V2 top-5 durs: {[round(x) for x in top]}")
    nt = [st["nontarget_ar"] for st in steps]
    print(f"non-target AR/step: n={[x['n'] for x in nt]} sum/step={[x['sum'] for x in nt]}")
    print("streams (step0):", json.dumps(steps[0]["streams"]))
    return res

sres, srows = analyze(STOCK_DIR, "stock")
vres, vrows = analyze(V2_DIR, "v2")
summarize(sres, "stock")
summarize(vres, "v2")

# cross-rank skew for V2 fused kernels: need rank1 v2 dir
if len(sys.argv) > 4:
    _, vrows1 = analyze(sys.argv[4], "v2")
    v0 = [(r["start"], r["dur"]) for r in vrows if is_fused(r)]
    v1 = [(r["start"], r["dur"]) for r in vrows1 if is_fused(r)]
    if v0 and v1:
        n = min(len(v0), len(v1))
        skews = []
        for i in range(n):
            d = v0[i][0] - v1[i][0]
            skews.append(d)
        print(f"\ncross-rank V2 kernel start skew (rank0-rank1): p50={statistics.median(skews):.0f}us max|{max(map(abs,skews)):.0f}|us")

json.dump({"stock": sres, "v2": vres}, open("/tmp/opencode/prefill_compare_rank0.json", "w"), indent=1, default=str)
print("\nwrote /tmp/opencode/prefill_compare_rank0.json")
