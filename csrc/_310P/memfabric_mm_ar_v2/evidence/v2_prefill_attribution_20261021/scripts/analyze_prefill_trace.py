#!/usr/bin/env python3
"""Prefill trace attribution analysis for Stock/V2 arm ascend_pt exports.

Parses ASCEND_PROFILER_OUTPUT/kernel_details.csv (proper quoted-CSV parsing),
clusters device kernels into prefill steps by time gaps, and for each step
extracts: total wall, per-kernel-name stats, the 40 target o_proj/out_proj
sites (Stock: big-M MatMulV2 + following hcom_allReduce; V2: the fused
kernel), site-local gaps, and device idle on the main compute stream.

Usage: analyze_prefill_trace.py <ascend_pt_dir> <out_json> [target_m]
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

TRACE_DIR = sys.argv[1]
OUT_JSON = sys.argv[2]
TARGET_M = int(sys.argv[3]) if len(sys.argv) > 3 else 4095

KO = os.path.join(TRACE_DIR, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
rows = []
with open(KO, newline="") as f:
    for r in csv.DictReader(f):
        try:
            rows.append({
                "name": r["Name"],
                "stream": int(r["Stream ID"]),
                "start": float(r["Start Time(us)"].strip()),
                "dur": float(r["Duration(us)"].strip()),
                "wait": float(r.get("Wait Time(us)", "0") or 0),
                "shapes": r.get("Input Shapes", ""),
            })
        except (ValueError, KeyError):
            continue
rows.sort(key=lambda x: x["start"])
print(f"kernels: {len(rows)}  t0={rows[0]['start']:.0f} t1={rows[-1]['start']+rows[-1]['dur']:.0f}")

# ---- cluster into steps by >20ms gaps ----
STEP_GAP_US = 20_000
steps = []
cur = [rows[0]]
for k in rows[1:]:
    if k["start"] - (cur[-1]["start"] + cur[-1]["dur"]) > STEP_GAP_US:
        steps.append(cur)
        cur = [k]
    else:
        cur.append(k)
steps.append(cur)
# drop micro-steps (sampling-only leftovers < 50 kernels)
steps = [s for s in steps if len(s) > 50]
print(f"steps: {len(steps)} sizes={[len(s) for s in steps]}")

def stats(vals):
    if not vals:
        return None
    sv = sorted(vals)
    return {"n": len(vals), "sum": round(sum(vals), 1),
            "min": round(sv[0], 1), "p50": round(sv[len(sv)//2], 1),
            "max": round(sv[-1], 1)}

result = {"trace_dir": TRACE_DIR, "target_m": TARGET_M, "steps": []}
for si, s in enumerate(steps):
    t_start = s[0]["start"]
    t_end = max(k["start"] + k["dur"] for k in s)
    # per-name stats
    byname = defaultdict(list)
    for k in s:
        byname[k["name"]].append(k["dur"])
    name_stats = {n: stats(v) for n, v in byname.items() if sum(v) > 1000 or len(v) >= 40}
    # per-stream busy/idle within step (top streams)
    bystream = defaultdict(list)
    for k in s:
        bystream[k["stream"]].append((k["start"], k["start"] + k["dur"]))
    stream_info = {}
    for st, ivs in sorted(bystream.items(), key=lambda x: -sum(i[1]-i[0] for i in x[1]))[:6]:
        ivs.sort()
        busy = sum(e - b for b, e in ivs)
        # merge overlap
        merged = []
        for b, e in ivs:
            if merged and b <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([b, e])
        gaps = [merged[i+1][0] - merged[i][1] for i in range(len(merged)-1)]
        stream_info[st] = {"kernels": len(ivs), "busy_us": round(busy, 1),
                           "gaps_n": len(gaps),
                           "gaps_sum": round(sum(gaps), 1),
                           "gaps_top5": [round(g, 1) for g in sorted(gaps, reverse=True)[:5]]}

    # ---- target sites ----
    # Stock: unquantized row-parallel MM = *MatmulWeightNz*MatMulV2 with M in shapes,
    #        followed by hcom_allReduce. V2: fused kernel name TBD (catch by name pattern).
    is_stock_mm = lambda k: "MatmulWeightNz" in k["name"] and str(TARGET_M) in k["shapes"]
    is_hccl = lambda k: k["name"].startswith("hcom_")
    is_fused = lambda k: ("mf310p" in k["name"].lower() or "memfabric" in k["name"].lower()
                          or "V2Trial" in k["name"] or "MmArV2" in k["name"])
    sites = []
    for i, k in enumerate(s):
        if is_fused(k):
            nxt = s[i+1] if i + 1 < len(s) else None
            sites.append({"type": "v2", "t": k["start"], "dur": k["dur"],
                          "wait": k["wait"], "name": k["name"], "stream": k["stream"],
                          "gap_after": round((nxt["start"] - (k["start"] + k["dur"])), 1) if nxt else None})
        elif is_stock_mm(k):
            # find next kernel (any stream) after MM end within 5ms and the AR after it
            mm_end = k["start"] + k["dur"]
            nxt = next((x for x in s[i+1:] if x["start"] > mm_end), None)
            ar = next((x for x in s[i+1:i+6] if is_hccl(x)), None)
            sites.append({"type": "stock_mm", "t": k["start"], "dur": k["dur"],
                          "wait": k["wait"], "name": k["name"], "stream": k["stream"],
                          "shapes": k["shapes"][:60],
                          "gap_to_next": round(nxt["start"] - mm_end, 1) if nxt else None,
                          "ar": ({"t": ar["start"], "dur": ar["dur"], "wait": ar["wait"],
                                  "stream": ar["stream"],
                                  "gap_mm_to_ar": round(ar["start"] - mm_end, 1)} if ar else None)})
    # stock sites: attach gap AR->next kernel
    for i, site in enumerate(sites):
        if site["type"] == "stock_mm" and site.get("ar"):
            ar_end = site["ar"]["t"] + site["ar"]["dur"]
            nxt = next((x for x in s if x["start"] > ar_end), None)
            site["ar"]["gap_ar_to_next"] = round(nxt["start"] - ar_end, 1) if nxt else None

    result["steps"].append({
        "step": si, "wall_us": round(t_end - t_start, 1),
        "kernels": len(s),
        "streams": stream_info,
        "name_stats": dict(sorted(name_stats.items(), key=lambda x: -x[1]["sum"])[:25]),
        "sites": sites,
        "n_sites": len(sites),
        "stock_mm_durs": [x["dur"] for x in sites if x["type"] == "stock_mm"],
        "ar_durs": [x["ar"]["dur"] for x in sites if x["type"] == "stock_mm" and x.get("ar")],
        "v2_durs": [x["dur"] for x in sites if x["type"] == "v2"],
    })
    print(f"step {si}: wall={t_end-t_start:.0f}us kernels={len(s)} sites={len(sites)}")

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=1)
print("wrote", OUT_JSON)
