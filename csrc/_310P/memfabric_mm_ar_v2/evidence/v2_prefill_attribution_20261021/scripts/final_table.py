#!/usr/bin/env python3
"""Final attribution table: per-arm per-rank per-step site economics + step periods.

Outputs /tmp/opencode/final_attribution.json consumed by the report.
"""
import csv
import json
import statistics

CAP = "/tmp/opencode/v2_prefill_captures"
ARMS = {
    ("stock", 0): f"{CAP}/stock_prefill_v2attrib/dp0_pp0_tp0_dcp0_ep0_rank0_463132_20261021142115014_ascend_pt",
    ("stock", 1): f"{CAP}/stock_prefill_v2attrib/dp0_pp0_tp1_dcp0_ep1_rank1_463204_20261021142115014_ascend_pt",
    ("v2", 0): f"{CAP}/v2_prefill_v2attrib/dp0_pp0_tp0_dcp0_ep0_rank0_477537_20261021143336925_ascend_pt",
    ("v2", 1): f"{CAP}/v2_prefill_v2attrib/dp0_pp0_tp1_dcp0_ep1_rank1_477749_20261021143336925_ascend_pt",
}
M = 4096

def load(d):
    rows = [r for r in csv.DictReader(open(d + "/ASCEND_PROFILER_OUTPUT/kernel_details.csv", newline="")) if r["Name"]]
    for r in rows:
        r["start"] = float(r["Start Time(us)"].strip())
        r["dur"] = float(r["Duration(us)"].strip())
        r["wait"] = float(r.get("Wait Time(us)", "0") or 0)
    rows = [r for r in rows if "HybmSdmaOrchEpoch" not in r["Name"]]
    rows.sort(key=lambda r: r["start"])
    return rows

out = {}
for (arm, rank), d in ARMS.items():
    rows = load(d)
    if arm == "stock":
        anchors = [r for r in rows if "MatmulWeightNz" in r["Name"]
                   and r["Output Shapes"].startswith(f'"{M},2048"')
                   and r["Input Shapes"].startswith(f'"{M},2048;')]
    else:
        anchors = [r for r in rows if "V2TrialKernel" in r["Name"]]
    n_steps = len(anchors) // 40
    res = {"n_sites": len(anchors), "n_steps": n_steps}
    # step periods: first-anchor to first-anchor
    periods = [(anchors[(i + 1) * 40]["start"] - anchors[i * 40]["start"]) / 1000
               for i in range(n_steps - 1)]
    res["step_periods_ms"] = [round(p, 1) for p in periods]
    # site costs, excluding outlier kernels and using steady steps (all but step0 for v2)
    if arm == "stock":
        mm = [a["dur"] for a in anchors]
        ars = []
        for i, a in enumerate(anchors):
            ar = next((x for x in rows[i + 1:i + 5] if x["Name"].startswith("hcom_")), None) if False else None
        # proper: use index in rows
        idx = {id(a): i for i, a in enumerate(rows)}
        for a in anchors:
            i = idx[id(a)]
            ar = next((x for x in rows[i + 1:i + 5] if x["Name"].startswith("hcom_")), None)
            if ar:
                ars.append(ar["dur"])
        res["mm_p50_us"] = round(statistics.median(mm))
        res["mm_sum_per_step_us"] = round(sum(mm) / n_steps)
        res["ar_p50_us"] = round(statistics.median(ars))
        res["ar_sum_per_step_us"] = round(sum(ars) / n_steps)
        res["site_sum_per_step_us"] = res["mm_sum_per_step_us"] + res["ar_sum_per_step_us"]
    else:
        steady = [a for a in anchors if a["dur"] < 10000]
        outliers = [a for a in anchors if a["dur"] >= 10000]
        res["v2_p50_us"] = round(statistics.median([a["dur"] for a in steady]))
        res["v2_p90_us"] = round(sorted(a["dur"] for a in steady)[int(len(steady) * 0.9)])
        res["v2_sum_per_step_us"] = round(sum(a["dur"] for a in steady) / n_steps)
        res["outliers"] = [{"dur_us": round(o["dur"]), "step": 1 + anchors.index(o) // 40}
                           for o in outliers]
    # non-target AR per step
    t_ar = set()
    if arm == "stock":
        idx = {id(a): i for i, a in enumerate(rows)}
        for a in anchors:
            i = idx[id(a)]
            ar = next((x for x in rows[i + 1:i + 5] if x["Name"].startswith("hcom_")), None)
            if ar:
                t_ar.add(ar["start"])
    all_ar = [r for r in rows if r["Name"].startswith("hcom_")]
    nt = [r for r in all_ar if r["start"] not in t_ar]
    res["nontarget_ar_per_step"] = round(len(nt) / n_steps, 1)
    res["nontarget_ar_sum_per_step_us"] = round(sum(r["dur"] for r in nt) / n_steps)
    out[f"{arm}_r{rank}"] = res

# top kernels by total time per arm r0 (context table)
for arm in ("stock", "v2"):
    rows = load(ARMS[(arm, 0)])
    agg = {}
    for r in rows:
        agg.setdefault(r["Name"][:52], []).append(r["dur"])
    top = sorted(((sum(v), k, len(v)) for k, v in agg.items()), reverse=True)[:12]
    out[f"{arm}_r0_topkernels"] = [
        {"sum_per_step_us": round(s / out[f"{arm}_r0"]["n_steps"]), "n_per_step": round(n / out[f"{arm}_r0"]["n_steps"], 1), "name": k}
        for s, k, n in top]

json.dump(out, open("/tmp/opencode/final_attribution.json", "w"), indent=1)
for k, v in out.items():
    if not k.endswith("topkernels"):
        print(k, json.dumps(v))
