#!/usr/bin/env python3
"""Gate A/B raw-trace audit pass 3 (final): steps by anchor spacing,
same-stream lookback for contiguous/memcpy attribution."""

import csv
import glob
import json
import statistics as st
from collections import defaultdict

BASE = "/tmp/opencode/v2_prefill_captures"
OUT_DIR = "/tmp/opencode/gate_ab_audit"
FUSED_PREFIX = "_ZN12_GLOBAL__N_113V2TrialKernel"
TARGET_MM = "aclnnMatmulWeightNz_MatMulV3Common_MatMulV3"
CONTIG_PREFIX = "aclnnContiguous"
ANCHOR_BOUNDARY_MS = 80.0
LOOKBACK_US = 5000.0


def fnum(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def load_kernels(arm, rank):
    d = glob.glob(f"{BASE}/{arm}_prefill_v2attrib/dp0*{rank}*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
    rows = []
    with open(d, newline="") as f:
        for r in csv.DictReader(f):
            start = fnum(r["Start Time(us)"])
            dur = fnum(r["Duration(us)"])
            if start is None or dur is None:
                continue
            rows.append(
                {
                    "name": r["Name"],
                    "core": r["Accelerator Core"],
                    "stream": r["Stream ID"],
                    "start": start,
                    "dur": dur,
                    "wait": fnum(r.get("Wait Time(us)")) or 0.0,
                    "shapes": r.get("Input Shapes", ""),
                }
            )
    rows.sort(key=lambda x: x["start"])
    return rows


def load_memcpy(arm, rank):
    d = glob.glob(f"{BASE}/{arm}_prefill_v2attrib/dp0*{rank}*_ascend_pt/ASCEND_PROFILER_OUTPUT/task_time.csv")[0]
    rows = []
    with open(d, newline="") as f:
        for r in csv.DictReader(f):
            if r["kernel_type"] != "MEMCPY_ASYNC":
                continue
            s = fnum(r["task_start(us)"])
            if s is None:
                continue
            rows.append({"start": s, "dur": fnum(r["task_time(us)"]) or 0.0, "stream": r["stream_id"]})
    rows.sort(key=lambda x: x["start"])
    return rows


def q(v, p):
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, int(len(s) * p))], 1)


def audit(arm):
    out = {}
    for rank in ("rank0", "rank1"):
        rows = load_kernels(arm, rank)
        mc = load_memcpy(arm, rank)
        is_site = (lambda k: k["name"].startswith(FUSED_PREFIX)) if arm == "v2" else (lambda k: k["name"] == TARGET_MM)
        site_kernels = [k for k in rows if is_site(k)]
        # assign steps by anchor spacing > boundary
        steps, step_id = [], 0
        prev_t = None
        for k in site_kernels:
            if prev_t is not None and (k["start"] - prev_t) / 1e3 > ANCHOR_BOUNDARY_MS:
                step_id += 1
            steps.append(step_id)
            prev_t = k["start"]
        r = {"n_sites": len(site_kernels), "n_steps": step_id + 1, "sites": []}
        stream_sorted = defaultdict(list)
        for k in rows:
            stream_sorted[k["stream"]].append(k)
        mc_by_stream = defaultdict(list)
        for m in mc:
            mc_by_stream[m["stream"]].append(m)

        for idx, (k, sidx) in enumerate(zip(site_kernels, steps)):
            # look back on same stream for nearest contiguous + memcpy
            sl = stream_sorted[k["stream"]]
            contig = None
            for j in range(len(sl) - 1, -1, -1):
                p = sl[j]
                if p["start"] >= k["start"]:
                    continue
                if k["start"] - p["start"] > LOOKBACK_US:
                    break
                if p["name"].startswith(CONTIG_PREFIX):
                    contig = p
                    break
            mcp = None
            ml = mc_by_stream[k["stream"]]
            for j in range(len(ml) - 1, -1, -1):
                p = ml[j]
                if p["start"] >= k["start"]:
                    continue
                if k["start"] - p["start"] > LOOKBACK_US:
                    break
                mcp = p
                break
            entry = {
                "site": idx,
                "step": sidx,
                "t_ms": round(k["start"] / 1e6, 3),
                "dur_us": round(k["dur"], 1),
                "wait_us": round(k["wait"], 1),
                "stream": k["stream"],
            }
            if arm == "stock":
                ar = next(
                    (
                        x
                        for x in rows
                        if x["name"].startswith("hcom_") and 0 <= x["start"] - (k["start"] + k["dur"]) < 5000
                    ),
                    None,
                )
                entry["ar_dur_us"] = round(ar["dur"], 1) if ar else None
                entry["ar_gap_us"] = round(ar["start"] - (k["start"] + k["dur"]), 1) if ar else None
                entry["ar_name"] = ar["name"] if ar else None
            if contig:
                entry["contig_name"] = contig["name"][:60]
                entry["contig_dur_us"] = round(contig["dur"], 1)
                entry["contig_lead_us"] = round(k["start"] - contig["start"], 1)
            if mcp and mcp["dur"] > 1:
                entry["mc_dur_us"] = round(mcp["dur"], 1)
                entry["mc_lead_us"] = round(k["start"] - mcp["start"], 1)
            r["sites"].append(entry)

        # per-step sums/periods
        per_step = defaultdict(float)
        anchors = defaultdict(list)
        for e in r["sites"]:
            d = e["dur_us"] + (e.get("ar_dur_us") or 0 if arm == "stock" else 0)
            per_step[e["step"]] += d
            anchors[e["step"]].append(e["t_ms"])
        keys = sorted(anchors)
        r["step_site_sums_ms"] = {str(k): round(per_step[k], 3) for k in keys}
        r["step_site_counts"] = {str(k): len(anchors[k]) for k in keys}
        r["step_periods_ms"] = [round(anchors[keys[i + 1]][0] - anchors[keys[i]][0], 3) for i in range(len(keys) - 1)]
        anom_steps = {e["step"] for e in r["sites"] if e["dur_us"] >= 10_000}
        stable = [k for k in keys if k not in anom_steps]
        r["anomaly_steps"] = sorted(anom_steps)
        r["stable_steps"] = stable
        r["stable_site_sum_mean_ms"] = round(st.mean(per_step[k] for k in stable), 3) if stable else None
        stable_periods = [
            r["step_periods_ms"][i] for i in range(len(keys) - 1) if keys[i] in stable and keys[i + 1] in stable
        ]
        r["stable_period_mean_ms"] = round(st.mean(stable_periods), 3) if stable_periods else None

        # contig/memcpy adjacency stats
        cg = [e for e in r["sites"] if "contig_dur_us" in e]
        r["contig_adjacent"] = {
            "n": len(cg),
            "of": len(r["sites"]),
            "dur_p50_us": q([e["contig_dur_us"] for e in cg], 0.5),
            "dur_sum_ms": round(sum(e["contig_dur_us"] for e in cg) / 1e3, 3),
            "lead_p50_us": q([e["contig_lead_us"] for e in cg], 0.5),
        }
        mg = [e for e in r["sites"] if "mc_dur_us" in e]
        r["memcpy_adjacent"] = {
            "n": len(mg),
            "of": len(r["sites"]),
            "dur_p50_us": q([e["mc_dur_us"] for e in mg], 0.5),
            "dur_sum_ms": round(sum(e["mc_dur_us"] for e in mg) / 1e3, 3),
            "lead_p50_us": q([e["mc_lead_us"] for e in mg], 0.5),
        }
        out[rank] = r
    return out


def main():
    res = {}
    for arm in ("stock", "v2"):
        res[arm] = audit(arm)
    with open(f"{OUT_DIR}/audit_pass3.json", "w") as f:
        json.dump(res, f, indent=1)
    for arm in ("stock", "v2"):
        print(f"===== {arm} =====")
        for rank in ("rank0", "rank1"):
            r = res[arm][rank]
            print(f" {rank}: sites={r['n_sites']} steps={r['n_steps']}")
            print(f"   sums_ms={r['step_site_sums_ms']}")
            print(f"   counts={r['step_site_counts']}")
            print(f"   periods={r['step_periods_ms']}  stable_mean_period={r['stable_period_mean_ms']}")
            print(f"   anomaly_steps={r['anomaly_steps']} stable={r['stable_steps']}")
            print(f"   stable_sum_mean={r['stable_site_sum_mean_ms']}")
            print(f"   contig_adj={r['contig_adjacent']}")
            print(f"   memcpy_adj={r['memcpy_adjacent']}")
        print()


if __name__ == "__main__":
    main()
