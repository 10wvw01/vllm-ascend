#!/usr/bin/env python3
"""Gate A/B raw-trace offline audit (2026-10-22).

Reads the preserved dual-rank ascend_pt exports of the Stock/V2 prefill
attribution round directly (kernel_details.csv, task_time.csv,
api_statistic.csv, operator_details.csv) and independently recomputes:

  1. per-step target-site statistics (40 sites/step) and step periods;
  2. exact localization of the V2 74ms fused-kernel anomaly;
  3. cross-rank same-generation pairing via unique hcom kernel names,
     with the corrected clock transform (A-02) and drift check;
  4. copy attribution: aclnnContiguous device kernels / MEMCPY_ASYNC tasks
     adjacent to target sites in both arms, with net per-site cost;
  5. presence and span of HybmSdmaOrchEpoch (AICPU v11 orchestrator).

Output: one JSON per arm + one summary JSON. Read-only on raw traces.
"""

import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

BASE = "/tmp/opencode/v2_prefill_captures"
OUT_DIR = "/tmp/opencode/gate_ab_audit"
STEP_GAP_US = 20_000
MIN_STEP_KERNELS = 50
FUSED_PREFIX = "_ZN12_GLOBAL__N_113V2TrialKernel"
TARGET_MM = "aclnnMatmulWeightNz_MatMulV3Common_MatMulV3"
CONTIG_SLICE = "aclnnContiguous_SliceAiCore_Slice"
CONTIG_TRANSPOSE = "aclnnContiguous_TransposeAiCore_Transpose"
SITES_DIR = (
    "/home/w00804146/310P/vllm-ascend/csrc/_310P/memfabric_mm_ar_v2/evidence/v2_prefill_attribution_20261021/sites"
)


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
                    "type": r["Type"],
                    "core": r["Accelerator Core"],
                    "stream": r["Stream ID"],
                    "start": start,
                    "dur": dur,
                    "wait": fnum(r.get("Wait Time(us)")) or 0.0,
                    "shapes": r.get("Input Shapes", ""),
                    "aicore": fnum(r.get("aicore_time(us)")) or 0.0,
                    "mac": fnum(r.get("mac_time(us)")) or 0.0,
                    "mte2": fnum(r.get("mte2_time(us)")) or 0.0,
                    "mte3": fnum(r.get("mte3_time(us)")) or 0.0,
                    "scalar": fnum(r.get("scalar_time(us)")) or 0.0,
                    "vec": fnum(r.get("vec_time(us)")) or 0.0,
                }
            )
    rows.sort(key=lambda x: x["start"])
    return rows


def load_memcpy_tasks(arm, rank):
    d = glob.glob(f"{BASE}/{arm}_prefill_v2attrib/dp0*{rank}*_ascend_pt/ASCEND_PROFILER_OUTPUT/task_time.csv")[0]
    rows = []
    with open(d, newline="") as f:
        for r in csv.DictReader(f):
            if r["kernel_type"] != "MEMCPY_ASYNC":
                continue
            start = fnum(r["task_start(us)"])
            dur = fnum(r["task_time(us)"])
            if start is None:
                continue
            rows.append({"start": start, "dur": dur or 0.0, "stream": r["stream_id"], "task_id": r["task_id"]})
    rows.sort(key=lambda x: x["start"])
    return rows


def cluster_steps(rows):
    steps, cur = [], [rows[0]]
    for k in rows[1:]:
        if k["start"] - (cur[-1]["start"] + cur[-1]["dur"]) > STEP_GAP_US:
            steps.append(cur)
            cur = [k]
        else:
            cur.append(k)
    steps.append(cur)
    return [s for s in steps if len(s) > MIN_STEP_KERNELS]


def q(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(len(s) * p))], 1)


def audit_arm(arm):
    out = {"arm": arm, "ranks": {}}
    for rank in ("rank0", "rank1"):
        rows = load_kernels(arm, rank)
        steps = cluster_steps(rows)
        r = {
            "n_kernels": len(rows),
            "n_steps": len(steps),
            "step_sizes": [len(s) for s in steps],
            "sites": [],
            "hybm_orch": [],
            "anomalies": [],
            "copies": {},
        }

        for k in rows:
            if k["name"] == "HybmSdmaOrchEpoch":
                r["hybm_orch"].append({"start": k["start"], "dur": k["dur"], "core": k["core"]})

        for si, s in enumerate(steps):
            t0 = s[0]["start"]
            for i, k in enumerate(s):
                if arm == "v2" and k["name"].startswith(FUSED_PREFIX):
                    prev = s[i - 1] if i > 0 else None
                    nxt = s[i + 1] if i + 1 < len(s) else None
                    r["sites"].append(
                        {
                            "step": si,
                            "t_rel_ms": round((k["start"] - t0) / 1e3, 3),
                            "dur_us": k["dur"],
                            "wait_us": k["wait"],
                            "stream": k["stream"],
                            "prev_name": prev["name"][:60] if prev else None,
                            "prev_gap_us": round(k["start"] - (prev["start"] + prev["dur"]), 1) if prev else None,
                            "prev_dur_us": prev["dur"] if prev else None,
                            "prev_same_stream": bool(prev and prev["stream"] == k["stream"]),
                            "next_name": nxt["name"][:60] if nxt else None,
                            "mac": k["mac"],
                            "mte2": k["mte2"],
                            "mte3": k["mte3"],
                            "scalar": k["scalar"],
                            "vec": k["vec"],
                        }
                    )
                elif arm == "stock" and k["name"] == TARGET_MM:
                    nxts = s[i + 1 : i + 6]
                    ar = next((x for x in nxts if x["name"].startswith("hcom_")), None)
                    prev = s[i - 1] if i > 0 else None
                    r["sites"].append(
                        {
                            "step": si,
                            "t_rel_ms": round((k["start"] - t0) / 1e3, 3),
                            "dur_us": k["dur"],
                            "wait_us": k["wait"],
                            "stream": k["stream"],
                            "shapes": k["shapes"][:80],
                            "ar_dur_us": ar["dur"] if ar else None,
                            "ar_gap_us": round(ar["start"] - (k["start"] + k["dur"]), 1) if ar else None,
                            "ar_name": ar["name"] if ar else None,
                            "prev_name": prev["name"][:60] if prev else None,
                            "prev_gap_us": round(k["start"] - (prev["start"] + prev["dur"]), 1) if prev else None,
                            "prev_dur_us": prev["dur"] if prev else None,
                            "prev_same_stream": bool(prev and prev["stream"] == k["stream"]),
                            "mac": k["mac"],
                            "mte2": k["mte2"],
                            "mte3": k["mte3"],
                            "scalar": k["scalar"],
                            "vec": k["vec"],
                        }
                    )

        # step sums of site durations (device events; may overlap -> NOT wall)
        per_step = defaultdict(float)
        for site in r["sites"]:
            d = site["dur_us"] + (site.get("ar_dur_us") or 0 if arm == "stock" else 0)
            per_step[site["step"]] += d
        r["step_site_sums_ms"] = {str(k): round(v / 1e3, 3) for k, v in sorted(per_step.items())}
        anchors = defaultdict(list)
        for site in r["sites"]:
            anchors[site["step"]].append(site["t_rel_ms"])
        keys = sorted(anchors)
        r["step_anchor_first_ms"] = {str(k): anchors[k][0] for k in keys}
        r["step_periods_ms"] = {
            str(keys[i + 1]): round(anchors[keys[i + 1]][0] - anchors[keys[i]][0], 3) for i in range(len(keys) - 1)
        }
        r["site_count_per_step"] = {str(k): len(anchors[k]) for k in keys}

        durs = [s["dur_us"] for s in r["sites"]]
        r["site_dur_stats_us"] = {
            "n": len(durs),
            "p50": q(durs, 0.5),
            "p90": q(durs, 0.9),
            "min": q(durs, 0),
            "max": q(durs, 1),
        }

        # anomalies: any site >= 10ms
        r["anomalies"] = [
            {"step": s["step"], "t_rel_ms": s["t_rel_ms"], "dur_us": s["dur_us"], "wait_us": s["wait_us"]}
            for s in r["sites"]
            if s["dur_us"] >= 10_000
        ]

        # contiguous attribution: which sites have a preceding
        # aclnnContiguous device kernel within 2ms on the same stream?
        if arm == "v2":
            fused = [s for s in r["sites"]]
        else:
            fused = []
        contig_all = [k for k in rows if k["name"] in (CONTIG_SLICE, CONTIG_TRANSPOSE)]
        r["copies"]["contiguous_device_kernels"] = {
            "slice_n": sum(1 for k in contig_all if k["name"] == CONTIG_SLICE),
            "transpose_n": sum(1 for k in contig_all if k["name"] == CONTIG_TRANSPOSE),
            "slice_dur_us": {
                "p50": q([k["dur"] for k in contig_all if k["name"] == CONTIG_SLICE], 0.5),
                "sum_ms": round(sum(k["dur"] for k in contig_all if k["name"] == CONTIG_SLICE) / 1e3, 3),
            },
        }

        # MEMCPY_ASYNC tasks
        mc = load_memcpy_tasks(arm, rank)
        r["copies"]["memcpy_async"] = {
            "n": len(mc),
            "dur_us_p50": q([m["dur"] for m in mc], 0.5),
            "dur_us_sum_ms": round(sum(m["dur"] for m in mc) / 1e3, 3),
            "streams": sorted(set(m["stream"] for m in mc)),
        }

        # adjacency: for v2, count fused sites whose same-stream predecessor
        # within 2ms is a contiguous kernel; capture its duration.
        adj_durs, adj_n = [], 0
        for site in fused:
            if (
                site.get("prev_same_stream")
                and site.get("prev_name", "").startswith("aclnnContiguous")
                and site.get("prev_gap_us") is not None
                and site["prev_gap_us"] < 2000
            ):
                adj_n += 1
                adj_durs.append(site["prev_dur_us"])
        r["copies"]["fused_preceded_by_contig_same_stream"] = {
            "n": adj_n,
            "of": len(fused),
            "prev_dur_us_p50": q(adj_durs, 0.5),
            "prev_dur_us_sum_ms": round(sum(adj_durs) / 1e3, 3),
        }
        out["ranks"][rank] = r
    return out


def cross_rank_pairing(arm):
    """Pair same-generation hcom kernels by unique name; corrected clock."""
    res = {}
    data = {}
    for rank in ("rank0", "rank1"):
        rows = load_kernels(arm, rank)
        data[rank] = {k["name"]: k for k in rows if k["name"].startswith("hcom_")}
    common = set(data["rank0"]) & set(data["rank1"])
    only0 = set(data["rank0"]) - set(data["rank1"])
    only1 = set(data["rank1"]) - set(data["rank0"])
    pairs = []
    for n in sorted(common):
        k0, k1 = data["rank0"][n], data["rank1"][n]
        pairs.append({"name": n, "t0": k0["start"], "t1": k1["start"], "d0": k0["dur"], "d1": k1["dur"]})
    if pairs:
        offs = [p["t0"] - p["t1"] for p in pairs]
        med = st.median(offs)
        # corrected: t1 on rank0 clock = t1 + med ; skew = t0 - (t1+med)
        skews = [p["t0"] - (p["t1"] + med) for p in pairs]
        dur_ratios = [p["d1"] / p["d0"] if p["d0"] else None for p in pairs]
        dur_ratios = [x for x in dur_ratios if x is not None]
        res = {
            "n_pairs": len(pairs),
            "only_rank0": len(only0),
            "only_rank1": len(only1),
            "offset_r0_minus_r1_us_p50": round(st.median(offs), 1),
            "offset_min": round(min(offs), 1),
            "offset_max": round(max(offs), 1),
            "skew_after_correction_us": {
                "p50": q(skews, 0.5),
                "p90_abs": q([abs(x) for x in skews], 0.9),
                "max_abs": round(max(abs(x) for x in skews), 1),
            },
            "dur_ratio_r1_over_r0_p50": round(st.median(dur_ratios), 3) if dur_ratios else None,
            "first_pair": pairs[0]["name"],
            "last_pair": pairs[-1]["name"],
            "note": "offset=median(t0-t1); corrected t1c=t1+offset; skew=t0-t1c (A-02 sign fix)",
        }
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    for arm in ("stock", "v2"):
        audit = audit_arm(arm)
        with open(f"{OUT_DIR}/{arm}_raw_audit.json", "w") as f:
            json.dump(audit, f, indent=1)
        pair = cross_rank_pairing(arm)
        audit["cross_rank_hcom_pairing"] = pair
        summary[arm] = {
            "n_steps": {r: audit["ranks"][r]["n_steps"] for r in ("rank0", "rank1")},
            "site_counts": {r: audit["ranks"][r]["site_dur_stats_us"]["n"] for r in ("rank0", "rank1")},
            "site_p50_us": {r: audit["ranks"][r]["site_dur_stats_us"]["p50"] for r in ("rank0", "rank1")},
            "step_site_sums_ms": {r: audit["ranks"][r]["step_site_sums_ms"] for r in ("rank0", "rank1")},
            "step_periods_ms": {r: audit["ranks"][r]["step_periods_ms"] for r in ("rank0", "rank1")},
            "anomalies": {r: audit["ranks"][r]["anomalies"] for r in ("rank0", "rank1")},
            "hybm_orch": {r: audit["ranks"][r]["hybm_orch"] for r in ("rank0", "rank1")},
            "copies": {r: audit["ranks"][r]["copies"] for r in ("rank0", "rank1")},
            "cross_rank_hcom_pairing": pair,
        }
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
