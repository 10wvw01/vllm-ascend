#!/usr/bin/env python3
"""Gate A/B raw-trace audit pass 2: fixed step clustering, same-stream
neighbor attribution, per-step rank-offset evolution, memcpy forensics."""

import csv
import glob
import json
import statistics as st
from collections import defaultdict

BASE = "/tmp/opencode/v2_prefill_captures"
OUT_DIR = "/tmp/opencode/gate_ab_audit"
STEP_GAP_US = 20_000
MIN_STEP_KERNELS = 50
FUSED_PREFIX = "_ZN12_GLOBAL__N_113V2TrialKernel"
TARGET_MM = "aclnnMatmulWeightNz_MatMulV3Common_MatMulV3"
CONTIG_PREFIX = "aclnnContiguous"


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


def cluster_steps(rows):
    """Cluster on non-persistent (dur<1s) AI_CORE/COMMUNICATION/MIX_AICORE
    kernels only; persistent AICPU kernels span the whole window."""
    timeline = [k for k in rows if k["dur"] < 1_000_000]
    steps, cur = [], [timeline[0]]
    for k in timeline[1:]:
        if k["start"] - (cur[-1]["start"] + cur[-1]["dur"]) > STEP_GAP_US:
            steps.append(cur)
            cur = [k]
        else:
            cur.append(k)
    steps.append(cur)
    return [s for s in steps if len(s) > MIN_STEP_KERNELS]


def q(v, p):
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, int(len(s) * p))], 1)


def same_stream_prev(step_rows, idx):
    """Previous kernel on the SAME stream before step_rows[idx]."""
    k = step_rows[idx]
    for j in range(idx - 1, -1, -1):
        if step_rows[j]["stream"] == k["stream"]:
            return step_rows[j]
    return None


def audit(arm):
    out = {}
    for rank in ("rank0", "rank1"):
        rows = load_kernels(arm, rank)
        mc = load_memcpy(arm, rank)
        steps = cluster_steps(rows)
        r = {
            "n_steps": len(steps),
            "step_sizes": [len(s) for s in steps],
            "sites": [],
            "memcpy": {},
            "anomaly_detail": [],
        }

        # memcpy forensics
        durs = [m["dur"] for m in mc]
        nonzero = [m for m in mc if m["dur"] > 1]
        r["memcpy"] = {
            "n_tasks": len(mc),
            "n_dur_gt1us": len(nonzero),
            "dur_p50_all": q(durs, 0.5),
            "dur_sum_ms": round(sum(durs) / 1e3, 3),
            "dur_gt1_p50": q([m["dur"] for m in nonzero], 0.5),
            "dur_gt1_sum_ms": round(sum(m["dur"] for m in nonzero) / 1e3, 3),
            "dur_gt1_range_us": [q([m["dur"] for m in nonzero], 0), q([m["dur"] for m in nonzero], 1)],
            "streams_all": sorted(set(m["stream"] for m in mc)),
        }

        for si, s in enumerate(steps):
            t0 = s[0]["start"]
            # step-local memcpy tasks
            step_mc = [m for m in mc if t0 <= m["start"] <= s[-1]["start"] + s[-1]["dur"]]
            site_idx = 0
            for i, k in enumerate(s):
                site = None
                if arm == "v2" and k["name"].startswith(FUSED_PREFIX):
                    prev = same_stream_prev(s, i)
                    site = {
                        "step": si,
                        "site_idx": site_idx,
                        "t_rel_ms": round((k["start"] - t0) / 1e3, 3),
                        "dur_us": round(k["dur"], 1),
                        "wait_us": round(k["wait"], 1),
                        "stream": k["stream"],
                        "ss_prev_name": prev["name"][:70] if prev else None,
                        "ss_prev_gap_us": round(k["start"] - (prev["start"] + prev["dur"]), 1) if prev else None,
                        "ss_prev_dur_us": round(prev["dur"], 1) if prev else None,
                    }
                    site_idx += 1
                elif arm == "stock" and k["name"] == TARGET_MM:
                    ar = next((x for x in s[i + 1 : i + 6] if x["name"].startswith("hcom_")), None)
                    prev = same_stream_prev(s, i)
                    site = {
                        "step": si,
                        "site_idx": site_idx,
                        "t_rel_ms": round((k["start"] - t0) / 1e3, 3),
                        "dur_us": round(k["dur"], 1),
                        "wait_us": round(k["wait"], 1),
                        "stream": k["stream"],
                        "shapes": k["shapes"][:90],
                        "ar_dur_us": round(ar["dur"], 1) if ar else None,
                        "ar_gap_us": round(ar["start"] - (k["start"] + k["dur"]), 1) if ar else None,
                        "ss_prev_name": prev["name"][:70] if prev else None,
                        "ss_prev_gap_us": round(k["start"] - (prev["start"] + prev["dur"]), 1) if prev else None,
                        "ss_prev_dur_us": round(prev["dur"], 1) if prev else None,
                    }
                    site_idx += 1
                if site:
                    r["sites"].append(site)
            r.setdefault("step_memcpy", {})[str(si)] = {
                "n": len(step_mc),
                "sum_ms": round(sum(m["dur"] for m in step_mc) / 1e3, 3),
                "n_gt1us": sum(1 for m in step_mc if m["dur"] > 1),
            }

        # per-step site sums and periods
        per_step = defaultdict(float)
        anchors = defaultdict(list)
        for site in r["sites"]:
            d = site["dur_us"] + (site.get("ar_dur_us") or 0 if arm == "stock" else 0)
            per_step[site["step"]] += d
            anchors[site["step"]].append(site["t_rel_ms"])
        r["step_site_sums_ms"] = {str(k): round(v / 1e3, 3) for k, v in sorted(per_step.items())}
        r["step_site_counts"] = {str(k): len(anchors[k]) for k in sorted(anchors)}
        keys = sorted(anchors)
        r["step_periods_ms"] = [round(anchors[keys[i + 1]][0] - anchors[keys[i]][0], 3) for i in range(len(keys) - 1)]
        stable = [k for k in keys if not any(s["step"] == k and s["dur_us"] >= 10000 for s in r["sites"])]
        r["stable_steps"] = stable
        r["stable_site_sum_mean_ms"] = round(st.mean([per_step[k] / 1e3 for k in stable]), 3) if stable else None

        # anomaly detail with neighbors
        for site in r["sites"]:
            if site["dur_us"] >= 10_000:
                r["anomaly_detail"].append(site)

        # contiguous adjacency per site (same-stream, within 3ms)
        adj = [
            s
            for s in r["sites"]
            if s.get("ss_prev_name", "").startswith(CONTIG_PREFIX)
            and s.get("ss_prev_gap_us") is not None
            and s["ss_prev_gap_us"] < 3000
        ]
        r["contig_adjacent_sites"] = {
            "n": len(adj),
            "of": len(r["sites"]),
            "prev_dur_p50_us": q([s["ss_prev_dur_us"] for s in adj], 0.5),
            "prev_dur_sum_ms": round(sum(s["ss_prev_dur_us"] for s in adj) / 1e3, 3),
            "prev_gap_p50_us": q([s["ss_prev_gap_us"] for s in adj], 0.5),
        }
        out[rank] = r
    return out


def pair_evolution(arm):
    data = {}
    for rank in ("rank0", "rank1"):
        rows = load_kernels(arm, rank)
        data[rank] = {k["name"]: k for k in rows if k["name"].startswith("hcom_")}
    common = sorted(set(data["rank0"]) & set(data["rank1"]))
    pairs = [(n, data["rank0"][n]["start"], data["rank1"][n]["start"]) for n in common]
    pairs.sort(key=lambda x: x[1])  # by rank0 time
    # split into 10 chronological buckets, report median offset per bucket
    n = len(pairs)
    buckets = []
    for b in range(10):
        seg = pairs[b * n // 10 : (b + 1) * n // 10]
        if not seg:
            continue
        offs = [t0 - t1 for _, t0, t1 in seg]
        buckets.append(
            {
                "bucket": b,
                "n": len(seg),
                "offset_p50_us": round(st.median(offs), 1),
                "t0_first_s": round((seg[0][1] - pairs[0][1]) / 1e6, 3),
            }
        )
    return {
        "n_pairs": n,
        "only_r0": len(set(data["rank0"]) - set(data["rank1"])),
        "only_r1": len(set(data["rank1"]) - set(data["rank0"])),
        "offset_evolution_buckets": buckets,
        "note": "offset=t0-t1 per same-name hcom event; drift=rank lag, not clock",
    }


def main():
    res = {}
    for arm in ("stock", "v2"):
        res[arm] = audit(arm)
        res[arm]["pair_evolution"] = pair_evolution(arm)
    with open(f"{OUT_DIR}/audit_pass2.json", "w") as f:
        json.dump(res, f, indent=1)
    # compact printout
    for arm in ("stock", "v2"):
        print(f"===== {arm} =====")
        for rank in ("rank0", "rank1"):
            r = res[arm][rank]
            print(f" {rank}: steps={r['n_steps']} sizes={r['step_sizes']}")
            print(f"   site_sums_ms={r['step_site_sums_ms']}")
            print(f"   site_counts={r['step_site_counts']}")
            print(f"   periods_ms={r['step_periods_ms']}")
            print(f"   stable_steps={r['stable_steps']} stable_mean={r['stable_site_sum_mean_ms']}")
            print(f"   contig_adj={r['contig_adjacent_sites']}")
            print(f"   memcpy={r['memcpy']}")
            if r["anomaly_detail"]:
                for a in r["anomaly_detail"]:
                    print(f"   ANOMALY: {a}")
        pe = res[arm]["pair_evolution"]
        print(f" pair: n={pe['n_pairs']} orphan_r0={pe['only_r0']} orphan_r1={pe['only_r1']}")
        for b in pe["offset_evolution_buckets"]:
            print(f"   bucket {b['bucket']}: n={b['n']} offset_p50={b['offset_p50_us']}us @t0+{b['t0_first_s']}s")
        print()


if __name__ == "__main__":
    main()
