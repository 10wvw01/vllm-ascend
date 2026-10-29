import json
from pathlib import Path
import statistics

out = Path("/tmp/opencode/310p-v32a2-20261029-001/results")

def pct(vals, p):
    s = sorted(vals)
    import math
    k = (len(s)-1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c: return s[int(k)]
    return s[f]*(c-k) + s[c]*(k-f)

rows = []
for m in (2048, 4096):
    for g in (512, 1024, 2048):
        per_round = {}
        for r in (0, 1):
            data = {}
            for rank in (0, 1):
                j = json.loads((out / f"timing_r{r}_m{m}_g{g}_rank{rank}.json").read_text())
                data[rank] = j["samples"]
            n = min(len(data[0]), len(data[1]))
            ev_max = [max(data[0][i]["event_us"], data[1][i]["event_us"]) for i in range(n)]
            wall_max = [max(data[0][i]["wall_us"], data[1][i]["wall_us"]) for i in range(n)]
            ev0 = [data[0][i]["event_us"] for i in range(n)]
            ev1 = [data[1][i]["event_us"] for i in range(n)]
            per_round[r] = {
                "event_p50": pct(ev_max, 0.50), "event_p90": pct(ev_max, 0.90), "event_p99": pct(ev_max, 0.99),
                "wall_p50": pct(wall_max, 0.50), "wall_p90": pct(wall_max, 0.90), "wall_p99": pct(wall_max, 0.99),
                "rank0_event_p50": statistics.median(ev0), "rank1_event_p50": statistics.median(ev1),
                "samples": n,
            }
        d = per_round[0]; d2 = per_round[1]
        drift = abs(d["event_p50"] - d2["event_p50"]) / min(d["event_p50"], d2["event_p50"])
        rows.append({
            "M": m, "group": g,
            "round0": d, "round1": d2,
            "r0_r1_p50_drift_pct": round(drift*100, 2),
            "best_round_p50": min(d["event_p50"], d2["event_p50"]),
            "path": "V32_PLAN_A_POOL_RETURN",
        })

print(f"{'M':>5} {'g':>5} | r0 ev p50/p90/p99 | r1 ev p50/p90/p99 | drift% | wall p50 (r0/r1)")
for r in rows:
    a, b = r["round0"], r["round1"]
    print(f"{r['M']:>5} {r['group']:>5} | "
          f"{a['event_p50']:8.1f} {a['event_p90']:8.1f} {a['event_p99']:8.1f} | "
          f"{b['event_p50']:8.1f} {b['event_p90']:8.1f} {b['event_p99']:8.1f} | "
          f"{r['r0_r1_p50_drift_pct']:5.2f} | {a['wall_p50']:8.1f}/{b['wall_p50']:8.1f}")
    print(f"      rank-split r0: {a['rank0_event_p50']:.1f}/{a['rank1_event_p50']:.1f}  r1: {b['rank0_event_p50']:.1f}/{b['rank1_event_p50']:.1f}  n={a['samples']}")

json.dump(rows, open("/tmp/opencode/310p-v32a2-20261029-001/perf_summary.json", "w"), indent=2)
