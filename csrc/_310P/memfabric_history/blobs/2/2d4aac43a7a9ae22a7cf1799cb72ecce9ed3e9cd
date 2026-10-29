#!/usr/bin/env python3
"""batch-r1 partial analysis: c512 complete bracket + c1024 partial (2/4 stages).

Methodology follows A3_SEAT_SCHEDULE: per-sample paired two-rank max, then
quantiles; two-round p50 median as the arm center; A3 bracket = median(before,
after); drift>5% marks the case unstable. c1024 lacks a3-tail and a3-after, so
it is reported as partial/directional only and cannot be signed.
"""

import json
import math
from pathlib import Path
from statistics import median

ROOT = Path("/tmp/opencode/310p-a3cmp-20261029-001/batch-r1")
CONFIGS = [(m, g) for m in (2048, 4096) for g in (512, 1024, 2048)]


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def paired_samples(sess, m, g, rep):
    vals, walls, correct = [], [], True
    for rank in (0, 1):
        f = ROOT / sess / f"timing_r{rep}_m{m}_g{g}_rank{rank}.json"
        if not f.is_file():
            return None, None, False
        t = json.loads(f.read_text())
        correct &= all(s["correct"] for s in t["samples"])
        vals.append([s["event_us"] for s in t["samples"]])
        walls.append([s["wall_us"] for s in t["samples"]])
    paired = [max(vals[0][i], vals[1][i]) for i in range(len(vals[0]))]
    paired_wall = [max(walls[0][i], walls[1][i]) for i in range(len(walls[0]))]
    return paired, paired_wall, correct


def arm_stats(sess):
    """Return {(m,g,rep): (p50,p90,p99,wall_p50)} plus correctness flag."""
    out, ok = {}, True
    for m, g in CONFIGS:
        for rep in (0, 1):
            paired, wall, correct = paired_samples(sess, m, g, rep)
            if paired is None:
                return None, False
            ok &= correct
            s = sorted(paired)
            out[(m, g, rep)] = (quantile(s, 0.5), quantile(s, 0.9), quantile(s, 0.99),
                                quantile(sorted(wall), 0.5))
    return out, ok


def main():
    sessions = ["a3-before-c512", "a3-2-c512", "a3-tail-c512", "a3-after-c512",
                "a3-before-c1024", "a3-2-c1024"]
    stats = {}
    for s in sessions:
        st, ok = arm_stats(s)
        if st is None:
            print(f"!! {s}: incomplete timing data")
            continue
        print(f"{s}: all configs x 2 rounds present, all samples correct={ok}")
        stats[s] = st

    print("\n=== compute_rows=512 (complete bracket, same batch) ===")
    print(f"{'M':>5} {'comm':>5} | {'A3前p50':>8} {'A3-2':>8} {'A3-tail':>8} {'A3后p50':>8} "
          f"| {'前/后漂移':>8} {'A3-2两轮':>8} {'tail两轮':>8} | {'A3-2缩短':>8} {'tail缩短':>8} {'稳定':>3}")
    summary = []
    for m, g in CONFIGS:
        before = [stats["a3-before-c512"][(m, g, r)][0] for r in (0, 1)]
        after = [stats["a3-after-c512"][(m, g, r)][0] for r in (0, 1)]
        a2 = [stats["a3-2-c512"][(m, g, r)][0] for r in (0, 1)]
        tl = [stats["a3-tail-c512"][(m, g, r)][0] for r in (0, 1)]
        b, a = median(before), median(after)
        drift = max(b, a) / min(b, a) - 1
        d_a2 = max(a2) / min(a2) - 1
        d_tl = max(tl) / min(tl) - 1
        base = median([b, a])
        red_a2 = 1 - median(a2) / base
        red_tl = 1 - median(tl) / base
        stable = drift <= 0.05 and d_a2 <= 0.05 and d_tl <= 0.05
        summary.append({"M": m, "comm_rows": g, "compute_rows": 512,
                        "a3_bracket_p50_us": [round(b, 1), round(a, 1)],
                        "a3_2_p50_us": [round(x, 1) for x in a2],
                        "a3_tail_p50_us": [round(x, 1) for x in tl],
                        "a3_bracket_drift": round(drift, 4),
                        "a3_2_round_drift": round(d_a2, 4),
                        "a3_tail_round_drift": round(d_tl, 4),
                        "a3_2_reduction": round(red_a2, 4),
                        "a3_tail_reduction": round(red_tl, 4),
                        "stable_screen": stable})
        print(f"{m:>5} {g:>5} | {b:8.1f} {median(a2):8.1f} {median(tl):8.1f} {a:8.1f} "
              f"| {drift:8.2%} {d_a2:8.2%} {d_tl:8.2%} | {red_a2:+8.2%} {red_tl:+8.2%} "
              f"{'YES' if stable else 'NO':>3}")

    print("\n=== p90/p99 (paired rank max, per round, c512) ===")
    for m, g in CONFIGS:
        for label, sess in (("A3前", "a3-before-c512"), ("A3-2", "a3-2-c512"),
                            ("A3-tail", "a3-tail-c512"), ("A3后", "a3-after-c512")):
            for r in (0, 1):
                p50, p90, p99, w = stats[sess][(m, g, r)]
                print(f"M{m} g{g} {label} r{r}: p50={p50:.1f} p90={p90:.1f} "
                      f"p99={p99:.1f} wall_p50={w:.1f}")

    print("\n=== compute_rows=1024 (PARTIAL: a3-before & a3-2 only; NO bracket tail) ===")
    print(f"{'M':>5} {'comm':>5} | {'A3前p50':>8} {'A3-2':>8} | {'A3-2缩短':>8}")
    for m, g in CONFIGS:
        before = [stats["a3-before-c1024"][(m, g, r)][0] for r in (0, 1)]
        a2 = [stats["a3-2-c1024"][(m, g, r)][0] for r in (0, 1)]
        b = median(before)
        red = 1 - median(a2) / b
        d_a2 = max(a2) / min(a2) - 1
        d_b = max(before) / min(before) - 1
        print(f"{m:>5} {g:>5} | {b:8.1f} {median(a2):8.1f} | {red:+8.2%}   "
              f"(A3前两轮漂移 {d_b:.2%}, A3-2两轮漂移 {d_a2:.2%})")

    out = {"batch": "batch-r1 (STOPPED at a3-tail-c1024 by lease guard)",
           "method": "paired two-rank max per sample; two-round p50 median; A3 bracket median",
           "compute_512_complete_bracket": summary,
           "compute_1024_partial": "a3-tail-c1024 and a3-after-c1024 missing; not signable"}
    Path("/tmp/opencode/310p-a3cmp-20261029-001/batch-r1-partial-analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("\nwritten: batch-r1-partial-analysis.json")


if __name__ == "__main__":
    main()
