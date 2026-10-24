#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""C3-3 ABBA 结果分析：TP2 配对统计（fail-closed）。

配对定义（沿 D2/D3/D4 口径）：每个双 Rank 配对请求取 max(rank0, rank1)；
external 无 Profiler event_us 为主验收口径；异常/长尾不删除。
基线 = D4 Tile2048（2a19390c，与 D4 构建字节一致），候选 = C3（00b42653）。
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path("/tmp/opencode/c3/sessions")
SESSIONS = ["base_r1", "cand_r1", "cand_r2", "base_r2", "base_r3", "cand_r3", "cand_r4", "base_r4"]
D4_TILE2048_DEVO1_P50 = 2556.3  # D4-3 同形状（M4096/TP2）候选配对 p50，dev0/1 参考锚
NOISE = 0.033
STALL_US = 6000.0
EXPECTED_SO = {
    "base": "2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19",
    "cand": "00b42653be2c7951398929cfa07696914c70215770854a6ab89ec5c5cb735564",
}


def load(name):
    out = {}
    for r in (0, 1):
        with open(ROOT / f"abba_{name}" / f"{name}_rank{r}_c3.json") as fh:
            d = json.load(fh)
        assert d["status"] == 0 and d["bit_exact"], f"{name} rank{r} gate failed"
        assert d["device_so"] == EXPECTED_SO[d["label"].split("_")[0]], f"{name} rank{r} SO mismatch: {d['device_so']}"
        assert d["M"] == 4096 and d["chunk_rows"] == 1024 and d["iters"] == 30, f"{name} rank{r} shape/iters"
        out[r] = d
    n0, n1 = len(out[0]["samples"]), len(out[1]["samples"])
    assert n0 == n1 == out[0]["iters"], f"{name} sample count mismatch {n0} vs {n1}"
    return out


def stats(vals):
    sv = sorted(vals)
    n = len(sv)
    p50 = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    p90 = sv[min(n - 1, int(round(0.9 * n)) - 1)]
    return {
        "p50": round(p50, 1),
        "p90": round(p90, 1),
        "min": round(sv[0], 1),
        "max": round(sv[-1], 1),
        "mean": round(st.mean(sv), 1),
    }


def main():
    data = {name: load(name) for name in SESSIONS}
    print(
        f"{'session':10} {'r0 ev p50':>10} {'r1 ev p50':>10} {'pair ev p50':>11} "
        f"{'pair ev p90':>11} {'pair w p50':>10} {'stall':>5} {'so':>6}"
    )
    per_cond = {"base": [], "cand": []}
    all_stalls = []
    for name in SESSIONS:
        d0, d1 = data[name][0], data[name][1]
        ev0 = [s["event_us"] for s in d0["samples"]]
        ev1 = [s["event_us"] for s in d1["samples"]]
        w0 = [s["wall_us"] for s in d0["samples"]]
        w1 = [s["wall_us"] for s in d1["samples"]]
        pair_ev = [max(a, b) for a, b in zip(ev0, ev1)]
        pair_w = [max(a, b) for a, b in zip(w0, w1)]
        stalls = [v for v in pair_ev if v > STALL_US] + [v for v in pair_w if v > STALL_US]
        all_stalls += [(name, v) for v in stalls]
        cond = name.split("_")[0]
        per_cond[cond].append({"name": name, "pair_ev": pair_ev, "pair_w": pair_w, "r0": ev0, "r1": ev1})
        s, sw = stats(pair_ev), stats(pair_w)
        print(
            f"{name:10} {stats(ev0)['p50']:>10.0f} {stats(ev1)['p50']:>10.0f} "
            f"{s['p50']:>11.0f} {s['p90']:>11.0f} {sw['p50']:>10.0f} {len(stalls):>5} {cond:>6}"
        )

    print("\n=== per-condition aggregate (paired TP2 event_us, primary) ===")
    agg = {}
    for cond in ("base", "cand"):
        p50s = [stats(x["pair_ev"])["p50"] for x in per_cond[cond]]
        pooled = [v for x in per_cond[cond] for v in x["pair_ev"]]
        pooled_w = [v for x in per_cond[cond] for v in x["pair_w"]]
        agg[cond] = {"p50s": p50s, "stats": stats(pooled), "wall": stats(pooled_w)}
        disp = (max(p50s) - min(p50s)) / min(p50s) * 100
        print(
            f"{cond}: session p50s={[round(p) for p in p50s]} dispersion={disp:.1f}% "
            f"pooled={agg[cond]['stats']} wall_p50={agg[cond]['wall']['p50']:.0f}"
        )

    b, c = agg["base"]["stats"]["p50"], agg["cand"]["stats"]["p50"]
    delta = (b - c) / b * 100
    print("\n=== verdict (pooled paired event p50; 3% gate) ===")
    print(f"base(D4 Tile2048)={b:.0f}us cand(C3)={c:.0f}us  delta={(b - c):.0f}us ({delta:+.2f}%)")
    print(f"reference: D4-3 Tile2048 paired p50 on dev0/1 = {D4_TILE2048_DEVO1_P50}us")

    print("\n=== pairwise adjacent ABBA deltas (base->cand) ===")
    pairwise = {}
    for bn, cn in [("base_r1", "cand_r1"), ("base_r2", "cand_r2"), ("base_r3", "cand_r3"), ("base_r4", "cand_r4")]:
        pb = stats(next(x["pair_ev"] for x in per_cond["base"] if x["name"] == bn))["p50"]
        pc = stats(next(x["pair_ev"] for x in per_cond["cand"] if x["name"] == cn))["p50"]
        pairwise[f"{bn}->{cn}"] = round((pb - pc) / pb * 100, 2)
        print(f"{bn} vs {cn}: {pb:.0f} -> {pc:.0f} ({(pb - pc) / pb * 100:+.2f}%)")
    print(f"\nstall census (>6ms paired): {all_stalls if all_stalls else 'NONE'}")

    out = {
        "per_session": {
            n: {
                "r0": stats(s["event_us"] for s in data[n][0]["samples"]),
                "r1": stats(s["event_us"] for s in data[n][1]["samples"]),
                "pair_event": stats(
                    max(a["event_us"], b["event_us"]) for a, b in zip(data[n][0]["samples"], data[n][1]["samples"])
                ),
                "pair_wall": stats(
                    max(a["wall_us"], b["wall_us"]) for a, b in zip(data[n][0]["samples"], data[n][1]["samples"])
                ),
            }
            for n in SESSIONS
        },
        "aggregate": {k: {"p50s": v["p50s"], "stats": v["stats"], "wall": v["wall"]} for k, v in agg.items()},
        "delta_pct_paired_event_p50": round(delta, 2),
        "delta_us": round(b - c, 1),
        "pairwise_adjacent_pct": pairwise,
        "stalls": all_stalls,
        "anchor": {"d4_tile2048_dev01_paired_p50_us": D4_TILE2048_DEVO1_P50},
    }
    Path("/tmp/opencode/c3/c3_abba_analysis.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
