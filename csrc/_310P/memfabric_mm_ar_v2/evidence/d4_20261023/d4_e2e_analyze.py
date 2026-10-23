#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D4-4 E2E ABBA 最终分析：仅干净会话（挂死会话无有效指标，两臂同环境暴露）。

会话集：dev6/7 默认 epoch 预算 —— base n=2（base_r1/r2）、cand n=3（cand_r1/r3/r4）。
诊断性会话（dev4/5、budget=5s）不进入性能对比，见报告 §5.2。
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "e2e_dev67"
SESSIONS = {"base": ["base_r1", "base_r2"], "cand": ["cand_r1", "cand_r3", "cand_r4"]}


def pct(vals, p):
    sv = sorted(vals)
    return sv[min(len(sv) - 1, int(round(p * len(sv))) - 1)]


def tpot_of(d):
    rows = [r for r in d["itls"] if any(x > 0 for x in r)]
    return st.median([sum(r) / len([x for x in r if x > 0]) for r in rows])


def main():
    agg = {}
    for cond, names in SESSIONS.items():
        rows = []
        for n in names:
            f = next((ROOT / n).glob("*rep1_run.json"))
            with open(f) as fh:
                d = json.load(fh)
            assert d["completed"] == d["num_prompts"] and d["failed"] == 0
            rows.append(
                {
                    "name": n,
                    "ttft_p50": pct(d["ttfts"], 0.5),
                    "ttft_p95": pct(d["ttfts"], 0.95),
                    "tpot": tpot_of(d),
                    "tput": d["output_throughput"],
                    "duration": d["duration"],
                }
            )
        agg[cond] = rows
        for r in rows:
            print(
                f"{r['name']}: ttft_p50={r['ttft_p50']:.2f}s p95={r['ttft_p95']:.2f}s "
                f"tpot={r['tpot']:.3f}s tput={r['tput']:.2f} dur={r['duration']:.0f}s"
            )
    print("=== verdict (mean of clean sessions) ===")
    for k in ("ttft_p50", "ttft_p95", "tpot", "tput"):
        b = st.mean(r[k] for r in agg["base"])
        c = st.mean(r[k] for r in agg["cand"])
        print(f"{k}: base={b:.3f} cand={c:.3f} delta={(c - b) / b * 100:+.2f}%")
    out = Path(__file__).resolve().parent / "d4_e2e_final_analysis.json"
    out.write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
