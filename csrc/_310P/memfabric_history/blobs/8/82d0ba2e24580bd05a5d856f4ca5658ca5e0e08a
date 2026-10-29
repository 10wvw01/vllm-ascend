#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1-R1周期性长尾纯CPU分析。仅定位现象，绝不推断MemFabric根因。"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

ORDER = ("S1", "F1", "F2", "S2")
ARMS = ("stock", "v2", "v2", "stock")


def examine(report: dict) -> dict:
    if report.get("schema") != "A1_L3_NO_PROFILER_ABBA_V1":
        raise ValueError("not L3 no-profiler device evidence")
    sessions = report.get("four_serve_sessions")
    if not isinstance(sessions, list) or len(sessions) != 4:
        raise ValueError("need precisely four independently measured sessions")
    parsed = []
    for s, sid, arm in zip(sessions, ORDER, ARMS):
        if (s.get("session_id"), s.get("arm")) != (sid, arm):
            raise ValueError("invalid ABBA session identity/order")
        r0, r1, paired = (s.get(k) for k in ("rank0_ms", "rank1_ms", "paired_critical_ms"))
        if any(not isinstance(v, list) or len(v) != 30 for v in (r0, r1, paired)):
            raise ValueError("30 complete dual-rank events required")
        if any(not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) and x > 0 for x in v)
               for v in (r0, r1, paired)):
            raise ValueError("missing/nonfinite event timing")
        if any(abs(max(a, b) - c) > .005 for a, b, c in zip(r0, r1, paired)):
            raise ValueError("not paired by request max(rank0,rank1)")
        median = statistics.median(paired)
        mad = statistics.median(abs(x - median) for x in paired)
        # 以20ms绝对门排除Stock的细小漂移；鲁棒MAD门防止融合慢稳态误判。
        threshold = max(20.0, 6 * 1.4826 * mad)
        spikes = [i for i, x in enumerate(paired) if x - median > threshold]
        parsed.append({
            "id": sid, "arm": arm, "median_ms": round(median, 5),
            "mad_ms": round(mad, 5), "spike_threshold_ms": round(threshold, 5),
            "spike_indices_zero_based": spikes,
            "gap_indices": [b-a for a, b in zip(spikes, spikes[1:])],
            "spike_excess_ms": [round(paired[i] - median, 4) for i in spikes],
            "rank_skew_at_spikes_ms": [round(abs(r0[i]-r1[i]), 4) for i in spikes],
        })
    f1, f2 = parsed[1], parsed[2]
    pattern = (
        not parsed[0]["spike_indices_zero_based"]
        and not parsed[3]["spike_indices_zero_based"]
        and len(f1["spike_indices_zero_based"]) >= 4
        and f1["spike_indices_zero_based"] == f2["spike_indices_zero_based"]
        and all(5 <= gap <= 8 for gap in f1["gap_indices"])
        and all(x < 2.0 for s in (f1, f2)
                for x in s["rank_skew_at_spikes_ms"])
    )
    return {
        "schema": "A1_R1_PERIODIC_TAIL_V1",
        "status": ("REPRODUCIBLE_PERIODIC_V2_TAIL_CAUSE_UNKNOWN"
                   if pattern else "PERIODIC_TAIL_NOT_PROVEN"),
        "sessions": parsed,
        "limitations": [
            "Zero-based indices are request order after six warmup queries; not device epoch numbers.",
            "No causality: correlation does NOT identify MM/SDMA/IB/pool/host mechanisms.",
            "Same-rank timing matches cannot establish a shared physical clock; only per-request elapsed durations compared.",
            "Measurement uses Forward-to-Logits NPU event; no formal L4/TTFT claims."
        ],
        "next_discriminator": (
            "Capture immutable request-index -> time -> MemFabric pool epoch/generation/"
            "mail tag and progress -> Rank0/1 completion -> kernel and host trace; "
            "profile only a short targeted window and avoid inserted synchronization.")
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l3-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not a.l3_json.is_absolute() or not a.out.is_absolute() or a.out.exists():
        ap.error("use absolute input and fresh absolute output")
    result = examine(json.loads(a.l3_json.read_text(encoding="utf-8")))
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
