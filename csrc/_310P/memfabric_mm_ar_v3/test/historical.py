# SPDX-License-Identifier: Apache-2.0
"""Historical baselines are read from archived evidence; never execute old arms."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
V2 = REPO / "csrc/_310P/memfabric_mm_ar_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baselines():
    paths = [
        V2 / "evidence/e3_chunk2048_20261028/e3_assessment.json",
        V2 / "evidence/e4_layered_20261028/attempt2_3_mainline_archive/attempt2_e4_screen_summary.json",
    ]
    e3, e4 = [json.loads(p.read_text()) for p in paths]
    if not e3["stock_native_NZ_route_verified"]:
        raise ValueError("historical Stock native NZ route missing")
    rows = []
    for c in e3["comparisons"]:
        for arm, value in c["arm_time_us"].items():
            if arm == "e3t":
                continue
            rows.append(
                {
                    "batch": "E3",
                    "arm": arm,
                    "M": c["M"],
                    "p50_us": value,
                    "source": str(paths[0].relative_to(REPO)),
                    "source_sha256": digest(paths[0]),
                    "original_status": e3["status"],
                    "shape_stable": c["stable"],
                }
            )
    for arm, shapes in e4["rank_pair_p50_us"].items():
        for m, value in shapes.items():
            rows.append(
                {
                    "batch": "E4_attempt2",
                    "arm": arm,
                    "M": int(m),
                    "p50_us": value,
                    "source": str(paths[1].relative_to(REPO)),
                    "source_sha256": digest(paths[1]),
                    "original_status": e4["status"],
                    "shape_stable": None,
                }
            )
    table = V2 / "docs/PERFORMANCE_BASELINES.md"
    matches = re.findall(r"\| \*\*R2 Screen M(2048|4096)\*\* \| ([0-9.]+)µs \| ([0-9.]+)µs", table.read_text())
    if len(matches) != 2:
        raise ValueError("historical D4 table changed; audit baselines before comparison")
    for m, _stock, d4 in matches:
        rows.append(
            {
                "batch": "R2_screen",
                "arm": "v2_d4",
                "M": int(m),
                "p50_us": float(d4),
                "source": str(table.relative_to(REPO)),
                "source_sha256": digest(table),
                "original_status": "SCREEN_ONLY",
                "shape_stable": None,
            }
        )
    # V3.1 default 4MiB fixed session: recompute from paired raw samples,
    # never select a minimum across different runs.
    for directory, batch, prefix in (
        ("quick-20261029-v31/default-r2-pass", "V3.1_default_r2", "v31"),
        ("quick-20261029-v32a2-plan-a/pass", "A2_fixed_session", "a2"),
    ):
        archived = REPO / "csrc/_310P/memfabric_mm_ar_v3/experiments" / directory
        files = [archived / f"rank{r}.json" for r in (0, 1)]
        ranks = [json.loads(p.read_text()) for p in files]
        paired = {}
        for a, b in zip(ranks[0]["measurements"], ranks[1]["measurements"], strict=True):
            key = (a["M"], a["group"])
            if any(a[k] != b[k] for k in ("M", "group", "repeat")):
                raise ValueError("archived V3.1 rank case mismatch")
            samples = []
            for x, y in zip(a["samples"], b["samples"], strict=True):
                if x["i"] != y["i"] or not x["correct"] or not y["correct"]:
                    raise ValueError("archived V3.1 sample mismatch")
                samples.append(max(x["event_us"], y["event_us"]))
            paired.setdefault(key, []).append(percentile(samples, 0.5))
        for (m, group), times in paired.items():
            rows.append(
                {
                    "batch": batch,
                    "arm": f"{prefix}_g{group}",
                    "M": m,
                    "p50_us": statistics.median(times),
                    "source": str(archived.relative_to(REPO)),
                    "source_sha256": {str(p.relative_to(REPO)): digest(p) for p in files},
                    "original_status": "HISTORICAL_SCREEN_ONLY",
                    "shape_stable": None,
                }
            )
    return rows


def percentile(values, q):
    a = sorted(values)
    if not a or not all(math.isfinite(x) and x > 0 for x in a):
        raise ValueError("latencies must be finite and positive")
    at = (len(a) - 1) * q
    lo = int(at)
    return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (at - lo)


def assess(rank0, rank1):
    if rank0["rank"] != 0 or rank1["rank"] != 1:
        raise ValueError("rank identity mismatch")
    for key in ("run_id", "build_manifest_sha256", "configuration"):
        if rank0[key] != rank1[key]:
            raise ValueError(f"rank-pair {key} mismatch")
    required_gates = {"ib", "add", "protocol", "fused", "tail"}
    if rank0["configuration"].get("path") in ("V32_PLAN_A_POOL_RETURN", "A3_DIRECT_ADD"):
        required_gates.add("output_lifetime")
    for r in (rank0, rank1):
        if r.get("status") != "PASS" or set(r.get("gates", {})) != required_gates:
            raise ValueError("incomplete or failed V3 prerequisites")
        if not all(v == "PASS" for v in r["gates"].values()):
            raise ValueError("failed V3 gate")
    paired = []
    config = rank0["configuration"]
    expected_cases = {
        (m, g, config["pipeline"], r) for m in config["M"] for g in config["groups"] for r in range(config["repeats"])
    }
    for rank in (rank0, rank1):
        actual = [(z["M"], z["group"], z["pipeline"], z["repeat"]) for z in rank["measurements"]]
        if len(actual) != len(expected_cases) or set(actual) != expected_cases:
            raise ValueError("missing or duplicate measurement case")
    if len(rank0["measurements"]) != len(rank1["measurements"]):
        raise ValueError("incomplete rank measurements")
    for a, b in zip(rank0["measurements"], rank1["measurements"]):
        for key in ("M", "group", "pipeline", "repeat"):
            if a[key] != b[key]:
                raise ValueError("case identity mismatch")
        if len(a["samples"]) != len(b["samples"]) or len(a["samples"]) != config["samples"]:
            raise ValueError("sample count mismatch")
        ev, wall = [], []
        for x, y in zip(a["samples"], b["samples"]):
            if x["i"] != y["i"] or x["variant"] != y["variant"] or not x["correct"] or not y["correct"]:
                raise ValueError("sample identity or correctness mismatch")
            if not all(math.isfinite(z[k]) and z[k] > 0 for z in (x, y) for k in ("event_us", "wall_us")):
                raise ValueError("invalid latency")
            ev.append(max(x["event_us"], y["event_us"]))
            wall.append(max(x["wall_us"], y["wall_us"]))
        paired.append(
            {
                **{k: a[k] for k in ("M", "group", "pipeline", "repeat")},
                "samples": len(ev),
                "p50_us": percentile(ev, 0.5),
                "p90_us": percentile(ev, 0.9),
                "p99_us": percentile(ev, 0.99),
                "wall_p50_us": percentile(wall, 0.5),
            }
        )
    comparisons = []
    for shape in sorted({(z["M"], z["group"], z["pipeline"]) for z in paired}):
        sessions = [z for z in paired if (z["M"], z["group"], z["pipeline"]) == shape]
        medians = [z["p50_us"] for z in sessions]
        drift = max(medians) / min(medians) - 1
        value = statistics.median(medians)
        historical = []
        for old in baselines():
            if old["M"] == shape[0]:
                historical.append(
                    {
                        **old,
                        "v3_speedup_pct": (old["p50_us"] / value - 1) * 100,
                        "latency_reduction_pct": (1 - value / old["p50_us"]) * 100,
                    }
                )
        comparisons.append(
            {
                "M": shape[0],
                "group": shape[1],
                "pipeline": shape[2],
                "v3_p50_us": value,
                "v3_session_drift": drift,
                "direction_usable": len(sessions) >= 2 and drift <= 0.05,
                "historical_comparisons": historical,
            }
        )
    return {
        "schema": "MF310P_V3_HISTORICAL_SCREEN_V1",
        "status": "HISTORICAL_COMPARISON_ONLY",
        "same_batch_controls_run": False,
        "formal_performance_go": False,
        "scope": "operator_only; no model/TTFT/Graph or release stability claim",
        "paired_sessions": paired,
        "comparisons": comparisons,
        "limitations": [
            "Historical binaries, MemFabric KVER, card, workload and launch overhead can differ",
            "Small improvements inside the historical/session variation remain inconclusive",
            "No V2/Stock performance arm is automatically scheduled",
        ],
    }
