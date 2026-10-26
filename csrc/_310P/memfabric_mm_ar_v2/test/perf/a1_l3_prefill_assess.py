#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L3 Eager Prefill step no-profiler dual-rank ABBA CPU-only checker.

Requires 4 independent serve sessions S1,F1,F2,S2, each measured with
A1_PREFILL_FORWARD_TO_LOGITS_EVENT_V1. Never substitutes HTTP latency,
Profiler time, or sum of 40 layer durations for L3 device time.
Gates false unless a separate evidence-backed L2/step calibration audit
proves current-stream events actually bracket relevant full dependency.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

SCHEDULE = (("S1", "stock"), ("F1", "v2"), ("F2", "v2"), ("S2", "stock"))
AUDIT_TRUE = (
    "actual_40_target_sites_each_rank",
    "same_model_input_and_matmul_M_4096",
    "stock_hccl_and_d4_kernel_route_verified",
    "rank0_rank1_requests_aligned",
    "model_forward_to_logits_completed_dependencies_verified",
    "no_extra_device_sync_by_probe",
    "all_four_sessions_no_profiler",
    "same_card_and_software_fingerprints",
    "numerical_correctness_verified",
    "no_unresolved_device_errors",
)
EVENT_SCHEMA = "A1_PREFILL_FORWARD_TO_LOGITS_EVENT_V1"


def fail_unless(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def check_one_session(root: Path, sid: str, arm: str, warmup: int, measured: int) -> dict:
    session = root / "sessions" / sid
    meta = json.loads((session / "manifest.json").read_text())
    for k, value in {"session_id": sid, "arm": arm, "input_tokens": 4096,
                     "output_tokens": 1, "concurrency": 1, "tp": 2,
                     "eager": True, "profiler": False,
                     "warmup_requests": warmup, "measured_requests": measured}.items():
        fail_unless(meta.get(k) == value, f"{sid}: invalid {k}")
    identity = meta.get("identity")
    fail_unless(isinstance(identity, dict)
                and all(isinstance(identity.get(k), str) and identity[k]
                        for k in ("model", "model_revision", "physical_devices",
                                  "host_so_sha256", "device_so_sha256",
                                  "plugin_so_sha256", "kver_sha256")),
                f"{sid}: incomplete identity")
    files = sorted(session.glob("a1_prefill_rank*_pid*.jsonl"))
    events = {}
    for rank in (0, 1):
        matching = [p for p in files if p.name.startswith(f"a1_prefill_rank{rank}_pid")]
        fail_unless(len(matching) == 1, f"{sid}: Rank{rank} has missing/multiple event files")
        rows = [json.loads(v) for v in matching[0].read_text().splitlines() if v.strip()]
        selected = [v for v in rows if v.get("num_scheduled_tokens") == 4096]
        fail_unless(len(selected) == warmup + measured,
                    f"{sid}: rank{rank} prefill count not 1-to-1 with requests")
        idx = [v.get("step_id") for v in selected]
        fail_unless(all(type(v) is int for v in idx) and idx == sorted(set(idx)),
                    f"{sid}: Rank{rank} step_id not strictly increasing")
        for v in selected:
            t = v.get("forward_to_logits_device_ms")
            fail_unless(v.get("schema") == EVENT_SCHEMA and
                        v.get("rank") == rank and
                        v.get("profiler") is False and
                        v.get("synchronization_inserted") is False and
                        type(t) in (float, int) and math.isfinite(t) and t > 0,
                        f"{sid}: invalid Rank{rank} event identity or timing")
        events[rank] = [float(x["forward_to_logits_device_ms"]) for x in selected[warmup:]]
    paired = [max(x, y) for x, y in zip(events[0], events[1])]
    return {"session_id": sid, "arm": arm,
            "identity": identity, "rank0_ms": events[0], "rank1_ms": events[1],
            "paired_critical_ms": paired, "p50_ms": statistics.median(paired),
            "mean_ms": statistics.mean(paired)}


def evaluate(root: Path, audit_path: Path, warmup: int = 6, measured: int = 30) -> dict:
    fail_unless(root.is_dir() and audit_path.is_file(), "missing actual run evidence")
    fail_unless(sorted(x.name for x in (root / "sessions").iterdir()) ==
                sorted(s for s, _ in SCHEDULE), "missing/extra ABBA sessions")
    items = [check_one_session(root, sid, arm, warmup, measured)
             for sid, arm in SCHEDULE]
    identity = items[0]["identity"]
    fail_unless(all(x["identity"] == identity for x in items),
                "model/device/SO identity drift across ABBA")
    audit = json.loads(audit_path.read_text())
    fail_unless(audit.get("schema") == "A1_L2_L3_CALIBRATION_ATTESTED_V1",
                "missing L2/L3 calibration audit")
    missing = [key for key in AUDIT_TRUE if audit.get(key) is not True]
    evid = audit.get("evidence_files")
    fail_unless(isinstance(evid, list) and len(evid) >= 4 and
                all(isinstance(x, str) and not Path(x).is_absolute()
                    and ".." not in Path(x).parts and
                    (root / x).is_file() and (root / x).stat().st_size > 0 for x in evid),
                "missing original site/trace/clock calibration evidence")
    pairs = []
    for stock, fused in ((items[0], items[1]), (items[3], items[2])):
        delta_ms = stock["p50_ms"] - fused["p50_ms"]
        pairs.append({"stock_ms": stock["p50_ms"], "v2_ms": fused["p50_ms"],
                      "gain_ms": round(delta_ms, 5)})
    stock_spread = abs(items[0]["p50_ms"] - items[3]["p50_ms"])
    stock_within_variance = max(
        (statistics.pstdev(items[0]["paired_critical_ms"]) if measured > 1 else 0),
        (statistics.pstdev(items[3]["paired_critical_ms"]) if measured > 1 else 0))
    device_event_resolution_ms = audit.get("event_resolution_ms")
    resolution_ok = (type(device_event_resolution_ms) in (int, float)
                     and 0 < device_event_resolution_ms < 1)
    noise_floor = max(stock_spread, stock_within_variance,
                      2 * device_event_resolution_ms if resolution_ok else float("inf"))
    gain_is_clear = min(x["gain_ms"] for x in pairs) > noise_floor
    good_audit = not missing and resolution_ok
    decision = ("L3_MEASUREMENT_NOT_READY" if not good_audit
                else "L3_GO" if gain_is_clear else "L3_INCONCLUSIVE_OR_NO_GAIN")
    return {
        "schema": "A1_L3_NO_PROFILER_ABBA_V1", "status": decision,
        "event_semantics": "current-stream model-forward entry to logits production; calibrated dependency required",
        "calibration_missing_or_false": missing,
        "four_serve_sessions": items, "pair_gains": pairs,
        "stock_cross_session_spread_ms": round(stock_spread, 5),
        "stock_within_session_pstdev_max_ms": round(stock_within_variance, 5),
        "noise_floor_ms": round(noise_floor, 5) if math.isfinite(noise_floor) else None,
        "limitations": [
            "Only one query per prefill device step is accepted; mixed M or chunking invalidates",
            "Full-step closure requires separate L2 device dependence/clock calibration",
            "No automatic L4: user has not authorized the formal 4096/2048 workload",
            "The ABBA verdict describes only this version/card/steady single-request load",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--measured", type=int, default=30)
    a = p.parse_args()
    if not all(v.is_absolute() for v in (a.root, a.audit, a.out)):
        p.error("all paths must be absolute")
    if a.warmup < 6 or a.measured < 30 or a.out.exists():
        p.error("invalid samples or existing output")
    try:
        r = evaluate(a.root, a.audit, a.warmup, a.measured)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        r = {"schema": "A1_L3_NO_PROFILER_ABBA_V1",
             "status": "L3_MEASUREMENT_NOT_READY", "error": str(exc)[:600]}
    a.out.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    print(r["status"])


if __name__ == "__main__":
    main()
