#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1大M完整算子基线离线验收与瓶颈分流（纯Python，无设备调用）。

原始双Rank记录必须由现有u1_analyze.evaluate()重新验收；screen不是GO，
confirm没有独立HCCL Event/路由证据不得授予Candidate。
历史Prefill与D4的设备耗时来自不同工况，仅独立列入背景，不参与运算。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from u1_analyze import evaluate
from u1_spec import LARGE_MS

FORMAL_SERVING_M = frozenset((2048, 4096))
# No measured new Stock comparison exists in the source branch as of 2026-10-26.
HISTORICAL = {
    "d4_internal": {
        "baseline_v2_add1024_event_p50_us": 2726.9,
        "optimized_v2_add2048_event_p50_us": 2556.3,
        "scope": "V2-to-V2 only, M4096, D4 standalone TP2 event/ABBA",
        "report": "V2_D4_ROUND1_ADD_TILE2048_ONDEVICE_REPORT_20261023.md",
    },
    "old_prefill_attribution": {
        "stock_mm_p50_us": 795,
        "stock_hccl_ar_p50_us": 1149,
        "v2_kernel_p50_us": 2573,
        "input_contiguous_per_site_us_approx": 66,
        "hardware_counters_rank0_p50_us": {
            "cube_v2": 458.9, "cube_stock_mm": 456.6,
            "mte2_v2": 1711.4, "mte2_stock_mm": 335.9,
            "scalar_v2": 872.8, "scalar_stock_mm": 186.5,
            "mte3_v2": 191.1, "mte3_stock_mm": 91.5,
        },
        "counter_caveat": "Hardware counters overlap; NEVER sum as phase duration or extrapolate to D4.",
        "scope": "2026-10-21 Serving Profiler, NOT simultaneous with D4",
        "report": "V2_PREFILL_ATTRIBUTION_REPORT_20261021.md",
    },
}


def _positive(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _load_identity(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("U1 manifest missing")
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = ("repo_head", "physical_devices", "device_so_sha256",
                "plugin_sha256", "host_so_sha256", "kver11_sha256")
    if any(not isinstance(raw.get(k), str) or not raw[k] for k in required):
        raise ValueError("U1 identity fields missing")
    # U1 analyze separately verifies exact expected SHA fingerprints and sessions.
    if not raw["physical_devices"].count(",") == 1:
        raise ValueError("unknown physical device mapping")
    return {k: raw[k] for k in required}


def _rows(report: dict, phase: str) -> dict[int, dict]:
    if (report.get("experiment") != "U1_FULLCHAIN"
            or report.get("scope") != "large"
            or report.get("phase") != phase):
        raise ValueError("not U1 large-M fullchain " + phase)
    rows = report.get("performance_map")
    if not isinstance(rows, list) or len(rows) != len(LARGE_MS):
        raise ValueError("incomplete large-M measurement map")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("M")) is not int:
            raise ValueError("invalid M row")
        m = row["M"]
        if m in result or m not in LARGE_MS or row.get("route") != "memfabric_d4":
            raise ValueError("unknown/duplicate M or incorrect fused arm")
        stock = _positive(row.get("stock_p50_us"), "Stock complete-event")
        fused = _positive(row.get("fused_p50_us"), "V2 complete-event")
        if (type(row.get("stock_sessions")) is not int or
                type(row.get("fused_sessions")) is not int or
                row["stock_sessions"] < (4 if phase == "confirm" else 1) or
                row["fused_sessions"] < (4 if phase == "confirm" else 1)):
            raise ValueError("insufficient arm sessions")
        result[m] = row
    if set(result) != set(LARGE_MS):
        raise ValueError("missing one of U1 large M=1024/2048/4096")
    return result


def build_report(screen: dict, confirm: dict | None = None,
                 *, same_identity: bool = True) -> dict:
    """Consume ONLY U1 already-validated reports, never manufacture device data.

    Call evaluate() on actual evidence first. Standalone unit tests may supply
    synthetic reports to validate only math and fail-closed decision logic.
    """
    screen_rows = _rows(screen, "screen")
    confirm_rows = _rows(confirm, "confirm") if confirm else None
    if confirm_rows and not same_identity:
        raise ValueError("screen/confirm device/SO/repo identity changed")
    measurements = []
    for m in LARGE_MS:
        srow = screen_rows[m]
        crow = confirm_rows[m] if confirm_rows else None
        chosen = crow or srow
        stock = _positive(chosen["stock_p50_us"], "stock_p50_us")
        fused = _positive(chosen["fused_p50_us"], "fused_p50_us")
        if crow:
            decision = crow.get("decision")
            if decision not in (
                "NO_GO", "INCONCLUSIVE_DRIFT", "INCONCLUSIVE_STOCK_VARIATION",
                "CANDIDATE_REQUIRES_ROUTE_AUDIT",
                "OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS",
            ):
                raise ValueError("unrecognized U1 confirm verdict")
            for key in ("stock_p90_us", "fused_p90_us", "stock_p99_us", "fused_p99_us"):
                _positive(crow.get(key), key)
            windows = crow.get("abba_speedup_pct")
            if (not isinstance(windows, list) or len(windows) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in windows)):
                raise ValueError("two ABBA windows required")
            if decision == "OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS":
                if not confirm.get("audit_sha256"):
                    raise ValueError("cannot promote unaudited confirm")
                if (not all(v >= 5.0 for v in windows)
                        or fused >= stock
                        or crow["fused_p90_us"] > crow["stock_p90_us"] * 1.05
                        or crow["fused_p99_us"] > crow["stock_p99_us"] * 1.10
                        or crow.get("stock_session_drift_pct", 0) > 5.0):
                    raise ValueError("promoted result contradicts ABBA/tail/Stock evidence")
        else:
            decision = "SCREEN_ONLY_NO_GO"
        delta_us = round(fused - stock, 3)
        measurements.append({
            "M": m,
            "serving_route_eligible": m in FORMAL_SERVING_M,
            "measured_scope": "isolated U1 TP2 complete operator, not Serving",
            "measurement_phase": "confirm" if crow else "screen",
            "stock_complete_p50_us": round(stock, 3),
            "v2_complete_p50_us": round(fused, 3),
            "v2_extra_latency_us": delta_us,
            "v2_extra_latency_pct_of_stock": round(100 * delta_us / stock, 3),
            "v2_vs_stock_rate_pct": round(100 * (stock / fused - 1), 3),
            "verdict": decision,
            "priority_for_bottleneck": (
                "P0_REGRESSION" if delta_us > 0 and m in FORMAL_SERVING_M
                else "P1_EXPERIMENTAL_M" if m not in FORMAL_SERVING_M
                else "VERIFY_SERVING_TRANSFER"
            ),
            "scope_caveat": (
                "M1024 not enabled by current Serving threshold"
                if m == 1024 else "Nonaligned 2049..4095/tail staging not represented"
            ),
        })
    evidence_audited = bool(confirm_rows and confirm.get("audit_sha256"))
    eligible_rows = [r for r in measurements if r["serving_route_eligible"]]
    operator_candidate = (evidence_audited and all(
        r["verdict"] == "OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS"
        for r in eligible_rows
    ))
    regressions = sorted(
        (x for x in measurements if x["serving_route_eligible"]
         and x["v2_extra_latency_us"] > 0),
        key=lambda x: x["v2_extra_latency_us"], reverse=True,
    )
    return {
        "contract": "A1_LARGE_M_FULL_OPERATOR_V1",
        "data_origin": ("U1 real-evidence revalidation required; "
                        "source historical timing never mixed into performance rows"),
        "state": ("CONFIRM_REVIEWABLE_NOT_MODEL_GO" if confirm_rows
                  else "SCREEN_ONLY_NOT_GO"),
        "measurements": measurements,
        "largest_current_gap_M": regressions[0]["M"] if regressions else None,
        "historical_separate_reference": HISTORICAL,
        "investigation_queue": [
            {"order": 0, "target": "Stock HCCL cross-stream event and route audit (measurement prerequisite)",
             "why": "W1/W4 Stock timings were distorted by sync tax or back-to-back AR",
             "discriminator": "prove asynchronous HCCL completion is inside paired event window",
             "phase_duration_us": None},
            {"order": 1, "target": "SDMA arrival wait / MTE2 versus MM overlap",
             "why": "old Serving Profiler MTE2 1711us V2 vs 336us Stock MM; overlapping counter, NOT new D4",
             "discriminator": "approved formal-overlap vs loop4 + short TP2 device timeline",
             "phase_duration_us": None},
            {"order": 2, "target": "Core0 mail/signal and intercore IB protocol scalar path",
             "why": "old scalar 873us V2 vs 187us Stock; may include waits, not pure scalar cycles",
             "discriminator": "timestamped wait/ready/DONE critical path if safe instrumentation validated",
             "phase_duration_us": None},
            {"order": 3, "target": "real Serving input.contiguous and tail staging",
             "why": "historical Prefill profiler saw ~66us device input copy per site",
             "discriminator": "same Stock+DFlash model per-layer route and copy timeline",
             "phase_duration_us": None},
            {"order": 4, "target": "MM tiling and NZ8-core utilization",
             "why": "old Cube math 459us V2 vs 457us Stock, lower priority until current data changes",
             "discriminator": "same M independent Cube/format diagnostics, never subtract disparate medians",
             "phase_duration_us": None},
            {"order": 5, "target": "MTE3, Add and final protocol cleanup",
             "why": "old MTE3 191us V2 vs 92us Stock and Add Tile2048 already improved internal latency",
             "discriminator": "audited timeline only; do not equate full counter to nonoverlapped tail",
             "phase_duration_us": None},
        ],
        "gates": {
            "L1_complete_op_confirmed": bool(operator_candidate),
            "L1_independent_route_evidence_audited": evidence_audited,
            "L1_all_formal_M_performance_candidate": bool(operator_candidate),
            "L2_actual_qwen_or_dflash": False,
            "device_phase_timing_available": False,
            "known_stability_p0_closed": False,
            "requires_stock_dflash_runner_rebuild": False,
        },
        "warnings": [
            "U1 M1024 is diagnostic-only, never assume Serving routes M1024 to V2.",
            "D4 V2-to-V2 and 2026-10-21 Serving Profiler are different experiments.",
            "No subtracting non-equivalent kernel medians to allocate MM/SDMA/IB/Add time.",
            "Event durations require independent HCCL cross-stream inclusion audit.",
            "No claim of positive Stock speedup or actual DFlash model gain from screen.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen-root", type=Path, required=True)
    p.add_argument("--confirm-root", type=Path)
    p.add_argument("--route-audit", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.route_audit and not a.confirm_root:
        p.error("--route-audit requires --confirm-root")
    if not a.screen_root.is_dir() or (a.confirm_root and not a.confirm_root.is_dir()):
        p.error("existing absolute U1 evidence directories required")
    screen = evaluate(a.screen_root, "large", "screen")
    confirm = (evaluate(a.confirm_root, "large", "confirm", a.route_audit)
               if a.confirm_root else None)
    same = (not a.confirm_root
            or _load_identity(a.screen_root) == _load_identity(a.confirm_root))
    report = build_report(screen, confirm, same_identity=same)
    if a.out.exists() or not a.out.is_absolute():
        p.error("output must be a fresh absolute file path")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    print(f"A1 {report['state']} => {a.out}; no NPU or Serving executed")


if __name__ == "__main__":
    main()
