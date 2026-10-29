#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1大M离线归因合同测试；仅合成数学与fail-closed语义，不是设备性能。"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

PERF = Path(__file__).resolve().parents[1] / "perf"
sys.path.insert(0, str(PERF))
FILE = PERF / "a1_large_m_assess.py"


def load_module():
    spec = importlib.util.spec_from_file_location("a1_large_m_assess", FILE)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


M = load_module()


def evidence(phase="screen", *, audit=False, regression=True):
    data = []
    for m in (1024, 2048, 4096):
        stock = {1024: 900.0, 2048: 1500.0, 4096: 2000.0}[m]
        fused = stock + (100.0 if regression else -100.0)
        row = {
            "M": m, "route": "memfabric_d4",
            "stock_p50_us": stock, "fused_p50_us": fused,
            "stock_sessions": 4 if phase == "confirm" else 1,
            "fused_sessions": 4 if phase == "confirm" else 1,
            "decision": "CANDIDATE_REQUIRES_ROUTE_AUDIT"
                        if phase == "confirm" else "SCREEN_ONLY_NO_GO_POSSIBLE",
        }
        if phase == "confirm":
            row.update({
                "stock_p90_us": stock * 1.1,
                "fused_p90_us": fused * 1.1,
                "stock_p99_us": stock * 1.15,
                "fused_p99_us": fused * 1.15,
                "abba_speedup_pct": [5.5, 6.1],
            })
            if audit:
                row["decision"] = "OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS"
        data.append(row)
    return {
        "experiment": "U1_FULLCHAIN", "scope": "large",
        "phase": phase, "performance_map": data,
        "audit_sha256": ("sha256-dummy-for-unit-test-only" if audit else None),
    }


class TestA1LargeMOfflineContract(unittest.TestCase):
    def test_screen_never_grants_go(self):
        out = M.build_report(evidence())
        self.assertEqual(out["state"], "SCREEN_ONLY_NOT_GO")
        self.assertFalse(out["gates"]["L1_complete_op_confirmed"])
        self.assertFalse(out["gates"]["L2_actual_qwen_or_dflash"])
        self.assertEqual([x["M"] for x in out["measurements"]], [1024, 2048, 4096])
        self.assertTrue(all(x["verdict"] == "SCREEN_ONLY_NO_GO"
                            for x in out["measurements"]))

    def test_1024_cannot_claim_serving_eligibility(self):
        rows = M.build_report(evidence())["measurements"]
        self.assertFalse(rows[0]["serving_route_eligible"])
        self.assertTrue(rows[1]["serving_route_eligible"])
        self.assertTrue(rows[2]["serving_route_eligible"])

    def test_overhead_is_latency_not_throughput_percent(self):
        row = M.build_report(evidence())["measurements"][-1]
        self.assertEqual(row["v2_extra_latency_us"], 100.0)
        self.assertEqual(row["v2_extra_latency_pct_of_stock"], 5.0)
        self.assertAlmostEqual(row["v2_vs_stock_rate_pct"], -4.762, places=3)

    def test_positive_speedup_is_not_blindly_go(self):
        out = M.build_report(evidence(regression=False))
        self.assertEqual(out["state"], "SCREEN_ONLY_NOT_GO")
        self.assertIsNone(out["largest_current_gap_M"])
        self.assertTrue(all(x["verdict"] == "SCREEN_ONLY_NO_GO"
                            for x in out["measurements"]))
        self.assertGreater(out["measurements"][2]["v2_vs_stock_rate_pct"], 0)

    def test_confirm_without_audit_must_not_promote(self):
        out = M.build_report(evidence(), evidence("confirm"))
        self.assertEqual(out["state"], "CONFIRM_REVIEWABLE_NOT_MODEL_GO")
        self.assertEqual(out["measurements"][-1]["verdict"],
                         "CANDIDATE_REQUIRES_ROUTE_AUDIT")
        self.assertFalse(out["gates"]["L1_complete_op_confirmed"])

    def test_audited_confirm_only_operator_not_model_go(self):
        out = M.build_report(evidence(regression=False),
                             evidence("confirm", audit=True, regression=False))
        self.assertTrue(out["gates"]["L1_complete_op_confirmed"])
        self.assertFalse(out["gates"]["L2_actual_qwen_or_dflash"])
        self.assertEqual(out["measurements"][-1]["verdict"],
                         "OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS")

    def test_forged_promoted_confirm_rejected(self):
        c = evidence("confirm", audit=False)
        for row in c["performance_map"]:
            row["decision"] = "OPERATOR_CANDIDATE_ONLY_MODEL_GATE_REMAINS"
        with self.assertRaisesRegex(ValueError, "unaudited"):
            M.build_report(evidence(), c)

    def test_all_formal_must_pass_even_with_valid_route_audit(self):
        # M2048 has valid independent route evidence but is slower than Stock:
        # the other M must not turn this into an aggregate performance GO.
        screen = evidence(regression=False)
        confirm = evidence("confirm", audit=True, regression=False)
        confirm["performance_map"][1].update({
            "decision": "NO_GO",
            "fused_p50_us": 1600.0,
            "abba_speedup_pct": [-6.3, -6.2],
        })
        out = M.build_report(screen, confirm)
        self.assertTrue(out["gates"]["L1_independent_route_evidence_audited"])
        self.assertFalse(out["gates"]["L1_complete_op_confirmed"])
        self.assertEqual(out["measurements"][1]["verdict"], "NO_GO")

    def test_no_promoted_go_with_slow_kernel_or_bad_tail(self):
        for field in ("fused_p50_us", "fused_p90_us", "abba_speedup_pct"):
            confirm = evidence("confirm", audit=True, regression=False)
            row = confirm["performance_map"][1]
            if field == "fused_p50_us":
                row[field] = row["stock_p50_us"] + 100
            elif field == "fused_p90_us":
                row[field] = row["stock_p90_us"] * 1.25
            else:
                row[field] = [1.0, 2.0]
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "contradicts"
            ):
                M.build_report(evidence(regression=False), confirm)

    def test_route_evidence_does_not_override_regression_in_all_shapes(self):
        confirm = evidence("confirm", audit=True, regression=True)
        for row in confirm["performance_map"]:
            row["decision"] = "NO_GO"
        result = M.build_report(evidence(), confirm)
        self.assertFalse(result["gates"]["L1_complete_op_confirmed"])
        self.assertTrue(result["gates"]["L1_independent_route_evidence_audited"])

    def test_identity_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity changed"):
            M.build_report(evidence(), evidence("confirm"), same_identity=False)

    def test_invalid_and_missing_shapes_fail_closed(self):
        for action in ("missing", "duplicate", "unknown", "wrong_route"):
            s = evidence()
            rows = s["performance_map"]
            if action == "missing":
                rows.pop()
            elif action == "duplicate":
                rows[1]["M"] = 1024
            elif action == "unknown":
                rows[2]["M"] = 8192
            else:
                rows[1]["route"] = "memfabric_t1"
            with self.subTest(action=action), self.assertRaises(ValueError):
                M.build_report(s)

    def test_nonfinite_or_boolean_times_rejected(self):
        for invalid in (float("nan"), float("inf"), -1, 0, True, "2000"):
            s = evidence()
            s["performance_map"][1]["fused_p50_us"] = invalid
            with self.subTest(value=str(invalid)), self.assertRaises(ValueError):
                M.build_report(s)

    def test_confirm_tail_and_abba_windows_required(self):
        c = evidence("confirm")
        c["performance_map"][0].pop("fused_p99_us")
        with self.assertRaises(ValueError):
            M.build_report(evidence(), c)
        c = evidence("confirm")
        c["performance_map"][0]["abba_speedup_pct"] = [5.0]
        with self.assertRaises(ValueError):
            M.build_report(evidence(), c)

    def test_data_scope_and_historical_separation(self):
        s = M.build_report(evidence())
        self.assertEqual(s["historical_separate_reference"]["d4_internal"]
                         ["optimized_v2_add2048_event_p50_us"], 2556.3)
        self.assertTrue(all(x["measured_scope"].startswith("isolated U1")
                            for x in s["measurements"]))
        self.assertTrue(all(q["phase_duration_us"] is None
                            for q in s["investigation_queue"]))
        self.assertFalse(s["gates"]["device_phase_timing_available"])

    def test_profiler_counters_are_old_and_never_claimed_as_phase_seconds(self):
        old = M.HISTORICAL["old_prefill_attribution"]
        self.assertEqual(old["hardware_counters_rank0_p50_us"]["cube_v2"], 458.9)
        self.assertEqual(old["hardware_counters_rank0_p50_us"]["mte2_v2"], 1711.4)
        self.assertEqual(old["hardware_counters_rank0_p50_us"]["scalar_v2"], 872.8)
        self.assertIn("NEVER sum", old["counter_caveat"])
        q = M.build_report(evidence())["investigation_queue"]
        self.assertEqual(q[0]["order"], 0)
        self.assertEqual(q[1]["target"], "SDMA arrival wait / MTE2 versus MM overlap")
        self.assertTrue(all(item["phase_duration_us"] is None for item in q))

    def test_wrong_experiment_or_phase_rejected(self):
        for change in ("experiment", "scope", "phase"):
            s = evidence()
            s[change] = "wrong"
            with self.subTest(change=change), self.assertRaises(ValueError):
                M.build_report(s)

    def test_source_scope_no_runner_or_dflash_rebuild(self):
        source = FILE.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("torch_npu", source)
        self.assertNotIn("subprocess.run", source)
        self.assertNotIn("mf310p_v2_trial_run(", source)
        self.assertIn("from u1_analyze import evaluate", source)
        self.assertIn("stock_complete_p50_us", source)


if __name__ == "__main__":
    unittest.main()
