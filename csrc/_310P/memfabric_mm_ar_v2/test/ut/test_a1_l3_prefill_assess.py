#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L3双RankABBA归因指标纯CPU合成测试（不是实机测量）。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PERF = Path(__file__).resolve().parents[1] / "perf"
spec = importlib.util.spec_from_file_location("a1_l3_prefill_assess", PERF / "a1_l3_prefill_assess.py")
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class L3EventAssessUT(unittest.TestCase):
    def evidence(self, path: Path):
        root = path / "batch"
        sessions = root / "sessions"
        sessions.mkdir(parents=True)
        for sid, arm in tool.SCHEDULE:
            sp = sessions / sid
            sp.mkdir()
            ident = {
                "model": "qwen", "model_revision": "frozen",
                "physical_devices": "2,3",
                "host_so_sha256": "a", "device_so_sha256": "b",
                "plugin_so_sha256": "c", "kver_sha256": "d",
            }
            (sp / "manifest.json").write_text(json.dumps({
                "session_id": sid, "arm": arm,
                "input_tokens": 4096, "output_tokens": 1, "concurrency": 1,
                "tp": 2, "eager": True, "profiler": False, "warmup_requests": 6,
                "measured_requests": 30, "identity": ident,
            }))
            for rank in (0, 1):
                v = 100.0 if arm == "stock" else 90.0
                if sid == "S2":
                    v = 100.5
                ev = []
                for i in range(36):
                    ev.append({
                        "schema": tool.EVENT_SCHEMA, "rank": rank, "step_id": i,
                        "num_scheduled_tokens": 4096,
                        "forward_to_logits_device_ms": v + .1 * rank,
                        "profiler": False, "synchronization_inserted": False,
                    })
                (sp / f"a1_prefill_rank{rank}_pid999.jsonl").write_text(
                    "\n".join(json.dumps(x) for x in ev) + "\n")
        for i in range(4):
            (root / f"device_trace_{i}.txt").write_text("real sample in synthetic test")
        (root / "l2_verified_report.json").write_text(json.dumps({
            "status": "L2_VERIFIED", "sites_per_rank_per_step": 40,
        }))
        audit = {
            "schema": "A1_L2_L3_CALIBRATION_ATTESTED_V1",
            "event_resolution_ms": .01,
            "l2_verified_report": "l2_verified_report.json",
            "evidence_files": [f"device_trace_{i}.txt" for i in range(4)],
            **{key: True for key in tool.AUDIT_TRUE},
        }
        audit_path = root / "l2_route_calibration.json"
        audit_path.write_text(json.dumps(audit))
        return root, audit_path

    def test_positive_signal_with_cross_session_matching(self):
        with tempfile.TemporaryDirectory() as d:
            root, audit = self.evidence(Path(d))
            obj = tool.evaluate(root, audit)
            self.assertEqual(obj["status"], "L3_GO")
            self.assertGreater(obj["pair_gains"][0]["gain_ms"], 9)

    def test_unverified_rank_alignment_or_dependency_is_not_go(self):
        with tempfile.TemporaryDirectory() as d:
            root, audit = self.evidence(Path(d))
            j = json.loads(audit.read_text())
            j["model_forward_to_logits_completed_dependencies_verified"] = False
            audit.write_text(json.dumps(j))
            self.assertEqual(tool.evaluate(root, audit)["status"], "L3_MEASUREMENT_NOT_READY")

    def test_missing_rank_and_unsynchronized_event_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root, audit = self.evidence(Path(d))
            f = root / "sessions" / "F2" / "a1_prefill_rank1_pid999.jsonl"
            f.unlink()
            with self.assertRaises(ValueError):
                tool.evaluate(root, audit)
            root, audit = self.evidence(Path(d) / "new")
            f = root / "sessions" / "F2" / "a1_prefill_rank1_pid999.jsonl"
            rows = [json.loads(x) for x in f.read_text().splitlines()]
            rows[0]["synchronization_inserted"] = True
            f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            with self.assertRaises(ValueError):
                tool.evaluate(root, audit)

    def test_stock_drift_blocks_go(self):
        with tempfile.TemporaryDirectory() as d:
            root, audit = self.evidence(Path(d))
            f = root / "sessions" / "S2" / "a1_prefill_rank0_pid999.jsonl"
            rows = [json.loads(x) for x in f.read_text().splitlines()]
            for row in rows:
                row["forward_to_logits_device_ms"] = 111.0
            f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            result = tool.evaluate(root, audit)
            self.assertEqual(result["status"], "L3_INCONCLUSIVE_OR_NO_GAIN")


if __name__ == "__main__":
    unittest.main()
