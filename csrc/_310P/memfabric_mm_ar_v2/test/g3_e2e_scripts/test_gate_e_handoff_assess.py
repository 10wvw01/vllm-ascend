#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate E 移交用离线判读逻辑的纯CPU、无模型、无设备回归。"""
import json
from pathlib import Path
import tempfile
import unittest

from gate_e_handoff_assess import assess, main


class GateEHandoffAssessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def write(self, name, value):
        (self.d / name).write_text(value, encoding="utf-8")

    def session(self, completed=25, errors=0, fusion=True, rc="0"):
        self.write("runner_exit_code.txt", rc)
        self.write("stability_rep1.json", json.dumps({"completed": completed, "errors": errors}))
        self.write("serve.log", "Enable 310P3 TP=2 MemFabric unquantized mm_ar\n" if fusion else "")
        self.write("runner.log", "complete")

    def test_dual_rank_507014_is_only_signature(self):
        self.session()
        self.write("serve.log", "Worker_TP0 507014\nWorker_TP1 507014\n")
        obj = assess(self.d, "fused")
        self.assertEqual(obj["verdict"], "FAULT_SIGNATURE_OBSERVED")
        self.assertEqual(obj["raw_507014_lines"], 2)
        self.assertFalse(obj["root_cause_confirmed"])

    def test_clean_fused_has_no_equivalence_proof(self):
        self.session()
        obj = assess(self.d, "fused")
        self.assertEqual(obj["verdict"], "NO_FAULT_OBSERVED_ROUTE_NOT_DEVICE_VERIFIED")
        self.assertFalse(obj["ondevice_verified"])

    def test_stock_fusion_contamination(self):
        self.session()
        self.assertEqual(assess(self.d, "stock")["verdict"], "INVALID_STOCK_CONTAMINATION")

    def test_no_fused_route(self):
        self.session(fusion=False)
        self.assertEqual(assess(self.d, "fused")["verdict"], "INCONCLUSIVE_NO_FUSION_SELECTION")

    def test_nonzero_runner(self):
        self.session(rc="1")
        self.assertEqual(assess(self.d, "fused")["verdict"], "FAIL_OR_INCOMPLETE")

    def test_unknown_bench_errors_field(self):
        self.session()
        self.write("stability_rep1.json", '{"completed":25}')
        self.assertEqual(assess(self.d, "fused")["verdict"], "INCONCLUSIVE_EVIDENCE")

    def test_corrupt_bench(self):
        self.session()
        self.write("stability_rep1.json", 'broken-json')
        obj = assess(self.d, "fused")
        self.assertEqual(obj["verdict"], "INCONCLUSIVE_EVIDENCE")
        self.assertEqual(len(obj["rejected_bench_files"]), 1)

    def test_bool_completed_rejected(self):
        self.session()
        self.write("stability_rep1.json", '{"completed":true,"errors":0}')
        self.assertEqual(assess(self.d, "stock")["verdict"], "INCONCLUSIVE_EVIDENCE")

    def test_missing_runner_exit_rejected(self):
        self.session()
        (self.d / "runner_exit_code.txt").unlink()
        self.assertEqual(assess(self.d, "fused")["verdict"], "INCONCLUSIVE_EVIDENCE")

    def test_output_does_not_overwrite(self):
        self.session()
        out = self.d / "assess.json"
        self.assertEqual(main([str(self.d), "--arm", "fused", "--out", str(out)]), 0)
        self.assertTrue(out.exists())
        self.assertEqual(main([str(self.d), "--arm", "fused", "--out", str(out)]), 2)


if __name__ == "__main__":
    unittest.main()
