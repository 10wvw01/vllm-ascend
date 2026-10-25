"""R2 admission inventory unit tests; never grants NPU execution."""
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "tools/310p_mm_ar/r2_static_admission.py"
spec = importlib.util.spec_from_file_location("r2_static_admission", PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class TestR2StaticAdmission(unittest.TestCase):
    def test_current_branch_remains_no_go(self):
        result = gate.audit(ROOT)
        self.assertEqual(result["verdict"], "STATIC-NO-GO")
        self.assertEqual(result["total"], 44)
        self.assertEqual(result["checks"]["T01"]["verdict"], "NO-GO")
        self.assertEqual(result["checks"]["ABI-15"]["verdict"], "NO-GO")
        self.assertEqual(result["checks"]["T16"]["matrix"], "BLOCKED")
        for key in ("T26", "T27", "T28", "T29"):
            self.assertEqual(result["checks"][key]["verdict"], "NO-GO")

    def test_discovered_extra_kernel_id_blocks_static_go(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            docs = root / gate.DOCS
            docs.mkdir(parents=True)
            (docs / "V2_TRITON_REPLACEMENT_MATRIX.md").write_text(
                "| T30 | discovered kernel | reachable | BLOCKED |\n"
            )
            (docs / "V2_API_ABI_MATRIX.md").write_text("")
            (docs / "R2_STATIC_ADMISSION_EVIDENCE.json").write_text(
                json.dumps({"upstream_sha": gate.UPSTREAM_SHA,
                            "device_scope": "ONDEVICE_NOT_RUN",
                            "entries": {}})
            )
            report = gate.audit(root)
            self.assertEqual(report["verdict"], "STATIC-NO-GO")
            self.assertTrue(any("untracked mandatory matrix IDs" in reason
                                for reason in report["reasons"]))

    def test_duplicate_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate matrix ID"):
            gate.parse_table(
                "| T01 | source | blocked | BLOCKED |\n"
                "| T01 | duplicate | blocked | READY |\n", "T"
            )

    def test_all_required_evidence_still_only_a_necessary_condition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            docs = root / gate.DOCS
            docs.mkdir(parents=True)
            (root / "impl.py").write_text("# test fixture\n")
            (root / "test_impl.py").write_text("# test fixture\n")
            for name, prefix in (
                ("V2_TRITON_REPLACEMENT_MATRIX.md", "T"),
                ("V2_API_ABI_MATRIX.md", "ABI"),
            ):
                rows = []
                for key in gate.REQUIRED:
                    if key.startswith(prefix):
                        state = "READY" if prefix == "T" else "PASS"
                        rows.append(f"| {key} | fixture | fixture | {state} |")
                (docs / name).write_text("\n".join(rows))
            entries = {
                key: {
                    "source": "impl.py", "test": "test_impl.py",
                    "runtime_wired": True, "cpu_contract_pass": True,
                    "device_ops_static_reviewed": True,
                    "review_round_1": "review-1-sha",
                    "review_round_2": "review-2-sha",
                }
                for key in gate.REQUIRED
            }
            (docs / "R2_STATIC_ADMISSION_EVIDENCE.json").write_text(json.dumps({
                "upstream_sha": gate.UPSTREAM_SHA,
                "device_scope": "ONDEVICE_NOT_RUN",
                "static_review_verdict": "PASS",
                "entries": entries,
            }))
            result = gate.audit(root)
            self.assertEqual(result["verdict"], "STATIC-CHECKS-COMPLETE")
            self.assertEqual(result["ready"], 44)
            # This is NOT a real static review or a device GO; only
            # checks the parser's necessary-condition mechanics.


if __name__ == "__main__":
    unittest.main()
