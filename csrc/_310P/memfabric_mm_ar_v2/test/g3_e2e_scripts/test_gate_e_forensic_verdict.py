#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate E 判定器离线测试：无需 310P3、CANN 或模型。"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("gate_e_forensic_verdict.py")


class VerdictTests(unittest.TestCase):
    def run_verdict(self, rounds=(), serve="healthy", runner="load done after 1 reps",
                    rc="0", prompts=25, write_runner=True, write_serve=True):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for rep, obj in rounds:
                (root / f"stability_rep{rep}.json").write_text(
                    json.dumps(obj) if isinstance(obj, dict) else obj, encoding="utf-8"
                )
            if write_serve:
                (root / "serve.log").write_text(serve, encoding="utf-8")
            if write_runner:
                (root / "runner.log").write_text(runner, encoding="utf-8")
            if rc is not None:
                (root / "runner_exit_code.txt").write_text(rc, encoding="utf-8")
            process = subprocess.run(
                [sys.executable, str(SCRIPT), str(root),
                 "--prompts-per-rep", str(prompts)],
                capture_output=True, text=True, check=False,
            )
            return process.returncode, json.loads(process.stdout)

    def test_complete_25_is_clean(self):
        rc, data = self.run_verdict([(1, {"completed": 25, "errors": 0})])
        self.assertEqual((rc, data["verdict"]), (0, "NO_OBSERVED_FAILURE"))

    def test_f01_partial_is_failure_even_runner_exit_zero(self):
        rc, data = self.run_verdict([
            (1, {"completed": 25, "errors": 0}),
            (2, {"completed": 4, "errors": 0}),
            (3, {"completed": 0, "errors": 0}),
        ])
        self.assertEqual((rc, data["verdict"]), (1, "FAIL"))
        self.assertEqual(data["completed"], 29)
        self.assertTrue(any("rep2" in x for x in data["failures"]))

    def test_explicit_request_error_fails(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 25, "errors": ["rpc error"]})])[0], 1
        )

    def test_timeout_fails_with_all_prompts_complete(self):
        rc, data = self.run_verdict(
            [(1, {"completed": 25, "errors": 0})],
            serve="aclnnSwiGlu failed, error code is 507014"
        )
        self.assertEqual((rc, data["verdict"]), (1, "FAIL"))

    def test_bench_subtest_failed_fails(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 25, "errors": 0})],
                             runner="bench rep 2 failed")[0], 1
        )

    def test_nonzero_exit_fails(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 25, "errors": 0})], rc="1")[0], 1
        )

    def test_no_result_unknown(self):
        self.assertEqual(self.run_verdict([])[0], 2)

    def test_missing_errors_unknown(self):
        self.assertEqual(self.run_verdict([(1, {"completed": 25})])[0], 2)

    def test_missing_rep_unknown(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 25, "errors": 0}),
                              (3, {"completed": 25, "errors": 0})])[0], 2
        )

    def test_corrupted_rep_unknown(self):
        self.assertEqual(self.run_verdict([(1, "{broken")])[0], 2)

    def test_missing_serve_log_unknown(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 25, "errors": 0})],
                             write_serve=False)[0], 2
        )

    def test_missing_runner_rc_unknown(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 25, "errors": 0})], rc=None)[0], 2
        )

    def test_stock_graph_50_prompt_rep(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 50, "errors": 0})], prompts=50)[0], 0
        )

    def test_stock_graph_50_prompt_partial(self):
        self.assertEqual(
            self.run_verdict([(1, {"completed": 49, "errors": 0})], prompts=50)[0], 1
        )


if __name__ == "__main__":
    unittest.main()
