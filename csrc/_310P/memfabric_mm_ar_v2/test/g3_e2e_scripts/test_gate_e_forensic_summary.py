#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""无 NPU 依赖的取证摘要测试：只验证错误分类和缺失数据保护。"""
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().with_name("gate_e_forensic_summary.py")


class GateESummaryTests(unittest.TestCase):
    def run_summary(self, json_text=None, runner_log="", serve_log="", rc="0"):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            if json_text is not None:
                (root / "stability_rep1.json").write_text(json_text, encoding="utf-8")
            (root / "serve.log").write_text(serve_log, encoding="utf-8")
            (root / "runner.log").write_text(runner_log, encoding="utf-8")
            (root / "runner_exit_code.txt").write_text(rc, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                text=True, capture_output=True, check=True
            )
            return result.stdout

    def test_no_results_is_unknown(self):
        self.assertIn("UNKNOWN", self.run_summary())

    def test_g3_772_errors_must_fail(self):
        output = self.run_summary('{"completed": 3, "errors": 772}')
        self.assertIn("已完成请求: 3", output)
        self.assertIn("FAIL", output)

    def test_missing_error_count_is_unknown(self):
        self.assertIn("UNKNOWN", self.run_summary('{"completed": 25}'))

    def test_clean_results_do_not_prove_root_cause_fixed(self):
        self.assertIn("NO_OBSERVED_FAILURE", self.run_summary('{"completed":25,"errors":0}'))

    def test_bench_failed_even_if_previous_rep_succeeded(self):
        self.assertIn("FAIL", self.run_summary(
            '{"completed":25,"errors":0}', runner_log="bench rep 2 failed"))

    def test_aicore_timeout_with_zero_json_errors(self):
        self.assertIn("FAIL", self.run_summary(
            '{"completed":25,"errors":0}', serve_log="The aicore execution times out"))

    def test_nonzero_runner_exit(self):
        self.assertIn("FAIL", self.run_summary('{"completed":25,"errors":0}', rc="1"))


if __name__ == "__main__":
    unittest.main()
