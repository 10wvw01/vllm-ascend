#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-C trace 解析器离线单测，不依赖 CANN / 服务器。"""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("g1c_trace_summary.py")


class TraceSummaryTests(unittest.TestCase):
    def run_trace(self, lines):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "g1c_rank0_pid101.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), tmp],
                capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout)["files"][0]

    @staticmethod
    def submit(seq=7):
        return (f"G1C_SUBMIT ms=2001 rank=0 seq={seq} m=4096 chunk_rows=1024 "
                "overlap=1 x=0x1 w=0x2 y=0x3")

    @staticmethod
    def snap(seq=7, phase=99, sample=3000):
        rows = " ".join(
            f"core{c}=47314331:{c}:{seq}:{phase}:3:48"
            for c in range(8)
        )
        return f"G1C_SNAP ms={sample} seq_latest={seq} rc=0 {rows}"

    def test_normal_complete_wave(self):
        data = self.run_trace([self.submit(), self.snap()])
        self.assertEqual(data["submit_count"], 1)
        self.assertEqual(data["snapshot_count"], 1)
        self.assertEqual(data["last_non_exit_core_count"], 0)
        self.assertEqual(len(data["last_snapshot"]["cores"]), 8)

    def test_kernel_wait_remains_incomplete(self):
        data = self.run_trace([self.submit(), self.snap(phase=40)])
        self.assertEqual(data["last_non_exit_core_count"], 8)

    def test_one_stale_core_warns_not_definitive_cause(self):
        lines = [self.submit(seq=7)]
        parts = [
            f"core{c}=47314331:{c}:{6 if c == 3 else 7}:99:3:12"
            for c in range(8)
        ]
        lines.append("G1C_SNAP ms=3000 seq_latest=7 rc=0 " + " ".join(parts))
        data = self.run_trace(lines)
        self.assertTrue(any("不同波" in s for s in data["warnings"]))

    def test_incomplete_sample_ignored(self):
        data = self.run_trace([self.submit(), self.snap(),
                               "G1C_SNAP ms=9000 seq_latest=8 rc=0 core"])
        self.assertEqual(data["last_sample_ms"], 3000)

    def test_host_log_without_device_sample_is_not_pass(self):
        data = self.run_trace([self.submit()])
        self.assertTrue(any("没有 G1C_SNAP" in s for s in data["warnings"]))

    def test_bad_magic_flags_invalid_core(self):
        line = self.snap().replace("core1=47314331", "core1=00000000")
        data = self.run_trace([self.submit(), line])
        self.assertTrue(any("7/8" in s for s in data["warnings"]))


if __name__ == "__main__":
    unittest.main()
