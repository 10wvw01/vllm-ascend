#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-B 离线设备日志示例测试。"""
import pathlib
import subprocess
import sys
import tempfile
import unittest
import json

SCRIPT = pathlib.Path(__file__).with_name("g1b_epoch_trace.py")


class EpochTraceTests(unittest.TestCase):
    def analyze(self, lines):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "serve.log"
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            r = subprocess.run([sys.executable, str(SCRIPT), str(log)],
                               capture_output=True, text=True, check=True)
            return json.loads(r.stdout)

    @staticmethod
    def tick(hb, head, tail, dev=0, valid=1):
        return (f"[MF_DIAG] tick epoch: 1 devId: {dev} hb: {hb} "
                f"reqHead: {head} reqTail: {tail} reqPending: {tail-head} "
                f"cursorsValid: {valid} exitReason: 4294967295 mainRet: 0")

    def test_empty_logs_give_no_false_diagnosis(self):
        result = self.analyze(["normal kernel execution"])
        self.assertEqual(result["devices"], {})

    def test_running_progress_is_not_backlog_stall(self):
        result = self.analyze([self.tick(t, t, t + 2) for t in range(1, 8)])
        self.assertFalse(result["devices"]["0"]["alerts"])

    def test_backlog_stall_is_only_suspected(self):
        result = self.analyze([self.tick(t, 7, 9) for t in range(1, 8)])
        alerts = result["devices"]["0"]["alerts"]
        self.assertEqual(alerts[0]["type"], "request_ring_backlog_no_head_progress")
        self.assertEqual(alerts[0]["samples"], 7)

    def test_zero_pending_not_a_stall(self):
        result = self.analyze([self.tick(t, 7, 7) for t in range(1, 8)])
        self.assertFalse(result["devices"]["0"]["alerts"])

    def test_epoch_error_is_distinct_from_sqe_error(self):
        result = self.analyze([
            "[MF_DIAG] start epoch: 1 devId: 0 sqId: 5",
            "[MF_DIAG] exit epoch: 1 devId: 0 durationMs: 16000 "
            "exitReason: 2 mainRet: 1360 consumed: 12 sqHeadAfter: 8 "
            "lastReqHead: 12 lastReqTail: 16"
        ])
        dev = result["devices"]["0"]
        self.assertEqual(dev["exit_reason_counts"], {"2": 1})
        self.assertEqual(dev["alerts"][0]["main_ret"], 1360)

    def test_invalid_cursor_not_proof_of_dropped_updates(self):
        result = self.analyze([self.tick(1, 10, 5, valid=0)])
        self.assertEqual(result["devices"]["0"]["alerts"][0]["type"], "invalid_cursor_snapshot")


if __name__ == "__main__":
    unittest.main()
