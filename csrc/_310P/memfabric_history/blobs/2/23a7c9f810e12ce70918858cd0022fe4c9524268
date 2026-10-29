#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-C 不依赖 NPU 的进度解析与相容性测试。"""
import importlib.util
import pathlib
import unittest

SCRIPT = pathlib.Path(__file__).with_name("g1c_trace_analyze.py")
SPEC = importlib.util.spec_from_file_location("g1c_trace_analyze", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MAGIC = MODULE.MAGIC


def line(t, **kwargs):
    import json
    return json.dumps({"type": t, **kwargs})


def valid_sample(seq=12, phase=40, core=0, tm=5000):
    return line("snap", time_ms=tm, core=core, magic=MAGIC,
                seq=seq, chunk=1, phase=phase, step=8)


class G1CTraceTests(unittest.TestCase):
    def test_empty_does_not_claim_first_stop(self):
        r = MODULE.analyze("")
        self.assertIn("no_complete_polls", r["limitations"])
        self.assertIsNone(r["cores"]["0"])

    def test_wave_and_core_are_linked(self):
        r = MODULE.analyze("\n".join([
            line("submit", time_ms=4000, seq=12, rank=0, rows=4096,
                 padded=4096, chunk_rows=1024, weight=999, output=888,
                 slot=2, stream=444),
            line("poll_begin", time_ms=5000),
            line("poll_end", time_ms=5010, rc=0, duration_ms=10),
            *(valid_sample(core=c) for c in range(8)),
        ]))
        self.assertEqual(r["cores"]["0"]["phase_name"], "WAIT_ENTER")
        self.assertEqual(r["cores"]["0"]["matching_submit"]["slot"], 2)
        self.assertEqual(r["polls_completed"], 1)
        self.assertEqual(r["snapshots"], 8)
        self.assertEqual(r["cores"]["0"]["behind_latest_submit"], 0)

    def test_stale_sample_detected(self):
        r = MODULE.analyze("\n".join([
            line("submit", seq=20), line("poll_begin", time_ms=6000),
            line("poll_end", time_ms=6010, rc=0, duration_ms=10),
            valid_sample(seq=12, phase=110),
        ]))
        self.assertEqual(r["cores"]["0"]["behind_latest_submit"], 8)
        self.assertIn("not_all_8_cores_observed", r["limitations"])

    def test_d2h_block_is_not_misdiagnosed_as_device_kernel_stall(self):
        r = MODULE.analyze("\n".join([
            line("submit", seq=12), line("poll_begin", time_ms=5000),
            valid_sample(),
        ]))
        self.assertTrue(r["unfinished_last_poll"])
        self.assertIn("last_d2h_poll_incomplete", r["limitations"])

    def test_invalid_magic_never_accepted(self):
        r = MODULE.analyze(line("snap", magic=123, seq=2, core=1, phase=40))
        self.assertEqual(r["snapshots"], 0)

    def test_phase_contract_matches_device_header(self):
        header = SCRIPT.parents[2] / "src/op_kernel/mf310p_v2_g1c_trace.h"
        src = header.read_text(encoding="utf-8")
        for name in ("kG1CWaitEnter", "kG1CSignalEnter", "kG1CQuietEnter",
                     "kG1CExit"):
            self.assertIn(name, src)
        self.assertIn("0x4731435048415345ULL", src)

    def test_production_kernel_is_not_instrumented(self):
        root = SCRIPT.parents[2] / "src/op_kernel"
        prod = (root / "mf310p_v2_kernel.asc").read_text(encoding="utf-8")
        diag = (root / "mf310p_v2_g1c_kernel.asc").read_text(encoding="utf-8")
        self.assertIn("V2TrialKernel(", prod)
        self.assertNotIn("G1CMark(", prod)
        self.assertIn("G1CTrialKernel(", diag)
        self.assertIn("G1CMark(", diag)


if __name__ == "__main__":
    unittest.main()
