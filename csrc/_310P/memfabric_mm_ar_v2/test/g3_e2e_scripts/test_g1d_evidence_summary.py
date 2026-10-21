#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-D 离线证据解析器的 CPU-only 单元测试，无 NPU 依赖。"""
import unittest

from g1d_evidence_summary import analyze


class EvidenceSummaryTests(unittest.TestCase):
    def test_boundary_is_observation_not_missing_proof(self):
        lines = [
            "[2026-10-21] [MF_G1D] devId: 0 epoch: 5 arrHead: 1023 "
            "headSlot: 1023 headStamp: 0 headMailSeq: 1024 "
            "stamp1020: 1021 stamp1021: 1022 stamp1022: 1023 "
            "stamp1023: 0 stamp0: 0 stamp1: 0 stamp2: 0 "
            "mail1023Seq: 1024 mail1023Dst: 256 mail1023Len: 4096 "
            "mail1023Imm: 7 mail0Seq: 0 quiet60: 0 quiet61: 0 "
            "quiet62: 0 quiet63: 0 quiet0: 0 quiet1: 0"
        ]
        d = analyze(lines)
        r = d["ranks"]["0"]
        self.assertEqual(r["mf_frames"], 1)
        self.assertEqual(r["head_snapshots"][0]["slot"], 1023)
        self.assertEqual(r["head_snapshots"][0]["matches_expect"], False)
        self.assertEqual(r["near_boundary"][0]["slot"], 1023)
        self.assertEqual(r["near_boundary"][0]["observed_stamp"], 0)
        self.assertEqual(r["near_boundary"][0]["matches_expect"], False)
        self.assertIn("single non-atomic", r["near_boundary"][0]["note"])

    def test_wait_return_and_error(self):
        lines = [
            "[V2_G1D] rank=1 generation=208 captured=1 mailSeq=1023 "
            "waitChunk=1 mailDst=0xabc mailLen=4194304 mailImm=0x4d46 mailStatus=0",
            "[MF_G1D] devId: 1 epoch: 6 snapshotError: 0,507014,0,0,0",
        ]
        d = analyze(lines)
        self.assertEqual(d["ranks"]["1"]["v2_returns"], 1)
        self.assertEqual(d["ranks"]["1"]["mail_returns"][0]["mailDst"], 0xABC)
        self.assertEqual(d["ranks"]["1"]["mail_returns"][0]["waitChunk"], 1)
        self.assertEqual(d["ranks"]["1"]["mf_errors"], 1)

    def test_empty_evidence_must_be_unknown(self):
        d = analyze(["some other log line"])
        self.assertEqual(d["ranks"], {})
        self.assertEqual(d["evidence_level"], "OBSERVATION_ONLY")


if __name__ == "__main__":
    unittest.main()
