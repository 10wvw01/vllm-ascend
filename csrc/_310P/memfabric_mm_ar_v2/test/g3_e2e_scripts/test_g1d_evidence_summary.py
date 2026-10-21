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

    def test_tail_frame_write_landed_branch(self):
        # 环冻结于 arrHead=8632：t1=槽439，stamp==arrHead → 最后落地封写已落
        # （读可见）。若此时 AICore 等待仍不返回 → 读不可见分支。
        lines = [
            "[MF_G1D_TAIL] devId: 0 epoch: 30 arrHead: 8632 "
            "t1Slot: 439 t1Stamp: 8632 t1Seq: 8632 t1Dst: 0x1000 "
            "t1Len: 4194304 t1Imm: 0x4d46 "
            "t2Slot: 438 t2Stamp: 8631 t2Seq: 8631 t2Dst: 0x1000 "
            "t2Len: 4194304 t2Imm: 0x4d46 "
            "t3Slot: 437 t3Stamp: 8630 t3Seq: 8630 t3Dst: 0x1000 "
            "t3Len: 4194304 t3Imm: 0x4d46 "
            "t4Slot: 436 t4Stamp: 8629 t4Seq: 8629 t4Dst: 0x1000 "
            "t4Len: 4194304 t4Imm: 0x4d46",
        ]
        d = analyze(lines)
        t = d["ranks"]["0"]["last_tail"]
        self.assertEqual(d["ranks"]["0"]["tail_frames"], 1)
        self.assertTrue(t["t1_matches_arrHead"])
        self.assertEqual(t["t1_lag"], 0)
        self.assertEqual(t["t1Seq"], 8632)

    def test_tail_frame_write_loss_branch(self):
        # 同一冻结点：t1Stamp 仍持上圈旧值(7608) → 最后封写入从未落地。
        lines = [
            "[MF_G1D_TAIL] devId: 0 epoch: 30 arrHead: 8632 "
            "t1Slot: 439 t1Stamp: 7608 t1Seq: 7608 t1Dst: 0x1000 "
            "t1Len: 4194304 t1Imm: 0x4d46 "
            "t2Slot: 438 t2Stamp: 8631 t2Seq: 8631 t2Dst: 0x1000 "
            "t2Len: 4194304 t2Imm: 0x4d46 "
            "t3Slot: 437 t3Stamp: 8630 t3Seq: 8630 t3Dst: 0x1000 "
            "t3Len: 4194304 t3Imm: 0x4d46 "
            "t4Slot: 436 t4Stamp: 8629 t4Seq: 8629 t4Dst: 0x1000 "
            "t4Len: 4194304 t4Imm: 0x4d46",
        ]
        d = analyze(lines)
        t = d["ranks"]["0"]["last_tail"]
        self.assertFalse(t["t1_matches_arrHead"])
        self.assertEqual(t["t1_lag"], 1024)

    def test_tail_snapshot_error_counts(self):
        lines = [
            "[MF_G1D_TAIL] devId: 1 epoch: 3 snapshotError: 0,-1,0,0,0,0,0,0",
        ]
        d = analyze(lines)
        self.assertEqual(d["ranks"]["1"]["tail_errors"], 1)
        self.assertEqual(d["ranks"]["1"]["tail_frames"], 0)


if __name__ == "__main__":
    unittest.main()
