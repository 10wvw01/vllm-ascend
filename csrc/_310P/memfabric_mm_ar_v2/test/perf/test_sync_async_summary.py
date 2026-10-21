#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""无设备依赖的 ABBA 汇总测试。服务器可运行 python3 -m unittest ..."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_sync_async import extract, summarize


def make_log(arm: str, rows: int, speed: float, *, omit_rank=None) -> str:
    lines = []
    for rank in (0, 1):
        windows = (1, 4)
        for window in windows:
            lines.append(json.dumps({
                "tag": "V2_SYNC_ASYNC", "rank": rank, "phase": "result",
                "arm": arm, "rows": rows, "window": window,
                "steps": 40, "validated": True,
                "wall": {
                    "count": 8,
                    "median_ms": speed * (0.7 if window == 4 else 1) * (1 + rank * .05),
                    "p95_ms": speed * (0.7 if window == 4 else 1) * (1 + rank * .08),
                },
            }))
        if rank != omit_rank:
            lines.append(json.dumps({
                "tag": "V2_SYNC_ASYNC", "rank": rank, "phase": "complete",
                "arm": arm, "status": "PASS",
            }))
    return "\n".join(lines)


class SummaryTests(unittest.TestCase):
    def test_max_rank_is_used(self):
        a = extract(make_log("stock", 4096, 100.0),
                    rows=4096, arm="stock", windows={1, 4})
        self.assertAlmostEqual(a[1]["max_rank_median_ms"], 105)
        self.assertAlmostEqual(a[4]["max_rank_median_ms"], 73.5)

    def test_missing_rank_rejected(self):
        with self.assertRaises(ValueError):
            extract(make_log("v2", 4096, 90.0, omit_rank=1),
                    rows=4096, arm="v2", windows={1, 4})

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            extract(make_log("stock", 2048, 100.0),
                    rows=4096, arm="stock", windows={1, 4})

    def test_concatenated_rank_records_accepted(self):
        # 回归 2026-10-21 实机 smoke：两 Rank 共享 stdout fd 时，记录
        # 交错拼接在同一物理行，逐行 json.loads 会同时丢失两条记录。
        lines = make_log("stock", 4096, 100.0).splitlines()
        joined = []
        for i in range(0, len(lines), 2):
            joined.append(lines[i] + lines[i + 1]
                          if i + 1 < len(lines) else lines[i])
        text = "\n\n".join(joined) + "\n"
        a = extract(text, rows=4096, arm="stock", windows={1, 4})
        self.assertAlmostEqual(a[1]["max_rank_median_ms"], 105)
        self.assertAlmostEqual(a[4]["max_rank_median_ms"], 73.5)

    def test_abba_and_gain_sign(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for round_no in (1, 2):
                for arm, ms in (("stock", 100.0), ("v2", 80.0)):
                    (root / f"r{round_no}_{arm}_m4096.log").write_text(
                        make_log(arm, 4096, ms))
            result = summarize(root, [4096], {1, 4})
            self.assertAlmostEqual(result["rows"]["4096"]["1"]["median_gain_pct"], 20)
            self.assertTrue(result["rows"]["4096"]["4"]["both_positive"])
            self.assertAlmostEqual(
                result["rows"]["4096"]["window4_minus_window1_gain_pp"], 0)


if __name__ == "__main__":
    unittest.main()
