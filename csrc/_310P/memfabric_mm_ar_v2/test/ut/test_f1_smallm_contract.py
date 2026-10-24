# SPDX-License-Identifier: Apache-2.0
"""F1 小M筛选CPU静态测试（不需要NPU）。"""
from __future__ import annotations
import ast
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "test/perf"
AN = runpy.run_path(str(PERF / "f1_smallm_analyze.py"))


def make_log(*, bad_m: int | None = None, fail: bool = False) -> str:
    lines = ["G0\tib_probe\tPASS",
             "M\tMM\tADD\tstock_ms\tv2_med_ms\tv2_first_ms\tv2_p95_ms\tstatus\tbitexact"]
    for m in AN["REQUIRED_M"]:
        for mm in AN["MM"]:
            for add in AN["ADD"]:
                lines.append(f"{bad_m if bad_m and m == 10 else m}\t{mm}\t{add}"
                             f"\t0.300\t0.330\t0.390\t0.350\t0\t"
                             f"{'FAIL' if fail and m == 10 and mm == 4 and add == 1 else 'PASS'}")
    lines.append("MATRIX ALL PASS")
    return "\n".join(lines) + "\n"


def test_f1_m10_and_neighbors_required():
    a = AN["parse"](make_log())
    assert [x["M"] for x in a["matrix"]] == [1, 8, 10, 16, 32, 64]
    assert a["matrix"][2]["best_delta_pct"] < 0
    assert a["verdict"].startswith("SCREEN_ONLY")
    runner = (PERF / "f1_run_smallm.sh").read_text()
    assert "--rows 1 8 10 16 32 64" in runner
    assert "--mm-cores 4 8" in runner
    assert "--add-cores 1 4 8" in runner
    assert "MF310P_V2_NOWARMUP" in runner and "MF_KFC_BYPASS" in runner


def test_cannot_hide_shape_or_correctness_failures():
    for s in (make_log(bad_m=9), make_log(fail=True),
              make_log().replace("G0\tib_probe\tPASS", ""),
              make_log().replace("0.330", "nan", 1),
              make_log().replace("MATRIX ALL PASS", "")
              ):
        with pytest.raises(ValueError):
            AN["parse"](s)


def test_t1_driver_retains_stock_before_pool_and_exact_check():
    src = (ROOT / "test/t1_nz_direct_trial.py").read_text()
    ast.parse(src)
    assert "M\tMM\tADD" in src or '"M\\tMM\\tADD' in src
    assert "if args.mode == \"full\"" in src or 'elif args.mode == "full"' in src
    assert "stock_ms[m]" in src
    assert "torch.equal" in src
    assert "trial.check_status()" in src
