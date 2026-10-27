# SPDX-License-Identifier: Apache-2.0
"""R2 D4机制诊断原始双Rank健全性CPU测试，不虚构Kernel内分段。"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf"))
from a1_r2_diag_assess import CASES, evaluate  # noqa: E402


def fixture(root: Path):
    for case in CASES:
        d=root/case
        d.mkdir(parents=True)
        for rank in (0,1):
            row={"experiment":"U1_LARGE_MECHANISM_DIAG",
                 "case":case,"rank":rank,
                 "M":1024 if case.startswith("diag") else 4096,
                 "chunks":1 if case.startswith("diag") else 4,
                 "chunk_rows":1024,"bit_exact":True,"status":0,
                 "profile":False, "iters":8,"warmup":3,
                 "samples":[{"i":i,"event_us":1000+rank*10+i,"wall_us":1100+i}
                            for i in range(8)]}
            (d/f"{case}_rank{rank}_result.json").write_text(json.dumps(row))
    return root


def test_five_cases_pass_exterior_only(tmp_path):
    r=evaluate(fixture(tmp_path))
    assert r["status"]=="DIAG_CASES_COMPLETE_INNER_PHASE_UNKNOWN"
    assert r["inner_phase_status"]=="INNER_PHASE_UNKNOWN"
    assert r["no_algebraic_phase_breakdown"] is True
    assert set(r["cases"])==set(CASES)


def test_reject_missing_rank(tmp_path):
    root=fixture(tmp_path)
    (root/"loop4/loop4_rank1_result.json").unlink()
    with pytest.raises(ValueError,match="missing loop4 rank1"):
        evaluate(root)


def test_reject_wrong_chunk_and_status(tmp_path):
    root=fixture(tmp_path)
    p=root/"diag4/diag4_rank0_result.json"
    d=json.loads(p.read_text())
    d["chunks"]=4
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="invalid chunks"):
        evaluate(root)
    d["chunks"]=1
    d["status"]=507899
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="invalid status"):
        evaluate(root)


def test_reject_missing_measurement(tmp_path):
    root=fixture(tmp_path)
    p=root/"formal/formal_rank0_result.json"
    d=json.loads(p.read_text())
    d["samples"][0]["event_us"]=0
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="bad sample"):
        evaluate(root)
