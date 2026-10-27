# SPDX-License-Identifier: Apache-2.0
"""U1 NZ-Stock可选主参照：原row-major保持不变，严拒没有真实Trace的假NZ GO。"""
from __future__ import annotations
import ast
import csv
import json
import sys
from pathlib import Path

import pytest

PERF = Path(__file__).resolve().parents[1] / "perf"
sys.path.insert(0, str(PERF))
import u1_analyze as analyzer  # noqa: E402
import a1_nz_route_gate as nz  # noqa: E402
from test_u1_matrix_contract import fixture  # noqa: E402


def set_layout(root: Path, layout: str):
    manifest = root / "manifest.json"
    data = json.loads(manifest.read_text())
    data["stock_weight_layout"] = layout
    manifest.write_text(json.dumps(data))
    for p in (root / "sessions").glob("*/*rank?.json"):
        rank = json.loads(p.read_text())
        rank["stock_weight_layout"] = layout
        p.write_text(json.dumps(rank))


def test_nz_screen_keeps_separate_identity(tmp_path):
    fixture(tmp_path, scope="large", phase="screen")
    set_layout(tmp_path, "nz")
    result = analyzer.evaluate(tmp_path, "large", "screen")
    assert result["stock_weight_layout"] == "nz"
    assert result["nz_route_proof_required"] is True
    assert all(r["decision"] == "SCREEN_ONLY_NO_GO_POSSIBLE"
               for r in result["performance_map"])


def test_reject_mixed_weight_layout_rank(tmp_path):
    fixture(tmp_path, scope="large", phase="screen")
    set_layout(tmp_path, "nz")
    p = tmp_path / "sessions/01_large_stock_fp16/01_large_stock_fp16_rank0.json"
    row = json.loads(p.read_text())
    row["stock_weight_layout"] = "row_major"
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="stock_weight_layout"):
        analyzer.evaluate(tmp_path, "large", "screen")


def test_confirm_nz_requires_independent_profiler_route(tmp_path):
    fixture(tmp_path, scope="large", phase="confirm")
    set_layout(tmp_path, "nz")
    output = analyzer.evaluate(tmp_path, "large", "confirm")
    assert all(r["decision"] == "CANDIDATE_REQUIRES_ROUTE_AUDIT"
               for r in output["performance_map"])
    evidence = tmp_path / "route.json"
    (tmp_path / "trace.txt").write_text("synthetic CPU fixture only")
    (tmp_path / "trace2.txt").write_text("synthetic CPU fixture only")
    evidence.write_text(json.dumps({
        "stock_hccl_fp16_verified": True, "memfabric_route_verified": True,
        "hccl_stream_end_to_end_verified": True, "no_backend_coexistence": True,
        "evidence_files": ["trace.txt", "trace2.txt"],
    }))
    with pytest.raises(ValueError, match="NZ route"):
        analyzer.evaluate(tmp_path, "large", "confirm", evidence)


def generate_trace(path: Path, m: int):
    columns = ["Name", "Type", "Start Time(us)", "Duration(us)",
               "Input Shapes", "Output Shapes", "Stream ID"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for i in range(3):
            t = i * 6000
            w.writerow({"Name": "aclnnMatmulWeightNz_MatMulV3Common_MatMulV3",
                        "Type": "AI_CORE", "Start Time(us)": t,
                        "Duration(us)": 760, "Input Shapes": f'"{m},2048;128,128,16,16"',
                        "Output Shapes": f'"{m},2048"', "Stream ID": 1})
            w.writerow({"Name": "hcom_allReduce__503_1_1", "Type": "HCCL",
                        "Start Time(us)": t + 761, "Duration(us)": 1200,
                        "Input Shapes": f'"{m},2048"', "Output Shapes": f'"{m},2048"',
                        "Stream ID": 2})


def test_nz_trace_gate_requires_4_real_route_captures(tmp_path):
    for rank in (0, 1):
        for m in (2048, 4096):
            generate_trace(tmp_path / f"rank{rank}_m{m}_kernel_details.csv", m)
    report = nz.verify(tmp_path)
    assert report["status"] == "NZ_STOCK_ROUTE_VERIFIED"
    assert len(report["proofs"]) == 4
    csv_path = tmp_path / "rank1_m4096_kernel_details.csv"
    csv_path.write_text("not native csv")
    with pytest.raises(ValueError, match="native CANN"):
        nz.verify(tmp_path)


def test_nz_source_default_is_nonbreaking():
    src = (PERF / "u1_arm.py").read_text()
    runner = (PERF / "u1_run.sh").read_text()
    ana = (PERF / "u1_analyze.py").read_text()
    for path in ("u1_arm.py", "u1_analyze.py", "a1_nz_route_gate.py"):
        ast.parse((PERF / path).read_text())
    assert 'os.environ.get("U1_STOCK_LAYOUT", "nz" if a.scope == "large" else "row_major")' in src
    assert 'stock_weight = w_nz if a.stock_weight_layout == "nz" else w' in src
    assert "F.linear(x, stock_weight)" in src
    assert "export U1_STOCK_LAYOUT=" in runner
    assert "nz_stock_route_report" in ana
    probe = (PERF / "a1_nz_stock_profile_probe.py").read_text()
    ast.parse(probe)
    assert 'dist.init_process_group(backend="hccl")' in probe
    assert 'dist.new_group(backend="gloo")' in probe
    assert "with profile(" in probe
    assert "not verified" not in src.lower()
