# SPDX-License-Identifier: Apache-2.0
"""使用已提交真实20份站点提取数据，验证NPU外部耗时预算与拒绝缺证。"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "perf"))
from a1_r1_l2_budget import evaluate  # noqa: E402

EVIDENCE = ROOT / "evidence" / "a1_r1_batch_20261027"


def test_real_r1_nz_mm_and_fused_outlier():
    result = evaluate(EVIDENCE)
    assert result["status"] == "REAL_SITE_EXTERIOR_MEASURED_INNER_PHASES_UNKNOWN"
    assert result["per_rank"]["stock/rank0"]["n"] == 200
    assert 2100 < result["per_rank"]["stock/rank0"]["per_site_chain_p50_us"] < 2250
    assert 2400 < result["per_rank"]["v2/rank0"]["per_site_chain_p50_us"] < 2600
    assert 1800 < result["per_rank"]["stock/rank1"]["per_site_chain_p50_us"] < 2050
    spikes = result["fused_kernel_outliers_gt10ms"]
    assert len(spikes) >= 2
    assert {(row["rank"], row["step"], row["layer_index"]) for row in spikes} >= {
        (0, 0, 10), (1, 0, 10)}
    assert all("INNER" not in str(v) for v in spikes)


def test_reject_missing_site_evidence(tmp_path):
    (tmp_path / "s4_l2/analysis").mkdir(parents=True)
    bad = tmp_path / "s4_l2/analysis/stock_rank0_step0_sites.json"
    bad.write_text(json.dumps({"arm": "stock", "rank": 0, "step": 0, "sites": []}))
    with pytest.raises(ValueError, match="not 40"):
        evaluate(tmp_path)


def test_bad_status_does_not_pretend_pass(tmp_path):
    with pytest.raises(ValueError, match="evidence root"):
        evaluate(tmp_path / "doesnotexist")
