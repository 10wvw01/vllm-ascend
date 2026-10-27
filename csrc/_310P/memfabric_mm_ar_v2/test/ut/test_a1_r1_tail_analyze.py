# SPDX-License-Identifier: Apache-2.0
"""对真实A1-R1已归档事件做只读CPU回归；负向数据应拒绝。"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "perf"))
from a1_r1_tail_analyze import examine  # noqa: E402

L3 = ROOT / "evidence" / "a1_r1_batch_20261027" / "s5_l3" / "l3_result.json"


def real():
    return json.loads(L3.read_text(encoding="utf-8"))


def test_real_archived_r1_two_arms_periodic_correlated():
    actual = examine(real())
    assert actual["status"] == "REPRODUCIBLE_PERIODIC_V2_TAIL_CAUSE_UNKNOWN"
    assert actual["sessions"][0]["spike_indices_zero_based"] == []
    assert actual["sessions"][3]["spike_indices_zero_based"] == []
    for index in (1, 2):
        s = actual["sessions"][index]
        assert s["spike_indices_zero_based"] == [0, 6, 12, 19, 25]
        assert s["gap_indices"] == [6, 6, 7, 6]
        assert max(s["rank_skew_at_spikes_ms"]) < 0.1


def test_reject_unaligned_dual_rank():
    data = real()
    data["four_serve_sessions"][1]["rank0_ms"][0] += 15
    with pytest.raises(ValueError, match="not paired"):
        examine(data)


def test_missing_request_is_not_accepted():
    data = real()
    data["four_serve_sessions"][1]["rank1_ms"].pop()
    with pytest.raises(ValueError, match="30 complete"):
        examine(data)


def test_stock_outlier_does_not_prove_v2_only_period():
    data = real()
    st = data["four_serve_sessions"][0]
    st["rank0_ms"][0] += 120
    st["rank1_ms"][0] += 120
    st["paired_critical_ms"][0] += 120
    assert examine(data)["status"] == "PERIODIC_TAIL_NOT_PROVEN"


def test_same_pattern_but_rank_desync_does_not_prove_correlated_tail():
    data = real()
    fused = data["four_serve_sessions"][1]
    # Keep max(row) valid while making two Rank observations disagree.
    fused["rank1_ms"][0] -= 10
    assert examine(data)["status"] == "PERIODIC_TAIL_NOT_PROVEN"
