# SPDX-License-Identifier: Apache-2.0
"""CPU-only：防止把有限Prefill算子预算误写成5%长输出Eager收益。"""
from pathlib import Path
import runpy
import pytest

P = Path(__file__).resolve().parents[1] / "perf" / "c3_roi_screen.py"
BUDGET = runpy.run_path(str(P))["budget"]


def test_sel01_actual_sample_prefill_only_limit():
    # SEL-01: 03-v2 50/50、输入4096、输出2048，2177.193s；
    # route-audit V2每Rank 3请求共120个V2TrialKernel，D4 Tile2048 paired p50约2556.3us。
    actual = BUDGET(
        requests=50, trace_requests=3, trace_kernel_count_per_rank=120,
        bench_duration_s=2177.19272222, baseline_kernel_us=2556.3,
    )
    assert actual["fusion_calls_per_request"] == 40
    assert actual["fusion_calls_whole_bench"] == 2000  # 不乘2；TP2 Rank并行
    assert 5.10 < actual["sum_prefill_fusion_kernel_time_s"] < 5.12
    assert 0.20 < actual["full_kernel_elimination_estimated_throughput_gain_pct"] < 0.25
    assert actual["fixed_profile"]["output_tokens"] == 2048


def test_100us_kernel_improvement_has_tiny_direct_eager_budget():
    v = BUDGET(
        requests=50, trace_requests=3, trace_kernel_count_per_rank=120,
        bench_duration_s=2177.19272222, baseline_kernel_us=2556.3,
        candidate_kernel_us=2456.3,
    )
    assert 0.19 < v["predicted_duration_reduction_s"] < 0.21
    assert 0 < v["predicted_throughput_gain_pct"] < 0.02


@pytest.mark.parametrize("changes", [
    {"trace_requests": 0},
    {"requests": -1},
    {"trace_kernel_count_per_rank": -1},
    {"trace_kernel_count_per_rank": 121},
    {"bench_duration_s": 0},
    {"bench_duration_s": 1},
    {"baseline_kernel_us": -10},
    {"candidate_kernel_us": 0},
])
def test_invalid_inputs_fail_closed(changes):
    args=dict(requests=50, trace_requests=3, trace_kernel_count_per_rank=120,
              bench_duration_s=2177.19272222, baseline_kernel_us=2556.3)
    args.update(changes)
    with pytest.raises(ValueError):
        BUDGET(**args)
