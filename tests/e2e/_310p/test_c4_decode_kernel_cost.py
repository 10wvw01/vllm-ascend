# SPDX-License-Identifier: Apache-2.0
"""CPU-only C4 归因合同测试：不代表设备Profiler已实际运行。"""
import csv
import json
import tempfile
from pathlib import Path

import pytest
from tests.e2e._310p.c4_decode_kernel_cost import SHAPE, aggregate, analyze, category


def proof(**kwargs):
    p = dict(SHAPE)
    p.update(phase="decode", decode_phase_verified=True,
             max_num_scheduled_tokens_per_request=1,
             running_requests=10, source_trace="trace_view.json",
             source_server_log="serve.log",
             rank_count=2, failed_requests=0)
    p.update(kwargs)
    return p


def dump_csv(p, rows):
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("Name", "Duration(us)"))
        w.writerows(rows)


def test_decode_categories_and_precedence():
    expected = {
        "npu_quant_grouped_matmul_dequant": "MOE_GROUPED_GEMM",
        "aclnnMoeInitRoutingV2": "MOE_ROUTE_COMBINE",
        "npu_swiglu": "MOE_SWIGLU_OR_OTHER",
        "aclnnCausalConv1d": "GDN_CONV",
        "aclnnRecurrentGatedDeltaRule": "GDN_RECURRENCE",
        "_npu_paged_attention": "FULL_ATTENTION",
        "HcclAllReduce": "TP_COMMUNICATION",
        "aclnnQuantMatmul": "QUANT_MATMUL_OR_CONVERT",
        "aclnnRmsNorm": "NORM",
    }
    for op, want in expected.items():
        assert category(op) == want, (op, want)


def test_aggregation_and_two_rank_safety():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        p = d / "proof.json"
        p.write_text(json.dumps(proof()), encoding="utf-8")
        r0, r1 = d / "rank0.csv", d / "rank1.csv"
        rows = [("npu_quant_grouped_matmul_dequant", "50")] * 12
        rows += [("aclnnRecurrentGatedDeltaRule", "20")] * 11
        rows += [("npu_paged_attention", "10")] * 12
        dump_csv(r0, rows)
        dump_csv(r1, list(reversed(rows)))
        result = analyze(p, r0, r1)
        assert result["window"]["actual_batch_verified"] == 10
        assert result["ranks"]["rank0"]["kernel_events"] == 35
        assert result["ranks"]["rank0"]["categories"][0]["category"] == "MOE_GROUPED_GEMM"
        assert result["verdict"] == "WORK_VOLUME_RANKING_ONLY_NOT_LATENCY_CAUSALITY"
        assert result["ranks"]["rank0"]["input_csv_sha256"] == result["ranks"]["rank1"]["input_csv_sha256"] is False


def test_refuse_wrong_shape_phase_failure_and_duplicate_rank():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        p = d / "proof.json"
        csv0 = d / "rank0.csv"
        dump_csv(csv0, [("HcclAllReduce", 1)] * 20)
        for drift in ({"output_tokens": 512},
                      {"input_tokens": 2048},
                      {"tp": 1},
                      {"running_requests": 8},
                      {"decode_phase_verified": False},
                      {"max_num_scheduled_tokens_per_request": 4},
                      {"failed_requests": 1}):
            p.write_text(json.dumps(proof(**drift)), encoding="utf-8")
            with pytest.raises(ValueError):
                analyze(p, csv0, csv0)
        p.write_text(json.dumps(proof()), encoding="utf-8")
        with pytest.raises(ValueError, match="different profiler"):
            analyze(p, csv0, csv0)


def test_reject_missing_duration_and_invalid_values():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        bad = d / "bad.csv"
        bad.write_text("Name,Duration(us)\nHcclAllReduce,-1\n")
        with pytest.raises(ValueError):
            aggregate(bad)
        bad.write_text("Name,Duration(us)\nHcclAllReduce,nan\n")
        with pytest.raises(ValueError):
            aggregate(bad)
