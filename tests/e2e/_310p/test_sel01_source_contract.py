# SPDX-License-Identifier: Apache-2.0
"""SEL-01 CPU-only fail-closed contracts (does not need torch_npu)."""
from __future__ import annotations
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MOD = runpy.run_path(str(ROOT / "sel01_analyze.py"))


def valid_record():
    return {
        "num_prompts": 50, "completed": 50, "failed": 0,
        "max_concurrency": 10, "input_lens": [4096] * 50,
        "output_lens": [2048] * 50, "errors": [],
        "output_throughput": 47.2, "median_ttft_ms": 12345.0,
        "p90_ttft_ms": 14000.0, "median_tpot_ms": 210.0,
        "p90_tpot_ms": 220.0, "median_itl_ms": 210.0,
    }


def test_strict_4096_input_2048_output():
    assert MOD["valid_result"](valid_record())["output"] == 2048
    for field, bad in (
        ("input_lens", [4095] + [4096] * 49),
        ("output_lens", [512] * 50),  # D4 数据必须拒绝
        ("completed", 49),
        ("failed", 1),
        ("max_concurrency", 16),
    ):
        row = valid_record()
        row[field] = bad
        try:
            MOD["valid_result"](row)
        except ValueError:
            pass
        else:
            raise AssertionError(f"missing fail-closed guard for {field}")


def test_route_logs_and_cross_order():
    f = MOD["check_route"]
    assert f("stock", "")["enabled_log_count"] == 0
    s = "Enable 310P3 TP=2 MemFabric unquantized mm_ar\n" * 40
    assert f("v1", s)["enabled_log_count"] == 40
    assert f("v2", s + "[mf310p-v2] M=4096 within [2048,4096]\n")["v2_execution_marker"]
    for arm, log in (("stock", s), ("v1", ""), ("v2", s)):
        try:
            f(arm, log)
        except ValueError:
            pass
        else:
            raise AssertionError(f"incorrectly accepted {arm}")
    assert MOD["ORDER"] == ["stock", "v1", "v2", "v2", "v1", "stock"] * 2


def test_driver_prohibits_wrong_shape_and_variable_drift():
    script = (ROOT / "sel01_three_arm_eager.sh").read_text()
    bench = (ROOT / "bench_memfabric_mm_ar_serve.sh").read_text()
    for key, value in (
        ("MEMFABRIC_MM_AR_E2E_INPUT_LEN", "4096"),
        ("MEMFABRIC_MM_AR_E2E_OUTPUT_LEN", "2048"),
        ("MEMFABRIC_MM_AR_E2E_CONCURRENCY", "10"),
        ("MEMFABRIC_MM_AR_E2E_MAX_NUM_SEQS", "10"),
        ("MEMFABRIC_MM_AR_E2E_DFLASH", "0"),
    ):
        assert f"export {key}={value}" in script
    assert "libmf_sdma_orch_v11.json" in script
    assert "--enforce-eager" in bench
    assert '--max-num-seqs "$MAX_NUM_SEQS"' in bench
    assert "MEMFABRIC_MM_AR_E2E_ORCH_JSON" in bench
    assert "torchrun" not in script  # Real vLLM Eager, not a standalone kernel-only microbenchmark
