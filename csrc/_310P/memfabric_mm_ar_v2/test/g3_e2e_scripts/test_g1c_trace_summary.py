"""G1-C 离线汇总器的 CPU-only 行为回归。"""
import importlib.util
from pathlib import Path

_MODULE = Path(__file__).with_name("g1c_trace_summary.py")
_spec = importlib.util.spec_from_file_location("g1c_trace_summary", _MODULE)
_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parser)


def _v2_line(rank=0, generation=1, phase=120):
    cores = " ".join(
        f"c{i}={generation}:3:{phase if i == 0 else 120}" for i in range(8)
    )
    return f"[V2_G1C] rank={rank} changed=1 {cores}"


def test_last_stage_is_observation_not_root_cause():
    lines = [
        _v2_line(0, 14, 50),
        _v2_line(1, 14, 120),
        "[MF_G1C] devId: 0 epoch: 26 valid: 1 phase: 4 reqSeq: 99 "
        "shadowTail: 42 sqDepth: 1024 hwHeadLast: 38 hwTailLast: 42 "
        "queryHeadRcLast: 0 queryTailRcLast: 0 error: 0",
        "[MF_G1C] devId: 1 epoch: 26 valid: 0 phase: 0",
    ]
    report = _parser.summarize(lines)
    assert report["ranks"]["0"]["v2_last"]["core0"]["stage_name"] == "mail_wait_before"
    assert report["ranks"]["0"]["mf_last"]["hwHeadLast"] == 38
    assert report["ranks"]["1"]["invalid_mf"] == 1
    assert not report["ranks"]["1"]["mf_last"]
    assert "OBSERVATION_ONLY" in report["interpretation"]


def test_incomplete_v2_frame_does_not_count_as_evidence():
    report = _parser.summarize(["[V2_G1C] rank=0 changed=1 c0=2:0:50"])
    assert report["ranks"]["0"]["v2_samples"] == 0
    assert report["ranks"]["0"]["v2_last"] == {}


def test_sampler_errors_are_explicit():
    report = _parser.summarize([
        "[V2_G1C] rank=0 snapshot_error=507014",
        "[MF_G1C] devId: 0 epoch: 2 snapshotError: 507014",
    ])
    assert report["errors"] == {
        "v2_snapshot_error": 1, "mf_snapshot_error": 1,
    }
