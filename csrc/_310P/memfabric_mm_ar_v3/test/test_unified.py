# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from unified_metrics import CONTRACT, STAGES, decode_trace, paired_report, timed_call


def test_tensor_boundary_and_join():
    calls = []
    event = SimpleNamespace(record=lambda stream: calls.append("event"), elapsed_time=lambda other: 0.123)
    stream = SimpleNamespace(synchronize=lambda: calls.append("join"))
    ticks = iter([1000, 21000])
    tensor = object()
    out, metrics = timed_call(lambda: calls.append("call") or tensor, stream, (event, event), lambda: next(ticks))
    assert out is tensor and calls == ["event", "call", "event", "join"]
    assert metrics == {"wall_us": 20, "event_us": 123}


def trace(rows=512, comm=512, rank=0):
    raw = [0] * 1024
    raw[:7] = [0x41335432, rows, 512, comm, 99, rank, 3302]
    for c in range(8):
        raw[16 + c * 8 : 20 + c * 8] = [100, 900, 950, 1000]
    for c in range(rows // 64):
        raw[80 + c * 8 : 85 + c * 8] = [110, 200, 220, 260, 270]
    for g in range(max(rows // 512, rows // comm)):
        raw[592 + g * 16 : 598 + g * 16] = [280, 300, 310, 320, 400, 410]
    raw[720:725] = [500, 600, 650, 700, 800]
    return raw


def test_trace_geometry_monotonicity_and_raw_preservation():
    raw = trace()
    result = decode_trace(raw, 512, 512, 512, 99)
    assert result["seats"][0]["mm"] == 90
    assert result["communication"][0]["peer_ready_pump"] == 80
    assert result["tail"]["drain"] == 100
    assert result["raw"] == raw and not result["clock_cross_core_calibrated"]
    for index, value in [(0, 0), (4, 98), (81, 109), (724, 0)]:
        damaged = raw.copy()
        damaged[index] = value
        with pytest.raises(ValueError):
            decode_trace(damaged, 512, 512, 512, 99)


def ranks():
    a = {
        "contract": CONTRACT,
        "run_id": "test",
        "identity": {"sha": "frozen"},
        "rank": 0,
        "status": "PASS",
        "configuration": {"arm": "a3", "M": [2048], "comm_rows": [1024], "samples": 2, "compute_rows": 512},
        "cases": [],
    }
    for stage in STAGES:
        for repeat in range(2):
            a["cases"].append(
                {
                    "M": 2048,
                    "comm_rows": 1024,
                    "stage": stage,
                    "repeat": repeat,
                    "samples": [{"i": i, "correct": True, "wall_us": 100 + i, "event_us": 80 + i} for i in range(2)],
                }
            )
    for case in a["cases"]:
        for sample in case["samples"]:
            if case["stage"] == "host":
                sample["host_us"] = dict.fromkeys(("validate_reap", "reserve", "wrap", "enqueue", "input_event"), 1)
            elif case["stage"] == "device":
                sample["device"] = decode_trace(trace(2048, 1024), 2048, 512, 1024)
    b = copy.deepcopy(a)
    b["rank"] = 1
    for case in b["cases"]:
        for sample in case["samples"]:
            if case["stage"] == "device":
                sample["device"] = decode_trace(trace(2048, 1024, 1), 2048, 512, 1024)
    b["cases"][0]["samples"][0]["wall_us"] = 200
    return a, b


def test_pair_max_controls_and_fail_closed():
    a, b = ranks()
    result = paired_report(a, b)
    assert result["cases"][0]["wall_us"]["p50"] == 150.5
    assert result["configuration"] == a["configuration"] and result["identity"] == a["identity"]
    assert len(result["attribution"]) == 8
    assert not result["formal_performance_go"]
    for kind in ("missing", "duplicate", "bad_value", "wrong_contract", "incorrect"):
        a, b = ranks()
        if kind == "missing":
            b["cases"].pop()
        elif kind == "duplicate":
            b["cases"][-1] = b["cases"][0]
        elif kind == "bad_value":
            b["cases"][0]["samples"][0]["wall_us"] = float("nan")
        elif kind == "wrong_contract":
            b["contract"] = "old kernel only"
        else:
            b["cases"][0]["samples"][0]["correct"] = False
        with pytest.raises(ValueError):
            paired_report(a, b)


def test_trace_cacheline_ownership_and_no_default_trace_clock():
    source = (Path(__file__).parents[1] / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    assert "A3Kernel<tiling##M, M, MM, TR>" in source
    assert "bool TRACE = false" in source
    assert "16 + core * 8" in source and "80 + band * 8" in source
    assert "i = 592; i < 728; i += 8" in source
    assert "if (tr) { tr[80 + band * 8 + 4]" in source


def test_complete_campaign_controls_and_identity(monkeypatch):
    import importlib.util

    tools = Path(__file__).parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("campaign", tools / "unified_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reports = {}
    for label, arm, compute in mod.ORDER:
        comms = [1024] if arm in ("stock", "v2") else [512, 1024, 2048]
        reports[label] = {
            "contract": CONTRACT,
            "identity": {"manifest": "same"},
            "configuration": {"arm": arm, "compute_rows": compute, "M": [2048, 4096], "comm_rows": comms},
            "controls": [
                {
                    "M": m,
                    "comm_rows": c,
                    "metric": metric,
                    "centers": {"off-before": 1000, "off-after": 1000},
                    "stable_bracket": True,
                }
                for m in (2048, 4096)
                for c in comms
                for metric in ("wall_us", "event_us")
            ],
        }
    result = mod.compare(reports)
    assert len(result["comparisons"]) == 84 and not result["formal_performance_go"]
    assert all(x["comparisons"]["stock"]["stable"] for x in result["comparisons"])
    for damage in ("absent", "boundary", "geometry", "manifest"):
        broken = copy.deepcopy(reports)
        if damage == "absent":
            broken.pop("stock-after")
        elif damage == "boundary":
            broken["a3-c512"]["contract"] = "kernel-only"
        elif damage == "geometry":
            broken["a3-c512"]["configuration"]["M"] = [4096]
        else:
            broken["v2-after"]["identity"] = {"manifest": "different"}
        with pytest.raises(ValueError):
            mod.compare(broken)
