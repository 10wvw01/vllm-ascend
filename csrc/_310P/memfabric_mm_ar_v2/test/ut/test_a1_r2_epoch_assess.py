# SPDX-License-Identifier: Apache-2.0
"""R2 Epoch 15/5/5/15纯CPU合成回归与拒绝性测试；不是NPU实测。"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf"))
from a1_r2_epoch_assess import ORDER, evaluate  # noqa: E402

def fixture(root: Path, short_gaps: bool = True):
    sessions = root / "sessions"
    sessions.mkdir(parents=True)
    ident = {"model":"Qwen3.6","physical_devices":"2,3",
             "host_so_sha256":"fixed","device_so_sha256":"d4",
             "plugin_so_sha256":"plugin","kver_sha256":"kver",
             "model_revision":"frozen"}
    for sid, budget in ORDER:
        d = sessions / sid
        d.mkdir()
        m={"session_id":sid,"arm":"v2","input_tokens":4096,"output_tokens":1,
           "concurrency":1,"tp":2,"eager":True,"profiler":False,
           "warmup_requests":6,"measured_requests":30,"identity":ident,
           "epoch_budget_s":budget,"epoch_budget_source":"explicit_R2_controlled_env"}
        (d / "manifest.json").write_text(json.dumps(m))
        spike_gap = (6 if budget == 15 or not short_gaps else 2)
        peaks = set(range(0,30,spike_gap))
        for rank in (0,1):
            rows=[]
            for i in range(36):
                idx=i-6
                v=2300.+rank+ (120. if idx in peaks and idx>=0 else 0.)
                rows.append({"schema":"A1_PREFILL_FORWARD_TO_LOGITS_EVENT_V1",
                            "rank":rank,"num_scheduled_tokens":4096,"step_id":i,
                            "forward_to_logits_device_ms":v,"profiler":False,
                            "synchronization_inserted":False})
            (d/f"a1_prefill_rank{rank}_pid111.jsonl").write_text(
                "\n".join(json.dumps(x) for x in rows)+"\n")
        req=[]
        for phase,n in (("warmup",6),("measured",30)):
            for i in range(n):
                idx=i if phase=="measured" else i-6
                begin=(idx+6)*2400000000
                req.append({"session":sid,"phase":phase,"index":i,
                           "prompt_tokens":4096,"completion_tokens":1,
                           "begin_ns":begin,"end_ns":begin+2350000000})
        (d/"client_requests.jsonl").write_text(
            "\n".join(json.dumps(x) for x in req)+"\n")
    return root


def test_budget_shift_is_timing_correlation_only(tmp_path):
    result = evaluate(fixture(tmp_path))
    assert result["status"] == "FREQUENCY_SHIFT_SUPPORTS_EPOCH_HYPOTHESIS_ONLY"
    assert result["actual_epoch_boundary_causality_proven"] is False
    assert all(x["spike_count"]>=4 for x in result["observations"])


def test_no_frequency_shift_does_not_pass(tmp_path):
    result = evaluate(fixture(tmp_path, short_gaps=False))
    assert result["status"] == "EPOCH_FREQUENCY_SHIFT_NOT_PROVEN"


def test_reject_wrong_epoch_or_missing_rank(tmp_path):
    root=fixture(tmp_path)
    manifest=root/"sessions/E5A/manifest.json"
    m=json.loads(manifest.read_text())
    m["epoch_budget_s"]=15
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="epoch_budget_s"):
        evaluate(root)
    m["epoch_budget_s"]=5
    manifest.write_text(json.dumps(m))
    (root/"sessions/E5A/a1_prefill_rank1_pid111.jsonl").unlink()
    with pytest.raises(ValueError,match="rank1"):
        evaluate(root)


def test_reject_identity_drift(tmp_path):
    root=fixture(tmp_path)
    m=root/"sessions/E15B/manifest.json"
    data=json.loads(m.read_text())
    data["identity"]["device_so_sha256"]="wrongD4"
    m.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="identity drift"):
        evaluate(root)


def test_reject_overlapping_requests(tmp_path):
    root=fixture(tmp_path)
    p=root/"sessions/E15A/client_requests.jsonl"
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    measurement=[i for i,x in enumerate(rows) if x["phase"]=="measured"]
    rows[measurement[3]]["begin_ns"]=rows[measurement[2]]["begin_ns"]+1
    p.write_text("\n".join(json.dumps(x) for x in rows))
    with pytest.raises(ValueError,match="overlap"):
        evaluate(root)
