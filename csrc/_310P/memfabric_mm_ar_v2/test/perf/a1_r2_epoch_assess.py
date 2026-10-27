#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R2 15/5/5/15s Epoch周期对照纯CPU分析：复现/频率位移≠已证明根因。"""
from __future__ import annotations
import argparse
import json
import math
import re
import statistics
from pathlib import Path

ORDER = (("E15A", 15), ("E5A", 5), ("E5B", 5), ("E15B", 15))
SCHEMA = "A1_PREFILL_FORWARD_TO_LOGITS_EVENT_V1"


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def valid(x: object) -> bool:
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def session(root: Path, name: str, budget: int) -> dict:
    sub = root / "sessions" / name
    meta = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
    expected = {"session_id": name, "arm": "v2", "input_tokens": 4096,
                "output_tokens": 1, "concurrency": 1, "tp": 2,
                "eager": True, "profiler": False,
                "warmup_requests": 6, "measured_requests": 30,
                "epoch_budget_s": budget,
                "epoch_budget_source": "explicit_R2_controlled_env"}
    for k, v in expected.items():
        require(meta.get(k) == v, f"{name}: metadata mismatch {k}")
    require(isinstance(meta.get("identity"), dict), "missing software identity")
    events = []
    for rank in (0, 1):
        found = list(sub.glob(f"a1_prefill_rank{rank}_pid*.jsonl"))
        require(len(found) == 1, f"{name}: rank{rank} trace missing or multiple")
        rows = [json.loads(x) for x in found[0].read_text().splitlines() if x.strip()]
        rows = [x for x in rows if x.get("num_scheduled_tokens") == 4096]
        require(len(rows) == 36, f"{name}: rank{rank} expected 36 prefill events")
        for i, row in enumerate(rows):
            require(row.get("schema") == SCHEMA and row.get("rank") == rank
                    and row.get("profiler") is False
                    and row.get("synchronization_inserted") is False
                    and valid(row.get("forward_to_logits_device_ms")),
                    f"{name}: bad rank{rank} event[{i}]")
        events.append([x["forward_to_logits_device_ms"] for x in rows[6:]])
    req = [json.loads(x) for x in (sub / "client_requests.jsonl").read_text().splitlines()
           if x.strip()]
    measured = [x for x in req if x.get("phase") == "measured"]
    require(len(measured) == 30 and all(x.get("index") == i and
            x.get("session") == name and x.get("prompt_tokens") == 4096
            and x.get("completion_tokens") == 1 for i,x in enumerate(measured)),
            f"{name}: no exact 30 measured requests")
    last = None
    for item in measured:
        begin, end = item.get("begin_ns"), item.get("end_ns")
        require(type(begin) is int and type(end) is int and begin < end,
                f"{name}: invalid host monotonic clock")
        require(last is None or begin >= last, f"{name}: requests overlap or time rewinds")
        last = end
    # MemFabric监督器本身在ConfigureOrch日志中给出epochBudgetS。
    # 若BM_LOG_INFO被过滤，环境变量正确也不能冒充实际预算已生效。
    serve_log = sub / "serve.log"
    runtime_budget_verified = False
    budget_die_ids = []
    if serve_log.is_file():
        configured = [
            (int(dev), int(value)) for dev,value in re.findall(
                r"sdma orch configured, devId:\s*(\d+)\s+epochBudgetS:\s*(\d+)",
                serve_log.read_text(encoding="utf-8",errors="replace"))
        ]
        budget_die_ids = sorted(set(dev for dev,val in configured if val == budget))
        runtime_budget_verified = len(budget_die_ids) == 2 and all(
            val == budget for _,val in configured)
        if re.search(r"epoch relaunch failed|epoch exited with error|507014|507899",
                     serve_log.read_text(encoding="utf-8",errors="replace"), re.I):
            raise ValueError(f"{name}: unresolved device/Epoch runtime error in serve.log")
    paired = [max(a,b) for a,b in zip(*events)]
    med = statistics.median(paired)
    mad = statistics.median(abs(x-med) for x in paired)
    threshold = max(20., 6 * 1.4826 * mad)
    spikes = [i for i,x in enumerate(paired) if x-med > threshold]
    gaps = [(measured[j]["begin_ns"]-measured[i]["begin_ns"])/1e9
            for i,j in zip(spikes, spikes[1:])]
    return {"session": name, "epoch_budget_s": budget,
            "identity": meta["identity"],
            "paired_device_p50_ms": round(med,4),
            "spike_count": len(spikes), "spike_indices": spikes,
            "time_between_spikes_s": [round(g,3) for g in gaps],
            "spike_excess_ms": [round(paired[i]-med,3) for i in spikes],
            "rank_skew_at_spikes_ms": [
                round(abs(events[0][i]-events[1][i]),3) for i in spikes],
            "threshold_ms": round(threshold,4),
            "observed_epoch_boundary_in_device_or_host_log": False,
            "runtime_epoch_budget_verified": runtime_budget_verified,
            "runtime_configured_die_ids": budget_die_ids}


def evaluate(root: Path) -> dict:
    require(root.is_dir(), "epoch R2 batch root missing")
    sessions = root / "sessions"
    require(sessions.is_dir() and sorted(x.name for x in sessions.iterdir()) ==
            sorted(x for x,_ in ORDER), "incomplete ABBA four fresh Serve sessions")
    results = [session(root, name, budget) for name,budget in ORDER]
    identity = results[0]["identity"]
    require(all(r["identity"] == identity for r in results),
            "binary/model/physical Die identity drift across controlled ABBA")
    def median_gap(rows: list[dict]) -> float | None:
        vals = [x for row in rows for x in row["time_between_spikes_s"]]
        return statistics.median(vals) if len(vals) >= 4 else None
    gap15 = median_gap([results[0],results[3]])
    gap5 = median_gap([results[1],results[2]])
    all_config_verified = all(s["runtime_epoch_budget_verified"] for s in results)
    replicated = (all(len(r["spike_indices"]) >= 3 for r in results)
                  and gap15 is not None and gap5 is not None)
    shift = (replicated and 9 <= gap15 <= 21 and 2.5 <= gap5 <= 8
             and gap15 / gap5 > 1.8)
    return {"schema": "A1_R2_EPOCH_ABBA_V1",
            "status": ("EPOCH_RUNTIME_BUDGET_UNVERIFIED" if not all_config_verified
                       else "FREQUENCY_SHIFT_SUPPORTS_EPOCH_HYPOTHESIS_ONLY"
                       if shift else "EPOCH_FREQUENCY_SHIFT_NOT_PROVEN"),
            "session_order": [n for n,_ in ORDER],
            "observations": results,
            "median_spike_gap_budget15_s": round(gap15,4) if gap15 else None,
            "median_spike_gap_budget5_s": round(gap5,4) if gap5 else None,
            "runtime_epoch_budget_verified_all_four": all_config_verified,
            "actual_epoch_boundary_causality_proven": False,
            "next_evidence": (
                "Original KFC/AICPU epoch boundary transitions with time reference "
                "calibrated to the request monotonic clock, or controlled cross-"
                "correlation from genuine runtime events; absence is UNKNOWN"),
            "limitations": [
                "Only controlled budget and repeatable timing shift are screened, not actual epoch events.",
                "Request begin_ns uses host perf_counter_ns; cannot compare directly to device cycle timestamps.",
                "If epoch source log missing, never claim epoch is the cause, even if 15:5 frequency ratio looks similar.",
                "No formal 4096/2048 concurrency10 TTFT and no deployment GO."
            ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not a.root.is_absolute() or not a.out.is_absolute() or a.out.exists():
        p.error("absolute real root and new absolute output only")
    result = evaluate(a.root)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
        f.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
