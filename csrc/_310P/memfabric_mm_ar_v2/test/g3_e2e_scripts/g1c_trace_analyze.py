#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-C 飞行日志离线判读：阶段戳与 Host 提交序号对齐，不把旧波残留视作阻塞。"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import re

MAGIC = 0x4731435048415345
PHASES = {
    1: "ENTRY", 10: "MM_ENTER", 11: "MM_EXIT",
    20: "READY_ENTER", 21: "READY_EXIT",
    30: "SIGNAL_ENTER", 31: "SIGNAL_EXIT",
    40: "WAIT_ENTER", 41: "WAIT_EXIT",
    50: "RELEASE_ENTER", 51: "RELEASE_EXIT",
    60: "ADD_ENTER", 61: "ADD_EXIT",
    70: "ADD_DONE_ENTER", 71: "ADD_DONE_EXIT",
    80: "QUIET_ENTER", 81: "QUIET_EXIT",
    90: "DONE_SIGNAL_ENTER", 91: "DONE_SIGNAL_EXIT",
    100: "DONE_WAIT_ENTER", 101: "DONE_WAIT_EXIT",
    110: "EXIT",
}
FNAME = re.compile(r"g1c_dev(?P<dev>\d+)_rank(?P<rank>\d+)_pid(?P<pid>\d+)\.jsonl\Z")


def analyze(text: str) -> dict:
    submits = {}
    latest = {}
    progressed = collections.defaultdict(set)
    polls = []
    errors = []
    all_snapshots = 0
    intermediate_snapshots = 0
    invalid_lines = 0
    dangling_polls = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        kind = obj.get("type")
        if kind == "submit":
            seq = obj.get("seq")
            if isinstance(seq, int) and seq > 0:
                submits[seq] = {k: obj.get(k) for k in (
                    "time_ms", "rows", "padded", "chunk_rows", "weight",
                    "output", "slot", "stream", "rank")}
        elif kind == "poll_begin":
            polls.append({"time_ms": obj.get("time_ms"), "rc": None,
                          "duration_ms": None, "line": line_no})
        elif kind == "poll_end":
            if polls and polls[-1]["rc"] is None:
                polls[-1]["rc"] = obj.get("rc")
                polls[-1]["duration_ms"] = obj.get("duration_ms")
            elif obj.get("rc") != 0:
                errors.append({"type": "orphan_poll_error", "line": line_no})
        elif kind == "snap":
            if obj.get("magic") != MAGIC:
                continue
            core, seq = obj.get("core"), obj.get("seq")
            if not isinstance(core, int) or not 0 <= core < 8 or not isinstance(seq, int) or seq <= 0:
                invalid_lines += 1
                continue
            phase = obj.get("phase")
            if phase not in PHASES:
                invalid_lines += 1
                continue
            latest[core] = {k: obj.get(k) for k in (
                "time_ms", "seq", "chunk", "phase", "step")}
            latest[core]["phase_name"] = PHASES[phase]
            progressed[core].add(seq)
            all_snapshots += 1
            if phase != 110:
                intermediate_snapshots += 1
        elif kind == "invalid_slot":
            errors.append({"type": "invalid_slot_magic", "line": line_no, "core": obj.get("core")})
        elif kind == "invalid_snapshot":
            errors.append({"type": "torn_or_invalid_snapshot", "line": line_no,
                           "core": obj.get("core")})
        elif kind == "observer_start" and obj.get("device_rc") != 0:
            errors.append({"type": "observer_device_failure", "line": line_no})
    if polls and polls[-1]["rc"] is None:
        dangling_polls = 1
    complete_polls = [p for p in polls if p["rc"] is not None]
    slow_polls = [p for p in complete_polls
                  if isinstance(p["duration_ms"], int) and p["duration_ms"] > 1000]
    failed_polls = [p for p in complete_polls if p["rc"] != 0]
    last_seq = max(submits) if submits else 0
    last_time = max(
        [p["time_ms"] for p in complete_polls if isinstance(p["time_ms"], int)]
        + [item["time_ms"] for item in latest.values()
           if isinstance(item.get("time_ms"), int)],
        default=None,
    )
    # 只有最后一次已经持久化的可信快照可用于有限的“候选首停点”推断。
    uncertain = []
    if last_time is None:
        uncertain.append("no_complete_polls")
    if len(latest) != 8:
        uncertain.append("not_all_8_cores_observed")
    if not submits:
        uncertain.append("no_submit_metadata")
    if dangling_polls:
        uncertain.append("last_d2h_poll_incomplete")
    if invalid_lines or errors:
        uncertain.append("trace_integrity_warning")
    core_summary = {}
    for core in range(8):
        item = latest.get(core)
        if item is None:
            core_summary[str(core)] = None
            continue
        item = dict(item)
        item["matching_submit"] = submits.get(item["seq"])
        item["behind_latest_submit"] = max(0, last_seq - item["seq"])
        item["age_ms_from_last_poll"] = (
            last_time - item["time_ms"] if isinstance(item["time_ms"], int)
            and last_time is not None else None)
        core_summary[str(core)] = item
    return {
        "last_submitted_seq": last_seq,
        "submit_count": len(submits),
        "snapshots": all_snapshots,
        "intermediate_snapshots": intermediate_snapshots,
        "polls_started": len(polls),
        "polls_completed": len(complete_polls),
        "polls_failed": len(failed_polls),
        "polls_over_1000ms": len(slow_polls),
        "unfinished_last_poll": bool(dangling_polls),
        "invalid_lines": invalid_lines,
        "errors": errors[:30],
        "cores": core_summary,
        "unique_sequences_observed_by_core": {
            str(i): len(progressed[i]) for i in range(8)},
        "limitations": uncertain,
        "note": "最后一个阶段只是最后可观测进度；若 observer 停止或 D2H 被计算流阻塞，不可断言第一停点。",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=pathlib.Path)
    args = p.parse_args()
    files = sorted(args.directory.glob("g1c_dev*_rank*_pid*.jsonl"))
    if not files:
        p.error("没有发现 G1-C Worker 日志")
    report = {}
    for name in files:
        match = FNAME.fullmatch(name.name)
        if match:
            report[name.name] = analyze(name.read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
