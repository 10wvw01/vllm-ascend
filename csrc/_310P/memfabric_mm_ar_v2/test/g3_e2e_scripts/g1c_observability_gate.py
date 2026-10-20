#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-C 观测通道准入：双 Rank 各 8 核、可见真实调用、不阻塞同步 D2H。"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
from g1c_trace_analyze import analyze, FNAME


def gate(directory: pathlib.Path) -> dict:
    files = list(directory.glob("g1c_dev*_rank*_pid*.jsonl"))
    failures = []
    summaries = {}
    ranks = set()
    for file in files:
        match = FNAME.fullmatch(file.name)
        if not match:
            continue
        rank = int(match.group("rank"))
        ranks.add(rank)
        data = analyze(file.read_text(encoding="utf-8", errors="replace"))
        summaries[file.name] = data
        if data["submit_count"] < 1:
            failures.append(file.name + ": no serving wave submit records")
        if data["polls_completed"] < 2:
            failures.append(file.name + ": fewer than 2 D2H reads")
        if data["polls_failed"] or data["polls_over_1000ms"]:
            failures.append(file.name + ": D2H failed or took over 1000ms")
        if data["intermediate_snapshots"] < 1:
            failures.append(file.name + ": 没有观察到仍在执行的中间阶段；不能证明运行中可读")
        if data["cores"]["0"] is not None and (
            data["unique_sequences_observed_by_core"]["0"] < 2
        ):
            failures.append(file.name + ": Core0 阶段序号未持续推进，需延长短测或调整采样")
        if data["unfinished_last_poll"]:
            failures.append(file.name + ": observer stuck in last D2H")
        if data["invalid_lines"] or data["errors"]:
            failures.append(file.name + ": corrupt/invalid trace")
        for core, item in data["cores"].items():
            if item is None or not item.get("seq") or item.get("matching_submit") is None:
                failures.append(f"{file.name}: core {core} missing valid seq/submit match")
    if ranks != {0, 1}:
        failures.append(f"expected both TP ranks 0/1, got {sorted(ranks)}")
    return {"verdict": "PASS" if not failures else "STOP",
            "failures": failures, "files": summaries,
            "note": "PASS 仅证明有限样本具备观测性；不能保证硬件超时后仍可读取 GM。"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=pathlib.Path)
    args = p.parse_args()
    if not args.directory.is_dir():
        p.error("G1-C 证据目录不存在")
    report = gate(args.directory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
