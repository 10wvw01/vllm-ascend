#!/usr/bin/env python3
"""G1-C 只读离线证据汇总：不会从最后阶段直接推断根因。"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

V2_LINE = re.compile(r"\[V2_G1C\]\s+rank=(\d+)\s+changed=(\d+)(.*)")
CORE = re.compile(r"\bc([0-7])=(\d+):(\d+):(\d+)")
MF_LINE = re.compile(r"\[MF_G1C\].*?devId:\s*(\d+).*?epoch:\s*(\d+).*?valid:\s*(\d+)(.*)")
FIELD = re.compile(r"\b(phase|reqSeq|shadowTail|sqDepth|hwHeadLast|hwTailLast|queryHeadRcLast|queryTailRcLast|error):\s*(-?\d+)")

STAGES = {
    10: "entry", 20: "mm_before", 21: "mm_after", 30: "ib_ready_before",
    31: "ib_ready_after", 40: "signal_before", 41: "signal_after",
    50: "mail_wait_before", 51: "mail_wait_after",
    60: "add_release_before", 61: "add_release_after",
    70: "add_before", 71: "add_after", 80: "add_done_before",
    81: "add_done_after", 90: "quiet_before", 91: "quiet_after",
    100: "done_signal_before", 101: "done_signal_after",
    110: "done_wait_before", 111: "done_wait_after", 120: "exit",
}


def summarize(lines):
    ranks = defaultdict(lambda: {"v2_samples": 0, "v2_last": {}, "mf_samples": 0, "mf_last": {}, "invalid_mf": 0})
    errors = {"v2_snapshot_error": 0, "mf_snapshot_error": 0}
    for line_no, line in enumerate(lines, 1):
        if "[V2_G1C]" in line and "snapshot_error=" in line:
            errors["v2_snapshot_error"] += 1
        if "[MF_G1C]" in line and "snapshotError:" in line:
            errors["mf_snapshot_error"] += 1
        v = V2_LINE.search(line)
        if v:
            rank = v.group(1)
            core_entries = CORE.findall(v.group(3))
            # 只有 8 个核均有一个代际/阶段时才算作完整采样。
            if len(core_entries) != 8 or len({c for c, _, _, _ in core_entries}) != 8:
                continue
            item = ranks[rank]
            item["v2_samples"] += 1
            item["v2_last"] = {
                f"core{c}": {
                    "generation": int(g), "chunk": int(k), "stage": int(st),
                    "stage_name": STAGES.get(int(st), "unknown"),
                }
                for c, g, k, st in core_entries
            }
            item["v2_line"] = line_no
        m = MF_LINE.search(line)
        if m:
            rank = m.group(1)
            item = ranks[rank]
            if m.group(3) != "1":
                item["invalid_mf"] += 1
                continue
            fields = {key: int(val) for key, val in FIELD.findall(m.group(4))}
            item["mf_samples"] += 1
            item["mf_last"] = {"epoch": int(m.group(2)), **fields}
            item["mf_line"] = line_no
    return {
        "interpretation": "OBSERVATION_ONLY: a last-stage stamp alone cannot prove the first faulting kernel",
        "ranks": dict(sorted(ranks.items(), key=lambda t: int(t[0]))),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="保存的 serve.log（可合并双 rank）")
    parser.add_argument("-o", "--output", type=Path, help="可选 JSON 输出文件")
    args = parser.parse_args()
    report = summarize(args.log.read_text(errors="replace").splitlines())
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
