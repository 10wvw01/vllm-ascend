#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-D D1 Host/D2H/AI Core 日志离线摘要。只输出 OBSERVED，不根据空 SQ 推断成功写入。"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

MF = re.compile(r"\[MF_G1D\].*?devId:\s*(\d+).*?epoch:\s*(\d+)(.*)")
V2 = re.compile(r"\[V2_G1D\]\s+rank=(\d+)\s+generation=(\d+)(.*)")
FIELD = re.compile(r"\b([A-Za-z][A-Za-z0-9]*):\s*(0x[0-9a-fA-F]+|\d+)")
KV = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)=(0x[0-9a-fA-F]+|\d+)")


def analyze(lines):
    result = {
        "evidence_level": "OBSERVATION_ONLY",
        "limits": [
            "1Hz 读出为非原子多次 D2H，不能凭单帧判定持续丢写",
            "CPU D2H 与 AI Core 读取不一定采用同一缓存视角",
            "SQ 排空不能证明 SQE 正确落地",
            "源侧 quiet 和对端 ArrHead 不属于同一瞬时快照",
        ],
        "ranks": {},
    }
    by_rank = defaultdict(lambda: {"mf_frames": 0, "v2_returns": 0,
                                   "mf_errors": 0, "near_boundary": [],
                                   "mail_returns": [], "head_snapshots": [], "last_mf": {}})
    for i, line in enumerate(lines, 1):
        match = MF.search(line)
        if match:
            rank = match.group(1)
            r = by_rank[rank]
            if "snapshotError" in match.group(3):
                r["mf_errors"] += 1
                continue
            fields = {k: int(v, 0) if v.lower().startswith("0x") else int(v)
                      for k, v in FIELD.findall(match.group(3))}
            if "arrHead" not in fields:
                r["mf_errors"] += 1
                continue
            r["mf_frames"] += 1
            r["last_mf"] = {"line": i, "epoch": int(match.group(2)), **fields}
            head = fields["arrHead"]
            slot = head % 1024
            if "headSlot" in fields and "headStamp" in fields:
                r["head_snapshots"].append({
                    "line": i, "arrHead": head, "slot": fields["headSlot"],
                    "headStamp": fields["headStamp"],
                    "headMailSeq": fields.get("headMailSeq"),
                    "matches_expect": fields["headStamp"] == head + 1,
                    "note": "non-atomic D2H; no proof of persistent loss",
                })
                r["head_snapshots"] = r["head_snapshots"][-80:]
            if slot in {1020, 1021, 1022, 1023, 0, 1, 2}:
                key = "stamp" + str(slot)
                stamp = fields.get(key)
                r["near_boundary"].append({
                    "line": i, "arrHead": head, "slot": slot,
                    "expect": head + 1, "observed_stamp": stamp,
                    "matches_expect": (stamp == head + 1 if stamp is not None else None),
                    "note": "single non-atomic sample; no proof of loss",
                })
                r["near_boundary"] = r["near_boundary"][-80:]
        v = V2.search(line)
        if v:
            rank = v.group(1)
            f = {k: int(x, 0) if x.lower().startswith("0x") else int(x)
                 for k, x in KV.findall(v.group(3))}
            if f.get("captured") != 1:
                continue
            r = by_rank[rank]
            r["v2_returns"] += 1
            r["mail_returns"].append({"line": i, "generation": int(v.group(2)), **f})
            r["mail_returns"] = r["mail_returns"][-16:]
    result["ranks"] = dict(sorted(by_rank.items(), key=lambda x: int(x[0])))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("serve_log", type=Path)
    p.add_argument("-o", "--output", type=Path)
    a = p.parse_args()
    data = json.dumps(analyze(a.serve_log.read_text(errors="replace").splitlines()),
                      ensure_ascii=False, indent=2) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
