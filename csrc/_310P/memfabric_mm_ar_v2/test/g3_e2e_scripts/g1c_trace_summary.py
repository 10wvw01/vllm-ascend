#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""离线判读 G1-C 每核阶段戳；不能由最后阶段直接推导根因。"""
import argparse
import collections
import json
import pathlib
import re

PHASES = {
    1: "KERNEL_ENTER",
    10: "MM_ENTER", 11: "MM_EXIT",
    20: "IB_READY_ENTER", 21: "IB_READY_EXIT",
    30: "SIGNAL_ENTER", 31: "SIGNAL_RETURN",
    40: "WAIT_ENTER", 41: "WAIT_RETURN",
    42: "TAIL_WAIT_ENTER", 43: "TAIL_WAIT_RETURN",
    50: "IB_RELEASE_ENTER", 51: "IB_RELEASE_EXIT",
    60: "ADD_ENTER", 61: "ADD_EXIT",
    70: "IB_DONE_ENTER", 71: "IB_DONE_EXIT",
    80: "QUIET_ENTER", 81: "QUIET_RETURN",
    90: "DONE_SIGNAL_ENTER", 91: "DONE_SIGNAL_RETURN",
    92: "DONE_WAIT_ENTER", 93: "DONE_WAIT_RETURN",
    99: "KERNEL_EXIT",
}
MAGIC = 0x47314331
SUBMIT = re.compile(
    r"^G1C_SUBMIT ms=(\d+) rank=(\d+) seq=(\d+) m=(\d+) "
    r"chunk_rows=(\d+) overlap=(\d+) x=([^ ]+) w=([^ ]+) y=([^ ]+)$"
)
SNAP = re.compile(r"^G1C_SNAP ms=(\d+) seq_latest=(\d+) rc=(-?\d+)(.*)$")
ROW = re.compile(r"\bcore(\d+)=([0-9a-fA-F]{8}):(\d+):(\d+):(\d+):(\d+):(\d+)")


def analyze(path: pathlib.Path) -> dict:
    submissions = []
    snapshots = []
    read_errors = 0
    bind_errors = 0
    last_read_begin = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("G1C_BIND") and not line.endswith("rc=0"):
            bind_errors += 1
        if line.startswith("G1C_PRE_READ"):
            last_read_begin = line
        match = SUBMIT.fullmatch(line)
        if match:
            t = match.groups()
            submissions.append({
                "ms": int(t[0]), "rank": int(t[1]), "seq": int(t[2]),
                "m": int(t[3]), "chunk_rows": int(t[4]), "overlap": int(t[5]),
                "x": t[6], "w": t[7], "y": t[8],
            })
            continue
        match = SNAP.fullmatch(line)
        if not match:
            continue
        ms, latest, rc, tail = match.groups()
        if int(rc):
            read_errors += 1
        cores = {}
        for m in ROW.finditer(tail):
            core, magic, recorded_core, seq, phase, chunk, step = m.groups()
            core_id = int(core)
            cores[core_id] = {
                "valid": int(magic, 16) == MAGIC and int(recorded_core) == core_id,
                "seq": int(seq), "phase": int(phase), "phase_name": PHASES.get(int(phase), "UNKNOWN"),
                "chunk": int(chunk), "step": int(step),
            }
        snapshots.append({
            "ms": int(ms), "submitted_seq": int(latest), "rc": int(rc),
            "cores": cores,
        })
    last = next((x for x in reversed(snapshots) if x["rc"] == 0 and x["cores"]), None)
    warnings = []
    if not submissions:
        warnings.append("没有 G1C_SUBMIT：未证明加载并运行了诊断 Kernel")
    if not snapshots:
        warnings.append("没有 G1C_SNAP：Host 观察者没有成功采样")
    if bind_errors:
        warnings.append(f"观察者绑定设备失败次数: {bind_errors}")
    if read_errors:
        warnings.append(f"D2H 读取异常次数: {read_errors}")
    if last:
        valid = {k: v for k, v in last["cores"].items() if v["valid"] and v["seq"] > 0}
        if len(valid) < 8:
            warnings.append(f"最后快照仅 {len(valid)}/8 核为有效阶段")
        seen = {v["seq"] for v in valid.values()}
        if len(seen) > 1:
            warnings.append("最后快照包含不同波的 Core 阶段，可能为核间不齐或采样非原子")
        if any(v["seq"] < last["submitted_seq"] for v in valid.values()):
            warnings.append("至少一个 Core 的已执行 seq 落后于 Host 最新发射 seq：需核查执行队列")
        if len(valid) == 8 and all(v["phase"] == 99 for v in valid.values()) and len(seen) == 1:
            warnings.append("最后可见波所有核均退出；故障可能出现在后续波或其他算子")
    uncompleted = [x for x in (last or {}).get("cores", {}).values()
                   if x["valid"] and x["phase"] != 99]
    return {
        "file": str(path),
        "submit_count": len(submissions),
        "last_submit": submissions[-1] if submissions else None,
        "snapshot_count": len(snapshots),
        "first_sample_ms": snapshots[0]["ms"] if snapshots else None,
        "last_sample_ms": snapshots[-1]["ms"] if snapshots else None,
        "read_errors": read_errors,
        "last_pre_read": last_read_begin,
        "last_snapshot": last,
        "last_non_exit_core_count": len(uncompleted),
        "warnings": warnings,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", type=pathlib.Path, help="MF310P_G1C_TRACE_DIR 对应目录")
    p.add_argument("--output", type=pathlib.Path, help="可选 JSON 输出")
    args = p.parse_args()
    if not args.trace_dir.is_dir():
        p.error("诊断目录不存在")
    files = sorted(args.trace_dir.glob("g1c_rank*_pid*.log"))
    results = [analyze(f) for f in files]
    output = {
        "files": results,
        "limitations": [
            "每核快照在不同内存位置，非跨核原子采样",
            "最后一条 SNAP 是最后成功观测到的状态，不必然等于设备首次阻塞时刻",
            "阶段 99 只表示 V2 Kernel 内核阶段已走完，不代表下一个异步 Stock Kernel 健康",
            "Host SUBMIT 可领先于设备实际执行，不能用 latest seq 证明设备已进入该波",
            "诊断埋点本身可能改变时序，需与非埋点对照分开记录",
        ],
        "status": "READY_FOR_MANUAL_REVIEW" if files else "NO_TRACE_FILES",
    }
    data = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    print(data, end="")
    if args.output:
        args.output.write_text(data, encoding="utf-8")


if __name__ == "__main__":
    main()
