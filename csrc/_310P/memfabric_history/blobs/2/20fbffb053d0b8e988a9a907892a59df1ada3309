#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate E 交付前的离线证据完整性与故障字样判读；不访问 NPU、不签发设备 GO。"""
import argparse
import json
from pathlib import Path
import re
import sys

FUSION_SELECT = "Enable 310P3 TP=2 MemFabric unquantized mm_ar"
FUSION_RUNTIME = "V2 MM+MemFabric"
EXACT_FIRST = "first fused call verified bit-exact"
DEVICE_ERRORS = ("507014", "507899")


def _text(path):
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _nonnegative_int(obj):
    return type(obj) is int and obj >= 0


def _error_count(obj):
    if _nonnegative_int(obj):
        return obj
    if isinstance(obj, list):
        return len(obj)
    return None


def assess(directory, arm):
    """只产生预判，不能以单纯日志关键词替代双Rank设备Trace人工核验。"""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(d)
    serve = _text(d / "serve.log")
    bench = _text(d / "bench.log")
    runner = _text(d / "runner.log")
    logs = serve + "\n" + bench + "\n" + runner
    rc_file = d / "runner_exit_code.txt"
    rc = _text(rc_file).strip() if rc_file.exists() else None

    candidates = sorted(d.glob("stability_rep*.json"))
    valid = 0
    completed = 0
    errors = 0
    unknown_error_field = 0
    rejected = []
    for f in candidates:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(obj, dict) or not _nonnegative_int(obj.get("completed")):
                raise ValueError("missing/invalid completed")
            count = _error_count(obj.get("errors"))
            if count is None:
                unknown_error_field += 1
            else:
                errors += count
            completed += obj["completed"]
            valid += 1
        except (OSError, UnicodeError, ValueError, TypeError) as ex:
            rejected.append({"file": f.name, "error": str(ex)})

    counts = {err: len(re.findall(re.escape(err), logs, re.I)) for err in DEVICE_ERRORS}
    marks = {
        "fusion_selection_lines": serve.count(FUSION_SELECT),
        "fusion_runtime_lines": serve.count(FUSION_RUNTIME),
        "first_call_bitexact_lines": serve.count(EXACT_FIRST),
        "tp0_mentions": len(re.findall(r"Worker_TP0|rank[ _=:]*0", logs, re.I)),
        "tp1_mentions": len(re.findall(r"Worker_TP1|rank[ _=:]*1", logs, re.I)),
    }
    notes = ["设备上是否实际执行融合、双Rank是否同源超时、真实SO/KVER和恢复状态仍需人工审计。"]
    if counts["507014"] or counts["507899"]:
        verdict = "FAULT_SIGNATURE_OBSERVED"
        notes.append("故障字样已出现；先保全原始设备slog/plog/Rank，再判断是否复现同一阶段停滞。")
    elif not candidates or rejected or unknown_error_field or rc is None:
        verdict = "INCONCLUSIVE_EVIDENCE"
    elif rc != "0" or errors or re.search(r"bench rep [0-9]+ failed|serve exited early|health timeout", runner, re.I):
        verdict = "FAIL_OR_INCOMPLETE"
    elif completed == 0:
        verdict = "INCONCLUSIVE_NO_REQUESTS"
    elif arm == "stock" and (marks["fusion_selection_lines"] or marks["fusion_runtime_lines"]):
        verdict = "INVALID_STOCK_CONTAMINATION"
    elif arm == "fused" and not marks["fusion_selection_lines"]:
        verdict = "INCONCLUSIVE_NO_FUSION_SELECTION"
    else:
        verdict = "NO_FAULT_OBSERVED_ROUTE_NOT_DEVICE_VERIFIED"
        notes.append("本轮未见错误不代表已修复；fused选层日志不等于双Rank设备Kernel实际命中。")
    return {
        "schema": "gate-e-handoff-v1",
        "arm": arm,
        "verdict": verdict,
        "raw_507014_lines": counts["507014"],
        "raw_507899_lines": counts["507899"],
        "runner_exit_code": rc,
        "bench_files": len(candidates),
        "valid_bench_files": valid,
        "rejected_bench_files": rejected,
        "unknown_error_fields": unknown_error_field,
        "completed_requests": completed,
        "explicit_bench_errors": errors,
        "route_indicators": marks,
        "notes": notes,
        "ondevice_verified": False,
        "root_cause_confirmed": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence_dir", type=Path)
    p.add_argument("--arm", choices=("fused", "stock"), required=True)
    p.add_argument("--out", type=Path, help="机器可读JSON输出；只允许新路径，不覆盖原始证据")
    args = p.parse_args(argv)
    try:
        data = assess(args.evidence_dir, args.arm)
        serialized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8") as f:
                f.write(serialized)
        print(serialized, end="")
    except (OSError, ValueError) as exc:
        print(f"Gate E assess failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
