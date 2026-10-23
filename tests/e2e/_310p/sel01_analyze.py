#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SEL-01 严格校验和三臂汇总。绝不把 4096/512 或失败会话算作性能。"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import re
import statistics

ARMS = ("stock", "v1", "v2")
ORDER = ["stock", "v1", "v2", "v2", "v1", "stock"] * 2


def valid_result(d: dict) -> dict:
    # 对实际请求数据而非只对命令行 flags 做门禁，尤其 2048 输出。
    if d.get("num_prompts") != 50 or d.get("completed") != 50:
        raise ValueError("not exactly 50/50 successful requests")
    if d.get("failed") not in (0, None):
        raise ValueError("failed requests present")
    if d.get("max_concurrency") != 10:
        raise ValueError("wrong concurrent-request cap")
    if len(d.get("input_lens", [])) != 50 or set(d["input_lens"]) != {4096}:
        raise ValueError("actual input lengths not all 4096")
    if len(d.get("output_lens", [])) != 50 or set(d["output_lens"]) != {2048}:
        raise ValueError("actual output lengths not all 2048")
    if d.get("errors") and any(d["errors"]):
        raise ValueError("nonempty request errors")
    keys = ("output_throughput", "median_ttft_ms", "p90_ttft_ms",
            "median_tpot_ms", "p90_tpot_ms", "median_itl_ms")
    out = {}
    for key in keys:
        value = d.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid performance field {key}={value}")
        out[key] = float(value)
    out.update(num_prompts=50, input=4096, output=2048, concurrency=10, tp=2)
    return out


def check_route(arm: str, log: str) -> dict:
    # 这只是日志层面的先决证据，最终仍必须附独立设备核符号/Trace。
    count = len(re.findall("Enable 310P3 TP=2 MemFabric unquantized mm_ar", log))
    v2 = "[mf310p-v2] M=" in log or "[mf310p-v2] first fused call" in log
    if arm == "stock":
        if count or v2:
            raise ValueError("stock route polluted by fused layer")
    else:
        if count < 40 or count % 40 != 0:
            raise ValueError(f"{arm} enabled layer count {count} is not 40 per TP rank")
        if (arm == "v2") != v2:
            raise ValueError(f"{arm} real route marker mismatch v2={v2}")
    return {"enabled_log_count": count, "v2_execution_marker": v2,
            "device_kernel_trace_still_required": True}


def verify_one(session: pathlib.Path, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    spec = "stock" if arm == "stock" else "fused"
    records = list(session.glob(f"{spec}_eager*_rep1_run.json"))
    if len(records) != 1:
        raise ValueError(f"expected exactly one Eager result JSON, got {records}")
    d = json.loads(records[0].read_text())
    result = valid_result(d)
    logs = list(session.glob(f"{spec}_eager*.log"))
    if len(logs) != 1:
        raise ValueError(f"expected 1 server log, got {logs}")
    result["route"] = check_route(arm, logs[0].read_text(errors="replace"))
    result["arm"] = arm
    result["bench_json"] = str(records[0])
    result["server_log"] = str(logs[0])
    return result


def summarize(root: pathlib.Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    fixed = {"input_tokens": 4096, "output_tokens": 2048, "tp": 2,
             "max_num_seqs": 10, "max_concurrency": 10,
             "num_prompts": 50, "mode": "eager"}
    for k, v in fixed.items():
        if manifest.get(k) != v:
            raise ValueError(f"manifest drift: {k}")
    sessions = sorted(root.glob("[0-9][0-9]-*/verified.json"))
    if len(sessions) != 12:
        raise ValueError(f"expected 12 sessions, got {len(sessions)}")
    rows = []
    for i, path in enumerate(sessions):
        rec = json.loads(path.read_text())
        if rec["arm"] != ORDER[i]:
            raise ValueError(f"bad crossover order session {i+1}: {rec['arm']}")
        rows.append(rec)
    groups = {}
    for arm in ARMS:
        vals = [r for r in rows if r["arm"] == arm]
        groups[arm] = {
            "n": len(vals),
            "output_throughput_median": statistics.median(v["output_throughput"] for v in vals),
            "ttft_median_ms": statistics.median(v["median_ttft_ms"] for v in vals),
            "tpot_median_ms": statistics.median(v["median_tpot_ms"] for v in vals),
            "raw_output_throughput": [v["output_throughput"] for v in vals],
        }
    stock = groups["stock"]["output_throughput_median"]
    for arm in ("v1", "v2"):
        groups[arm]["throughput_delta_vs_stock_pct"] = 100 * (
            groups[arm]["output_throughput_median"] / stock - 1)
    audit_path = root / "route_audit.json"
    audits = None
    if audit_path.is_file():
        audits = json.loads(audit_path.read_text())
    required = all((audits or {}).get(a, {}).get("device_trace_verified") is True
                   for a in ARMS)
    return {
        "experiment": "SEL-01",
        "fixed_profile": fixed,
        "sessions": rows,
        "group_summary": groups,
        "route_audit": audits,
        "actual_scheduler_batch10": "UNKNOWN: concurrent requests and max-num-seqs capped at 10; device scheduling must be independently evidenced",
        "selection_verdict": "REVIEW_REQUIRED" if required else "NO_DECISION_MISSING_DEVICE_ROUTE_PROOF",
        "safety": "Gate E P0 remains OPEN; no arm is production-approved",
        "notice": "Historical 4096/512 D4 and previous 4096/2048 Stock/V2 runs are NOT included",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify-one")
    v.add_argument("--session", required=True, type=pathlib.Path)
    v.add_argument("--arm", required=True, choices=ARMS)
    v.add_argument("--output", required=True, type=pathlib.Path)
    s = sub.add_parser("summarize")
    s.add_argument("--root", required=True, type=pathlib.Path)
    s.add_argument("--output", required=True, type=pathlib.Path)
    a = p.parse_args()
    if a.output.exists():
        p.error(f"refuse to overwrite {a.output}")
    report = verify_one(a.session, a.arm) if a.cmd == "verify-one" else summarize(a.root)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cmd": a.cmd, "status": "PASS", "selection_verdict": report.get("selection_verdict")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
