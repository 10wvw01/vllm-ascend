#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SEL-01 最终分组分析：容忍失败场（保留不删），按臂汇总有效会话。

与 sel01_analyze.py summarize 的区别：summarize 是 12 场全净的 happy path；
本脚本面向主驱动停止门触发后的续跑/重试结果——
  - 每场仍以 verify-one 的严格校验为准（verified.json 存在才算有效）；
  - 失败场（.failed 标记）全部保留进台账，绝不计入性能；
  - 按臂汇总中位数吞吐/延迟、相对 Stock 的增益、跨时段方向一致性；
  - 输出 SEL-01 选型判定（含 PERF_EVALUATION_BLOCKED_BY_STABILITY 分支）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

ARMS = ("stock", "v1", "v2")


def scheduler_batch_hist(server_log: pathlib.Path) -> dict:
    hist = {}
    running_re = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs")
    for m in running_re.finditer(server_log.read_text(errors="replace")):
        running = int(m.group(1))
        hist[running] = hist.get(running, 0) + 1
    return dict(sorted(hist.items()))


def collect(root: pathlib.Path) -> list[dict]:
    rows = []
    for session in sorted(root.glob("[0-9][0-9]-*")):
        if not session.is_dir():
            continue
        name = session.name
        m = re.match(r"^(\d+)-(stock|v1|v2)$", name)
        if not m:
            continue
        sid, arm = int(m.group(1)), m.group(2)
        verified = session / "verified.json"
        failed = session / ".failed"
        server_logs = [p for p in session.glob("*_eager*.log") if not p.name.endswith("_bench.log")]
        row = {"session": name, "sid": sid, "arm": arm}
        if verified.is_file():
            d = json.loads(verified.read_text())
            row.update(
                valid=True,
                output_throughput=d["output_throughput"],
                median_ttft_ms=d["median_ttft_ms"],
                p90_ttft_ms=d["p90_ttft_ms"],
                median_tpot_ms=d["median_tpot_ms"],
                p90_tpot_ms=d["p90_tpot_ms"],
                median_itl_ms=d["median_itl_ms"],
                route=d["route"],
                scheduler_batch_hist=scheduler_batch_hist(server_logs[0]) if server_logs else {},
            )
        elif failed.is_file():
            row.update(valid=False, failure_reason=failed.read_text().strip())
        else:
            row.update(valid=False, failure_reason="INCOMPLETE (no verified.json, no .failed)")
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    a = p.parse_args()
    rows = collect(a.root)
    groups = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        valid = [r for r in arm_rows if r["valid"]]
        failed = [r for r in arm_rows if not r["valid"]]
        g = {
            "attempts": len(arm_rows),
            "valid_sessions": len(valid),
            "failed_sessions": len(failed),
            "failure_ledger": [{"session": r["session"], "reason": r["failure_reason"]} for r in failed],
            "sessions": [
                {
                    k: r[k]
                    for k in (
                        "session",
                        "output_throughput",
                        "median_ttft_ms",
                        "p90_ttft_ms",
                        "median_tpot_ms",
                        "p90_tpot_ms",
                        "median_itl_ms",
                        "scheduler_batch_hist",
                    )
                }
                for r in valid
            ],
        }
        if valid:
            tps = [r["output_throughput"] for r in valid]
            g["output_throughput_median"] = statistics.median(tps)
            g["output_throughput_min_max"] = [min(tps), max(tps)]
            g["ttft_p50_of_session_medians_ms"] = statistics.median(r["median_ttft_ms"] for r in valid)
            g["tpot_p50_of_session_medians_ms"] = statistics.median(r["median_tpot_ms"] for r in valid)
            g["itl_p50_of_session_medians_ms"] = statistics.median(r["median_itl_ms"] for r in valid)
        groups[arm] = g

    stock = groups["stock"]
    verdict_notes = []
    if stock["valid_sessions"] == 0:
        selection = "NO_DECISION_STOCK_NO_VALID_SAMPLES"
    else:
        base = stock["output_throughput_median"]
        for arm in ("v1", "v2"):
            g = groups[arm]
            if g["valid_sessions"] == 0:
                g["throughput_delta_vs_stock_pct"] = None
                ledger = g["failure_ledger"]
                natural = sum(1 for f in ledger if "ABORTED_BY_EXPERIMENTER" not in f["reason"])
                aborted = len(ledger) - natural
                verdict_notes.append(
                    f"{arm}: PERF_EVALUATION_BLOCKED_BY_STABILITY "
                    f"({natural} natural failures + {aborted} experimenter aborts "
                    f"of {g['attempts']} attempts, no valid sample)"
                )
                continue
            g["throughput_delta_vs_stock_pct"] = round(100 * (g["output_throughput_median"] / base - 1), 2)
            # 跨时段方向一致性：每个有效会话相对 Stock 中位数都同向
            deltas = [100 * (s["output_throughput"] / base - 1) for s in g["sessions"]]
            g["per_session_delta_vs_stock_median_pct"] = [round(d, 2) for d in deltas]
            g["direction_consistent"] = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)

        v1, v2 = groups["v1"], groups["v2"]

        def beats(d):
            return d is not None and d >= 5.0

        if beats(v1.get("throughput_delta_vs_stock_pct")) and v1.get("direction_consistent"):
            selection = "V1_CANDIDATE_GE5PCT_CONSISTENT"
        elif beats(v2.get("throughput_delta_vs_stock_pct")) and v2.get("direction_consistent"):
            selection = "V2_CANDIDATE_GE5PCT_CONSISTENT"
        else:
            selection = "KEEP_STOCK_NO_ARM_MEETS_5PCT_CONSISTENT_BAR"
        blocked = [a for a in ("v1", "v2") if groups[a]["valid_sessions"] == 0]
        if blocked:
            selection = "PERF_EVALUATION_BLOCKED_BY_STABILITY(" + ",".join(blocked) + ")"

    report = {
        "experiment": "SEL-01",
        "fixed_profile": {
            "input_tokens": 4096,
            "output_tokens": 2048,
            "tp": 2,
            "max_num_seqs": 10,
            "max_concurrency": 10,
            "num_prompts": 50,
            "mode": "eager",
            "note": "batch10 = cap; steady decode shows Running:10 (see scheduler_batch_hist)",
        },
        "session_rows": rows,
        "group_summary": groups,
        "selection_verdict": selection,
        "verdict_notes": verdict_notes,
        "safety": (
            "Gate E P0 remains OPEN (intermittent dual-die spin hang, G3 section-8); no arm is production-approved"
        ),
        "notice": "Failed sessions preserved and excluded from performance; D4 4096/512 data NOT included",
    }
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verdict": selection,
                **{
                    arm: {
                        "valid": groups[arm]["valid_sessions"],
                        "failed": groups[arm]["failed_sessions"],
                        "thr_median": groups[arm].get("output_throughput_median"),
                        "delta_pct": groups[arm].get("throughput_delta_vs_stock_pct"),
                    }
                    for arm in ARMS
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
