#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TP2 V2 sync/async ABBA 独立进程结果汇总；无 torch/设备依赖。

使用较慢 Rank 的各自耗时中位数，作为 max-rank median 近似值；
不同 Rank 的单次样本未成对，不能声称这是每轮 max 的精确中位数。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def extract(text: str, *, rows: int, arm: str, windows: set[int]) -> dict:
    records = []
    for line in text.splitlines():
        if '"V2_SYNC_ASYNC"' not in line:
            continue
        pos = line.find("{")
        if pos < 0:
            continue
        try:
            row = json.loads(line[pos:])
        except json.JSONDecodeError:
            continue
        if row.get("tag") == "V2_SYNC_ASYNC":
            records.append(row)
    result = {}
    for rank in (0, 1):
        rs = [x for x in records if x.get("rank") == rank]
        comp = [x for x in rs if x.get("phase") == "complete"
                and x.get("status") == "PASS" and x.get("arm") == arm]
        if len(comp) != 1:
            raise ValueError(f"{arm} rank{rank}: missing unique PASS completion")
        found = [x for x in rs if x.get("phase") == "result"]
        if len(found) != len(windows):
            raise ValueError(f"{arm} rank{rank}: result count {len(found)}")
        rank_windows = {}
        for x in found:
            win = x.get("window")
            if (win not in windows or x.get("rows") != rows or
                    x.get("arm") != arm or not x.get("validated") or
                    x.get("wall", {}).get("count", 0) < 2 or win in rank_windows):
                raise ValueError(f"{arm} rank{rank}: invalid window result {x}")
            rank_windows[win] = x
        result[rank] = rank_windows
    return {
        w: {
            "max_rank_median_ms": max(result[0][w]["wall"]["median_ms"],
                                      result[1][w]["wall"]["median_ms"]),
            "max_rank_p95_ms": max(result[0][w]["wall"]["p95_ms"],
                                   result[1][w]["wall"]["p95_ms"]),
            "rank0_ms": result[0][w]["wall"]["median_ms"],
            "rank1_ms": result[1][w]["wall"]["median_ms"],
            "steps": result[0][w]["steps"],
        }
        for w in sorted(windows)
    }


def percent_gain(stock: float, v2: float) -> float:
    return round(100 * (stock - v2) / stock, 4)


def summarize(root: Path, rows: list[int], windows: set[int]) -> dict:
    out = {
        "evidence_level": "TP2_MICROBENCH_CONTROLLED",
        "estimator": "max of per-rank session medians, not median of per-repetition max",
        "limits": [
            "40 independent calls are a bounded async submission envelope, not Qwen's 40 sequential dependent layers",
            "arm runs in separate processes; ABBA balances time order but not all thermal/network variance",
            "host end-to-end wall includes enqueue, work.wait, stream synchronize and Python/launch costs",
            "no profiler in timed runs; benchmark cannot independently attribute MM/SDMA/HCCL cache effects",
            "rank medians do not establish statistical confidence or production stability",
        ],
        "rows": {},
    }
    for m in rows:
        by_round = {}
        for round_no in (1, 2):
            x = {}
            for arm in ("stock", "v2"):
                path = root / f"r{round_no}_{arm}_m{m}.log"
                if not path.is_file():
                    raise FileNotFoundError(path)
                x[arm] = extract(path.read_text(errors="replace"),
                                 rows=m, arm=arm, windows=windows)
            by_round[round_no] = x
        summary = {}
        for window in sorted(windows):
            per_round = []
            for round_no in (1, 2):
                s = by_round[round_no]["stock"][window]["max_rank_median_ms"]
                v = by_round[round_no]["v2"][window]["max_rank_median_ms"]
                per_round.append({
                    "round": round_no,
                    "stock_max_rank_median_ms": s,
                    "v2_max_rank_median_ms": v,
                    "v2_latency_gain_pct": percent_gain(s, v),
                    "stock_max_rank_p95_ms":
                        by_round[round_no]["stock"][window]["max_rank_p95_ms"],
                    "v2_max_rank_p95_ms":
                        by_round[round_no]["v2"][window]["max_rank_p95_ms"],
                })
            gains = [z["v2_latency_gain_pct"] for z in per_round]
            summary[window] = {
                "paired_runs": per_round,
                "median_gain_pct": round(statistics.median(gains), 4),
                "min_gain_pct": min(gains),
                "max_gain_pct": max(gains),
                "both_positive": all(v > 0 for v in gains),
            }
        out["rows"][str(m)] = {str(k): v for k, v in summary.items()}
        if 1 in summary and 4 in summary:
            out["rows"][str(m)]["window4_minus_window1_gain_pp"] = round(
                summary[4]["median_gain_pct"] - summary[1]["median_gain_pct"], 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--rows", type=int, nargs="+", default=[4096])
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        result = summarize(args.session_dir, args.rows, set(args.windows))
        encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(encoded)
        else:
            print(encoded, end="")
    except (ValueError, OSError, KeyError) as e:
        ap.exit(3, f"SUMMARY FAIL: {e}\n")


if __name__ == "__main__":
    main()
