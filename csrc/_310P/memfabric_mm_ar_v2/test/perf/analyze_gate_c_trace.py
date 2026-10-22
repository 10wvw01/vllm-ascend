#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""汇总 Gate C 两 Rank 设备 Kernel Trace；只比较整体 Kernel 耗时。

不把诊断变体时间差解释成正式 V2 的阶段时间，或当作 Qwen Forward。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

NAMES = {
    "formal": "V2TrialKernel",
    "loop4": "V2DiagLoopKernel",
    "diag0": "V2DiagMmKernel",
    "diag1": "V2DiagNoIbKernel",
    "diag3": "V2DiagIbKernel",
    "diag4": "V2DiagIbDoneKernel",
    "diag5": "V2TrialKernel",
}


def collect(path: Path, needle: str):
    content = json.loads(path.read_text(encoding="utf-8"))
    events = content.get("traceEvents", []) if isinstance(content, dict) else content
    if not isinstance(events, list):
        raise ValueError(f"traceEvents must be array: {path}")
    vals = [
        float(e["dur"]) for e in events
        if isinstance(e, dict) and e.get("ph") == "X"
        and isinstance(e.get("dur"), (int, float))
        and needle in str(e.get("name", ""))
    ]
    if not vals:
        raise ValueError(f"no kernel '{needle}' in {path}; cannot report performance")
    return sorted(vals)


def quantile(vals, fract):
    return vals[min(len(vals) - 1, int((len(vals) - 1) * fract))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    report = {
        "note": (
            "Each case captured in a separate TP2 torchrun session; numbers are device "
            "Kernel durations, NOT full Qwen Forward, host sync wall or SDMA/IB phase times."
        ),
        "cases": {},
    }
    for case, needle in NAMES.items():
        paths = sorted(a.root.glob(f"**/{case}_rank*_result.json"))
        if not paths:
            continue
        if len(paths) != 2:
            raise ValueError(f"{case}: need exactly two rank results, found {len(paths)}")
        pairs = {}
        for result_path in paths:
            meta = json.loads(result_path.read_text(encoding="utf-8"))
            rank = int(meta["rank"])
            if rank in pairs:
                raise ValueError(f"{case}: duplicate rank{rank}")
            if not (meta["bit_exact"] and meta["status"] == 0 and meta["profile"]):
                raise ValueError(f"{case} rank{rank}: invalid correctness/profile gate")
            trace = Path(meta["trace"])
            if not trace.is_file():
                raise FileNotFoundError(f"{case} rank{rank} trace missing: {trace}")
            vals = collect(trace, needle)
            if len(vals) != int(meta["iters"]):
                raise ValueError(f"{case} rank{rank}: expected {meta['iters']} kernels, got {len(vals)}")
            pairs[rank] = {
                "n": len(vals),
                "p50_us": round(statistics.median(vals), 2),
                "p90_us": round(quantile(vals, .90), 2),
                "max_us": round(max(vals), 2),
                "min_us": round(min(vals), 2),
                "trace": str(trace),
                "M": int(meta["M"]),
                "chunks": int(meta["chunks"]),
            }
        if set(pairs) != {0, 1}:
            raise ValueError(f"{case}: expected ranks 0,1 got {sorted(pairs)}")
        report["cases"][case] = pairs
    if not report["cases"]:
        raise ValueError("no completed profiled case results")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({case: {r: p["p50_us"] for r, p in rs.items()}
                      for case, rs in report["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
