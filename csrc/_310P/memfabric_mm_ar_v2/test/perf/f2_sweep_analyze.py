#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""F2：TP2成对 Event 数据离线审计；同一 D4 Kernel 参数网格筛选。
不能以 max(p50(rank0),p50(rank1)) 冒充成对时延；一场失败立刻拒绝选优。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

MS = (2048, 4096)
CHUNKS = (256, 512, 1024)
OVERLAPS = (0, 1)
MIN_SAMPLES = 30


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(rank0: dict, rank1: dict, *, m: int, chunk: int, overlap: int,
           expected_sha: str) -> dict:
    for r, data in enumerate((rank0, rank1)):
        if data.get("rank") != r or data.get("M") != m:
            raise ValueError(f"rank{r}: mismatched rank/M")
        if data.get("chunk_rows") != chunk or data.get("chunks") != m // chunk:
            raise ValueError(f"rank{r}: chunk count mismatch")
        if data.get("overlap") != overlap:
            raise ValueError(f"rank{r}: overlap mismatch")
        if not data.get("bit_exact") or data.get("status") != 0:
            raise ValueError(f"rank{r}: correctness or protocol failure")
        if data.get("profile") is not False:
            raise ValueError(f"rank{r}: profiler contaminated baseline")
        if data.get("device_so") != expected_sha:
            raise ValueError(f"rank{r}: device SO SHA mismatch")
        if int(data.get("warmup", -1)) < 2:
            raise ValueError(f"rank{r}: missing warmup")
        if data.get("iters") != MIN_SAMPLES or len(data.get("samples", [])) != MIN_SAMPLES:
            raise ValueError(f"rank{r}: missing samples")
    ev = []
    wall = []
    for i, (a, b) in enumerate(zip(rank0["samples"], rank1["samples"])):
        if a.get("i") != i or b.get("i") != i:
            raise ValueError("rank samples must be aligned by index")
        pairs = []
        for entry in (a, b):
            one = []
            for name in ("event_us", "wall_us"):
                v = entry.get(name)
                if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise ValueError(f"nonfinite/negative {name}")
                one.append(float(v))
            pairs.append(one)
        ev.append(max(pairs[0][0], pairs[1][0]))
        wall.append(max(pairs[0][1], pairs[1][1]))
    # 不截尾、不删挂起长样本
    return {
        "M": m, "chunk_rows": chunk, "overlap": overlap, "samples": len(ev),
        "paired_event_p50_us": round(statistics.median(ev), 3),
        "paired_event_p90_us": round(sorted(ev)[int(math.ceil(.90 * len(ev))) - 1], 3),
        "paired_wall_p50_us": round(statistics.median(wall), 3),
        "stalls_gt_6ms": sum(x > 6000 for x in ev),
        "paired_event_us": ev,
    }


def analyze(root: Path, expected_sha: str) -> dict:
    if len(expected_sha) != 64 or set(expected_sha.lower()) - set("0123456789abcdef"):
        raise ValueError("invalid SO SHA256")
    sessions = sorted(root.glob("*/"))
    if not sessions:
        raise ValueError("no session directories")
    rows = []
    for d in sessions:
        js = sorted(d.glob("*_rank*_f2.json"))
        if len(js) != 2:
            raise ValueError(f"{d}: expected two rank JSON files, got {len(js)}")
        ds = [json.loads(p.read_text()) for p in js]
        by_rank = {a.get("rank"): a for a in ds}
        if set(by_rank) != {0, 1}:
            raise ValueError(f"{d}: must have rank0 and rank1")
        r0, r1 = by_rank[0], by_rank[1]
        if r0.get("label") != r1.get("label"):
            raise ValueError("ranks have different labels")
        m = r0.get("M")
        chunk = r0.get("chunk_rows")
        overlap = r0.get("overlap")
        if m not in MS or chunk not in CHUNKS or overlap not in OVERLAPS:
            raise ValueError(f"unsupported shape in {d}")
        row = paired(r0, r1, m=m, chunk=chunk, overlap=overlap,
                     expected_sha=expected_sha)
        row["session"] = d.name
        row["rank_json_sha256"] = {f"rank{r}": _hash(js[0 if ds[0].get("rank") == r else 1])
                                   for r in (0, 1)}
        rows.append(row)
    groups = []
    for m in MS:
        reference = [x["paired_event_p50_us"] for x in rows
                     if x["M"] == m and x["chunk_rows"] == 1024 and x["overlap"] == 1]
        if not reference:
            raise ValueError(f"no D4 baseline M{m} C1024 overlap1")
        baseline = statistics.median(reference)
        for chunk in CHUNKS:
            for overlap in OVERLAPS:
                sel = [x for x in rows if (x["M"], x["chunk_rows"], x["overlap"]) == (m, chunk, overlap)]
                if not sel:
                    raise ValueError(f"M{m} C{chunk} O{overlap}: no session")
                values = [x["paired_event_p50_us"] for x in sel]
                if any(x["stalls_gt_6ms"] for x in sel):
                    raise ValueError("stalled session: fail closed and preserve evidence")
                med = statistics.median(values)
                groups.append({
                    "M": m, "chunk_rows": chunk, "overlap": overlap,
                    "sessions": len(sel), "median_pair_event_us": round(med, 3),
                    "baseline_c1024_o1_us": round(baseline, 3),
                    "delta_vs_baseline_pct": round(100 * (baseline / med - 1), 3),
                    "per_session_pair_p50_us": values,
                })
    return {
        "fixed_device_geometry": "TP=2, K=N=2048, FP16 NZ MM + MemFabric AR",
        "baseline": "same SO, M-specific C1024/O1/D4 Tile2048",
        "rows": rows, "groups": groups,
        "verdict": "SCREENING_ONLY_REQUIRES_CONCURRENT_STOCK_BASELINE_AND_ABBA_RETEST",
        "notes": [
            "This sweep compares V2 scheduling variants, NOT Stock MM+HCCL.",
            "Same-process stock vs fused comparison must include real HCCL and synchronization.",
            "Independent dual-rank samples are paired by sample index; no per-rank p50 max shortcut.",
            "No device trace or phase segmentation has yet proven which stage is the bottleneck.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device-so-sha", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("refuse to overwrite evidence")
    report = analyze(a.root, a.device_so_sha)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"F2 reviewed {len(report['rows'])} sessions; no winner until Stock head-to-head.")


if __name__ == "__main__":
    main()
