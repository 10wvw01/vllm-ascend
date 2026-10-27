#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""基于独立短Profiler原始Kernel CSV，证明U1的NZ-Stock确实命中真实aclnnMatmulWeightNz。
只做路由准入，不从Profiler数据产生正式无Profiler性能数字。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def verify(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError("trace root absent")
    proofs = []
    for rank in (0, 1):
        for m in (2048, 4096):
            path = root / f"rank{rank}_m{m}_kernel_details.csv"
            if not path.is_file() or path.stat().st_size < 100:
                raise ValueError(f"missing actual profiler kernel CSV: {path.name}")
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError(f"empty trace: {path.name}")
            fields = ("Name", "Start Time(us)", "Duration(us)",
                      "Input Shapes", "Output Shapes", "Stream ID")
            if any(any(k not in r for k in fields) for r in rows):
                raise ValueError(f"not native CANN kernel details CSV: {path.name}")
            mm = []
            hccl = []
            for row in rows:
                try:
                    beg = float(row["Start Time(us)"])
                    dur = float(row["Duration(us)"])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(beg) and math.isfinite(dur) and dur > 0):
                    continue
                if row["Name"].startswith("aclnnMatmulWeightNz_") and (
                    row["Input Shapes"].startswith(f'"{m},2048;')
                    and row["Output Shapes"] == f'"{m},2048"'
                ):
                    mm.append((beg, dur, row))
                if "hcom_allReduce" in row["Name"]:
                    hccl.append((beg, dur, row))
            if not mm or not hccl:
                raise ValueError(f"{path.name}: no verified real NZ-mm + HCCL")
            # U1独立进程只有该Stock链；检查AR紧邻MM结束，排除旁观的HCCL。
            pair = []
            for beg, dur, row in mm:
                following = [h for h in hccl if -50 <= h[0] - (beg+dur) <= 10000]
                if following:
                    h = min(following, key=lambda x: abs(x[0]-(beg+dur)))
                    pair.append({"nz_kernel": row["Name"], "mm_us": dur,
                                 "hccl_kernel": h[2]["Name"], "hccl_us": h[1],
                                 "gap_us": round(h[0]-(beg+dur), 3),
                                 "mm_stream": row["Stream ID"],
                                 "hccl_stream": h[2]["Stream ID"]})
            if not pair:
                raise ValueError(f"{path.name}: NZ MM not paired with adjacent HCCL")
            proofs.append({
                "rank": rank, "M": m, "trace": path.name,
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pairs": len(pair), "first_pair": pair[0],
            })
    return {"schema": "A1_NZ_U1_NATIVE_ROUTE_V1", "status": "NZ_STOCK_ROUTE_VERIFIED",
            "proofs": proofs, "scope": "short profiler route-only, NOT ABBA timings",
            "warning": "Actual real-model NM/stride equivalence still needs visual tensor/profile audit"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not a.trace_root.is_absolute() or not a.out.is_absolute() or a.out.exists():
        p.error("absolute input and unused absolute output required")
    r = verify(a.trace_root)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(r["status"])


if __name__ == "__main__":
    main()
