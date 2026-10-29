#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R2机制诊断五种Case的双Rank纯CPU验收；不给非等价内核做时间相减。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

CASES = ("formal", "loop4", "diag0", "diag4", "diag5")
SOURCE = "U1_LARGE_MECHANISM_DIAG"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError("missing actual R2 mechanism session root")
    items = {}
    for case in CASES:
        case_dir = root / case
        if not case_dir.is_dir():
            raise ValueError(f"missing independent {case} process/pool case")
        reports = []
        for rank in (0, 1):
            p = case_dir / f"{case}_rank{rank}_result.json"
            if not p.is_file():
                raise ValueError(f"missing {case} rank{rank} raw result")
            rec = json.loads(p.read_text(encoding="utf-8"))
            required = {"experiment": SOURCE, "case": case, "rank": rank,
                        "M": 1024 if case.startswith("diag") else 4096,
                        "chunks": 1 if case.startswith("diag") else 4,
                        "chunk_rows": 1024, "bit_exact": True,
                        "status": 0, "profile": False}
            for k, val in required.items():
                if rec.get(k) != val:
                    raise ValueError(f"{case}/rank{rank}: invalid {k}")
            data = rec.get("samples")
            if not isinstance(data, list) or len(data) < 3:
                raise ValueError(f"{case}/rank{rank}: incomplete no-profiler samples")
            timing = []
            for i, item in enumerate(data):
                a, b = item.get("event_us"), item.get("wall_us")
                if (item.get("i") != i or type(a) not in (float, int)
                        or type(b) not in (float, int)
                        or not math.isfinite(a) or not math.isfinite(b)
                        or a <= 0 or b <= 0):
                    raise ValueError(f"{case}/rank{rank}: bad sample")
                timing.append(float(a))
            reports.append({"rank": rank, "sample_count": len(data),
                            "event_p50_us": round(statistics.median(timing), 3),
                            "raw_sha256": digest(p)})
        items[case] = reports
    return {
        "schema": "A1_R2_D4_DIAG_EXTERNAL_V1",
        "status": "DIAG_CASES_COMPLETE_INNER_PHASE_UNKNOWN",
        "cases": items,
        "comparisons": {
            "formal": "4-chunk overlapping formal D4 run, M4096",
            "loop4": "4-chunk SERIAL diagnostic; does NOT isolate one stage",
            "diag0": "1-chunk M1024 diagnostic MM-only-like behavior; not formal MM phase",
            "diag4": "1-chunk diagnostic; not a genuine full M4096 phase",
            "diag5": "1-chunk diagnostic; not a genuine full M4096 phase",
        },
        "no_algebraic_phase_breakdown": True,
        "inner_phase_status": "INNER_PHASE_UNKNOWN",
        "next_requirement": (
            "CANN 9.1 device-level trustworthy MM/SDMA/IB/Add phase evidence or "
            "supported cycle profiling; cannot subtract non-equivalent diagnostic cases"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not args.root.is_absolute() or not args.out.is_absolute() or args.out.exists():
        p.error("absolute root and new absolute output required")
    r = evaluate(args.root)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(r["status"])


if __name__ == "__main__":
    main()
