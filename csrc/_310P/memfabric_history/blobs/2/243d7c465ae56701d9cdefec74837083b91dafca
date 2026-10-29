#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R1真实Serving L2 20份逐站点原始提取离线复算，区分已测外部耗时与未知内核分段。"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

L2 = Path("s4_l2/analysis")
SITES = tuple(f"o_proj_{i}" for i in range(10)) + tuple(f"out_proj_{i}" for i in range(30))


def finite(v):
    return type(v) in (float, int) and math.isfinite(v) and v >= 0


def evaluate(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError("evidence root is absent")
    out = {}
    outliers = []
    for arm in ("stock", "v2"):
        for rank in (0, 1):
            per_step = []
            values = []
            for step in range(5):
                path = root / L2 / f"{arm}_rank{rank}_step{step}_sites.json"
                doc = json.loads(path.read_text(encoding="utf-8"))
                if (doc.get("arm"), doc.get("rank"), doc.get("step")) != (arm, rank, step):
                    raise ValueError(f"{path}: arm/rank/step mismatch")
                sites = doc.get("sites")
                if not isinstance(sites, list) or len(sites) != 40:
                    raise ValueError(f"{path}: not 40 real sites")
                if set(x.get("site_id") for x in sites) != set(SITES):
                    raise ValueError(f"{path}: site identities duplicated or missing")
                rows = []
                for item in sites:
                    if (item.get("physical_m"), item.get("K"), item.get("N")) != (4096, 2048, 2048):
                        raise ValueError("M/K/N mismatch")
                    if item.get("downstream_visible_verified") is not True:
                        raise ValueError("completion not proven")
                    kd = item.get("kernel_dur_us")
                    if not finite(kd) or kd <= 0:
                        raise ValueError("bad kernel duration")
                    if arm == "stock":
                        if (item.get("native_matmul_verified") is not True
                                or item.get("hccl_ar_completion_verified") is not True):
                            raise ValueError("not native NZ+HCCL Stock")
                        if not item.get("kernel_name", "").startswith("aclnnMatmulWeightNz_"):
                            raise ValueError("Stock not real NZ MatMul")
                        gap, ar = item.get("ar_gap_after_mm_us"), item.get("ar_dur_us")
                        if not finite(gap) or not finite(ar) or ar <= 0:
                            raise ValueError("invalid Stock AR/gap")
                        value = kd + gap + ar
                    else:
                        if item.get("d4_kernel_verified") is not True or item.get("memfabric_ar_completion_verified") is not True:
                            raise ValueError("not fused D4 completion")
                        value = kd
                        if kd > 10000:
                            outliers.append({"arm": arm, "rank": rank, "step": step,
                                             "site_id": item["site_id"], "layer_index": item["layer_index"],
                                             "fused_kernel_us": round(kd, 3)})
                    rows.append(value)
                per_step.append(round(sum(rows), 3))
                values.extend(rows)
            sorted_values = sorted(values)
            out[f"{arm}/rank{rank}"] = {
                "n": len(values), "per_step_site_chain_sum_us": per_step,
                "per_site_chain_p50_us": round(statistics.median(values), 3),
                "per_site_chain_p95_us": round(sorted_values[189], 3),
                "per_site_chain_mean_us": round(statistics.mean(values), 3),
                "per_site_chain_max_us": round(max(values), 3),
            }
    return {
        "schema": "A1_R1_L2_SITE_BUDGET_V1",
        "status": "REAL_SITE_EXTERIOR_MEASURED_INNER_PHASES_UNKNOWN",
        "per_rank": out, "fused_kernel_outliers_gt10ms": outliers,
        "warnings": [
            "For Stock each per-site duration sums same-site recorded MM+gap+AR; step sums are not full Prefill critical-path time.",
            "V2 has a single fused kernel; MM/SDMA/mail/Add/IB overlap CANNOT be decomposed from total kernel duration.",
            "Profiler step and no-profiler L3 sessions are not interchangeable.",
            "No device clock alignment is assumed between ranks.",
        ]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not a.evidence_root.is_absolute() or not a.out.is_absolute() or a.out.exists():
        p.error("must use absolute paths and a fresh output")
    result = evaluate(a.evidence_root)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
