#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L2 真实Serving双Rank40站点核验：不把Kernel名字计数当作真实路由。

Input --audit-json: external, site-indexed device-provenance observations from
one Stock + one V2 short Profiler capture (5 windows each).
This is a CPU-only strict verifier; no NPU, no fabricated layer labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ARMS = ("stock", "v2")
RANKS = (0, 1)
NUM_SITES = 40
EXPECTED = tuple([f"o_proj_{i}" for i in range(10)] +
                 [f"out_proj_{i}" for i in range(30)])


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def evaluate(root: Path, observations: Path) -> dict:
    require(root.is_dir() and observations.is_file(), "missing L2 evidence")
    doc = json.loads(observations.read_text(encoding="utf-8"))
    require(doc.get("schema") == "A1_L2_REAL_MODEL_SITES_V1", "schema mismatch")
    require(doc.get("model_eager") is True and doc.get("tp") == 2
            and doc.get("input_tokens") == 4096 and doc.get("output_tokens") == 1,
            "L2 model workload mismatch")
    captures = doc.get("captures")
    require(isinstance(captures, dict) and set(captures) == set(ARMS),
            "missing Stock/V2 arm")
    report = {}
    all_proofs = []
    for arm in ARMS:
        ranks = captures[arm].get("ranks")
        require(isinstance(ranks, dict) and set(ranks) == {"0", "1"},
                arm + ": missing Rank0/1")
        for rank in RANKS:
            obj = ranks[str(rank)]
            steps = obj.get("steps")
            require(isinstance(steps, list) and len(steps) == 5,
                    f"{arm}/rank{rank}: expected five profiled Prefill steps")
            summary = []
            for step_i, step in enumerate(steps):
                sites = step.get("sites")
                require(isinstance(sites, list) and len(sites) == NUM_SITES,
                        f"{arm}/rank{rank}/step{step_i}: not 40 actual sites")
                keys = []
                for site in sites:
                    k = site.get("site_id")
                    require(k in EXPECTED, f"{arm}/rank{rank}: unknown target {k}")
                    keys.append(k)
                    require(site.get("physical_m") == 4096
                            and site.get("K") == 2048 and site.get("N") == 2048,
                            f"{arm}/rank{rank}: mismatched M/K/N")
                    require(site.get("kernel_execution_verified") is True
                            and site.get("downstream_visible_verified") is True,
                            f"{arm}/rank{rank}: not real executed/consumable Kernel")
                    if arm == "stock":
                        require(site.get("native_matmul_verified") is True
                                and site.get("hccl_ar_completion_verified") is True,
                                f"{arm}/rank{rank}: HCCL completion not proven")
                    else:
                        require(site.get("d4_kernel_verified") is True
                                and site.get("memfabric_ar_completion_verified") is True,
                                f"{arm}/rank{rank}: D4/MemFabric completion not proven")
                    require(type(site.get("copy_device_us")) in (int, float)
                            and site["copy_device_us"] >= 0,
                            f"{arm}/rank{rank}: missing copy cost (zero allowed)")
                    path = site.get("device_trace")
                    require(isinstance(path, str) and path and
                            not Path(path).is_absolute() and ".." not in Path(path).parts,
                            "unsafe raw trace path")
                    real = (root / path).resolve()
                    require(real.is_relative_to(root.resolve())
                            and real.is_file() and real.stat().st_size > 0,
                            "missing raw per-site device trace")
                    all_proofs.append(path)
                require(set(keys) == set(EXPECTED) and len(set(keys)) == NUM_SITES,
                        f"{arm}/rank{rank}/step{step_i}: 40 real layer identities not unique")
                summary.append({"step": step_i, "n_sites": NUM_SITES,
                                "total_copy_device_us": round(sum(x["copy_device_us"] for x in sites), 4)})
            report[f"{arm}/rank{rank}"] = summary
    require(len(set(all_proofs)) >= 4, "not enough independent original device trace files")
    return {"schema": "A1_L2_SITE_AUDIT_V1", "status": "L2_VERIFIED",
            "sites_per_rank_per_step": NUM_SITES,
            "historical_site_pattern_verified": list(EXPECTED),
            "trace_file_count": len(set(all_proofs)), "summary": report,
            "warning": "Explicit layer/kernel/stream provenance required; names cannot be inferred from kernel count alone"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not all(x.is_absolute() for x in (a.root, a.audit_json, a.out)):
        p.error("all paths must be absolute")
    if a.out.exists():
        p.error("output exists")
    try:
        result = evaluate(a.root, a.audit_json)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = {"schema": "A1_L2_SITE_AUDIT_V1",
                  "status": "L2_INCOMPLETE", "reason": str(exc)[:700]}
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
