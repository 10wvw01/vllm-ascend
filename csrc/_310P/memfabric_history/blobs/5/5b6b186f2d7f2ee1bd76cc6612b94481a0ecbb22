#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SEL-01 路由审计分析：serve 日志 + 双 Rank 设备 Trace 核符号双证据。"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

FUSED_LOG = "Enable 310P3 TP=2 MemFabric unquantized mm_ar"
V2_MARKERS = ("[mf310p-v2] M=", "[mf310p-v2] first fused call")


def rank_trace_kernels(trace_dir: pathlib.Path) -> dict:
    stats = {}
    # torch_npu ascend_pt 导出后需 msprof --export 离线后处理：
    # CSV 位于 PROF_*/mindstudio_profiler_output/op_statistic_*.csv
    csvs = list(trace_dir.glob("PROF_*/mindstudio_profiler_output/op_statistic_*.csv"))
    csvs += list(trace_dir.glob("ASCEND_PROFILER_OUTPUT/op_statistic.csv"))
    for csv in csvs:
        lines = csv.read_text(errors="replace").splitlines()
        if not lines:
            continue
        for row in lines[1:]:
            cols = row.split(",")
            if len(cols) < 5:
                continue
            op, count = cols[1], cols[3]
            try:
                stats[op] = stats.get(op, 0) + int(count)
            except ValueError:
                continue
    return stats


def match_kernels(stats: dict, needles: list[str]) -> dict[str, int]:
    out = {}
    for needle in needles:
        total = sum(c for op, c in stats.items() if needle in op)
        out[needle] = total
    return out


def audit_arm(arm: str, session: pathlib.Path) -> dict:
    log = (session / "serve.log").read_text(errors="replace")
    enable = log.count(FUSED_LOG)
    v2_marker_lines = [line for line in log.splitlines() if any(m in line for m in V2_MARKERS)]
    # 每 Rank 40 层：两个 worker 进程各打 40 条
    ranks = sorted(p for p in session.glob("*_ascend_pt") if p.is_dir())
    per_rank = {}
    for rd in ranks:
        stats = rank_trace_kernels(rd)
        # V1 融合路径在设备上以 mf310p 协作核家族执行：
        # Mf310pDirectProducerKernel（融合MM生产者）+ WaitBatch/Add/Quie/Gate/Ack 等
        v1_family = sum(match_kernels(stats, ["mf310p", "Mf310p"]).values())
        per_rank[rd.name] = {
            "fused_v1_mf310p_family_total": v1_family,
            "fused_v1_Mf310pDirectProducerKernel": match_kernels(stats, ["Mf310pDirectProducerKernel"])[
                "Mf310pDirectProducerKernel"
            ],
            "fused_v2_V2TrialKernel": match_kernels(stats, ["V2TrialKernel"])["V2TrialKernel"],
            "orch_HybmSdmaOrchEpoch": match_kernels(stats, ["HybmSdmaOrchEpoch"])["HybmSdmaOrchEpoch"],
            "native_MatMulV3": match_kernels(stats, ["MatMulV3"])["MatMulV3"],
            "native_MatMulV2": match_kernels(stats, ["MatMulV2"])["MatMulV2"],
        }
    report = {
        "arm": arm,
        "enable_log_total": enable,
        "enable_log_per_rank_expect_40": enable == 80,
        "v2_marker_sample": v2_marker_lines[:6],
        "v2_marker_count": len(v2_marker_lines),
        "num_rank_traces": len(ranks),
        "per_rank_kernel_counts": per_rank,
    }
    ok = len(ranks) == 2
    fused_counts_v1 = [v["fused_v1_Mf310pDirectProducerKernel"] for v in per_rank.values()]
    fused_counts_v2 = [v["fused_v2_V2TrialKernel"] for v in per_rank.values()]
    v1_family = [v["fused_v1_mf310p_family_total"] for v in per_rank.values()]
    orch = [v["orch_HybmSdmaOrchEpoch"] for v in per_rank.values()]
    native = [v["native_MatMulV3"] + v["native_MatMulV2"] for v in per_rank.values()]
    if arm == "stock":
        ok = ok and enable == 0 and all(c == 0 for c in fused_counts_v1 + fused_counts_v2 + orch + v1_family)
        report["native_mm_present"] = all(c > 0 for c in native)
        ok = ok and report["native_mm_present"]
        report["description"] = "Stock: zero fused enable logs, zero mf310p/V2/orch kernels, native MM+HCCL only"
    elif arm == "v1":
        ok = ok and enable == 80 and all(c > 0 for c in fused_counts_v1) and all(c > 0 for c in v1_family)
        ok = ok and all(c == 0 for c in fused_counts_v2)
        report["description"] = "V1: 40 Enable per rank + Mf310pDirectProducerKernel/mf310p family executed on device"
    elif arm == "v2":
        ok = ok and enable == 80 and all(c > 0 for c in fused_counts_v2) and report["v2_marker_count"] > 0
        ok = ok and all(c == 0 for c in fused_counts_v1) and all(c == 0 for c in v1_family)
        report["description"] = "V2: 40 Enable per rank + V2TrialKernel executed on device + M= route markers"
    report["device_trace_verified"] = bool(ok)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    a = p.parse_args()
    report = {}
    for arm in ("stock", "v1", "v2"):
        session = a.root / arm
        if not session.is_dir():
            report[arm] = {"device_trace_verified": False, "error": "session missing"}
            continue
        report[arm] = audit_arm(arm, session)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for arm, r in report.items():
        print(
            f"{arm}: device_trace_verified={r.get('device_trace_verified')} :: {r.get('description', r.get('error'))}"
        )
    sys.exit(0 if all(r.get("device_trace_verified") for r in report.values()) else 1)


if __name__ == "__main__":
    main()
