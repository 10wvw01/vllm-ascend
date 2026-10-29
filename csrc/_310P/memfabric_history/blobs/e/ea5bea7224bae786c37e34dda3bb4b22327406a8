#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""C3 现实收益预算（CPU-only）：用真实 vLLM 耗时和路由核数量筛掉低价值候选。

重要：此脚本仅测算'候选只缩短既有 Prefill Kernel 且不改变调用数量/路由/
排队/并行/跨阶段耦合'的直达时间预算，不是对所有潜在系统联动收益的数学上界。
当前 Stock/V1/V2 选型口径锁定输入4096/输出2048/TP2/batch10/Eager。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def budget(*, requests: int, trace_requests: int, trace_kernel_count_per_rank: int,
           bench_duration_s: float, baseline_kernel_us: float,
           candidate_kernel_us: float | None = None) -> dict:
    if requests <= 0 or trace_requests <= 0 or trace_kernel_count_per_rank < 0:
        raise ValueError("invalid request and route count")
    if trace_kernel_count_per_rank % trace_requests:
        raise ValueError("route kernel count is not divisible by profiled requests")
    if bench_duration_s <= 0 or baseline_kernel_us <= 0:
        raise ValueError("nonpositive bench duration or kernel time")
    if candidate_kernel_us is not None and (
        not math.isfinite(candidate_kernel_us) or candidate_kernel_us <= 0
    ):
        raise ValueError("candidate kernel must be positive")
    if not (math.isfinite(bench_duration_s) and math.isfinite(baseline_kernel_us)):
        raise ValueError("nonfinite durations")

    calls_per_request = trace_kernel_count_per_rank // trace_requests
    # TP2 Rank0/Rank1 concurrently execute; do not double-count per-rank kernels.
    total_calls = calls_per_request * requests
    kernel_s = total_calls * baseline_kernel_us * 1e-6
    if kernel_s >= bench_duration_s:
        raise ValueError("kernel sum >= whole run: not a valid single-critical-path budget")
    # If only these invocations are completely removed and everything else is unchanged.
    whole_elimination_pct = (bench_duration_s / (bench_duration_s - kernel_s) - 1.0) * 100
    result = {
        "fixed_profile": {"input_tokens": 4096, "output_tokens": 2048, "tp": 2,
                          "batch_cap": 10, "mode": "eager"},
        "assumed_requests": requests,
        "profiled_requests": trace_requests,
        "kernel_calls_per_rank_in_trace": trace_kernel_count_per_rank,
        "fusion_calls_per_request": calls_per_request,
        "fusion_calls_whole_bench": total_calls,
        "bench_duration_s": round(bench_duration_s, 3),
        "baseline_fusion_kernel_us": round(baseline_kernel_us, 3),
        "sum_prefill_fusion_kernel_time_s": round(kernel_s, 4),
        "fraction_of_duration_pct": round(100 * kernel_s / bench_duration_s, 5),
        "full_kernel_elimination_estimated_throughput_gain_pct": round(whole_elimination_pct, 5),
        "interpretation": (
            "ONLY a simplified accounting limit for direct shortening of these observed "
            "Prefill kernels. Reassess if routing, kernel count, queues, scheduling, "
            "cache or communication overlap changes."
        ),
        "not_a_device_measurement": True,
    }
    if candidate_kernel_us is not None:
        saved_s = total_calls * (baseline_kernel_us - candidate_kernel_us) * 1e-6
        if bench_duration_s <= saved_s:
            raise ValueError("saved time >= whole run")
        result.update(
            candidate_kernel_us=round(candidate_kernel_us, 3),
            predicted_duration_reduction_s=round(saved_s, 5),
            predicted_throughput_gain_pct=round(
                100 * (bench_duration_s / (bench_duration_s - saved_s) - 1), 5
            ),
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-json", type=Path, required=True,
                   help="A full-success Stock/V2 SEL01 *_run.json, never a failed session")
    p.add_argument("--trace-kernels-per-rank", type=int, required=True,
                   help="Observed V2TrialKernel launch count, from an independent route-audit trace")
    p.add_argument("--trace-requests", type=int, required=True,
                   help="Number of requests corresponding to that trace kernel count")
    p.add_argument("--base-kernel-us", type=float, required=True,
                   help="Paired TP2 no-profiler event median for baseline (NOT profiler kernel p50)")
    p.add_argument("--candidate-kernel-us", type=float,
                   help="Optional paired TP2 no-profiler event median for C3 candidate")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("refuse to overwrite previous evidence")
    d = json.loads(a.bench_json.read_text())
    if d.get("num_prompts") != 50 or d.get("completed") != 50 or d.get("failed") != 0:
        p.error("bench data must be 50/50 complete with zero failures")
    if len(d.get("input_lens", [])) != 50 or set(d["input_lens"]) != {4096}:
        p.error("only real 4096-token request inputs are accepted")
    if len(d.get("output_lens", [])) != 50 or set(d["output_lens"]) != {2048}:
        p.error("only real 2048-token request outputs are accepted")
    if d.get("max_concurrency") != 10:
        p.error("concurrency must be 10")
    v = budget(requests=50, trace_requests=a.trace_requests,
               trace_kernel_count_per_rank=a.trace_kernels_per_rank,
               bench_duration_s=float(d["duration"]),
               baseline_kernel_us=a.base_kernel_us,
               candidate_kernel_us=a.candidate_kernel_us)
    a.out.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(v, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
