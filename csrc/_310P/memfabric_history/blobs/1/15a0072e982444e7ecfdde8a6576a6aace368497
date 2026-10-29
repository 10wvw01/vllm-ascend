#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""C4: 310P3 Eager Decode-only vLLM Profiler CSV 归因；不替代 NPU/TP2 关键路径时间。

输入是两 Rank 分别导出的 kernel_details.csv，和从真实 4096/2048、TP2、
batch-cap10 的 vLLM Eager 运行人工核验的 Decode 窗口证明文件。
总 kernel duration 是设备计算工作量画像，不是延迟，也不等同独占耗时。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SHAPE = {"input_tokens": 4096, "output_tokens": 2048,
         "tp": 2, "max_num_seqs": 10, "max_concurrency": 10,
         "mode": "eager"}


def category(name: str) -> str:
    n = name.casefold()
    # 注意按具体算子优先匹配，不要让 GroupedMatmul 误归到通用 Matmul。
    if re.search(r"(grouped.?matmul|quant.?gmm|grouped.?mm)", n):
        return "MOE_GROUPED_GEMM"
    if re.search(r"(moe.?init.?routing|moe.?token.?unpermute|moe.?token.?permute|dispatch.?moe|combine.?moe|moe.?gating|expert.?select|group.?topk)", n):
        return "MOE_ROUTE_COMBINE"
    if re.search(r"(swiglu|swi.?glu)", n):
        return "MOE_SWIGLU_OR_OTHER"
    if re.search(r"(recurrent.?gated.?delta|gated.?delta.?rule|delta.?state)", n):
        return "GDN_RECURRENCE"
    if re.search(r"(causal.?conv1d|causal.?conv|conv1d)", n):
        return "GDN_CONV"
    if re.search(r"(gdn.?gating|gated.?delta.?gating|fused.?gating)", n):
        return "GDN_GATING"
    if re.search(r"(paged.?attention|paged.?attn|splitfuse|flash.?attention)", n):
        return "FULL_ATTENTION"
    if re.search(r"(all.?reduce|all.?gather|reduce.?scatter|hccl|collective)", n):
        return "TP_COMMUNICATION"
    if re.search(r"(quantize|dequantize|quant.?matmul)", n):
        return "QUANT_MATMUL_OR_CONVERT"
    if re.search(r"(matmul|batchmatmul|gemm|mm[_\(])", n):
        return "OTHER_MATMUL"
    if re.search(r"(rms.?norm|layer.?norm|norm.?quant)", n):
        return "NORM"
    if re.search(r"(topk|softmax|sample|sort|argmax|logits)", n):
        return "SAMPLING_ROUTING_MISC"
    if re.search(r"(memcpy|copy|transdata|format.?cast|reshape)", n):
        return "MEMORY_TRANSFORM"
    return "UNCLASSIFIED"


def _normalize(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.casefold())


def _field(row: dict, keys: tuple[str, ...]) -> str | None:
    normal = {_normalize(k): v for k, v in row.items() if k}
    for key in keys:
        if key in normal:
            return normal[key]
    return None


def _duration_us(row: dict) -> float:
    raw_ns = _field(row, ("durationns", "taskdurationns"))
    raw_us = _field(row, ("durationus", "taskdurationus", "deviceexecutiontimeus",
                          "duration", "taskduration"))
    if raw_ns is None and raw_us is None:
        raise ValueError("kernel_details.csv lacks Duration(us) / Duration(ns)")
    val = str(raw_ns if raw_ns is not None else raw_us).strip().replace(",", "")
    try:
        x = float(val)
    except ValueError as e:
        raise ValueError(f"non-numeric duration: {val}") from e
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"invalid duration {val}")
    return x / 1000 if raw_ns is not None else x


def _name(row: dict) -> str:
    v = _field(row, ("name", "opname", "kernelname", "taskname", "optype", "type"))
    if not v or not v.strip():
        raise ValueError("kernel row lacks op/kernel name")
    return v.strip()


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def validate_proof(proof: dict) -> None:
    for key, val in SHAPE.items():
        if proof.get(key) != val:
            raise ValueError(f"expected {key}={val!r}, got {proof.get(key)!r}")
    if proof.get("phase") != "decode" or proof.get("decode_phase_verified") is not True:
        raise ValueError("Decode-only profiling window was not verified")
    if proof.get("max_num_scheduled_tokens_per_request") != 1:
        raise ValueError("profile must isolate one-token-per-request Decode")
    if proof.get("running_requests") != 10:
        raise ValueError("cannot claim batch10 window if running_requests != 10")
    if not proof.get("source_trace") or not proof.get("source_server_log"):
        raise ValueError("must identify actual scheduler/trace evidence")
    if proof.get("rank_count") != 2:
        raise ValueError("TP2 requires two rank profiler files")
    if proof.get("failed_requests", 0) != 0:
        raise ValueError("failure-tainted capture cannot be clean attribution")


def aggregate(path: Path) -> dict:
    group = defaultdict(lambda: {"count": 0, "sum_kernel_us": 0.0})
    examples = defaultdict(Counter)
    total = 0.0
    events = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not row or not any(row.values()):
                continue
            name = _name(row)
            us = _duration_us(row)
            if not us:
                continue
            label = category(name)
            group[label]["count"] += 1
            group[label]["sum_kernel_us"] += us
            examples[label][name] += 1
            total += us
            events += 1
    if events < 10:
        raise ValueError(f"{path} has too few valid kernel events ({events})")
    cats = []
    for label, val in sorted(group.items(),
                             key=lambda x: -x[1]["sum_kernel_us"]):
        cats.append({
            "category": label,
            "count": val["count"],
            "sum_kernel_us": round(val["sum_kernel_us"], 3),
            "work_share_pct": round(100 * val["sum_kernel_us"] / total, 2),
            "examples": [n for n, _ in examples[label].most_common(8)],
        })
    return {"input_csv_sha256": _hash(path), "kernel_events": events,
            "summed_kernel_us": round(total, 3), "categories": cats}


def analyze(proof_path: Path, r0: Path, r1: Path) -> dict:
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    validate_proof(proof)
    if r0.resolve() == r1.resolve():
        raise ValueError("rank0 and rank1 must be different profiler files")
    ranks = {"rank0": aggregate(r0), "rank1": aggregate(r1)}
    dominant = {
        rank: [v["category"] for v in d["categories"][:5]]
        for rank, d in ranks.items()
    }
    return {
        "experiment": "C4_DECODE_COST_ATTRIBUTION",
        "shape": SHAPE,
        "window": {"source_proof_sha256": _hash(proof_path),
                   "phase": "decode", "actual_batch_verified": 10},
        "ranks": ranks,
        "top_categories_per_rank": dominant,
        "verdict": "WORK_VOLUME_RANKING_ONLY_NOT_LATENCY_CAUSALITY",
        "next": "Review kernel timeline, step_trace_time.csv and CPU gaps; "
                "time each leading candidate using controlled A/B, "
                "not by adding PMU activities or category durations.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proof", type=Path, required=True)
    p.add_argument("--rank0", type=Path, required=True)
    p.add_argument("--rank1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for x in (args.proof, args.rank0, args.rank1):
        if not x.is_file():
            p.error(f"missing input {x}")
    if args.output.exists():
        p.error(f"will not overwrite existing artifact {args.output}")
    data = analyze(args.proof, args.rank0, args.rank1)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"C4 PASS: {args.output} -- attribution only, no performance GO")


if __name__ == "__main__":
    main()
