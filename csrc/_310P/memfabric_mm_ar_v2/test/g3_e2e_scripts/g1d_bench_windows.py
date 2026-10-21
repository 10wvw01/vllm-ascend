#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-D3 对 Gate E bench 逐轮汇总，只读离线性能分窗，不做协议故障因果归因。"""
import argparse
import json
import re
from pathlib import Path


def choose(d, *keys):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return d[k]
    return None


def summarize(root):
    rows = []
    for p in sorted(root.glob("stability_rep*.json"),
                    key=lambda p: int(re.search(r"rep(\d+)", p.name).group(1))
                    if re.search(r"rep(\d+)", p.name) else 0):
        try:
            d = json.loads(p.read_text())
        except (ValueError, OSError):
            rows.append({"file": p.name, "valid": False})
            continue
        rows.append({
            "file": p.name, "valid": True,
            "completed": choose(d, "completed", "completed_requests"),
            "failed": choose(d, "failed", "failed_requests", "errors"),
            "output_tok_s": choose(d, "output_throughput", "output_throughput_tok_s"),
            "ttft_p50_ms": choose(d, "median_ttft_ms", "p50_ttft_ms"),
            "ttft_p99_ms": choose(d, "p99_ttft_ms"),
            "tpot_p50_ms": choose(d, "median_tpot_ms", "p50_tpot_ms"),
            "tpot_p99_ms": choose(d, "p99_tpot_ms"),
            "request_throughput": choose(d, "request_throughput"),
        })
    return {
        "evidence_level": "BENCH_WINDOW_ONLY",
        "warning": ("每轮结果仅为独立时间窗口的汇总；没有每请求/每 seq 时间戳时，"
                    "不能把某个慢窗口直接归因于 MemFabric wait。Gate E 失败会话"
                    "不应参与健康稳态 throughput A/B。"),
        "windows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_dir", type=Path)
    p.add_argument("-o", "--output", type=Path)
    a = p.parse_args()
    payload = json.dumps(summarize(a.session_dir), ensure_ascii=False, indent=2) + "\n"
    if a.output:
        a.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
