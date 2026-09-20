# SPDX-License-Identifier: Apache-2.0
"""Analyze MemFabric o_proj overlap from a CANN task_time CSV export.

Input: the ``task_time_*.csv`` produced by a torch_npu.profiler text export
(``export_only_prof_dir/<host>_ascend_pt/PROF_*/mindstudio_profiler_output``)
of a fused ``memfabric_o_proj_allreduce`` workload.

Output: per-wave overlap report proving (or refuting) that later matmul
tiles execute on the compute stream while the reduce-consumer kernel and the
AICPU SDMA orchestrator run on their own streams (R4 acceptance evidence).

Usage:
  python3 benchmarks/scripts/analyze_310p_memfabric_overlap.py <task_time.csv>
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

# Wave boundary heuristic: matmul gap larger than this (us) starts a new wave.
WAVE_GAP_US = 3000.0


def load(path: str) -> list[dict]:
    evs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                evs.append(
                    dict(
                        name=r["kernel_name"],
                        ktype=r["kernel_type"],
                        stream=int(r["stream_id"]),
                        dur=float(r["task_time(us)"]),
                        start=float(r["task_start(us)"].strip()),
                        stop=float(r["task_stop(us)"].strip()),
                    )
                )
            except (ValueError, KeyError):
                pass
    return evs


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    evs = load(sys.argv[1])
    by = defaultdict(list)
    for e in evs:
        by[e["name"]].append(e)

    mm = sorted([e for e in evs if "te_matmul" in e["name"]], key=lambda e: e["start"])
    if not mm:
        raise SystemExit("no matmul events found; is this a fused o_proj trace?")
    pub = [e for e in evs if e["name"] == "mf310pPublishKernel"]
    red = [e for e in evs if e["name"] == "mf310pReduceConsumerKernel"]
    aicpu = [e for e in evs if e["name"] == "HybmSdmaOrchBatch"]
    print(f"events: matmul={len(mm)} publish={len(pub)} reduce={len(red)} "
          f"aicpu_orchestrator={len(aicpu)}")

    waves = []
    cur = [mm[0]]
    for e in mm[1:]:
        if e["start"] - cur[-1]["stop"] > WAVE_GAP_US:
            waves.append(cur)
            cur = [e]
        else:
            cur.append(e)
    waves.append(cur)
    print(f"waves detected: {[len(w) for w in waves]}")

    # The last wave is post-warmup in our capture scripts.
    w = waves[-1]
    ws, we = w[0]["start"], w[-1]["stop"]
    act = red + aicpu
    print(f"\n== wave: {len(w)} matmul tiles, compute span {(we - ws)/1e3:.3f} ms, "
          f"tile avg {sum(e['dur'] for e in w)/len(w):.1f} us ==")
    for label, seq in (("reduce consumer", red), ("AICPU orchestrator", aicpu)):
        for e in seq:
            if e["start"] < we and e["stop"] > ws:
                ov = min(we, e["stop"]) - max(ws, e["start"])
                print(f"{label}: active window overlaps compute span by {ov/1e3:.3f} ms "
                      f"(kernel dur {e['dur']/1e3:.3f} ms, stream {e['stream']})")
    n_overlap = sum(
        1
        for i in range(1, len(w))
        if any(a["start"] < w[i]["stop"] and a["stop"] > w[i]["start"] for a in act)
    )
    total = len(w) - 1
    print(f"\nmatmul tiles (t>=1) overlapping reduce/AICPU activity: {n_overlap}/{total}")
    if n_overlap == total and total > 0:
        print("OVERLAP PROVEN: every later matmul tile ran while the "
              "communication/reduction streams were active.")
    elif n_overlap == 0:
        print("NO OVERLAP: compute waited for communication to finish.")
    else:
        print(f"PARTIAL OVERLAP: {n_overlap}/{total} tiles.")


if __name__ == "__main__":
    main()
