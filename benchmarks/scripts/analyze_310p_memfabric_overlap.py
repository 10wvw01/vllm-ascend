# SPDX-License-Identifier: Apache-2.0
"""Analyze public-API MemFabric o_proj overlap from a CANN task_time CSV export.

Input: the ``task_time.csv`` produced offline from a torch_npu.profiler
text export of a fused ``memfabric_o_proj_allreduce`` workload (see the
development plan R3 notes for the exact collection recipe - the device
slices must be flushed by a normal process exit and the CSV is produced by
``torch_npu.profiler.profiler.analyse(<export_only_prof_dir>)`` in a
separate process).

Output: per-call fused wave timeline and the steady-state overlap argument:

  - producer (AICore matmul+clean+post, multi-block) window
  - waiter (quiet + wait x chunks) window: the proof point. Chunks are
    128 KiB each; a wave of N chunks has moved N*128 KiB by the time the
    waiter's public wait loop drains its mails. If the transfer had to happen
    serially after the producer, the waiter window would imply an SDMA
    bandwidth far beyond the physical limit - i.e. the epoch kernel's SDMA
    engine moves the chunks while the producer's matmuls are still running
    (device-side overlap).
  - add / gate / ack windows.

Usage:
  python3 benchmarks/scripts/analyze_310p_memfabric_overlap.py <task_time.csv> \
      [--chunk-kib 128] [--bandwidth-gbps 20]
"""

from __future__ import annotations

import argparse
import contextlib
import csv

PRODUCER_MARK = "Mf310pDirectProducerKernel"
WAITER_MARK = "mf310pWaitKernel"
ADD_MARK = "mf310pAddKernel"
GATE_MARK = "mf310pGateKernel"
ACK_MARK = "mf310pAckKernel"


def load(path: str) -> list[dict]:
    evs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            with contextlib.suppress(ValueError, KeyError):
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
    return evs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--chunk-kib", type=int, default=128, help="send-arena chunk size in KiB")
    ap.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=20.0,
        help="assumed sustainable SDMA bandwidth for the serial lower bound (GB/s)",
    )
    ap.add_argument("--rows", type=int, default=2048, help="rows per fused call (for the chunk count)")
    ap.add_argument("--tile-m", type=int, default=32, help="tile_m (rows per chunk)")
    args = ap.parse_args()

    evs = load(args.csv)
    by = {
        mark: sorted((e for e in evs if mark in e["name"]), key=lambda e: e["start"])
        for mark in (PRODUCER_MARK, WAITER_MARK, ADD_MARK, GATE_MARK, ACK_MARK)
    }
    print(
        f"events: producer={len(by[PRODUCER_MARK])} "
        f"waiter={len(by[WAITER_MARK])} add={len(by[ADD_MARK])} "
        f"gate={len(by[GATE_MARK])} ack={len(by[ACK_MARK])}"
    )
    if not by[PRODUCER_MARK] or not by[WAITER_MARK]:
        raise SystemExit("no fused o_proj events found; wrong trace?")

    chunks = (args.rows + args.tile_m - 1) // args.tile_m
    wave_bytes = chunks * args.chunk_kib * 1024
    print(
        f"wave shape: rows={args.rows} tile_m={args.tile_m} chunks={chunks} ({wave_bytes / 1024 / 1024:.1f} MiB/rank)\n"
    )

    calls = min(len(by[PRODUCER_MARK]), len(by[WAITER_MARK]))
    print(f"{'#':>2} {'gate_us':>8} {'producer_us':>11} {'waiter_us':>9} {'add_us':>7} {'ack_us':>6} {'cycle_us':>9}")
    steady = []
    for i in range(calls):
        g = by[GATE_MARK][i] if i < len(by[GATE_MARK]) else None
        p, w, a, k = (by[PRODUCER_MARK][i], by[WAITER_MARK][i], by[ADD_MARK][i], by[ACK_MARK][i])
        cycle = k["stop"] - (g["start"] if g else p["start"])
        print(
            f"{i:>2} {(g['dur'] if g else 0):>8.1f} {p['dur']:>11.1f} "
            f"{w['dur']:>9.1f} {a['dur']:>7.1f} {k['dur']:>6.1f} "
            f"{cycle:>9.1f}"
        )
        if i > 0:  # drop the first call (tail of warmup effects)
            steady.append((p["dur"], w["dur"], a["dur"], cycle))

    sp = [s[0] for s in steady]
    sw = [s[1] for s in steady]
    sc = [s[3] for s in steady]
    producer_med = sorted(sp)[len(sp) // 2]
    waiter_med = sorted(sw)[len(sw) // 2]
    cycle_med = sorted(sc)[len(sc) // 2]

    # Overlap argument: the waiter drains all chunk mails within its
    # window, so the whole wave payload crossed the fabric by then. A
    # serial transfer would need the full SDMA time inside (or after) the
    # waiter window.
    implied_gbps = wave_bytes / (waiter_med * 1e-6) / 1e9
    serial_sdma_us = wave_bytes / (args.bandwidth_gbps * 1e9) * 1e6
    serial_cycle_us = producer_med + serial_sdma_us + (cycle_med - producer_med - waiter_med)

    print(f"\nsteady medians: producer={producer_med:.0f}us waiter={waiter_med:.0f}us cycle={cycle_med:.0f}us")
    print(
        f"bandwidth implied if transfer fit inside the waiter window: "
        f"{implied_gbps:.0f} GB/s (physical SDMA assumption "
        f"{args.bandwidth_gbps:.0f} GB/s)"
    )
    print(
        f"serial lower bound (matmul + {serial_sdma_us:.0f}us SDMA + "
        f"reduce tail): {serial_cycle_us:.0f}us vs measured cycle "
        f"{cycle_med:.0f}us"
    )
    if implied_gbps > 2 * args.bandwidth_gbps and cycle_med < serial_cycle_us:
        print(
            "\nOVERLAP PROVEN: the wave payload cannot fit the waiter "
            "window at physical bandwidth, so the epoch kernel's SDMA "
            "transfers ran while the producer matmuls were still "
            "executing (measured cycle also beats the serial lower "
            "bound)."
        )
    else:
        print(
            "\nINCONCLUSIVE: waiter window does not rule out serial "
            "transfer at the assumed bandwidth; lower --bandwidth-gbps "
            "or inspect the timeline manually."
        )


if __name__ == "__main__":
    main()
