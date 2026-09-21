# SPDX-License-Identifier: Apache-2.0
"""Analyze MM/SDMA overlap for the current 310P3 MemFabric o_proj pipeline.

Input: CANN task_time.csv captured from the current fused workload.

The producer kernel contains seven MM workers and one communication
coordinator. signal() asynchronously submits already-ready chunks, so SDMA for
earlier chunks can progress while later chunks are still being computed.

The script reports gate/producer/waiter/add/ack timing and uses the waiter
duration plus an assumed sustainable SDMA bandwidth as a lower-bound argument.
For final acceptance, also inspect the CANN timeline directly.

Usage:
  python3 benchmarks/scripts/analyze_310p_memfabric_overlap.py task_time.csv \
      --rows 2048 --tile-m 32 --bandwidth-gbps 20
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
        for row in csv.DictReader(f):
            with contextlib.suppress(ValueError, KeyError):
                evs.append(
                    dict(
                        name=row["kernel_name"],
                        ktype=row["kernel_type"],
                        stream=int(row["stream_id"]),
                        dur=float(row["task_time(us)"]),
                        start=float(row["task_start(us)"].strip()),
                        stop=float(row["task_stop(us)"].strip()),
                    )
                )
    return evs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--chunk-kib", type=int, default=128)
    ap.add_argument("--bandwidth-gbps", type=float, default=20.0)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--tile-m", type=int, default=32)
    args = ap.parse_args()

    evs = load(args.csv)
    by = {
        mark: sorted(
            (e for e in evs if mark in e["name"]), key=lambda e: e["start"]
        )
        for mark in (
            PRODUCER_MARK,
            WAITER_MARK,
            ADD_MARK,
            GATE_MARK,
            ACK_MARK,
        )
    }

    print(
        f"events: producer={len(by[PRODUCER_MARK])} "
        f"waiter={len(by[WAITER_MARK])} "
        f"add={len(by[ADD_MARK])} "
        f"gate={len(by[GATE_MARK])} "
        f"ack={len(by[ACK_MARK])}"
    )
    if not by[PRODUCER_MARK] or not by[WAITER_MARK]:
        raise SystemExit("no current fused o_proj events found")

    if args.rows % args.tile_m != 0:
        raise SystemExit(
            "this analyzer currently requires rows % tile_m == 0 so each "
            "wave has one producer launch; use a divisible performance shape"
        )

    total_chunks = args.rows // args.tile_m
    max_wave_chunks = 64
    waves_per_call = (total_chunks + max_wave_chunks - 1) // max_wave_chunks
    wave_chunks = [
        min(max_wave_chunks, total_chunks - i * max_wave_chunks)
        for i in range(waves_per_call)
    ]
    print(
        f"shape: rows={args.rows} tile_m={args.tile_m} "
        f"total_chunks={total_chunks} waves={wave_chunks}\n"
    )

    calls = min(len(by[PRODUCER_MARK]), len(by[WAITER_MARK]))
    print(
        f"{'#':>2} {'gate_us':>8} {'producer_us':>11} "
        f"{'waiter_us':>9} {'add_us':>7} {'ack_us':>7} {'cycle_us':>9}"
    )

    steady = []
    for i in range(calls):
        gate = by[GATE_MARK][i] if i < len(by[GATE_MARK]) else None
        prod = by[PRODUCER_MARK][i]
        wait = by[WAITER_MARK][i]
        add = by[ADD_MARK][i]
        ack = by[ACK_MARK][i]
        cycle = ack["stop"] - (gate["start"] if gate else prod["start"])
        chunks_this_wave = wave_chunks[i % waves_per_call]
        bytes_this_wave = chunks_this_wave * args.chunk_kib * 1024
        print(
            f"{i:>2} {(gate['dur'] if gate else 0):>8.1f} "
            f"{prod['dur']:>11.1f} {wait['dur']:>9.1f} "
            f"{add['dur']:>7.1f} {ack['dur']:>7.1f} {cycle:>9.1f}"
        )
        if i >= waves_per_call:  # drop the first full fused call
            steady.append(
                (prod["dur"], wait["dur"], add["dur"], cycle, bytes_this_wave)
            )

    if not steady:
        raise SystemExit("need at least two fused calls for steady-state analysis")

    # Evaluate each steady-state wave with its actual payload. Multiple-wave
    # calls (for example rows=4096,tile_m=32) must not be treated as one 128
    # chunk transfer.
    records = []
    for prod_us, wait_us, add_us, cycle_us, wave_bytes in steady:
        implied = wave_bytes / (wait_us * 1e-6) / 1e9
        serial_sdma = wave_bytes / (args.bandwidth_gbps * 1e9) * 1e6
        serial_cycle = prod_us + serial_sdma + (cycle_us - prod_us - wait_us)
        records.append((prod_us, wait_us, cycle_us, implied, serial_cycle))

    producer_med = sorted(x[0] for x in records)[len(records) // 2]
    waiter_med = sorted(x[1] for x in records)[len(records) // 2]
    cycle_med = sorted(x[2] for x in records)[len(records) // 2]
    implied_gbps = sorted(x[3] for x in records)[len(records) // 2]
    serial_cycle_us = sorted(x[4] for x in records)[len(records) // 2]

    print(
        f"\nsteady medians: producer={producer_med:.0f}us "
        f"waiter={waiter_med:.0f}us cycle={cycle_med:.0f}us"
    )
    print(
        "waiter-only bandwidth implication: "
        f"{implied_gbps:.1f} GB/s; assumed sustainable SDMA "
        f"{args.bandwidth_gbps:.1f} GB/s"
    )
    print(
        f"median serial lower bound: {serial_cycle_us:.0f}us; "
        f"median measured cycle: {cycle_med:.0f}us"
    )

    if (
        implied_gbps > 2 * args.bandwidth_gbps
        and cycle_med < serial_cycle_us
    ):
        print(
            "\nOVERLAP EVIDENCE: the payload cannot be explained as a "
            "purely serial post-producer transfer at the assumed bandwidth. "
            "Confirm with the CANN timeline before final acceptance."
        )
    else:
        print(
            "\nINCONCLUSIVE: this lower-bound test alone does not prove "
            "overlap. Inspect the CANN timeline."
        )


if __name__ == "__main__":
    main()
