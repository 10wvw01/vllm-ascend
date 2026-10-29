# SPDX-License-Identifier: Apache-2.0
"""Plan A device gates and full pool-return timing; called under the existing guard."""

import gc
import time

from quick_worker import write


def run_flow(
    session,
    a,
    dist,
    torch,
    xs,
    refs,
    partials,
    partial_refs,
    weight,
    checked,
    gate,
    measures,
    e0,
    e1,
    rank,
    outputs,
    stage,
):
    # Wrap metadata, downstream arithmetic, aliases, slot exhaustion, reuse, close.
    first = session.run_pool(partials[4096][0], group=1024, partial=True)
    checked(first, partial_refs[4096][0], "pool wrap")
    alias = first[:17]
    snapshot = outputs[4096]
    snapshot.copy_(first)
    del first
    second = session.run_pool(partials[4096][1], group=1024, partial=True)
    checked(second, partial_refs[4096][1], "second live slot")
    if not torch.equal(alias.view(torch.int16), snapshot[:17].view(torch.int16)):
        raise AssertionError("held output alias overwritten")
    # Asymmetric lifetime: rank1 frees its alias, rank0 keeps it.
    if rank == 1:
        del alias
        gc.collect()
    dist.barrier()
    try:
        session.run_pool(partials[4096][0], group=1024, partial=True)
    except RuntimeError as exc:
        if "release an output slot" not in str(exc):
            raise
    else:
        raise AssertionError("asymmetric held slot must reject allocation on both ranks")
    try:
        session.close()
    except RuntimeError as exc:
        if "release all outputs" not in str(exc):
            raise
    else:
        raise AssertionError("close destroyed live output")
    if rank == 0:
        del alias
    del second, snapshot
    gc.collect()
    gate("output_lifetime")
    for group in (512, 1024, 2048):
        # Smallest shape is a single chunk, exercising every drain boundary.
        for rows in (group, 2048, 4096):
            for i in range(a.protocol_reps):
                dist.barrier()
                v = i % 2
                out = session.run_pool(partials[4096][v][:rows], group=group, partial=True)
                checked(out, partial_refs[4096][v][:rows], "Plan A protocol")
                del out
    # group512 creates 25+ mails/wave; 48 waves crosses the 1024 arrival ring.
    for i in range(48):
        out = session.run_pool(partials[4096][i % 2], group=512, partial=True)
        checked(out, partial_refs[4096][i % 2], "Plan A wrap")
        del out
    gate("protocol")
    for m in (2048, 4096):
        for group in (512, 1024, 2048):
            out = session.run_pool(xs[m][0], weight, group=group)
            checked(out, refs[m][0], "Plan A fused")
            # Force a real downstream op to read format/Storage metadata.
            readback = outputs[m]
            torch.add(out, out, alpha=0, out=readback)
            checked(readback, refs[m][0], "ND consumer")
            del out, readback
    gate("fused")
    # Tail path includes explicit zero padding and output slicing; no hidden copy.
    stage.zero_()
    stage[:2049].copy_(xs[4096][0][:2049])
    out = session.run_pool(stage, weight, group=1024)
    checked(out[:2049], refs[4096][0][:2049], "Plan A tail")
    del out, stage
    gate("tail")
    for repeat in range(2):
        order = [(m, g) for m in (2048, 4096) for g in (512, 1024, 2048)]
        if repeat:
            order.reverse()
        for m, group in order:
            samples = []
            for i in range(a.warmup + a.samples):
                dist.barrier()
                v = i % 2
                start = time.perf_counter_ns()
                e0.record(session.stream)
                out = session.run_pool(xs[m][v], weight, group=group)
                e1.record(session.stream)
                session.stream.synchronize()
                wall = (time.perf_counter_ns() - start) / 1000
                event = e0.elapsed_time(e1) * 1000
                checked(out, refs[m][v], "Plan A timed")
                del out
                if i >= a.warmup:
                    samples.append(
                        {"i": i - a.warmup, "variant": v, "event_us": event, "wall_us": wall, "correct": True}
                    )
            row = {
                "M": m,
                "group": group,
                "pipeline": True,
                "repeat": repeat,
                "path": "V32_PLAN_A_POOL_RETURN",
                "samples": samples,
            }
            measures.append(row)
            write(a.out / f"timing_r{repeat}_m{m}_g{group}_rank{rank}.json", row)
