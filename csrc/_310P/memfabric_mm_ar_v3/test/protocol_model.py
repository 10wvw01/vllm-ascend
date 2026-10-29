# SPDX-License-Identifier: Apache-2.0
"""CPU executable specification, not a substitute for FP16/IB device tests."""

from __future__ import annotations

import random


def tag(phase, epoch, group, packet):
    if not (1 <= phase <= 5 and 1 <= epoch < 1 << 24 and 0 <= group < 64 and 0 <= packet < 64):
        raise ValueError("tag field overflow")
    return phase << 36 | epoch << 12 | group << 6 | packet


def verify_mail(mail, expected, seen):
    if mail != expected or mail[2] in seen:
        raise ValueError("wrong address/length/tag or duplicate; ADD is never retried")
    seen.add(mail[2])


def simulate(seed, groups=2, packets=2, pipeline=True):
    """Random legal interleavings, in-place sources, asymmetric progress, FIFO mail streams.

    Tracks source versions: a GATHER must not overwrite any pending REDUCE source.
    Kernel waits are cooperative here; missing predecessors raise a deadlock error.
    """
    rng = random.Random(seed)
    original = [[[10000 * r + 100 * g + j for j in range(2 * packets)] for g in range(groups)] for r in range(2)]
    work = [[list(a) for a in b] for b in original]
    queues = [[], []]
    arrivals = [[], []]
    program = []
    for rank in range(2):
        ops = []
        for g in range(groups):
            ops.append(("mm", g))
            if pipeline and g:
                ops.extend((op, g - 1) for op in ("wait_reduce", "quiet", "gather", "wait_gather", "quiet", "copy"))
            ops.extend((op, g) for op in ("ready", "wait_ready", "reduce"))
            if not pipeline:
                ops.extend((op, g) for op in ("wait_reduce", "quiet", "gather", "wait_gather", "quiet", "copy"))
        if pipeline:
            ops.extend((op, groups - 1) for op in ("wait_reduce", "quiet", "gather", "wait_gather", "quiet", "copy"))
        program.append(ops)
    pc = [0, 0]
    complete = set()
    outputs = {}
    for _ in range(10000):
        actions = []
        for r in range(2):
            if queues[r]:
                actions.append(("sdma", r))
            if pc[r] < len(program[r]):
                op, g = program[r][pc[r]]
                if op.startswith("wait_"):
                    phase = op[5:]
                    needed = 1 if phase == "ready" else packets
                    if len(arrivals[r]) < needed:
                        continue
                    if any(z != (phase, g) for z in arrivals[r][:needed]):
                        raise AssertionError("mail phase/order mismatch")
                if op == "quiet" and queues[r]:
                    continue
                actions.append(("core", r))
        if not actions:
            if all(pc[r] == len(program[r]) for r in range(2)) and not any(queues):
                break
            raise AssertionError("deadlock")
        kind, r = rng.choice(actions)
        peer = 1 - r
        if kind == "sdma":
            phase, g, part, value = queues[r].pop(0)
            if phase == "reduce":
                j = peer * packets + part
                if work[r][g][j] != value:
                    raise AssertionError("REDUCE source overwritten before read")
                work[peer][g][j] += work[r][g][j]
                complete.add((r, g, j))
            elif phase == "gather":
                j = r * packets + part
                if (peer, g, j) not in complete:
                    raise AssertionError("GATHER overwrites a still-live peer source")
                work[peer][g][j] = work[r][g][j]
            arrivals[peer].append((phase, g))
            continue
        op, g = program[r][pc[r]]
        pc[r] += 1
        if op in ("ready", "reduce", "gather"):
            phase = op
            count = 1 if phase == "ready" else packets
            for p in range(count):
                value = work[r][g][peer * packets + p] if phase == "reduce" else None
                queues[r].append((phase, g, p, value))
        elif op.startswith("wait_"):
            count = 1 if op == "wait_ready" else packets
            del arrivals[r][:count]
        elif op == "copy":
            outputs[r, g] = list(work[r][g])
    else:
        raise AssertionError("unbounded protocol")
    for r in range(2):
        for g in range(groups):
            assert outputs[r, g] == [a + b for a, b in zip(original[0][g], original[1][g])]
    assert not any(arrivals)
    return outputs
