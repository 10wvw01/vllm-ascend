# SPDX-License-Identifier: Apache-2.0
"""Bounded timing units. This module performs no barrier, comparison or status read."""
import math
import random
import statistics


def groups(total, size):
    return [(first, min(size, total-first)) for first in range(0, total, size)]


def units(total, size, scenario):
    return groups(total, 1 if scenario == 'ready' else size)


def timed(op, stream, events, clock, host=False):
    start, end = events
    begin = clock()
    start.record(stream)
    call_begin = clock() if host else None
    output = op()
    call_end = clock() if host else None
    end.record(stream)
    stream.synchronize()
    row = dict(wall_us=(clock()-begin)/1000, event_us=start.elapsed_time(end)*1000)
    if host:
        row['collection'] = {'host_us': {'adapter_call_including_blocking': (call_end-call_begin)/1000}}
    return output, row


def measure_group(op, first, count, scenario, stream, events, clock, host=False):
    """Retain ALL outputs until the caller joins/checks/releases the completed group."""
    outputs, rows = [], []
    if scenario == 'ready':
        for i in range(first, first+count):
            output, row = timed(lambda: op(i), stream, events, clock, host)
            outputs.append(output)
            rows.append(dict(i=i, calls=1, **row))
    elif scenario == 'steady':
        start, end = events
        begin = clock()
        start.record(stream)
        call_begin = clock() if host else None
        for i in range(first, first+count):
            outputs.append(op(i))
        call_end = clock() if host else None
        end.record(stream)
        stream.synchronize()
        row = dict(i=first, calls=count, wall_us=(clock()-begin)/1000,
                   event_us=start.elapsed_time(end)*1000)
        if host:
            row['collection'] = {'host_us': {'adapter_call_including_blocking': (call_end-call_begin)/1000/count}}
        rows.append(row)
    else:
        raise ValueError('unknown measurement scenario')
    return outputs, rows


def disjoint(spans):
    ordered = sorted(spans)
    return all(start > 0 and end > start for start, end in ordered) and all(
        right[0] >= left[1] for left, right in zip(ordered, ordered[1:]))


def warmup_quality(values):
    """A timing convergence observation, NOT a hardware clock/temperature certificate."""
    if len(values) < 6:
        return dict(status='UNVERIFIED_SHORT_WARMUP', samples=len(values))
    tail = values[-8:]
    center = statistics.median(tail)
    deviation = (max(tail)-min(tail))/center if center else math.inf
    return dict(status='STABLE_TIMING_WINDOW' if deviation <= .1 else 'UNSTABLE_TIMING_WINDOW',
                samples=len(values), tail=tail, relative_span=deviation,
                hardware_frequency_and_temperature='NOT_CERTIFIED')


def speedup_interval(stock, candidate, seed=4242):
    """Resample independent paired rounds, never individual correlated calls."""
    if len(stock) != len(candidate) or not stock or any(x <= 0 for x in stock+candidate):
        raise ValueError('invalid paired rounds')
    if len(stock) < 3:
        return dict(status='INSUFFICIENT_INDEPENDENT_ROUNDS', rounds=len(stock), low=None, high=None)
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        indexes = [rng.randrange(len(stock)) for _ in stock]
        draws.append(math.fsum(stock[i] for i in indexes)/math.fsum(candidate[i] for i in indexes))
    draws.sort()
    return dict(status='PAIRED_ROUND_BOOTSTRAP', rounds=len(stock), confidence=.95,
                low=draws[49], high=draws[1949], replicates=2000,
                interpretation='conditional on measured rounds; does not remove systematic bias')
