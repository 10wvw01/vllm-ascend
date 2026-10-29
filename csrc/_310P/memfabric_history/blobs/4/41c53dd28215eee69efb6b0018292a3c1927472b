# SPDX-License-Identifier: Apache-2.0
"""One full-input/output contract and conservative A3 cycle attribution."""

import math
from statistics import median

CONTRACT = "ND_X_NZ_W_TO_READY_ND_Y_V1"
STAGES = ("off-before", "host", "device", "off-after")


def percentile(values, p):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("missing/invalid timing")
    at = (len(values) - 1) * p
    lo = int(at)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)


def timed_call(op, stream, events, clock):
    """Tensor input prepared; returns Tensor output after its owning-stream join."""
    e0, e1 = events
    start = clock()
    e0.record(stream)
    out = op()
    e1.record(stream)
    stream.synchronize()
    wall = (clock() - start) / 1000
    return out, {"wall_us": wall, "event_us": e0.elapsed_time(e1) * 1000}


def device_clock_unavailable(raw):
    """dav-2002 (Ascend 310P3) has no AICore-readable cycle source: AscendC
    GetSystemCycle() is REPORT_NOT_SUPPORT there and returns 0, and the SYS_CNT
    instruction is rejected by the dav-2002 ISA. A fresh valid header with all
    stamp slots zero proves the kernel ran but the clock reads zero."""
    return raw[0] == 0x41335432 and not any(raw[8:])


def decode_trace(raw, rows, compute, comm, epoch=None):
    if len(raw) != 1024 or raw[0] != 0x41335432 or list(raw[1:4]) != [rows, compute, comm]:
        raise ValueError("wrong trace geometry/version")
    if epoch is not None and raw[4] != epoch:
        raise ValueError("stale trace epoch")
    if device_clock_unavailable(raw):
        raise ValueError("device clock unavailable: all stamps are zero")
    h, bands, groups = compute // 8, rows // (compute // 8), rows // comm

    def delta(a, b):
        if not a or not b or b < a:
            raise ValueError("missing/nonmonotonic stamp")
        return b - a

    seats = []
    for band in range(bands):
        t = raw[80 + band * 8 : 85 + band * 8]
        seats.append(
            dict(zip(("mm", "s_visibility", "seed_y_visibility", "publish"), (delta(a, b) for a, b in zip(t, t[1:]))))
        )
    cores = []
    for core in range(8):
        t = raw[16 + core * 8 : 20 + core * 8]
        cores.append(
            {
                "pre_final_join": delta(t[0], t[1]),
                "final_join": delta(t[1], t[2]),
                "output_visibility": delta(t[2], t[3]),
                "total": delta(t[0], t[3]),
            }
        )
    communication = []
    for g in range(groups):
        t = raw[592 + g * 16 : 608 + g * 16]
        # Original A3 records producer waits per compute wave, not per comm group.
        communication.append(
            {"ready_submit": delta(t[2], t[3]), "peer_ready_pump": delta(t[3], t[4]), "add_submit": delta(t[4], t[5])}
        )
    tail = dict(zip(("drain", "quiet", "done", "final_quiet"), (delta(raw[i], raw[i + 1]) for i in range(720, 724))))
    profile = raw[6]
    if profile not in (3302, 3303, 3304):
        raise ValueError("unknown A3 profile")
    leading = bands - ((bands + 7) // 8 if profile == 3304 else 0)
    for core, summary in enumerate(cores):
        owned = [
            seat
            for band, seat in enumerate(seats)
            if (band % 8 if profile == 3302 else 0 if band >= leading else band % 7 + 1) == core
        ]
        busy = {
            name: sum(seat[name] for seat in owned) for name in ("mm", "s_visibility", "seed_y_visibility", "publish")
        }
        summary["producer_ticks"] = busy
        summary["producer_fraction"] = {k: v / summary["total"] for k, v in busy.items()}
    wait_count = rows // compute if profile == 3302 else groups
    waits = [delta(raw[592 + g * 16], raw[593 + g * 16]) for g in range(wait_count)]
    # Do not mix clocks across cores or turn overlapping busy times into 100% shares.
    return {
        "units": "device_counter_ticks",
        "clock_cross_core_calibrated": False,
        "rows_per_seat": h,
        "seats": seats,
        "cores": cores,
        "communication": communication,
        "local_wait_ticks": waits,
        "tail": tail,
        "raw": list(raw),
        "warning": "MM includes Matmul movement/output; wait includes inbox processing, not pure network latency",
    }


def paired_report(a, b):
    for key in ("contract", "run_id", "configuration", "identity"):
        if a[key] != b[key]:
            raise ValueError(f"rank mismatch: {key}")
    if a["contract"] != CONTRACT:
        raise ValueError("unsupported timing contract")
    if [a["rank"], b["rank"]] != [0, 1] or any(x["status"] != "PASS" for x in (a, b)):
        raise ValueError("rank status/identity")
    config = a["configuration"]
    stages = STAGES if config["arm"].startswith("a3") else ("off-before", "off-after")
    expected = {
        (m, c, stage, repeat)
        for m in config["M"]
        for c in config["comm_rows"]
        for stage in stages
        for repeat in range(2)
    }
    for rank in (a, b):
        keys = [(x["M"], x["comm_rows"], x["stage"], x["repeat"]) for x in rank["cases"]]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("missing/duplicate cases")
    right = {(x["M"], x["comm_rows"], x["stage"], x["repeat"]): x for x in b["cases"]}
    result = []
    for x in a["cases"]:
        key = x["M"], x["comm_rows"], x["stage"], x["repeat"]
        y = right[key]
        if len(x["samples"]) != config["samples"] or len(y["samples"]) != config["samples"]:
            raise ValueError("wrong sample count")
        for i, (s, t) in enumerate(zip(x["samples"], y["samples"])):
            if s["i"] != i or t["i"] != i or not s["correct"] or not t["correct"]:
                raise ValueError("sample identity/correctness")
        metrics = {}
        for metric in ("wall_us", "event_us"):
            for sample in x["samples"] + y["samples"]:
                if not math.isfinite(sample[metric]) or sample[metric] <= 0:
                    raise ValueError("invalid rank timing")
            values = [max(s[metric], t[metric]) for s, t in zip(x["samples"], y["samples"])]
            metrics[metric] = {f"p{int(p * 100)}": percentile(values, p) for p in (0.5, 0.9, 0.99)}
        result.append({"M": key[0], "comm_rows": key[1], "stage": key[2], "repeat": key[3], **metrics})
    controls = []
    for m in config["M"]:
        for c in config["comm_rows"]:
            for metric in ("wall_us", "event_us"):
                center = {
                    stage: median(
                        x[metric]["p50"] for x in result if x["M"] == m and x["comm_rows"] == c and x["stage"] == stage
                    )
                    for stage in stages
                }
                drift = (
                    max(center["off-before"], center["off-after"]) / min(center["off-before"], center["off-after"]) - 1
                )
                round_drift = {
                    stage: max(
                        x[metric]["p50"] for x in result if x["M"] == m and x["comm_rows"] == c and x["stage"] == stage
                    )
                    / min(
                        x[metric]["p50"] for x in result if x["M"] == m and x["comm_rows"] == c and x["stage"] == stage
                    )
                    - 1
                    for stage in stages
                }
                controls.append(
                    {
                        "M": m,
                        "comm_rows": c,
                        "metric": metric,
                        "centers": center,
                        "bracket_drift": drift,
                        "round_drift": round_drift,
                        "stable_bracket": drift <= 0.05 and all(v <= 0.05 for v in round_drift.values()),
                        "instrumentation_delta": {
                            s: center[s] / median([center["off-before"], center["off-after"]]) - 1
                            for s in stages
                            if s not in ("off-before", "off-after")
                        },
                    }
                )
    attribution = []
    for rank in (a, b):
        for case in rank["cases"]:
            samples = case["samples"]
            item = {k: case[k] for k in ("M", "comm_rows", "stage", "repeat")}
            item["rank"] = rank["rank"]
            if case["stage"] == "host":
                if not all("host_us" in sample for sample in samples):
                    raise ValueError("missing Host attribution")
                item["host_phase_p50_us"] = {
                    name: median(sample["host_us"][name] for sample in samples)
                    for name in ("validate_reap", "reserve", "wrap", "enqueue", "input_event")
                }
            elif case["stage"] == "device":
                if not all("device" in sample for sample in samples):
                    raise ValueError("missing Device attribution")
                if all(sample["device"].get("unavailable") for sample in samples):
                    # 310P3 hardware has no AICore cycle source; recorded per session
                    # with raw-header evidence by the worker. Timing stages stay valid.
                    item["device_attribution"] = "UNAVAILABLE_DAV2002_NO_AICORE_CLOCK"
                elif any(sample["device"].get("unavailable") for sample in samples):
                    raise ValueError("mixed device clock availability within one case")
                else:
                    expected_profile = {"a3": 3302, "a3-2": 3303, "a3-tail": 3304}[config["arm"]]
                    for sample in samples:
                        raw = sample["device"]["raw"]
                        decoded = decode_trace(raw, case["M"], config["compute_rows"], case["comm_rows"])
                        if raw[5] != rank["rank"] or raw[6] != expected_profile or decoded != sample["device"]:
                            raise ValueError("trace rank/profile/derived evidence mismatch")
                    item["units"] = "device_counter_ticks"
                    item["core_p50_producer_fraction"] = [
                        {
                            name: median(sample["device"]["cores"][c]["producer_fraction"][name] for sample in samples)
                            for name in ("mm", "s_visibility", "seed_y_visibility", "publish")
                        }
                        for c in range(8)
                    ]
                    item["core0_tail_p50_ticks"] = {
                        name: median(sample["device"]["tail"][name] for sample in samples)
                        for name in ("drain", "quiet", "done", "final_quiet")
                    }
                    item["core0_local_wait_p50_ticks"] = median(
                        sum(sample["device"]["local_wait_ticks"]) for sample in samples
                    )
                    item["core0_communication_p50_ticks"] = {
                        name: median(sum(g[name] for g in sample["device"]["communication"]) for sample in samples)
                        for name in ("ready_submit", "peer_ready_pump", "add_submit")
                    }
            else:
                continue
            attribution.append(item)
    return {
        "contract": CONTRACT,
        "configuration": config,
        "identity": a["identity"],
        "formal_performance_go": False,
        "cases": result,
        "controls": controls,
        "attribution": attribution,
    }
