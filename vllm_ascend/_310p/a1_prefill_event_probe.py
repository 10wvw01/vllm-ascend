# SPDX-License-Identifier: Apache-2.0
"""A1-only no-profiler NPU Event observer; never runs unless explicitly opted in.

Records the device interval from model forward entry through logits production.
Does NOT claim to measure scheduler queue/host input preparation/sampling,
or independent-stream work that is not dependent on the recorded stream.
A later already-existing synchronization must complete the events; this
module never invokes event/stream/device.synchronize().
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def _target() -> Path | None:
    value = os.environ.get("A1_310P_PREFILL_EVENTS_DIR")
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute() or not p.is_dir() or p.is_symlink():
        raise RuntimeError("A1 event directory must be an existing private absolute directory")
    return p


def begin(runner: Any, scheduler_output: Any) -> dict | None:
    """Enqueue start Event on the current NPU stream; never synchronize."""
    root = _target()
    if root is None or not bool(getattr(runner, "with_prefill", False)):
        return None
    import torch
    start = torch.npu.Event(enable_timing=True)
    start.record(torch.npu.current_stream())
    step_id = getattr(runner, "_a1_prefill_counter", 0)
    runner._a1_prefill_counter = step_id + 1
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_initialized():
        rank = str(dist.get_rank())
    else:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "UNKNOWN"))
    return {
        "start": start, "step_id": step_id,
        "num_scheduled_tokens": int(scheduler_output.total_num_scheduled_tokens),
        "root": root,
        "rank": rank,
    }


def end(runner: Any, sample: dict | None) -> None:
    """Enqueue end Event after logits are produced, no host/device sync."""
    if sample is None:
        return
    import torch
    finish = torch.npu.Event(enable_timing=True)
    finish.record(torch.npu.current_stream())
    sample["finish"] = finish
    pending = getattr(runner, "_a1_prefill_pending", None)
    if pending is None:
        pending = []
        runner._a1_prefill_pending = pending
    pending.append(sample)


def drain(runner: Any) -> None:
    """After natural bookkeeping sync, read only *completed* events."""
    pending = getattr(runner, "_a1_prefill_pending", None)
    if not pending:
        return
    keep = []
    for item in pending:
        if not item["start"].query() or not item["finish"].query():
            keep.append(item)
            continue
        value = float(item["start"].elapsed_time(item["finish"]))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("A1 Prefill event duration invalid")
        rank = item["rank"]
        if not rank.isdecimal():
            raise RuntimeError("A1 rank identity invalid")
        out = item["root"] / f"a1_prefill_rank{rank}_pid{os.getpid()}.jsonl"
        entry = {
            "schema": "A1_PREFILL_FORWARD_TO_LOGITS_EVENT_V1",
            "rank": int(rank), "pid": os.getpid(),
            "step_id": item["step_id"],
            "num_scheduled_tokens": item["num_scheduled_tokens"],
            "forward_to_logits_device_ms": round(value, 6),
            "profiler": False,
            "synchronization_inserted": False,
            "scope": "current-stream forward-entry to logits; full-step/other-stream completion UNPROVEN",
        }
        with out.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    runner._a1_prefill_pending = keep
