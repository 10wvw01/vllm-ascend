#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E4最小现场守护：目标卡双Die只读检查 + 每阶段运行中的主动STOP。

本守护不是排他锁：容器PID命名空间可能无法映射宿主PID；仅对可观测的
跨卡进程/目标卡进程数量异常 fail-closed，目标卡单独入侵仍需真实独占租约保护。
只终止自己启动的torchrun进程组，绝不kill外部租户。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

PROC = re.compile(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|")
CHIP = re.compile(r"^\|\s*(\d+)\s+310P3\s*\|\s*(OK|Alarm|Warning|Fault)\s*\|")
STOP_EXIT = 97


class LeaseLost(RuntimeError):
    pass


def parse_info(raw: str, card: int) -> dict:
    """根据npu-smi 26.1.1的进程表而非内存估计身份。解析失败即STOP。"""
    if "Process id" not in raw or "Chip" not in raw:
        raise LeaseLost("npu-smi process header missing")
    head, tail = raw.split("Process id", 1)
    health = [state for line in head.splitlines()
              for n, state in CHIP.findall(line) if int(n) == card]
    if len(health) != 2 or any(x != "OK" for x in health):
        raise LeaseLost(f"target card {card} health entries={health!r}, need two OK dies")
    by_card: dict[int, set[int]] = {}
    for line in tail.splitlines():
        for n, die, pid in PROC.findall(line):
            n, die, pid = int(n), int(die), int(pid)
            if die not in (0, 1) or pid <= 0:
                raise LeaseLost("malformed process row")
            by_card.setdefault(n, set()).add(pid)
    return {"card": card, "target": by_card.get(card, set()),
            "outside": set().union(*(v for k, v in by_card.items() if k != card)),
            "all": by_card}


def assess(snap: dict, *, stage_running: bool, max_rank_pids: int = 2) -> dict:
    target = snap["target"]
    outside = snap["outside"]
    if not stage_running and target:
        raise LeaseLost(f"target must be idle before stage, found PIDs={sorted(target)}")
    cross = target & outside
    if cross:
        raise LeaseLost(f"external cross-card PID on target: {sorted(cross)}")
    if len(target) > max_rank_pids:
        raise LeaseLost(f"unexpected target process count {len(target)}>{max_rank_pids}: {sorted(target)}")
    return {"target_pids": sorted(target),
            "cross_card": sorted(cross),
            "target_pid_origin": "unverified_host_namespace_requires_exclusive_lease"}


def sample(card: int) -> tuple[str, dict]:
    try:
        run = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=8, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeaseLost(f"npu-smi unavailable/failed: {exc}") from exc
    return run.stdout, parse_info(run.stdout, card)


def record(path: Path, event: str, **fields: object) -> None:
    entry = {"at_utc": datetime.now(timezone.utc).isoformat(),
             "event": event, **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stop_owned(child: subprocess.Popen) -> None:
    """只信任本脚本新建的独立PGID；不根据npu-smi PID杀进程。"""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run_stage(card: int, log: Path, poll: float, command: list[str]) -> int:
    if not log.is_absolute() or not log.parent.is_dir() or log.is_symlink():
        raise ValueError("absolute log under an existing directory required")
    if not command:
        raise ValueError("stage command required")
    record(log, "PREFLIGHT_START", argv=command, card=card)
    try:
        _, before = sample(card)
        checked = assess(before, stage_running=False)
    except LeaseLost as exc:
        record(log, "STOP", phase="PREFLIGHT", reason=str(exc))
        raise
    record(log, "PREFLIGHT_PASS", **checked)
    child = subprocess.Popen(command, start_new_session=True)
    record(log, "STAGE_STARTED", child_pid=child.pid)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    def interrupted(signum: int, _frame: object) -> None:
        raise LeaseLost(f"guard received signal {signum}; stop owned stage")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            _, snapshot = sample(card)
            verified = assess(snapshot, stage_running=True)
            record(log, "LEASE_SAMPLE", **verified)
            ret = child.poll()
            if ret is not None:
                record(log, "STAGE_EXIT", exit_code=ret)
                return ret
            time.sleep(poll)
    except BaseException as exc:
        record(log, "STOP", reason=str(exc))
        stop_owned(child)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return STOP_EXIT
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", type=int, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--poll-sec", type=float, default=2.0)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.card < 0 or not (0.5 <= a.poll_sec <= 10):
        p.error("invalid physical NPU card or polling interval")
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    try:
        return run_stage(a.card, a.log, a.poll_sec, cmd)
    except (ValueError, LeaseLost) as exc:
        print(f"E4_LEASE_STOP {exc}", file=sys.stderr, flush=True)
        return STOP_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
