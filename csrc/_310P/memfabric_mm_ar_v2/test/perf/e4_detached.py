#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E4最小脱管启动/查询：父级OpenCode命令退出不影响独立Session的Campaign。

必须在同一隔离工作区、环境变量已设置且G0获准后调用。
从不杀外部PID；只管理本次worker启动的子进程组。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def guard_inputs(state: Path, out: Path, build: Path, repo_root: Path) -> str:
    if any(not p.is_absolute() for p in (state, out, build)):
        raise ValueError("state/out/build must all be absolute")
    if state.exists() or out.exists():
        raise ValueError("state and run output must be fresh, never overwrite evidence")
    if not state.parent.is_dir() or not out.parent.is_dir():
        raise ValueError("parents of state/out directories must already exist")
    if not (build / "manifest.json").is_file():
        raise ValueError("build manifest not found")
    if os.environ.get("E4_CONFIRM_APPROVED") == "YES":
        raise ValueError("this detached launcher authorizes Screen only; Confirm not permitted")
    if not os.environ.get("E4_PHYSICAL_CARD_ID", "").isdigit():
        raise ValueError("E4_PHYSICAL_CARD_ID=16384 (from fresh G0) required")
    head = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                               text=True).strip():
        raise ValueError("dirty tracked worktree")
    data = json.loads((build / "manifest.json").read_text(encoding="utf-8"))
    if data.get("work_head") != head or data.get("schema") != "E4_LAYERED_ADD_BUILD_V1":
        raise ValueError("stale/mismatched build manifest; rebuild E4 before starting")
    return head


def start(state: Path, out: Path, build: Path) -> int:
    perf = Path(__file__).resolve().parent
    root = perf.parents[3]
    head = guard_inputs(state, out, build, root)
    state.mkdir(mode=0o700)
    info = {"state": "STARTING", "repo_head": head, "out": str(out),
            "build": str(build), "created_utc": datetime.now(timezone.utc).isoformat(),
            "physical_card": os.environ["E4_PHYSICAL_CARD_ID"],
            "screen_only": True}
    atomic_json(state / "launch.json", info)
    with (state / "worker.log").open("ab") as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker",
             "--state", str(state), "--out", str(out), "--build", str(build)],
            start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    info["worker_pid"] = child.pid
    info["state"] = "LAUNCHED"
    atomic_json(state / "launch.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def stop_group(child: subprocess.Popen) -> None:
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


def worker(state: Path, out: Path, build: Path) -> int:
    perf = Path(__file__).resolve().parent
    cmd = ["bash", str(perf / "e4_campaign.sh"), str(out), str(build)]
    campaign_log = state / "campaign.log"
    child = None
    stopped = False

    def on_signal(signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True
        if child is not None:
            stop_group(child)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    code = 125
    try:
        with campaign_log.open("ab") as log:
            child = subprocess.Popen(cmd, start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, close_fds=True)
            code = child.wait()
    except BaseException as exc:
        if child is not None:
            stop_group(child)
        code = 128 + signal.SIGTERM if stopped else 125
        with (state / "worker.log").open("a", encoding="utf-8") as log:
            log.write(f"worker exception: {exc!r}\n")
    finally:
        progress = out / "progress.txt"
        progress_text = progress.read_text(encoding="utf-8") if progress.is_file() else ""
        accepted = code == 0 and "E4_SCREEN_COMPLETE / CONFIRM_NOT_AUTHORIZED" in progress_text
        final = {"state": "SCREEN_COMPLETE" if accepted else "STOP",
                 "exit_code": code if code else (0 if accepted else 96),
                 "out": str(out), "completed_utc": datetime.now(timezone.utc).isoformat(),
                 "progress_tail": progress_text[-1800:],
                 "confirm_authorized": False}
        atomic_json(state / "result.json", final)
    return code if code else (0 if accepted else 96)


def status(state: Path) -> int:
    p = state / "result.json"
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return 0 if d.get("state") == "SCREEN_COMPLETE" else 1
    launch = state / "launch.json"
    if not launch.is_file():
        print("UNKNOWN: missing launch.json", file=sys.stderr)
        return 2
    j = json.loads(launch.read_text(encoding="utf-8"))
    pid = j.get("worker_pid")
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, TypeError):
        print("STOP_OR_CRASH: worker absent, no result.json", file=sys.stderr)
        return 1
    print(json.dumps({"state": "RUNNING_OR_CLEANING", "worker_pid": pid,
                      "out": j.get("out")}, ensure_ascii=False))
    return 3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("start")
    for x in ("state", "out", "build"):
        a.add_argument("--" + x, type=Path, required=True)
    a = sub.add_parser("worker")
    for x in ("state", "out", "build"):
        a.add_argument("--" + x, type=Path, required=True)
    a = sub.add_parser("status")
    a.add_argument("--state", type=Path, required=True)
    opts = p.parse_args()
    if opts.cmd == "status":
        return status(opts.state)
    if opts.cmd == "worker":
        return worker(opts.state, opts.out, opts.build)
    try:
        return start(opts.state, opts.out, opts.build)
    except (ValueError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"E4_LAUNCH_STOP {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
