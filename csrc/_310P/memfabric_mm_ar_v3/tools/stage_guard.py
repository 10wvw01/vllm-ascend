#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V3现场守护（从E4 5cff1b2d逐文件回收）：目标卡双Die只读检查 + 每阶段运行中的主动STOP。

本守护不是排他锁：容器PID命名空间可能无法映射宿主PID；仅对可观测的
跨卡进程/目标卡进程数量异常 fail-closed，目标卡单独入侵仍需真实独占租约保护。
只终止自己启动的torchrun进程组，绝不kill外部租户。

部署窗口例外（V3 2026-10-29）：库强制 fork+exec `orch_deploy_helper` 完成设备侧
kernel 部署——两个短命助手子进程与两个 Rank 同卡出现是本方案的合法形态，npu-smi
宿主 PID 无法映射回容器进程，故以 worker 声明的 `deploy_window_rank*.flag` 标记
识别该窗口：窗口内允许至多 max+2 个目标卡进程且持续超上限不得超过有界时限；
无标记或持续超限仍立即 STOP，跨卡/健康/解析失败不受窗口影响。

窗口自有形态扩展（V3.1 2026-10-29，诊断轮 STOP 根因修复）：torch_npu 在进程首个
设备算子处初始化其异步编译基础设施，会从 worker fork 出携带继承设备映射的短命
子进程（实测一次在驱动进程表短暂浮现，宿主 PID 4158731，~103MB，存活约8秒），
同样无法映射回容器。worker 现已把窗口标记提前到首个设备算子之前声明；窗口内容忍
上限相应为 max_rank_pids + deploy_extra_pids + window_extra_pids（2+2+2=6，覆盖
两 Rank、两助手与至多两个编译基础设施形态）。超出该上限、无标记超额或持续超时仍
立即 STOP；60 秒持续界限与全部 fail-closed 检查不变。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

PROC = re.compile(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|")
CHIP = re.compile(r"^\|\s*(\d+)\s+310P3\s*\|\s*(OK|Alarm|Warning|Fault)\s*\|")
STOP_EXIT = 97


class LeaseLost(RuntimeError):
    pass


class ExtraTargetPids(LeaseLost):
    """目标卡进程数超限（部署窗口内可被有界容忍的自有助手子进程形态）。"""


def parse_info(raw: str, card: int) -> dict:
    """根据npu-smi 26.1.1的进程表而非内存估计身份。解析失败即STOP。"""
    if "Process id" not in raw or "Chip" not in raw:
        raise LeaseLost("npu-smi process header missing")
    head, tail = raw.split("Process id", 1)
    health = [state for line in head.splitlines() for n, state in CHIP.findall(line) if int(n) == card]
    if len(health) != 2 or any(x != "OK" for x in health):
        raise LeaseLost(f"target card {card} health entries={health!r}, need two OK dies")
    by_card: dict[int, set[int]] = {}
    for line in tail.splitlines():
        if re.match(r"^\|\s*\d+\s+", line) and not PROC.match(line):
            raise LeaseLost("unparseable npu-smi process row; STOP")
        for n, die, pid in PROC.findall(line):
            n, die, pid = int(n), int(die), int(pid)
            if die not in (0, 1) or pid <= 0:
                raise LeaseLost("malformed process row")
            by_card.setdefault(n, set()).add(pid)
    return {
        "card": card,
        "target": by_card.get(card, set()),
        "outside": set().union(*(v for k, v in by_card.items() if k != card)),
        "all": by_card,
    }


def assess(
    snap: dict,
    *,
    stage_running: bool,
    max_rank_pids: int = 2,
    deploy_allowance: int = 0,
    allow_external_target: bool = False,
) -> dict:
    """allow_external_target: operator-authorized shared-card lease; skips target occupancy
    exclusivity checks (idle preflight, cross-card-on-target, own-PID count cap) while health,
    parse and signal fail-closed checks stay active. Occupancy is still recorded, never waived."""
    target = snap["target"]
    outside = snap["outside"]
    if allow_external_target:
        return {
            "target_pids": sorted(target),
            "cross_card": sorted(target & outside),
            "target_pid_origin": "shared_card_mode_external_pids_tolerated",
        }
    if not stage_running and target:
        raise LeaseLost(f"target must be idle before stage, found PIDs={sorted(target)}")
    cross = target & outside
    if cross:
        raise LeaseLost(f"external cross-card PID on target: {sorted(cross)}")
    if len(target) > max_rank_pids + deploy_allowance:
        raise ExtraTargetPids(
            f"unexpected target process count {len(target)}>{max_rank_pids + deploy_allowance}: {sorted(target)}"
        )
    return {
        "target_pids": sorted(target),
        "cross_card": sorted(cross),
        "target_pid_origin": "unverified_host_namespace_requires_exclusive_lease",
    }


def sample(card: int) -> tuple[str, dict]:
    try:
        run = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=8, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LeaseLost(f"npu-smi unavailable/failed: {exc}") from exc
    return run.stdout, parse_info(run.stdout, card)


def record(path: Path, event: str, **fields: object) -> None:
    entry = {"at_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stop_owned(child: subprocess.Popen) -> None:
    """只信任本脚本新建的独立PGID；不根据npu-smi PID杀进程。"""
    if child.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def deploy_window_open(flag_dir: Path | None) -> bool:
    """worker 在池创建前后声明/撤销的部署窗口标记（每 Rank 一个文件）。"""
    if flag_dir is None:
        return False
    return any(flag_dir.glob("deploy_window_rank*.flag"))


def run_stage(
    card: int,
    log: Path,
    poll: float,
    command: list[str],
    timeout: float = 600,
    deploy_flag_dir: Path | None = None,
    deploy_extra_pids: int = 2,
    window_extra_pids: int = 2,
    deploy_persist_s: float = 60.0,
    allow_external_target: bool = False,
) -> int:
    if not log.is_absolute() or not log.parent.is_dir() or log.is_symlink():
        raise ValueError("absolute log under an existing directory required")
    if not command:
        raise ValueError("stage command required")
    if deploy_extra_pids < 0 or window_extra_pids < 0 or not (1.0 <= deploy_persist_s <= 600.0):
        raise ValueError("deploy window bounds invalid")
    record(
        log,
        "PREFLIGHT_START",
        argv=command,
        card=card,
        lease_mode="SHARED_CARD_OPERATOR_AUTHORIZED" if allow_external_target else "EXCLUSIVE",
    )
    last_raw = ""
    try:
        last_raw, before = sample(card)
        checked = assess(before, stage_running=False, allow_external_target=allow_external_target)
    except LeaseLost as exc:
        record(log, "STOP", phase="PREFLIGHT", reason=str(exc))
        if last_raw:
            (log.parent / "lease_guard_stop.npu.txt").write_text(last_raw, encoding="utf-8")
        raise
    record(log, "PREFLIGHT_PASS", **checked)
    child = subprocess.Popen(command, start_new_session=True)
    record(log, "STAGE_STARTED", child_pid=child.pid)
    deadline = time.monotonic() + timeout
    over_since: float | None = None
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def window_allowance() -> int:
        # Declared window tolerance: 2 helpers + bounded torch_npu compile-infra forms.
        return deploy_extra_pids + window_extra_pids if deploy_window_open(deploy_flag_dir) else 0

    def interrupted(signum: int, _frame: object) -> None:
        raise LeaseLost(f"guard received signal {signum}; stop owned stage")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            if time.monotonic() >= deadline:
                raise LeaseLost("V3 stage wall-clock deadline exceeded")
            last_raw, snapshot = sample(card)
            declared = deploy_window_open(deploy_flag_dir)
            verified = assess(
                snapshot,
                stage_running=True,
                deploy_allowance=window_allowance(),
                allow_external_target=allow_external_target,
            )
            if not allow_external_target and len(snapshot["target"]) > 2:
                # 分层：无窗口或超额>allowance 已在 assess 内 STOP；此处只剩
                # "声明窗口内 3~6 个进程"——自有助手/编译基础设施子进程形态，仅有界容忍。
                if not declared:
                    raise LeaseLost(f"unexpected target process count without deploy window: {verified}")
                if over_since is None:
                    over_since = time.monotonic()
                    record(log, "DEPLOY_WINDOW_TRANSIENT", **verified)
                elif time.monotonic() - over_since > deploy_persist_s:
                    raise LeaseLost(f"deploy window extras persisted >{deploy_persist_s}s: {verified['target_pids']}")
            elif over_since is not None:
                over_since = None
                record(log, "DEPLOY_WINDOW_HEALED", **verified)
            record(log, "LEASE_SAMPLE", **verified)
            ret = child.poll()
            if ret is not None:
                # 短Screen可能不足2秒：结束时强制再采一帧，不能只靠定时采样。
                last_raw, final = sample(card)
                record(
                    log,
                    "POST_STAGE_LEASE",
                    **assess(
                        final,
                        stage_running=True,
                        deploy_allowance=window_allowance(),
                        allow_external_target=allow_external_target,
                    ),
                )
                record(log, "STAGE_EXIT", exit_code=ret)
                return ret
            time.sleep(poll)
    except BaseException as exc:
        record(log, "STOP", reason=str(exc))
        if last_raw:
            (log.parent / "lease_guard_stop.npu.txt").write_text(last_raw, encoding="utf-8")
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
