#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""离线分析 G1-B 的 MF_DIAG 行（只读）；不推断硬件 SQ 占用或根因。"""
import argparse
import collections
import json
import pathlib
import re

TICK = re.compile(
    r"\[MF_DIAG\] tick epoch: (?P<epoch>\d+) devId: (?P<dev>\d+) "
    r"hb: (?P<hb>\d+) reqHead: (?P<head>\d+) reqTail: (?P<tail>\d+) "
    r"reqPending: (?P<pending>\d+) cursorsValid: (?P<valid>true|false|0|1) "
    r"exitReason: (?P<reason>\d+) mainRet: (?P<ret>\d+)", re.I
)
EXIT = re.compile(
    r"\[MF_DIAG\] exit epoch: (?P<epoch>\d+) devId: (?P<dev>\d+) "
    r"durationMs: (?P<duration>\d+) exitReason: (?P<reason>\d+) "
    r"mainRet: (?P<ret>\d+) consumed: (?P<consumed>\d+) "
    r"sqHeadAfter: (?P<sq_head>\d+) lastReqHead: (?P<head>\d+) "
    r"lastReqTail: (?P<tail>\d+)", re.I
)
START = re.compile(
    r"\[MF_DIAG\] start epoch: (?P<epoch>\d+) devId: (?P<dev>\d+)", re.I
)


def parse_log(text):
    devices = collections.defaultdict(
        lambda: {"ticks": [], "exits": [], "starts": [], "invalid_samples": 0}
    )
    for line_no, line in enumerate(text.splitlines(), 1):
        for kind, pattern in (("tick", TICK), ("exit", EXIT), ("start", START)):
            match = pattern.search(line)
            if not match:
                continue
            values = match.groupdict()
            valid = values.pop("valid", "1").lower() not in ("0", "false")
            data = {key: int(value) for key, value in values.items()}
            dev = data.pop("dev")
            data["line"] = line_no
            if kind == "tick":
                data["valid"] = valid
                devices[dev]["ticks"].append(data)
                if not valid:
                    devices[dev]["invalid_samples"] += 1
            else:
                devices[dev][kind + "s"].append(data)
            break
    summary = {}
    for dev, data in sorted(devices.items()):
        ticks = data["ticks"]
        alerts = []
        prev = None
        for tick in ticks:
            if not tick["valid"]:
                alerts.append({"type": "invalid_cursor_snapshot", "line": tick["line"]})
                prev = None
                continue
            if prev is not None and (
                tick["head"] < prev["head"] or tick["tail"] < prev["tail"]
            ):
                alerts.append({
                    "type": "cursor_decrease_between_samples",
                    "line": tick["line"], "prev_line": prev["line"]
                })
            prev = tick

        # 连续多次观测到 head 固定、请求积压；只是可疑停滞，不等于 SQ 溢出。
        streak = []
        for tick in ticks:
            if not tick["valid"] or tick["pending"] <= 0:
                if len(streak) >= 5:
                    alerts.append({
                        "type": "request_ring_backlog_no_head_progress",
                        "from_line": streak[0]["line"], "to_line": streak[-1]["line"],
                        "samples": len(streak), "head": streak[0]["head"],
                    })
                streak = []
            elif streak and tick["head"] != streak[-1]["head"]:
                if len(streak) >= 5:
                    alerts.append({
                        "type": "request_ring_backlog_no_head_progress",
                        "from_line": streak[0]["line"], "to_line": streak[-1]["line"],
                        "samples": len(streak), "head": streak[0]["head"],
                    })
                streak = [tick]
            else:
                streak.append(tick)
        if len(streak) >= 5:
            alerts.append({
                "type": "request_ring_backlog_no_head_progress",
                "from_line": streak[0]["line"], "to_line": streak[-1]["line"],
                "samples": len(streak), "head": streak[0]["head"],
            })
        exit_counts = collections.Counter(x["reason"] for x in data["exits"])
        for record in data["exits"]:
            if record["reason"] == 2 or record["ret"] != 0:
                alerts.append({
                    "type": "epoch_error_exit",
                    "line": record["line"], "main_ret": record["ret"],
                    "reason": record["reason"],
                })
            elif record["duration"] > 22500:
                alerts.append({
                    "type": "epoch_duration_over_22_5s",
                    "line": record["line"], "duration_ms": record["duration"]
                })
        summary[str(dev)] = {
            "tick_count": len(ticks), "valid_ticks": sum(t["valid"] for t in ticks),
            "starts": len(data["starts"]), "exits": len(data["exits"]),
            "exit_reason_counts": dict(sorted(exit_counts.items())),
            "first_req_head_tail": [ticks[0]["head"], ticks[0]["tail"]] if ticks else None,
            "last_req_head_tail": [ticks[-1]["head"], ticks[-1]["tail"]] if ticks else None,
            "alerts": alerts,
        }
    return {
        "devices": summary,
        "limitations": [
            "ReqHead/ReqTail 属于请求环，不是硬件 SQ 的 head/tail",
            "16B D2H 是非原子采样，单次游标异常只能作为线索",
            "start epoch 表示 Host 已下发，不能直接证明设备已开始执行",
            "日志静默与 1Hz 采样缺口不等于 AICPU 停止运行",
            "检测到积压不意味着 MemFabric 必为第一个故障点",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve_log", type=pathlib.Path)
    args = parser.parse_args()
    if not args.serve_log.is_file():
        parser.error("serve.log 不存在")
    result = parse_log(args.serve_log.read_text(encoding="utf-8", errors="replace"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
