#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU模型：E4 packet-ready Add的信件唯一性、Core释放因果和组间事件复用。

仅为协议model-check，无法代替Ascend IB缓存可见性或真实设备压测。
"""
from dataclasses import dataclass
from typing import Iterable

ROW_BYTES = 4096
PACKET_ROWS = 1024
PACKET_BYTES = ROW_BYTES * PACKET_ROWS
TAG_BASE = 0x4D46325600000000
CORES = tuple(range(8))


@dataclass(frozen=True)
class Mail:
    dst: int
    length: int
    imm: int
    ok: bool = True


def packet_mail(group: int, part: int, recv_base: int) -> Mail:
    if not 0 <= group <= 3 or part not in (0, 1):
        raise ValueError("group 0..3, part0/1")
    tag = group*2+part
    return Mail(recv_base+tag*PACKET_BYTES, PACKET_BYTES, TAG_BASE | tag)


def simulate_group(group: int, recv_base: int, arrival: Iterable[Mail]) -> dict:
    """Core0唯一消费者；每次mail校验前不可释放相应Core半组。

    Core0自己的Add只能在两封邮件全部收到后执行。
    """
    seen=set()
    releases=[]
    for mail in arrival:
        valid_part=None
        for part in (0, 1):
            if part not in seen and mail == packet_mail(group,part,recv_base):
                valid_part=part
                break
        if valid_part is None:
            raise ValueError("duplicate, stale, out-of-range, corrupted or bad-status mail")
        seen.add(valid_part)
        releases.append((valid_part, tuple((1,2,3) if valid_part==0 else (4,5,6,7))))
    if seen!={0,1}:
        raise ValueError("not both independent 4MiB packets received")
    # Core0 is the unique mail reader and consumes both before doing its own Add.
    fired=[c for _,cores in releases for c in cores]
    if sorted(fired)!=list(range(1,8)) or len(set(fired))!=7:
        raise ValueError("IBRelease event reuse or wrong owner mapping")
    return {"group":group,"releases":tuple(releases),"core0_after_two_mails":True,
            "wait_add_done_core_set":CORES,"can_quiet_and_done":True}


def simulate_campaign(group_orders: Iterable[tuple[int,int]], recv_base: int) -> dict:
    out=[]
    count={c:0 for c in range(1,8)}
    for g,order in enumerate(group_orders):
        if tuple(sorted(order))!=(0,1):
            raise ValueError("must deliver two unique parts per group")
        result=simulate_group(g,recv_base,
                              (packet_mail(g,part,recv_base) for part in order))
        for _,cores in result["releases"]:
            for core in cores:count[core]+=1
        # In actual device this WaitAddDone is the barrier before quiet/DONE;
        # a CPU simulator only validates event cardinality, not physical visibility.
        if any(count[c]!=g+1 for c in count):
            raise AssertionError("cross-group IB release count mismatch")
        out.append(result)
    return {"groups":len(out),"release_events":count,"group_trace":out}
