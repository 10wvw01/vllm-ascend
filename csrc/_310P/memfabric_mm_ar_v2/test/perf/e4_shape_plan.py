#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E4-B: 2048<=真实M<=4096 的纯CPU分组规划与保守路由合同。

本模块是离线研究规划器，不执行设备Kernel、不启用融合Serving。
2048/1024/512/256分别是“逻辑组行数”，不是AddTile；
尾块的独立MM/SDMA/接收缓存协议尚无E4设备实现，禁止据此宣布支持真实尾块。
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

MIN_M, MAX_M, ROW_BYTES = 2048, 4096, 4096
LEGAL_GROUPS = (2048, 1024, 512, 256)
MAX_SDMA_PACKET_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Group:
    start: int
    rows: int
    valid_rows: int
    packet_rows: tuple[int, ...]


@dataclass(frozen=True)
class Plan:
    requested_m: int
    padded_m: int
    groups: tuple[Group, ...]

    @property
    def padding_rows(self) -> int:
        return self.padded_m - self.requested_m


def _packets(group_rows: int) -> tuple[int, ...]:
    # E3已经验证2048拆成2x1024，其他尺寸仅是待验证的逻辑规划能力。
    return (1024, 1024) if group_rows == 2048 else (group_rows,)


def _compositions(rows: int) -> tuple[int, ...] | None:
    @lru_cache(None)
    def choose(left: int) -> tuple[int, ...] | None:
        if left == 0:
            return ()
        candidates = []
        for q in LEGAL_GROUPS:
            if q > left:
                continue
            tail = choose(left - q)
            if tail is not None:
                candidate = (q,) + tail
                # 先少组，再优先保留大组作为前缀；这里只做逻辑分组规划。
                candidates.append(candidate)
        if not candidates:
            return None
        return min(candidates, key=lambda cs: (len(cs), tuple(-c for c in cs)))
    return choose(rows)


def plan(m: int) -> Plan:
    if isinstance(m, bool) or not isinstance(m, int) or not MIN_M <= m <= MAX_M:
        raise ValueError("E4 only plans integer M=2048..4096")
    # 理论最少尾部补齐（到256行），不意味着相应实机Kernel已支持。
    padded = ((m + 255) // 256) * 256
    if padded > MAX_M:
        raise ValueError("padding exceeds available 4096-row window")
    chunks = _compositions(padded)
    if not chunks:
        raise ValueError("no legal coverage")
    out = []
    off = 0
    for q in chunks:
        packets = _packets(q)
        assert sum(packets) == q
        assert all(p * ROW_BYTES <= MAX_SDMA_PACKET_BYTES for p in packets)
        valid = max(0, min(q, m - off))
        out.append(Group(off, q, valid, packets))
        off += q
    assert off == padded
    assert sum(g.valid_rows for g in out) == m
    return Plan(m, padded, tuple(out))


def choose_default_route(m: int, verified: Mapping[int, Mapping] | None = None) -> str:
    """未获得独立同M/TP/NZ/硬件/Serving验收时一律退回Stock。

    verified只接受上游审签后的同一精确M的不可变决策；E3端点的L1
    不能推出2049..4095任何点。返回值严格只有stock/e4。
    """
    plan(m)
    evidence = (verified or {}).get(m)
    if not isinstance(evidence, dict):
        return "stock"
    if (evidence.get("m") != m or evidence.get("tp") != 2 or
        evidence.get("kn") != (2048, 2048) or
        evidence.get("stock_nz") is not True or
        evidence.get("rank_bit_exact") is not True or
        evidence.get("l1_stable") is not True or
        evidence.get("l3_confirmed") is not True or
        evidence.get("no_graph_mismatch") is not True or
        evidence.get("gain_vs_stock_pct", -1.0) <= 0):
        return "stock"
    return "e4"


def audit_all_shapes() -> dict[str, int]:
    plans = tuple(plan(m) for m in range(MIN_M, MAX_M + 1))
    assert len(plans) == 2049
    return {
        "shapes": len(plans),
        "zero_padding": sum(p.padding_rows == 0 for p in plans),
        "max_padding_rows": max(p.padding_rows for p in plans),
        "max_groups": max(len(p.groups) for p in plans),
        "unknown_m_routes_to_stock": sum(choose_default_route(p.requested_m) == "stock"
                                         for p in plans),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(audit_all_shapes(), ensure_ascii=False, indent=2))
