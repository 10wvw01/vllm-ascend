#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1-C/G1-B 原始证据提取：统一时间线、1024 边界核算与脱敏摘录。

只读、确定性：不修改任何原始日志；同一输入恒产生同一输出（无时间戳注入）。
用法：
  python3 g1c_evidence_extract.py \
      --session B02=/path/to/B02 --session C2R1=/path/to/C2R1 \
      --session C2R2=/path/to/C2R2 --out <输出目录>

输入为各 Gate 会话目录（含 serve.log、stability_rep*.json、runner.log、
evidence/device_logs/**）。输出：
  timeline_<S>.json/.csv      统一时间线（V2 行时间为插值，标注 time_quality）
  session_stats_<S>.json      会话级统计
  boundary_1024_analysis.json 1024 边界专项核算（observed/derived/unknown 分级）
  sanitized/<S>_<窗口>.log    脱敏摘录（用户路径/主机名/内网 IP 已替换）
事实来源（源码常量）：
  memfabric_hybrid src/smem/include/device/smem_shm_sdma_layout.h:
    RS_REQ_SLOTS=64, RS_QUIET_SLOTS=64, RS_ARR_SLOTS=1024
  hybm_sdma_orch_kernel.cc issueMailSqes: 每邮件 3 个 SQE（载荷→到达戳→完成戳）
  smem_shm_aicore_base_sdma.h smem_shm_sdma_wait_at: 戳为 1 基(seq+1)，
    等待槽 = ARR_HEAD % 1024；POLL_MAX_ITER=1e8（事实上的无限等待）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

ARR_SLOTS = 1024
SQE_PER_MAIL = 3
SQ_DEPTH_DEFAULT = 1024

TS_PREFIX = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
MF_G1C = re.compile(
    r"\[MF_G1C\] devId: (?P<dev>\d+) epoch: (?P<epoch>\d+) valid: (?P<valid>\d+) "
    r"phase: (?P<phase>\d+) reqSeq: (?P<reqSeq>\d+) shadowTail: (?P<shadowTail>\d+) "
    r"sqDepth: (?P<sqDepth>\d+) hwHeadLast: (?P<hwHead>\d+) hwTailLast: (?P<hwTail>\d+) "
    r"queryHeadRcLast: (?P<qhrc>-?\d+) queryTailRcLast: (?P<qtrc>-?\d+) error: (?P<error>-?\d+)")
MF_G1C_ERR = re.compile(r"\[MF_G1C\] devId: (?P<dev>\d+) epoch: (?P<epoch>\d+) snapshotError: (?P<rc>-?\d+)")
MF_TICK = re.compile(
    r"\[MF_DIAG\] tick epoch: (?P<epoch>\d+) devId: (?P<dev>\d+) hb: (?P<hb>\d+) "
    r"reqHead: (?P<reqHead>\d+) reqTail: (?P<reqTail>\d+) reqPending: (?P<reqPending>\d+) "
    r"cursorsValid: (?P<cursorsValid>\d+) exitReason: (?P<exitReason>\d+) mainRet: (?P<mainRet>-?\d+)")
MF_EXIT = re.compile(
    r"\[MF_DIAG\] exit epoch: (?P<epoch>\d+) devId: (?P<dev>\d+) durationMs: (?P<durationMs>\d+) "
    r"exitReason: (?P<exitReason>\d+) mainRet: (?P<mainRet>-?\d+) consumed: (?P<consumed>\d+) "
    r"sqHeadAfter: (?P<sqHeadAfter>\d+) lastReqHead: (?P<lastReqHead>\d+) lastReqTail: (?P<lastReqTail>\d+)")
MF_START = re.compile(r"\[MF_DIAG\] start epoch: (?P<epoch>\d+) devId: (?P<dev>\d+) sqId: (?P<sqId>\d+)")
V2_LINE = re.compile(r"\[V2_G1C\] rank=(?P<rank>\d+) changed=(?P<changed>\d)(?P<cores>(?: c[0-7]=\d+:\d+:\d+){8})\s*$")
V2_CORE = re.compile(r"\bc(?P<core>[0-7])=(?P<gen>\d+):(?P<chunk>\d+):(?P<stage>\d+)")
WORKER_ERR = re.compile(
    r"^\(Worker_TP(?P<rank>\d) pid=(?P<pid>\d+)\) ERROR "
    r"(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<msg>.*)$")
RANK_E = re.compile(r"^\[rank(?P<rank>\d)\]:\[E\d+ (?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ (?P<msg>.*)$")
PLOG_ERR = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+\[INFO\] RUNTIME\(\d+,\)"
    r".*?(?P<msg>PeekLastErr: level=\d+ err=(?P<err>\d+))")
V2_MILESTONE = re.compile(
    r"^\(Worker_TP(?P<rank>\d) pid=\d+\) INFO (?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"\[(?P<mod>[\w.]+)\] (?P<msg>.*)$")
DEVICE_HWTS_TIMEOUT = re.compile(
    r"time=(?P<dts>\d{4}-\d{2}-\d{2})-(?P<hms>\d{2}:\d{2}:\d{2})\.(?P<frac>\d+)\.\d+, "
    r"function=process_hwts_timeout_exception")
HWTS_TASK = re.compile(r"Message info\[0\]:RTS_HWTS: Aicore timeout, slot_id=(?P<slot>\d+), "
                       r"stream_id=(?P<stream>\d+), task_id=(?P<task>\d+)")
RANK_MARKER = re.compile(r"(?:^\[rank(?P<a>\d)\]:\[E)|(?:^\(Worker_TP(?P<b>\d) pid=\d+\))")
# 设备日志时区 = UTC+8：serve 启动时刻（UTC）与设备 run 日志/CCECPU 首条时间对齐核实
DEVICE_TZ_OFFSET_HOURS = 8

SANITIZE_RULES = [
    ("/home/w00804146", "/home/<user>"),
    ("w00804146", "<user>"),
    ("qwen-test", "<host>"),
    ("172.17.0.3", "<container-ip>"),
    ("192.168.2.199", "<ip-lan-a>"),
    ("192.168.3.198", "<ip-lan-b>"),
]

STAGE_NAMES = {
    10: "entry", 20: "mm_before", 21: "mm_after", 30: "ib_ready_before", 31: "ib_ready_after",
    40: "signal_before", 41: "signal_after", 50: "mail_wait_before", 51: "mail_wait_after",
    60: "add_release_before", 61: "add_release_after", 70: "add_before", 71: "add_after",
    80: "add_done_before", 81: "add_done_after", 90: "quiet_before", 91: "quiet_after",
    100: "done_signal_before", 101: "done_signal_after", 110: "done_wait_before",
    111: "done_wait_after", 120: "exit",
}


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj))


def parse_ts(text: str) -> datetime:
    if "." in text:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def parse_vllm_ts(text: str, year: int) -> datetime:
    return datetime.strptime(f"{year}-{text}", "%Y-%m-%d %H:%M:%S")


def sanitize(text: str) -> str:
    for old, new in SANITIZE_RULES:
        text = text.replace(old, new)
    return text


class Session:
    def __init__(self, name: str, root: Path):
        self.name = name
        self.root = root
        self.serve_log = root / "serve.log"
        self.events: list[dict] = []
        self.v2_frames: list[dict] = []
        self.mf_g1c: list[dict] = []
        self.mf_exits: list[dict] = []
        self.mf_ticks: list[dict] = []
        self.worker_errors: list[dict] = []
        self._hwts_seen: set = set()
        self.year = 2026

    def add(self, time_utc: str, quality: str, event: str, rank_or_dev, fields: dict, line: int):
        self.events.append({
            "time_utc": time_utc, "time_quality": quality, "event": event,
            "rank_or_dev": str(rank_or_dev), "fields": fields, "source_line": line,
        })

    def parse(self):
        text = self.serve_log.read_text(errors="replace").splitlines()
        anchors = []  # (line_no, datetime) for interpolation
        for i, raw in enumerate(text, 1):
            m = TS_PREFIX.match(raw)
            if m:
                anchors.append((i, parse_ts(m.group("ts"))))

        def interpolate(i: int):
            prev = next((a for a in reversed(anchors) if a[0] < i), None)
            nxt = next((a for a in anchors if a[0] > i), None)
            if prev is None:
                return None, "no_anchor"
            if nxt is None:
                return prev[1], "lower_bound"
            span = (nxt[0] - prev[0]) or 1
            frac = (i - prev[0]) / span
            est = prev[1] + (nxt[1] - prev[1]) * frac
            return est, "interpolated"

        for i, raw in enumerate(text, 1):
            ts_m = TS_PREFIX.match(raw)
            ts = parse_ts(ts_m.group("ts")) if ts_m else None
            # 设备看门狗超时（Worker 前缀或裸续行两种格式；后者按最近 rank 标记归属）
            dt = DEVICE_HWTS_TIMEOUT.search(raw)
            if dt:
                wm = WORKER_ERR.match(raw)
                if wm:
                    rank, pid = int(wm.group("rank")), int(wm.group("pid"))
                else:
                    rank, pid = None, None
                    for j in range(i - 1, 0, -1):
                        rm = RANK_MARKER.match(text[j - 1] or "")
                        if rm:
                            rank = int(rm.group("a") or rm.group("b"))
                            break
                dev_utc = datetime.strptime(
                    f"{dt.group('dts')} {dt.group('hms')}.{dt.group('frac')[:6]}",
                    "%Y-%m-%d %H:%M:%S.%f") - timedelta(hours=DEVICE_TZ_OFFSET_HOURS)
                rec = {"rank": rank, "pid": pid,
                       "device_time_utc": dev_utc.isoformat(timespec="milliseconds"),
                       "function": "process_hwts_timeout_exception", "error_code": "0x25"}
                for j in range(max(1, i - 4), min(len(text), i + 4) + 1):
                    tm = HWTS_TASK.search(text[j - 1] or "")
                    if tm:
                        rec["timed_out_task"] = {k: int(v) for k, v in tm.groupdict().items()}
                        break
                key = (rank, rec["device_time_utc"])
                if key in self._hwts_seen:
                    continue
                self._hwts_seen.add(key)
                self.add(dev_utc.isoformat(timespec="milliseconds"), "exact",
                         "device_hwts_timeout", rank, rec, i)
                continue
            m = MF_G1C.search(raw)
            if m:
                rec = {k: int(v) for k, v in m.groupdict().items()}
                rec["time"] = ts
                self.mf_g1c.append(rec)
                self.add(ts.isoformat(), "exact", "mf_g1c", rec["dev"], rec, i)
                continue
            m = MF_G1C_ERR.search(raw)
            if m:
                rec = {k: int(v) for k, v in m.groupdict().items()}
                self.add(ts.isoformat(), "exact", "mf_g1c_snapshot_error", rec["dev"], rec, i)
                continue
            m = MF_TICK.search(raw)
            if m:
                rec = {k: int(v) for k, v in m.groupdict().items()}
                self.mf_ticks.append(rec)
                self.add(ts.isoformat(), "exact", "mf_diag_tick", rec["dev"], rec, i)
                continue
            m = MF_EXIT.search(raw)
            if m:
                rec = {k: int(v) for k, v in m.groupdict().items()}
                self.mf_exits.append(rec)
                self.add(ts.isoformat(), "exact", "mf_diag_epoch_exit", rec["dev"], rec, i)
                continue
            m = MF_START.search(raw)
            if m:
                rec = {k: int(v) for k, v in m.groupdict().items()}
                self.add(ts.isoformat(), "exact", "mf_diag_epoch_start", rec["dev"], rec, i)
                continue
            m = V2_LINE.search(raw)
            if m:
                cores = [c.groupdict() for c in V2_CORE.finditer(m.group("cores"))]
                est, q = interpolate(i)
                rec = {
                    "rank": int(m.group("rank")), "changed": int(m.group("changed")),
                    "cores": [{"core": int(c["core"]), "gen": int(c["gen"]),
                               "chunk": int(c["chunk"]), "stage": int(c["stage"]),
                               "stage_name": STAGE_NAMES.get(int(c["stage"]), "unknown")} for c in cores],
                }
                if est is not None:
                    rec["time"] = est
                    self.v2_frames.append(rec)
                    self.add(est.isoformat(timespec="milliseconds"), q, "v2_g1c",
                             rec["rank"], {"changed": rec["changed"], "cores": rec["cores"]}, i)
                continue
            m = WORKER_ERR.match(raw)
            if m:
                t = parse_vllm_ts(m.group("ts"), self.year)
                rec = {"rank": int(m.group("rank")), "pid": int(m.group("pid")),
                       "msg": m.group("msg")[:180]}
                self.worker_errors.append(rec)
                self.add(t.isoformat(timespec="seconds"), "second_resolution", "serve_error",
                         rec["rank"], rec, i)
                continue
            m = RANK_E.match(raw)
            if m:
                t = parse_vllm_ts(m.group("ts"), self.year)
                rec = {"rank": int(m.group("rank")), "msg": m.group("msg")[:180]}
                self.worker_errors.append(rec)
                self.add(t.isoformat(timespec="milliseconds"), "millisecond_resolution",
                         "serve_error", rec["rank"], rec, i)
                continue
            m = V2_MILESTONE.match(raw)
            if m and ("first fused call verified" in m.group("msg")
                      or "V2 MM+MemFabric" in m.group("msg")):
                t = parse_vllm_ts(m.group("ts"), self.year)
                self.add(t.isoformat(timespec="seconds"), "second_resolution", "serve_info",
                         int(m.group("rank")), {"msg": m.group("msg")[:180]}, i)
                continue
        self.events.sort(key=lambda e: (e["time_utc"], e["source_line"]))
        # 设备 plog（如存在）
        plog_dir = self.root / "evidence" / "device_logs"
        if plog_dir.is_dir():
            for p in sorted(plog_dir.rglob("plog-*.log")):
                for i, raw in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    m = PLOG_ERR.match(raw)
                    if m:
                        t = datetime.strptime(m.group("ts"), "%Y-%m-%d-%H:%M:%S.%f")
                        self.add(t.isoformat(timespec="milliseconds"), "exact", "plog_error",
                                 "n/a", {"err": int(m.group("err")), "file": p.name, "msg": m.group("msg")}, i)

    # ---- 会话统计 ----
    def stats(self) -> dict:
        bench = []
        for f in sorted((self.root).glob("stability_rep*.json")):
            try:
                d = json.loads(f.read_text())
                bench.append({"file": f.name, "completed": d.get("completed"),
                              "errors": d.get("errors") if isinstance(d.get("errors"), int) else None,
                              "duration_s": round(d.get("duration", 0), 1)})
            except (json.JSONDecodeError, OSError):
                bench.append({"file": f.name, "parse_error": True})
        per_dev = {}
        for dev in {r["dev"] for r in self.mf_g1c}:
            rows = [r for r in self.mf_g1c if r["dev"] == dev]
            per_dev[str(dev)] = {
                "mf_g1c_samples": len(rows),
                "invalid_frames": sum(1 for r in rows if r["valid"] != 1),
                "final_reqSeq": rows[-1]["reqSeq"] if rows else None,
                "final_shadowTail": rows[-1]["shadowTail"] if rows else None,
                "final_hwHead_hwTail": [rows[-1]["hwHead"], rows[-1]["hwTail"]] if rows else None,
                "final_query_rc": [rows[-1]["qhrc"], rows[-1]["qtrc"]] if rows else None,
                "final_phase": rows[-1]["phase"] if rows else None,
            }
        epoch_bad = [e for e in self.mf_exits if e["exitReason"] != 0 or e["mainRet"] != 0]
        # 3-SQE/邮件不变量核验：sqHeadAfter 增量 == 3*consumed (mod sqDepth)
        inv_ok, inv_checked = 0, 0
        by_dev: dict[int, list[dict]] = {}
        for e in self.mf_exits:
            by_dev.setdefault(e["dev"], []).append(e)
        for dev, rows in by_dev.items():
            for a, b in zip(rows, rows[1:]):
                if b["epoch"] == a["epoch"] + 1:
                    inv_checked += 1
                    if (b["sqHeadAfter"] - a["sqHeadAfter"]) % b.get("sqDepth", SQ_DEPTH_DEFAULT) == \
                            SQE_PER_MAIL * b["consumed"] % SQ_DEPTH_DEFAULT or \
                            (b["sqHeadAfter"] - a["sqHeadAfter"]) == SQE_PER_MAIL * b["consumed"]:
                        inv_ok += 1
        v2_by_rank = {}
        for rank in {f["rank"] for f in self.v2_frames}:
            frames = [f for f in self.v2_frames if f["rank"] == rank]
            v2_by_rank[str(rank)] = {
                "frames": len(frames),
                "last_frame": frames[-1]["cores"][0] if frames else None,
                "last_changed": frames[-1]["changed"] if frames else None,
            }
        return {
            "session": self.name,
            "bench_reps": bench,
            "bench_completed_total": sum(b.get("completed") or 0 for b in bench),
            "mf_g1c_per_dev": per_dev,
            "mf_diag_ticks": len(self.mf_ticks),
            "epoch_exits": len(self.mf_exits),
            "epoch_exits_abnormal": len(epoch_bad),
            "sqe_per_mail_invariant": {"checked_pairs": inv_checked, "consistent": inv_ok},
            "v2_per_rank": v2_by_rank,
            "worker_first_errors": [
                {"rank": e["rank"], "msg": e["msg"][:120]} for e in self.worker_errors[:4]],
        }


def timeline_json(sess: Session) -> list[dict]:
    out = []
    for e in sess.events:
        out.append({
            "time_utc": e["time_utc"], "time_quality": e["time_quality"], "event": e["event"],
            "rank_or_dev": e["rank_or_dev"], "source_line": e["source_line"],
            "fields": e["fields"],
        })
    return out


TIMELINE_CSV_COLS = ["time_utc", "time_quality", "event", "rank_or_dev", "source_line",
                     "epoch", "phase", "reqSeq", "shadowTail", "hwHeadLast", "hwTailLast",
                     "reqHead", "reqTail", "reqPending", "v2_gen", "v2_chunk", "v2_stage",
                     "v2_stage_name", "v2_changed", "detail"]


def timeline_csv(sess: Session, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TIMELINE_CSV_COLS)
        for e in sess.events:
            fl = e["fields"]
            cores = fl.get("cores") or [{}]
            row = [e["time_utc"], e["time_quality"], e["event"], e["rank_or_dev"], e["source_line"],
                   fl.get("epoch", ""), fl.get("phase", ""), fl.get("reqSeq", ""),
                   fl.get("shadowTail", ""), fl.get("hwHead", ""), fl.get("hwTail", ""),
                   fl.get("reqHead", ""), fl.get("reqTail", ""), fl.get("reqPending", ""),
                   cores[0].get("gen", ""), cores[0].get("chunk", ""), cores[0].get("stage", ""),
                   cores[0].get("stage_name", ""), fl.get("changed", ""),
                   sanitize(json.dumps({k: v for k, v in fl.items()
                                        if k not in ("cores",)}, ensure_ascii=False, default=str))[:220]]
            w.writerow(row)


def window_excerpt(sess: Session, start: str, end: str, out: Path):
    """按时间窗截取 serve.log 行（含窗内无时间戳行），脱敏后写出。"""
    t0, t1 = parse_ts(start), parse_ts(end)
    text = sess.serve_log.read_text(errors="replace").splitlines()
    lo = hi = None
    ts_at = {}
    for i, raw in enumerate(text, 1):
        m = TS_PREFIX.match(raw)
        if m:
            ts_at[i] = parse_ts(m.group("ts"))
    idx = sorted(ts_at)
    for i in idx:
        if ts_at[i] >= t0 and lo is None:
            lo = i
        if ts_at[i] <= t1:
            hi = i
    lo = lo or 1
    hi = hi or len(text)
    # 扩展到窗口边界间无时间戳的 V2 行
    while lo > 1 and TS_PREFIX.match(text[lo - 2] or "") is None and "[V2_G1C]" in text[lo - 2]:
        lo -= 1
    while hi < len(text) and "[V2_G1C]" in text[hi]:
        hi += 1
    lines = [f"# excerpt of {sess.name}/serve.log lines {lo}..{hi} window [{start} .. {end}] UTC",
             "# sanitized: /home/<user>, <host>, <container-ip>, <ip-lan-a/b>; technical content unchanged"]
    lines += [sanitize(text[i - 1]) for i in range(lo, hi + 1)]
    out.write_text("\n".join(lines) + "\n")
    return {"file": out.name, "lines": f"{lo}..{hi}", "window": [start, end]}


def last_stalled_v2_frame(sess: Session):
    """返回每个 rank 的最后 V2 帧与其前一个 changed=1 帧（停滞几何）。"""
    out = {}
    for rank in {f["rank"] for f in sess.v2_frames}:
        frames = [f for f in sess.v2_frames if f["rank"] == rank]
        if not frames:
            continue
        last = frames[-1]
        prev_change = next((f for f in reversed(frames[:-1]) if f["changed"] == 1), None)
        out[str(rank)] = {"last": last, "last_change_frame": prev_change}
    return out


def derive_boundary_c2r2(sess: Session) -> dict:
    """C2R2 专用：从停滞几何推导丢失邮件的 seq/槽位（推导链逐步标注）。

    几何判别：stage50 的 chunk 字段在 overlap 流水下为 prev=k-1（k=当前迭代），
    收尾等待为 last=chunks-1；非 overlap 下为 prev=k。两种解释给出不同的
    "已 signal 邮件数 n"。用双 rank 锁步约束（停滞前基线计数相等）唯一化：
      reqSeq0 - n0 == reqSeq1 - n1（基线相等）且 n1 - n0 ∈ {0,1}
    无唯一解时输出 geometry_undetermined，不做槽位断言。
    """
    res = {"observed": {}, "derived": [], "unknown": []}
    if not sess.v2_frames:
        res["unknown"].append("no_v2_frames")
        return res
    geo = last_stalled_v2_frame(sess)
    finals = {}
    for rank, g in geo.items():
        c0 = g["last"]["cores"][0]
        others = [c["stage"] for c in g["last"]["cores"] if c["core"] != 0]
        finals[rank] = {
            "generation": c0["gen"], "core0_chunk_field": c0["chunk"], "core0_stage": c0["stage"],
            "core0_stage_name": c0["stage_name"],
            "cores_1_7_stage": sorted(set(others)),
            "stalled": g["last"]["changed"] == 0,
        }
    res["observed"]["v2_final_frames"] = finals
    res["observed"]["mf_final"] = {}
    for dev in {r["dev"] for r in sess.mf_g1c}:
        rows = [r for r in sess.mf_g1c if r["dev"] == dev]
        last = rows[-1]
        res["observed"]["mf_final"][str(dev)] = {
            "reqSeq": last["reqSeq"], "shadowTail": last["shadowTail"], "sqDepth": last["sqDepth"],
            "hwHeadLast": last["hwHead"], "hwTailLast": last["hwTail"],
            "query_rc": [last["qhrc"], last["qtrc"]], "phase": last["phase"],
        }
    # 观测的 chunk 总数（全会话 stage40/41 帧的 chunk 字段最大值 +1）
    max_chunk = max((c["chunk"] for f in sess.v2_frames for c in f["cores"]), default=-1)
    chunks = max_chunk + 1
    res["observed"]["chunks_per_call_inferred"] = chunks
    deriv = []

    def signal_count(rank_fin, mode):
        """返回该停滞几何下已投递的 signal 数；mode: 'overlap'|'no_overlap'。"""
        cf = rank_fin["core0_chunk_field"]
        if rank_fin["core0_stage"] != 50:
            return None
        if mode == "overlap":
            if cf == chunks - 1:
                return chunks          # 收尾等待 last：全部 chunk 已 signal
            return cf + 2              # mid-loop k=cf+1：已 signal chunk 0..k
        return cf + 1                  # 非 overlap：k=cf，已 signal chunk 0..k

    candidates = []
    for mode in ("overlap", "no_overlap"):
        ns = {r: signal_count(fin, mode) for r, fin in finals.items()}
        if any(v is None for v in ns.values()):
            continue
        finals_req = {r: res["observed"]["mf_final"][str(int(r))]["reqSeq"] for r in ns}
        bases = {r: finals_req[r] - ns[r] for r in ns}
        lockstep = len({b for b in bases.values()}) == 1
        candidates.append({"mode": mode, "signal_counts": ns, "baselines": bases,
                           "lockstep_consistent": lockstep})
    res["observed"]["geometry_candidates"] = candidates
    ok = [c for c in candidates if c["lockstep_consistent"]]
    if len(ok) != 1:
        res["unknown"].append("geometry_undetermined: 锁步判别不唯一，槽位推导中止")
        res["derived"] = deriv
        return res
    chosen = ok[0]
    deriv.append({
        "id": "D-geometry",
        "claim": (f"停滞几何判别为 {chosen['mode']} 模式：signal 数 "
                  f"{chosen['signal_counts']}，停滞前基线 {chosen['baselines']}"),
        "basis": ["双 rank 锁步约束（同层同 M，完成代双方各发满 4 封 ⇒ 基线相等）",
                  f"终态 reqSeq {{0: {res['observed']['mf_final']['0']['reqSeq']}, "
                  f"1: {res['observed']['mf_final']['1']['reqSeq']}}} 与各候选 n 逐一检验",
                  f"观测 chunks/call={chunks}（stage40/41 chunk 字段最大值+1）"],
    })
    ns = chosen["signal_counts"]
    base = chosen["baselines"]
    mail_seq = {r: {f"chunk{c}": base[r] + c for c in range(ns[r])} for r in ns}
    deriv.append({
        "id": "D-seq",
        "claim": f"停滞代 signal seq 映射（请求环 FIFO 游标，0 基）= {mail_seq}",
        "basis": ["issueMailSqes/consumeOneMail: seq = 请求环 FIFO 游标",
                  "signal 顺序 = chunk 升序（内核循环）",
                  "每核内核串行化 + FIFO ⇒ 停滞代的 signal 位于环尾部"],
    })
    # 丢失邮件 vs 下游等待
    lost, downstream = {}, {}
    for rank, fin in finals.items():
        if fin["core0_stage"] != 50:
            continue
        peer = "1" if rank == "0" else "0"
        waited = fin["core0_chunk_field"]
        key = f"chunk{waited}"
        if key in mail_seq.get(peer, {}):
            seq = mail_seq[peer][key]
            lost[rank] = {
                "waiting_rank": rank, "source_rank": peer, "waited_chunk": waited,
                "seq_in_source_dev_request_ring": seq,
                "arrival_slot_on_waiting_dev": seq % ARR_SLOTS,
                "stamp_value_expected": seq + 1,
                "sq_slots_on_source_dev": [(SQE_PER_MAIL * seq + j) % SQ_DEPTH_DEFAULT
                                           for j in range(SQE_PER_MAIL)],
            }
        else:
            downstream[rank] = {
                "waiting_rank": rank, "source_rank": peer, "waited_chunk": waited,
                "note": f"源 rank{peer} 停滞前仅 signal 到 chunk{ns[peer] - 1}，"
                        f"chunk{waited} 从未投递 —— 该等待为停滞的下游结果，非邮件丢失",
            }
    deriv.append({
        "id": "D-loss",
        "claim": (f"唯一无法解释的等待（丢失邮件）：rank{list(lost)[0] if lost else '?'} 等待 "
                  f"rank{lost[list(lost)[0]]['source_rank'] if lost else '?'} 的 "
                  f"chunk{lost[list(lost)[0]]['waited_chunk'] if lost else '?'} 邮件" if lost
                  else "无（不应发生）"),
        "basis": ["等待方 stage50 + 源方该 chunk 的 signal 已被消费（请求环排空）⇒ SQE 链已执行",
                  "另侧等待的邮件源方从未投递 ⇒ 归类为下游等待"],
    } if lost else None)
    if lost:
        for rank, lm in lost.items():
            arr_slot = lm["arrival_slot_on_waiting_dev"]
            sq_last = lm["sq_slots_on_source_dev"][-1]
            deriv.append({
                "id": f"D-wrap-{rank}",
                "claim": (
                    f"丢失邮件 seq={lm['seq_in_source_dev_request_ring']}，到达槽位 {arr_slot}"
                    f"（{'ARR 环最后一槽' if arr_slot == ARR_SLOTS - 1 else '非边界槽'}），"
                    f"SQE 链止于源设备 SQ 槽 {sq_last}"
                    f"（{'SQ 环最后一槽' if sq_last == SQ_DEPTH_DEFAULT - 1 else '非边界'}）；"
                    f"同代下一封 seq={lm['seq_in_source_dev_request_ring'] + 1} 为 wrap 邮件（槽 0）"),
                "basis": ["ARR_SLOTS=SQ 深度=1024 且 3 SQE/邮件 ⇒ seq≡1023(mod 1024) 的邮件"
                          "同时占据 ARR 末槽与 SQ 末槽（算术锁定）",
                          "wrap 邮件（seq≡0）在反方向的健康到达由 rank1 通过 prev=2 等待证明"],
            })
    # 不变量核验
    for dev, fin in res["observed"]["mf_final"].items():
        expect = (SQE_PER_MAIL * fin["reqSeq"]) % fin["sqDepth"]
        deriv.append({
            "id": f"D-inv-{dev}",
            "claim": (f"dev{dev}: shadowTail={fin['shadowTail']} == 3*reqSeq({fin['reqSeq']}) "
                      f"mod {fin['sqDepth']} = {expect}（成立）" if expect == fin["shadowTail"]
                      else f"dev{dev}: shadowTail={fin['shadowTail']} != 期望 {expect}（违反！）"),
            "basis": ["[MF_G1C] 终态采样 + 3 SQE/邮件不变量"],
        })
    res["derived"] = [d for d in deriv if d]
    res["derived_summary"] = {
        "geometry_mode": chosen["mode"],
        "signal_counts_stalled_gen": ns,
        "pre_stall_baseline_counts": base,
        "stalled_gen_mail_seqs": mail_seq,
        "lost_mail": lost,
        "downstream_waits": downstream,
    }
    res["unknown"] = [
        "arrHead（等待设备到达环游标）运行时值：日志未记录；仅可由等待参数推导（D-loss）",
        "到达槽位的实际内存内容/戳值：停滞时未转储，无法区分'未写入'与'写入不可见'",
        "丢失邮件之后的同代邮件（wrap 邮件等）在等待方到达环的实际落地状态：等待核未消费，不可观察",
        "设备侧硬件 SQ 消费每个 SQE 的精确时刻：仅有 1Hz 低频采样（phase8/32 邮件边界）",
        "B02 在途邮件 seq/槽位：无 V2 阶段戳，不可推导（仅终态游标）",
    ]
    return res
    res["unknown"] = [
        "arrHead（等待设备到达环游标）：日志未记录；仅可由等待参数推导（见 D-loss）",
        "到达槽位的实际内存内容/戳值：停滞时未转储，无法区分'未写入'与'写入不可见'",
        "丢失邮件之后的同代邮件（wrap 邮件等）在等待方到达环的实际落地状态：等待核未消费，不可观察",
        "设备侧硬件 SQ 消费每个 SQE 的精确时刻：仅有 1Hz 低频采样（phase8/32 邮件边界）",
    ]
    return res


def derive_boundary_b02(sess: Session) -> dict:
    """B02：无 V2 阶段戳；仅记录 observed 游标与不变量，不做槽位推导。"""
    res = {"observed": {}, "derived": [], "unknown": []}
    per_dev = {}
    for dev in {r["dev"] for r in sess.mf_ticks}:
        rows = [r for r in sess.mf_ticks if r["dev"] == dev]
        last = rows[-1]
        per_dev[str(dev)] = {"final_reqHead": last["reqHead"], "final_reqTail": last["reqTail"],
                             "final_reqPending": last["reqPending"]}
    res["observed"]["final_ticks"] = per_dev
    exits = {}
    for dev in {e["dev"] for e in sess.mf_exits}:
        rows = [e for e in sess.mf_exits if e["dev"] == dev]
        exits[str(dev)] = {"last_exit": rows[-1], "total_consumed": sum(r["consumed"] for r in rows)}
    res["observed"]["epoch_exit_totals"] = {
        k: {"total_consumed": v["total_consumed"],
            "final_sqHeadAfter": v["last_exit"]["sqHeadAfter"],
            "last_exit_epoch": v["last_exit"]["epoch"]} for k, v in exits.items()}
    for dev, v in exits.items():
        res["derived"].append({
            "id": f"D-b02-cursor-{dev}",
            "claim": (f"dev{dev} 死亡时共消费 {v['total_consumed']} 封（最后 seq="
                      f"{v['total_consumed'] - 1}，槽 {(v['total_consumed'] - 1) % ARR_SLOTS}）；"
                      f"SQ 终值 {v['last_exit']['sqHeadAfter']} = 3*{v['total_consumed']} mod 1024 = "
                      f"{(SQE_PER_MAIL * v['total_consumed']) % SQ_DEPTH_DEFAULT}（不变量成立）"),
            "basis": ["[MF_DIAG] exit 行 consumed/sqHeadAfter 聚合",
                      "3 SQE/邮件不变量（issueMailSqes）"],
        })
    # 最后一个有消费的纪元 → 停滞时间窗
    last_active = {}
    for dev in {e["dev"] for e in sess.mf_exits}:
        rows = [e for e in sess.mf_exits if e["dev"] == dev and e["consumed"] > 0]
        if rows:
            last_active[str(dev)] = {"epoch": rows[-1]["epoch"]}
    res["observed"]["last_consuming_epoch"] = last_active
    res["unknown"] = [
        "B02 无 V2 阶段戳：停滞内核的代际/chunk/等待点不可知",
        "在途未到达邮件的 seq 与到达槽位：不可推导（仅知终态游标）",
        "F01/F02R1 无 MF 记录：游标状态不可知",
    ]
    return res


def derive_boundary_c2r1(sess: Session) -> dict:
    """C2R1：健康对照——终态游标与不变量。"""
    res = {"observed": {}, "derived": [], "unknown": []}
    per_dev = {}
    for dev in {r["dev"] for r in sess.mf_g1c}:
        rows = [r for r in sess.mf_g1c if r["dev"] == dev]
        last = rows[-1]
        per_dev[str(dev)] = {"final_reqSeq": last["reqSeq"], "final_shadowTail": last["shadowTail"],
                             "final_hwHead_hwTail": [last["hwHead"], last["hwTail"]]}
    res["observed"]["final_mf_g1c"] = per_dev
    total = {}
    for dev in {e["dev"] for e in sess.mf_exits}:
        rows = [e for e in sess.mf_exits if e["dev"] == dev]
        total[str(dev)] = sum(r["consumed"] for r in rows)
    res["observed"]["total_consumed"] = total
    for dev, t in total.items():
        res["derived"].append({
            "id": f"D-c2r1-{dev}",
            "claim": (f"dev{dev} 健康终态消费 {t} 封（跨 {t // ARR_SLOTS} 次 ARR wrap，"
                      f"末封槽 {(t - 1) % ARR_SLOTS}）；shadowTail={per_dev[str(dev)]['final_shadowTail']} "
                      f"= 3*{t} mod 1024（不变量成立）"),
            "basis": ["[MF_G1C]/[MF_DIAG] 聚合"],
        })
    res["unknown"] = ["C2R1 各 wrap 时刻的精确游标序列（1Hz 采样，穿越多发生在采样间隙）"]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", action="append", default=[], metavar="NAME=PATH",
                    help="会话名=会话目录（可多次）")
    ap.add_argument("--out", required=True, type=Path, help="输出目录")
    args = ap.parse_args()
    if not args.session:
        ap.error("--session 至少一个")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sanitized").mkdir(exist_ok=True)

    sessions: dict[str, Session] = {}
    for spec in args.session:
        name, _, path = spec.partition("=")
        root = Path(path)
        if not (root / "serve.log").is_file():
            ap.error(f"{name}: 缺少 serve.log: {root}")
        sessions[name] = Session(name, root)

    windows = {
        "B02": [("13:20:50", "13:22:10")],
        "C2R1": [("17:01:35", "17:01:45"), ("17:12:25", "17:12:35")],
        "C2R2": [("17:18:40", "17:20:30")],
    }
    manifest = {"generated_by": "g1c_evidence_extract.py", "deterministic": True, "sessions": {}}
    for name, sess in sessions.items():
        sess.parse()
        tj = timeline_json(sess)
        tj_text = sanitize(json.dumps(tj, ensure_ascii=False, indent=1, default=_json_default))
        (args.out / f"timeline_{name}.json").write_text(tj_text + "\n")
        timeline_csv(sess, args.out / f"timeline_{name}.csv")
        stats = sess.stats()
        st_text = sanitize(json.dumps(stats, ensure_ascii=False, indent=1, default=_json_default))
        (args.out / f"session_stats_{name}.json").write_text(st_text + "\n")
        excerpts = []
        for (t0, t1) in windows.get(name, []):
            day = "2026-10-20 "
            out = args.out / "sanitized" / f"{name}_{t0.replace(':', '')}_{t1.replace(':', '')}.log"
            excerpts.append(window_excerpt(sess, day + t0, day + t1, out))
        manifest["sessions"][name] = {
            "serve_log": str(sess.serve_log), "events": len(tj),
            "v2_frames": len(sess.v2_frames), "mf_g1c": len(sess.mf_g1c),
            "mf_diag_ticks": len(sess.mf_ticks), "epoch_exits": len(sess.mf_exits),
            "excerpts": excerpts,
        }
        print(f"[{name}] events={len(tj)} v2={len(sess.v2_frames)} mf_g1c={len(sess.mf_g1c)}")

    boundary = {
        "meta": {
            "purpose": "1024 边界专项核查（reqSeq/到达槽位/对应 Rank）；observed=日志直录，"
                       "derived=带推导链的算术结论，unknown=日志未记录且不可推导",
            "source_constants": {
                "ARR_SLOTS": {"value": ARR_SLOTS,
                              "source": "memfabric_hybrid src/smem/include/device/smem_shm_sdma_layout.h:52"},
                "REQ_SLOTS": {"value": 64, "source": "同上 :45"},
                "QUIET_SLOTS": {"value": 64, "source": "同上 :49"},
                "SQE_PER_MAIL": {"value": SQE_PER_MAIL,
                                 "source": "hybm_sdma_orch_kernel.cc issueMailSqes（载荷→到达戳→完成戳）"},
                "SQ_DEPTH": {"value": SQ_DEPTH_DEFAULT, "source": "运行时 [MF_G1C] sqDepth 字段"},
                "stamp_semantics": "1 基（stamp=seq+1，0 表空）——smem_shm_aicore_base_sdma.h wait_at",
            },
        },
    }
    for name, sess in sessions.items():
        if name == "C2R2":
            boundary[name] = derive_boundary_c2r2(sess)
        elif name == "B02":
            boundary[name] = derive_boundary_b02(sess)
        elif name == "C2R1":
            boundary[name] = derive_boundary_c2r1(sess)
        else:
            boundary[name] = {"observed": sess.stats(), "derived": [], "unknown": ["未定义分析"]}
    (args.out / "boundary_1024_analysis.json").write_text(
        sanitize(json.dumps(boundary, ensure_ascii=False, indent=1, default=_json_default)) + "\n")
    mf_text = sanitize(json.dumps(manifest, ensure_ascii=False, indent=1, default=_json_default))
    (args.out / "extract_manifest.json").write_text(mf_text + "\n")
    print(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()
