#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 R1同批自动续跑门: 严格验收M2048双Rank Stock/V2 Smoke。

输入必须是现场本次新建路径中的原始U1输出、实际装载和KFC/安全取证。
仅生成ALLOW_SCREEN或STOP，不运行torchrun/Kernel，不设置任何批准变量。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from u1_spec import K, N, ITERS, WARMUP, D4_DEVICE_SHA, PLUGIN_SHA, MEMFABRIC_HOST_SHA, KVER_SHA, session_id

SHA = {"device_so_sha256": D4_DEVICE_SHA, "plugin_sha256": PLUGIN_SHA,
       "host_so_sha256": MEMFABRIC_HOST_SHA, "kver11_sha256": KVER_SHA}
REQUIRED_CHECKS = (
    "fresh_same_card_dual_die_exclusive",
    "rank0_runtime_host_fixed",
    "rank1_runtime_host_fixed",
    "rank0_runtime_d4_kernel",
    "rank1_runtime_d4_kernel",
    "two_die_kver11_kfc_autodeploy_verified",
    "first_write_protocol_and_numeric_clean",
    "no_system_tenant_files_changed",
    "no_unresolved_device_error",
)


def fail_unless(b: bool, reason: str) -> None:
    if not b:
        raise ValueError(reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def samples(obj: object, n: int, offset: int, label: str):
    fail_unless(isinstance(obj, list) and len(obj) == n, label + " count mismatch")
    for i, item in enumerate(obj):
        fail_unless(isinstance(item, dict) and item.get("i") == i
                    and item.get("variant") == (i + offset) % 2, label + " order mismatch")
        for key in ("event_us", "wall_us"):
            val = item.get(key)
            fail_unless(type(val) in (int, float) and math.isfinite(val) and val > 0,
                        label + " invalid timing")


def rank_check(record: dict, stage: int, rank: int):
    arm = "stock_fp16" if stage == 1 else "memfabric_d4"
    sid = session_id("large", stage)
    want = {"experiment": "U1_SMOKE_ONLY", "scope": "large", "phase": "smoke",
            "session": sid, "stage": stage, "arm": arm, "rank": rank,
            "TP": 2, "K": K, "N": N, "dtype": "fp16",
            "M_order": [2048], "profile": False,
            "reference": "CPU_GLOO_FP32_SUM_TO_FP16",
            "stock_collective": "FP16_HCCL_no_FP32_conversion",
            "comm_isolation": "separate process; never HCCL with MemFabric pool",
            **SHA}
    for key, val in want.items():
        fail_unless(record.get(key) == val, f"{sid}/rank{rank} invalid {key}")
    result = record.get("results")
    fail_unless(isinstance(result, dict) and set(result) == {"2048", "_anchor_after"},
                f"{sid}/rank{rank} incorrect result keys")
    dat = result["2048"]
    for k, val in {"M": 2048, "arm": arm, "warmup": WARMUP["smoke"],
                   "iters": ITERS["smoke"], "bit_exact": True,
                   "status": None if stage == 1 else 0}.items():
        fail_unless(dat.get(k) == val, f"{sid}/rank{rank} invalid {k}")
    samples(dat.get("samples"), ITERS["smoke"], WARMUP["smoke"], sid)
    late = result["_anchor_after"]
    fail_unless(late.get("M") == 2048, sid + " missing late anchor")
    samples(late.get("samples"), 8, 0, sid + " late")


def evaluate(smoke_root: Path, evidence_root: Path, attestation: Path) -> dict:
    fail_unless(smoke_root.is_dir() and evidence_root.is_dir()
                and attestation.is_file(), "missing real smoke/evidence paths")
    manifest_path = smoke_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, val in {"experiment": "U1_FULLCHAIN", "scope": "large",
                     "phase": "smoke", **SHA}.items():
        fail_unless(manifest.get(key) == val, f"invalid manifest {key}")
    fail_unless(manifest.get("session_ids") == [session_id("large", 1),
                                                 session_id("large", 2)],
                "wrong smoke session order")
    physical = manifest.get("physical_devices", "")
    parts = physical.split(",")
    fail_unless(len(parts) == 2 and all(x.isdecimal() for x in parts)
                and parts[0] != parts[1], "physical die pair invalid")
    sessions = smoke_root / "sessions"
    fail_unless(sorted(x.name for x in sessions.iterdir()) ==
                sorted([session_id("large", 1), session_id("large", 2)]),
                "missing/extra smoke session")
    proofs = {}
    for stage in (1, 2):
        sid = session_id("large", stage)
        sd = sessions / sid
        log = sd / "session.log"
        fail_unless(log.is_file() and log.stat().st_size > 0,
                    sid + " missing log")
        pair = []
        for rank in (0, 1):
            rp = sd / f"{sid}_rank{rank}.json"
            data = json.loads(rp.read_text(encoding="utf-8"))
            rank_check(data, stage, rank)
            pair.append(sha(rp))
        proofs[sid] = {"rank_json_sha256": pair, "session_log_sha256": sha(log)}
    audit = json.loads(attestation.read_text(encoding="utf-8"))
    fail_unless(audit.get("scope") == "A1_R1_M2048_SMOKE_RUNTIME_ATTESTED",
                "wrong external audit scope")
    fail_unless(audit.get("physical_devices") == physical,
                "audit die pair disagrees with Smoke manifest")
    for k in REQUIRED_CHECKS:
        fail_unless(audit.get(k) is True, f"unverified runtime or safety gate: {k}")
    raw = audit.get("evidence_files")
    fail_unless(isinstance(raw, list) and len(raw) >= 6
                and len(set(raw)) == len(raw), "not enough independent trace artifacts")
    evidence_hashes = {}
    for rel in raw:
        fail_unless(isinstance(rel, str) and rel and not Path(rel).is_absolute()
                    and ".." not in Path(rel).parts, "unsafe evidence path")
        f = (evidence_root / rel).resolve()
        fail_unless(f.is_relative_to(evidence_root.resolve()) and f.is_file()
                    and f.stat().st_size > 0, "runtime proof file missing")
        evidence_hashes[rel] = sha(f)
    return {"schema": "A1_R1_SMOKE_GATE_V1", "decision": "ALLOW_SCREEN",
            "meaning": "Correctness/identity-only; NOT L1/L3/L4 performance GO",
            "smoke_manifest_sha256": sha(manifest_path),
            "runtime_attestation_sha256": sha(attestation),
            "session_proofs": proofs, "runtime_evidence_sha256": evidence_hashes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-root", type=Path, required=True)
    ap.add_argument("--evidence-root", type=Path, required=True)
    ap.add_argument("--attestation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not all(p.is_absolute() for p in (args.smoke_root, args.evidence_root,
                                           args.attestation, args.out)):
        ap.error("all paths must be absolute")
    if args.out.exists() or not args.out.parent.is_dir():
        ap.error("output must be new in existing directory")
    report = evaluate(args.smoke_root, args.evidence_root, args.attestation)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("ALLOW_SCREEN: safe to continue within user's conditional authorization; still NO L1 GO")


if __name__ == "__main__":
    main()
