#!/usr/bin/env python3
"""310P3 V2 R2-A strict static-admission inventory (stdlib only).

A PASS here is a *necessary* documentation/evidence condition, never proof
of NPU correctness. The tool does not edit code, disable the Worker guard,
trigger an NPU run, or infer device support from CPU Torch tests.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path("csrc/_310P/memfabric_mm_ar_v2/docs")
UPSTREAM_SHA = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
REQUIRED = (
    *(f"T{i:02d}" for i in range(1, 26)),
    *(f"ABI-{i:02d}" for i in range(1, 16)),
)
REQUIRED_EVIDENCE = (
    "source", "test", "runtime_wired", "cpu_contract_pass",
    "device_ops_static_reviewed", "review_round_1", "review_round_2",
)


def parse_table(text: str, prefix: str) -> dict[str, str]:
    """Only parse explicit matrix rows, never narrative 'READY' claims."""
    result = {}
    pattern = re.compile(r"^(T\d{2}|ABI-\d{2})$")
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not pattern.fullmatch(cells[0]):
            continue
        key = cells[0]
        if prefix == "T" and not key.startswith("T"):
            continue
        if prefix == "ABI" and not key.startswith("ABI-"):
            continue
        if key in result:
            raise ValueError(f"duplicate matrix ID: {key}")
        result[key] = cells[-1]
    return result


def audit(root: Path) -> dict:
    reasons = []
    matrices = {}
    for name, prefix in (
        ("V2_TRITON_REPLACEMENT_MATRIX.md", "T"),
        ("V2_API_ABI_MATRIX.md", "ABI"),
    ):
        path = root / DOCS / name
        if not path.is_file():
            reasons.append(f"missing matrix: {path}")
            continue
        matrices.update(parse_table(path.read_text(encoding="utf-8"), prefix))
    ledger_path = root / DOCS / "R2_STATIC_ADMISSION_EVIDENCE.json"
    if ledger_path.is_file():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            reasons.append(f"invalid evidence JSON: {exc}")
            ledger = {}
    else:
        reasons.append("missing R2_STATIC_ADMISSION_EVIDENCE.json")
        ledger = {}
    if ledger.get("upstream_sha") != UPSTREAM_SHA:
        reasons.append("upstream vLLM 0.27.1 SHA is not frozen in evidence")
    entries = ledger.get("entries", {})
    if not isinstance(entries, dict):
        reasons.append("evidence entries must be a dictionary")
        entries = {}
    # Fail closed if the matrix discovers a new kernel/ABI without adding it
    # to the mandatory admission inventory. Otherwise T26+ could silently
    # escape the STATIC-GO evidence check.
    untracked = sorted(set(matrices) - set(REQUIRED))
    if untracked:
        reasons.append(f"untracked mandatory matrix IDs: {untracked}")
    checks = {}
    for key in REQUIRED:
        state = matrices.get(key, "MISSING")
        entry = entries.get(key, {})
        if not isinstance(entry, dict):
            entry = {}
        ready_state = state == ("READY" if key.startswith("T") else "PASS")
        # N/A is permitted only after two reviews of an explicit
        # unreachable-in-TP2/PP1 Eager call graph.
        if state == "NOT_APPLICABLE":
            ready_state = bool(entry.get("unreachable_proof"))
        missing = [f for f in REQUIRED_EVIDENCE if not entry.get(f)]
        for f in ("source", "test"):
            if entry.get(f) and not (root / entry[f]).is_file():
                missing.append(f"{f}_file_missing")
        verdict = "READY" if ready_state and not missing else "NO-GO"
        checks[key] = {"matrix": state, "verdict": verdict, "missing": missing}
    if ledger.get("static_review_verdict") != "PASS":
        reasons.append("two-round static-review final verdict not PASS")
    if ledger.get("device_scope") != "ONDEVICE_NOT_RUN":
        reasons.append("R2-A device scope must remain ONDEVICE_NOT_RUN")
    no_go = [k for k, v in checks.items() if v["verdict"] != "READY"]
    if no_go:
        reasons.append(f"{len(no_go)} mandatory ABI/Triton nodes not ready")
    return {
        "verdict": "STATIC-NO-GO" if reasons else "STATIC-CHECKS-COMPLETE",
        "upstream_sha": UPSTREAM_SHA,
        "ready": len(REQUIRED) - len(no_go),
        "total": len(REQUIRED),
        "checks": checks,
        "reasons": reasons,
        "warning": "Never interpret this as NPU execution or permission to remove NOT_IMPLEMENTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 2 on STATIC-NO-GO")
    mode.add_argument("--report", action="store_true", help="report NO-GO without failing governance CI")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(ROOT)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"R2-A {result['verdict']}: {result['ready']}/{result['total']} mandatory nodes")
        for key, value in result["checks"].items():
            print(f"  {key}: {value['verdict']} ({value['matrix']})")
        for reason in result["reasons"]:
            print("BLOCKER:", reason)
        print(result["warning"])
    return 2 if args.check and result["verdict"] != "STATIC-CHECKS-COMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
