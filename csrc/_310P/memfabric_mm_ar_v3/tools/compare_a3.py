#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detached same-card A3 / A3-2 / A3-tail screen with bracketed A3 drift."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

from build import PROJECT
from quick import verify_manifest

ORDER = [("a3-before", "a3"), ("a3-2", "a3-2"), ("a3-tail", "a3-tail"), ("a3-after", "a3")]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def summarize(reports):
    if set(reports) != {(label, compute) for label, _ in ORDER for compute in (512, 1024)}:
        raise ValueError("incomplete batch stages")
    cases = {}
    for (label, compute), report in reports.items():
        if report.get("formal_performance_go") or report.get("status") != "HISTORICAL_COMPARISON_ONLY":
            raise ValueError("unexpected stage report")
        rows = report["paired_sessions"]
        expected = {(m, g, r) for m in (2048, 4096) for g in (512, 1024, 2048) for r in (0, 1)}
        actual = [(row["M"], row["group"], row["repeat"]) for row in rows]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("missing or duplicate stage cases")
        for row in rows:
            if not math.isfinite(row["p50_us"]) or row["p50_us"] <= 0:
                raise ValueError("invalid stage latency")
            cases.setdefault((row["M"], row["group"], compute), {}).setdefault(label, []).append(row["p50_us"])
    result = []
    for (m, comm, compute), values in sorted(cases.items()):
        if set(values) != {x[0] for x in ORDER} or any(len(x) != 2 for x in values.values()):
            raise ValueError("incomplete bracketed cases")
        centers = {k: median(v) for k, v in values.items()}
        before, after = centers["a3-before"], centers["a3-after"]
        drift = max(before, after) / min(before, after) - 1
        stable = drift <= 0.05 and all(max(v) / min(v) - 1 <= 0.05 for v in values.values())
        baseline = median([before, after])
        result.append(
            {
                "M": m,
                "comm_rows": comm,
                "mm_tile_rows": compute // 8,
                "compute_rows": compute,
                "paired_p50_us": centers,
                "a3_bracket_drift": drift,
                "stable_screen": stable,
                "latency_reduction_vs_bracketed_a3": {k: 1 - centers[k] / baseline for k in ["a3-2", "a3-tail"]},
            }
        )
    return {"same_batch_candidate_screen": True, "formal_performance_go": False, "cases": result}


def run(args):
    report = {"status": "STOP", "order": ORDER, "compute_order": [512, 1024]}
    try:
        manifest = verify_manifest(args.manifest)
        report["head"] = manifest["head"]
        stages = {}
        for compute in (512, 1024):
            for label, variant in ORDER:
                path = args.out / f"{label}-c{compute}"
                cmd = [
                    sys.executable,
                    str(PROJECT / "tools/quick.py"),
                    "start",
                    "--a3",
                    "--a3-variant",
                    variant,
                    "--compute-rows",
                    str(compute),
                    "--manifest",
                    str(args.manifest),
                    "--out",
                    str(path),
                    "--card",
                    str(args.card),
                    "--devices",
                    args.devices,
                    "--timeout",
                    str(args.timeout),
                    "--lease-note",
                    args.lease_note,
                ]
                subprocess.run(cmd, check=True, timeout=30)
                deadline = time.monotonic() + args.timeout + 60
                while not (path / "result.json").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{label}: no completed result; STOP, do not launch next stage")
                    time.sleep(2)
                exit_state = json.loads((path / "result.json").read_text())
                if exit_state.get("status") != "PASS_QUICK_SCREEN_ONLY":
                    raise RuntimeError(f"{label} failed: {exit_state}")
                for rank in (0, 1):
                    config = json.loads((path / f"rank{rank}.json").read_text())["configuration"]
                    if (config.get("variant"), config.get("compute_rows"), config.get("mm_tile_rows")) != (
                        variant,
                        compute,
                        compute // 8,
                    ):
                        raise ValueError("stage variant/tiling identity mismatch")
                stages[(label, compute)] = json.loads((path / "historical_comparison.json").read_text())
        report.update(status="PASS_CANDIDATE_SCREEN_ONLY", comparison=summarize(stages))
    except BaseException as exc:
        report["error"] = repr(exc)
    save(args.out / "batch_result.json", report)
    return 0 if report["status"] == "PASS_CANDIDATE_SCREEN_ONLY" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "_run", "status"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--card", type=int, default=38144)
    parser.add_argument("--devices", default="4,5")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--lease-note", default="exclusive lease must cover all eight stage sessions")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.action == "status":
        result = args.out / "batch_result.json"
        print(result.read_text() if result.exists() else '{"status":"RUNNING_OR_INTERRUPTED"}')
        return 0
    if not args.manifest:
        parser.error("--manifest required")
    args.manifest = args.manifest.resolve(strict=True)
    if args.action == "_run":
        return run(args)
    verify_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, str(Path(__file__).resolve()), "_run", *sys.argv[2:]]
    with (args.out / "batch_console.log").open("x") as log:
        child = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
        )
    save(args.out / "batch_launch.json", {"pid": child.pid, "argv": cmd})
    print(json.dumps({"status": "STARTED_DETACHED", "pid": child.pid, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
