#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sequential independent-process controls; STOP at first failure, never auto-retry."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

from build import PROJECT
from quick import verify_manifest

ORDER = (
    [("stock-before", "stock", 1024), ("v2-before", "v2", 1024), ("v3", "v3", 1024)]
    + [(f"{arm}-c{c}", arm, c) for c in (512, 1024) for arm in ("a3", "a3-2", "a3-tail")]
    + [("v2-after", "v2", 1024), ("stock-after", "stock", 1024)]
)


def compare(reports):
    if set(reports) != {label for label, _, _ in ORDER}:
        raise ValueError("incomplete campaign")
    identities = [x["identity"] for x in reports.values()]
    if any(x != identities[0] for x in identities):
        raise ValueError("mixed manifests/builds")
    result = []
    for label, arm, compute in ORDER:
        report = reports[label]
        if report.get("contract") != "ND_X_NZ_W_TO_READY_ND_Y_V1":
            raise ValueError("inconsistent timing boundary")
        expected_comms = [1024] if arm in ("stock", "v2") else [512, 1024, 2048]
        if report["configuration"]["M"] != [2048, 4096] or report["configuration"]["comm_rows"] != expected_comms:
            raise ValueError("incomplete campaign geometry")
        if report["configuration"]["arm"] != arm or report["configuration"]["compute_rows"] != compute:
            raise ValueError("wrong arm/tiling")
        if arm in ("stock", "v2"):
            continue
        for m in (2048, 4096):
            for comm in report["configuration"]["comm_rows"]:
                for metric in ("wall_us", "event_us"):
                    own = next(
                        x for x in report["controls"] if (x["M"], x["comm_rows"], x["metric"]) == (m, comm, metric)
                    )
                    own_time = median([own["centers"]["off-before"], own["centers"]["off-after"]])
                    comparisons = {}
                    for baseline in ("stock", "v2"):
                        controls = [
                            next(
                                x
                                for x in reports[f"{baseline}-{suffix}"]["controls"]
                                if x["M"] == m and x["metric"] == metric
                            )
                            for suffix in ("before", "after")
                        ]
                        times = [median([x["centers"]["off-before"], x["centers"]["off-after"]]) for x in controls]
                        drift = max(times) / min(times) - 1
                        comparisons[baseline] = {
                            "baseline_us": median(times),
                            "bracket_drift": drift,
                            "stable": drift <= 0.05
                            and own["stable_bracket"]
                            and all(x["stable_bracket"] for x in controls),
                            "latency_reduction": 1 - own_time / median(times),
                        }
                    result.append(
                        {
                            "arm": arm,
                            "compute_rows": compute,
                            "M": m,
                            "comm_rows": comm,
                            "metric": metric,
                            "candidate_us": own_time,
                            "comparisons": comparisons,
                        }
                    )
    return {"formal_performance_go": False, "comparisons": result, "scope": "unified-boundary operator attribution"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["start", "_run", "status"])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--card", type=int, default=38144)
    p.add_argument("--devices", default="4,5")
    p.add_argument("--lease-note", default="real exclusive lease required for whole campaign")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument(
        "--allow-shared-card",
        action="store_true",
        help="operator-authorized shared-card run: tolerate external target PIDs; health stays fail-closed",
    )
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.action == "status":
        path = a.out / "campaign.json"
        print(path.read_text() if path.exists() else '{"status":"RUNNING_OR_INTERRUPTED"}')
        return
    if not a.manifest:
        p.error("--manifest required")
    a.manifest = a.manifest.resolve(strict=True)
    verify_manifest(a.manifest)
    if a.action == "start":
        a.out.mkdir(parents=True, exist_ok=False)
        with (a.out / "console.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "_run", *sys.argv[2:]],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        print(json.dumps({"pid": child.pid, "out": str(a.out)}))
        return
    state = {"status": "STOP", "completed": [], "order": ORDER}
    try:
        reports = {}
        for label, arm, compute in ORDER:
            path = a.out / label
            command = [
                sys.executable,
                str(PROJECT / "tools/unified.py"),
                "start",
                "--manifest",
                str(a.manifest),
                "--out",
                str(path),
                "--arm",
                arm,
                "--compute-rows",
                str(compute),
                "--card",
                str(a.card),
                "--devices",
                a.devices,
                "--lease-note",
                a.lease_note,
                "--timeout",
                str(a.timeout),
            ]
            if a.allow_shared_card:
                command.append("--allow-shared-card")
            subprocess.run(command, check=True, timeout=30)
            deadline = time.monotonic() + a.timeout + 120
            while not (path / "result.json").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("child state unknown; no next stage")
                time.sleep(2)
            if json.loads((path / "result.json").read_text())["status"] != "PASS_UNIFIED_ONLY":
                raise RuntimeError(f"{label}: STOP; retain scene")
            reports[label] = json.loads((path / "paired.json").read_text())
            state["completed"].append(label)
        state.update(status="PASS_ATTRIBUTION_ONLY", **compare(reports))
    except BaseException as exc:
        state["error"] = repr(exc)
    (a.out / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["status"] != "PASS_ATTRIBUTION_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
