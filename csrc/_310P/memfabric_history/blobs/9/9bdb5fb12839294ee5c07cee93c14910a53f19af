#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detached launcher for the A3 pipe-utilization probe, guarded by stage_guard
shared mode (user-authorized): health/parse fail-closed, external PIDs tolerated."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/w00804146/310P/a3ut-20261029/vllm-ascend")
sys.path.insert(0, str(REPO / "csrc/_310P/memfabric_mm_ar_v3/tools"))
from stage_guard import run_stage  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--card", type=int, required=True)
p.add_argument("--devices", required=True)
p.add_argument("--store-url", required=True)
p.add_argument("--variant", default="a3")
p.add_argument("--compute-rows", type=int, default=1024)
p.add_argument("--calls", type=int, default=10)
p.add_argument("--timeout", type=int, default=900)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc-per-node=2",
    str(Path(__file__).resolve().parent / "probe_a3_prof.py"),
    "--manifest",
    str(a.manifest.resolve()),
    "--out",
    str(a.out.resolve()),
    "--variant",
    a.variant,
    "--compute-rows",
    str(a.compute_rows),
    "--physical-devices",
    a.devices,
    "--store-url",
    a.store_url,
    "--calls",
    str(a.calls),
]
rc = run_stage(
    a.card,
    a.out / "lease_guard.jsonl",
    2,
    command,
    a.timeout,
    allow_external_target=True,
)
print(json.dumps({"exit_code": rc}))
sys.exit(rc)
