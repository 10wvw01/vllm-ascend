#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detached single-arm unified-boundary runner; independent HCCL/MemFabric processes.

Full input/output Wall + Event with optional A3 host/device attribution controls.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from quick import PROJECT, REPO, SERVER_CARDS, git, sha, verify_manifest
from stage_guard import run_stage

sys.path.insert(0, str(PROJECT / "test"))


def save(path, data):
    with path.open("x") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run(a):
    result = {"schema": "MF310P_V3_UNIFIED_EXIT_V1", "status": "STOP", "device_test": "NOT_RUN"}
    try:
        manifest = verify_manifest(a.manifest)
        if SERVER_CARDS.get(a.card) != a.devices:
            raise ValueError("card/device pair differs from the documented server mapping")
        if a.out.joinpath("result.json").exists():
            raise FileExistsError("refuse to rerun/overwrite a completed attempt")
        with (a.out / "cpu.log").open("x") as log:
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(PROJECT / "test"), "--import-mode=importlib"],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=90,
                check=True,
            )
        save(
            a.out / "g0.json",
            {
                "head": manifest["head"],
                "manifest_sha256": sha(a.manifest),
                "card": a.card,
                "physical_devices": a.devices,
                "cpu": "PASS",
                "lease_note": a.lease_note,
                "arms": [a.arm],
                "timeout_seconds": a.timeout,
                "scope": "full Tensor input/output, independent process per arm; paired controls",
            },
        )
        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        os.environ["MF_SDMA_ORCH_JSON"] = manifest["dependency"]["orch_json"]
        os.environ["MEMFABRIC_HYBRID_HOME_PATH"] = manifest["dependency"]["home"]
        os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"] = str(Path(manifest["dependency"]["smem"]).parent)
        forbidden = ("MF_KFC_BYPASS", "MF_SDMA_ORCH_DISABLE", "MF310P_V2_NOWARMUP")
        if any(k in os.environ for k in forbidden):
            raise ValueError("KFC/first-write bypass or SDMA disable present")
        if a.arm == "v2":
            spec = manifest.get("v2")
            if not spec or spec.get("kind") != "D4_TRIAL_KVER11" or not spec.get("environment"):
                raise ValueError("V2 requires separately frozen D4/KVER11 specification")
            for filename, expected in {spec["library"]: spec["sha256"], **spec["dependencies"]}.items():
                if sha(filename) != expected:
                    raise ValueError("V2 library/dependency fingerprint changed")
        mf_lib_dirs = sorted(
            {str(Path(p).parent) for p in manifest["dependency"]["installed_files"] if "/lib64/" in p or "/lib/" in p}
        )
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [
                str(a.manifest.parent),
                *mf_lib_dirs,
                str(Path(manifest["cann"]) / "aarch64-linux/lib64"),
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        )
        if a.arm == "v2":
            allowed = {
                "LD_LIBRARY_PATH",
                "MF_SDMA_ORCH_JSON",
                "MEMFABRIC_HYBRID_HOME_PATH",
                "MEMFABRIC_HYBRID_EXTEND_LIB_PATH",
                "V2_HOST_LIB",
            }
            if set(spec["environment"]) - allowed:
                raise ValueError("unapproved V2 environment keys")
            if not spec["environment"].get("MF_SDMA_ORCH_JSON", "").endswith("libmf_sdma_orch_v11.json"):
                raise ValueError("V2 must use frozen KVER11 environment, not V3 KVER28")
            os.environ.update(spec["environment"])
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(PROJECT / "test/unified_worker.py"),
            "--manifest",
            str(a.manifest),
            "--out",
            str(a.out),
            "--run-id",
            a.out.name,
            "--store-url",
            a.store_url,
            "--physical-devices",
            a.devices,
            "--samples",
            str(a.samples),
            "--warmup",
            str(a.warmup),
            "--arm",
            a.arm,
            "--compute-rows",
            str(a.compute_rows),
        ]
        result["device_test"] = "STARTED"
        ret = run_stage(a.card, a.out / "lease_guard.jsonl", 2, command, a.timeout, deploy_flag_dir=a.out)
        result["exit_code"] = ret
        if ret:
            raise RuntimeError(f"unified worker/lease guard stopped with {ret}; do not auto-retry")
        rank0 = json.loads((a.out / "unified_rank0.json").read_text())
        rank1 = json.loads((a.out / "unified_rank1.json").read_text())
        if rank0["status"] != "PASS" or rank1["status"] != "PASS":
            raise RuntimeError("attribution rank reports are not PASS")
        from unified_metrics import paired_report

        comparison = paired_report(rank0, rank1)
        save(a.out / "paired.json", comparison)
        result.update(status="PASS_UNIFIED_ONLY", device_test="PASS")
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        result["finished_unix"] = time.time()
        save(a.out / "result.json", result)
    return 0 if result["status"] == "PASS_UNIFIED_ONLY" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("start", "_run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--manifest", required=True, type=Path)
        cmd.add_argument("--out", required=True, type=Path)
        cmd.add_argument("--card", required=True, type=int)
        cmd.add_argument("--devices", required=True)
        cmd.add_argument("--store-url", default="tcp://127.0.0.1:8594")
        cmd.add_argument("--samples", type=int, default=30)
        cmd.add_argument("--warmup", type=int, default=4)
        cmd.add_argument("--arm", choices=["stock", "v2", "v3", "a3", "a3-2", "a3-tail"], required=True)
        cmd.add_argument("--compute-rows", type=int, choices=[512, 1024], default=1024)
        cmd.add_argument("--timeout", type=int, default=900)
        cmd.add_argument("--lease-note", default="operator must obtain an exclusive dual-die window")
    status = sub.add_parser("status")
    status.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.action == "status":
        result = a.out / "result.json"
        print(
            result.read_text()
            if result.exists()
            else json.dumps({"status": "RUNNING_OR_INTERRUPTED", "launch": str(a.out / "launch.json")})
        )
        return 0
    a.manifest = a.manifest.resolve(strict=True)
    if not (1 <= a.samples <= 100 and 1 <= a.warmup <= 20 and 60 <= a.timeout <= 1800):
        p.error("attribution bounds: samples1..100/warmup1..20/timeout60..1800")
    if a.action == "_run":
        return run(a)
    verify_manifest(a.manifest)
    if SERVER_CARDS.get(a.card) != a.devices:
        p.error("wrong physical card/device pair")
    a.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "_run", *sys.argv[2:]]
    with (a.out / "console.log").open("x") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL
        )
    save(
        a.out / "launch.json",
        {
            "pid": child.pid,
            "argv": command,
            "head": git(REPO, "rev-parse", "HEAD"),
            "started_unix": time.time(),
            "baseline_performance_arms": [a.arm] if a.arm in ("stock", "v2") else [],
        },
    )
    print(json.dumps({"status": "STARTED_DETACHED", "pid": child.pid, "out": str(a.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
