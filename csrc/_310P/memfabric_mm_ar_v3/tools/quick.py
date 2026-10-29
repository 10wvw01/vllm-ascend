#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded, detached V3-only quick screen. Never launches V2/Stock performance arms."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from build import PROJECT, REPO, git, sha, source_files
from stage_guard import run_stage

sys.path.insert(0, str(PROJECT / "test"))
from historical import assess  # noqa: E402

# This project's documented four-card 310P3 server; worker verifies CANN physical IDs too.
SERVER_CARDS = {5664: "0,1", 16384: "2,3", 38144: "4,5", 43520: "6,7"}


def save(path, data):
    with path.open("x") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def verify_manifest(path):
    m = json.loads(path.read_text())
    if m["schema"] != "MF310P_V3_BUILD_V1" or m["head"] != git(REPO, "rev-parse", "HEAD"):
        raise ValueError("build HEAD does not match this checkout; rebuild privately")
    if source_files() != m["sources"]:
        raise ValueError("V3 sources changed after build")
    files = {
        **m["artifacts"],
        **m["dependency"]["installed_files"],
        **m["dependency"]["public_headers"],
        m["dependency"]["orch_json"]: m["dependency"]["orch_sha256"],
        m["dependency"]["deploy_helper"]: m["dependency"]["deploy_helper_sha256"],
    }
    for name, expected in files.items():
        if sha(name) != expected:
            raise ValueError(f"frozen dependency/artifact changed: {name}")
    if not os.access(m["dependency"]["deploy_helper"], os.X_OK):
        raise ValueError("frozen deploy helper lost the executable bit")
    return m


def run(a):
    result = {"schema": "MF310P_V3_QUICK_EXIT_V1", "status": "STOP", "device_test": "NOT_RUN"}
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
                "baseline_arms": [],
                "timeout_seconds": a.timeout,
                "dependency_findings": manifest["dependency"]["findings"],
                "scope": "short operator screen only, no release/Graph/model claim",
            },
        )
        # Avoid the historical logical/physical SetAccess split: select direct CANN IDs.
        # This is a new child process; the caller's environment is unchanged.
        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        os.environ["MF_SDMA_ORCH_JSON"] = manifest["dependency"]["orch_json"]
        os.environ["MEMFABRIC_HYBRID_HOME_PATH"] = manifest["dependency"]["home"]
        os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"] = str(Path(manifest["dependency"]["smem"]).parent)
        forbidden = ("MF_KFC_BYPASS", "MF_SDMA_ORCH_DISABLE", "MF310P_V2_NOWARMUP")
        if any(k in os.environ for k in forbidden):
            raise ValueError("KFC/first-write bypass or SDMA disable present")
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
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(PROJECT / "test/quick_worker.py"),
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
            "--protocol-reps",
            str(a.protocol_reps),
            "--packet-bytes",
            str(a.packet_bytes),
        ]
        if a.plan_a or a.a3:
            if a.plan_a and a.a3:
                raise ValueError("choose exactly one protocol")
            tensor = manifest.get("tensor_adapter")
            if not tensor or sha(tensor["path"]) != tensor["sha256"]:
                raise ValueError("build/freeze the V3.2 Tensor adapter before device admission")
            command.append("--a3" if a.a3 else "--plan-a")
        result["device_test"] = "STARTED"
        ret = run_stage(a.card, a.out / "lease_guard.jsonl", 2, command, a.timeout, deploy_flag_dir=a.out)
        result["exit_code"] = ret
        if ret:
            raise RuntimeError(f"V3 worker/lease guard stopped with {ret}; do not auto-retry")
        rank0, rank1 = [json.loads((a.out / f"rank{r}.json").read_text()) for r in (0, 1)]
        report = assess(rank0, rank1)
        report["dependency_findings"] = manifest["dependency"]["findings"]
        save(a.out / "historical_comparison.json", report)
        lines = [
            "# V3 算子级快速历史对照",
            "",
            "只运行V3；下表不是同批对照，不签正式性能GO。",
            "",
            "| M | Group | V3 p50 µs | 历史批次/臂 | 历史 p50 µs | 时延缩短 | V3跨轮漂移 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in report["comparisons"]:
            for old in row["historical_comparisons"]:
                lines.append(
                    f"| {row['M']} | {row['group']} | {row['v3_p50_us']:.3f} | "
                    f"{old['batch']}/{old['arm']} | {old['p50_us']:.3f} | "
                    f"{old['latency_reduction_pct']:+.2f}% | {row['v3_session_drift']:.2%} |"
                )
        lines.extend(["", "正值表示V3比该历史数更快；小差异、漂移或环境不一致应判为暂不能确定。", ""])
        (a.out / "REPORT.md").write_text("\n".join(lines))
        result.update(status="PASS_QUICK_SCREEN_ONLY", device_test="PASS")
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        result["finished_unix"] = time.time()
        save(a.out / "result.json", result)
    return 0 if result["status"] == "PASS_QUICK_SCREEN_ONLY" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("start", "_run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--plan-a", action="store_true")
        cmd.add_argument("--a3", action="store_true")
        cmd.add_argument("--manifest", required=True, type=Path)
        cmd.add_argument("--out", required=True, type=Path)
        cmd.add_argument("--card", required=True, type=int)
        cmd.add_argument("--devices", required=True)
        cmd.add_argument("--store-url", default="tcp://127.0.0.1:8593")
        cmd.add_argument("--samples", type=int, default=30)
        cmd.add_argument("--warmup", type=int, default=4)
        cmd.add_argument("--protocol-reps", type=int, default=8)
        cmd.add_argument("--packet-bytes", type=int, choices=[1 << 20, 2 << 20, 4 << 20], default=4 << 20)
        cmd.add_argument("--timeout", type=int, default=600)
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
    if not (1 <= a.samples <= 100 and 1 <= a.warmup <= 20 and 1 <= a.protocol_reps <= 64 and 60 <= a.timeout <= 1800):
        p.error("quick-only bounds: samples1..100/warmup1..20/protocol1..64/timeout60..1800")
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
            "baseline_performance_arms": [],
        },
    )
    print(json.dumps({"status": "STARTED_DETACHED", "pid": child.pid, "out": str(a.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
