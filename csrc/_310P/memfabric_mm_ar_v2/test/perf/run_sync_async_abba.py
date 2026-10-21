#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stock/V2 独立进程 ABBA 顺序 TP2 计时。外部先 source CANN/MemFabric env。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BRANCH = "perf/310p-v2-sync-async-value-20261021"


def cmd(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, text=True, cwd=cwd).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_case(root: Path, output: Path, env: dict[str, str], round_no: int,
             arm: str, rows: int, windows: list[int], args) -> None:
    name = f"r{round_no}_{arm}_m{rows}.log"
    path = output / name
    command = [
        "torchrun", "--standalone", "--nproc-per-node=2",
        str(Path(__file__).parent / "sync_async_tp2.py"),
        "--arm", arm, "--rows", str(rows),
        "--chunk-rows", str(args.chunk_rows),
        "--steps", str(args.steps), "--warmup", str(args.warmup),
        "--repeat", str(args.repeat), "--windows", *map(str, windows),
    ]
    print(f"START {name} windows={windows}", flush=True)
    with path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            rc = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
    raw = path.read_text(errors="replace")
    complete = []
    for line in raw.splitlines():
        if '"V2_SYNC_ASYNC"' not in line or '"complete"' not in line:
            continue
        pos = line.find("{")
        if pos < 0:
            continue
        try:
            obj = json.loads(line[pos:])
        except json.JSONDecodeError:
            continue
        if (obj.get("tag") == "V2_SYNC_ASYNC" and
                obj.get("phase") == "complete" and obj.get("status") == "PASS"
                and obj.get("arm") == arm):
            complete.append(obj.get("rank"))
    with (output / "run_index.jsonl").open("a") as f:
        f.write(json.dumps({
            "name": name, "rc": rc, "sha256": sha256(path),
            "completed_ranks": sorted(complete), "rows": rows,
            "arm": arm, "round": round_no, "windows": windows,
        }, ensure_ascii=False) + "\n")
    if rc != 0 or sorted(complete) != [0, 1]:
        raise RuntimeError(f"STOP {name} rc={rc}, completed={complete}; raw={path}")
    print(f"PASS  {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[4096])
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if (not args.out.is_absolute() or args.out.exists()
            or not args.rows or not args.windows
            or any(m not in (2048, 3072, 4096) for m in args.rows)
            or any(w < 1 or w > min(16, args.steps) for w in args.windows)
            or len(set(args.windows)) != len(args.windows)
            or not 1 <= args.steps <= 40 or args.timeout < 60
            or args.chunk_rows not in (256, 512, 1024)):
        parser.error("非法参数：out 必须是不存在的绝对目录，M 2048/3072/4096")
    root = Path(__file__).resolve().parents[5]
    if cmd("git", "branch", "--show-current", cwd=root) != BRANCH:
        parser.error(f"当前不在独立性能分支 {BRANCH}")
    if cmd("git", "status", "--porcelain", "--untracked-files=no", cwd=root):
        parser.error("源码工作树不洁净：拒绝缺乏溯源的 A/B")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1") != "0,1":
        parser.error("只允许设备 0,1")
    if os.environ.get("MF_KFC_BYPASS") == "1":
        parser.error("拒绝 KFC 绕过环境")
    mf_json = Path(os.environ.get("MF_SDMA_ORCH_JSON", ""))
    if mf_json.name != "libmf_sdma_orch_v11.json" or not mf_json.is_file():
        parser.error("必须显式提供仍在现行基线的 KVER11 JSON，不能换 wgm-dev-310p")
    args.out.mkdir(parents=True, mode=0o700)
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    env["V2_SYNC_ASYNC_REPO_ROOT"] = str(root)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR"] = "1"
    env["VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2"] = "1"
    env.setdefault("VLLM_ASCEND_310P_MEMFABRIC_STORE_URL", "tcp://127.0.0.1:8581")
    provenance = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "repo_sha": cmd("git", "rev-parse", "HEAD", cwd=root),
        "repo_branch": BRANCH,
        "memfabric_json_path": str(mf_json.resolve()),
        "memfabric_json_sha256": sha256(mf_json),
        "hccl_expansion_mode": env.get("HCCL_OP_EXPANSION_MODE", "<unset>"),
        "device_map": env["ASCEND_RT_VISIBLE_DEVICES"],
        "rows": args.rows, "steps": args.steps, "windows": args.windows,
        "chunk_rows": args.chunk_rows,
        "note": "运行前由 OpenCode 审核现装 Host SO/KVER11 和 GPU 独占状态",
    }
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    try:
        for m in args.rows:
            run_case(root, args.out, env, 1, "stock", m, args.windows, args)
            run_case(root, args.out, env, 1, "v2", m, args.windows, args)
            run_case(root, args.out, env, 2, "v2", m, args.windows[::-1], args)
            run_case(root, args.out, env, 2, "stock", m, args.windows[::-1], args)
        command = [
            sys.executable, str(Path(__file__).parent / "summarize_sync_async.py"),
            str(args.out), "--rows", *map(str, args.rows),
            "--windows", *map(str, args.windows), "--output",
            str(args.out / "sync_async_summary.json"),
        ]
        subprocess.run(command, cwd=root, env=env, check=True)
        print(f"ALL PASS; {args.out / 'sync_async_summary.json'}", flush=True)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"ABORT; evidence retained at {args.out}: {e}",
              file=sys.stderr, flush=True)
        raise SystemExit(3) from e


if __name__ == "__main__":
    main()
