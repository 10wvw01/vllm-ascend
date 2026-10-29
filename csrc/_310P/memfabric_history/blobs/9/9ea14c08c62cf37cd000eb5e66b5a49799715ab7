#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""310P3服务器分支切换后的只读环境盘点。

不导入torch/torch_npu，不运行Device Kernel/KFC，不修改或部署SO，
不执行git checkout/fetch/pull/reset/clean。结果只供人工G0/编译部署评审。
用法:
  python3 a1_reentry_readonly.py --vllm-root /绝对路径/隔离A1工作树 \\
      --memfabric-root /绝对路径/MemFabric源码树 \\
      --out /全新绝对路径/reentry_inventory [--paths-json /绝对路径/已有SO路径清单.json]
路径清单可包含 host_system,host_python,plugin,device_so,kver_json,model。
禁止把本脚本生成的快照视作当前SO已装载的证明。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

MAX_OUTPUT = 65536
PATH_NAMES = ("host_system", "host_python", "plugin", "device_so", "kver_json", "model")
SAFE_ENVS = (
    "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME",
    "MF_SDMA_ORCH_JSON", "V2_HOST_LIB", "U1_DEVICE_SO_PATH",
    "U1_CORE_SO_LIB64", "U1_CORE_SO_PY",
    "MF_KFC_BYPASS", "MF310P_V2_NOWARMUP", "MF310P_V2_AR_ONLY_NOWARMUP",
    "MF_SDMA_ORCH_DEPLOY_HELPER", "MF_SDMA_ORCH_EPOCH_BUDGET_S",
    "LD_PRELOAD",
)
SOURCE_PATHS = (
    "src/hybm/csrc/common/hybm_physical_device_id.h",
    "src/hybm/csrc/driver/hybm_gva.cpp",
    "src/hybm/csrc/mm/hybm_acl_vmm_segment.cpp",
    "src/hybm/csrc/entity/hybm_sdma_orchestrator.cpp",
)
EXPECTED_BASENAMES = {
    "host_system": "libmf_hybm_core.so",
    "host_python": "libmf_hybm_core.so",
    "kver_json": "libmf_sdma_orch_v11.json",
}


def safe_env_view(env: dict[str, str]) -> dict:
    return {name: {"present": name in env, "value": env.get(name)}
            for name in SAFE_ENVS}


def run_readonly(argv: list[str], timeout: int = 12) -> dict:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, errors="replace",
            timeout=timeout, check=False,
        )
        return {
            "argv": argv, "returncode": proc.returncode,
            "stdout": proc.stdout[:MAX_OUTPUT], "stderr": proc.stderr[:MAX_OUTPUT],
            "truncated": len(proc.stdout) > MAX_OUTPUT or len(proc.stderr) > MAX_OUTPUT,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": type(exc).__name__ + ": " + str(exc)[:400]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def file_facts(value: str | None) -> dict:
    if not value:
        return {"status": "NOT_SUPPLIED"}
    p = Path(value).expanduser()
    out = {"requested": str(p), "absolute": p.is_absolute()}
    if not p.is_absolute():
        out["status"] = "NOT_ABSOLUTE"
        return out
    if not p.exists():
        out["status"] = "MISSING"
        return out
    try:
        real = p.resolve(strict=True)
        out.update(realpath=str(real), is_symlink=p.is_symlink())
        if real.is_file():
            out.update(status="FILE", size=real.stat().st_size, sha256=sha256_file(real))
        elif real.is_dir():
            out.update(status="DIRECTORY")
        else:
            out.update(status="OTHER")
    except OSError as exc:
        out["status"] = "UNREADABLE"
        out["error"] = str(exc)[:200]
    return out


def git_facts(root: Path) -> dict:
    if not root.is_dir():
        return {"status": "MISSING_ROOT", "path": str(root)}
    calls = {
        "head": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(root), "branch", "--show-current"],
        "status": ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        "worktrees": ["git", "-C", str(root), "worktree", "list", "--porcelain"],
        "origin_name": ["git", "-C", str(root), "remote"],
    }
    data = {key: run_readonly(cmd) for key, cmd in calls.items()}
    return {"path": str(root), "commands": data,
            "head": data["head"].get("stdout", "").strip(),
            "branch": data["branch"].get("stdout", "").strip(),
            "tracked_clean": data["status"].get("returncode") == 0
            and not data["status"].get("stdout", "").strip()}


def source_facts(mfroot: Path) -> dict:
    evidence = {}
    for rel in SOURCE_PATHS:
        p = mfroot / rel
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            evidence[rel] = {
                "sha256": sha256_file(p),
                "physical_mapper": "HybmPhysicalDeviceIdOf" in text,
                "setaccess_call_count": len(re.findall(r"AclrtMemSetAccess\s*\(", text)),
                "logical_malloc_call_count": len(re.findall(r"AclrtMallocPhysical\s*\(", text)),
                "uses_epoch_v11": "libmf_sdma_orch_v11" in text,
            }
        else:
            evidence[rel] = {"status": "MISSING"}
    meta = evidence.get("src/hybm/csrc/driver/hybm_gva.cpp", {})
    pool = evidence.get("src/hybm/csrc/mm/hybm_acl_vmm_segment.cpp", {})
    hdr = evidence.get("src/hybm/csrc/common/hybm_physical_device_id.h", {})
    evidence["three_setaccess_mapper_static_hint"] = bool(
        hdr.get("physical_mapper") and
        meta.get("physical_mapper") and pool.get("physical_mapper") and
        meta.get("setaccess_call_count") == 1 and
        pool.get("setaccess_call_count") == 2
    )
    return evidence


def gather(vllm_root: Path, mf_root: Path, files: dict, env: dict) -> dict:
    paths = {name: file_facts(files.get(name)) for name in PATH_NAMES}
    hazard_vars = [k for k in (
        "MF_KFC_BYPASS", "MF310P_V2_NOWARMUP",
        "MF310P_V2_AR_ONLY_NOWARMUP", "MF_SDMA_ORCH_DEPLOY_HELPER",
    ) if k in env]
    epoch = env.get("MF_SDMA_ORCH_EPOCH_BUDGET_S")
    hosta, hostb = paths["host_system"], paths["host_python"]
    same_host = (hosta.get("status") == "FILE" and hostb.get("status") == "FILE"
                 and hosta.get("sha256") == hostb.get("sha256"))
    results = {
        "schema": "A1_SERVER_REENTRY_READONLY_V1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "vllm": git_facts(vllm_root),
        "memfabric": git_facts(mf_root),
        "memfabric_source": source_facts(mf_root),
        "provided_paths": paths,
        "host_system_python_same_hash": same_host,
        "environment": safe_env_view(env),
        "prohibited_bypass_variables_present": hazard_vars,
        "prohibited_epoch_sentinel": epoch == "4294967295",
        "tool_versions": {
            "python": run_readonly([sys.executable, "--version"]),
            "cmake": run_readonly(["cmake", "--version"]),
            "ninja": run_readonly(["ninja", "--version"]),
            "c++": run_readonly(["c++", "--version"]),
            "pip_packages": run_readonly([sys.executable, "-m", "pip", "show",
                "memfabric_hybrid", "vllm-ascend", "torch", "torch_npu"], timeout=15),
        },
        "host_readonly": {
            "uname": run_readonly(["uname", "-a"]),
            "id": run_readonly(["id"]),
            "npu_smi": run_readonly(["npu-smi", "info"], timeout=12),
            "npu_process": run_readonly(["npu-smi", "info", "-t", "proc"], timeout=12),
            "socket_ports": run_readonly(["ss", "-ltn"], timeout=8),
            "process_names": run_readonly(["ps", "-eo", "pid,uid,comm"], timeout=8),
        },
        "unproven": [
            "same-card dual-Die health/topology/tenant ownership",
            "runtime dlopen/plugin/Host SO/KVER device registration",
            "VMM physical-vs-logical probe on current CANN",
            "KFC installed/auto-deploys on each selected Die",
            "bit-exact and actual Stock/V2 performance",
        ],
        "G0_APPROVED": False,
        "DEPLOY_APPROVED": False,
        "SMOKE_APPROVED": False,
    }
    return results


def parse_paths(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("paths JSON must be object")
    unexpected = set(obj) - set(PATH_NAMES)
    if unexpected:
        raise ValueError("unsupported path keys: " + ",".join(sorted(unexpected)))
    if any(v is not None and not isinstance(v, str) for v in obj.values()):
        raise ValueError("all path entries must be strings or null")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vllm-root", required=True, type=Path)
    ap.add_argument("--memfabric-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--paths-json", type=Path)
    args = ap.parse_args()
    for name in ("vllm_root", "memfabric_root", "out"):
        if not getattr(args, name).is_absolute():
            ap.error(name + " must be an absolute path")
    if not args.out.parent.is_dir() or args.out.exists():
        ap.error("--out must be a new absolute directory with existing parent")
    paths = parse_paths(args.paths_json)
    args.out.mkdir(mode=0o700)
    data = gather(args.vllm_root, args.memfabric_root, paths, dict(os.environ))
    outfile = args.out / "environment_inventory.json"
    with outfile.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("READ_ONLY_INVENTORY_COMPLETED", outfile)
    print("G0=NOT_GRANTED DEPLOY=NOT_GRANTED SMOKE=NOT_GRANTED")
    print("SOURCE_THREE_SETACCESS_MAPPER_HINT=",
          data["memfabric_source"]["three_setaccess_mapper_static_hint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
