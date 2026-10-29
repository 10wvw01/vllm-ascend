#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 R1 Host/Device/KVER静态装载身份预审：只读，不导入torch/CANN，不加载SO。

用法：
 python3 a1_r1_loader_readonly.py --paths-json /ABS/paths.json --out /NEW/ABS/DIR

paths.json 必须包含：
 host_archive_fixed, host_system, host_python, plugin_so,
 d4_device_candidate, device_so_installed, kver11_json
可选 memfabric_mirror_root, vllm_root。不会自动设置U1批准变量或拷贝/部署库。
此脚本不能证明进程实际dlopen、设备Kernel注册、KFC启动、同卡独占。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

HOST_FIXED = "1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30"
PLUGIN = "8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1"
D4 = "2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19"
KVER = "25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78"
REQUIRED = (
    "host_archive_fixed", "host_system", "host_python", "plugin_so",
    "d4_device_candidate", "device_so_installed", "kver11_json",
)
OPTIONAL = ("memfabric_mirror_root", "vllm_root")
MAX_OUTPUT = 12000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def paths_contract(obj: object) -> dict[str, Path]:
    if not isinstance(obj, dict) or any(k not in obj for k in REQUIRED):
        raise ValueError("missing required identity paths")
    if set(obj) - set(REQUIRED + OPTIONAL):
        raise ValueError("unknown identity path name")
    if any(not isinstance(v, str) or not v.startswith("/") for v in obj.values()):
        raise ValueError("every identity path must be an absolute string")
    return {k: Path(v) for k, v in obj.items()}


def inspect(path: Path, *, is_elf: bool) -> dict:
    result = {"path": str(path)}
    try:
        original = path.lstat()
        real = path.resolve(strict=True)
        result.update(realpath=str(real), symlink=path.is_symlink())
        if not real.is_file():
            result["status"] = "NOT_FILE"
            return result
        result.update(status="FILE", bytes=real.stat().st_size, sha256=digest(real),
                      st_dev=original.st_dev, st_ino=original.st_ino)
    except OSError as exc:
        result.update(status="UNAVAILABLE", error=str(exc)[:300])
        return result
    if is_elf:
        try:
            p = subprocess.run(["readelf", "-d", str(real)],
                               capture_output=True, text=True, errors="replace",
                               timeout=10, check=False)
            result["readelf"] = {"returncode": p.returncode,
                                  "output": p.stdout[:MAX_OUTPUT],
                                  "stderr": p.stderr[:300]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["readelf"] = {"error": type(exc).__name__ + ": " + str(exc)[:200]}
    return result


def repo_info(root: Path | None) -> dict:
    if root is None:
        return {"status": "NOT_SUPPLIED"}
    if not root.is_dir():
        return {"status": "UNAVAILABLE", "path": str(root)}
    out = {"path": str(root)}
    for key, cmd in {
        "head": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(root), "branch", "--show-current"],
        "status": ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
    }.items():
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                               timeout=10, check=False)
            out[key] = {"rc": p.returncode, "text": p.stdout[:MAX_OUTPUT],
                        "stderr": p.stderr[:300]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            out[key] = {"error": str(exc)[:200]}
    return out


def evaluate(facts: dict) -> dict:
    def sha(name: str) -> str | None:
        return facts[name].get("sha256")
    return {
        "historical_fixed_host_archive_valid": sha("host_archive_fixed") == HOST_FIXED,
        "host_system_installed_fixed": sha("host_system") == HOST_FIXED,
        "host_python_installed_fixed": sha("host_python") == HOST_FIXED,
        "host_two_installed_paths_equal": bool(
            sha("host_system") and sha("host_system") == sha("host_python")),
        "plugin_file_frozen_valid": sha("plugin_so") == PLUGIN,
        "d4_candidate_frozen_valid": sha("d4_device_candidate") == D4,
        "device_installed_is_frozen_d4": sha("device_so_installed") == D4,
        "kver11_file_frozen_valid": sha("kver11_json") == KVER,
        "all_frozen_candidate_files_present": (
            sha("host_archive_fixed") == HOST_FIXED
            and sha("plugin_so") == PLUGIN
            and sha("d4_device_candidate") == D4
            and sha("kver11_json") == KVER),
        "actual_host_dlopen_path_proven": False,
        "actual_device_kernel_so_loaded_proven": False,
        "python_editable_import_frozen_proven": False,
        "card_exclusivity_current_proven": False,
        "kfc_and_first_write_correctness_proven": False,
        "permission_to_deploy": False,
        "permission_to_smoke": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.paths_json.is_absolute() or not args.out.is_absolute():
        ap.error("all arguments must have absolute paths")
    if args.out.exists() or not args.out.parent.is_dir():
        ap.error("--out must be NEW absolute directory with an existing parent")
    data = paths_contract(json.loads(args.paths_json.read_text(encoding="utf-8")))
    args.out.mkdir(mode=0o700)
    facts = {name: inspect(data[name], is_elf=name not in ("kver11_json",))
             for name in REQUIRED}
    obj = {
        "schema": "A1_R1_LOADER_READONLY_V1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "files": facts,
        "source_repos": {x: repo_info(data.get(x)) for x in OPTIONAL},
        "checks": evaluate(facts),
        "status": "STATIC_EVIDENCE_ONLY",
        "unproven": [
            "actual library loaded by fresh child rank0 and rank1 process",
            "kernel ELF/CANN registration path and loaded Device SO",
            "V2_HOST_LIB and MemFabric runtime resolving identical bytes",
            "healthy idle exclusive 16384 dev2,3 at device-start time",
            "KFC auto-deploy and first-write correctness",
        ],
        "DEVICE_ACTION_APPROVED": False,
        "DEPLOY_APPROVED": False,
        "SMOKE_APPROVED": False,
    }
    target = args.out / "r1_loader_static_inventory.json"
    with target.open("x", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    print("A1_R1_STATIC_EVIDENCE_COLLECTED", target)
    print("DEPLOY=NOT_APPROVED SMOKE=NOT_APPROVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
