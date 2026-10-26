#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1只读取证：不编译、不导入torch_npu、不运行Kernel，不签发U1实机许可。

用法: python3 a1_g0_readonly.py --out /NEW/ABSOLUTE/a1_g0.json
失败也写入独立取证文件（如果指定路径可新建）；脚本不设置任何 *_APPROVED 环境变量。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from u1_spec import D4_DEVICE_SHA, PLUGIN_SHA, MEMFABRIC_HOST_SHA, KVER_SHA

ROOT = Path(__file__).resolve().parents[4]
ENV_ARTIFACTS = {
    "device_so_sha256": ("U1_DEVICE_SO_PATH", D4_DEVICE_SHA),
    "plugin_sha256": ("V2_HOST_LIB", PLUGIN_SHA),
    "host_so_sha256": ("U1_CORE_SO_LIB64", MEMFABRIC_HOST_SHA),
    "host_python_sha256": ("U1_CORE_SO_PY", MEMFABRIC_HOST_SHA),
    "kver11_sha256": ("MF_SDMA_ORCH_JSON", KVER_SHA),
}
DANGEROUS_FLAGS = (
    "MF_KFC_BYPASS", "MF310P_V2_NOWARMUP", "MF310P_V2_AR_ONLY_NOWARMUP",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for buf in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(buf)
    return digest.hexdigest()


def judge(facts: dict) -> dict:
    """Fail-closed static fingerprint judgment; never proves device exclusivity.

    Pure function for stdlib CPU tests. Flags are presence-based: even '0'
    still means an unsafe bypass variable has been explicitly defined.
    """
    errors: list[str] = []
    devs = facts.get("visible_devices", "")
    if (not isinstance(devs, str) or
            not re.fullmatch(r"[0-9]+,[0-9]+", devs) or
            devs.split(",")[0] == devs.split(",")[-1]):
        errors.append("Invalid/duplicate physical TP2 device IDs")
    if facts.get("git_dirty") is not False:
        errors.append("Git tracked tree dirty or not observed")
    if not isinstance(facts.get("git_head"), str) or len(facts["git_head"]) != 40:
        errors.append("Git HEAD not trustworthy")
    sha_map = facts.get("sha256", {})
    for field, (_, expected) in ENV_ARTIFACTS.items():
        if sha_map.get(field) != expected:
            errors.append(f"Missing/wrong approved fingerprint: {field}")
    if facts.get("dangerous_flags"):
        errors.append("Unsafe bypass variables present")
    if not facts.get("npu_smi_readonly_ok"):
        errors.append("npu-smi read-only snapshot missing/error")
    if not facts.get("required_tools"):
        errors.append("Missing executable tools")
    # Loaded-plugin identity is a separate *manual* runtime gate.
    # It must never authorize kernel launch, but must not permanently
    # mark otherwise valid on-disk fingerprints as inconsistent either.
    loaded = facts.get("python_plugin_loaded_path_verified") is True
    return {
        "static_fingerprint_consistent": not errors,
        "runtime_loaded_plugin_verified": loaded,
        "G0_APPROVED": False,
        "SMOKE_APPROVED": False,
        "errors": errors,
        "human_required": [
            "Verify both device IDs belong to one physical 310P3 card; npu-smi snapshot alone does not prove it",
            "Verify both dies exclusive and currently idle (processes/other tenant)",
            "Verify actual loaded SO and KVER11 AICPU registration, not only on-disk SHA",
            "Verify independent restore/rollback plan and backup SHA prior to any swap",
            "Review Stock HCCL versus MemFabric independent communication process isolation",
            "Approve G0 explicitly; then approve a separate M2048 smoke before screen",
        ],
    }


def _command(argv: list[str], *, timeout: int = 10) -> dict:
    try:
        out = subprocess.run(argv, check=False, capture_output=True, text=True,
                             timeout=timeout)
        return {"ok": out.returncode == 0,
                "returncode": out.returncode,
                "stdout": out.stdout[-16000:], "stderr": out.stderr[-3000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:300]}


def collect(root: Path, env: dict[str, str]) -> dict:
    """Only filesystem reads, git status and npu-smi info; no device initialization."""
    git_head = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    git_dirty = _command(["git", "-C", str(root), "status", "--porcelain",
                          "--untracked-files=no"])
    device_info = (_command(["npu-smi", "info"], timeout=15)
                   if shutil.which("npu-smi") else {"ok": False, "error": "npu-smi absent"})
    sha_map: dict[str, str | None] = {}
    artifacts: dict[str, str] = {}
    for field, (var, _) in ENV_ARTIFACTS.items():
        raw = env.get(var, "")
        path = Path(raw) if raw.startswith("/") else None
        if path and path.is_file() and not path.is_symlink():
            sha_map[field] = _sha(path)
            artifacts[field] = str(path)
        else:
            sha_map[field] = None
            artifacts[field] = "missing/relative/symlink"
    required = ("python3", "torchrun", "timeout", "sha256sum", "nm",
                "npu-smi", "readlink")
    tools = {name: shutil.which(name) is not None for name in required}
    # Dynamic Python import can initialize NPU/modify runtime: do NOT do it here.
    facts = {
        "repo_root": str(root),
        "git_head": git_head.get("stdout", "").strip() if git_head["ok"] else None,
        "git_dirty": (bool(git_dirty.get("stdout", "").strip())
                      if git_dirty["ok"] else None),
        "visible_devices": env.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "sha256": sha_map,
        "artifact_paths": artifacts,
        "dangerous_flags": [f for f in DANGEROUS_FLAGS if f in env],
        "npu_smi_readonly_ok": device_info["ok"],
        "npu_smi_output": device_info.get("stdout", "")[-16000:],
        "npu_smi_error": device_info.get("stderr", "")[-1500:],
        "required_tools": all(tools.values()),
        "tool_availability": tools,
        "python_plugin_loaded_path_verified": False,
        "loaded_plugin_note": ("Read-only probe intentionally does not import "
                               "vllm_ascend_C; the separate approved G0 "
                               "process must verify Python-loaded plugin SO path"),
    }
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if (not args.out.is_absolute() or args.out.exists()
            or not args.repo_root.is_dir()):
        parser.error("new absolute output and existing repository root required")
    facts = collect(args.repo_root.resolve(), dict(os.environ))
    report = {
        "contract": "A1_G0_READONLY_V1",
        "hardware_executed": False,
        "device_kernel_executed": False,
        "facts": facts,
        "decision": judge(facts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("A1 G0 READONLY evidence recorded; no run approval emitted: " + str(args.out))


if __name__ == "__main__":
    main()
