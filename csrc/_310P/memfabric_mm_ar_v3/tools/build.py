#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Private CANN build. Does not replace installed libraries or deploy an AICPU kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
MF_BASE = "db38d75e8485d31451302219eb7a1bfeb1dd107c"
MF_UPSTREAM = "ba3e7f42873b6f2a4e8d118c03a14d1f43d39ab4"
MF_TREE = "4fe624cabbf62d6475f79cc63bd19dcbedacb0e7"
MF_FIX_BRANCH = "fix/310p-v3-orch-deploy-helper-20261029"
MF_SYNC_BASE = "b9d46f97955b11a8b7a892e9fd8d516e66346a65"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def only(paths, label):
    found = sorted({p.resolve() for p in paths if p.is_file()})
    if len(found) != 1:
        raise ValueError(f"need exactly one {label}, found {found}")
    return found[0]


def dependency(source, home):
    head = git(source, "rev-parse", "HEAD")
    if head != MF_BASE:
        raise ValueError(
            f"MemFabric must be pinned to {MF_BASE} ({MF_FIX_BRANCH}); found {head}. Audit a new dependency explicitly."
        )
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("MemFabric source contains unrecorded changes")
    if git(source, "rev-parse", "HEAD^{tree}") != MF_TREE:
        raise ValueError("MemFabric source tree differs from the audited fix branch")
    base_check = subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", MF_SYNC_BASE, head])
    if base_check.returncode != 0:
        raise ValueError("fix branch is not rooted at the audited sync commit b9d46f97")
    version_info = only(home.glob("**/version.info"), "installed package version.info")
    if not re.search(rf"^CommitId:\s*{head}\s*$", version_info.read_text(), re.MULTILINE):
        raise ValueError("installed package CommitId does not match the pinned MemFabric source")
    inc = only(home.glob("**/include/smem/device/smem_shm_aicore_base_sdma.h"), "installed SDMA header").parent
    host = only(home.glob("**/include/smem/host/smem_shm.h"), "installed host header").parent
    smem = only(home.glob("**/libmf_smem.so"), "public SMEM library")
    hybm = only(home.glob("**/lib64/libmf_hybm_core.so"), "orchestrator host library")
    helper = only(home.glob("**/lib64/orch_deploy_helper"), "out-of-process deploy helper")
    if not os.access(helper, os.X_OK):
        raise ValueError("orch_deploy_helper is not executable")
    if helper.parent != hybm.parent:
        raise ValueError("deploy helper must sit next to libmf_hybm_core.so for dladdr discovery")
    contract = source / "src/hybm/ops/hybm_kernel/hybm_sdma_contract.h"
    kver = int(re.search(r"HYBM_SDMA_ORCH_KVER\s*=\s*(\d+)", contract.read_text()).group(1))
    orch = only(home.glob(f"**/libmf_sdma_orch_v{kver}.json"), "versioned launch JSON")
    op = json.loads(orch.read_text())["HybmSdmaOrchEpoch"]["opInfo"]
    if op["kernelSo"] != f"libmf_sdma_orch_v{kver}.so" or op["opKernelLib"] != "KFCKernel":
        raise ValueError("launch JSON does not match source KVER/type5")
    for name in ("smem_shm_aicore_base_sdma.h", "smem_shm_sdma_layout.h"):
        if sha(inc / name) != sha(source / "src/smem/include/device" / name):
            raise ValueError(f"installed/source public header mismatch: {name}")
    header = (inc / "smem_shm_aicore_base_sdma.h").read_text()
    if "SMEM_SHM_SDMA_OP_ADD_FP16" not in header or not re.search(r"SMEM_SDMA_SQE_OP_FP16_ADD\s*=\s*0x03", header):
        raise ValueError("FP16 ADD opcode is not 0x03")
    kernel = (source / "src/hybm/ops/hybm_kernel/hybm_sdma_orch_kernel.cc").read_text()
    findings = []
    if re.search(r"scratch,\s*scratch,\s*8", kernel):
        findings.append("KVER28_PADDING_SELF_COPY_PRESENT: quick screen cannot certify long-run stability")
    libs = sorted(p for p in home.rglob("*.so") if p.is_file())
    identities = {str(p.resolve()): sha(p) for p in libs}
    default_json = f"libmf_sdma_orch_v{kver}.json".encode()
    if not any(default_json in p.read_bytes() for p in libs if "hybm_core" in p.name):
        raise ValueError("installed Host binary does not embed the expected default KVER")
    return {
        "source_head": head,
        "upstream_head": MF_UPSTREAM,
        "source_tree": MF_TREE,
        "fix_branch": MF_FIX_BRANCH,
        "fix_base": MF_SYNC_BASE,
        "home": str(home),
        "kver": kver,
        "orch_json": str(orch),
        "orch_sha256": sha(orch),
        "deploy_helper": str(helper),
        "deploy_helper_sha256": sha(helper),
        "installed_files": identities,
        "smem": str(smem),
        "device_include": str(inc),
        "host_include": str(host),
        "public_headers": {
            str(inc / n): sha(inc / n) for n in ("smem_shm_aicore_base_sdma.h", "smem_shm_sdma_layout.h")
        },
        "findings": findings,
    }


def source_files():
    files = list((PROJECT / "src").rglob("*")) + list((PROJECT / "tools").glob("*.py"))
    files += list((PROJECT / "test").glob("*.py"))
    files += [REPO / "vllm_ascend/_310p/ops/memfabric_mm_ar_v3.py"]
    return {str(p.relative_to(REPO)): sha(p) for p in files if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memfabric-source", type=Path, required=True)
    parser.add_argument("--memfabric-home", type=Path, required=True)
    parser.add_argument("--cann", type=Path, default=os.environ.get("ASCEND_HOME_PATH"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cann is None:
        parser.error("--cann or ASCEND_HOME_PATH required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dep = dependency(args.memfabric_source.resolve(), args.memfabric_home.resolve())
    cann = args.cann.resolve()
    compiler = next(
        (p for p in (cann / "bin/bisheng", cann / "compiler/aarch64-linux/bin/bisheng") if p.is_file()), None
    )
    if compiler is None:
        raise FileNotFoundError("CANN BiSheng not found")
    host_compiler = shutil.which("g++")
    if not host_compiler:
        raise FileNotFoundError("g++ not found")
    includes = [cann / "aarch64-linux/include", cann / "aarch64-linux/ascendc/include"]
    libdir = cann / "aarch64-linux/lib64"
    device_so = out / "libmf310p_v3_device.so"
    host_so = out / "libmf310p_v3_host.so"
    commands = [
        [
            str(compiler),
            "--npu-arch=dav-2002",
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-x",
            "asc",
            str(PROJECT / "src/op_kernel/mf310p_v3_kernel.asc"),
            "-x",
            "none",
            "-o",
            str(device_so),
            *[f"-I{p}" for p in includes],
            f"-I{dep['device_include']}",
            f"-L{libdir}",
            "-lascendcl",
            "-lruntime",
            "-ldl",
            "-pthread",
            "-lm",
            f"-Wl,-rpath,{libdir}",
        ],
        [
            host_compiler,
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wl,-z,defs",
            str(PROJECT / "src/op_host/mf310p_v3_host.cpp"),
            "-o",
            str(host_so),
            f"-I{includes[0]}",
            f"-I{dep['host_include']}",
            str(device_so),
            dep["smem"],
            f"-L{libdir}",
            "-lascendcl",
            "-pthread",
            f"-Wl,-rpath,$ORIGIN:{Path(dep['smem']).parent}:{libdir}",
        ],
    ]
    for i, cmd in enumerate(commands):
        with (out / f"build_{i}.log").open("x") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = {
        "schema": "MF310P_V3_BUILD_V1",
        "created_unix": time.time(),
        "head": git(REPO, "rev-parse", "HEAD"),
        "sources": source_files(),
        "dependency": dep,
        "cann": str(cann),
        "compiler": str(compiler),
        "compiler_version": subprocess.check_output([str(compiler), "--version"], text=True),
        "commands": commands,
        "host_so": str(host_so),
        "device_so": str(device_so),
        "artifacts": {str(p): sha(p) for p in (host_so, device_so)},
        "device_compile": "PASS",
        "device_execution": "NOT_RUN",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(out / "manifest.json"), "device_compile": "PASS", "device_execution": "NOT_RUN"}))


if __name__ == "__main__":
    main()
