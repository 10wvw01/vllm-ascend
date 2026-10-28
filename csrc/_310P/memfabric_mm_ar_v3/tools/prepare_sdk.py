#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned MemFabric in a fresh clone; stage a complete private SDK.

The upstream .run packer does not copy output/hybm/aicpu_kernel. This
stager explicitly includes SDMA source SO + CUST JSON + versioned launch
JSON, so the executor must not improvise packaging or copy old KVER11 files.
No shared installation, shell profile, Python wheel, or device is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from build import MF_BASE, MF_TREE, MF_UPSTREAM, git, sha


def copy_file(source, target, records):
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != sha(source):
            raise ValueError(f"conflicting artifact names: {target}")
        return
    shutil.copy2(source, target)
    records.append({"source": str(source), "target": str(target), "sha256": sha(target)})


def stage(source, dest):
    arch = "aarch64-linux"
    records = []
    for kind in ("host", "device"):
        headers = sorted((source / "src/smem/include" / kind).glob("*.h"))
        if not headers:
            raise FileNotFoundError(f"missing public {kind} headers")
        for header in headers:
            copy_file(header, dest / "include/smem" / kind / header.name, records)
    # Flatten the actual host libraries built in this isolated source tree.
    libraries = sorted(
        p for p in (source / "output").rglob("lib*.so*") if p.is_file() and "aicpu_kernel" not in p.parts
    )
    for library in libraries:
        copy_file(library, dest / arch / "lib64" / library.name, records)
    for name in ("libmf_smem.so", "libmf_hybm_core.so"):
        if not (dest / arch / "lib64" / name).is_file():
            raise FileNotFoundError(f"build did not produce {name}")
    kernel = source / "build/src/hybm/ops/hybm_kernel"
    for name in ("libmf_sdma_orch.so", "libmf_sdma_orch.json", "libmf_sdma_orch_v28.json"):
        copy_file(kernel / name, dest / arch / "hybm/aicpu_kernel" / name, records)
    (dest / "version.info").write_text(
        f"Version:{(source / 'VERSION').read_text().strip()}\nPlatform:aarch64\nKernel:linux\n"
        f"CommitId:{MF_BASE}\nUpstreamCommitId:{MF_UPSTREAM}\nTree:{MF_TREE}\n"
        "PackageKind:V3_PRIVATE_STAGED_SDK\n"
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    source, out, cann = args.source.resolve(strict=True), args.out.resolve(), args.cann.resolve(strict=True)
    if platform.machine() != "aarch64":
        raise ValueError("MemFabric SDK must be built on the aarch64 310P3 host")
    if git(source, "rev-parse", "HEAD") != MF_BASE or git(source, "rev-parse", "HEAD^{tree}") != MF_TREE:
        raise ValueError("wrong MemFabric commit/tree; use the exact GitHub mirror lock")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("dirty MemFabric source")
    if (source / "build").exists() or (source / "output").exists():
        raise ValueError("fresh isolated clone required: upstream wrapper deletes build/output")
    if out.is_relative_to(source):
        raise ValueError("private SDK output must be outside the isolated source clone")
    if not 1 <= args.jobs <= 32:
        raise ValueError("jobs must be 1..32")
    hcc = cann / "toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++"
    if not hcc.is_file():
        raise FileNotFoundError(f"required AICPU cross compiler missing: {hcc}")
    out.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    # Do not let an inherited CMake default redirect upstream's install target
    # to an existing shared prefix. Fresh default /usr/local maps to source/output.
    environment.pop("CMAKE_INSTALL_PREFIX", None)
    environment.update(ASCEND_HOME_PATH=str(cann), MF_BUILD_JOBS=str(args.jobs))
    command = [
        "bash",
        "script/build_and_pack_run.sh",
        "--build_python",
        "OFF",
        "--build_test",
        "OFF",
        "--build_hcom",
        "OFF",
        "--build_mode",
        "RELEASE",
        "--xpu_type",
        "NPU",
    ]
    with (out / "memfabric_build.log").open("x") as log:
        subprocess.run(
            command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True
        )
    # Stage from the build graph, not from the incomplete .run archive; do not execute its installer.
    records = stage(source, out)
    receipt = {
        "schema": "V3_PRIVATE_SDK_V1",
        "source": str(source),
        "head": MF_BASE,
        "upstream_head": MF_UPSTREAM,
        "tree": MF_TREE,
        "kver": 28,
        "build_command": command,
        "files": records,
        "device_deployment": "NOT_RUN",
        "scope": "host private staging only",
    }
    (out / "sdk_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"sdk": str(out), "head": MF_BASE, "kver": 28, "device_deployment": "NOT_RUN"}))


if __name__ == "__main__":
    main()
