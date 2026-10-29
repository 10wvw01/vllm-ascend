#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Restore an immutable project branch snapshot into a NEW directory; never checkout the current tree."""

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "csrc/_310P/memfabric_history"


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def restore(snapshot, out, reader):
    files = snapshot["files"]
    for name in files:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"invalid snapshot path: {name}")
    out.mkdir(parents=True, exist_ok=False)
    for name, entry in files.items():
        data = reader(entry["sha"])
        if blob_sha(data) != entry["sha"]:
            raise ValueError(f"corrupt historical object: {name}")
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry["mode"] == "120000":
            target.symlink_to(data.decode())
        else:
            target.write_bytes(data)
            target.chmod(0o755 if entry["mode"] == "100755" else 0o644)
    (out / "PROJECT_SNAPSHOT_ORIGIN.json").write_text(
        json.dumps({"branch": snapshot["branch"], "head": snapshot["head"], "files": len(files)}, indent=2) + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", help="exact branch from manifest")
    p.add_argument("--out", type=Path)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    manifest = json.loads((HISTORY / "manifest.json").read_text())
    if a.list:
        for b in manifest["branches"]:
            print(b["branch"], b["head"], b["file_count"])
        return
    if not a.branch or not a.out:
        p.error("--branch and --out required")
    entries = [x for x in manifest["branches"] if x["branch"] == a.branch]
    if len(entries) != 1:
        p.error("branch not in immutable inventory")
    snapshot = json.loads(gzip.decompress((HISTORY / entries[0]["snapshot"]).read_bytes()))

    def read(sha):
        path = HISTORY / "blobs" / sha[0] / sha
        if path.is_file():
            return path.read_bytes()
        return subprocess.check_output(["git", "cat-file", "blob", sha], cwd=ROOT)

    restore(snapshot, a.out.resolve(), read)
    print(f"restored {len(snapshot['files'])} project files from {snapshot['head']}; no runtime defaults changed")


if __name__ == "__main__":
    main()
