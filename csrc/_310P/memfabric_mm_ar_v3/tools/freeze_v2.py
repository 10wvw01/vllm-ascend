#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze explicitly audited D4/KVER11 files into an existing private unified manifest."""
import argparse
import json
from pathlib import Path

from quick import sha, verify_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    a = p.parse_args()
    manifest = verify_manifest(a.manifest)
    if "v2" in manifest:
        raise ValueError("already frozen; use fresh manifest/build")
    spec = json.loads(a.spec.read_text())
    if spec["kind"] != "D4_TRIAL_KVER11" or not spec["dependencies"] or not spec["source_head"]:
        raise ValueError("audited V2 source and dependency identity required")
    if not spec["environment"].get("MF_SDMA_ORCH_JSON", "").endswith("libmf_sdma_orch_v11.json"):
        raise ValueError("V2 KVER11 launch identity required")
    paths = {spec["library"]: spec["sha256"], **spec["dependencies"]}
    for filename, expected in paths.items():
        if not Path(filename).is_absolute() or sha(filename) != expected:
            raise ValueError(f"audited V2 file mismatch: {filename}")
    if spec["environment"]["MF_SDMA_ORCH_JSON"] not in paths:
        raise ValueError("V2 launch JSON must be among frozen dependencies")
    manifest["v2"] = spec
    a.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": sha(a.manifest), "v2_library_sha256": spec["sha256"]}))


if __name__ == "__main__":
    main()
