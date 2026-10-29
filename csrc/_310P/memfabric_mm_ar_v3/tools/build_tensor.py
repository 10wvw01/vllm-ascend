#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Privately compile and freeze the TorchNPU Storage adapter before G0/device work."""

import argparse
import json
import sys
from pathlib import Path

from build import REPO, sha, source_files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    if m["sources"] != source_files():
        raise RuntimeError("source fingerprint changed; rebuild host/device first")
    if "tensor_adapter" in m:
        raise RuntimeError("Tensor adapter already frozen; create a new build directory")
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(REPO / "vllm_ascend/_310p/ops"))
    import torch_npu  # noqa: F401 - initializes the NPU Storage factory
    from memfabric_mm_ar_v32 import load_tensor_module

    module = load_tensor_module(a.out)
    so = Path(module.__file__).resolve()
    m["tensor_adapter"] = {"path": str(so), "sha256": sha(so), "compile": "PASS", "storage_device_gate": "NOT_RUN"}
    a.manifest.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["tensor_adapter"]))


if __name__ == "__main__":
    main()
