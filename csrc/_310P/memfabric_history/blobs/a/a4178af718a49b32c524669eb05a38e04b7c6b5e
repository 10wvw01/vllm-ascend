#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic validation of real-layer proof constraints."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PERF = Path(__file__).resolve().parents[1] / "perf"
spec = importlib.util.spec_from_file_location("a1_l2_site_audit", PERF / "a1_l2_site_audit.py")
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class L2AuditUT(unittest.TestCase):
    def make(self, temp: Path):
        temp.mkdir(exist_ok=True, parents=True)
        captures = {}
        for arm in tool.ARMS:
            ranks = {}
            for rank in tool.RANKS:
                raw = f"raw_{arm}_rank{rank}.csv"
                (temp / raw).write_text("real device-trace placeholder for CPU fixture\n")
                steps = []
                for i in range(5):
                    sites = []
                    for site_id in tool.EXPECTED:
                        sites.append({
                            "site_id": site_id, "physical_m": 4096,
                            "K": 2048, "N": 2048,
                            "kernel_execution_verified": True,
                            "downstream_visible_verified": True,
                            "native_matmul_verified": True,
                            "hccl_ar_completion_verified": True,
                            "d4_kernel_verified": True,
                            "memfabric_ar_completion_verified": True,
                            "copy_device_us": 0,
                            "device_trace": raw,
                        })
                    steps.append({"sites": sites})
                ranks[str(rank)] = {"steps": steps}
            captures[arm] = {"ranks": ranks}
        obs = temp / "site_audit.json"
        obs.write_text(json.dumps({
            "schema": "A1_L2_REAL_MODEL_SITES_V1",
            "model_eager": True, "tp": 2, "input_tokens": 4096,
            "output_tokens": 1, "captures": captures,
        }))
        return obs

    def test_complete_40_site_proof(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            obs = self.make(path)
            result = tool.evaluate(path, obs)
            self.assertEqual(result["status"], "L2_VERIFIED")

    def test_missing_site_or_noncompleted_hccl_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            obs = self.make(path)
            data = json.loads(obs.read_text())
            data["captures"]["stock"]["ranks"]["1"]["steps"][0]["sites"][0]["hccl_ar_completion_verified"] = False
            obs.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                tool.evaluate(path, obs)
            data["captures"]["stock"]["ranks"]["1"]["steps"][0]["sites"][0]["hccl_ar_completion_verified"] = True
            data["captures"]["v2"]["ranks"]["0"]["steps"][3]["sites"].pop()
            obs.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                tool.evaluate(path, obs)

    def test_kernel_count_alone_cannot_prove_unique_layers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            obs = self.make(path)
            data = json.loads(obs.read_text())
            sites = data["captures"]["v2"]["ranks"]["1"]["steps"][2]["sites"]
            sites[0]["site_id"] = sites[1]["site_id"]
            obs.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                tool.evaluate(path, obs)


if __name__ == "__main__":
    unittest.main()
