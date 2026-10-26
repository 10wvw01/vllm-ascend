#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 read-only G0 evidence safety contract (pure Python / mock facts)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PERF = Path(__file__).resolve().parents[1] / "perf"
sys.path.insert(0, str(PERF))
src = PERF / "a1_g0_readonly.py"
spec = importlib.util.spec_from_file_location("a1_g0_readonly", src)
assert spec is not None and spec.loader is not None
g0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g0
spec.loader.exec_module(g0)


def facts():
    return {
        "git_head": "a" * 40, "git_dirty": False,
        "visible_devices": "0,1",
        "sha256": {name: h for name, (_, h) in g0.ENV_ARTIFACTS.items()},
        "dangerous_flags": [], "npu_smi_readonly_ok": True,
        "required_tools": True,
        "python_plugin_loaded_path_verified": False,
    }


class TestA1ReadonlyG0(unittest.TestCase):
    def test_no_auto_approval_even_when_fingerprints_match(self):
        v = g0.judge(facts())
        self.assertFalse(v["G0_APPROVED"])
        self.assertFalse(v["SMOKE_APPROVED"])
        self.assertTrue(v["static_fingerprint_consistent"])
        self.assertFalse(v["runtime_loaded_plugin_verified"])
        self.assertGreaterEqual(len(v["human_required"]), 4)

    def test_independent_plugin_fingerprint_is_still_not_human_go(self):
        v = facts()
        v["python_plugin_loaded_path_verified"] = True
        r = g0.judge(v)
        self.assertTrue(r["static_fingerprint_consistent"])
        self.assertTrue(r["runtime_loaded_plugin_verified"])
        self.assertFalse(r["G0_APPROVED"])
        self.assertGreaterEqual(len(r["human_required"]), 4)

    def test_any_fingerprint_mismatch_blocks(self):
        for name in g0.ENV_ARTIFACTS:
            with self.subTest(name=name):
                v = facts()
                v["sha256"][name] = "b" * 64
                self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])

    def test_git_status_failure_not_confused_with_clean(self):
        v = facts()
        v["git_dirty"] = None
        self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])

    def test_dirty_git_tree_blocks(self):
        v = facts()
        v["git_dirty"] = True
        self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])

    def test_invalid_visible_die_mapping_blocks(self):
        for devices in ("", "0", "0,0", "0,1,2", "a,b", "0, 1"):
            with self.subTest(devices=devices):
                v = facts()
                v["visible_devices"] = devices
                self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])

    def test_unsafe_bypass_variable_defined_as_zero_still_blocks(self):
        v = facts()
        v["dangerous_flags"] = ["MF_KFC_BYPASS"]
        self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])

    def test_npu_smi_or_required_tools_missing_blocks(self):
        v = facts()
        v["npu_smi_readonly_ok"] = False
        self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])
        v = facts()
        v["required_tools"] = False
        self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])

    def test_g0_script_does_not_import_torch_or_execute_device_commands(self):
        source = src.read_text(encoding="utf-8")
        for prohibited in ("import torch", "torchrun --", "npu-smi set",
                           "npu-smi reset", "os.system", "shell=True",
                           "U1_G0_APPROVED=YES", "pip install", "torch_npu"):
            self.assertNotIn(prohibited, source)
        self.assertIn('["npu-smi", "info"]', source)
        self.assertIn('G0_APPROVED": False', source)

    def test_collect_without_server_never_approves(self):
        with tempfile.TemporaryDirectory() as directory:
            v = g0.collect(Path(directory), {})
            self.assertFalse(g0.judge(v)["G0_APPROVED"])
            self.assertFalse(g0.judge(v)["static_fingerprint_consistent"])


if __name__ == "__main__":
    unittest.main()
