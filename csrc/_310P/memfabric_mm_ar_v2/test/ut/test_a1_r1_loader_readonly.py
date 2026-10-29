#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 R1只读静态验收器的CPU合同测试。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "perf" / "a1_r1_loader_readonly.py"
spec = importlib.util.spec_from_file_location("a1_r1_loader_readonly", TOOL)
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class A1R1LoaderReadonlyUT(unittest.TestCase):
    def test_requires_complete_absolute_path_map(self):
        complete = {name: "/tmp/" + name for name in tool.REQUIRED}
        self.assertEqual(set(tool.paths_contract(complete)), set(tool.REQUIRED))
        with self.assertRaises(ValueError):
            tool.paths_contract({**complete, "extra": "/tmp/evil"})
        with self.assertRaises(ValueError):
            tool.paths_contract({**complete, "host_system": "../relative"})
        with self.assertRaises(ValueError):
            tool.paths_contract({k: v for k, v in complete.items() if k != "host_system"})
        with self.assertRaises(ValueError):
            tool.paths_contract({"host_system": 3})
        with self.assertRaises(ValueError):
            tool.paths_contract([])

    def test_digest_symlink_and_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "libmf_hybm_core.so"
            p.write_bytes(b"STATIC_SO")
            link = root / "system.so"
            link.symlink_to(p)
            obj = tool.inspect(link, is_elf=False)
            self.assertEqual(obj["status"], "FILE")
            self.assertTrue(obj["symlink"])
            self.assertEqual(obj["sha256"], tool.digest(p))
            self.assertEqual(tool.inspect(root / "notfound.so", is_elf=False)["status"],
                             "UNAVAILABLE")

    def test_never_claim_runtime_ready_from_file_hashes(self):
        fake = {name: {"sha256": "unrelated"} for name in tool.REQUIRED}
        fake.update({
            "host_archive_fixed": {"sha256": tool.HOST_FIXED},
            "host_system": {"sha256": tool.HOST_FIXED},
            "host_python": {"sha256": tool.HOST_FIXED},
            "plugin_so": {"sha256": tool.PLUGIN},
            "d4_device_candidate": {"sha256": tool.D4},
            "device_so_installed": {"sha256": tool.D4},
            "kver11_json": {"sha256": tool.KVER},
        })
        result = tool.evaluate(fake)
        self.assertTrue(result["all_frozen_candidate_files_present"])
        self.assertTrue(result["device_installed_is_frozen_d4"])
        for key in ("actual_host_dlopen_path_proven",
                    "actual_device_kernel_so_loaded_proven",
                    "python_editable_import_frozen_proven",
                    "card_exclusivity_current_proven",
                    "kfc_and_first_write_correctness_proven",
                    "permission_to_deploy", "permission_to_smoke"):
            self.assertFalse(result[key])

    def test_stale_installed_device_fails_even_with_valid_candidate(self):
        fake = {name: {"sha256": ""} for name in tool.REQUIRED}
        fake["host_archive_fixed"]["sha256"] = tool.HOST_FIXED
        fake["plugin_so"]["sha256"] = tool.PLUGIN
        fake["d4_device_candidate"]["sha256"] = tool.D4
        fake["kver11_json"]["sha256"] = tool.KVER
        fake["device_so_installed"]["sha256"] = "b8ca96f0"
        result = tool.evaluate(fake)
        self.assertTrue(result["all_frozen_candidate_files_present"])
        self.assertFalse(result["device_installed_is_frozen_d4"])
        self.assertFalse(result["permission_to_smoke"])

    def test_script_contains_no_runtime_approval_or_import(self):
        source = TOOL.read_text()
        for prohibited in ("import torch", "import torch_npu",
                           "ctypes.CDLL(", "torchrun", "os.system(",
                           "subprocess.Popen(", "os.replace("):
            self.assertNotIn(prohibited, source)
        self.assertIn('"DEPLOY_APPROVED": False', source)
        self.assertIn('"SMOKE_APPROVED": False', source)


if __name__ == "__main__":
    unittest.main()
